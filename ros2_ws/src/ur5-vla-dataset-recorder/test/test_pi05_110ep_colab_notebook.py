"""Offline regression checks for the Pi0.5 notebook and local checkpoint pruning."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/train_ur5_cube_110ep_v3_pi05_a100_colab.ipynb'


def cell(index):
    return ''.join(json.loads(NOTEBOOK.read_text())['cells'][index]['source'])


def helpers():
    namespace = dict(Path=Path, hashlib=hashlib, shutil=shutil, json=json)
    wanted = {'sha256_file', 'tree_hashes', 'copy_verified_tree', 'checkpoint_complete',
              'sync_checkpoints', 'prune_verified_local_checkpoints'}
    for index in (5, 12):
        nodes = ast.parse(cell(index)).body
        selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in wanted]
        selected += [n for n in nodes if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in {'REQUIRED_CHECKPOINT_FILES', 'synced_steps'} for t in n.targets)]
        exec(compile(ast.Module(body=selected, type_ignores=[]), 'helpers', 'exec'), namespace)
    return namespace


def test_cells_compile_without_saved_outputs():
    notebook = json.loads(NOTEBOOK.read_text())
    for item in notebook['cells']:
        if item['cell_type'] == 'code':
            code = '\n'.join(s for s in ''.join(item['source']).splitlines() if not s.startswith('%pip '))
            compile(code, 'cell', 'exec')
            assert item['outputs'] == [] and item['execution_count'] is None


def test_prunes_only_verified_local_copies_after_new_checkpoint_sync(tmp_path):
    h = helpers()
    output = tmp_path / 'output'; checkpoint_dir = output / 'checkpoints'; checkpoint_dir.mkdir(parents=True)
    drive = tmp_path / 'drive'; drive.mkdir(); destination = drive / 'checkpoints'; destination.mkdir()
    h.update(TRAIN_OUTPUT=output, DRIVE_RUN_DIR=drive, DRIVE_CHECKPOINTS_DIR=destination, LOCAL_CHECKPOINTS_TO_KEEP=2)
    names = ['002500', '005000', '007500']
    for name in names:
        for relative in h['REQUIRED_CHECKPOINT_FILES']:
            artifact = checkpoint_dir / name / relative
            artifact.parent.mkdir(parents=True, exist_ok=True); artifact.write_text(name)
    # An in-progress fourth checkpoint must never be copied or deleted.
    incomplete = checkpoint_dir / '010000'; incomplete.mkdir()
    (incomplete / 'partial').write_text('writing')
    (checkpoint_dir / 'last').symlink_to(names[-1])
    h['sync_checkpoints']()
    assert not (checkpoint_dir / names[0]).exists()
    assert all((checkpoint_dir / n).exists() for n in names[1:])
    assert all(h['checkpoint_complete'](destination / n) for n in names)
    assert incomplete.is_dir() and not (destination / incomplete.name).exists()


def test_corrupted_drive_copy_prevents_local_pruning(tmp_path):
    h = helpers(); local = tmp_path / 'local'; drive = tmp_path / 'drive'
    local.mkdir(); drive.mkdir()
    for name in ['0001', '0002', '0003']:
        p = local / name; p.mkdir(); (p / 'weights').write_text('valid')
        shutil.copytree(p, drive / name)
    (drive / '0001/weights').write_text('corrupted')
    h.update(DRIVE_CHECKPOINTS_DIR=drive, synced_steps={'0001', '0002', '0003'}, LOCAL_CHECKPOINTS_TO_KEEP=2)
    with pytest.raises(AssertionError, match='Drive copy changed'):
        h['prune_verified_local_checkpoints'](local)
    assert (local / '0001/weights').read_text() == 'valid'


def test_policy_arguments_are_pi05_absolute_and_quantile_normalized():
    namespace = dict(json=json, BASE_MODEL_PATH='/cached/pi05', TRAIN_EXPERT_ONLY=True,
                     NORMALIZATION={'STATE': 'QUANTILES', 'ACTION': 'QUANTILES', 'VISUAL': 'IDENTITY'})
    fn = next(n for n in ast.parse(cell(9)).body if isinstance(n, ast.FunctionDef) and n.name == 'policy_arguments')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'policy_arguments', 'exec'), namespace)
    args = namespace['policy_arguments']()
    assert '--policy.type=pi05' in args and '--policy.use_relative_actions=false' in args
    assert '--policy.train_expert_only=true' in args and '--policy.dtype=bfloat16' in args
    assert not any(arg.startswith('--policy.path=') for arg in args)
    assert json.loads(next(s.split('=', 1)[1] for s in args if s.startswith('--policy.normalization_mapping='))) == namespace['NORMALIZATION']


@pytest.mark.parametrize('secret,cached,prompted', [('secret-token', None, False), (None, 'cached-token', False), (None, None, True)])
def test_authentication_accepts_secret_cached_login_or_hidden_prompt(secret, cached, prompted, capsys):
    from types import SimpleNamespace
    from unittest.mock import Mock
    userdata = Mock()
    if secret is None:
        userdata.get.side_effect = RuntimeError('Secret unavailable')
    else:
        userdata.get.return_value = secret
    prompt = Mock(return_value='entered-token')
    login = Mock()
    tokenizer_loader = Mock(return_value=lambda text: {'input_ids': [1]})
    namespace = dict(userdata=userdata, get_token=lambda: cached, getpass=prompt,
                     hf_login=login, AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader))
    fn = next(n for n in ast.parse(cell(7)).body if isinstance(n, ast.FunctionDef) and n.name == 'authenticate_hugging_face')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'authentication', 'exec'), namespace)
    namespace['authenticate_hugging_face']()
    expected = secret or cached or 'entered-token'
    login.assert_called_once_with(token=expected, add_to_git_credential=False)
    tokenizer_loader.assert_called_once_with('google/paligemma-3b-pt-224', token=expected)
    assert prompt.called == prompted
    assert expected not in capsys.readouterr().out
