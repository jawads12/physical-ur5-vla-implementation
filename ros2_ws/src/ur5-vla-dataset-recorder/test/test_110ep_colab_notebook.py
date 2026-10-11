"""Offline checks for the new notebook's copy and checkpoint persistence logic."""
import ast
import hashlib
import json
import math
from pathlib import Path
import shutil

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/train_ur5_cube_110ep_v3_smolvla_a100_colab.ipynb'


def cell(index):
    return ''.join(json.loads(NOTEBOOK.read_text())['cells'][index]['source'])


def helpers():
    namespace = dict(Path=Path, hashlib=hashlib, shutil=shutil, json=json)
    for index in (5, 12):
        nodes = ast.parse(cell(index)).body
        selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in {
            'sha256_file', 'tree_hashes', 'copy_verified_tree', 'checkpoint_complete', 'sync_checkpoints'}]
        selected += [n for n in nodes if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in {'REQUIRED_CHECKPOINT_FILES', 'synced_steps'} for t in n.targets)]
        exec(compile(ast.Module(body=selected, type_ignores=[]), 'notebook_helpers', 'exec'), namespace)
    return namespace


def test_code_cells_compile_and_have_no_saved_outputs():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook['nbformat'] == 4
    for item in notebook['cells']:
        if item['cell_type'] == 'code':
            # Only install magics require IPython; the remaining cells are plain Python.
            code = '\n'.join(line for line in ''.join(item['source']).splitlines() if not line.startswith('%pip '))
            compile(code, 'notebook_cell', 'exec')
            assert item['outputs'] == [] and item['execution_count'] is None


def test_verified_copy_preserves_source_and_rejects_changed_destination(tmp_path):
    h = helpers()
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir(); (source / 'weights').write_bytes(b'complete weights')
    expected = h['tree_hashes'](source)
    assert h['copy_verified_tree'](source, target) == expected
    assert h['copy_verified_tree'](source, target) == expected
    (target / 'weights').write_bytes(b'corruption')
    with pytest.raises(AssertionError, match='different content'):
        h['copy_verified_tree'](source, target)
    assert h['tree_hashes'](source) == expected


def test_partial_copy_is_not_overwritten(tmp_path):
    h = helpers()
    source = tmp_path / 'source'; source.mkdir(); (source / 'file').write_text('content')
    partial = tmp_path / '.target.copying'; partial.mkdir(); (partial / 'file').write_text('partial')
    with pytest.raises(AssertionError, match='Partial copy'):
        h['copy_verified_tree'](source, tmp_path / 'target')
    assert (partial / 'file').read_text() == 'partial'


def test_checkpoint_sync_waits_for_completion_pointer(tmp_path):
    h = helpers()
    output = tmp_path / 'training'; checkpoints = output / 'checkpoints'; checkpoints.mkdir(parents=True)
    drive = tmp_path / 'drive'; drive.mkdir(); destination = drive / 'checkpoints'; destination.mkdir()
    h.update(TRAIN_OUTPUT=output, DRIVE_RUN_DIR=drive, DRIVE_CHECKPOINTS_DIR=destination)
    first, second = checkpoints / '003000', checkpoints / '006000'
    for p in (first, second):
        for name in h['REQUIRED_CHECKPOINT_FILES']:
            artifact = p / name; artifact.parent.mkdir(parents=True, exist_ok=True); artifact.write_text('complete')
    h['sync_checkpoints']()
    assert not list(destination.iterdir())
    last = checkpoints / 'last'; last.symlink_to(first.name)
    h['sync_checkpoints']()
    assert (destination / first.name).is_dir() and not (destination / second.name).exists()
    (second / 'training_state/scheduler_state.json').unlink()
    last.unlink(); last.symlink_to(second.name)
    with pytest.raises(RuntimeError, match='missing required artifacts'):
        h['sync_checkpoints']()
    assert not (destination / second.name).exists()


def test_budget_uses_new_dataset_size():
    nodes = ast.parse(cell(10)).body
    selected = [n for n in nodes if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {'TRAIN_STEPS', 'SAVE_FREQ', 'WARMUP_STEPS'} for t in n.targets)]
    from types import SimpleNamespace
    namespace = dict(math=math, dataset=SimpleNamespace(meta=SimpleNamespace(total_frames=285204)), TARGET_EPOCHS=8.0, BATCH_SIZE=96)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'budget', 'exec'), namespace)
    assert (namespace['TRAIN_STEPS'], namespace['SAVE_FREQ'], namespace['WARMUP_STEPS']) == (24000, 3000, 1000)
