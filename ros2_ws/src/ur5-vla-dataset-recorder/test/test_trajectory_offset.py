"""Geometry and queue regressions for the continuous Cartesian modifier."""
import ast
from collections import deque
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
from trajectory_offset import CartesianOffsetModel, rotation_error


def model():
    # A bounded six-joint chain exercises XML origins, frame handling and IK
    # without relying on a running ROS installation or robot.
    xml = ['<robot name="test"><link name="base"/>']
    previous = 'base'
    for i in range(6):
        link = f'link_{i}'
        xml.append(f'<link name="{link}"/><joint name="q{i}" type="revolute">'
                   f'<parent link="{previous}"/><child link="{link}"/>'
                   '<origin xyz="0.2 0 0"/><axis xyz="0 0 1"/>'
                   '<limit lower="-3" upper="3" effort="1" velocity="1"/></joint>')
        previous = link
    xml.append(f'<link name="tool0"/><joint name="tip" type="fixed">'
               f'<parent link="{previous}"/><child link="tool0"/>'
               '<origin xyz="0.1 0 0"/></joint></robot>')
    return CartesianOffsetModel(''.join(xml), [f'q{i}' for i in range(6)])


def test_every_target_has_one_translation_and_preserves_orientation():
    m = model()
    for i in range(50):
        q = [.5 + .0001*i, -.8, .6, .3, -.7, .2]
        original, _ = m.fk_jacobian(q)
        corrected, _ = m.fk_jacobian(m.translate(q, [.005, 0, 0]))
        assert np.linalg.norm(corrected[:3, 3]-original[:3, 3]-[.005, 0, 0]) < .00005
        assert np.linalg.norm(rotation_error(original[:3, :3], corrected[:3, :3])) < .001


def test_analytic_jacobian_matches_finite_difference():
    m = model()
    q = [.5, -.8, .6, .3, -.7, .2]
    original, jacobian = m.fk_jacobian(q)
    for i in range(6):
        perturbed = list(q)
        perturbed[i] += 1e-6
        changed, _ = m.fk_jacobian(perturbed)
        assert np.linalg.norm((changed[:3, 3]-original[:3, 3])/1e-6-jacobian[:3, i]) < 1e-5


@pytest.mark.parametrize('vector', [[.008, 0, 0], [float('nan'), 0, 0], [0, 0]])
def test_invalid_offsets_are_rejected(vector):
    with pytest.raises(ValueError):
        model().translate([.5, -.8, .6, .3, -.7, .2], vector)


def test_blending_uses_raw_boundary_and_keeps_gripper_sequence():
    # Exercise the actual executor methods with fake model/ROS endpoints.
    source = ast.parse((ROOT / 'scripts/policy_executor.py').read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'PolicyExecutor')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('infer', 'publish_action')]
    module = ast.Module(body=[ast.ClassDef(name='Harness', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    class Message:
        pass
    calls = []
    def transform(actions, reference):
        calls.append([list(a) for a in actions])
        return [[*[v+.01 for v in a[:6]], a[6]] for a in actions], {}
    chunk = [[.1]*6+[float(i<25)] for i in range(50)]
    namespace = dict(time=time, Float64MultiArray=Message,
                     request_prediction=lambda *args: (chunk, 0),
                     SERVER_HOST='localhost', SERVER_PORT=8765, TASK='test', BLEND_ACTIONS=10)
    exec(compile(module, 'executor_methods', 'exec'), namespace)
    h = namespace['Harness']()
    h.queue_lock = threading.Lock()
    h.action_queue = deque([[.06]*6+[1.]])
    h.raw_action_queue = deque([[.05]*6+[1.]])
    h.last_action = h.last_raw_action = None
    h.trajectory_offset = SimpleNamespace(plan_actions=transform)
    h.grasp_assist = None
    h.offset_fault = False
    h.inference_generation = 0
    h.inference_running = True
    h.measured_joints = [.06]*6
    h.measured_joints_at = time.monotonic()
    h.get_logger = lambda: SimpleNamespace(info=lambda msg: None, error=lambda msg: None)
    sent = []
    h.publisher = SimpleNamespace(publish=lambda msg: sent.append(msg.data))
    h.infer([0]*7, [], 0)
    assert not h.offset_fault and len(h.action_queue) == len(h.raw_action_queue) == 51
    weight = .1*.1*(3-2*.1)
    assert calls[0][0][0] == pytest.approx(.05+weight*(.1-.05))
    for _ in range(51):
        h.measured_joints_at = time.monotonic()
        h.publish_action()
    assert [a[6] for a in sent[1:]] == [a[6] for a in chunk]
    assert sent[-1][0] == pytest.approx(.11)
    assert not h.raw_action_queue
    h.inference_running = True
    h.infer([0]*7, [], 0)
    # The next blend begins at 0.1 raw, never the already corrected 0.11.
    assert calls[1][0][0] == pytest.approx(.1)
