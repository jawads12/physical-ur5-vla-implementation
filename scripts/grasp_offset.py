"""Opt-in, one-grasp Cartesian correction for a supervised experiment.

Planning uses MoveIt services only. The policy executor owns publication.
After an acknowledged close, the executor replans without changing carry targets.
"""
from __future__ import annotations

import copy
import json
import math
import threading
import time
from pathlib import Path


def load_offset(path: Path) -> tuple[str, list[float]]:
    data = json.loads(path.read_text())
    frame = data.get('offset_frame')
    vector = [float(x) for x in data.get('offset_m', [])]
    if frame != 'base' or len(vector) != 3 or not all(math.isfinite(x) for x in vector):
        raise ValueError('offset requires three finite metre values in the measured base frame')
    distance = math.sqrt(sum(x*x for x in vector))
    if not 0 < distance <= .00701:
        raise ValueError('this experiment permits at most 7 mm Cartesian correction')
    return frame, vector


def shifted_pose(pose, vector):
    result = copy.deepcopy(pose)
    for axis, delta in zip(('x', 'y', 'z'), vector):
        setattr(result.position, axis, getattr(result.position, axis) + delta)
    return result


def checked_joints(values):
    values = [float(x) for x in values]
    if len(values) != 6 or not all(math.isfinite(x) for x in values):
        raise ValueError('six finite joint values required')
    return values


class OffsetPlanner:
    def __init__(self, node, order, frame, vector):
        from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetStateValidity
        self.node, self.order, self.frame, self.vector = node, list(order), frame, vector
        self.fk = node.create_client(GetPositionFK, '/compute_fk')
        self.ik = node.create_client(GetPositionIK, '/compute_ik')
        self.validity = node.create_client(GetStateValidity, '/check_state_validity')

    @staticmethod
    def call(client, request, timeout=2.0):
        if not client.service_is_ready():
            raise RuntimeError('required MoveIt planning service unavailable')
        completed = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _: completed.set())
        if not completed.wait(timeout):
            future.cancel()
            raise TimeoutError(f'MoveIt offset planning timed out: {client.srv_name}')
        result = future.result()
        if result is None:
            raise RuntimeError('MoveIt planning returned no result')
        return result

    def pose(self, joints):
        from moveit_msgs.srv import GetPositionFK
        req = GetPositionFK.Request()
        req.header.frame_id = self.frame
        req.fk_link_names = ['tool0']
        req.robot_state.joint_state.name = self.order
        req.robot_state.joint_state.position = checked_joints(joints)
        result = self.call(self.fk, req)
        if result.error_code.val != 1 or len(result.pose_stamped) != 1:
            raise RuntimeError('forward kinematics failed')
        if result.pose_stamped[0].header.frame_id != self.frame:
            raise RuntimeError('forward kinematics returned an unexpected frame')
        return result.pose_stamped[0]

    def plan(self, target, measured):
        from moveit_msgs.srv import GetPositionIK, GetStateValidity
        target, measured = checked_joints(target), checked_joints(measured)
        if max(abs(a-b) for a,b in zip(target, measured)) > .05:
            raise ValueError('uncorrected closing target is more than 0.05 rad from measured pose')
        original = self.pose(target)
        desired = copy.deepcopy(original)
        desired.pose = shifted_pose(original.pose, self.vector)
        req = GetPositionIK.Request()
        req.ik_request.group_name = 'ur_manipulator'
        req.ik_request.ik_link_name = 'tool0'
        req.ik_request.pose_stamped = desired
        req.ik_request.robot_state.joint_state.name = self.order
        req.ik_request.robot_state.joint_state.position = target
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec = 1
        result = self.call(self.ik, req)
        if result.error_code.val != 1:
            raise RuntimeError(f'inverse kinematics rejected correction: {result.error_code.val}')
        state = result.solution.joint_state
        corrected = checked_joints([state.position[state.name.index(k)] for k in self.order])
        if max(abs(a-b) for a,b in zip(corrected, measured)) > .05:
            raise ValueError('corrected target exceeds the 0.05 rad experiment envelope')
        reached = self.pose(corrected).pose
        position_error = math.sqrt(sum((getattr(reached.position,k)-getattr(desired.pose.position,k))**2 for k in ('x','y','z')))
        qa = [getattr(reached.orientation,k) for k in ('x','y','z','w')]
        qb = [getattr(desired.pose.orientation,k) for k in ('x','y','z','w')]
        cosine = abs(sum(a*b for a,b in zip(qa,qb))) / math.sqrt(sum(a*a for a in qa)*sum(b*b for b in qb))
        angle_error = 2*math.acos(min(1,cosine))
        if position_error > .0005 or angle_error > .01:
            raise ValueError('IK solution does not reproduce requested translation/orientation')
        # Validate the interpolated joint path, not just its endpoint.
        for step in range(11):
            req = GetStateValidity.Request()
            req.group_name = 'ur_manipulator'
            req.robot_state.is_diff = True
            req.robot_state.joint_state.name = self.order
            req.robot_state.joint_state.position = [a+(b-a)*step/10 for a,b in zip(measured, corrected)]
            if not self.call(self.validity, req).valid:
                raise RuntimeError('MoveIt rejected the interpolated correction path')
        return corrected, {'frame':self.frame, 'offset_m':self.vector, 'fk_position_error_m':position_error, 'fk_orientation_error_rad':angle_error}


class GraspOffsetAssist:
    def __init__(self, node, order, calibration):
        from std_msgs.msg import Float64
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        self.node = node
        frame, vector = load_offset(calibration)
        self.planner = OffsetPlanner(node, order, frame, vector)
        self.stage = 'idle'
        self.target = None
        self.error = None
        self.started = 0.
        self.close_started = 0.
        self.aligned_since = None
        self.confirmed_closed_at = 0.
        self.confirmed_gripper = None
        self.confirmed_gripper_at = 0.
        self.plan_lock = threading.Lock()
        self.subscription = node.create_subscription(
            Float64, '/control_engine/gripper_state', self.io,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        node.get_logger().warning(f'ONE-GRASP ASSIST ENABLED: {vector} metres in {frame}; fresh policy inference resumes after confirmed close')

    @property
    def active(self):
        return self.stage not in ('idle', 'complete')

    def io(self, message):
        value = float(message.data)
        if value not in (0., 1.):
            return
        self.confirmed_gripper = value
        self.confirmed_gripper_at = time.monotonic()
        if value == 0.:
            self.confirmed_closed_at = self.confirmed_gripper_at

    @property
    def open_confirmed(self):
        return self.confirmed_gripper == 1. and time.monotonic() - self.confirmed_gripper_at <= .2

    def start(self, target, measured):
        self.stage = 'planning'
        self.started = time.monotonic()
        def worker():
            try:
                corrected, report = self.planner.plan(target, measured)
                with self.plan_lock:
                    if self.stage == 'planning':
                        self.target = corrected
                        self.stage = 'aligning'
                        self.started = time.monotonic()
                self.node.get_logger().warning('OFFSET PLAN CHECKED: '+json.dumps(report))
            except Exception as exc:
                with self.plan_lock:
                    self.error = str(exc)
                    self.stage = 'fault'
                self.node.get_logger().error('OFFSET BLOCKED: '+str(exc))
        threading.Thread(target=worker, daemon=True).start()

    def tick(self, measured, measured_at, open_confirmed):
        now = time.monotonic()
        if self.stage in ('planning','fault','complete','idle'):
            return None
        if measured is None or now-measured_at > .1:
            self.stage = 'fault'
            self.node.get_logger().error('OFFSET BLOCKED: stale measured joints')
            return None
        if self.stage == 'aligning':
            if not open_confirmed:
                self.stage = 'fault'
                self.node.get_logger().error('OFFSET BLOCKED: gripper-open IO is not confirmed')
                return None
            if now-self.started > 5:
                self.stage = 'fault'
                self.node.get_logger().error('OFFSET BLOCKED: arm failed to settle at corrected target')
                return None
            if max(abs(a-b) for a,b in zip(measured,self.target)) <= .0015:
                if self.aligned_since is None:self.aligned_since = now
                if now-self.aligned_since >= .2:
                    self.stage = 'closing'
                    self.close_started = now
                    self.node.get_logger().warning('OFFSET REACHED: commanding one gripper close')
            else:self.aligned_since = None
            return [*self.target, 0. if self.stage == 'closing' else 1.]
        if now-self.close_started > 2:
            self.stage = 'fault'
            self.node.get_logger().error('OFFSET BLOCKED: no fresh gripper-close IO confirmation')
            return None
        if self.confirmed_closed_at > self.close_started and now-self.close_started >= .5:
            self.stage = 'complete'
            self.node.get_logger().warning('ASSISTED GRASP FINISHED: closed IO confirmed; resuming from a fresh policy observation')
            return None
        return [*self.target, 0.]
