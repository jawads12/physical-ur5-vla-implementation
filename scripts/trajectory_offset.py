"""Constant Cartesian translation of every policy target, using the MoveIt model."""
from __future__ import annotations

import math
import threading
import xml.etree.ElementTree as ET

import numpy as np

from grasp_offset import OffsetPlanner, checked_joints


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew@skew)


def origin_transform(element):
    result = np.eye(4)
    if element is None:
        return result
    result[:3, 3] = [float(x) for x in element.get('xyz', '0 0 0').split()]
    roll, pitch, yaw = [float(x) for x in element.get('rpy', '0 0 0').split()]
    result[:3, :3] = rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)
    return result


def rotation_error(desired, actual):
    r = desired @ actual.T
    angle = math.acos(float(np.clip((np.trace(r)-1)/2, -1, 1)))
    vector = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])
    return .5*vector if angle < 1e-7 else angle/(2*math.sin(angle))*vector


class CartesianOffsetModel:
    def __init__(self, urdf, order, frame='base', tip='tool0'):
        xml = ET.fromstring(urdf)
        links = {link.get('name') for link in xml.findall('link')}
        if frame not in links or tip not in links:
            raise ValueError('offset frame or tool link missing from robot description')
        self.order = list(order)
        self.parents = {}
        self.limits = {}
        for joint in xml.findall('joint'):
            name, kind = joint.get('name'), joint.get('type')
            axis = joint.find('axis')
            limit = joint.find('limit')
            data = {'name': name, 'type': kind, 'parent': joint.find('parent').get('link'),
                    'origin': origin_transform(joint.find('origin')),
                    'axis': np.array([float(x) for x in (axis.get('xyz') if axis is not None else '1 0 0').split()])}
            self.parents[joint.find('child').get('link')] = data
            if name in self.order:
                if kind != 'revolute' or limit is None:
                    raise ValueError('offset model requires bounded UR5 revolute joints')
                self.limits[name] = (float(limit.get('lower')), float(limit.get('upper')))
        self.chain = self.path(tip)
        if {j['name'] for j in self.chain if j['type'] != 'fixed'} != set(self.order):
            raise ValueError('robot model joint chain differs from the policy joint order')
        base_path = self.path(frame)
        if any(j['type'] != 'fixed' for j in base_path):
            raise ValueError('offset reference frame must be fixed')
        transform = np.eye(4)
        for joint in base_path:
            transform = transform @ joint['origin']
        self.root_to_base = np.linalg.inv(transform)

    def path(self, link):
        path, visited = [], set()
        while link in self.parents:
            if link in visited:
                raise ValueError('cyclic robot description')
            visited.add(link)
            joint = self.parents[link]
            path.append(joint)
            link = joint['parent']
        return list(reversed(path))

    def fk_jacobian(self, values):
        q = dict(zip(self.order, checked_joints(values)))
        transform = self.root_to_base.copy()
        axes = {}
        for joint in self.chain:
            transform = transform @ joint['origin']
            if joint['type'] != 'fixed':
                axes[joint['name']] = (transform[:3, 3].copy(), transform[:3, :3] @ joint['axis'])
                turn = np.eye(4)
                turn[:3, :3] = rotation(joint['axis'], q[joint['name']])
                transform = transform @ turn
        jacobian = np.zeros((6, 6))
        for i, name in enumerate(self.order):
            point, axis = axes[name]
            jacobian[:3, i] = np.cross(axis, transform[:3, 3]-point)
            jacobian[3:, i] = axis
        return transform, jacobian

    def translate(self, target, vector):
        original = np.asarray(checked_joints(target))
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) > .00701:
            raise ValueError('invalid or excessive Cartesian translation')
        desired, _ = self.fk_jacobian(original)
        desired[:3, 3] += np.asarray(vector)
        solution = original.copy()
        for _ in range(12):
            actual, jacobian = self.fk_jacobian(solution)
            position = desired[:3, 3]-actual[:3, 3]
            orientation = rotation_error(desired[:3, :3], actual[:3, :3])
            if np.linalg.norm(position) <= .00001 and np.linalg.norm(orientation) <= .0001:
                break
            error = np.concatenate([position, orientation])
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-8*np.eye(6), error)
            largest = np.max(np.abs(delta))
            if largest > .01:
                delta *= .01/largest
            solution += delta
        actual, _ = self.fk_jacobian(solution)
        if (np.linalg.norm(desired[:3, 3]-actual[:3, 3]) > .00005 or
                np.linalg.norm(rotation_error(desired[:3, :3], actual[:3, :3])) > .001):
            raise ValueError('continuous offset IK did not converge')
        if np.max(np.abs(solution-original)) > .05:
            raise ValueError('offset correction exceeds 0.05 rad relative to its original target')
        for name, value in zip(self.order, solution):
            lo, hi = self.limits[name]
            if not lo <= value <= hi:
                raise ValueError(f'corrected {name} exceeds joint limits')
        return solution.tolist()


class TrajectoryOffsetPlanner(OffsetPlanner):
    def __init__(self, node, order, frame, vector):
        super().__init__(node, order, frame, vector)
        from rcl_interfaces.srv import GetParameters
        self.description = node.create_client(GetParameters, '/move_group/get_parameters')
        self.model = None

    def initialize(self, reference):
        from rcl_interfaces.srv import GetParameters
        request = GetParameters.Request()
        request.names = ['robot_description']
        response = self.call(self.description, request, timeout=5.0)
        if len(response.values) != 1 or not response.values[0].string_value:
            raise ValueError('MoveIt robot_description is unavailable')
        model = CartesianOffsetModel(response.values[0].string_value, self.order, self.frame)
        actual, _ = model.fk_jacobian(reference)
        pose = self.pose(reference).pose
        if np.linalg.norm(actual[:3, 3]-np.array([pose.position.x, pose.position.y, pose.position.z])) > .0001:
            raise ValueError('local kinematic model does not match MoveIt FK')
        # Compare the local and MoveIt solutions for the initial correction.
        local = model.translate(reference, self.vector)
        reached, _ = model.fk_jacobian(local)
        # Orientation comparison uses the MoveIt quaternion directly.
        q = pose.orientation
        x, y, z, w = q.x, q.y, q.z, q.w
        norm = math.sqrt(x*x+y*y+z*z+w*w)
        if norm == 0:
            raise ValueError('MoveIt returned invalid orientation')
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        orientation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        if np.linalg.norm(rotation_error(orientation, reached[:3, :3])) > .001:
            raise ValueError('local kinematic orientation differs from MoveIt FK')
        self.model = model

    def plan_actions(self, actions, measured):
        from moveit_msgs.srv import GetStateValidity
        if not actions:
            raise ValueError('empty trajectory')
        measured = checked_joints(measured)
        if self.model is None:
            self.initialize(measured)
        corrected = []
        for action in actions:
            if len(action) != 7 or not all(math.isfinite(float(x)) for x in action):
                raise ValueError('invalid policy action')
            corrected.append([*self.model.translate(action[:6], self.vector), action[6]])
        previous = measured
        requests = []
        for index, action in enumerate(corrected):
            jump = max(abs(a-b) for a,b in zip(previous, action[:6]))
            if jump > (.05 if index == 0 else .15):
                raise ValueError('corrected trajectory exceeds the existing target-step envelope')
            steps = max(1, math.ceil(jump/.005))
            for step in range(1, steps+1):
                request = GetStateValidity.Request()
                request.group_name = 'ur_manipulator'
                request.robot_state.is_diff = True
                request.robot_state.joint_state.name = self.order
                request.robot_state.joint_state.position = [a+(b-a)*step/steps for a,b in zip(previous, action[:6])]
                requests.append(request)
            previous = action[:6]
        if not self.validity.service_is_ready():
            raise RuntimeError('MoveIt state-validity service unavailable')
        # Keep in-flight requests below the default ROS service queue depth.
        # Sending a whole chunk at once can drop requests under keep-last QoS.
        for first in range(0, len(requests), 4):
            completed = threading.Event()
            lock = threading.Lock()
            batch = requests[first:first+4]
            remaining = len(batch)
            rejected = []
            def done(future):
                nonlocal remaining
                try:
                    result = future.result()
                    if result is None or not result.valid:
                        rejected.append('collision/invalid state')
                except Exception as exc:
                    rejected.append(str(exc))
                with lock:
                    remaining -= 1
                    if remaining == 0:
                        completed.set()
            futures = [self.validity.call_async(request) for request in batch]
            for future in futures:
                future.add_done_callback(done)
            if not completed.wait(2.):
                for future in futures:
                    future.cancel()
                raise TimeoutError('continuous offset path checking timed out')
            if rejected:
                raise ValueError('MoveIt rejected the continuously offset path: '+rejected[0])
        return corrected, {'frame':self.frame, 'offset_m':self.vector, 'actions':len(corrected), 'path_checks':len(requests)}
