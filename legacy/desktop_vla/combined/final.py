#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Header
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


# ----------------------------
# Arm joints (what you control with joystick)
# ----------------------------
CTRL_JOINTS = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',  # axes[5] (slow)
    'elbow_joint',          # axes[4] (slow like lift)
    'wrist_1_joint',        # buttons 2/3
    'wrist_3_joint',        # buttons 4/5
]

# Joints you lock for sim/control (NOT part of dataset action)
LOCK_JOINTS = ['wrist_2_joint']
PUBLISH_JOINTS = CTRL_JOINTS + LOCK_JOINTS

LOCK_WRIST_2 = 1.5708

# Reset pose (arm action server expects all 6 arm joints)
RESET_ARM_JOINTS = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint'
]
RESET_ARM_POSITIONS = [0.3932, -1.0305, -2.0361, -1.0351, 1.5708, 0.0]

# Gripper (action server)
GRIPPER_JOINTS = ['left_inner_finger_joint', 'right_inner_finger_joint']


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class UR5TeleopDataset(Node):
    def __init__(self):
        super().__init__('ur5_teleop_dataset')

        # Publishers / Subscribers
        self.arm_pub = self.create_publisher(JointState, '/joint_command', 10)
        self.ds_pub  = self.create_publisher(JointState, '/dataset_joint_commands', 10)  # ✅ dataset actions

        self.create_subscription(JointState, '/joint_states', self.on_js, 10)
        self.create_subscription(Joy, '/joy', self.on_joy, 10)

        # Action clients
        self.arm_client  = ActionClient(self, FollowJointTrajectory, '/ur5/follow_joint_trajectory')
        self.grip_client = ActionClient(self, FollowJointTrajectory, '/ur5_gripper/follow_joint_trajectory')

        # Latest joint state
        self.latest = {}

        # Teleop target for CTRL_JOINTS only (5 joints)
        self.q_cmd = None

        # Joystick state
        self.axes = []
        self.buttons = []
        self.prev_buttons = []

        # ✅ Dataset action message layout:
        #   name = CTRL_JOINTS + ['gripper_cmd']
        #   position = q_cmd(5) + [0/1]
        self.dataset_names = list(CTRL_JOINTS) + ['gripper_cmd']
        self.gripper_cmd = 0  # 0=open, 1=close

        # Control frequency
        self.rate_hz = 20.0
        self.dt = 1.0 / self.rate_hz
        self.timer = self.create_timer(self.dt, self.update_arm)

        # Tuning
        self.deadzone = 0.07
        self.expo = 1.6

        self.arm_speed = 0.9
        self.lift_speed_scale = 0.20
        self.elbow_speed_scale = 0.20

        self.wrist1_speed = 0.8
        self.wrist3_speed = 1.2

        self.ctrl_limits = [(-2 * math.pi, 2 * math.pi)] * 5

        # Axis mapping (your controller)
        self.AX_PAN = 0
        self.AX_LIFT = 5   # shoulder_lift
        self.AX_ELBOW = 4  # elbow

        # axis_idx -> (q_cmd_index, sign)
        self.axis_map = {
            self.AX_PAN:   (0, +1.0),  # shoulder_pan
            self.AX_LIFT:  (1, -1.0),  # shoulder_lift invert
            self.AX_ELBOW: (2, +1.0),  # elbow
        }

        # Buttons (Xbox-style)
        self.btn_reset = 7
        self.btn_grip_close = 0
        self.btn_grip_open = 1

        self.btn_wrist1_neg = 2
        self.btn_wrist1_pos = 3

        self.btn_wrist3_neg = 4
        self.btn_wrist3_pos = 5

        # State flags
        self.in_reset = False
        self.reset_goal_active = False
        self.grip_goal_active = False
        self.wait_center_after_reset = False

        # Action timing
        self.reset_time = 3.0
        self.reset_goal_tol = 2

        self.grip_time = 1.0
        self.grip_goal_tol = 1
        self.grip_open_pos = 0.0
        self.grip_close_pos = 0.5

        self._reset_timer = None

        self.get_logger().info(
            "Teleop ready:\n"
            " START(7)=RESET\n"
            " A(0)=close gripper, B(1)=open gripper\n"
            " Elbow = axes[4] (slow)\n"
            " Shoulder lift = axes[5] (slow)\n"
            " Buttons 2/3 = wrist_1 -/+\n"
            " Buttons 4/5 = wrist_3 -/+\n"
            " Dataset actions on /dataset_joint_commands (CTRL_JOINTS + gripper_cmd)\n"
        )

    # ----------------------------
    # Callbacks
    # ----------------------------
    def on_js(self, msg: JointState):
        for n, p in zip(msg.name, msg.position):
            self.latest[n] = float(p)

        # Initialize teleop targets from the actual robot pose
        if self.q_cmd is None and all(j in self.latest for j in CTRL_JOINTS):
            self.q_cmd = [self.latest[j] for j in CTRL_JOINTS]
            self.get_logger().info("Initialized q_cmd from /joint_states.")

    def on_joy(self, msg: Joy):
        self.axes = list(msg.axes)
        self.buttons = list(msg.buttons)

        if not self.prev_buttons:
            self.prev_buttons = [0] * len(self.buttons)

        if self.rising_edge(self.btn_reset):
            self.reset_sequence()

        self.handle_gripper_buttons()

        self.prev_buttons = list(self.buttons)

    # ----------------------------
    # Helpers
    # ----------------------------
    def rising_edge(self, i: int) -> bool:
        if i >= len(self.buttons) or i >= len(self.prev_buttons):
            return False
        return self.buttons[i] == 1 and self.prev_buttons[i] == 0

    def axis_centered(self) -> bool:
        if not self.axes:
            return True
        for a in [self.AX_PAN, self.AX_LIFT, self.AX_ELBOW]:
            if a < len(self.axes) and abs(self.axes[a]) > self.deadzone:
                return False
        return True

    def shape(self, v: float) -> float:
        if abs(v) < self.deadzone:
            return 0.0
        return (abs(v) ** self.expo) * (1.0 if v >= 0 else -1.0)

    def btn(self, idx: int) -> int:
        if idx < 0 or idx >= len(self.buttons):
            return 0
        return 1 if self.buttons[idx] == 1 else 0

    # ----------------------------
    # Main loop
    # ----------------------------
    def update_arm(self):
        if self.in_reset or self.q_cmd is None:
            return

        # After reset: require sticks centered once
        if self.wait_center_after_reset:
            if self.axis_centered():
                self.wait_center_after_reset = False
                self.get_logger().info("Sticks centered. Teleop active again.")
            else:
                self.publish_hold()
                return

        if not self.axes and not self.buttons:
            self.publish_hold()
            return

        # Axes -> shoulder_pan, shoulder_lift, elbow
        for axis_idx, (jidx, sign) in self.axis_map.items():
            if axis_idx < len(self.axes):
                v = self.shape(self.axes[axis_idx])

                speed = self.arm_speed
                if jidx == 1:  # shoulder_lift
                    speed = self.arm_speed * self.lift_speed_scale
                elif jidx == 2:  # elbow
                    speed = self.arm_speed * self.elbow_speed_scale

                self.q_cmd[jidx] += sign * speed * v * self.dt

        # Buttons -> wrist_1 (q_cmd index 3)
        wrist1_dir = (self.btn(self.btn_wrist1_pos) - self.btn(self.btn_wrist1_neg))
        if wrist1_dir != 0:
            self.q_cmd[3] += wrist1_dir * self.wrist1_speed * self.dt

        # Buttons -> wrist_3 (q_cmd index 4)
        wrist3_dir = (self.btn(self.btn_wrist3_pos) - self.btn(self.btn_wrist3_neg))
        if wrist3_dir != 0:
            self.q_cmd[4] += wrist3_dir * self.wrist3_speed * self.dt

        # Clamp
        for i in range(5):
            lo, hi = self.ctrl_limits[i]
            self.q_cmd[i] = clamp(self.q_cmd[i], lo, hi)

        self.publish_hold()

    def publish_hold(self):
        # Publish actual sim control topic (/joint_command) including locked wrist_2
        cmd = JointState()
        cmd.header = Header()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = PUBLISH_JOINTS
        cmd.position = list(self.q_cmd) + [LOCK_WRIST_2]
        self.arm_pub.publish(cmd)

        # ✅ Publish dataset-only actions (CTRL_JOINTS + gripper_cmd)
        ds = JointState()
        ds.header = Header()
        ds.header.stamp = cmd.header.stamp  # keep same timestamp
        ds.name = self.dataset_names
        ds.position = list(self.q_cmd) + [float(self.gripper_cmd)]
        self.ds_pub.publish(ds)

    # ----------------------------
    # RESET SEQUENCE
    # ----------------------------
    def reset_sequence(self):
        if self.reset_goal_active:
            self.get_logger().warn("Reset already running.")
            return
        if not self.arm_client.server_is_ready():
            self.get_logger().warn("Arm action server not ready.")
            return

        self.in_reset = True
        self.reset_goal_active = True
        self.wait_center_after_reset = False

        # Reset should always open gripper
        self.open_gripper_for_reset()

        if self._reset_timer is not None:
            self._reset_timer.cancel()
        self._reset_timer = self.create_timer(0.2, self._send_arm_reset_once)

        self.get_logger().info("RESET: opening gripper then moving arm to preset pose...")

    def open_gripper_for_reset(self):
        # ✅ Update dataset gripper command immediately
        self.gripper_cmd = 0

        if not self.grip_client.server_is_ready():
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = GRIPPER_JOINTS
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()

        pt = JointTrajectoryPoint()
        pt.positions = [self.grip_open_pos, self.grip_open_pos]
        pt.velocities = [0.0, 0.0]
        pt.time_from_start.sec = int(self.grip_time)
        pt.time_from_start.nanosec = 0

        goal.trajectory.points = [pt]
        goal.goal_time_tolerance.sec = int(self.grip_goal_tol)

        self.grip_client.send_goal_async(goal)

    def _send_arm_reset_once(self):
        if self._reset_timer is not None:
            self._reset_timer.cancel()
            self._reset_timer = None

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = RESET_ARM_JOINTS
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()

        pt = JointTrajectoryPoint()
        pt.positions = list(RESET_ARM_POSITIONS)
        pt.velocities = [0.0] * len(RESET_ARM_POSITIONS)

        t = float(self.reset_time)
        pt.time_from_start.sec = int(t)
        pt.time_from_start.nanosec = int((t - int(t)) * 1e9)

        goal.trajectory.points = [pt]
        goal.goal_time_tolerance.sec = int(self.reset_goal_tol)

        fut = self.arm_client.send_goal_async(goal)
        fut.add_done_callback(self._reset_goal_response_cb)

    def _reset_goal_response_cb(self, fut):
        gh = fut.result()
        if not gh or not gh.accepted:
            self.get_logger().warn("RESET: arm goal rejected.")
            self.reset_goal_active = False
            self.in_reset = False
            return
        gh.get_result_async().add_done_callback(self._reset_done_cb)

    def _reset_done_cb(self, fut):
        # Sync q_cmd from actual /joint_states pose (prevents weird rotation)
        if all(j in self.latest for j in CTRL_JOINTS):
            self.q_cmd = [self.latest[j] for j in CTRL_JOINTS]
        else:
            # Fallback only
            self.q_cmd = [
                RESET_ARM_POSITIONS[0],
                RESET_ARM_POSITIONS[1],
                RESET_ARM_POSITIONS[2],
                RESET_ARM_POSITIONS[3],
                RESET_ARM_POSITIONS[5],
            ]

        self.reset_goal_active = False
        self.in_reset = False
        self.wait_center_after_reset = True
        self.get_logger().info("RESET done. Synced q_cmd from /joint_states. Center sticks once -> teleop resumes.")

    # ----------------------------
    # Gripper control (A/B)
    # ----------------------------
    def handle_gripper_buttons(self):
        if self.grip_goal_active:
            return
        if not self.grip_client.server_is_ready():
            return

        if self.rising_edge(self.btn_grip_close):
            self.send_gripper_goal(self.grip_close_pos)
        elif self.rising_edge(self.btn_grip_open):
            self.send_gripper_goal(self.grip_open_pos)

    def send_gripper_goal(self, pos: float):
        # ✅ Update dataset gripper command immediately (intent)
        self.gripper_cmd = 1 if pos > 0.25 else 0

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = GRIPPER_JOINTS
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()

        pt = JointTrajectoryPoint()
        pt.positions = [pos, pos]
        pt.velocities = [0.0, 0.0]
        pt.time_from_start.sec = int(self.grip_time)
        pt.time_from_start.nanosec = 0

        goal.trajectory.points = [pt]
        goal.goal_time_tolerance.sec = int(self.grip_goal_tol)

        self.grip_goal_active = True
        fut = self.grip_client.send_goal_async(goal)
        fut.add_done_callback(self._grip_resp_cb)

    def _grip_resp_cb(self, fut):
        gh = fut.result()
        if not gh or not gh.accepted:
            self.grip_goal_active = False
            return
        gh.get_result_async().add_done_callback(self._grip_done_cb)

    def _grip_done_cb(self, fut):
        self.grip_goal_active = False


def main():
    rclpy.init()
    node = UR5TeleopDataset()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

