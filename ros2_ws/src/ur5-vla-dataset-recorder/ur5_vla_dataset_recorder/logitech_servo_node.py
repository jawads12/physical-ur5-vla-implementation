"""Original Logitech teleoperation behavior plus dataset gripper topics."""

import os
import signal
import subprocess
import threading
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64
from ur_msgs.msg import IOStates
from ur_msgs.srv import SetIO


# Logitech joystick mapping
DEADMAN_BUTTON = 0
GRIPPER_CLOSE_BUTTON = 1
GRIPPER_OPEN_BUTTON = 2
Z_DOWN_BUTTON = 4
Z_UP_BUTTON = 5
STANDARD_POSE_BUTTON = 6  # Physical button 7; Joy indexes are zero-based.
STANDARD_POSE_HOLD_SECONDS = 2.0

AXIS_LEFT_RIGHT = 0
AXIS_FORWARD_BACKWARD = 1
AXIS_HANDLE_TWIST = 2

# Conservative Cartesian speeds for real-robot teleoperation.
LINEAR_SPEED = 0.30
ANGULAR_SPEED = 0.50

# Keep joystick directions fixed relative to the robot base instead of rotating
# them with the tool pose.
COMMAND_FRAME = "base_link"

# UR digital outputs
IO_FUNCTION_SET_DIGITAL_OUT = 1
GRIPPER_CLOSE_PIN = 0
GRIPPER_OPEN_PIN = 1
OUTPUT_ON = 1.0
OUTPUT_OFF = 0.0

# Dataset convention
GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 1.0


class LogitechServo(Node):
    def __init__(self):
        super().__init__("logitech_servo_mapper")

        self.twist_publisher = self.create_twist_publisher()

        self.joy_subscription = self.create_subscription(
            Joy,
            "/joy",
            self.joy_callback,
            qos_profile_sensor_data,
        )

        self.io_client = self.create_client(
            SetIO,
            "/io_and_status_controller/set_io",
        )
        self.io_state_subscription = self.create_subscription(
            IOStates,
            "/io_and_status_controller/io_states",
            self.io_state_callback,
            qos_profile_sensor_data,
        )

        # Added only for dataset recording. Transient-local QoS preserves the
        # latest value for a recorder that starts after the mapper.
        gripper_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.gripper_command_publisher = self.create_publisher(
            Float64,
            "/gripper/command",
            gripper_qos,
        )
        self.gripper_state_publisher = self.create_publisher(
            Float64,
            "/gripper/state",
            gripper_qos,
        )

        self.previous_buttons = []
        self.last_gripper_value = None
        self.standard_pose_hold_started = None
        self.standard_pose_latched = False
        self.standard_pose_in_progress = False
        self.standard_pose_result = None
        self.standard_pose_process = None
        self.standard_pose_monitor = self.create_timer(0.1, self.monitor_standard_pose)
        self.gripper_heartbeat = self.create_timer(
            0.05,
            self.publish_gripper_heartbeat,
        )

        self.get_logger().info("Logitech UR5 teleoperation started.")
        self.get_logger().info("Hold trigger to move the arm.")
        self.get_logger().info(
            f"Command frame={COMMAND_FRAME} | linear speed={LINEAR_SPEED:.2f} m/s "
            f"| angular speed={ANGULAR_SPEED:.2f} rad/s"
        )
        self.get_logger().info(
            "Button index 1: CLOSE gripper | Button index 2: OPEN gripper"
        )
        self.get_logger().info(
            "Hold physical button 7 for 2 seconds with deadman released: "
            "guarded standard-pose reset, then resume Servo"
        )
        self.get_logger().info(
            "Operate the gripper once before recording to initialize its value."
        )

        if self.io_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("UR SetIO service is available.")
        else:
            self.get_logger().warning(
                "UR SetIO service is not currently available."
            )

    def create_twist_publisher(self):
        return self.create_publisher(
            TwistStamped,
            "/servo_node/delta_twist_cmds",
            10,
        )

    @staticmethod
    def axis(message: Joy, index: int) -> float:
        if 0 <= index < len(message.axes):
            return float(message.axes[index])
        return 0.0

    @staticmethod
    def button(message: Joy, index: int) -> int:
        if 0 <= index < len(message.buttons):
            return int(message.buttons[index])
        return 0

    def publish_gripper_value(self, value: float) -> None:
        self.last_gripper_value = float(value)
        message = Float64()
        message.data = self.last_gripper_value
        self.gripper_command_publisher.publish(message)
        self.gripper_state_publisher.publish(message)

    def publish_gripper_heartbeat(self) -> None:
        if self.last_gripper_value is not None:
            self.publish_gripper_value(self.last_gripper_value)

    def io_state_callback(self, message: IOStates) -> None:
        outputs = {int(item.pin): bool(item.state) for item in message.digital_out_states}
        close_active = outputs.get(GRIPPER_CLOSE_PIN, False)
        open_active = outputs.get(GRIPPER_OPEN_PIN, False)
        if open_active and not close_active:
            self.last_gripper_value = GRIPPER_OPEN
        elif close_active and not open_active:
            self.last_gripper_value = GRIPPER_CLOSED

    def send_io(
        self,
        pin: int,
        state: float,
        description: str,
        dataset_value=None,
    ) -> None:
        if not self.io_client.service_is_ready():
            self.get_logger().warning(
                "Cannot control gripper: SetIO service unavailable."
            )
            return

        request = SetIO.Request()
        request.fun = IO_FUNCTION_SET_DIGITAL_OUT
        request.pin = pin
        request.state = float(state)

        self.get_logger().info(f"{description}: pin={pin}, state={state}")

        future = self.io_client.call_async(request)
        future.add_done_callback(
            lambda completed_future: self.io_response_callback(
                completed_future,
                description,
                dataset_value,
            )
        )

    def io_response_callback(
        self,
        future,
        description: str,
        dataset_value,
    ) -> None:
        try:
            response = future.result()

            if response.success:
                self.get_logger().info(
                    f"{description} completed successfully."
                )
                if dataset_value is not None:
                    self.publish_gripper_value(dataset_value)
                    self.get_logger().info(
                        f"Dataset gripper value={dataset_value:.1f}"
                    )
            else:
                self.get_logger().warning(
                    f"{description} was rejected by the UR controller."
                )

        except Exception as error:
            self.get_logger().error(f"{description} failed: {error}")

    def open_gripper(self) -> None:
        # Original I/O behavior: disable CLOSE, then activate OPEN.
        self.send_io(
            GRIPPER_CLOSE_PIN,
            OUTPUT_OFF,
            "Disable CLOSE output",
        )
        self.send_io(
            GRIPPER_OPEN_PIN,
            OUTPUT_ON,
            "OPEN gripper",
            dataset_value=GRIPPER_OPEN,
        )

    def close_gripper(self) -> None:
        # Original I/O behavior: disable OPEN, then activate CLOSE.
        self.send_io(
            GRIPPER_OPEN_PIN,
            OUTPUT_OFF,
            "Disable OPEN output",
        )
        self.send_io(
            GRIPPER_CLOSE_PIN,
            OUTPUT_ON,
            "CLOSE gripper",
            dataset_value=GRIPPER_CLOSED,
        )

    def process_gripper_buttons(self, joy: Joy) -> None:
        if len(self.previous_buttons) != len(joy.buttons):
            self.previous_buttons = [0] * len(joy.buttons)

        close_pressed = self.button(joy, GRIPPER_CLOSE_BUTTON) == 1
        open_pressed = self.button(joy, GRIPPER_OPEN_BUTTON) == 1

        close_was_pressed = (
            self.previous_buttons[GRIPPER_CLOSE_BUTTON] == 1
            if GRIPPER_CLOSE_BUTTON < len(self.previous_buttons)
            else False
        )
        open_was_pressed = (
            self.previous_buttons[GRIPPER_OPEN_BUTTON] == 1
            if GRIPPER_OPEN_BUTTON < len(self.previous_buttons)
            else False
        )

        if close_pressed and not close_was_pressed:
            self.close_gripper()

        if open_pressed and not open_was_pressed:
            self.open_gripper()

        self.previous_buttons = list(joy.buttons)

    def process_standard_pose_button(self, joy: Joy) -> bool:
        """Return True while reset handling owns and suppresses teleoperation."""
        pressed = self.button(joy, STANDARD_POSE_BUTTON) == 1
        deadman_pressed = self.button(joy, DEADMAN_BUTTON) == 1

        if self.standard_pose_in_progress:
            return True

        if not pressed:
            self.standard_pose_hold_started = None
            self.standard_pose_latched = False
            return False

        if deadman_pressed:
            self.standard_pose_hold_started = None
            return False

        now = time.monotonic()
        if self.standard_pose_hold_started is None:
            self.standard_pose_hold_started = now
            self.get_logger().warning(
                "Standard-pose reset requested: keep physical button 7 held; "
                "deadman must remain released"
            )
            return True

        if (
            not self.standard_pose_latched
            and now - self.standard_pose_hold_started >= STANDARD_POSE_HOLD_SECONDS
        ):
            self.standard_pose_latched = True
            self.begin_standard_pose_reset()
        return True

    def begin_standard_pose_reset(self) -> None:
        if self.standard_pose_in_progress:
            return
        zero = TwistStamped()
        zero.header.stamp = self.get_clock().now().to_msg()
        zero.header.frame_id = COMMAND_FRAME
        if self.twist_publisher is not None:
            self.twist_publisher.publish(zero)
            self.destroy_publisher(self.twist_publisher)
            self.twist_publisher = None
        self.standard_pose_in_progress = True
        self.standard_pose_result = None
        self.get_logger().warning(
            "TELEOP DISABLED: executing guarded standard-pose reset"
        )
        threading.Thread(target=self._run_standard_pose_reset, daemon=True).start()

    def _run_standard_pose_reset(self) -> None:
        try:
            # Allow DDS discovery to remove this mapper's Twist endpoint before
            # the guarded reset process checks for competing publishers.
            time.sleep(1.0)
            command = [
                "/opt/ros/humble/bin/ros2",
                "run",
                "ur5_vla_dataset_recorder",
                "prepare_dataset_start",
            ]
            self.standard_pose_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            output, _ = self.standard_pose_process.communicate()
            self.standard_pose_result = (
                int(self.standard_pose_process.returncode),
                output.strip(),
            )
        except Exception as error:
            self.standard_pose_result = (1, f"failed to start reset command: {error}")
        finally:
            self.standard_pose_process = None

    def monitor_standard_pose(self) -> None:
        if not self.standard_pose_in_progress or self.standard_pose_result is None:
            return
        return_code, output = self.standard_pose_result
        self.standard_pose_result = None
        self.standard_pose_in_progress = False
        if self.twist_publisher is None:
            self.twist_publisher = self.create_twist_publisher()
        if return_code == 0:
            self.get_logger().info(
                "STANDARD POSE READY: Servo teleoperation publisher restored"
            )
        else:
            self.get_logger().error(
                f"STANDARD POSE RESET FAILED (code {return_code}); teleoperation "
                f"publisher restored but controller state must be checked. Output: {output}"
            )

    def stop_standard_pose_process(self) -> None:
        process = self.standard_pose_process
        if process is None or process.poll() is not None:
            return
        self.get_logger().error("Mapper stopping: interrupting active standard-pose reset")
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2.0)

    def joy_callback(self, joy: Joy) -> None:
        if self.process_standard_pose_button(joy):
            return
        self.process_gripper_buttons(joy)

        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = COMMAND_FRAME

        deadman_pressed = self.button(joy, DEADMAN_BUTTON) == 1

        if deadman_pressed:
            command.twist.linear.x = (
                -self.axis(joy, AXIS_FORWARD_BACKWARD) * LINEAR_SPEED
            )
            command.twist.linear.y = (
                self.axis(joy, AXIS_LEFT_RIGHT) * LINEAR_SPEED
            )
            command.twist.linear.z = (
                self.button(joy, Z_UP_BUTTON)
                - self.button(joy, Z_DOWN_BUTTON)
            ) * LINEAR_SPEED
            command.twist.angular.z = (
                self.axis(joy, AXIS_HANDLE_TWIST) * ANGULAR_SPEED
            )

        # Releasing the trigger publishes zero motion.
        if self.twist_publisher is not None:
            self.twist_publisher.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = LogitechServo()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping Logitech UR5 teleoperation.")
    finally:
        node.stop_standard_pose_process()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
