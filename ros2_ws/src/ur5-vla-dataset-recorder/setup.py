from glob import glob
from setuptools import find_packages, setup


package_name = "ur5_vla_dataset_recorder"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Grasp Lab",
    maintainer_email="maintainer@example.com",
    description="Safe synchronized staging-dataset recorder for UR5 VLA demonstrations.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dataset_recorder = ur5_vla_dataset_recorder.recorder_node:main",
            "logitech_servo = ur5_vla_dataset_recorder.logitech_servo_node:main",
            "convert_lerobot = ur5_vla_dataset_recorder.convert_lerobot:main",
            "validate_lerobot = ur5_vla_dataset_recorder.validate_lerobot:main",
            "offline_policy_check = ur5_vla_dataset_recorder.offline_inference:main",
            "policy_executor = ur5_vla_dataset_recorder.policy_executor:main",
            "policy_live_executor = ur5_vla_dataset_recorder.policy_live_executor:main",
            "policy_episode_executor = ur5_vla_dataset_recorder.policy_episode_executor:main",
            "move_to_standard_pose = ur5_vla_dataset_recorder.standard_pose_node:main",
            "prepare_dataset_start = ur5_vla_dataset_recorder.standard_pose_node:prepare_dataset_start",
            "package_lerobot_dataset = ur5_vla_dataset_recorder.package_dataset:main",
            "install_lerobot_checkpoint = ur5_vla_dataset_recorder.install_checkpoint:main",
            "split_lerobot_archive = ur5_vla_dataset_recorder.split_archive:main",
        ],
    },
)
