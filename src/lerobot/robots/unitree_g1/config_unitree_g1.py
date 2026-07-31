#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

_GAINS: dict[str, dict[str, list[float]]] = {
    "left_leg": {
        "kp": [150, 150, 150, 300, 40, 40],
        "kd": [2, 2, 2, 4, 2, 2],
    },  # pitch, roll, yaw, knee, ankle_pitch, ankle_roll
    "right_leg": {"kp": [150, 150, 150, 300, 40, 40], "kd": [2, 2, 2, 4, 2, 2]},
    "waist": {"kp": [250, 250, 250], "kd": [5, 5, 5]},  # yaw, roll, pitch
    "left_arm": {"kp": [50, 50, 80, 80], "kd": [3, 3, 3, 3]},  # shoulder_pitch/roll/yaw, elbow
    "left_wrist": {"kp": [40, 40, 40], "kd": [1.5, 1.5, 1.5]},  # roll, pitch, yaw
    "right_arm": {"kp": [50, 50, 80, 80], "kd": [3, 3, 3, 3]},
    "right_wrist": {"kp": [40, 40, 40], "kd": [1.5, 1.5, 1.5]},
}


def _build_gains() -> tuple[list[float], list[float]]:
    """Build kp and kd lists from body-part groupings."""
    kp = [v for g in _GAINS.values() for v in g["kp"]]
    kd = [v for g in _GAINS.values() for v in g["kd"]]
    return kp, kd


_DEFAULT_KP, _DEFAULT_KD = _build_gains()


@RobotConfig.register_subclass("unitree_g1")
@dataclass
class UnitreeG1Config(RobotConfig):
    kp: list[float] = field(default_factory=lambda: _DEFAULT_KP.copy())
    kd: list[float] = field(default_factory=lambda: _DEFAULT_KD.copy())

    # Default joint positions
    default_positions: list[float] = field(default_factory=lambda: [0.0] * 29)

    # Control loop timestep
    control_dt: float = 1.0 / 250.0  # 250Hz

    # Launch mujoco simulation
    is_simulation: bool = True

    # Socket config for ZMQ bridge
    robot_ip: str = "192.168.123.164"  # default G1 IP

    # Run the locomotion / whole-body controller ONBOARD the robot (policy on the G1
    # itself, against local DDS at full rate) instead of on the laptop over the ZMQ
    # socket bridge. In this mode the robot object uses the real Unitree SDK channels
    # and expects high-level actions (arm targets + joystick axes, or 64-D SONIC
    # tokens) fed via send_action -- e.g. by run_g1_server's serve_onboard_controller,
    # which receives them from the laptop over ZMQ. Mutually exclusive with is_simulation.
    onboard: bool = False
    # DDS network interface for onboard mode (None = SDK default, matching
    # run_g1_server.py's ChannelFactoryInitialize(0)).
    dds_interface: str | None = None
    # Onboard sub-flags. On a real G1 both are True: the built-in motion services
    # must be released before we can write lowcmd, and locomotion axes are read from
    # the physical wireless remote. Against a DDS sim neither applies (no
    # MotionSwitcher, no physical remote), so set both False so the controller takes
    # its locomotion axes purely from send_action (ZMQ) input.
    release_motion_control: bool = True
    physical_remote: bool = True
    # Onboard-only: read locomotion axes from an XInput USB gamepad plugged into the
    # robot. Read over libusb rather than /dev/input/jsN on purpose -- an XInput pad
    # presents a vendor-specific (0xff) interface that only the `xpad` driver binds, and
    # the G1's Tegra kernel ships no xpad.ko, so no jsN node ever appears. Nothing claims
    # the interface, which is exactly what lets libusb take it. Needs a udev rule giving
    # the user access, e.g.
    #   SUBSYSTEM=="usb", ATTR{idVendor}=="2f24", ATTR{idProduct}=="008f", MODE="0660", GROUP="plugdev"
    # The physical Unitree remote still takes priority whenever it is active.
    usb_pad: bool = False
    usb_pad_id: str = "2f24:008f"
    # Deadman button index into the XInput buttons2 byte (0=LB, 1=RB, 4=A, 5=B, 6=X,
    # 7=Y); -1 disables it, which is the default. A deadman is NOT a stop here: a
    # locomotion policy keeps walking on a zero command (v24 creeps ~0.2 m/s), so
    # releasing it does not halt the robot -- only shutting the controller down does.
    # It costs a held finger and buys little, so it is opt-in.
    usb_pad_deadman: int = -1
    # Stick response shaping. expo>1 flattens the curve near centre for finer control,
    # but the controller applies its OWN deadzone downstream (0.1 for zealot), so expo
    # pushes the point where the robot first moves UP the travel: at expo=2 nothing
    # happens until 0.1**(1/2) = 0.32 of stick. Default 1.0 (linear) keeps the deadband
    # at the controller's own 0.1 -- raise it only if the low end still feels twitchy,
    # and expect to lose the bottom of the range in exchange.
    usb_pad_expo: float = 1.0
    # Time constant of a low-pass on the axes: takes the step out of a flicked stick
    # without moving where the response starts. OFF by default -- it trades away
    # responsiveness, and v24 already feels sluggish on hardware (it commands ~40%
    # smaller joint excursions than v21). Fix twitchiness with the range mapping first,
    # which costs no latency, and only add smoothing if the stick itself is noisy.
    usb_pad_smoothing_s: float = 0.0

    # Cameras (ZMQ-based remote cameras)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Compensates for gravity on the unitree's arms using the arm ik solver
    gravity_compensation: bool = False

    # Controller class name, e.g. GrootLocomotionController / HolosomaLocomotionController /
    # SonicWholeBodyController / ZealotLocomotionController. None disables it.
    controller: str | None = None
