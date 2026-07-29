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

"""Zealot legs-only locomotion controller for the Unitree G1.

Runs a policy trained in `zealot` (nexus GPU physics) as the G1's locomotion
layer. The policy is a 12-action legs-only MLP over a 45-dim observation frame
stacked 5 deep; the upper body is PD-held and is NOT actuated here, so this
controller composes with an arm teleoperator the way the other legs-only
controllers do.

Weights load straight from the trainer's `.safetensors` (no ONNX export): the
checkpoint carries both the MLP and the Welford observation normalizer.

Weights resolve in this order: an explicit `policy_path=`, then
`ZEALOT_POLICY_PATH` (a local file), then the Hub — repo
`ZEALOT_POLICY_REPO` / file `ZEALOT_POLICY_FILE`, defaulting to the
released checkpoint. The Hub path is the one to prefer: it pins the
checkpoint by name so a controller and the policy it was validated with
travel together.

Conventions replicated exactly from the training env (`biped_env_nexus.rs` /
`velocity_flat.rs`) — every one of these is load-bearing:

  obs45 = [last_action(12), cmd(4), q-default(12), qdot_fd(12),
           projected_gravity(3), sin 2*pi*phi, cos 2*pi*phi]
  * `last_action` is LAG-2 (the frame at decision t carries the action from
    t-2), zeros for the first two steps of an episode.
  * `qdot` is the FINITE DIFFERENCE (q_t - q_{t-1}) / control_dt, NOT the
    encoder's dq. The policy never saw encoder velocity: at 50 Hz the two
    differ by more than 1 rad/s whenever the stance dithers, so feeding
    `motor_state.dq` here is a silent distribution shift.
  * gait clock is derived from the command, with no free-running phase:
    frozen while |cmd| < 0.1 (yaw included in the magnitude), otherwise the
    period lerps 0.8 s -> 0.55 s as speed goes 0.1 -> 0.5.
  * target = clamp(default + 0.5 * action, joint range), and the PD gains
    below must match the ones the policy trained with.

The G1's joint indices for the legs (`G1_29_JointIndex` 0-11) are already in
zealot's canonical order, so no remapping is needed.
"""

from __future__ import annotations

import json
import logging
import os
import struct

import numpy as np
from huggingface_hub import hf_hub_download

from .g1_utils import (
    REMOTE_AXES,
    G1_29_JointIndex,
    get_gravity_orientation,
)

logger = logging.getLogger(__name__)

# --- policy contract ------------------------------------------------------
NUM_LEG_JOINTS = 12
OBS_FRAME = 45
OBS_HISTORY = 5
CONTROL_DT = 0.02  # 50 Hz, zealot's control rate (decimation 4 x 5 ms physics)
ACTION_SCALE = 0.5

# Leg default pose, zealot canonical order == G1 indices 0-11.
DEFAULT_LEG = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2, dtype=np.float32)

# Per-joint position limits (official G1 URDF), used to clamp PD targets
# exactly as the trainer does. Hip roll is left/right asymmetric.
LEG_LIMITS = np.array(
    [
        (-2.5307, 2.8798),  # L hip pitch
        (-0.5236, 2.9671),  # L hip roll
        (-2.7576, 2.7576),  # L hip yaw
        (-0.087267, 2.8798),  # L knee
        (-0.87267, 0.5236),  # L ankle pitch
        (-0.2618, 0.2618),  # L ankle roll
        (-2.5307, 2.8798),  # R hip pitch
        (-2.9671, 0.5236),  # R hip roll (mirrored)
        (-2.7576, 2.7576),  # R hip yaw
        (-0.087267, 2.8798),  # R knee
        (-0.87267, 0.5236),  # R ankle pitch
        (-0.2618, 0.2618),  # R ankle roll
    ],
    dtype=np.float32,
)

# Gait clock (command-derived, no knobs) — mirrors the trainer.
GAIT_PERIOD_SLOW = 0.8
GAIT_PERIOD_FAST = 0.55
STANDING_SPEED = 0.1

# Command ranges the policy was trained on. Joystick axes map onto these;
# anything beyond is extrapolation the policy has never seen.
CMD_VX = 0.5
CMD_VY = 0.3
CMD_YAW = 0.6

# PD gains, zealot's `unitree_g1_agile` spec with the v19 ankle package
# (ankle kp 20 -> 40, kd 0.2 -> 2.0, matching the unitree_rl_gym deploy pair).
# A policy trained at one set of gains and run at another is a different robot,
# so these track the checkpoint, not the hardware defaults.
LEG_KP = np.array([100.0, 100.0, 100.0, 200.0, 40.0, 40.0] * 2, dtype=np.float32)
LEG_KD = np.array([2.5, 2.5, 2.5, 5.0, 2.0, 2.0] * 2, dtype=np.float32)
# Upper body: zealot's held-joint table (waist 12-14, arms 15-28).
HELD_KP = {"waist": 300.0, "shoulder_pitch": 90.0, "shoulder_roll": 60.0,
           "shoulder": 20.0, "elbow": 60.0, "wrist": 4.0}
HELD_KD = {"waist": 5.0, "shoulder_pitch": 2.0, "shoulder_roll": 1.0,
           "shoulder": 0.4, "elbow": 1.0, "wrist": 0.2}
# Upper-body pose the policy was TRAINED against (the playground "home"
# keyframe): arms bent, not hanging. This is part of the policy's world, not
# decoration -- the arms carry ~15 kg and straight-down arms move the CoM and
# the pendulum inertia the legs are balancing. Anything not listed holds 0.
HELD_HOME = {
    "shoulderpitch": 0.2,
    "leftshoulderroll": 0.2,
    "rightshoulderroll": -0.2,
    "elbow": 1.28,
}

# Default Hub location for the released checkpoint.
DEFAULT_REPO_ID = "haixuantao/zealot-g1-locomotion"
DEFAULT_FILENAME = "g1_v19_iter2740.safetensors"

# --- safety ---------------------------------------------------------------
# Blend from the pose the robot is actually in toward the policy's target over
# this many control steps, so engaging the controller can't step-jerk the legs.
RAMP_STEPS = 25  # 0.5 s
# Cap how far a target may move per control step (rad). At 50 Hz this bounds
# commanded joint speed to ~10 rad/s, well inside the hardware rating, and
# stops a single bad inference from slamming a joint into its endstop.
MAX_TARGET_DELTA = 0.2


def _load_safetensors(path: str) -> dict[str, np.ndarray]:
    """Minimal pure-numpy safetensors reader (no torch dependency)."""
    dtypes = {"F32": np.float32, "F64": np.float64, "I64": np.int64,
              "I32": np.int32, "U32": np.uint32, "F16": np.float16}
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        blob = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        arr = np.frombuffer(blob[start:end], dtype=dtypes[meta["dtype"]])
        out[name] = arr.reshape(meta["shape"])
    return out


class _Policy:
    """Welford observation normalizer + ELU MLP (deterministic mean action)."""

    def __init__(self, path: str):
        sd = _load_safetensors(path)
        self.weights, self.biases = [], []
        layer = 0
        while f"actor.w_{layer}" in sd:
            self.weights.append(sd[f"actor.w_{layer}"].astype(np.float64))
            self.biases.append(sd[f"actor.b_{layer}"].astype(np.float64))
            layer += 1
        if not self.weights:
            raise ValueError(f"no actor.w_* tensors in {path}")
        self.mean = sd["obs_norm.mean"].astype(np.float64)
        self.m2 = sd["obs_norm.m2"].astype(np.float64)
        self.count = float(sd["obs_norm.count"].reshape(-1)[0])
        self.obs_dim = self.weights[0].shape[1]
        self.act_dim = self.weights[-1].shape[0]
        if self.obs_dim != OBS_FRAME * OBS_HISTORY or self.act_dim != NUM_LEG_JOINTS:
            raise ValueError(
                f"checkpoint shape {self.obs_dim}->{self.act_dim}, expected "
                f"{OBS_FRAME * OBS_HISTORY}->{NUM_LEG_JOINTS}"
            )

    def act(self, obs: np.ndarray) -> np.ndarray:
        var = np.maximum(self.m2 / self.count, 1e-8)
        a = np.clip((obs - self.mean) / np.sqrt(var), -5.0, 5.0)
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = w @ a + b
            a = z if i == len(self.weights) - 1 else np.where(z > 0, z, np.expm1(z))
        return a


def gait_period_for(cmd_speed: float) -> float:
    """Cadence as a function of commanded speed (trainer's exact mapping)."""
    t = (min(abs(cmd_speed), 0.5) - STANDING_SPEED) / 0.4
    return GAIT_PERIOD_SLOW + (GAIT_PERIOD_FAST - GAIT_PERIOD_SLOW) * max(t, 0.0)


class ZealotLocomotionController:
    """Legs-only locomotion controller running a zealot-trained policy."""

    control_dt = CONTROL_DT  # read by unitree_g1.py to pace the control thread

    def __init__(self, policy_path: str | None = None, hold_upper_body: bool = True):
        path = policy_path or os.environ.get("ZEALOT_POLICY_PATH")
        if path:
            path = os.path.expanduser(path)
        else:
            repo_id = os.environ.get("ZEALOT_POLICY_REPO", DEFAULT_REPO_ID)
            filename = os.environ.get("ZEALOT_POLICY_FILE", DEFAULT_FILENAME)
            logger.info(f"Fetching zealot policy from the Hub: {repo_id}/{filename}")
            path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.policy = _Policy(path)
        logger.info(f"Zealot policy loaded from {path}")

        # Per-motor gains for all 29 joints; legs from the trained spec, upper
        # body from the held-joint table.
        self.kp = np.zeros(29, dtype=np.float32)
        self.kd = np.zeros(29, dtype=np.float32)
        self.kp[:NUM_LEG_JOINTS] = LEG_KP
        self.kd[:NUM_LEG_JOINTS] = LEG_KD
        for motor in G1_29_JointIndex:
            if motor.value < NUM_LEG_JOINTS:
                continue
            name = motor.name.lower()
            for frag, kp in HELD_KP.items():
                if frag in name:
                    self.kp[motor.value] = kp
                    self.kd[motor.value] = HELD_KD[frag]
                    break

        # Held-joint targets (waist + arms) at the trained home pose. The robot
        # layer only writes motor_cmd for joints present in the returned dict,
        # so joints we omit keep whatever was last commanded -- which at
        # startup is q=0, i.e. arms hanging straight down, NOT what the policy
        # trained with. Emitting them here keeps the mass distribution honest.
        # Set hold_upper_body=False when an arm teleoperator owns those joints.
        self.held_targets: dict[str, float] = {}
        if hold_upper_body:
            for motor in G1_29_JointIndex:
                if motor.value < NUM_LEG_JOINTS:
                    continue
                key = motor.name.lower().replace("_", "").replace("k", "", 1)
                target = 0.0
                for frag, val in HELD_HOME.items():
                    if frag in key:
                        target = val
                        break
                self.held_targets[f"{motor.name}.q"] = target

        self.reset()

    def reset(self) -> None:
        """Reset episode state (clock, history, action lag, ramp)."""
        self.cmd = np.zeros(4, dtype=np.float32)
        self.phase = 0.0
        self.frames: list[np.ndarray] | None = None
        self.act_hist = [np.zeros(NUM_LEG_JOINTS, dtype=np.float32)] * 2
        self.prev_q: np.ndarray | None = None
        self.prev_target: np.ndarray | None = None
        self.step_idx = 0

    def _command_from_remote(self, action: dict) -> None:
        lx, ly, rx, _ry = (float(action.get(k, 0.0)) for k in REMOTE_AXES)
        # Deadzone, then map the stick onto the TRAINED command ranges. Beyond
        # them the policy is extrapolating, which is where it falls over.
        dz = lambda v: v if abs(v) > 0.1 else 0.0  # noqa: E731
        self.cmd[0] = np.clip(dz(ly) * CMD_VX, -CMD_VX, CMD_VX)
        self.cmd[1] = np.clip(dz(-lx) * CMD_VY, -CMD_VY, CMD_VY)
        self.cmd[2] = np.clip(dz(-rx) * CMD_YAW, -CMD_YAW, CMD_YAW)
        self.cmd[3] = 0.0

    def run_step(self, action: dict, lowstate) -> dict:
        """One control step: joystick + lowstate -> leg position targets."""
        if lowstate is None:
            return {}

        self._command_from_remote(action)

        q = np.array(
            [lowstate.motor_state[i].q for i in range(NUM_LEG_JOINTS)], dtype=np.float32
        )
        # Finite-difference velocity, matching training (see module docstring).
        if self.prev_q is None:
            qdot = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)
        else:
            qdot = (q - self.prev_q) / CONTROL_DT
        self.prev_q = q.copy()

        gravity = np.asarray(
            get_gravity_orientation(lowstate.imu_state.quaternion), dtype=np.float32
        )

        # Command-derived clock: frozen when the command is a stand, so the
        # policy sees a distinct standing observation instead of a clock that
        # keeps waving "swing" at it.
        speed = float(np.linalg.norm(self.cmd[:3]))
        if speed >= STANDING_SPEED:
            self.phase = (self.phase + CONTROL_DT / gait_period_for(speed)) % 1.0

        frame = np.zeros(OBS_FRAME, dtype=np.float32)
        frame[0:12] = self.act_hist[0] if self.step_idx >= 2 else 0.0
        frame[12:16] = self.cmd
        frame[16:28] = q - DEFAULT_LEG
        frame[28:40] = qdot
        frame[40:43] = gravity
        frame[43] = np.sin(2 * np.pi * self.phase)
        frame[44] = np.cos(2 * np.pi * self.phase)

        # Reset-replicate the history on the first step, as the trainer does.
        if self.frames is None:
            self.frames = [frame.copy() for _ in range(OBS_HISTORY)]
        else:
            self.frames = self.frames[1:] + [frame.copy()]

        raw = self.policy.act(np.concatenate(self.frames).astype(np.float64))
        policy_action = np.clip(raw, -10.0, 10.0).astype(np.float32)
        self.act_hist = [self.act_hist[1], policy_action.copy()]

        target = np.clip(
            DEFAULT_LEG + ACTION_SCALE * policy_action,
            LEG_LIMITS[:, 0],
            LEG_LIMITS[:, 1],
        )

        # Safety 1: ease in from the measured pose so engaging the controller
        # never steps the legs discontinuously.
        if self.step_idx < RAMP_STEPS:
            alpha = (self.step_idx + 1) / RAMP_STEPS
            target = (1.0 - alpha) * q + alpha * target
        # Safety 2: slew-rate limit, so one bad inference cannot slam a joint.
        if self.prev_target is not None:
            target = np.clip(
                target,
                self.prev_target - MAX_TARGET_DELTA,
                self.prev_target + MAX_TARGET_DELTA,
            )
        self.prev_target = target.copy()
        self.step_idx += 1

        # Legs only: joints 12-14 (waist) and the arms stay wherever the
        # upper-body controller/teleop puts them.
        out = {
            f"{G1_29_JointIndex(i).name}.q": float(target[i])
            for i in range(NUM_LEG_JOINTS)
        }
        out.update(self.held_targets)
        return out
