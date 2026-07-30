#!/usr/bin/env python
"""Bring-up script: walk the G1 a fixed distance forward, then a fixed distance
backward, under the zealot locomotion policy.

There is NO odometry in this path: distance is dead-reckoned as
commanded_speed * time. Phase durations are computed from the requested
distances and the capped command speed (ZEALOT_MAX_VX, default 0.2 m/s), so
4 m forward at 0.2 m/s runs the forward phase for ~20 s. The policy is known
to overshoot its command, so expect the real distance to run LONG -- pace it
out before trusting it near a wall.

Still in place from the first-run version:

  * forward speed capped in the controller (ZEALOT_MAX_VX, default 0.2 m/s);
  * a zero-command hold before, between, and after the walk phases;
  * `finally: robot.disconnect()`, so a crash or Ctrl-C still ends with the
    motors released;
  * `--dry-run` runs the identical sequence against the MuJoCo sim.

IMPORTANT, and measured rather than assumed:
  * Releasing the stick does NOT stop this policy -- at zero command it keeps
    walking at ~0.26 m/s. Phase duration is the stop, not the command.
  * The robot is NEVER released while nobody is holding it. After the
    sequence completes the policy keeps balancing (and creeping -- see above)
    and the script waits for the operator to take the robot's weight and
    press Enter before releasing.
  * The release is Unitree damping mode (kp=0, kd=8), not zero-gain: the
    robot sinks instead of free-falling. It still does not hold itself up --
    take its weight before releasing. Use a gantry.
  * Ctrl-C is the operator stop: it skips any remaining phases (and the Enter
    prompt) and releases immediately, damped.
  * The control loop runs on THIS machine over the network, not onboard.

Usage:
    python run_zealot_g1.py --dry-run                 # simulation
    python run_zealot_g1.py --robot-ip 192.168.123.164  # hardware
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config
from lerobot.robots.unitree_g1.zealot_locomotion import CMD_VX

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
log = logging.getLogger("zealot-bringup")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="run in MuJoCo instead of on hardware")
    ap.add_argument("--robot-ip", default="192.168.123.164")
    ap.add_argument("--forward-m", type=float, default=2.0, help="distance to walk forward (m, dead-reckoned)")
    ap.add_argument("--backward-m", type=float, default=2.0, help="distance to walk backward (m, dead-reckoned)")
    ap.add_argument("--settle-seconds", type=float, default=1.0, help="zero-command hold before/between/after")
    ap.add_argument("--vx", type=float, default=1.0,
                    help="stick deflection magnitude 0-1 (direction comes from the phase)")
    args = ap.parse_args()

    # Tolerate an ssh-style "user@host" for --robot-ip: ZMQ needs a bare host,
    # and pasting the ssh target is the obvious mistake to make.
    robot_ip = args.robot_ip.split("@", 1)[-1].strip()
    if robot_ip != args.robot_ip:
        log.warning("stripped user prefix from --robot-ip: %r -> %r", args.robot_ip, robot_ip)

    cfg = UnitreeG1Config(
        is_simulation=args.dry_run,
        robot_ip=robot_ip,
        controller="ZealotLocomotionController",
    )
    robot = UnitreeG1(cfg)

    log.info("=" * 62)
    log.info("ZEALOT G1 BRING-UP  mode=%s", "SIMULATION" if args.dry_run else "*** HARDWARE ***")
    log.info("walk %.1fm forward then %.1fm backward, settle %.1fs around each",
             args.forward_m, args.backward_m, args.settle_seconds)
    log.info("distance is DEAD-RECKONED from the commanded speed -- no odometry.")
    log.info("The policy overshoots its command, so the real distance runs long.")
    log.info("After the sequence the robot HOLDS under the policy until you press Enter.")
    log.info("Ctrl-C stops the run and releases the motors (damped: it sinks, not free-fall).")
    log.info("=" * 62)

    dt = 0.02
    try:
        robot.connect()
        cap = robot.controller.max_vx
        # What the controller will actually command: stick deflection mapped
        # onto the trained range, then clipped by the safety cap.
        stick = float(np.clip(abs(args.vx), 0.0, 1.0))
        speed = min(stick * CMD_VX, cap)
        if speed <= 0.0:
            log.error("--vx %.2f commands no motion; nothing to do", args.vx)
            return 1
        fwd_s = args.forward_m / speed
        back_s = args.backward_m / speed
        log.info("connected; command speed %.2f m/s (cap %.2f)", speed, cap)
        log.info("estimated phase times: forward %.1fs, backward %.1fs", fwd_s, back_s)

        def phase(name: str, seconds: float, action: dict) -> None:
            log.info("phase %-8s %.1fs  action=%s", name, seconds, action or "{} (zero command)")
            for _ in range(int(seconds / dt)):
                robot.send_action(action)
                time.sleep(dt)

        phase("settle", args.settle_seconds, {})
        phase("forward", fwd_s, {"remote.ly": stick})
        # Zero-command hold between the two directions so the reversal is not
        # a full-speed sign flip in a single control step.
        phase("settle", args.settle_seconds, {})
        phase("backward", back_s, {"remote.ly": -stick})
        # NOTE: this does not bring the robot to a halt -- the policy keeps
        # walking at zero command. It only stops us ASKING for motion.
        phase("settle", args.settle_seconds, {})
        log.info("sequence complete")

        # Operator-gated release: the controller thread keeps the robot
        # balancing (and creeping -- zero command is not a stop) while we wait,
        # so it is never dropped with nobody holding it. Ctrl-C here releases
        # immediately, same as during a phase.
        log.warning("The robot is STILL under the policy and still moving.")
        log.warning("Take its weight (gantry / two people), THEN press Enter to release.")
        log.warning("Release is damped (kp=0, kd=8): it sinks, but it will NOT hold itself up.")
        try:
            input("ready to release? press Enter> ")
        except EOFError:
            log.warning("stdin closed; releasing now")
        return 0
    except KeyboardInterrupt:
        log.warning("interrupted by operator")
        return 130
    except Exception:
        log.exception("run failed")
        return 1
    finally:
        # disconnect() stops the controller loop first, THEN sends the damped
        # release (kp=0, kd=8), so the loop cannot re-publish stiff gains after
        # the release and the robot sinks instead of free-falling.
        log.info("disconnecting (stops the controller, then damped release)")
        try:
            robot.disconnect()
        except Exception:
            log.exception("disconnect failed -- USE THE E-STOP")
        thread = getattr(robot, "_controller_thread", None)
        if thread is not None and thread.is_alive():
            log.error("controller thread still alive -- USE THE E-STOP")


if __name__ == "__main__":
    raise SystemExit(main())
