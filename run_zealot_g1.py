#!/usr/bin/env python
"""Bring-up script: run the zealot locomotion policy on a G1 for a fixed, short
burst, then stop cleanly.

Designed for a FIRST hardware run, so everything is bounded and nothing depends
on the operator reacting in time:

  * fixed duration (default 1.0 s of walking) -- the loop exits on its own;
  * forward speed capped in the controller (ZEALOT_MAX_VX, default 0.2 m/s);
  * a hold phase before and after, where the command is zero;
  * `finally: robot.disconnect()`, which sends the zero-gain passive command,
    so a crash or Ctrl-C still ends with the motors released;
  * `--dry-run` runs the identical sequence against the MuJoCo sim.

IMPORTANT, and measured rather than assumed:
  * Releasing the stick does NOT stop this policy -- at zero command it keeps
    walking at ~0.26 m/s. Duration is the stop, not the command.
  * The stop path is kp=kd=tau=0 (fully passive, no damping): the robot goes
    limp and will drop if it is carrying its own weight. Use a gantry.
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
log = logging.getLogger("zealot-bringup")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="run in MuJoCo instead of on hardware")
    ap.add_argument("--robot-ip", default="192.168.123.164")
    ap.add_argument("--walk-seconds", type=float, default=1.0, help="how long to actually walk")
    ap.add_argument("--settle-seconds", type=float, default=2.0, help="zero-command hold before/after")
    ap.add_argument("--vx", type=float, default=1.0, help="joystick deflection (scaled by the speed cap)")
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
    log.info("walk %.1fs, settle %.1fs either side", args.walk_seconds, args.settle_seconds)
    log.info("STOP = duration, not the stick. Releasing it does NOT stop the robot.")
    log.info("Shutdown is kp=kd=0 (limp, no damping). Gantry required.")
    log.info("=" * 62)

    dt = 0.02
    try:
        robot.connect()
        cap = robot.controller.max_vx
        log.info("connected; forward speed capped at %.2f m/s", cap)

        def phase(name: str, seconds: float, action: dict) -> None:
            log.info("phase %-6s %.1fs  action=%s", name, seconds, action or "{} (zero command)")
            for _ in range(int(seconds / dt)):
                robot.send_action(action)
                time.sleep(dt)

        phase("settle", args.settle_seconds, {})
        phase("walk", args.walk_seconds, {"remote.ly": float(np.clip(args.vx, -1.0, 1.0))})
        # NOTE: this does not bring the robot to a halt -- the policy keeps
        # walking at zero command. It only stops us ASKING for motion.
        phase("settle", args.settle_seconds, {})
        log.info("sequence complete")
        return 0
    except KeyboardInterrupt:
        log.warning("interrupted by operator")
        return 130
    except Exception:
        log.exception("run failed")
        return 1
    finally:
        # Stop the controller thread BEFORE disconnecting. lerobot's
        # disconnect() sends the zero-gain passive command and only then sets
        # the shutdown flag, so the 50 Hz controller loop can re-publish normal
        # stiff gains in the gap -- leaving the robot rigid at its last target
        # instead of limp. Killing the loop first closes that window.
        try:
            robot._shutdown_event.set()
            thread = getattr(robot, "_controller_thread", None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    log.error("controller thread still alive -- USE THE E-STOP")
            log.info("controller loop stopped")
        except Exception:
            log.exception("could not stop the controller loop -- USE THE E-STOP")

        log.info("disconnecting (sends zero-gain passive command on hardware)")
        try:
            robot.disconnect()
        except Exception:
            log.exception("disconnect failed -- CHECK THE ROBOT")


if __name__ == "__main__":
    raise SystemExit(main())
