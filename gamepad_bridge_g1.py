#!/usr/bin/env python3
"""Drive the G1's onboard locomotion controller from a gamepad on THIS machine.

Runs wherever the pad is already paired -- a MacBook, a laptop, or the robot
itself -- and pushes joystick axes to the onboard controller over ZMQ. Nothing
needs to be re-paired to the robot: the pad stays connected here, and only the
axis values travel.

The receiving end is ``serve_onboard_controller`` in run_g1_server.py, which
binds a ZMQ PULL on :6004 with CONFLATE (it only ever acts on the freshest
message) and feeds each JSON dict straight into ``robot.send_action``. Any
``remote.*`` key is forwarded to the controller, so this script is a pure
producer of those four axes -- it holds no control logic of its own.

Setup on the machine with the pad::

    pip install pygame pyzmq

Then, with the robot already running ``run_g1_server --handshake`` and the
controller negotiated::

    python gamepad_bridge_g1.py --robot-ip 172.18.130.111

Check your pad's axis mapping first -- it varies by controller and OS::

    python gamepad_bridge_g1.py --list          # show pads and live axis values
    python gamepad_bridge_g1.py --dry-run       # print commands, send nothing

SAFETY, and none of this is theoretical:
  * Zero command does NOT stop the robot -- the policy keeps walking. Closing
    this script stops new axes arriving but does NOT halt the robot; the last
    command persists and the controller keeps stepping. Stopping is Ctrl-C on
    the ROBOT's server (damped release) or the e-stop.
  * ``--deadman N`` is strongly recommended: axes only pass while button N is
    held, and zeros are sent the moment you let go. It bounds operator error,
    not policy behaviour.
  * The physical Unitree remote OVERRIDES these axes whenever it is non-idle,
    so keep its sticks centred while flying from the pad.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

ACTION_PORT = 6004
SEND_HZ = 50.0  # matches the controller's 50 Hz control loop


def clamp(v: float) -> float:
    return max(-1.0, min(1.0, float(v)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-ip", default="172.18.130.111", help="robot IP running the onboard controller")
    ap.add_argument(
        "--port", type=int, default=ACTION_PORT, help=f"onboard action port (default {ACTION_PORT})"
    )
    ap.add_argument("--pad", type=int, default=0, help="joystick index (see --list)")
    # Axis indices differ per pad/OS. Defaults suit an Xbox-style pad: left
    # stick X/Y on 0/1, right stick X on 2 (macOS/SDL2 often puts it on 2).
    ap.add_argument("--axis-lx", type=int, default=0, help="left stick X axis index")
    ap.add_argument("--axis-ly", type=int, default=1, help="left stick Y axis index")
    ap.add_argument("--axis-rx", type=int, default=2, help="right stick X axis index")
    ap.add_argument(
        "--invert-ly",
        action="store_true",
        default=True,
        help="invert left stick Y (default on: SDL reports down as +)",
    )
    ap.add_argument("--no-invert-ly", dest="invert_ly", action="store_false")
    ap.add_argument(
        "--deadman",
        type=int,
        default=None,
        help="button index that must be HELD for axes to pass (recommended)",
    )
    ap.add_argument("--list", action="store_true", help="list pads and stream live axis values, send nothing")
    ap.add_argument("--dry-run", action="store_true", help="print what would be sent, send nothing")
    args = ap.parse_args()

    try:
        import pygame
    except ImportError:
        print("pygame missing:  pip install pygame pyzmq", file=sys.stderr)
        return 1

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No gamepad detected. Pair/connect it to THIS machine first.", file=sys.stderr)
        return 1

    if args.list:
        for i in range(pygame.joystick.get_count()):
            j = pygame.joystick.Joystick(i)
            j.init()
            print(f"[{i}] {j.get_name()}  axes={j.get_numaxes()} buttons={j.get_numbuttons()}")
        j = pygame.joystick.Joystick(args.pad)
        print(f"\nLive axes for pad {args.pad} -- move the sticks, Ctrl-C to stop:")
        try:
            while True:
                pygame.event.pump()
                axes = [f"{i}:{j.get_axis(i):+.2f}" for i in range(j.get_numaxes())]
                btns = [str(i) for i in range(j.get_numbuttons()) if j.get_button(i)]
                print(
                    "  " + " ".join(axes) + ("   held=" + ",".join(btns) if btns else ""),
                    end="\r",
                    flush=True,
                )
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n")
        return 0

    pad = pygame.joystick.Joystick(args.pad)
    pad.init()
    print(f"pad: {pad.get_name()} (axes={pad.get_numaxes()}, buttons={pad.get_numbuttons()})")

    sock = None
    if not args.dry_run:
        try:
            import zmq
        except ImportError:
            print("pyzmq missing:  pip install pyzmq", file=sys.stderr)
            return 1
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUSH)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDHWM, 2)  # never queue stale commands
        sock.connect(f"tcp://{args.robot_ip}:{args.port}")
        print(f"pushing axes to tcp://{args.robot_ip}:{args.port} at {SEND_HZ:.0f} Hz")

    if args.deadman is None:
        print("WARNING: no --deadman set. Axes pass whenever the stick moves.")
    else:
        print(f"deadman: hold button {args.deadman} to enable motion")
    print("Zero command does NOT stop the robot. Ctrl-C here stops SENDING only.")

    period = 1.0 / SEND_HZ
    try:
        while True:
            t0 = time.time()
            pygame.event.pump()

            live = args.deadman is None or pad.get_button(args.deadman)
            if live:
                ly = clamp(pad.get_axis(args.axis_ly)) * (-1.0 if args.invert_ly else 1.0)
                lx = clamp(pad.get_axis(args.axis_lx))
                rx = clamp(pad.get_axis(args.axis_rx)) if args.axis_rx < pad.get_numaxes() else 0.0
            else:
                ly = lx = rx = 0.0

            action = {"remote.ly": ly, "remote.lx": lx, "remote.rx": rx, "remote.ry": 0.0}
            if sock is not None:
                sock.send(json.dumps(action).encode("utf-8"))
            print(
                f"  ly {ly:+.2f}  lx {lx:+.2f}  rx {rx:+.2f}   {'LIVE ' if live else 'idle'}",
                end="\r",
                flush=True,
            )

            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        # Best-effort zero on the way out. This stops us ASKING for motion; it
        # does not stop the robot, which keeps walking on the last policy state.
        if sock is not None:
            zero = {"remote.ly": 0.0, "remote.lx": 0.0, "remote.rx": 0.0, "remote.ry": 0.0}
            for _ in range(5):
                sock.send(json.dumps(zero).encode("utf-8"))
                time.sleep(0.02)
            sock.close(linger=200)
        print("\nstopped sending (robot is NOT halted -- use the robot server's Ctrl-C or the e-stop)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
