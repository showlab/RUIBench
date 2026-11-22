"""Can this machine run the real station?  Read-only: nothing moves.

    python -m ruibench.engines.franka.check

Checks that the NUC's robot server answers (the arm's pose and the fingers'
width) and that the robot model is its robot's, that the hand's client runs on
the NUC, that every camera of the rig is plugged in and delivers a picture with
depth, that the station is calibrated, and that the operator's window can open.
Each failure says what fixes it.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from typing import List, Optional, Tuple

from . import calibration as calibration_mod
from . import rig as rig_mod
from .view import VIEW_ENV, can_show

#: how long the robot server has to answer a call
ANSWER_SECONDS = 3
#: the robot's joints and the flange it reports agree this well by the robot model
MODEL_AGREES_M, MODEL_AGREES_DEG = 0.002, 0.5
#: the NUC as ssh knows it (``RUIBENCH_FRANKA_NUC_SSH``; unset: the hand is not asked
#: about), the hand's client as its kernel names it (15 letters at most), and how long the
#: NUC has to say
NUC = os.environ.get("RUIBENCH_FRANKA_NUC_SSH", "")
HAND_CLIENT = "franka_hand_cli"
ASK_NUC_SECONDS = 8


def robot(rig: rig_mod.Rig) -> Tuple[bool, str]:
    """Does the robot server answer, and what does it say of the arm?"""
    try:
        import zerorpc
    except ImportError:
        return False, "zerorpc is not installed in this environment (pip install zerorpc)"
    from ...core.types import Pose
    from ..kinematics import forward
    from .arm import from_wire
    from .calibration import flange_to_tcp
    from .model import PANDA

    client = zerorpc.Client(heartbeat=None, timeout=ANSWER_SECONDS)
    try:
        client.connect(f"tcp://{rig.host}:{rig.port}")
        flange = from_wire(client.get_ee_pose())
        joints = client.get_joint_positions()
        width = float(list(client.get_gripper_position())[0])
    except Exception as exc:                            # a timeout says the server is down
        return False, (f"no answer from {rig.host}:{rig.port} ({type(exc).__name__}): start the "
                       "server on the NUC (ssh to it; cd franka_server; python "
                       "single_arm_server.py), with the arm unlocked and FCI on in Desk")
    finally:
        client.close()
    # the robot's joints and its flange put the tool centre point at one place, or the
    # robot model is not this robot's: said before anything moves
    by_joints, by_flange = forward(PANDA, Pose(), joints), flange_to_tcp(flange)
    apart, turned = by_joints.distance_to(by_flange), math.degrees(by_joints.angle_to(by_flange))
    said = (f"flange at {[round(v, 3) for v in flange.pos]} m (base frame), fingers "
            f"{width * 1000:.1f} mm apart; by the model, joints and flange agree within "
            f"{apart * 1000:.1f} mm and {turned:.2f} degrees")
    if apart > MODEL_AGREES_M or turned > MODEL_AGREES_DEG:
        return False, f"the robot model is not this robot's: {said}"
    return True, f"answers: {said}"


def hand() -> Tuple[bool, str]:
    """Is the Franka Hand's client running on the NUC?  When it dies the robot server goes
    on answering with the last width it had, so only the NUC can say.  A NUC that cannot be
    asked keeps nothing from running."""
    if not NUC:
        return True, "the hand's client is not checked (set RUIBENCH_FRANKA_NUC_SSH to the NUC's ssh name)"
    try:
        done = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", NUC, "pgrep",
                               "-x", HAND_CLIENT], capture_output=True, timeout=ASK_NUC_SECONDS,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        return True, f"the NUC cannot be asked about the hand ({type(exc).__name__})"
    if done.returncode == 1:
        return False, ("the Franka Hand's client has died on the NUC, and the fingers no longer "
                       f"answer: ssh {NUC}, then launch_gripper.py gripper=franka_hand")
    if done.returncode:
        return True, f"the NUC cannot be asked about the hand (ssh said {done.returncode})"
    return True, "the Franka Hand's client is running"


def cameras(rig: rig_mod.Rig) -> List[Tuple[bool, str]]:
    """Is each of the rig's cameras plugged in, and does it deliver colour and depth?"""
    try:
        import pyrealsense2 as rs
    except ImportError:
        return [(False, "pyrealsense2 is not installed in this environment (pip install "
                        "pyrealsense2)")]
    from .cameras import RealSenseCamera

    present = {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}
    out = []
    for name, c in rig.cameras.items():
        if c.serial not in present:
            out.append((False, f"{name}: RealSense {c.serial} is not plugged in (have "
                               f"{sorted(present) or 'none'})"))
            continue
        cam = RealSenseCamera(c.serial, c.width, c.height, c.fps)
        try:
            colour, depth = cam.start().capture()
            seen = float((depth > 0).mean())
            out.append((True, f"{name}: RealSense {c.serial} delivers {colour.shape[1]}x"
                              f"{colour.shape[0]}, depth on {seen:.0%} of it"))
        except Exception as exc:
            out.append((False, f"{name}: RealSense {c.serial}: {exc}"))
        finally:
            cam.close()
    return out


def calibrated(cal: calibration_mod.Calibration) -> Tuple[bool, str]:
    if cal.provisional:
        return False, ("the station is not calibrated: positions perception reports are "
                       "guesses (python -m ruibench.engines.franka.calibrate)")
    return True, f"calibrated on {cal.measured}: the table {cal.table_z_m:.4f} m above the base"


def viewer() -> Tuple[bool, str]:
    """Can the operator's window open?  P is how the operator pauses the arm."""
    if os.environ.get(VIEW_ENV) == "0":
        return True, f"none, as {VIEW_ENV}=0 asks: P will not pause the arm"
    return can_show()


def numbers() -> Tuple[bool, str]:
    """NumPy 1, as in simulation (``pyproject.toml``): under 2 an episode ends in a
    log it cannot write."""
    import numpy

    if numpy.__version__.startswith("1."):
        return True, f"NumPy {numpy.__version__}, as in simulation"
    return False, (f"NumPy {numpy.__version__}, and the shared code runs on 1: "
                   "uv pip install --python .venv/bin/python -e '.[dev,real]'")


def problems(rig: Optional[rig_mod.Rig] = None,
             cal: Optional[calibration_mod.Calibration] = None) -> List[str]:
    """What keeps this machine from running the real station; [] when nothing does."""
    rig = rig or rig_mod.load()
    checks = [robot(rig), hand(), *cameras(rig), calibrated(cal or calibration_mod.load()),
              viewer(), numbers()]
    return [said for ok, said in checks if not ok]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    rig = rig_mod.load()
    checks = [("robot", robot(rig)), ("hand", hand()), *(("camera", c) for c in cameras(rig)),
              ("calibration", calibrated(calibration_mod.load())), ("window", viewer()),
              ("numbers", numbers())]
    for what, (ok, said) in checks:
        print(f"{'ok ' if ok else 'NO '} {what:<12} {said}")
    return 0 if all(ok for _, (ok, _) in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
