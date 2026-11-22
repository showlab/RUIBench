"""Calibrate the real station: where the cameras are, and the table.

    python -m ruibench.engines.franka.calibrate moves       # the first moves on a robot
    python -m ruibench.engines.franka.calibrate run         # measure: the arm moves
    python -m ruibench.engines.franka.calibrate solve D     # solve again what a run kept in D
    python -m ruibench.engines.franka.calibrate verify      # check it: the arm moves
    python -m ruibench.engines.franka.calibrate board       # a board to print

One ChArUco board, lying flat on the table where the arm can get over it and the
fixed camera sees it, answers everything.  ``Board()`` is the lab's sheet (4 x 4
squares, an older OpenCV's layout); any other is given on the command line.

**What ``run`` does.**  The arm searches the table for the board from high up,
then the wrist camera sees it from ``FIRST_VIEWS`` and ``VIEWS``: camera poses
round the board, tilted and turned as the hand-eye problem needs, reached by the
wrist mount as far as it is known yet.  It goes from one to the next joint by
joint (``Station.swing``): views are far apart in turn, and a straight line of the
hand between them is what the robot's controller has the most trouble following.
The arm goes back to its start, and the fixed camera takes the board in its
largest colour picture.  All of it is kept as it is taken
(``runs/real/calibration/<time>/``), so that ``solve`` can answer again with no
robot, and a run cut short leaves what it saw.

**What ``solve`` answers.**

1. A first answer from each view alone (``cv2.calibrateHandEye``, eye in hand),
   then every view's every corner together: the wrist camera's mount, the
   board's pose in the robot's base frame and the size of a square, whatever
   puts all the corners where the pictures show them, the robot's readings
   taken as true (``fit_views``).
2. The size of a square comes out of the arm's own travel against what the
   pictures show; depth gives a second opinion.  ``--square-mm`` fixes it.
3. The fixed camera, from the board's corners in its picture and their place in
   the base frame.
4. The table's height is the board's plane, less its thickness.
5. Each camera's depth against that plane: what is off is ``depth_scale``.

Everything but the fixed camera's sharper look is measured on the pictures the
tools get -- rectified to the model's lens (``cameras``) -- so it is the cameras
perception reads that are calibrated.  ``verify`` brings the tool centre point 3
cm over each corner of the board, as the fixed camera and the calibration place
it, and keeps the pictures.  ``moves`` and ``verify`` move the hand in straight
lines, as the tools do.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ...core.geometry import compose, inverse, matrix_to_quat, quat_mul, rotate
from ...core.types import GRASP_DOWN, Pose
from ..base import NEAR_POS_TOL, NEAR_ROT_TOL
from . import calibration as calibration_mod
from .cameras import Intrinsics, RealSenseCamera, Rectifier
from .terminal import question

#: where a run keeps what it saw
RUNS = Path("runs/real/calibration")
#: says the first moves have passed on this robot (``moves --once``)
MOVES_PASSED = RUNS / "moves-passed"
#: where the search looks for the board: offsets (m) of the tool centre point from
#: its start, at the start's height, where the wrist camera sees about 40 cm across
SEARCH = ((0.0, 0.0), (0.12, 0.0), (0.24, 0.0), (0.12, 0.15), (0.12, -0.15), (0.24, 0.15),
          (0.24, -0.15), (0.0, 0.15), (0.0, -0.15), (0.34, 0.0))
#: the wrist camera's views of the board: how far it is tilted from over the board
#: (degrees), toward where (azimuth: 0 beyond the board, 180 the robot's side), its
#: turn about its own axis (degrees), and its distance from the board's middle (m).
#: The first are taken by the mount as it is roughly known, the others by what they gave.
#: A camera on the robot's side of the board, looking away from the robot, has the
#: hand bent forward and joint 6 near the wrist's singularity: only slightly tilted
#: there, the other sides make up the turns the problem needs
FIRST_VIEWS = ((0, 0, 0, 0.36), (12, 90, 0, 0.36), (12, 270, 0, 0.36), (12, 0, 0, 0.36),
               (0, 0, 25, 0.32), (0, 0, -25, 0.32), (10, 180, 0, 0.38))
VIEWS = ((25, 90, 0, 0.34), (25, 270, 0, 0.34), (25, 45, 15, 0.34), (25, 315, -15, 0.34),
         (25, 90, 20, 0.30), (25, 270, -20, 0.30), (30, 60, -15, 0.38), (30, 300, 15, 0.38),
         (20, 0, 0, 0.36), (18, 45, -25, 0.40), (18, 315, 25, 0.40), (15, 135, 15, 0.40),
         (15, 225, -15, 0.40), (0, 0, 40, 0.42), (0, 0, -40, 0.42), (0, 0, 0, 0.26),
         (0, 0, 0, 0.46))
#: a view counts with at least this many of the board's corners found, chess corners
#: and markers' corners together; the hand-eye problem needs at least this many views
MIN_POINTS = 16
MIN_VIEWS = 6
#: a move gives up after this long, or this long after its path ended, and the arm
#: stands this long for a sharp picture
MOVE_SECONDS = 20.0
SETTLE_SECONDS = 1.5
STILL_SECONDS = 0.5
#: the fixed camera's sharper look: colour alone reaches this on its USB 2 port
FIXED_LOOK = (1920, 1080, 8)
FIXED_FRAMES = 8
#: what a corner is first found with where nothing says how big a square is
NOMINAL_SQUARE_M = 0.03
#: a view whose corners miss by this many pixels (and three times the middle view's)
#: is left out of the answer
BAD_VIEW_PX = 2.0
#: the pause's trial: a line this long, a pause this long, within this of where it was told
PAUSE_TRIAL_M = 0.2
PAUSE_TRIAL_SECONDS = 1.0
PAUSE_STOPS_WITHIN_M = 0.04
#: the fingertips beyond the tool centre point, along the tool (their contact
#: measurement: 11.23 cm from the flange)
FINGERTIP_M = 0.1123 - calibration_mod.TCP_FROM_FLANGE_M
#: a first move's joints further than this from where ``route`` has them: its
#: reckoning of the robot's controller is not to be trusted.  (Measured, 2026-11-02:
#: 0.02 rad after 5 cm, 0.12 rad after a 30 degree turn: friction shares a turn out
#: among the joints otherwise than the controller asks)
ROUTE_AGREES_RAD = 0.15
#: how near the table the first moves go, the tool centre point over it: the fingertips
#: 3 cm up
LOW_OVER_TABLE_M = 0.04
#: the board's plane, measured, further than this from level is not lying flat
MAX_BOARD_TILT_DEG = 2.0
#: the table's pixels, for its depth: inside this box of the base frame, and within
#: this of the board's plane
TABLE_X, TABLE_Y, TABLE_BAND_M = (0.25, 0.85), (-0.40, 0.40), 0.03


@dataclass(frozen=True)
class Board:
    """A ChArUco board, in squares.  ``legacy``: laid out as OpenCV before 4.6 laid
    a board with an even number of rows, as the lab's sheet is."""

    squares: Tuple[int, int] = (4, 4)
    dictionary: str = "DICT_4X4_50"
    marker_share: float = 0.75
    legacy: bool = True

    def charuco(self, square_m: float = 1.0):
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        b = cv2.aruco.CharucoBoard(self.squares, square_m, square_m * self.marker_share, d)
        b.setLegacyPattern(self.legacy)
        return b

    def as_dict(self) -> dict:
        return {"squares": list(self.squares), "dictionary": self.dictionary,
                "marker_share": round(self.marker_share, 4), "legacy": self.legacy}

    @classmethod
    def from_dict(cls, d: dict) -> "Board":
        return cls(tuple(d["squares"]), d["dictionary"], float(d["marker_share"]),
                   bool(d["legacy"]))


@functools.lru_cache(maxsize=8)
def _places(board: Board) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Where the board's chess corners, by id, and its markers' corners, by marker
    id, lie on it, in squares."""
    b = board.charuco()
    chess = np.asarray(b.getChessboardCorners(), float).reshape(-1, 3)
    markers = {int(i): np.asarray(p, float).reshape(4, 3)
               for i, p in zip(np.asarray(b.getIds()).ravel(), b.getObjPoints())}
    return chess, markers


@dataclass
class Seen:
    """The board in one picture: the pixels of its chess corners and markers' corners."""

    chess: Dict[int, np.ndarray]
    markers: Dict[int, np.ndarray]

    def points(self, board: Board) -> Tuple[np.ndarray, np.ndarray]:
        """(the places on the board, in squares, Nx3; the pixels, Nx2)."""
        chess, markers = _places(board)
        where = [chess[i] for i in sorted(self.chess)]
        px = [self.chess[i] for i in sorted(self.chess)]
        for i in sorted(self.markers):
            where += list(markers[i])
            px += list(self.markers[i])
        return np.asarray(where, float), np.asarray(px, float)

    def __len__(self) -> int:
        return len(self.chess) + 4 * len(self.markers)


def see(image, board: Board, upscale: int = 1) -> Optional[Seen]:
    """The board in a picture, or None.  ``upscale``: looked for in a larger copy of
    the picture, for a board small in it; the pixels are the picture's own."""
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    if upscale > 1:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.CharucoDetector(board.charuco(), cv2.aruco.CharucoParameters(), params)
    corners, ids, marks, mark_ids = detector.detectBoard(gray)
    if mark_ids is None or not len(mark_ids):
        return None

    def back(px):                                      # pixel middles at their index
        return (np.asarray(px, float) + 0.5) / upscale - 0.5

    chess = {} if ids is None else {int(i): back(c).ravel() for i, c in zip(ids.ravel(), corners)}
    return Seen(chess, {int(i): back(c).reshape(4, 2) for i, c in zip(mark_ids.ravel(), marks)})


def marker_share(board: Board, seen: Sequence[Seen]) -> float:
    """A marker's side in squares, as the pictures show it: each picture's markers
    taken to the board's plane by its chess corners."""
    chess, _ = _places(board)
    sides = []
    for s in seen:
        if len(s.chess) < 4:
            continue
        ids = sorted(s.chess)
        h, _ = cv2.findHomography(np.array([s.chess[i] for i in ids]), chess[ids][:, :2])
        for corners in s.markers.values():
            flat = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), h).reshape(4, 2)
            sides += list(np.linalg.norm(flat - np.roll(flat, 1, axis=0), axis=1))
    return float(np.median(sides)) if sides else board.marker_share


def pinhole(focal_px: float, width: int, height: int) -> np.ndarray:
    """A centred pinhole, its pixel middles at their index (OpenCV's way of a
    rectified picture whose middle is at width / 2, RUIBench's way)."""
    return np.array([[focal_px, 0, width / 2 - 0.5], [0, focal_px, height / 2 - 0.5], [0, 0, 1]])


def _matrix(pose: Pose) -> np.ndarray:
    w, x, y, z = pose.quat
    t = np.eye(4)
    t[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    t[:3, 3] = pose.pos
    return t


def _pose(t: np.ndarray) -> Pose:
    return Pose(tuple(float(v) for v in t[:3, 3]), matrix_to_quat(t[:3, :3].tolist()))


def _rms(obj: np.ndarray, px: np.ndarray, t: np.ndarray, k: np.ndarray) -> float:
    rvec, _ = cv2.Rodrigues(t[:3, :3])
    proj, _ = cv2.projectPoints(obj, rvec, t[:3, 3], k, None)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - px) ** 2, axis=1))))


def locate(obj: np.ndarray, px: np.ndarray, k: np.ndarray) -> Tuple[Pose, float]:
    """The board's pose in a camera's frame from its points (metres, on its plane)
    and their pixels, and how many pixels they then miss by.  A flat board has two
    answers that nearly fit: the better one."""
    _, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj.astype(np.float64), px.astype(np.float64), k, None,
                                             flags=cv2.SOLVEPNP_IPPE)
    best: Optional[Tuple[Pose, float]] = None
    for r, t in zip(rvecs, tvecs):
        r, t = cv2.solvePnPRefineLM(obj.astype(np.float64), px.astype(np.float64), k, None,
                                    r.copy(), t.copy())
        m = np.eye(4)
        m[:3, :3], _ = cv2.Rodrigues(r)
        m[:3, 3] = t.ravel()
        err = _rms(obj, px, m, k)
        if t[2] > 0 and (best is None or err < best[1]):
            best = (_pose(m), err)
    if best is None:
        raise ValueError("the board cannot be placed in front of the camera")
    return best


def hand_eye(flanges: Sequence[Pose], views: Sequence[Pose]) -> Pose:
    """The camera in the flange's frame, from the flange's poses in the base frame
    and the board's in the camera's (eye in hand)."""
    rg, tg = zip(*((_matrix(p)[:3, :3], _matrix(p)[:3, 3]) for p in flanges))
    rc, tc = zip(*((_matrix(p)[:3, :3], _matrix(p)[:3, 3]) for p in views))
    r, t = cv2.calibrateHandEye(list(rg), list(tg), list(rc), list(tc),
                                method=cv2.CALIB_HAND_EYE_PARK)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, t.ravel()
    return _pose(m)


def spread(poses: Sequence[Pose]) -> Tuple[Pose, float, float]:
    """The mean of nearly equal poses, and their RMS distance (m) and angle
    (degrees) from it."""
    from .arm import slerp

    n = len(poses)
    pos = tuple(sum(p.pos[i] for p in poses) / n for i in range(3))
    quat = poses[0].quat
    for i, p in enumerate(poses[1:], 2):
        quat = slerp(quat, p.quat, 1.0 / i)
    mean = Pose(pos, quat)
    dist = math.sqrt(sum(mean.distance_to(p) ** 2 for p in poses) / n)
    angle = math.degrees(math.sqrt(sum(mean.angle_to(p) ** 2 for p in poses) / n))
    return mean, dist, angle


@dataclass
class Fit:
    """The answer of all a wrist camera's views together (``fit_views``)."""

    mount: Pose                      # the camera in the flange's frame
    on_table: Pose                   # the board in the base frame
    square_m: float
    view_px: List[float]             # how far each view's corners miss, RMS
    used: List[bool]                 # the views the answer stands on
    spread_mm: float                 # each view by itself: where it puts the board
    spread_deg: float


def fit_views(board: Board, flanges: Sequence[Pose], seen: Sequence[Seen], k: np.ndarray,
              square_m: Optional[float] = None, guess_m: float = NOMINAL_SQUARE_M) -> Fit:
    """The wrist camera's mount, the board's pose and -- unless given -- a square's
    size that put every corner of every view where its picture shows it, the
    robot's readings taken as true.  The size is in it because moving the camera by
    a known distance moves the corners by as much as they are near: the arm's own
    travel measures the board.  Views that fit far worse than the rest are left out."""
    from scipy.optimize import least_squares

    points = [s.points(board) for s in seen]
    size = square_m or guess_m
    views = [locate(obj * size, px, k)[0] for obj, px in points]
    mount = hand_eye(flanges, views)
    on_table = spread([compose(compose(f, mount), v) for f, v in zip(flanges, views)])[0]
    down = [np.linalg.inv(_matrix(f)) for f in flanges]
    used = [True] * len(flanges)

    def vec(t):
        return np.concatenate([cv2.Rodrigues(t[:3, :3])[0].ravel(), t[:3, 3]])

    def mat(v):
        t = np.eye(4)
        t[:3, :3], _ = cv2.Rodrigues(v[:3])
        t[:3, 3] = v[3:]
        return t

    def solved(x):
        return mat(x[:6]), mat(x[6:12]), (square_m if square_m else math.exp(x[12]))

    def residuals(x, keep):
        m, b, a = solved(x)
        out = []
        for i in keep:
            t = np.linalg.inv(m) @ down[i] @ b
            r, _ = cv2.Rodrigues(t[:3, :3])
            obj, px = points[i]
            proj, _ = cv2.projectPoints(obj * a, r, t[:3, 3], k, None)
            out.append((proj.reshape(-1, 2) - px).ravel())
        return np.concatenate(out)

    x = np.concatenate([vec(_matrix(mount)), vec(_matrix(on_table)),
                        [] if square_m else [math.log(size)]])
    for _ in range(3):
        keep = [i for i, u in enumerate(used) if u]
        x = least_squares(residuals, x, args=(keep,), loss="soft_l1", f_scale=1.0,
                          x_scale="jac", max_nfev=100).x
        m, b, a = solved(x)
        errs = [_rms(points[i][0] * a, points[i][1], np.linalg.inv(m) @ down[i] @ b, k)
                for i in range(len(flanges))]
        worst = max(BAD_VIEW_PX, 3.0 * float(np.median([errs[i] for i in keep])))
        now = [e <= worst for e in errs]
        if now == used:
            break
        used = now
    keep = [i for i, u in enumerate(used) if u]
    alone = [compose(compose(flanges[i], _pose(m)),
                     locate(points[i][0] * a, points[i][1], k)[0]) for i in keep]
    _, dist, angle = spread(alone)
    return Fit(_pose(m), _pose(b), float(a), errs, used, dist * 1000.0, angle)


def board_point(board: Board, pose: Pose, square_m: float, u: float, v: float):
    """A point of the board's plane, as fractions (u across, v down) of its squares."""
    return compose(pose, Pose((u * board.squares[0] * square_m, v * board.squares[1] * square_m,
                               0.0))).pos


def turned_down(roll: float, pitch: float, yaw: float):
    """The tool's orientation at [roll, pitch, yaw] degrees from pointing straight
    down, in the station's convention (``ruikit.env.station.quat_from_rpy_deg``)."""
    def about(axis, deg):
        a = math.radians(deg) / 2
        return (math.cos(a), *(c * math.sin(a) for c in axis))

    rel = quat_mul(about((0, 0, 1), yaw), quat_mul(about((0, 1, 0), -pitch),
                                                   about((1, 0, 0), roll)))
    return quat_mul(rel, GRASP_DOWN)


def looking_at(middle, tilt: float, azimuth: float, turn: float, distance: float) -> Pose:
    """A camera (optical convention) ``distance`` from a point, ``tilt`` degrees from
    straight over it toward ``azimuth``, looking at it, turned ``turn`` degrees about
    its axis from the top of its picture being toward the robot."""
    t, a = math.radians(tilt), math.radians(azimuth)
    away = np.array([math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), math.cos(t)])
    z = -away
    y = np.array([1.0, 0.0, 0.0]) - z[0] * z           # the picture's down: away from the robot
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    r = np.column_stack([c * x + s * y, -s * x + c * y, z])
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = r, np.asarray(middle, float) + distance * away
    return _pose(pose)


# -- with the robot ----------------------------------------------------------------
class Station:
    """The real engine, its scene and its cameras' lenses, for calibrating.  Its
    depth is what the cameras measure, with no correction of an earlier run."""

    def __init__(self, engine=None) -> None:
        from ...core.config import RunConfig
        from ...runner.episode import _sized
        from ..base import camera_apertures, camera_focal_length
        from .engine import FrankaEngine
        from .scene import station_scene

        self.engine = engine or FrankaEngine(
            calibration=replace(calibration_mod.load(), depth_scale={}))
        scene = _sized(station_scene("calibrate", cal=self.engine.calibration), RunConfig())
        self.scene = replace(scene, cameras=tuple(replace(c, depth=True) for c in scene.cameras))
        self.engine.load(self.scene, 0)
        self.engine.reset()
        self.start_joints = self.engine.arm_joints(self.engine.arm_id)
        self.focal = {c.name: c.width * camera_focal_length(self.scene, c)
                      / camera_apertures(self.scene, c)[0] for c in self.scene.cameras}
        self.size = {c.name: (c.width, c.height) for c in self.scene.cameras}

    def k(self, camera: str) -> np.ndarray:
        return pinhole(self.focal[camera], *self.size[camera])

    def tcp(self) -> Pose:
        return self.engine.state().arm(self.engine.arm_id).eef_pose

    def go(self, pose: Pose) -> bool:
        """The tool centre point (world) in a straight line, as the tools move it,
        if all the way there is allowed; False where it is not, or where it did not
        get there."""
        if not self.engine.route(pose).whole:
            return False
        self.engine.set_arm_target(self.engine.arm_id, pose)
        return self._arrive()

    def swing(self, pose: Pose) -> bool:
        """The tool centre point (world) joint by joint, in the posture sought from
        the start's: no line for the robot's controller to follow, every joint within
        its speed, and no wandering along the arm's redundancy from view to view.
        False where no such posture is, or where it did not get there."""
        joints = self.engine.joints_for(pose, seed=self.start_joints)
        if joints is None:
            return False
        self.engine.set_arm_joints(self.engine.arm_id, joints)
        return self._arrive()

    def _arrive(self) -> bool:
        """Wait for the move, and stand still.  As near as a move gets at rest is as
        good for a picture: the arm's readings say where it is."""
        engine, arm = self.engine, self.engine.arm_id
        t0, ended = engine.sim_time, None
        while not engine.arm_at_target(arm) and engine.sim_time - t0 < MOVE_SECONDS:
            if ended is None and not engine.arm.busy():
                ended = engine.sim_time
            if ended is not None and engine.sim_time - ended > SETTLE_SECONDS:
                break
            engine.advance(0.05)
        engine.advance(STILL_SECONDS)
        return engine.arm_at_target(arm, NEAR_POS_TOL, NEAR_ROT_TOL)

    def start(self) -> None:
        """Back to the arm's start, out of the fixed camera's way."""
        self.engine.reset()

    def close(self) -> None:
        self.engine.close()


def _search(st: Station, board: Board, mount: Pose, say) -> Pose:
    """The board's pose in the world, roughly: the wrist camera looks from high up
    until it sees some of it."""
    start, k = st.tcp(), st.k("wrist")
    for dx, dy in SEARCH:
        if not st.swing(Pose((start.pos[0] + dx, start.pos[1] + dy, start.pos[2]), GRASP_DOWN)):
            continue
        seen = see(st.engine.render("wrist").image, board)
        if seen is not None:
            say(f"   the wrist camera sees {len(seen)} of its corners from "
                f"{[round(v, 2) for v in st.tcp().pos]}")
            obj, px = seen.points(board)
            pose = locate(obj * NOMINAL_SQUARE_M, px, k)[0]
            flange = st.engine.arm.reading().flange
            return compose(compose(st.engine.base, compose(flange, mount)), pose)
    raise SystemExit("the wrist camera finds no board anywhere over the table")


def _keep(folder: Path, views: List[dict], meta: dict) -> None:
    """What a run has so far, for ``solve``: a run cut short loses none of its views."""
    (folder / "views.json").write_text(json.dumps(views, indent=1))
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))


def capture(folder: Path, board: Board, say=print, engine=None) -> Path:
    """Search for the board and see it from every view; keep it all in ``folder``."""
    from PIL import Image

    from .calibration import flange_to_tcp
    from .rig import load as load_rig

    folder.mkdir(parents=True, exist_ok=True)
    st = Station(engine)
    views: List[dict] = []
    seens: List[Seen] = []
    flanges: List[Pose] = []
    meta = {"board": board.as_dict(), "wrist_focal_px": st.focal["wrist"],
            "wrist_size": list(st.size["wrist"]), "front_focal_px": st.focal["agentview"],
            "front_size": list(st.size["agentview"]),
            "taken": datetime.now().isoformat(timespec="seconds")}
    try:
        mount, k = st.engine.calibration.wrist, st.k("wrist")
        found = _search(st, board, mount, say)
        middle = board_point(board, found, NOMINAL_SQUARE_M, 0.5, 0.5)

        def take(plan, mount, middle):
            for tilt, azimuth, turn, distance in plan:
                camera = looking_at(middle, tilt, azimuth, turn, distance)
                # where the camera is wanted, the flange by the mount, the tool from it
                tcp = flange_to_tcp(compose(camera, inverse(mount)))
                if not st.swing(tcp):
                    say(f"   view {tilt, azimuth, turn, distance}: out of the arm's way, left out")
                    continue
                frame = st.engine.render("wrist")
                seen = see(frame.image, board)
                if seen is None or len(seen) < MIN_POINTS:
                    say(f"   view {tilt, azimuth, turn, distance}: the board is not in the "
                        f"picture ({len(seen) if seen else 0} corners), left out")
                    continue
                reading = st.engine.arm.reading()
                flange = reading.flange
                name = f"wrist_{len(views):02d}"
                Image.fromarray(frame.image).save(folder / f"{name}.png")
                if not views:
                    np.save(folder / "wrist_depth.npy", frame.depth)
                views.append({"file": f"{name}.png", "view": [tilt, azimuth, turn, distance],
                              "flange": {"pos": list(flange.pos), "quat_wxyz": list(flange.quat)},
                              "joints": list(reading.joints)})
                seens.append(seen)
                flanges.append(flange)
                _keep(folder, views, meta)
                say(f"   view {tilt, azimuth, turn, distance}: {len(seen)} corners")

        take(FIRST_VIEWS[:1], mount, middle)
        if not views:
            raise SystemExit("the wrist camera lost the board over where it was found")
        # the board's middle by a whole view, for the rest to look at
        obj, px = seens[0].points(board)
        whole = locate(obj * NOMINAL_SQUARE_M, px, k)[0]
        middle = board_point(board, compose(st.engine.base, compose(compose(flanges[0], mount),
                                                                    whole)),
                             NOMINAL_SQUARE_M, 0.5, 0.5)
        take(FIRST_VIEWS[1:], mount, middle)
        if len(views) >= 3:
            fit = fit_views(board, flanges, seens, k)
            mount = fit.mount
            middle = compose(st.engine.base, Pose(board_point(board, fit.on_table, fit.square_m,
                                                              0.5, 0.5))).pos
            say(f"   so far: a square {fit.square_m * 1000:.1f} mm, the views within "
                f"{np.median(fit.view_px):.2f} px")
        take(VIEWS, mount, middle)
        st.start()
        front = st.engine.render("agentview")
        Image.fromarray(front.image).save(folder / "front.png")
        np.save(folder / "front_depth.npy", front.depth)
    finally:
        st.close()
    # the fixed camera's sharper look, the arm holding at its start
    serial = load_rig().cameras["agentview"].serial
    width, height, fps = FIXED_LOOK
    cam = RealSenseCamera(serial, width, height, fps, depth=False).start()
    try:
        for i in range(FIXED_FRAMES):
            time.sleep(1.5 / fps)
            Image.fromarray(cam.capture()[0]).save(folder / f"front_look_{i}.png")
        meta["front_look"] = cam.intrinsics.as_dict()
    finally:
        cam.close()
    _keep(folder, views, meta)
    say(f"kept {len(views)} views in {folder}")
    return folder


def _depth_against_table(depth, camera: Pose, k: np.ndarray, table_z: float) -> dict:
    """What a camera's depth says of the table: over the table's pixels, how high
    above the board's plane it reads the table, how far its plane leans, and what
    its depth is to be multiplied by for the table to lie on the board's plane."""
    h, w = depth.shape
    us, vs = np.meshgrid((np.arange(w) - k[0, 2]) / k[0, 0], (np.arange(h) - k[1, 2]) / k[1, 1])
    m = _matrix(camera)
    rays = np.stack([us, vs, np.ones_like(us)], axis=-1) @ m[:3, :3].T
    points = m[:3, 3] + rays * depth[..., None]
    above = points[..., 2] - table_z
    mask = ((depth > 0) & (points[..., 0] > TABLE_X[0]) & (points[..., 0] < TABLE_X[1])
            & (points[..., 1] > TABLE_Y[0]) & (points[..., 1] < TABLE_Y[1])
            & (np.abs(above) < TABLE_BAND_M))
    if mask.sum() < 500:
        return {"pixels": int(mask.sum())}
    scale = float(np.median((table_z - m[2, 3]) / (depth[mask] * rays[..., 2][mask])))
    a = np.c_[points[mask][:, :2], np.ones(int(mask.sum()))]
    plane, *_ = np.linalg.lstsq(a, points[mask][:, 2], rcond=None)
    lean = math.degrees(math.atan(math.hypot(plane[0], plane[1])))
    return {"pixels": int(mask.sum()), "median_mm": round(float(np.median(above[mask])) * 1000, 2),
            "rms_mm": round(float(np.std(above[mask])) * 1000, 2), "lean_deg": round(lean, 3),
            "scale": round(scale, 5)}


def solve(folder: Path, square_m: Optional[float] = None, thickness_m: float = 0.0,
          say=print) -> calibration_mod.Calibration:
    """The calibration from what a run kept; saved (the module docstring)."""
    from PIL import Image

    meta = json.loads((folder / "meta.json").read_text())
    views = json.loads((folder / "views.json").read_text())
    if "front_look" not in meta:
        raise SystemExit(f"that run was cut short after {len(views)} views, before the fixed "
                         "camera looked: run again")
    board = Board.from_dict(meta["board"])
    images = [np.asarray(Image.open(folder / v["file"]).convert("RGB")) for v in views]
    flanges = [Pose(tuple(v["flange"]["pos"]), tuple(v["flange"]["quat_wxyz"])) for v in views]
    first = [see(i, board) for i in images]
    board = replace(board, marker_share=marker_share(board, first))
    seens = [see(i, board) for i in images]
    k = pinhole(meta["wrist_focal_px"], *meta["wrist_size"])
    say(f"{len(seens)} views; a marker is {board.marker_share:.3f} of a square")
    # a square's size by the wrist camera's depth, in the first view: a second opinion
    obj, px = seens[0].points(board)
    in_squares = locate(obj, px, k)[0]
    depth = np.load(folder / "wrist_depth.npy")
    low, high = px.min(0).astype(int), px.max(0).astype(int)
    patch = depth[low[1]:high[1], low[0]:high[0]]
    by_depth = float(np.median(patch[patch > 0])) / in_squares.pos[2]
    fit = fit_views(board, flanges, seens, k, square_m, guess_m=by_depth)
    bad = sum(not u for u in fit.used)
    say(f"a square: {fit.square_m * 1000:.2f} mm" + ("" if square_m else " by the arm's travel,")
        + f" {by_depth * 1000:.2f} mm by the wrist camera's depth")
    say(f"the views' corners miss by {np.median(fit.view_px):.2f} px (middle), "
        f"{max(fit.view_px):.2f} px (worst); {bad} of {len(fit.used)} left out; each view alone "
        f"puts the board within {fit.spread_mm:.2f} mm and {fit.spread_deg:.3f} degrees")
    normal = rotate(fit.on_table.quat, (0.0, 0.0, 1.0))
    tilt = math.degrees(math.acos(min(1.0, abs(normal[2]))))
    if tilt > MAX_BOARD_TILT_DEG:
        say(f"the board lies {tilt:.1f} degrees from level: is it flat on the table?")
    middle = board_point(board, fit.on_table, fit.square_m, 0.5, 0.5)
    table_z = middle[2] - thickness_m
    # the fixed camera: its sharper look, averaged over its frames
    raw = Intrinsics(*meta["front_look"]["size"], meta["front_look"]["fx"], meta["front_look"]["fy"],
                     meta["front_look"]["ppx"], meta["front_look"]["ppy"],
                     tuple(meta["front_look"]["coeffs"]), meta["front_look"]["model"])
    focal = (raw.fx + raw.fy) / 2.0
    sharp = Rectifier(raw, raw.width, raw.height, focal)
    looks = [see(sharp.colour(np.asarray(Image.open(p).convert("RGB"))), board)
             for p in sorted(folder.glob("front_look_*.png"))]
    looks = [s for s in looks if s is not None]
    if not looks:
        raise SystemExit("the fixed camera does not see the board")
    full = max(looks, key=len)
    mean = Seen({i: np.mean([s.chess[i] for s in looks if i in s.chess], axis=0) for i in full.chess},
                {i: np.mean([s.markers[i] for s in looks if i in s.markers], axis=0)
                 for i in full.markers})
    obj, px = mean.points(board)
    in_front, front_err = locate(obj * fit.square_m, px, pinhole(focal, raw.width, raw.height))
    front = compose(fit.on_table, inverse(in_front))
    say(f"the fixed camera sees {len(mean)} of the board's corners, within {front_err:.2f} px; "
        f"it stands at {[round(v, 3) for v in front.pos]} m (base frame)")
    # the same camera through the stream the tools get, against that
    k_front = pinhole(meta["front_focal_px"], *meta["front_size"])
    usual = see(np.asarray(Image.open(folder / "front.png").convert("RGB")), board, upscale=3)
    agrees: dict = {}
    if usual is not None:
        o, p = usual.points(board)
        also = compose(fit.on_table, inverse(locate(o * fit.square_m, p, k_front)[0]))
        agrees = {"mm": round(also.distance_to(front) * 1000, 2),
                  "deg": round(math.degrees(also.angle_to(front)), 3)}
        say(f"through the tools' own stream it stands {agrees['mm']} mm and {agrees['deg']} "
            "degrees from that")
    # each camera's depth against the table
    wrist_at = compose(flanges[0], fit.mount)
    depths = {"wrist": _depth_against_table(depth, wrist_at, k, table_z),
              "agentview": _depth_against_table(np.load(folder / "front_depth.npy"), front, k_front,
                                                table_z)}
    for name, d in depths.items():
        say(f"{name}'s depth reads the table {d.get('median_mm')} mm above the board's plane, "
            f"+-{d.get('rms_mm')} mm, leaning {d.get('lean_deg')} degrees; scale {d.get('scale')}")
    cal = calibration_mod.Calibration(
        table_z_m=round(table_z, 5), front=front, wrist=fit.mount,
        measured=date.today().isoformat(),
        depth_scale={name: d["scale"] for name, d in depths.items() if "scale" in d},
        fit={"run": folder.name, "board": board.as_dict(), "square_m": round(fit.square_m, 6),
             "square_by_depth_m": round(by_depth, 6), "views": len(fit.used),
             "views_used": sum(fit.used), "view_px_median": round(float(np.median(fit.view_px)), 3),
             "view_px_worst": round(max(fit.view_px), 3), "board_spread_mm": round(fit.spread_mm, 2),
             "board_spread_deg": round(fit.spread_deg, 3), "board_tilt_deg": round(tilt, 3),
             "board_thickness_m": thickness_m, "front_px": round(front_err, 3),
             "front_stream_vs_look": agrees, "depth": depths})
    say(f"the table is {cal.table_z_m * 1000:.1f} mm above the robot's base; "
        f"saved {calibration_mod.save(cal)}")
    return cal


def first_moves(say=print) -> bool:
    """Before the calibration, on a robot never driven by this engine: small moves
    from the start, then down near the table, where the wrist is nearest its singularity,
    and the hand closed and opened, each move's residual said, and how near its joints
    came to where following its line had them: the robot's own controller against
    ``route``'s reckoning of it (``check`` says, before anything moves, that the robot
    model is this robot's).  Then the operator's pause, half way along a line at full
    speed."""
    st = Station()
    ok = True
    try:
        engine = st.engine
        start = st.tcp()
        low = LOW_OVER_TABLE_M - start.pos[2]
        for dx, dy, dz, yaw in ((0.05, 0, 0, 0), (0, 0, 0, 0), (0, 0.05, 0, 0), (0, 0, 0, 0),
                                (0, 0, -0.05, 0), (0, 0, 0, 0), (0, 0, 0, 30), (0, 0, 0, 0),
                                (0, 0, low, 0), (0, 0.05, low, 0), (0, 0, low, 0), (0, 0, low, 30),
                                (0, 0, low, 0), (0, 0, 0, 0)):
            goal = Pose((start.pos[0] + dx, start.pos[1] + dy, start.pos[2] + dz),
                        turned_down(0, 0, yaw))
            way = engine.route(goal)
            arrived = st.go(goal)
            here = st.tcp()
            apart = max(abs(a - b) for a, b in zip(way.joints, engine.arm_joints(engine.arm_id)))
            arrived = arrived and apart <= ROUTE_AGREES_RAD
            ok &= arrived
            say(f"{'ok ' if arrived else 'NO '} to {[round(v, 3) for v in goal.pos]} yaw {yaw}: "
                f"{here.distance_to(goal) * 1000:.1f} mm, "
                f"{math.degrees(here.angle_to(goal)):.2f} degrees away, in {way.seconds:.1f} s; "
                f"the joints {apart:.3f} rad from where following the line has them")
        for value, word in ((1.0, "closed"), (0.0, "opened")):
            engine.set_gripper(engine.arm_id, value)
            t0 = time.monotonic()
            while not engine.gripper_settled(engine.arm_id):
                engine.advance(0.05)
            width = engine.state().arm(engine.arm_id).opening
            say(f"ok  the hand {word} in {time.monotonic() - t0:.1f} s: {width * 1000:.1f} mm")
        ok = _try_the_pause(st, start, say) and ok
        if ok and not engine.fake:                  # a stand-in's say nothing of the robot
            MOVES_PASSED.parent.mkdir(parents=True, exist_ok=True)
            MOVES_PASSED.write_text(datetime.now().isoformat(timespec="seconds") + "\n")
        return ok
    finally:
        st.close()


def _try_the_pause(st: Station, start: Pose, say=print) -> bool:
    """The pause at its hardest, half way along a line at full speed: how far the arm
    goes on once it is told, and whether it then gets to where it was going."""
    engine, arm = st.engine, st.engine.arm_id
    goal = Pose((start.pos[0] + PAUSE_TRIAL_M, start.pos[1], start.pos[2]), start.quat)
    seconds = engine.route(goal).seconds
    engine.set_arm_target(arm, goal)
    engine.advance(seconds / 2.0)
    told = st.tcp()
    engine.pause()
    time.sleep(PAUSE_TRIAL_SECONDS)
    rest = st.tcp()
    time.sleep(0.5)
    still = st.tcp().distance_to(rest)
    engine.resume()
    arrived = st._arrive()
    ok = arrived and rest.distance_to(told) <= PAUSE_STOPS_WITHIN_M and still <= 0.002
    say(f"{'ok ' if ok else 'NO '} paused half way at full speed: it stopped "
        f"{rest.distance_to(told) * 1000:.0f} mm on, and went on to "
        f"{st.tcp().distance_to(goal) * 1000:.1f} mm from its goal")
    return ok and st.go(start)


def verify(folder: Path, say=print, hover_m: float = 0.03) -> None:
    """The tool centre point ``hover_m`` over each of the board's corners, as the
    fixed camera and the calibration place them; the pictures kept, marked where the
    calibration has the tool centre point, the fingertips and the corner: on the
    real fingers, or the table's height is wrong.  The operator looks."""
    from PIL import Image

    cal = calibration_mod.load()
    if cal.provisional:
        raise SystemExit("the station is not calibrated yet: run first")
    board, square = Board.from_dict(cal.fit["board"]), float(cal.fit["square_m"])
    folder.mkdir(parents=True, exist_ok=True)
    st = Station()
    try:
        k = st.k("agentview")
        seen = see(st.engine.render("agentview").image, board, upscale=3)
        if seen is None:
            raise SystemExit("the fixed camera does not see the board")
        obj, px = seen.points(board)
        camera = st.engine.camera_pose("agentview")
        on_table = compose(camera, locate(obj * square, px, k)[0])
        to_camera = np.linalg.inv(_matrix(camera))

        def mark(image, point, colour):
            u, v, w = k @ (to_camera @ np.array([*point, 1.0]))[:3]
            cv2.drawMarker(image, (int(round(u / w)), int(round(v / w))), colour,
                           cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)

        for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
            x, y, z = board_point(board, on_table, square, u, v)
            target = Pose((x, y, z + hover_m), GRASP_DOWN)
            if not st.go(target):
                say(f"   the corner ({u}, {v}) at {x:.3f}, {y:.3f} is out of the arm's way")
                continue
            here = st.tcp()
            front = np.array(st.engine.render("agentview").image)
            mark(front, here.pos, (255, 0, 0))
            mark(front, (x, y, z), (0, 255, 0))
            width = st.engine.state().arm(st.engine.arm_id).opening
            for side in (-0.5, 0.5):
                mark(front, compose(here, Pose((0.0, side * width, FINGERTIP_M))).pos, (255, 255, 0))
            Image.fromarray(front).save(folder / f"front_{u}{v}.png")
            Image.fromarray(st.engine.render("wrist").image).save(folder / f"wrist_{u}{v}.png")
            say(f"   over the corner ({u}, {v}) at {x:.3f}, {y:.3f}, table height {z:.4f}: the "
                f"hand is {here.distance_to(target) * 1000:.1f} mm from its target")
            if sys.stdin.isatty():
                input(question(f"is the fingers' middle right over it, {hover_m * 100:g} cm up?  "
                               "Enter for the next one"))
        st.start()
    finally:
        st.close()
    say(f"pictures in {folder}")


def write_board(path: Path, board: Board, square_mm: float = 35.0, dpi: int = 300) -> Path:
    """A board as a PDF of its true size.  Printers scale: print at 100 %."""
    from PIL import Image

    px = square_mm / 25.4 * dpi
    w, h = (round(n * px) for n in board.squares)
    margin = round(10 / 25.4 * dpi)
    image = board.charuco(square_mm / 1000.0).generateImage((w, h), marginSize=0, borderBits=1)
    page = Image.new("L", (w + 2 * margin, h + 2 * margin), 255)
    page.paste(Image.fromarray(image), (margin, margin))
    page.save(path, resolution=dpi)
    return path


def _board_arguments(p) -> None:
    p.add_argument("--squares", type=int, nargs=2, default=list(Board.squares),
                   metavar=("ACROSS", "DOWN"), help="the board's squares (default: the lab's sheet)")
    p.add_argument("--dictionary", default=Board.dictionary)
    p.add_argument("--no-legacy", action="store_true",
                   help="laid out as OpenCV has since 4.6 (the lab's sheet is not)")


def _board(args) -> Board:
    return Board(tuple(args.squares), args.dictionary, legacy=not args.no_legacy)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    bd = sub.add_parser("board", help="write a board to print")
    bd.add_argument("--out", default="charuco_board.pdf")
    bd.add_argument("--square-mm", type=float, default=35.0)
    _board_arguments(bd)
    mv = sub.add_parser("moves", help="the first moves on the robot (the arm moves)")
    mv.add_argument("--once", action="store_true", help="nothing if they have passed already")
    rn = sub.add_parser("run", help="measure the station (the arm moves)")
    _board_arguments(rn)
    so = sub.add_parser("solve", help="solve again what a run kept")
    so.add_argument("folder")
    for p in (rn, so):
        p.add_argument("--square-mm", type=float, default=None,
                       help="a square of the board, if measured; else the arm measures it")
        p.add_argument("--thickness-mm", type=float, default=0.0,
                       help="what the board is printed on, if it is not paper on the table")
    ve = sub.add_parser("verify", help="check the calibration (the arm moves)")
    ve.add_argument("--hover-mm", type=float, default=30.0,
                    help="how far over the board's corners the tool centre point goes")
    args = ap.parse_args(argv)
    if args.cmd == "board":
        print(write_board(Path(args.out), _board(args), args.square_mm))
        return 0
    if args.cmd == "solve":
        solve(Path(args.folder), args.square_mm and args.square_mm / 1000.0,
              args.thickness_mm / 1000.0)
        return 0
    if args.cmd == "moves" and args.once and MOVES_PASSED.exists():
        return 0
    what = {"moves": "around its start, down to 3 cm over the table, then 20 cm out and back",
            "run": "slowly over the table, high up, and round the board",
            "verify": "slowly to over the board's corners"}[args.cmd]
    go = input(question(f"The arm will move, {what}. Is the table clear under it and a hand on "
                        "the stop? [y/N]"))
    if go.strip().lower() != "y":
        return 1
    stamp = datetime.now().strftime("%m%d-%H%M%S")
    if args.cmd == "moves":
        return 0 if first_moves() else 1
    if args.cmd == "verify":
        verify(RUNS / f"verify-{stamp}", hover_m=args.hover_mm / 1000.0)
        return 0
    folder = capture(RUNS / stamp, _board(args))
    solve(folder, args.square_mm and args.square_mm / 1000.0, args.thickness_mm / 1000.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
