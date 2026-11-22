"""CL6 "the knight's move" -- In-Context Learning: a rule told, and a king that moves.

Knob: how far the black king is (four moves of the told knight -> six).

A marble chess board before the arm with an endgame on it: a white knight,
the white king and pawns, and black castled in a corner -- its king beside
its rook, pawns before the rook and in the corner file, and the square
between them covered by a white pawn -- so that the king has two squares and
steps from one to the other each time white has moved.  "Capture the
black king with the white knight.  The knight moves as in chess, but like the
horse in Chinese chess it cannot move past a piece standing next to it in
the direction of its two-square step.  Black moves after each of your moves."
One move at a time; the other pieces stay where they are; to capture, the
king comes off the board and the knight takes its square.  A knight left on
the king's other square is taken by the king.  Easy: four moves.  Hard:
six.  At both levels a chess knight's shortest way is shorter -- and passes
a blocked leg: the rule told is what decides the route, and the king's
steps decide when the knight has to be where.  (A king walled in on every
side by its own men could never be taken: a knight's leg is always a square
diagonally next to where it lands.  So the square diagonally before the
corner is left open, and a white pawn keeps the king off it; the king is
taken in the corner, from the square past the open one.)

The board is Poly Haven's marble set at its own size (tournament: 5.79 cm
squares, the king 9.5 cm), turned so that white sits at the robot's side.
Each square is (file, rank): file 0 is the a-file, on the robot's left; rank
0 is white's first rank, nearest the robot.  Black's king stands on a plate
under its two squares, hidden and a millimetre over the board, which a slide
moves a square at a time -- as an automatic chess board moves its pieces
with a magnet from below.

A referee (``Referee``) watches the knight: whenever it comes to rest after
being lifted it must stand within ``ON_SQUARE`` of a square's middle, and
that square must be a knight's move from the last one, past no piece on the
knight's leg, and empty -- or the king's, once the king is off the board.
Anything else is a foul, as is any other piece moved or knocked over, or the
king put down on the board.  Then black moves.

The Franka's hand is 21 cm long along its fingers' travel and the squares
5.8 cm: the knight is taken high, by its head, the fingers opened only as far
as they need, so that they pass over the pawns beside it; the hand turns to
a diagonal where a king or the rook stands beside the knight, and the
positions are drawn so that every move of the route has a turn that keeps
the hand clear of every piece around it -- the first one straight.

v8 (2026-09-24): the advisor -- "drop the Chinese chess film, introduce the
hobbled leg in words, and let black move too after every move: more real".
Until then (v7) a film on a Chinese chess board showed the horse blocked and
let through, and black never moved.

v6 (2026-09-22), replacing the toy catapult: the advisor's choice -- chess
(Chinese or international, whichever looks more real; Poly Haven's set is
photographic), a run of knight's moves to take the king.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from ...core.config import RunConfig
from ...core.rng import SeedBundle
from ...core.types import Pose, WorldState
from ...engines import parts as P
from ...engines.base import ArticulationSpec, BodySpec, Engine
from ...events.actions import DriveJoint
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...policies.scripted import turned_down
from ...scoring import Custom, Finally, Milestone, Never, Scorer, Upright
from ..instance import TaskInstance
from ..loader import get
from . import _toys as T
from ._common import RIGHT, TABLE, Frame, ahead, asset_body, station_scene

Square = Tuple[int, int]

# -- the board ----------------------------------------------------------------------
BOARD = "board"
#: the board's middle, ahead of the station line; its half size and top
BOARD_AT, BOARD_HALF = (0.35, 0.0), 0.2766
BOARD_TOP = TABLE + 0.0174
SQUARE = 0.0579
#: every piece is drawn turned so that it faces black, as it would in play
FACING = math.pi / 2
#: the pieces: name -> (catalogue entry, half extents for the mock)
KNIGHT, KING = "white_knight", "black_king"
PIECES = {KNIGHT: ("chess_knight_white", (0.0194, 0.0227, 0.0375)),
          KING: ("chess_king_black", (0.0208, 0.0209, 0.0475)),
          "white_king": ("chess_king_white", (0.0208, 0.0209, 0.0475)),
          "white_pawn1": ("chess_pawn_white", (0.0145, 0.0144, 0.0269)),
          "white_pawn2": ("chess_pawn_white", (0.0145, 0.0144, 0.0269)),
          "white_pawn3": ("chess_pawn_white", (0.0145, 0.0144, 0.0269)),
          "black_pawn1": ("chess_pawn_black", (0.0145, 0.0144, 0.0269)),
          "black_pawn2": ("chess_pawn_black", (0.0145, 0.0144, 0.0269)),
          "black_rook": ("chess_rook_black", (0.0184, 0.0183, 0.0303))}
WHITE_PAWNS = ("white_pawn1", "white_pawn2", "white_pawn3")
#: black castled, on either wing: the king's two squares (where it starts
#: first, then the corner), the two pawns (before the rook, and in the
#: corner file), the rook beside the king, and the white pawn
#: (``white_pawn1``) that keeps the king off the square between the pawns --
#: the king has no other square.  It is taken in the corner, from past the
#: open square (h8 from g6, past g7; a8 from b6, past b7)
CASTLED = {"king": {"king": ((6, 7), (7, 7)), "pawns": ((5, 6), (7, 6)), "rook": (5, 7),
                    "guard": (7, 5)},
           "queen": {"king": ((1, 7), (0, 7)), "pawns": ((2, 6), (0, 6)), "rook": (2, 7),
                     "guard": (0, 5)}}
#: where a taken piece goes: a square of green baize on the table, on the
#: robot's right beside the board
TAKEN, TAKEN_AT, TAKEN_HALF = "taken_mat", (0.30, -0.37), (0.05, 0.05)
#: the knight has come to rest on a square within this of its middle: well
#: inside the square, 2.9 cm to its edge (v2.3.4; 1.5 cm before, when five of
#: the fifteen knights called between squares stood upright inside one, 1.5
#: to 2.4 cm off its middle)
ON_SQUARE = 0.022
#: another piece has been disturbed once it is this far from where it stood
DISTURBED = 0.015
#: how far the king is, in moves of the knight told about (the capture the
#: last), at each level.  Even: the king is in its corner at white's even
#: moves.  Not two: in two moves the leg never matters (the one way in, over
#: the open square, is a chess knight's shortest way too)
DISTANCE = {"easy": 4, "hard": 6}
KNIGHT_STEPS = ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))

# -- black's king: a hidden plate that slides it a square at a time -------------------
CARRIAGE, KING_SLIDE = "king_carriage", "king_slide"
#: the plate: under the king's foot, a millimetre over the board so that it
#: slides free of it; its travel a square either way
PLATE_HALF, PLATE_GAP = (0.022, 0.022, 0.0005), 0.001
PLATE_TOP = BOARD_TOP + PLATE_GAP + 2 * PLATE_HALF[2]
#: black's move takes this long, eased in and out: the king is 9.5 cm tall
#: on a 4.2 cm foot and must not topple
BLACK_SECONDS = 1.2

# -- the hand -----------------------------------------------------------------------
#: the knight by its head, the king by its collar: this far above the board.
#: Taken by its neck (5.6 cm) the knight was 62% hidden under the hand from
#: the fixed camera; by the head, 54%
KNIGHT_GRASP, KING_GRASP = 0.066, 0.075
#: the fingers closed this far (of their 8 cm): open 6 cm, round a knight 3.9
#: by 4.5 cm at its foot -- turned 45 degrees, they come down beside its
#: corners -- and still 5 mm short of a piece on the next diagonal square
OPEN = 0.25
#: carried this high over the board: the knight's foot clears the king's crown
CARRY = 0.19
#: the king set down on the felt beside the board: the grasp's height over
#: the board's top, for a king standing on the felt
KING_ON_FELT = KING_GRASP - (BOARD_TOP - TABLE) + 2 * T.MAT_HALF_Z
#: the turns of the hand, best first.  Straight (the fingers across the
#: files, on the knight's cheeks) the fixed camera sees three quarters of the
#: knight in the hand; turned, a third to a half -- the wrist camera's
#: housing comes round in front of it.  Taken this high the fingertips pass
#: over the pawns on the files either side; a king or the rook there, and
#: the hand turns to a diagonal
TURNS = (0.0, math.radians(45), math.radians(-45), math.radians(90))
#: how far either finger's inside stands from the middle, opened to OPEN
FINGER_IN = 0.04 * (1.0 - OPEN)


#: the part of the table the fixed camera shows, with the gripper where it
#: starts: the board, and the felt the king goes to
FRAME = Frame((0.06, 0.64), (-0.43, 0.29), 0.12)

INSTRUCTION = ("Capture the black king with the white knight. The knight moves as in chess, but like "
               "the horse in Chinese chess it cannot move past a piece standing next to it in the "
               "direction of its two-square step. Black moves after each of your moves.")


def centre(sq: Square) -> Tuple[float, float]:
    f, r = sq
    bx, by = ahead(*BOARD_AT)
    return (bx + (r - 3.5) * SQUARE, by + (3.5 - f) * SQUARE)


def square_of(xy) -> Optional[Square]:
    """The square whose middle is within ``ON_SQUARE`` of ``xy``, if any."""
    bx, by = ahead(*BOARD_AT)
    r = round((xy[0] - bx) / SQUARE + 3.5)
    f = round(3.5 - (xy[1] - by) / SQUARE)
    if not (0 <= f < 8 and 0 <= r < 8):
        return None
    cx, cy = centre((f, r))
    return (f, r) if math.hypot(xy[0] - cx, xy[1] - cy) <= ON_SQUARE else None


def on_board(xy) -> bool:
    bx, by = ahead(*BOARD_AT)
    return abs(xy[0] - bx) < BOARD_HALF and abs(xy[1] - by) < BOARD_HALF


def name_of(sq: Square) -> str:
    return "abcdefgh"[sq[0]] + str(sq[1] + 1)


def knight_move(a: Square, b: Square) -> bool:
    return (abs(a[0] - b[0]), abs(a[1] - b[1])) in ((1, 2), (2, 1))


def leg(a: Square, b: Square) -> Square:
    """The square a knight's move from ``a`` to ``b`` goes past: next to
    ``a``, toward the move's two-square side.  A piece there blocks the move
    -- the Chinese chess horse's rule, which the instruction tells."""
    df, dr = b[0] - a[0], b[1] - a[1]
    return (a[0] + df // 2, a[1]) if abs(df) == 2 else (a[0], a[1] + dr // 2)


def route(start: Square, kings: Tuple[Square, Square], taken, hobbled: bool = True,
          moving: bool = True) -> Optional[List[Square]]:
    """The shortest knight's route from ``start`` to the capture, with black's
    king on ``kings[0]`` now and stepping to the other square after every
    white move (``moving``; not: it stays where it is): landing on no square
    in ``taken`` -- the other pieces -- nor on the king's other square, where
    the king would take the knight; if ``hobbled`` (the rule told), past no
    piece on a leg, the king included.  The first of the moves in
    ``KNIGHT_STEPS`` order is preferred.  None if there is none."""
    a, b = kings

    def king(t: int) -> Square:
        return a if not moving or t % 2 == 0 else b

    seen: Dict[Tuple[Square, int], Optional[Tuple[Square, int]]] = {(start, 0): None}
    todo = deque([(start, 0)])
    while todo:
        sq, par = todo.popleft()
        here = king(par)
        standing = set(taken) | {here}
        for df, dr in KNIGHT_STEPS:
            nxt = (sq[0] + df, sq[1] + dr)
            if not (0 <= nxt[0] < 8 and 0 <= nxt[1] < 8):
                continue
            if hobbled and leg(sq, nxt) in standing:
                continue
            if nxt == here:
                path, state = [nxt], (sq, par)
                while state is not None:
                    path.append(state[0])
                    state = seen[state]
                return path[::-1]
            if nxt in standing or nxt in (a, b):
                continue
            state = (nxt, 1 - par)
            if state not in seen:
                seen[state] = (sq, par)
                todo.append(state)
    return None


def hand_clear(at: Tuple[float, float], yaw: float, grasp: float,
               others: Dict[str, Tuple[float, float]]) -> bool:
    """Is a hand over ``at`` turned ``yaw``, its fingers ``grasp`` over the
    board and opened only to ``OPEN``, clear of every other piece?  The same
    envelope and margin ``validate`` checks a grasp with (``clearance``): the
    Franka hand's parts as turned rectangles, the pieces as upright boxes."""
    from ...engines.robots import FR3
    from ..clearance import MARGIN, _turned_overlap, gripper_parts

    parts = gripper_parts((at[0], at[1], BOARD_TOP + grasp), FR3, yaw, FINGER_IN)
    for name, (x, y) in others.items():
        hx, hy, hz = PIECES[name][1]
        if name == KNIGHT:
            hx, hy = hy, hx                           # turned to face black
        box = ((x - hx, x + hx), (y - hy, y + hy), (BOARD_TOP, BOARD_TOP + 2 * hz))
        if any(_turned_overlap(corners, bottom, box, MARGIN) for _, corners, bottom in parts):
            return False
    return True


def turn_for(sq_from: Square, sq_to: Square, where: Dict[str, Square],
             turns=TURNS) -> Optional[Tuple[float, float]]:
    """The turns of the hand that take the knight up on ``sq_from`` and set it
    down on ``sq_to`` -- the first of ``turns`` at each end with every other
    piece clear of the hand (the king already off the board if ``sq_to`` is
    its square); the hand turns in the air between them.  None if either end
    has none."""
    others = {n: centre(s) for n, s in where.items() if n != KNIGHT and s != sq_to}
    up = next((yaw for yaw in turns if hand_clear(centre(sq_from), yaw, KNIGHT_GRASP, others)), None)
    down = next((yaw for yaw in turns if hand_clear(centre(sq_to), yaw, KNIGHT_GRASP, others)), None)
    return None if up is None or down is None else (up, down)


def position(level: str, seed: int) -> Tuple[Dict[str, Square], Tuple[Square, Square]]:
    """Where every piece stands, and the black king's two squares (where it
    stands first).  Black castled on a wing drawn per seed; the white king
    and white's other pawns in white's half, the knight anywhere up to the
    sixth rank -- close by, the capture it seems to have is blocked by a
    leg.  The king exactly
    ``DISTANCE`` moves of the knight told about away, as its steps go -- and
    fewer for a chess knight, whose shortest ways all pass a blocked leg.
    Every move of the reference's route can be made with the hand clear."""
    rng = SeedBundle(seed).stream(f"chess/{level}/castled")
    want = DISTANCE[level]
    for _ in range(40000):
        wing = CASTLED[rng.choice(("king", "queen"))]
        kings = wing["king"]
        knight = (rng.randrange(0, 8), rng.randrange(0, 6))
        white_king = (rng.randrange(0, 8), rng.randrange(0, 2))
        pawns = [wing["guard"]] + [(rng.randrange(0, 8), rng.randrange(1, 4)) for _ in WHITE_PAWNS[1:]]
        squares = [knight, kings[0], white_king] + pawns + list(wing["pawns"]) + [wing["rook"]]
        if len(set(squares) | {kings[1]}) != len(squares) + 1:
            continue
        where = dict(zip(PIECES, squares))
        taken = set(squares) - {knight, kings[0]}
        path = route(knight, kings, taken)
        if path is None or len(path) - 1 != want:
            continue
        chess = route(knight, kings, taken, hobbled=False)
        if chess is None or len(chess) - 1 >= want:
            continue                                  # the leg would not matter
        # every move with the hand clear, the king where it is by then (off
        # the board for the capture)
        at, ok = knight, True
        for t, sq in enumerate(path[1:]):
            now = dict(where, **{KNIGHT: at})
            now[KING] = kings[t % 2]
            if sq == now[KING]:
                del now[KING]
            if not ok or turn_for(at, sq, now) is None:
                ok = False
                break
            at = sq
        if ok and knight_in_sight(level, where, kings):
            return where, kings
    raise RuntimeError(f"no chess position for {level}/{seed}")


def king_carriage(sq: Square) -> Tuple[BodySpec, ArticulationSpec]:
    """The plate the black king stands on, and the slide that moves it along
    the rank: a square either way from where it starts."""
    x, y = centre(sq)
    plate = BodySpec(CARRIAGE, "compound", PLATE_HALF, Pose((x, y, BOARD_TOP + PLATE_GAP + PLATE_HALF[2])),
                     category="automatic board's carriage", mass=0.05,
                     attrs={"parts": (P.grip((0, 0, 0), PLATE_HALF),), "graspable": False,
                            "carried_by": KING_SLIDE, "driven": True, "footprint": "rect"})
    slide = ArticulationSpec(KING_SLIDE, "slide", axis=(0.0, 1.0, 0.0),
                             limits=(-SQUARE - 0.005, SQUARE + 0.005), initial=0.0, actuated=True,
                             parent=CARRIAGE, carries=True)
    return plate, slide


def scene_for(level: str, where: Dict[str, Square], kings: Tuple[Square, Square]):
    taken = set(where.values()) - {where[KNIGHT], where[KING]}
    first = route(where[KNIGHT], kings, taken)[1]
    bx, by = ahead(*BOARD_AT)
    board = asset_body(BOARD, "chess_board", (bx, by), fallback_size=(BOARD_HALF, BOARD_HALF, 0.0087),
                       fixed=True, category="chess board", graspable=False, yaw=FACING)
    bodies = [board]
    for name, (asset, half) in PIECES.items():
        x, y = centre(where[name])
        # the knight is taken as the reference takes it for its first move --
        # which is what the grasp check in ``validate`` looks at
        yaw = (turn_for(where[KNIGHT], first, where) or (0.0, 0.0))[0] if name == KNIGHT else 0.0
        attrs = {"driven": True} if name == KING else {}
        body = asset_body(name, asset, (x, y), fallback_size=half, mass=0.05, yaw=FACING,
                          graspable=name in (KNIGHT, KING), grasp_yaw=yaw, grasp_open=FINGER_IN,
                          grasp_height=(KNIGHT_GRASP if name == KNIGHT else KING_GRASP) - half[2],
                          **attrs)
        foot = PLATE_TOP if name == KING else BOARD_TOP
        bodies.append(replace(body, pose=Pose((x, y, foot + body.size[2]), body.pose.quat)))
    plate, slide = king_carriage(kings[0])
    bodies += [plate, T.mat(TAKEN, ahead(*TAKEN_AT), TAKEN_HALF, "fabric:baize")]
    return station_scene(f"cl6_chess_{level}", bodies, articulations=(slide,), frame=FRAME)


def build_scene(level: str, seed: int):
    where, kings = position(level, seed)
    return scene_for(level, where, kings), where, kings


def knight_in_sight(level: str, where: Dict[str, Square], kings) -> bool:
    """Can the fixed camera, beyond black's side of the board, see enough of
    the knight at the start -- the pieces before it may hide it -- and in the
    hand for its first move, turned as that move turns it?  The checks
    ``validate`` makes (``sightlines``)."""
    from ..sightlines import MAX_HIDDEN_AT_START, MAX_HIDDEN_IN_GRASP, hidden_at_start, hidden_in_grasp

    scene = scene_for(level, where, kings)
    return (hidden_at_start(scene, KNIGHT) <= MAX_HIDDEN_AT_START
            and hidden_in_grasp(scene, KNIGHT) <= MAX_HIDDEN_IN_GRASP)


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("CL6")
    scene, where, kings = build_scene(level, seed)
    moved = lambda w: sum(1 for e in w.events_fired if e.startswith("knight_to_"))  # noqa: E731
    foul = Custom(lambda w: any(e.startswith(FOUL) for e in w.events_fired), "a foul")
    taken = Custom(lambda w: CAPTURED in w.events_fired and not w.obj(KNIGHT).is_held
                   and not on_board(w.obj(KING).pose.pos) and Upright(KNIGHT).holds(w),
                   "the king taken, the knight on its square")
    milestones = (
        Milestone("a_move", Custom(lambda w: moved(w) >= 1, "a knight's move")),
        Milestone("black_moved", Custom(lambda w: any(e.startswith(BLACK) for e in w.events_fired),
                                        "black has answered")),
        Milestone("king_off", Custom(lambda w: not on_board(w.obj(KING).pose.pos)
                                     and not w.obj(KING).is_held, "the king off the board")),
        Milestone("captured", taken),
    )
    path = route(where[KNIGHT], kings, set(where.values()) - {where[KNIGHT], where[KING]})
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene, instruction=INSTRUCTION,
        scorer=Scorer(task_id="CL6", level=level, success=Finally(taken) & Never(foul),
                      milestones=milestones),
        milestones=milestones,
        reactor_factory=lambda w=dict(where), k=kings: Referee(w, k),
        reference_factory=lambda inst, w=dict(where), k=kings: KnightPolicy(w, k),
        metadata={"knob": spec.knob.field, "knob_value": DISTANCE[level],
                  # hidden from the agent: the position and a shortest route
                  "position": {n: name_of(s) for n, s in where.items()},
                  "king_squares": [name_of(s) for s in kings],
                  "route": [name_of(s) for s in path], "max_model_calls": 60},
        extra_metrics={"knight_moves": lambda traj: moved(traj[-1]) if traj else 0},
    )


FOUL = "foul"
CAPTURED = "king_captured"
BLACK = "black_king_to_"


def slide_value(kings: Tuple[Square, Square], sq: Square) -> float:
    """Where the slide stands with the king on ``sq``: 0 on its first square."""
    return centre(sq)[1] - centre(kings[0])[1]


class Referee(Reactor):
    """Watches the board, and plays black.  The knight lifted and come to
    rest again has made a move: on a square's middle, a knight's move from
    where it stood, past no piece on its leg, onto an empty square -- or the
    king's, the king already off the board (the capture).  Any other piece
    moved or knocked over, or the king put down on the board, is a foul.
    After each move black's king steps to its other square, carried by the
    plate under it -- and a knight standing there is taken."""

    def __init__(self, where: Dict[str, Square], kings: Tuple[Square, Square]) -> None:
        super().__init__()
        self.start = dict(where)
        self.kings = tuple(kings)
        self.at = where[KNIGHT]
        self.king_at = kings[0]
        self._lifted = False
        self._quiet = 0
        self._fouled = set()
        #: black's move under way: (from value, to value, started at)
        self._black: Optional[Tuple[float, float, float]] = None
        self._over = False

    def _foul(self, engine, world, what: str) -> None:
        if what not in self._fouled:
            self._fouled.add(what)
            self.note(engine, world, f"{FOUL}_{what}")
            self.say(world, KNIGHT, "foul", "That is not allowed.")

    def _black_step(self, engine: Engine, world: WorldState) -> None:
        a, b, since = self._black
        f = min(1.0, (world.sim_time - since) / BLACK_SECONDS)
        eased = 0.5 - 0.5 * math.cos(math.pi * f)
        if f < 1.0:
            engine.set_articulation_target(KING_SLIDE, a + (b - a) * eased)
            return
        self._black = None
        self.act(engine, world, f"{BLACK}{name_of(self.king_at)}", DriveJoint(KING_SLIDE, b))

    def _answer(self, engine: Engine, world: WorldState) -> None:
        """Black's move: the king to its other square -- taking the knight if
        it stands there."""
        other = self.kings[1] if self.king_at == self.kings[0] else self.kings[0]
        if self.at == other:
            self._over = True
            self._foul(engine, world, "knight_taken_by_the_king")
            return
        a = slide_value(self.kings, self.king_at)
        self.king_at = other
        self._black = (a, slide_value(self.kings, other), world.sim_time)

    def tick(self, engine: Engine, world: WorldState) -> None:
        if self._black is not None:
            self._black_step(engine, world)
        for name, sq in self.start.items():
            if name in (KNIGHT, KING):
                continue
            o = world.obj(name)
            cx, cy = centre(sq)
            if (o.is_held or math.hypot(o.pose.pos[0] - cx, o.pose.pos[1] - cy) > DISTURBED
                    or not Upright(name).holds(world)):
                self._foul(engine, world, f"{name}_moved")
        king = world.obj(KING)
        kx, ky = centre(self.king_at)
        if (self._black is None and not king.is_held and king.speed < 0.01 and on_board(king.pose.pos)
                and math.hypot(king.pose.pos[0] - kx, king.pose.pos[1] - ky) > DISTURBED):
            self._foul(engine, world, "king_left_on_the_board")
        knight = world.obj(KNIGHT)
        if knight.is_held:
            self._lifted, self._quiet = True, 0
            return
        if not self._lifted or self._over:
            return
        self._quiet = self._quiet + 1 if knight.speed < 0.01 else 0
        if self._quiet < 3:
            return                                      # still settling
        self._lifted = False
        sq = square_of(knight.pose.pos)
        if sq is None or not Upright(KNIGHT).holds(world):
            self._foul(engine, world, "knight_between_squares")
            return
        if sq == self.at:
            return                                      # put back where it was
        others = {s for n, s in self.start.items() if n not in (KNIGHT, KING)}
        king_on = on_board(king.pose.pos)
        standing = others | ({self.king_at} if king_on else set())
        capture = sq == self.king_at and not king_on
        if (not knight_move(self.at, sq) or sq in others or (sq == self.king_at and king_on)
                or leg(self.at, sq) in standing):
            self._foul(engine, world, f"illegal_move_{name_of(self.at)}_{name_of(sq)}")
            self.at = sq
            return
        self.note(engine, world, f"knight_to_{name_of(sq)}")
        self.at = sq
        if capture:
            self._over = True
            self.note(engine, world, CAPTURED)
            self.say(world, KING, "captured", "Check -- and taken!")
            return
        self._answer(engine, world)


class _Mover(ScriptedPolicy):
    """Moves one piece at a time: over it, down, closed, up to ``CARRY``,
    over the target, down, open, up -- the hand turned to a diagonal and the
    fingers only as open as ``OPEN``."""

    arm = RIGHT

    def _lift(self, world: WorldState, name: str, yaw: float, grasp: float) -> List[Command]:
        x, y, _ = world.obj(name).pose.pos
        q = turned_down(yaw)
        return [SetGripper(self.arm, OPEN), MoveTo(self.arm, Pose((x, y, BOARD_TOP + CARRY), q)),
                MoveTo(self.arm, Pose((x, y, BOARD_TOP + grasp), q)), SetGripper(self.arm, 1.0),
                Wait(0.3), MoveTo(self.arm, Pose((x, y, BOARD_TOP + CARRY), q))]

    def _set_down(self, xy, yaw: float, grasp: float, lift_to: float = CARRY) -> List[Command]:
        q = turned_down(yaw)
        x, y = xy
        return [MoveTo(self.arm, Pose((x, y, BOARD_TOP + CARRY), q)),
                MoveTo(self.arm, Pose((x, y, BOARD_TOP + grasp + 0.002), q)),
                SetGripper(self.arm, OPEN), Wait(0.4),
                MoveTo(self.arm, Pose((x, y, BOARD_TOP + lift_to), q))]


class KnightPolicy(_Mover):
    """Each turn, once black has answered: where the king stands now, the
    shortest route of the knight told about from where the knight stands --
    the king stepping to its other square after every move -- and its first
    move; before the capture, the king off the board onto the felt.
    ``ignore_rules`` (the blind control): a chess player's knight, past
    pieces on its leg -- which every short way is, a foul.  Under
    ``ignore_change`` the king is always where it was first seen, and the
    knight goes where the king will not be."""

    name = "knight"

    def __init__(self, where: Dict[str, Square], kings: Tuple[Square, Square]) -> None:
        self.where = dict(where)
        self.kings = tuple(kings)
        self.ignore_rules = False
        self.reset()

    def reset(self) -> None:
        self._sent: Optional[Square] = None
        self._capture_at: Optional[Square] = None
        self._tries = 0

    def _next(self, world: WorldState, at: Square) -> Optional[Square]:
        """The next square on the way: the king's own for the capture."""
        king = world.obj(KING)
        now = square_of(king.pose.pos) if on_board(king.pose.pos) else None
        if now is None:                             # taken off: its square is where it stood
            return self._capture_at
        other = self.kings[1] if now == self.kings[0] else self.kings[0]
        taken = {s for n, s in self.where.items() if n not in (KNIGHT, KING)}
        path = route(at, (now, other), taken, hobbled=not self.ignore_rules)
        return path[1] if path else None

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        fired = world.events_fired
        if CAPTURED in fired:
            king = world.obj(KING)
            return [MoveTo(self.arm, Pose((king.pose.pos[0], king.pose.pos[1], TABLE + 0.30),
                                          turned_down(0.0))), Wait(1.0), Done("the king is taken")]
        if any(e.startswith(FOUL) for e in fired):
            return [Done("a foul")]
        moves = sum(1 for e in fired if e.startswith("knight_to_"))
        answers = sum(1 for e in fired if e.startswith(BLACK))
        if answers < moves:
            return [Wait(0.3)]                           # black to move
        at = square_of(world.obj(KNIGHT).pose.pos)
        if at is None or (self._sent is not None and at != self._sent):
            self._tries += 1                            # still settling, or it did not get there
            if self._tries > 8:
                return [Done("the knight is not where it was sent")]
            return [Wait(0.3)]
        self._sent, self._tries = None, 0
        nxt = self._next(world, at)
        if nxt is None:
            return [Done("no way to the king")]
        king = world.obj(KING)
        where = {n: s for n, s in self.where.items() if n not in (KNIGHT, KING)}
        where[KNIGHT] = at
        king_sq = square_of(king.pose.pos) if on_board(king.pose.pos) else None
        if nxt == king_sq:
            # the capture: the king off the board first, onto the felt
            self._capture_at = nxt
            return (self._lift(world, KING, 0.0, KING_GRASP)
                    + self._set_down(ahead(*TAKEN_AT), 0.0, KING_ON_FELT) + [Wait(0.3)])
        if king_sq is not None:
            where[KING] = king_sq
        up, down = turn_for(at, nxt, where) or (0.0, 0.0)
        self._sent = nxt
        return (self._lift(world, KNIGHT, up, KNIGHT_GRASP)
                + self._set_down(centre(nxt), down, KNIGHT_GRASP) + [Wait(0.5)])

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        return [Done("no plan without watching black")]
