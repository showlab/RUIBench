"""WK6 "tic-tac-toe" -- Dynamic Interaction.

Knob: where the win comes from (one move on the board -> a single mistake
of an otherwise perfect opponent, early in the game).

A tic-tac-toe board with a game under way and spare Xs beside it.  "You are
X. Win the game."  The opponent plays O: once an X is down on a cell and the
hand is off the board, an O comes up out of the board.

Easy: two Xs and two Os; X wins with one move and O threatens nothing.
Hard (v3): an X and an O -- a game perfect play draws.  The opponent plays
perfectly but once: at its first or second reply (drawn per seed) it plays
a move that loses, though not at once -- X has no three in a row to take,
and wins only by seeing the chance and forking.  Play safe for a draw and the
chance passes; the win is not there at the start, it is the opponent's move
that gives it.  The openings are every opening where the reference, playing
this opponent, wins (a search, below).

Played out, not just judged: an X may not go on a taken cell, a placed X may
not be moved, and the hand stays off the board while an O comes up.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.types import Pose, WorldState
from ...engines.base import Engine
from ...events.actions import DriveJoint
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, Wait
from ...policies.base import ScriptedPolicy
from ...policies.scripted import MAX_GRASP_ATTEMPTS, PlacePosePolicy, turned_down
from ...scoring import AtRest, Custom, Finally, Milestone, Never, Scorer
from ..instance import TaskInstance
from ..loader import get
from . import _toys as T
from ._common import RIGHT, Frame, ahead, station_scene

# -- the game -------------------------------------------------------------------------
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
Cells = FrozenSet[int]


#: the part of the table the fixed camera shows, with the gripper where it
#: starts (``_common.Frame``)
FRAME = Frame((0.100, 0.440), (-0.300, 0.150), 0.09)


def winner(x: Cells, o: Cells) -> Optional[str]:
    for line in LINES:
        if all(c in x for c in line):
            return "X"
        if all(c in o for c in line):
            return "O"
    return None


def threats(mine: Cells, theirs: Cells) -> FrozenSet[int]:
    """The empty cells that would finish a line of ``mine``."""
    out = set()
    for line in LINES:
        have = [c for c in line if c in mine]
        free = [c for c in line if c not in mine and c not in theirs]
        if len(have) == 2 and len(free) == 1:
            out.add(free[0])
    return frozenset(out)


def empty(x: Cells, o: Cells) -> List[int]:
    return [c for c in range(9) if c not in x and c not in o]


@lru_cache(maxsize=None)
def value(x: Cells, o: Cells, x_moves: bool) -> int:
    """+1: X wins, -1: O wins, 0: a draw -- both playing perfectly."""
    w = winner(x, o)
    if w:
        return 1 if w == "X" else -1
    free = empty(x, o)
    if not free:
        return 0
    if x_moves:
        return max(value(x | {c}, o, False) for c in free)
    return min(value(x, o | {c}, True) for c in free)


def best_o(x: Cells, o: Cells) -> int:
    """The perfect opponent's move (the first, of equals)."""
    free = empty(x, o)
    win = [c for c in free if winner(x, o | {c}) == "O"]
    if win:
        return win[0]
    return min(free, key=lambda c: (value(x, o | {c}, True), c))


def best_x(x: Cells, o: Cells) -> int:
    """A win now if there is one, else the best move (the first, of equals)."""
    free = empty(x, o)
    win = [c for c in free if winner(x | {c}, o) == "X"]
    if win:
        return win[0]
    return max(free, key=lambda c: (value(x | {c}, o, False), -c))


def greedy_x(x: Cells, o: Cells) -> int:
    """Wins if it can; otherwise makes the most lines of its own -- and never
    looks at the opponent's."""
    free = empty(x, o)
    win = [c for c in free if winner(x | {c}, o) == "X"]
    if win:
        return win[0]
    return max(free, key=lambda c: (len(threats(x | {c}, o)), -c))


def mistakes(x: Cells, o: Cells) -> List[int]:
    """O's moves that lose, but not at once: after one, X can force a win
    (a fork) and has no three in a row to take."""
    return [c for c in empty(x, o)
            if value(x, o | {c}, True) == 1 and not threats(x, o | {c})]


def o_move(x: Cells, o: Cells, reply: int, mistake_at: Optional[int], pick: int) -> int:
    """The opponent's ``reply``-th move (from 1): perfect, except at the
    first reply from ``mistake_at`` on where a mistake is there to make."""
    if mistake_at is not None and reply >= mistake_at:
        bad = mistakes(x, o)
        if bad:
            return bad[pick % len(bad)]
    return best_o(x, o)


def play_out(xs, os_, mistake_at: Optional[int], pick: int, choose=None) -> Tuple[Optional[str], int]:
    """The game from (xs, os_), X choosing by ``choose`` (the reference's
    ``best_x``) against this opponent: (the winner, the reply it erred at)."""
    choose = choose or best_x
    x, o = frozenset(xs), frozenset(os_)
    reply, erred = 0, 0
    while not winner(x, o) and empty(x, o):
        x = x | {choose(x, o)}
        if winner(x, o) or not empty(x, o):
            break
        reply += 1
        c = o_move(x, o, reply, None if erred else mistake_at, pick)
        if not erred and mistake_at is not None and c != best_o(x, o):
            erred = reply
        o = o | {c}
    return winner(x, o), erred


#: which of its replies the hard opponent errs at, per seed
MISTAKE_AT = (1, 2)


@lru_cache(maxsize=None)
def puzzles(level: str) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...], Optional[int]], ...]:
    """Every start the level asks for: (Xs, Os, the reply O errs at).  Easy:
    two Xs and two Os, X to move, a win on the board and no O threat.  Hard:
    one X and one O, a draw with perfect play, where the reference wins
    against the opponent erring at that reply."""
    out = []
    if level == "easy":
        for xs in combinations(range(9), 2):
            for os_ in combinations([c for c in range(9) if c not in xs], 2):
                x, o = frozenset(xs), frozenset(os_)
                if not winner(x, o) and threats(x, o) and not threats(o, x):
                    out.append((xs, os_, None))
        return tuple(out)
    for m in MISTAKE_AT:
        for xc in range(9):
            for oc in range(9):
                if oc == xc or value(frozenset({xc}), frozenset({oc}), True) != 0:
                    continue
                won, erred = play_out((xc,), (oc,), m, 0)
                if won == "X" and erred == m:
                    out.append(((xc,), (oc,), m))
    # seed after seed, the mistake comes at the first reply, then the second
    rank = {m: [p for p in out if p[2] == m] for m in MISTAKE_AT}
    return tuple(sorted(out, key=lambda p: (rank[p[2]].index(p), p[2])))


def start(level: str, seed: int):
    bank = puzzles(level)
    return bank[seed % len(bank)]


# -- the board ------------------------------------------------------------------------
BOARD = "board"
XS = tuple(f"x{i}" for i in range(5))           # two on the board, three to play
RINGS = tuple(f"o{c}" for c in range(9))
LIFTS = tuple(f"o{c}_lift" for c in range(9))
BOARD_AT = (0.29, 0.0)
#: the spare Xs in a column beside the board, far enough out that the
#: Franka's hand (21 cm along its fingers) clears the 6 cm board over them
SPARE_X, SPARE_Y = (0.14, 0.22, 0.30, 0.38), -0.24
#: an X stands on a cell when it is this close to its middle, on the post
ON_CELL = 0.015
X_REST = T.TTT_TOP + T.CROSS_HALF_HEIGHT
#: the hand is off the board this far outside its edge
CLEAR = 0.03
RISE_SECONDS = 0.8
UP = -0.004                                     # an O this high is on the board


def build_scene(level: str, seed: int):
    xs, os_, _ = start(level, seed)
    board = T.ttt_board(BOARD, ahead(*BOARD_AT))
    bodies = [board]
    for name, cell in zip(XS, xs):
        # already played: never to be picked up again
        piece = T.x_piece(name, T.ttt_cell(board, cell), on=T.TTT_TOP)
        bodies.append(replace(piece, attrs={**piece.attrs, "graspable": False}))
    # the spares lie square (a hand turned 45 degrees would hide them), and
    # are turned into Xs on the way to the board
    for name, x in zip(XS[len(xs):], SPARE_X):
        bodies.append(T.x_piece(name, ahead(x, SPARE_Y), yaw=0.0))
    joints = []
    for cell, (name, lift) in enumerate(zip(RINGS, LIFTS)):
        ring, joint = T.o_ring(name, T.ttt_cell(board, cell), lift,
                               0.0 if cell in os_ else T.O_DOWN)
        bodies.append(ring)
        joints.append(joint)
    return station_scene(f"rs3_tic_tac_toe_{level}", bodies, articulations=joints, frame=FRAME), board, xs, os_


def x_cells(w: WorldState, board, stacked: bool = False) -> Dict[str, int]:
    """Which X stands on which cell's post -- or, ``stacked``, lies over a
    cell on something else (an O, another X)."""
    out = {}
    for name in XS:
        obj = w.obj(name)
        if obj.is_held:
            continue
        low = abs(obj.pose.pos[2] - X_REST) <= 0.008
        if stacked == low or (stacked and obj.pose.pos[2] < X_REST):
            continue
        for cell in range(9):
            if math.dist(obj.pose.pos[:2], T.ttt_cell(board, cell)) < ON_CELL:
                out[name] = cell
    return out


def o_cells(w: WorldState) -> FrozenSet[int]:
    return frozenset(c for c, lift in enumerate(LIFTS) if w.articulations.get(lift, T.O_DOWN) > UP)


def hand_on_board(w: WorldState, board) -> bool:
    bx, by, _ = board.pose.pos
    reach = T.TTT_HALF + CLEAR
    return any(abs(w.arm(a).eef_pose.pos[0] - bx) < reach and abs(w.arm(a).eef_pose.pos[1] - by) < reach
               for a in w.arm_ids)


def events(w: WorldState, prefix: str) -> List[str]:
    return [e[len(prefix):] for e in w.events_fired if e.startswith(prefix)]


def fouls_in_play(w: WorldState) -> List[str]:
    """The fouls before the game was over: a piece knocked once it is won or
    drawn changes nothing on the board that counts (v2.3.4: an X brushed as
    the hand left the winning move had failed a won game)."""
    out = []
    for e in w.events_fired:
        if e.startswith("won:"):
            break
        if e.startswith("foul:"):
            out.append(e[len("foul:"):])
    return out


class Opponent(Reactor):
    """Plays O, and keeps the score: records every move (``x:<cell>``,
    ``o:<cell>``), the result (``won:X`` / ``won:O`` / ``won:-``), and any
    foul (``foul:<what>``)."""

    def __init__(self, board, xs: Sequence[int], os_: Sequence[int],
                 mistake_at: Optional[int] = None, pick: int = 0) -> None:
        super().__init__()
        self.board = board
        self.mistake_at, self.pick = mistake_at, pick
        self._replies = 0
        self._erred = False
        self.x = frozenset(xs)
        self.o = frozenset(os_)
        self.placed: Dict[str, int] = dict(zip(XS, xs))
        #: pieces seen standing on their cell: only these can be moved off it
        self._seen: set = set()
        self._rising: Optional[Tuple[int, float]] = None
        self._to_answer = False
        self._over = False
        self._fouls = set()

    def foul(self, engine, world, what: str) -> None:
        if what not in self._fouls:
            self._fouls.add(what)
            self.note(engine, world, f"foul:{what}")

    def tick(self, engine: Engine, world: WorldState) -> None:
        if self._rising is not None:
            cell, t0 = self._rising
            if hand_on_board(world, self.board):
                self.foul(engine, world, "out_of_turn")
            frac = min(1.0, (world.sim_time - t0) / RISE_SECONDS)
            engine.set_articulation_target(LIFTS[cell], T.O_DOWN * (1.0 - frac))
            if frac >= 1.0:
                self._rising = None
            return
        standing = x_cells(world, self.board)
        for name, cell in self.placed.items():
            if standing.get(name) == cell:
                self._seen.add(name)
            elif name in self._seen:
                self.foul(engine, world, "moved_a_piece")
        for name in x_cells(world, self.board, stacked=True):
            if AtRest(name).holds(world):
                self.foul(engine, world, "taken_cell")
        for name, cell in standing.items():
            if name in self.placed or not AtRest(name).holds(world):
                continue
            self.placed[name] = cell
            if self._over:
                continue
            self.x = self.x | {cell}
            self.note(engine, world, f"x:{cell}", name)
            self._to_answer = True
            self._end(engine, world)
        if self._to_answer and not self._over and not hand_on_board(world, self.board):
            self._to_answer = False
            self._replies += 1
            cell = o_move(self.x, self.o, self._replies, None if self._erred else self.mistake_at,
                          self.pick)
            if self.mistake_at is not None and not self._erred and cell != best_o(self.x, self.o):
                self._erred = True
                self.note(engine, world, "o_erred")
            self.o = self.o | {cell}
            self._rising = (cell, world.sim_time)
            self.act(engine, world, f"o:{cell}", DriveJoint(LIFTS[cell], T.O_DOWN))
            self._end(engine, world)

    def _end(self, engine, world) -> None:
        w = winner(self.x, self.o)
        if w or not empty(self.x, self.o):
            self._over = True
            self._to_answer = False
            self.note(engine, world, f"won:{w or '-'}")
            # said out loud, as a person across the board would: on the easy
            # level the winning X is the last move, and with nothing said it
            # looked the same as an X that never sat on its post (v1.1)
            self.say(world, BOARD, "over", OVER[w])


#: what the opponent says when the game is over
OVER = {"X": "Three in a row -- X wins!", "O": "Three in a row -- O wins.",
        None: "The board is full -- a draw."}


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("WK6")
    scene, board, xs, os_ = build_scene(level, seed)
    _, _, mistake_at = start(level, seed)
    won = Custom(lambda w: events(w, "won:") == ["X"], "X has three in a row")
    fair = Custom(lambda w: not fouls_in_play(w), "no foul while the game was on")
    # the opponent answers once the hand has left the board, so an X set down
    # beside the posts could be played as a second move in the same turn
    # (v2.3); X moves first and O answers every move until the game is over
    turns = Custom(lambda w: len(events(w, "x:")) <= len(events(w, "o:")) + 1,
                   "one X move a turn")
    milestones = (
        Milestone("x_played", Custom(lambda w: len(events(w, "x:")) >= 1, "an X put down")),
        Milestone("o_answered", Custom(lambda w: len(events(w, "o:")) >= 1 or bool(events(w, "won:")),
                                       "the opponent has moved (or the game is over)")),
        Milestone("x_ahead", Custom(lambda w: _x_ahead(w, xs, os_),
                                    "after an X move: X one move from winning, O not")),
        Milestone("x_won", won),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="You are X. Win the game.",
        scorer=Scorer(task_id="WK6", level=level, success=Finally(won), milestones=milestones,
                      violations=(("fair_play", Never(Custom(lambda w: not fair.holds(w),
                                                             "a foul"))),
                                  ("one_move_a_turn", Never(Custom(lambda w: not turns.holds(w),
                                                                   "two X moves in one turn"))))),
        milestones=milestones,
        reactor_factory=lambda b=board, x=xs, o=os_, m=mistake_at, k=seed: Opponent(b, x, o, m, k),
        reference_factory=lambda inst, b=board: TicTacToePolicy(b),
        metadata={"knob": spec.knob.field, "knob_value": "on the board" if level == "easy" else "a mistake",
                  "position": {"x": list(xs), "o": list(os_)}, "mistake_at": mistake_at,
                  "puzzles": len(puzzles(level)), "max_model_calls": 40},
        extra_metrics={"x_moves": lambda traj: len(events(traj[-1], "x:")) if traj else 0},
    )


def _x_ahead(w: WorldState, start_x, start_o) -> bool:
    played = events(w, "x:")
    if not played:
        return False
    if events(w, "won:") == ["X"]:
        return True
    x = frozenset(int(c) for c in played) | frozenset(start_x)
    o = frozenset(int(c) for c in events(w, "o:")) | frozenset(start_o)
    return bool(threats(x, o)) and not threats(o, x)


# -- the reference --------------------------------------------------------------------
HAND = turned_down(T.X_YAW)
AWAY = (0.15, -0.19)                            # beside the spare Xs: off the board


class TicTacToePolicy(ScriptedPolicy):
    """Reads the board, waits its turn, and plays the perfect move -- or, as
    ``greedy`` (the stress agent), the move that makes most lines of its own."""

    name = "tic_tac_toe"

    def __init__(self, board, arm: str = RIGHT) -> None:
        self.board = board
        self.arm = arm
        self.greedy = False
        self.reset()

    def reset(self) -> None:
        self._place: Optional[PlacePosePolicy] = None
        self._cell: Optional[int] = None
        self._piece: Optional[str] = None
        self._played = 0

    def choose(self, x: Cells, o: Cells) -> int:
        return (greedy_x if self.greedy else best_x)(x, o)

    def _away(self) -> MoveTo:
        x, y = ahead(*AWAY)
        return MoveTo(self.arm, Pose((x, y, T.TTT_TOP + 0.15), HAND))

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        if events(world, "won:"):
            return [self._away(), Wait(1.0), Done("the game is over")]
        standing = x_cells(world, self.board)
        if self._place is not None:
            obj = world.obj(self._piece)
            if standing.get(self._piece) == self._cell and not obj.is_held:
                self._place, self._played = None, self._played + 1
                return [self._away(), Wait(0.5)]            # off the board: its turn
            cmds = self._place.commands(world, ctx)
            if self._place.attempts > MAX_GRASP_ATTEMPTS:
                return [Done(f"could not put {self._piece} down")]
            return [c for c in cmds if not isinstance(c, Done)] or [Wait(0.3)]
        # its turn is over when an O has come up for every X played
        answered = len(events(world, "o:"))
        if answered < self._played or any(
                T.O_DOWN + 0.001 < world.articulations.get(l, T.O_DOWN) <= UP for l in LIFTS):
            return [self._away(), Wait(0.5)]
        self._cell = self.choose(frozenset(standing.values()), o_cells(world))
        spare = [n for n in XS if n not in standing and not world.obj(n).is_held]
        if not spare:
            return [Done("no X left")]
        self._piece = min(spare, key=lambda n: world.obj(n).pose.pos[0])
        cx, cy = T.ttt_cell(self.board, self._cell)
        self._place = PlacePosePolicy(self._piece, (cx, cy, T.TTT_TOP), T.X_YAW, arm=self.arm,
                                      symmetry=math.pi / 2, release_clearance=0.003,
                                      tolerance=0.006, yaw_tolerance=math.radians(8))
        return [Wait(0.1)]
