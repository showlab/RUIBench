"""Reference expressors: the task's scripted reference, acted through an interface's tools.

"Can this interface's tools do this task at all?" is a different question from
"can a model do it".  Here the *decisions* come from the task's own scripted
reference, which reads the simulator (and, given direct control, solves the
task); the *actions* must go through the interface's tools and nothing else.
If the reference cannot be expressed, the interface is missing something the
task needs -- a coverage gap to fix by adding a tool, or to report.

One expressor per action abstraction, and one for the program over the pose:

* ``pose`` (``rui.pose/v1``) -- each commanded pose becomes ``robot_move_to``;
* ``step`` (``rui.step/v1``) -- each commanded pose becomes a walk of
  ``robot_step`` calls, closed-loop on the reported gripper pose;
* ``point`` (``rui.point/v1``) -- each commanded pose becomes a pixel of a
  fresh frame whose lifted point lies right under the target, plus a height
  and orientation for ``robot_move_to_pixel``;
* ``code`` (``rui.code/v1``) -- each turn's commands become one program run by
  ``code_run`` in the sandbox;
* ``delta`` (``rui.delta/v1``) -- each commanded pose becomes a
  ``robot_move_delta`` from where the hand reads it is;
* ``skill`` (``rui.skill/v1``) -- the reference's grasps, releases and
  presses become ``skill_pick`` / ``skill_place`` / ``skill_press``; the moves
  between them are the skills' own.  What is not one of these (a push, a turn,
  a held thing carried along a path that matters) has no skill, and the cell
  shows it;
* ``joint`` (``rui.joint/v1``) -- each commanded pose becomes
  ``robot_solve_ik`` and ``robot_move_joints`` calls along the straight line.

A program over another abstraction (``ProgramOver``) makes that one's calls
from inside programs.

The expressors are diagnostic tools: they read the simulator through
``Episode.oracle()`` to know *what* to do.  They never hand that to a tool.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ruibench.core.types import Observation, Pose
from ruibench.interfaces.base import ModelTurn, TurnContext
from ruibench.interfaces.commands import (
    Cancel,
    Done,
    MoveDelta,
    MoveTo,
    SetGripper,
    Wait,
)
from ruibench.interfaces.direct import DirectInterface
from ruikit.env.episode import Episode, Outcome


def _arm_name(arm_id: str) -> str:
    return arm_id.split("/")[-1]


class Expressor:
    style = ""

    def __init__(self, max_turns: Optional[int] = None, honor_waits: bool = True) -> None:
        self.max_turns = max_turns          # default: the task's own turn budget
        # True (as the benchmark's executor reads them): a Wait between two
        # commands is a pause right there ("settle, then open"); one naming an
        # arm is taken as a pause for both (a tool reply cannot pause one arm).
        # False: the old reading, the turn's waits run after its moves
        self.honor_waits = honor_waits
        self.unexpressed: List[str] = []

    # -- the loop --------------------------------------------------------------
    def run(self, ep: Episode) -> Outcome:
        oracle = ep.oracle()
        inst = oracle.instance
        policy = inst.reference_factory(inst)
        policy.reset()
        codec = DirectInterface()
        limit = self.max_turns or ep.budget.max_model_turns or 400
        for turn in range(limit + 1):       # one past: the episode itself calls the budget
            if ep.over:
                break
            ep.model_turn()
            self.on_turn(ep)
            world = oracle.world()
            said = oracle.said()
            obs = Observation(sim_time=world.sim_time, frames={},
                              proprioception={a: world.arm(a).public() for a in world.arm_ids},
                              instruction=world.instruction,
                              instruction_version=world.instruction_version,
                              feedback=tuple(m.as_text() for m in said))
            ctx = TurnContext(
                observation=obs, task_id=inst.task_id, instruction=world.instruction,
                hint=inst.hint, arm_ids=ep.scene.arm_ids, turn_index=turn,
                sim_time=world.sim_time, budget_left={}, history=[], services=None,
                extras={"world": world, "level": inst.level,
                        "knob": inst.metadata.get("knob_value"),
                        "executor_busy": oracle.executor_busy(), "residual_motion": {},
                        "feedback": tuple(said), "demonstrations": oracle.watched})
            raw = policy.act(ctx, ModelTurn(turn_index=turn, prompt=""))
            parsed = codec.parse(raw, ctx)
            cmds = parsed.commands if parsed.ok else []
            done = any(isinstance(c, Done) for c in cmds)
            acts = [c for c in cmds if isinstance(c, (MoveTo, MoveDelta, SetGripper, Cancel))]
            if acts:
                self.on_commands(ep, cmds)
            wait = max((c.seconds for c in cmds if isinstance(c, Wait)), default=0.0)
            t0 = ep.sim_time
            if self.honor_waits:
                self._in_order(ep, cmds)
            elif acts:
                self.express(ep, acts)
            if ep.over:
                break
            left = 0.0 if self.honor_waits else wait - (ep.sim_time - t0)
            if left > 1e-6:
                self.wait(ep, round(min(left, 60.0), 4))
            elif not acts and not done and not (self.honor_waits and wait > 0):
                self.wait(ep, 0.5)
            if done and not ep.over:
                ep.call("task_done", {"report": "reference finished"})
        if not ep.over:
            ep.finish("budget_turns")
        return ep.outcome

    def _in_order(self, ep: Episode, cmds: Sequence) -> None:
        """The moves between two waits together, each wait where it stands."""
        seg: List = []
        for c in cmds:
            if isinstance(c, Wait):
                if seg:
                    self.express(ep, seg)
                    seg = []
                if ep.over:
                    return
                self.wait(ep, round(min(c.seconds, 60.0), 4))
            elif isinstance(c, (MoveTo, MoveDelta, SetGripper, Cancel)):
                seg.append(c)
        if seg and not ep.over:
            self.express(ep, seg)

    # -- helpers ------------------------------------------------------------------
    @staticmethod
    def rounds(acts: Sequence) -> List[List]:
        """Commands per arm, interleaved: round k holds each arm's k-th command,
        so arms move together and each arm's commands stay in order."""
        per: Dict[str, List] = {}
        for c in acts:
            per.setdefault(getattr(c, "arm", None) or "*", []).append(c)
        out, k = [], 0
        while True:
            row = [cs[k] for cs in per.values() if k < len(cs)]
            if not row:
                return out
            out.append(row)
            k += 1

    @staticmethod
    def target_of(ep: Episode, c) -> Pose:
        if isinstance(c, MoveDelta):
            now = ep.engine.state().arm(c.arm).eef_pose
            return now.translated(c.dpos)
        return c.pose

    def express(self, ep: Episode, acts: Sequence) -> None:
        raise NotImplementedError

    def on_turn(self, ep: Episode) -> None:
        """A new model turn begins (the grounded expressors look afresh)."""

    def on_commands(self, ep: Episode, cmds: Sequence) -> None:
        """The turn's whole command list, before any of it is sent (the grounded
        expressors estimate everything it needs at once)."""

    def wait(self, ep: Episode, seconds: float) -> None:
        ep.call("robot_wait", {"seconds": seconds})


def _grip_args(ep, c: SetGripper) -> dict:
    arm = _arm_name(c.arm)
    if c.value >= 0.5:
        return {"arm": arm, "command": "close"}
    if c.value <= 0.0:
        return {"arm": arm, "command": "open"}
    return {"arm": arm, "width_m": round((1.0 - c.value) * ep.station.arm(arm).max_opening, 4)}
