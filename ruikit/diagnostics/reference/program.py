"""The reference as programs: the pose program, and a program over any other abstraction."""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

from ruibench.core.types import Pose
from ruibench.interfaces.commands import (
    Cancel,
    MoveDelta,
    MoveTo,
    SetGripper,
)
from ruikit.diagnostics.reference.base import Expressor, _arm_name, _grip_args
from ruikit.diagnostics.reference.grounding import _targets, _with_notes
from ruikit.env.episode import Episode, Outcome


class CodeExpressor(Expressor):
    """Each turn's commands as one program -- written the way a person would:
    one call a line, blocking, and the arms' moves of one round started
    together and waited for together on a two-arm station.

    A move that stops short of where it was sent ("blocked": a wrist turning
    a bulb in its socket ends every stroke a few degrees short) raises in a
    program, and the program ends there -- where the other expressors, a
    call a move, simply send the next one.  So a program's blocking moves go
    through ``go``, which takes that for an answer and goes on, as the
    reference does (CR2 hard through the program interface, on Isaac: every
    stroke ended after its first 80 degrees and the lamp never lit)."""

    style = "code"
    GO = ("def go(**move):\n"
          "    try:\n"
          "        return robot.move_to(**move)\n"
          "    except ToolCallError as short:        # stopped short: go on from there\n"
          "        return short.result")

    def wait(self, ep, seconds):
        ep.call("code_run", {"source": f"robot.wait(seconds={seconds!r})"})

    @staticmethod
    def _call(ep, c, two_arms: bool, wait: bool) -> str:
        arm = f"arm={_arm_name(c.arm)!r}" if two_arms and getattr(c, "arm", None) else ""
        tail = "" if wait else "wait=False"
        if isinstance(c, Cancel):
            return f"robot.stop({arm})"
        if isinstance(c, SetGripper):
            kv = [f"{k}={v!r}" for k, v in _grip_args(ep, c).items() if k != "arm"]
            return f"robot.gripper({', '.join(x for x in kv + [arm, tail] if x)})"
        d = ep.station.describe_pose(CodeExpressor.target_of(ep, c))
        rpy = [x + 0.0 for x in d["rpy_deg"]]            # no "-0.0" in the program
        turn = f"rpy_deg={rpy!r}" if any(abs(x) > 0.05 for x in rpy) else ""
        xyz = f"xyz_m={d['xyz_m']!r}"
        # a blocking move through ``go``; one only started answers at once
        call = "go" if wait else "robot.move_to"
        return f"{call}({', '.join(x for x in (xyz, turn, arm, tail) if x)})"

    def express(self, ep, acts):
        two_arms = ep.station.two_arms
        lines = []
        for row in self.rounds(acts):
            moving = [c for c in row if not isinstance(c, Cancel)]
            if len(moving) <= 1:
                lines += [self._call(ep, c, two_arms, wait=True) for c in row]
                continue
            # both arms at once: start each, then wait for both
            lines += [self._call(ep, c, two_arms, wait=True) for c in row if isinstance(c, Cancel)]
            names = []
            for k, c in enumerate(moving):
                name = f"h_{_arm_name(c.arm)}" if getattr(c, "arm", None) else f"h{k}"
                names.append(name)
                lines.append(f"{name} = {self._call(ep, c, two_arms, wait=False)}['data']['handle']")
            lines.append(f"robot.wait(handles=[{', '.join(names)}], mode='all')")
        if any(line.startswith("go(") for line in lines):
            lines.insert(0, self.GO)
        ep.call("code_run", {"source": "\n".join(lines)})


class _ProgramEpisode:
    """The episode as a program over another action abstraction sees it: a call
    the program interface cannot make by itself is made from inside a program --
    one call, one program; a batch of calls, one program making them in order,
    the arms' moves started together and waited for together as the pose
    program does -- and what the program's functions return is handed back as
    the tools' results."""

    def __init__(self, ep: Episode) -> None:
        self._ep = ep
        self._top = set(ep.profile.tools)

    def __getattr__(self, name):
        return getattr(self._ep, name)

    def call(self, name, args=None):
        if name in self._top:
            return self._ep.call(name, args)
        return self._program([[(name, args or {})]])[0]

    def call_batch(self, calls):
        calls = list(calls)
        if all(name in self._top for name, _ in calls):
            return self._ep.call_batch(calls)
        return self._program([calls])

    def call_rounds(self, rounds):
        """Rounds of calls, one program: a round's moves on several arms start
        together and are waited for together; one round after the other; an
        argument given as ``_Out`` is what an earlier call returned."""
        return self._program([list(r) for r in rounds])

    def _program(self, rounds):
        """One program for ``rounds`` of calls; an argument given as ``_Out``
        is what an earlier call of the same program returned."""
        from ruikit.core.result import ToolResult
        from ruikit.core.tool import all_tools

        tools = all_tools()

        def moves_arm(name, args) -> bool:
            spec = tools[name].spec if name in tools else None
            return (spec is not None and spec.controls == "arm" and "wait" in
                    {f.name for f in dataclasses.fields(spec.params)})

        lines, n = ["out = []"], 0
        for calls in rounds:
            started = [k for k, (name, args) in enumerate(calls) if moves_arm(name, args)]
            together = len({(calls[k][1] or {}).get("arm") for k in started}) > 1
            handles = []
            for k, (name, args) in enumerate(calls):
                family, _, rest = name.partition("_")
                args = dict(args or {})
                if together and k in started:
                    args["wait"] = False
                given = "{" + ", ".join(f"{key!r}: {_source(v)}" for key, v in args.items()) + "}"
                lines += ["try:", f"    out.append({family}.{rest}(**{given}))",
                          "except ToolCallError as e:", "    out.append(e.result)",
                          "except (KeyError, TypeError) as e:     # what it was to take failed",
                          "    out.append({'error': 'tool_failed', 'text': repr(e)})"]
                if together and k in started:
                    lines.append(f"h{n} = (out[-1].get('data') or {{}}).get('handle')")
                    handles.append(f"h{n}")
                    n += 1
            if handles:
                lines.append(f"robot.wait(handles=[h for h in [{', '.join(handles)}] if h], "
                             "mode='all')")
        lines.append("result = out")
        r = self._ep.call("code_run", {"source": "\n".join(lines)})
        got = (r.data or {}).get("result") if not r.is_error else None
        count = sum(len(c) for c in rounds)
        if not isinstance(got, list) or len(got) != count:
            return [r] * count                      # the program itself failed
        return [ToolResult(data=dict(g.get("data") or {}), text=str(g.get("text") or ""),
                           error=g.get("error") or None, hint=str(g.get("hint") or ""))
                for g in got]


@dataclasses.dataclass(frozen=True)
class _Out:
    """In a program: field ``key`` of what its ``index``-th call returned."""
    index: int
    key: str


def _source(value) -> str:
    if isinstance(value, _Out):
        return f"out[{value.index}]['data'][{value.key!r}]"
    return repr(value)


class ProgramOver:
    """A program over another action abstraction: that abstraction's reference,
    every move made from inside a program (``_ProgramEpisode``) -- a diagnostic
    of whether a program over those moves can do the task."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.style = "code-" + inner.style

    def __getattr__(self, name):                   # unexpressed, notes: the inner's
        return getattr(self.inner, name)

    def run(self, ep: Episode) -> Outcome:
        return self.inner.run(_ProgramEpisode(ep))


class GroundedCode(_with_notes(CodeExpressor)):
    """Two programs a turn when something must be found: the first looks,
    finds and estimates into ``seen`` -- the dependent calls chained in one
    reply -- and the second moves by those variables.  A program cannot see
    before it asks, so when the pictures show things a pose is by that it did
    not ask for, and none it asked for was found, a third asks for those."""

    via_code = True

    def on_commands(self, ep, cmds):
        targets = _targets(self, ep, cmds)
        if not targets:
            return
        asked = self._estimate(ep, targets)
        # as the tool-calling agents look up an anchor they did not plan for: when
        # the pictures show things a pose is by but none of them was asked for
        # (CL4: the two nearest, the well's water and the pebble already in it,
        # cannot be seen; the well can), one more program asks for those
        g = self.grounder
        missing: Dict[str, None] = {}
        for t in targets:
            ranked = g.anchors(t)
            if ranked and not any(g.seen.get(b) for b in ranked):
                for b in ranked[:2]:
                    if ep.perception.label(b)["label"] not in asked:
                        missing[b] = None
        if missing:
            self._estimate(ep, targets, bodies=list(missing))

    def _estimate(self, ep, targets: List[Pose], bodies: Optional[List[str]] = None) -> List[str]:
        """One program: look with every camera, then find each wanted kind of
        thing in every picture and estimate every match's pose into ``seen``.
        What to look for comes from the reference's intent (the things its
        poses are relative to); where they are, only from the tools."""
        g = self.grounder
        labels = list(dict.fromkeys(ep.perception.label(b)["label"]
                                    for b in (bodies if bodies is not None else g.wanted(targets))))
        if not labels:
            return []
        lines = ["frames = {f['camera']: f['frame_id'] for f in camera.look()['data']['frames']}",
                 "try:", "    seen", "except NameError:", "    seen = {}",
                 "def at(o, dx, dy, dz):",
                 "    x, y, z = o['xyz_m']",
                 "    return [round(x + dx, 4), round(y + dy, 4), round(z + dz, 4)]",
                 "found = {}",
                 "for fid in frames.values():",
                 f"    for q in {labels!r}:",
                 "        for o in vision.detect(frame_id=fid, query=q)['data']['objects']:",
                 "            p = vision.pose(frame_id=fid, id=o['id'])['data']",
                 "            # nothing lies below the table: its middle is at least half its thickness up",
                 "            p['xyz_m'][2] = max(p['xyz_m'][2], p['extent_m'][2] / 2)",
                 "            found[o['id']] = p",
                 "seen.update(found)",
                 "result = {'frames': frames,",
                 "          'found': {k: [v['label'], v['xyz_m'], v['yaw_deg']] for k, v in found.items()}}"]
        r = ep.call("code_run", {"source": "\n".join(lines)})
        res = r.data.get("result") if not r.is_error else None
        if not isinstance(res, dict):
            return labels
        g.frames = [{"frame_id": fid, "camera": cam} for cam, fid in (res.get("frames") or {}).items()]
        # each thing from the fixed camera's picture if it is there, else a wrist camera's
        order = {f.id: k for k, f in enumerate(g._frames())}
        for oid, (label, xyz, yaw) in sorted((res.get("found") or {}).items(),
                                             key=lambda kv: order.get(kv[0].split("#")[0], 99)):
            try:
                fr = ep.frames.get(oid.split("#")[0])
                name = ep.perception.resolve(fr, oid)
            except Exception:
                continue
            if name not in g.seen:
                g.seen[name] = g._seen(name, label, fr, oid, {"xyz_m": xyz, "yaw_deg": yaw})
                g.memory[name] = (g.seen[name], ep.sim_time)
        return labels

    def express(self, ep, acts):
        super().express(ep, acts)

    @staticmethod
    def _var(oid: str) -> str:
        return "seen[" + repr(oid) + "]"

    def _call(self, ep, c, two_arms: bool, wait: bool) -> str:
        if not isinstance(c, (MoveTo, MoveDelta)):
            return super()._call(ep, c, two_arms, wait)
        got = self.grounder.ground(self.target_of(ep, c))
        if got is None:
            self.unexpressed.append("nothing estimated to move by")
            return "pass  # 没有任何估计可以依据：这一步不做"
        arm = f"arm={_arm_name(c.arm)!r}" if two_arms and getattr(c, "arm", None) else ""
        tail = "" if wait else "wait=False"
        d = ep.station.describe_pose(got.pose)
        rpy = [x + 0.0 for x in d["rpy_deg"]]
        if got.seen is not None:
            o = self._var(got.seen.obj_id)
            xyz = f"xyz_m=at({o}, {', '.join(repr(x) for x in got.offset)})"
            yaw = f"{o}['yaw_deg'] + {got.turn_deg!r}" if abs(got.turn_deg) > 0.05 else f"{o}['yaw_deg']"
            turn = (f"rpy_deg=[{rpy[0]!r}, {rpy[1]!r}, {yaw}]"
                    if any(abs(x) > 0.05 for x in rpy) or abs(got.turn_deg) > 0.05 else "")
        else:
            xyz = f"xyz_m={d['xyz_m']!r}"
            turn = f"rpy_deg={rpy!r}" if any(abs(x) > 0.05 for x in rpy) else ""
        return f"robot.move_to({', '.join(x for x in (xyz, turn, arm, tail) if x)})"
