# The contract under every interface

The benchmark's interfaces are RUIKit's 13 RUIs: six action abstractions
(Joint, Delta, Step, Pose, Point, Skill) in two interaction forms (per-action
and program), and RUI-Free ([docs/ruikit/README.md](ruikit/README.md),
[docs/ruikit/interfaces.md](ruikit/interfaces.md)). A new interface is added
there ([docs/ruikit/extending.md](ruikit/extending.md)). This document is the
layer under them, which RUIKit's tools and the scripted references share: the
command vocabulary the executor runs, its rules, the interface contract and
what a turn carries.

## The executor-level interface

What the episode runner drives is a `ruibench.interfaces.base.Interface`:
three methods and a contract.

```python
from ruibench.interfaces.base import Interface, InterfaceContract, Expressivity

class MyInterface(Interface):
    id, name, source = "my_interface", "pose", "where its action language comes from"

    @property
    def contract(self) -> InterfaceContract: ...

    def system_prompt(self, ctx) -> str: ...         # the action language, in words
    def observe(self, ctx) -> ModelTurn: ...         # which part of the observation
    def parse(self, raw, ctx) -> ParsedAction: ...   # model text -> list[Command]
```

The scripted reference, the negative controls and the stress agents run
through `I0_direct` (`ruibench/interfaces/direct.py`), which takes the command
vocabulary verbatim and never appears on a leaderboard. The contract stubs
beside it (`semantic_step.py`, `pose.py`, `pointing.py`, `program.py`) predate
RUIKit; they are not the interfaces the benchmark reports.

## Two rules

**1. Never reach for the engine.** Emit commands; the executor runs them. The
vocabulary is in `ruibench/interfaces/commands.py`:

```
MoveTo(arm, pose)              MoveDelta(arm, dpos, drot, frame)
MoveJoints(arm, joints)        SetGripper(arm, value)
Wait(seconds, arm=None)        Cancel(arm, handle_id)
Done(note)
```

`SetGripper`: 0 is open and 1 closed. From 0.5 up the fingers close on whatever
is between them; below 0.5 they open to (1 - value) of the way, e.g. to take one
piece among others standing close. Both engines agree on this.

`MoveDelta` moves and turns the hand from where it is when the move starts:
`dpos` in metres and `drot` as a rotation vector in radians, in the `frame`'s
axes -- `world`, `eef` (the hand's own; its z points along the fingers) or
`camera` (the run's first camera, optical: x is the picture's right, y down,
z away from the lens). `MoveTo.speed` must be 1.0: every move runs at the
shared controller's speed. The executor **rejects** what it cannot do (the
handle comes back `rejected`, with the reason in its `detail`) -- it never
drops part of a command silently.

`Wait` is a pause at its place among the commands. What comes after it starts
once what came before it is done and `seconds` more have passed, so
`[MoveTo(down), Wait(0.3), SetGripper(0.0)]` lets the hand settle before it
opens. `Wait(seconds)` holds every arm: each finishes what came before, then
all pause together. `Wait(seconds, arm)` pauses only that arm, so two hands'
work can go in one batch without one waiting for the other. On its own, a
`Wait` lets the world run that long.

Work decided before a `Cancel` must not run after it. An interface that
submits late -- a chunk of actions that arrives after the model changed its
mind, a program's sub-call -- passes the arm's generation it started under
(`Executor.generation(arm)`, then `submit(cmd, generation=g)`); after a cancel
the submission is rejected with a detail starting "stale generation".

If you need something the vocabulary cannot express, that is a finding, not a
reason to widen one interface. Raise it before adding to `commands.py` — a
command that only one interface can use quietly becomes the thing being measured.

**2. "Cannot express" is not "failed."** If a task needs something your action
language genuinely cannot say, return a reason from `coverage_gap(task)`. The
episode is then recorded as a coverage gap and excluded from success rates
entirely. The clearest case: a pixel plus an action type cannot carry a 6-DoF
orientation, so a task that needs one -- DS1's hard level, where a present is
turned over to show its underside to the fixed camera; CL5's tangram pieces,
each set down at its own turn -- is at risk for an interface whose moves name
only a point. Decide it deliberately, per suite, and put the decision in the
contract. RUIKit's interfaces are held to more: they may differ in how well
they do a task, never in whether they can (`ruikit coverage`,
[docs/ruikit/experiments.md](ruikit/experiments.md#tool-solvability)).

## The contract

`ruibench.interfaces.contract_table()` builds the table. The fields are the
axes along which interfaces are allowed to differ:

| Field | Meaning |
|---|---|
| `position` / `orientation` / `gripper` | how each can be named: symbolic, relative, absolute, pixel, or not at all |
| `blocking` | does one model turn wait for the motion to finish? |
| `cancellable` | can the model take back an action already in flight? |
| `concurrent_arms` | can it drive two arms at once? |
| `camera_control` | `none` / `via_arm` / `explicit` |
| `services` | which perception tools it may call — **must be the same for every interface** |
| `feedback_delivery` | `none` / `between_turns` / `inline` — the program interface must also receive it |
| `reentry` | `per_step` / `per_target` / `per_program` |

If your implementation's behaviour and its contract disagree, the contract is
what gets reported — so fix whichever is wrong, and let the test tell you.

## Conformance

```bash
pytest tests/ruibench/test_interface_contract.py tests/ruikit/core/test_contract.py
```

These are acceptance criteria, not incidental unit tests. In particular:

- every interface offers the same perception tools;
- **a program and the per-action interface over the same abstraction expose an
  identical call surface** (pose and program-pose, and so for each
  abstraction). They are the cleanest contrast in the benchmark — same callable
  API, the only difference being one action a reply versus a written program.
  Anything you add to one, add to the other;
- the program interface can read the one-bit correct/wrong feedback signal.

## What a turn carries besides the picture

`TurnContext` carries the task's `instruction` (which may change during an
episode: DI4's calls, and any task whose instruction is revised) and its `hint`
-- how the things on the bench work, what a person teleoperating would also
be told; "" for most tasks.  An interface must present both.

`TurnContext.extras` holds what some tasks add to an observation. An interface
decides how to present it; it may not leave it out. An interface under test
finds exactly these keys (`runner.episode.PUBLIC_EXTRAS`), and a test holds it
to that:

| Key | When | What |
|---|---|---|
| `demonstrations` | CL1, CL2, CL4, CL5 | films played before the episode (`runner.demonstration.Film`): a `caption` and `frames` -- `(sim time, {camera: frame})`, one a simulated second, at the episode's cameras and resolution. **Frames only, never code, never the world states behind them**: the program interface gets exactly what the others get |
| `feedback` | a task with a correct/wrong signal (none of the current 30) | the one-bit messages delivered so far |
| `executor_busy` | always | whether submitted work is still running |
| `residual_motion` | always | what submitted work has still to do (a program interface can query its own in-flight work; a step-by-step one sees it in the picture) |
| `refused` | always | what the executor would not run of the last turn's commands, and why -- `{"kind", "arm", "why"}` for each (an arm the scene has not, a speed or a frame the shared controller has not); told once, empty otherwise |

The scripted agents' interface (`I0_direct`) also gets
`world` (the simulator's ground truth), `level` and `knob`, and the
demonstrations with their world states. None of these reaches an interface
under test: each carries answers (a knob is a tolerance or a speed; CL1's tune
and which of CL2's dolls is Momo are in the demonstration's states).

`observation.proprioception` gives, per arm, an `ArmReading`: where the hand
is, how far the fingers are commanded closed (`gripper`), how far apart they
really are (`opening`, metres: on what they hold, its width), whether they
close on something (`grasping`: stopped on it, not closed on nothing) and
whether it is still moving -- never *what* it holds (a held
body's name can be the answer: DS1's "present2").

## Perception

The benchmark answers no perception queries itself. Perception is RUIKit's,
part of its shared tools, and it answers from pixels only: every
interface's `vision_*` tools read the picture and a depth image (or the
picture alone, `depth=mono`) with a model that finds and outlines what a word
names (SAM 3 by default), the camera's calibration off by a real robot's error
and the depth with a real camera's noise, and nothing from the simulator: no
position, not even with noise added, no name, no segmentation
(`ruikit/perception/backends/pixels.py`;
[docs/ruikit/perception.md](ruikit/perception.md)). The same perception for
every interface, and never more precise than a real robot's. Whether a hand
holds something is told by its fingers' readings. On the mock engine, which
renders no pictures, a test double stands in
(`ruikit/perception/backends/synthetic.py`); the mock produces no result.

Everything an interface returns to its model is scanned for the task's
secrets (`TaskInstance.secrets`: the canary and the names that carry the
answer, e.g. the fake coin's body name) by RUIKit's leak test.

Two-arm tasks (WK4, CL1, DI1, DI5, CR6) have one robot, `robot0`, with two arms,
`robot0/left` and `robot0/right`; the cameras are the head camera `agentview`
(640 x 480) and the wrist cameras `wrist_left` and `wrist_right`, so
`observation.frames` has three images.  Single-arm tasks drive
`robot0/right` (an FR3) and see `agentview` and `wrist`.

## Smoke-testing on the mock engine

```bash
ruibench run DI4 --seeds 0 --engine mock --clock paused     # the scripted reference, through I0_direct
ruikit coverage --task DI4 --interface pose program-pose    # the reference through RUIKit's interfaces
```

```python
from ruikit import Episode

ep = Episode("DI4", level="easy", seed=0, profile="rui.pose/v1", engine="mock")
...                                                          # ep.call(tool, args), one call at a time
```

The mock engine is a kinematic stand-in with no physics, which is exactly what
you want while debugging an interface: episodes take milliseconds and the
world is deterministic.

## Things that will bite

- **Frames.** If your language says "left", say in the prompt whether that is the
  camera frame, the world frame or the end-effector frame, and keep it identical
  across tasks. A frame flip silently changes what the suite measures.
- **Prompts are frozen.** Tune on the dev split, then freeze. The hash goes into
  every episode log; changing it after the test split has been touched
  invalidates those results.
- **The re-entry period is part of the interface, not of the model.** A program
  interface is allowed to execute one primitive and return — that is how it can
  imitate step-by-step control when the world is moving, and it is why "plans
  well" and "cannot change its mind" are not necessarily the same axis.
- **There is no delegation primitive.** No `watch(condition)`, no late-binding
  `move_to(object)`. That is a deliberate decision: giving every interface an
  escape hatch would turn every interface into the same high-level call and erase the
  research object. A consequence to accept: the hard Dynamic interaction levels
  may be passable only by some interfaces. That is a result.
- **Time is in T_ref.** The Dynamic interaction tasks state their timing in
  `RunConfig.t_ref_seconds`, a benchmark-level response time: 8 s, set from
  three frontier models' measured reply times (8 s is about the 75th
  percentile of 4,067 replies). Under `--clock 8s`, the scripted reference
  taking 8 s a turn solves every easy level and almost none of the hard ones
  (at most 2 seeds in 20).
