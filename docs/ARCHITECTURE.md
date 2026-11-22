# Architecture

Written for anyone extending RUIBench: a new interface, a new task, a new engine.

## The one idea

The 13 robot-use interfaces (RUIs) share **one robot, one controller, one
clock, one set of perception tools and one scorer**. The only difference
between conditions is the interface: the action abstraction the model speaks
in (Joint, Delta, Step, Pose, Point, Skill) and the interaction form it speaks
through (per-action or program; RUI-Free offers every abstraction in both).
Everything in this codebase exists to make that true in code rather than only
in the paper.

Concretely: every interface's tools compile down to the same small command
vocabulary, and only the executor touches the engine. The scripted reference
sends that vocabulary directly.

```
  model's tool call (or program)
      │  RUIKit interface           ← the only per-interface code: one abstraction's tools
      ▼
  Command          MoveTo · MoveJoints · MoveDelta · SetGripper · Wait · Cancel · Done
      │  Executor                   ← queues, handles, cancellation, concurrency
      ▼
  Engine           isaac (results) · mock (CI) · franka (the real robot)
      │
      ▼
  WorldState       privileged ground truth
      │  predicates · milestones · events · scorer     ← all engine-agnostic
      ▼
  EpisodeLog → Results → report
```

## The three boundaries that matter

### 1. The engine boundary

Task logic — predicates, milestones, event scripts, knobs, scorers — is written
against `ruibench.core.types` only. Nothing above `ruibench/engines/` knows what
a simulator is.

The design document budgeted two engineer-days for this boundary on the argument
that drawing it early is an order of magnitude cheaper than refactoring later.
That bet was settled within a day: the plan moved from MuJoCo/robosuite/LIBERO to
Isaac Sim, and no task card, predicate, scorer, event or report changed. What
changed was one directory.

A scene is described declaratively by `SceneSpec` — bodies, articulations,
markers, cameras, robots — so the same task card instantiates on any backend.
Task cards name *logical* assets (`bowl`, `basket`); `engines/isaac/assets.yaml`
maps them to USD. A card never contains a file path.

What the backends build from that description, and how:

- **Built bodies.** The toys are rigid bodies built from parts -- boxes,
  cylinders, spheres, capsules (`engines/parts.py`, `tasks/impl/_toys.py`): a
  burger slice, a car, a shape sorter's lid, a tic-tac-toe board with wells.
  Decoration parts do not collide; a `hidden` part collides unseen (a round
  handle's square grip: flat fingers on a round handle let it turn out of the
  hand).  The mock's supports and the sight-line checks use the same parts,
  by their true shadows on the table.  Bins, upside-down cups and lids are
  older built shapes, boxes glued into one body (`engines/isaac/spawners.py`).
- **Joints.** A body an `ArticulationSpec` carries becomes the moving link of a
  one-joint articulation: a base pinned to the world, and a prismatic joint --
  or a revolute one, turning about `pivot` -- with a PhysX drive:
  position-driven for DI2's train, WK6's rising O pieces, DI6's turntable and
  WK2's ferry, sprung for DI1's buttons and CL1's xylophone keys, a spring a
  reactor takes away for WK3's pendulum (`Engine.set_joint_drive`).  Never a teleported body: only a joint carries
  what rests on it the way physics would.  A carried body rides a few
  millimetres clear of the table and of anything fixed along its travel (both
  checked by `validate`): a button cap resting on its housing could not be
  pressed.  The mock has no contact: a closed hand pushes a `pressable` body's
  joint (a button, a key, a lever), and a lid nothing holds drops shut -- or,
  past upright, open.
- **Cloth.** One kind of body is not rigid (`engines/cloth.py`): its shape is
  `cloth`, and `attrs["cloth"]` holds the points it is simulated with, lying
  flat, and the triangles between them.  Its state is an `ObjectState` with
  `points` -- where a sparse set of those points, the marks, are now -- and a
  `pose` and `size` that are the box round them; whatever only asks where a
  body is and how big (the checks, sight lines, the perception tools' boxes)
  takes it as that box.  The grasp is a rule, not friction, and the same on
  both engines: fingers that close with their tips at the cloth take the
  patch under them, however many layers, and carry it until they open.  On
  Isaac the cloth is a PhysX particle cloth and the patch is attached to the
  hand (`engines/isaac/cloth.py`).  PhysX solves cloth on the graphics card
  only, so a scene with a cloth in it is simulated there whole -- every other
  scene stays on the CPU -- and may hold nothing that is moved by writing its
  pose: no fixed body, no joint (`cloth.scene_problems`, checked by
  `validate`).  Two runs of one mission then agree to a hundredth of a
  millimetre, not to the bit.  The mock has no cloth physics: it folds the
  cloth flat when the last hand lets go, along the perpendicular bisector of
  the way each patch was carried.
- **Reactors.** Toys that answer (`events/reactor.py`): ticked with every
  physics slice, a reactor drives joints, recolours lamps, records events and
  says one-bit messages on the feedback path -- DI1's rocket, CL1's
  xylophone, WK6's opponent, DI3's traffic light, DI4's Simon, DI5's moles.
  One may run a finale after the agent says it is done (WK3's domino run).  What the mock cannot simulate -- a chain
  of dominoes -- the task's reactor plays out there
  (`EngineCapabilities.dynamics`).
- **Marks.** Crosses, squares, lines and lettering are flat strokes
  (`engines.base.marker_strokes`), drawn by the renderer, counted by the
  segmentation check, and drawn the same way in the viewer's plan.
- **Sight lines.** `engines/sight.py` answers "can this camera see that" by
  geometry: the mock's visibility when a task tracks it (other bodies, the
  hands, the camera's frame), and the design checks in `tasks/sightlines.py`.
- **Demonstrations.** A task may list films to play first
  (`TaskInstance.demonstrations`): the runner runs a scripted demonstrator in
  the demonstration's scene -- same structure, so the same stage -- films it
  from the episode's cameras, and hands the frames to the interface.

### 2. The interface boundary

The interfaces are RUIKit's (`ruikit/`, [docs/ruikit/](ruikit/README.md)). A
tool reaches the world only through its `ctx` and the shared executor; it never
reaches for the engine. Underneath, `ruibench/interfaces/` holds the command
vocabulary and the executor-level contract that the scripted reference runs
through (`I0_direct`). See [docs/INTERFACES.md](INTERFACES.md).

The contract is data, not prose: `ruikit check` and the tests
(`tests/ruikit/core/test_contract.py`, `tests/ruikit/interfaces/test_design_space.py`)
check the claims it makes -- that every interface offers the same 13 shared
tools, and that a program over an abstraction calls exactly what the
per-action interface over it offers (pose and program-pose, and so for each
abstraction), which is the cleanest contrast in the design.

### 3. The scoring boundary

The scorer is a **pure function of a trajectory**. No clock, no engine, no
randomness. A stored log (`ruibench-log/2`) keeps the trajectory whole, every
0.1 s: each body's pose, velocity, holder, changed colour and the cameras that
see it; each hand's pose, grip and load; the joints; the instruction; and the
events fired since the snapshot before, repeats included (`runner/snapshot.py`).
What never changes (sizes, categories, attributes) comes back from the scene,
which the task, level and seed rebuild.  So `ruibench rescore` scores an
episode again from its log alone, in another process than the one that ran
the agent, and reports where the verdict differs.  Logs from before v15 kept
positions only; for those it checks that the stored score is consistent.

Success criteria are trajectory predicates, not terminal-state predicates,
because "never did the thing it was not supposed to do" cannot be read off a
final state — and because constraint violations belong *inside* the success
criterion rather than in a second metric.

## Design rules that are enforced, not remembered

Everything below fails a test or `ruibench validate` if broken.

| Rule | Where it is enforced |
|---|---|
| every suite: 6 on the board + 1 held out, one development task; at least 4 two-arm tasks on the board | `tasks/validate.py` |
| one knob per task, exactly two levels | `tasks/loader.py`, `tasks/validate.py` |
| every card lists what else changes with the knob | `tasks/validate.py` |
| held-out seventh tasks stay unimplemented | `tasks/validate.py`, `tasks/instance.py` |
| main-board events cannot fire on a model-call count | `events/scheduler.py` |
| coverage gaps are never scored as zeros | `scoring/scorer.py`, `reports/metrics.py` |
| non-photorealistic engines cannot produce a leaderboard | `reports/metrics.py` |
| a program over an abstraction calls exactly what the per-action interface over it offers | `ruikit/interfaces/program.py`, `tests/ruikit/core/test_contract.py` |
| cancelled work never resumes | `executor/executor.py`, `test_harness.py` |
| every loose object can be grasped straight down (against the robot's measured hand) and is within its reach | `tasks/clearance.py`, `tasks/validate.py` |
| a body on a joint nothing drives rides clear of the table | `tasks/clearance.py`, `tasks/validate.py` |
| scorer, placement feedback and scripted references share one "in the container" | `scoring/predicates.py` (`inside`), `test_tasks.py` |
| the external camera and the arm's start pose are the same on every backend | `engines/base.py`, `test_isaac_backend.py` |
| the log records what the engine ran with; a rate it cannot honour is an error | `runner/harness.py`, `test_harness.py` |
| the fixed camera sees every object at the start -- past other bodies and the arms where they start -- unless a task hides it on purpose, and while the hand is over it | `tasks/sightlines.py`, `tasks/validate.py` |
| a prepared asset whose scan is missing fails loudly, never spawns an empty body | `engines/isaac/assets.py`, `engines/isaac/spawners.py` |
| every mark a task names is in the fixed camera's picture | `test_isaac_backend.py` (projection), `ruibench gallery` (pixels) |
| an implemented task names the agent whose failure the showcase shows | `tasks/validate.py` (`failure_example`) |
| a task's seeds share one scene structure | `test_isolated.py` |
| an interface under test gets the public extras only: no ground truth, level, knob or demonstration states; arm readings never name what the hand holds | `runner/episode.py` (`PUBLIC_EXTRAS`), `test_hardening.py` |
| perception reports only what a frame shows, by per-frame numbers, read off the pictures and the depth (v1.13), never from the simulator's positions and never more exactly than a real robot's; on the mock a test double stands in | `ruikit/perception/backends/pixels.py`, `ruikit/env/episode.py`, `tests/ruikit/perception/test_pixels.py`, `test_no_exact_positions.py` |
| work decided before a cancel is rejected after it; a command the controller cannot carry out is rejected, never partly dropped | `executor/executor.py`, `test_hardening.py` |
| a `Wait` is a pause where it stands in the batch: nothing after it starts until what came before is done and its time has passed | `executor/executor.py`, `test_hardening.py` |
| a hand holds something only while its fingers are stopped on it; closed on nothing, or pushed onto a body already closed, it holds nothing | `engines/isaac/engine.py`, `engines/mock.py`, `test_issue_fixes.py` |
| every body is found in Isaac's segmentation, whatever the case of its name | `engines/isaac/engine.py` (`_segmentation_names`), `test_issue_fixes.py` |
| a trajectory physics would not produce (a teleport, a throw, a joint past its stop, NaN) fails until reviewed | `scoring/audit.py`, `scoring/scorer.py`, `test_hardening.py` |
| an episode can be scored again from its log alone, in another process | `runner/snapshot.py`, `ruibench rescore`, `test_hardening.py` |
| every task's secrets are specific, and the canary is in every card and log | `tasks/instance.py` (`secrets`), `core/canary.py`, `test_hardening.py` |

## The three-layer solvability check

A task is not done until all three pass, and they are deliberately separate:

1. **physically solvable** — a privileged scripted reference reading ground truth
   finishes it. Proves the scene and the success criterion are valid.
   It runs through `I0_direct`, the executor's interface for scripted agents,
   which is not one of the benchmark's interfaces.
2. **expressible** — the same reference finishes it *through the interface under
   test*, within that interface's declared information and action permissions
   (`ruikit coverage`). Failing here is a **coverage gap**, reported separately.
3. **model-solvable** — a real model finishes it through the same interface.
   Failures split into decision error / not expressible / execution failure /
   timeout.

"The scripted policy can pass" proves nothing about layer 2: the script has
privileged state.

## The clock

The world advances while the model thinks. How much is the clock's only job:
`paused` (thinking costs nothing; diagnostic), `8s` (a fixed time a reply,
identical for every model in a run: the main setting; any `<N>s`), `measured`
(the real API latency). Nothing else in the harness
reads a wall clock, so an episode replays identically on a fast or slow machine.

Within one run every model faces the same clock. Tuning the world to a model's
measured latency would change the task instead of measuring the model.

## Adding things

**A task.** Write the card in `ruibench/tasks/cards/<suite>.yaml` (knob, success,
milestones, confounds, matched control, risks), then a module in
`ruibench/tasks/impl/` exposing `build(level, seed, config) -> TaskInstance`.
`di4_simon_says.py` is a worked example. Point `impl:` at it and set
`status: implemented`.

**An interface.** See [docs/ruikit/extending.md](ruikit/extending.md): an
action abstraction, an interaction form, or a YAML file of tools.
[docs/INTERFACES.md](INTERFACES.md) has the contract underneath.

**An engine.** Subclass `engines.base.Engine`, fill in `EngineCapabilities`,
register it. If it is not photorealistic, leave `results_grade=False` and the
reporting layer will keep it off the leaderboard.

## Running the Isaac backend

Two constraints Isaac imposes, both visible in the code:

* **`SimulationApp` must start before any `omni`, `pxr` or `isaaclab` import**,
  and only once per process. Every such import in RUIBench is therefore inside
  a function, and `ruibench validate` / `tasks` / `report` keep working on a
  machine with no GPU.
* **The IK controller takes commands in the robot's base frame**, while
  RUIBench speaks world poses everywhere else. The conversion lives in
  `IsaacEngine._apply_targets` and nowhere else.

Practical notes:

```bash
export OMNI_KIT_ACCEPT_EULA=YES     # your decision, not the harness's
export RUIBENCH_ISAAC_GPU=0         # pin one GPU; Kit otherwise opens all of them
ruibench setup                      # once: the ARX X5 description, textures, scans
ruibench doctor
```

The **first launch on a new machine takes tens of minutes** while the RTX shader
cache is built — it looks like a hang and is not. Later launches are quick.

Rendering is the most expensive thing the engine does, so it happens **on
demand**: `advance()` steps physics only, and a frame is produced when the
harness asks for one. That is also what makes the observation timestamp mean
something. Per-object visibility costs a render, so it is off unless a task
scores seeing (`needs_visibility`: DS1-DS3), and then worked out at most every
0.2 simulated seconds, from one render pass -- labels do not converge the way
lighting does.

An on-demand frame is **rendered several times**. RTX carries lighting from
frame to frame, and this engine renders long after the previous frame: a single
pass came out with a quarter of the image pure black, and after something moved
the old picture lingered as a faint ghost. So: FXAA rather than the default
DLSS (which still showed a translucent copy of a moved block 52 frames later),
up to eight passes for an observation (after the arm moved, the error where
things changed fell from 4.8/255 on the first pass to 1.5 on the eighth), one
per 50 ms of simulated time when frames come often (video), and thirty for the
first frame of an episode, when everything has just been put somewhere else.
The measurements are in `engines/isaac/engine.py`.

"Visible" on Isaac means two things at once: the renderer shows at least 16
pixels of the body (semantic segmentation), and, for a mark on one face, the
camera is in front of that face (`engines.base.legible_from`, the rule the mock
engine uses too).  Pixels alone called a mark seen from a couple of degrees
behind its plane.

### Reviewing a task's design

Two pages, for two questions.

* **Does the task read the way it is meant to?** `ruibench showcase` records,
  for every implemented task, the scripted reference succeeding on Isaac at
  the easy level and at the hard one (seed 0), and lays them out suite by
  suite, six tasks each: what the model gets -- the prompt, the hint, a film, the first
  frame from every camera -- then the two videos.  A clip that does not show
  its role stays off the page and is recorded again next time.  Failure clips
  (the agent a card names in `failure_example`) are not recorded yet.
* **Does every mission start right?**  A task is its missions -- every level at
  every seed -- and a layout that only one seed gets wrong is still a design
  bug.  `ruibench gallery --seeds 0-9` renders each mission's start on Isaac
  and counts, in the fixed camera, the pixels of the gripper and of every mark
  on the table; either being zero fails the command.

### Diagnosing a failed run

The order that has worked, cheapest first:

1. `ruibench view runs/x.jsonl` — the failed action is red on the timeline and
   says why ("blocked: stopped 7.5 cm short of the target"); the plan view shows
   where the tool stopped and what was around it; with `--record`, so do the
   cameras.
2. `python -m ruibench.engines.isaac.checks snapshot TASK LEVEL SEED` — where
   every body came to rest, every collider's bounds, how many pixels of each
   body every camera sees, and the images (in `runs/checks/`).
3. A focused experiment in its own process.  The ones used so far — the hand's
   collider envelope, render convergence, camera candidates, joint traces of a
   motion that looked wrong — are local research code (`research/`, not
   tracked); what they found is in the package, with the numbers in its
   comments.

A failure that shows up on Isaac and not on the mock is also a mock bug: the
mock laid objects in a bin side by side and so hid the tower that three drops
at one spot build under PhysX.  Fix both.

## Runs that do not hang

A benchmark that runs thousands of episodes on a simulator has to assume the
simulator will sometimes misbehave.  Three failure modes were observed on the
Isaac backend, and each has a structural answer rather than a note in a wiki:

| Failure | Answer |
|---|---|
| Tearing down a stage that has cameras hung for 15+ minutes | never tear a stage down in-process: one scene structure per worker process, episodes reset poses and colours in place, a structural change ends the worker (exit 75) |
| After Isaac starts, an ordinary exception becomes a hang: a normal interpreter exit runs Kit shutdown hooks that never returned | workers wrap everything in `except BaseException`, print the traceback, and leave with `os._exit` |
| A healthy long run looks the same as a hung one | workers write progress heartbeats (separate patience for "simulating" and "waiting on a model"); the parent kills a silent worker and retries its episode; worker output goes to `<run>.workers/*.log` |

Everything a worker loses this way is recorded as a `harness_error`: counted,
shown in the report, and excluded from every success rate.  The same rule
applies to `ruibench doctor`, which never starts the simulator in its own
process and runs each hardware check in a child with a timeout.
