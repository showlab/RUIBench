# The real station

Written for anyone running RUIBench on a real Franka.

An episode on the real robot is the episode the simulator runs: the same
interfaces, tools, executor, perception and Harnesses, the same ready-made
agents over MCP, and the same results layout. Only the engine changes:
`franka` (`ruibench/engines/franka/`) in place of `isaac`. Every difference
follows from three facts about it (`EngineCapabilities.real`):

- **Time is the wall clock.** The world goes on while a model thinks, so the
  measured clock is the only honest one; the episode refuses any other. Time
  thought is charged by having passed, never added again.
- **The sensors are real.** Pictures and depth come from the RealSense
  cameras, readings from the robot. Perception adds none of the simulated
  errors: a real camera errs as it errs.
- **Nothing knows where things are.** The engine's state is the robot alone,
  so no scorer can judge an episode. The operator judges it by the task card's
  success, and the verdict is the result.

## The rig

| | | Configured in |
|---|---|---|
| Arm | Franka Emika Panda with the Franka Hand. Its NUC runs Polymetis behind `franka_server/single_arm_server.py` (ZeroRPC, `<robot-host>:4242`: `nuc.host` and `nuc.port` in `rig.yaml`, or `RUIBENCH_FRANKA_NUC=<host>[:<port>]`) | `engines/franka/rig.yaml` |
| Fixed camera `front` | RealSense D435 (its serial in `rig.yaml`), upper mount across the table: 640 × 480 through a 21.5 mm lens (52 × 40 degrees) | `rig.yaml`, `model.py` |
| Wrist camera `wrist` | RealSense D405 (its serial in `rig.yaml`) on the hand: 512 × 512 through the simulated FR3's 18 mm lens (60 degrees) | `rig.yaml`, `model.py` |
| Robot model `panda` | The FR3's kinematic chain held to the Panda's joint limits (the server's, less its safety margin); the Franka Hand's tool centre point, 10.34 cm from the flange | `engines/franka/model.py` |
| Calibration | The table's height and both cameras' poses, from one ChArUco board | `engines/franka/calibration.json` |

The station frame is the simulated one: the origin is on the table top under
the robot base, x points forward, y left and z up, in metres. The robot
reports its flange in its base frame. The engine moves the flange to put the
tool centre point where a move asks.

The cameras' pictures are rectified to the robot model's lens
(`engines/franka/cameras.py`). The D405's distortion is undone, the principal
point is centred and the pixels are made square, so every frame is the
centred pinhole perception reads in simulation. The fixed camera keeps its
4:3 picture, as the simulated ARX station's head camera does. In the wrist
picture the robot's base is toward the top, and an open hand's fingertips just
show in the top corners; `task_info` says so.

## Once: the environments and the calibration

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e '.[dev,real]'
uv venv .venv-vision --python 3.11
uv pip install --python .venv-vision/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-vision/bin/python "transformers==5.17.*" pillow numpy
```

The robot's environment needs NumPy 1, as in simulation, and so an OpenCV before
4.12 (`.[real]` pins both). Under NumPy 2 a number worked out from perception's
`float32` stays one: the agent is shown `np.float32(0.5)`, and the episode ends in a
log it cannot write. The vision service's environment is the simulation's own
(torch for the RTX 5090, transformers 5.17 for SAM 3).

On a robot this engine has never driven, first try a few moves, with the table
clear. The arm moves 5 cm along each axis from its start and back, turns 30 degrees
and back, then goes down to 4 cm over the table, where joint 6 is nearest its limit,
moves and turns there, and comes back. It closes and opens the hand. Last it goes
20 cm out and is paused half way at full speed, which says how far it runs on after
P, and goes back. Before it moves,
the robot's joints and the flange it reports must put the tool centre point at the
same place, within two millimetres: that checks the robot model against Polymetis.

```bash
.venv/bin/python -m ruibench.engines.franka.calibrate moves
```

Calibrate once, and again whenever a camera or the table moves. The board lies
flat on the table, in reach and in the fixed camera's view. The default sheet
is (4 × 4 squares, `DICT_4X4_50`, an older OpenCV's layout); another is
given on the command line, or printed with `calibrate board`.

```bash
.venv/bin/python -m ruibench.engines.franka.calibrate run
.venv/bin/python -m ruibench.engines.franka.calibrate verify --hover-mm 80
.venv/bin/python -m ruibench.engines.franka.calibrate solve runs/real/calibration/<run>   # again, no robot
```

`run` takes about two minutes. It looks for the board from high up, then takes
up to 24 views of it with the wrist camera, going from one to the next joint by
joint. Back at the start, the fixed camera takes the board in its largest colour
picture. Each view is saved as it is taken. A square's size comes out of the
arm's own travel, with the wrist camera's depth as a second opinion; `--square-mm`
fixes it. `solve` gives the wrist camera's mount, the fixed camera's pose, the
table's height and each camera's `depth_scale`, and saves them in
`calibration.json` with its residuals.

`verify` brings the tool centre point over each corner of the board in straight
lines, as the tools move it. It keeps the fixed camera's pictures, marked where
the calibration has the corner (green), the tool centre point (red) and the
fingertips (yellow). The yellow marks should sit on the real fingertips: they
are off by about 1 px per millimetre the table's height is wrong. Hover high
enough to clear what is on the table. Until the station is calibrated its values
are provisional, and every position perception reports is a guess.

## Before a session

1. On the NUC, with the arm unlocked and FCI on in Desk, start the robot
   server: `ssh <robot-host>`, then `cd franka_server && python single_arm_server.py`.
   Keep a hand near the stop: nothing in RUIBench starts the server or the
   arm by itself.
2. Start the vision service (SAM 3, about 6.5 GB of the GPU):

   ```bash
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-vision/bin/python -m ruikit.perception.service.server
   ```

3. Check the station. This is read-only and nothing moves:

   ```bash
   .venv/bin/python -m ruibench.engines.franka.check
   ```

## The operator's window

Whenever the engine runs on the real robot (episodes, going home between them,
`calibrate`), a window shows both cameras live, the pictures the tools get
(`engines/franka/view.py`). **Press P to pause and P again to go on.** The window must
have the keyboard's focus; a red strip says when it has not, and then P does nothing.
**Press F to fail the episode under way.** The arm stops where it is and nothing moves it
again, a program's next moves among them, even from a pause. The episode ends, it counts
as a failure (`verdict.failed_by_key`), the runner asks only why, and the next interface starts. The
agent is told the episode has ended, and gives its summary first.

- **At once.** The arm holds where it is within a control period of the key being
  read. It holds softly: a push moves it, and it takes the new place.
- **What goes on.** A move under way continues from where the arm is, to the same
  goal, smooth from rest. What the agent asks meanwhile waits, the hand's commands
  too. A hand command already given finishes, since the hand cannot be stopped.
- **The clock stops too.** A move is neither blocked nor timed out by a pause, and
  no limit is charged for it. The agent is not told. It goes on thinking, and its
  next move simply takes longer to answer.
- **Things moved during a pause are the operator's word against the record.** Each
  pause is kept in `runs/real/flight/`; say so in the verdict's note.

P is not a safety device. It needs this machine, the window's focus and the NUC's
server, and the arm runs on for a moment first. In an emergency use the robot's own
stop. `RUIBENCH_FRANKA_VIEW=0` runs without the window, and `=1` opens it on a
stand-in robot too.

## Running episodes

The plan is `ruikit/configs/experiments/real.yaml`. The `bring-up` experiment
runs each of the 13 RUIs (the 12 fixed ones with the aids, and RUI-Free) once on
`R0`, putting the lemon on the plate; `suites` runs them on one task of each suite,
for both agents. `scripts/real_run.sh` lists the interfaces in the order they run,
and starts a task from one line once the NUC's server is up. It checks the station,
starts the vision service if it is down, tries the first moves if they have not
passed on this robot, and runs the episodes:

```bash
scripts/real_run.sh DS2 --test                     # first, a try: pose alone
scripts/real_run.sh DS2                            # every interface, Claude Code
scripts/real_run.sh DS2 --agent codex              # every interface, Codex (GPT-6 Astra)
scripts/real_run.sh DS2 --seed 16                  # another table
scripts/real_run.sh DS2 --interface pose rui-free  # only these
```

Before a suite's run, try it with `--test`: one episode through pose, as often as
wanted. It goes to `runs/real-test/<when>/`, and neither counts as a result nor keeps the
next run from running pose.

A seed is one table, laid out alike for every interface. Its number is in the cell's folder
(`DS2_easy_s16_...`) and its result, so another table never mixes with the first, and each
resumes on its own. A card lists its seeds and what sets each apart (`seeds`), and the plan
names the one a run takes without `--seed`. The numbers are simulation's: on `DS2`, the block
hides where the simulated seed of that number hides it. Each result records the table it ran
on (`layout`).

`suites` is set as simulation's `rui` experiment is, and a test holds it there: the same
agents and models at high effort, the same 13 interfaces, the easy level, every episode
filmed, and the limits of the simulated task a card stands for. Each episode is a fresh
agent in an empty home, with no memory, settings or session kept, through the same code as
in simulation, so one suite cannot reach another. What differs is what a real robot makes
differ: the clock is the wall clock, not 8 s a reply (the demo's agent thought about 3 s a
reply), and the operator lays out the table and judges it, one layout for all of a task's
interfaces, as one seed is in simulation.

The runner itself takes any other selection:

```bash
.venv/bin/python -m ruikit.experiments.real --dry-run                         # what it would run
.venv/bin/python -m ruikit.experiments.real --agent claude-code               # Claude Code's 13
.venv/bin/python -m ruikit.experiments.real --agent codex --interface pose rui-free
```

Episodes run one at a time:

1. The arm goes to its start. The runner shows the task card's setup; lay
   the table out, then press Enter (`s` skips the episode, `q` stops).
2. The episode runs as a ready-made agent's runs in simulation: a session on
   the real engine, a fresh agent with a clean context, filmed. The arm first
   goes to its start, about the robot's HOME: the tool centre point 38 cm out and
   29 cm over the table, pointing down, hand open. The terminal follows it as it
   happens: each of the agent's calls and its answer, with the time on the episode's
   clock; a program's source and its calls, numbered (`#4.2` is the second call of
   the agent's fourth); and what the agent says. The window's second line says the
   call under way, or how long the agent has been thinking.
3. When it ends, the runner shows the agent's summary. Judge the table by the
   card's success (`y`/`n`) and add a note if needed. The verdict becomes the
   result's `success`, with `judged_by: operator`. `scored_success` keeps the
   episode's own score, never a success on a real task.
4. An episode that does not count, because the agent was cut off or the robot
   faulted, can be run again.

Before the first episode the runner checks the station, the vision service and
each agent's self-test, as the batch runner does on Isaac. It resumes: a cell
with a result is not run again. Results go to `runs/real/`, in the simulation's
layout.

`RUIBENCH_FRANKA_FAKE=1` stands the robot and cameras in with
`engines/franka/fake.py`. The engine, its control thread and the whole runner
run with nothing moving and nothing counted, which is for trying the pipeline
and an agent's setup without the robot.

## Keeping the record

Each day's episodes go to a gated Hugging Face dataset of yours, `<your-hf-dataset>` (set
`RUIBENCH_REAL_DATASET=<owner>/<name>`), in a folder named `MMDD-<what>`, for those judged on
that day. Anyone sees its card and its files' names; to read them a collaborator requests
access on its page, and its owner lets them in:

```bash
.venv/bin/python -m ruikit.experiments.real_upload 1102-PickAndPlace --dry-run   # what would go
.venv/bin/python -m ruikit.experiments.real_upload 1102-PickAndPlace
.venv/bin/python -m ruikit.experiments.real_upload 1103-FindTheBlock --task DS2    # a suite's own
```

The folder's `README.md` is a table of its episodes. Each episode has its video, its
`details.md` (the times, the tokens, the tools chosen, each call and its answer, the code
written, what the agent said), its raw logs, the pictures the agent saw and the arm's flight.
It uses this machine's Hub login (`hf auth whoami`), which must be able to write there.
Uploading again replaces what is there.

## What the agent is told

- The ready-made agents' prompt is the simulation's word for word, except for
  what is real: a real robot and real cameras, a limit by the wall clock, and
  a person who stands by for safety only (`session_agent.REAL_PROMPT`).
- `task_info` gives the robot's own words where the simulated station's do not
  fit it (`model.CONVENTIONS`): how far its wrist turns, where its wrist camera
  looks, and what its sensors' errors are.
- The tools are the simulation's, descriptions included. A tool's "simulated
  time" is the episode's clock, here the wall clock.

## A game the program plays

`DI4`, Simon says, needs no one to say anything (`tasks/real/simon.py`). The program says
simulation's calls for the seed, on the same times, from the agent's first call, and the agent reads
each as it does in simulation: the instruction changing, with its next tool result, whichever tool
that is. Each call is said in its turn, however long the agent thought. A touch is the hand over a
toy's spot, as the card has it, down to within reach of its top; the operator's terminal follows
the calls and the touches, and is told at the end what the program's record makes of them. The
verdict is still the operator's: where a toy really stands the card only roughly knows.

## When the hand stops answering

Polymetis' `franka_hand_client` runs each command in a detached thread and catches nothing, so
a command that fails, or a second without a state from the hand (`libfranka: UDP receive:
Timeout`), ends the process with SIGABRT. The gripper's server lives on and goes on answering
with the last width, so the fingers stop without anyone being told. It happened at the moment of
a grip, on rigid things: the object stays on the table, and the width still says it is held.

- The NUC's client is patched to report such a failure and go on
  (`engines/franka/franka_hand_client.patch`, applied 2026-11-03; the original binary is
  `build/franka_hand_client.orig`). A failure shows in the terminal of `launch_gripper.py` as
  `Gripper command failed: ...` or `Gripper state not read: ...`.
- That keeps the client alive, and does not keep the grip: on 2026-11-03 the fingers still stopped
  at contact with a rigid block now and then, and the block stayed on the table. The server's
  old call closes at 0.5 m/s, five times libfranka's example, so the station closes through a
  separate `grasp(speed, force)` at `rig.yaml`'s 0.05 m/s and 60 N, added to the NUC's
  `single_arm_server.py` (the old call is untouched; a server without the new one is used as
  before, and the flight log says so). Whether this cures it has not been measured.
- To keep what the client says: `launch_gripper.py gripper=franka_hand 2>&1 | tee -a ~/gripper.log`.
- `check` and the runner, before each table, ask the NUC whether the client runs. An episode whose
  fingers did not move asks too, and ends in a fault if it has died, and so does not count. To
  restart it: `ssh <robot-host>`, then `launch_gripper.py gripper=franka_hand`; it re-homes the fingers.

## How the arm moves, and what keeps it safe

- One control thread owns the connection to the NUC (`engines/franka/arm.py`).
  At 50 Hz it reads the arm and streams setpoints to Polymetis' Cartesian
  impedance controller:
  - a move follows a straight line, minimum-jerk in time, at most 0.15 m/s and
    0.6 rad/s, and slower where its joints need it; going down to within about 18 cm of
    the table, at most 0.08 m/s (`descent_speed_m_s`);
  - a joint move streams joint targets to joint impedance, at most 0.6 rad/s;
  - at rest the setpoint is nudged by what still separates the hand from its
    goal, by at most 1 cm. A free hand so rests within the 2 mm the executor
    calls arrived, and one pressed on something pushes no harder than 7.5 N.
- Every move is followed before it starts, the way the robot's own controller
  would follow it (`engines/franka/route.py`). Polymetis turns each setpoint into
  joints from where the joints are, with no limits and nothing to keep the arm's
  redundancy in hand, so only following the line shows where the joints go. The
  tool centre point must stay inside the workspace box (`rig.yaml`; the fingertips
  stop at the table) and its axis within 100 degrees of pointing down. Each joint
  must stay inside its limit for a line: half way into the server's safety margin,
  and joint 6 up to 3.0 rad, short of pi, the wrist's singularity. Near the table
  joint 6 needs that much, and joint 4 keeps the hand from places nearer the base
  than about 38 cm. The move then takes as
  long as its fastest joint needs. A move that may not go all the way stops where
  it must, and the executor reports it blocked short, as in simulation.
- The control thread watches the arm whatever it was asked:
  - the hand too far from its setpoint (3 cm, plus what its speed lets it trail
    by; 17 degrees), or a joint going on within 0.05 rad of the server's limit:
    the arm holds where it is, and the move is reported blocked;
  - a joint faster than 1.2 rad/s: the arm holds where it is, and the episode
    ends as the harness's failure, never scored.
- A late answer from the server pauses a motion; the setpoint never jumps ahead
  to catch up. A server error that lost the controller means a reflex ended it:
  the controller restarts where the arm is, the arm holds there and the motion is
  over. The third time, or any other error, or no answer for a second, ends the
  episode as the harness's failure, never scored. A fault leaves the last setpoint
  where the arm is.
- What the arm did is kept as it happens in `runs/real/flight/`: its joints, flange
  and setpoint every period of a motion, and everything the watching did. The robot's
  own account of a reflex goes only to the terminal that started the NUC's server.
- The speeds are bring-up speeds. The simulator's controller moves at
  0.25 m/s and 1.5 rad/s; `rig.yaml` says where to change them. Polymetis' joint
  damping has the arm trail its setpoint by about 0.2 s in motion.

On 2026-11-02 the calibration's first run put the robot into what looked like a
reflex (red light, then a stuttering yellow): its views bent the hand forward toward
joint 6's pi. Replayed offline through the robot's own controller, a line there asks
joint 3 for 2.9 rad/s, against the server's 2.075; the check of the time solved each
point of the line alone and did not see it. The robot's own message went to the NUC's
terminal and was not kept. Most of the above comes from that.

When a move stops short, what the same command had queued behind it does not run, here where in
simulation it does: the descent after a blocked approach would drive the hand on into what stopped it,
after the agent has been told it is blocked. A hand stopped by something holds once its setpoint is
38 mm ahead at full speed, before the robot's own reflex at 40 N, and what pressing built up of the
setpoint's offset fades over the next move.

## Adding a real task

A real task is a card in `ruibench/tasks/real/tasks.yaml` with these fields:

- `instruction`, `hint`: what the agent is told;
- `things`: what is on the table, by name and category; the categories are
  the words perception lists everything by. Try them on the real table: plain words
  do best, since each detection comes with its colour;
- `setup`: how the operator lays the table out; only the operator sees it, so it may
  say where a hidden thing is;
- `picture`, `caption`: a picture the agent needs and what it is told of it: a file in
  `ruibench/tasks/real/cards/` (`CL3.png`, drawn by `scripts/real_cl3_card.py`), which
  `task_demo` shows through every interface. Nothing of it is on the table, and nothing moves;
- `success`: how the operator judges it;
- `max_model_calls`: its limits, as a simulated task's: 10 replies and 8 s a call;
- `game`, `toys`: a game the program plays with the agent, on these toys, each with what the calls
  call it and where it stands;
- `mirrors`: the simulated task it stands for, if any: its id, and the card takes that
  task's limits at the easy level. A card standing for a simulated task has that task's id;
- `seeds`: what sets each table apart, on top of `setup`, by its number. A card without them
  has one table, seed 0.

A real task is built as level `easy`. Add it to the plan's experiment, and run it.

## Not frozen

The real station lives outside the frozen benchmark's fingerprint
(`ruibench/tasks/freeze.py`): the engine, the robot model, the rig, the
calibration, the real tasks and their runner. They change with the station, and
no simulation result depends on them. Each real result records its commit,
and its log records the rig and the calibration it ran with.
