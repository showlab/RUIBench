# Getting started: testing an agent on RUIBench

How to install the benchmark, run your agent or model on it, and reproduce the
paper's experiments. Everything here was run on a fresh checkout.

## 1. Install

You need Python 3.10 or 3.11. The core (task cards, the mock engine, scoring,
reports, the RUIKit tool library) has almost no dependencies:

```bash
git clone <this repository> RUIBench && cd RUIBench
uv venv --python 3.10 .venv && source .venv/bin/activate   # or: python3.10 -m venv .venv
uv pip install -e ".[dev]"                                  #     pip install -e ".[dev]"
ruibench validate          # 0 errors expected
ruibench run DS4 --seeds 0-1 --engine mock   # two episodes of the reference, no GPU
```

(On Ubuntu, `python3.10 -m venv` needs the `python3.10-venv` package; `uv`
does not.)

Results come from **Isaac Sim 4.5.0 + Isaac Lab 2.1**, installed into their own
virtualenv: `ruibench doctor --hint` prints the commands (pin
`warp-lang==1.5.0`). Then, in that virtualenv:

```bash
pip install -e ".[isaac]"
export OMNI_KIT_ACCEPT_EULA=YES     # after reading NVIDIA's terms
ruibench setup                      # fetches the robots, textures and scanned
                                    # objects (~2.6 GB, every file sha256-pinned)
ruibench doctor                     # checks the machine
```

The downloads go to `third_party/`; set `RUIBENCH_THIRD_PARTY` to share one
copy between checkouts. The first Isaac launch on a machine compiles shaders
and takes several minutes.

**On a shared machine:** what an Isaac process is short of is CPU, not GPU:
about 6 cores while it runs (8-19 when it records video), about 40 for the
minute or two it starts.
Start them a minute apart, keep the machine's load below about 0.8 of its
cores, and say so before starting a batch (`ruikit run` does the first two by
itself). `ruibench stop` stops only what this checkout started.
CR6 is the exception: its cloth is solved on the graphics card (the first
one the process sees: choose it with `CUDA_VISIBLE_DEVICES`), and on a card
that another job keeps busy it runs several times slower.

## 2. What you run on

- **30 tasks** in five suites (World knowledge, In-context learning,
  Discovery, Dynamic interaction, Contact-rich manipulation), each at an
  **easy** and a **hard** level -- 60 task-levels.
  `ruibench tasks` lists them; `ruibench show DS1` prints one.
- **Seeds.** Every task is procedural: a seed lays a mission out.
  `ruibench/tasks/splits.json` deals each task-level's seeds into **15
  development (0-14), 20 test (15-34), 5 spare**. Develop on the development
  seeds (`--seeds dev`; `ruibench seeds DS1 easy` prints them); report on the
  test seeds (`--seeds test`). The paper's tables use the easy level and the
  first test seed (15) of every task.
- **Robots.** Single-arm tasks: a Franka Research 3 with a fixed camera
  across the table and a wrist camera that shows the fingers. Two-arm tasks
  (WK4, CL1, DI1, DI5, CR6): two ARX X5 arms with a head camera and a wrist camera
  each.
- **Scale.** The paper's main table is 30 tasks x 13 RUIs x one test seed = 390
  episodes per agent. An episode with a real model takes roughly 5-20 minutes on
  Isaac.

## 3. Connect your agent

An agent acts on the robot only through **RUIKit** tools. The 13 RUIs go by the
paper's names -- `joint`, `delta`, `step`, `pose`, `point`, `skill`,
`program-joint` ... `program-skill`, and `rui-free` -- and each has a versioned id
(`program-skill` is `rui.code-skill/v1`); a name runs the RUI as the paper did,
with the aids. `ruikit interfaces` lists them; `docs/ruikit/README.md` has the
details. Every command spells its options the same way: `--task`, `--level`,
`--seeds`, `--interface`, `--agent`, `--model`, `--clock`, `--engine`, `--gpus`,
`--out`.

The mock engine (`--engine mock`, no GPU) renders no pictures: a model there
sees only what the tools report, so a task that hinges on how things look --
CL2's look-alike dolls, told apart by their hair -- cannot be solved on it.
Develop those on Isaac. On Isaac every interface answers from pixels, so a
vision server must be up: set `RUIKIT_VISION_URL` (several, comma-separated, are
shared out among workers; `docs/ruikit/perception.md`). Four ways in:

**A model through its API**, RUIKit's own tool-calling loop (`minimal-loop`). The
channels are in `ruikit/configs/models.yaml`: `openai:<model>` (`OPENAI_API_KEY`;
`OPENAI_BASE_URL` for any OpenAI-compatible endpoint), `anthropic:<model>`
(`ANTHROPIC_API_KEY`), `local:<model>` (a vLLM server, `RUIKIT_LOCAL_BASE_URL`).
Keys come from the environment or from `.secrets/models.env` (gitignored).

```bash
ruikit episode --task DS1 --seeds dev:1 --interface pose --model openai:<model> --out runs/mine
ruikit run --agent minimal-loop --model openai:<model> --task WK6 WK5 --level easy \
    --seeds dev:3 --interface pose program-pose --out runs/mine      # --dry-run: the plan only
```

`ruikit run` runs agents x tasks x levels x seeds x interfaces, seed by seed, each
episode in a fresh Isaac process; stop it any time and run the same command again
to go on.

**A ready-made coding agent** -- Claude Code or Codex, each with its own login --
one fresh agent per episode, reaching the robot over MCP and nothing else:
`--agent claude-code` or `--agent codex` (`codex-code`: Codex in its own code
mode) with its `--model`.

**MCP over stdio**, one episode per server, for any MCP-speaking agent:
`ruikit serve-mcp --task DS2 --interface pose`.

**Python, in process.** Every call goes through the episode: the same clock,
limits and log.

```python
import ruikit
agent = ruikit.Agent(model="openai:<model>", interface="program-pose")
ruikit.run(agent, task="DS2", level="easy", seed=0, out="runs/mine")

from ruikit import Episode                       # or by hand, one tool call at a time
from ruikit.serve.sdk import Client
ep = Episode("DS2", level="easy", seed=0, profile="rui.pose/v1+aids", engine="isaac")
robot = Client(ep)
frame = robot.camera.look(cameras=["front"]).data["frames"][0]["frame_id"]
...
robot.task.done(report="the egg is in the basket")
print(ep.outcome)        # success, judged from the physical trajectory
ep.close()
```

**The paper's experiments** are a plan, `ruikit/configs/experiments/paper.yaml`:

```bash
ruikit run --plan paper --dry-run                               # every episode it asks for
ruikit run --plan paper --experiment rui --agent claude-code --model claude-opus-5-5
ruikit run --plan paper --experiment rui --interface program-delta program-step
ruikit report runs/paper                                        # the tables and the results page
```

## 4. Scores

An episode's success is decided by the benchmark from what physically
happened, never from what the agent reports. Each episode writes a
`ruibench-log/2` JSON line with the whole trajectory:

```bash
ruikit rescore runs/mine --task DS1         # scores a run's episodes again from their logs
                                            # alone; official scores come from this
ruibench view runs/mine/<agent>/<episode>/<episode>.episode.jsonl
                                            # replay: plan view, cameras, timeline
ruibench serve                              # open runs/ from another machine (port 8765)
```

`score.flags` lists anything physics would not do (a teleport, a throw, a
joint past its stop); a flagged episode fails until someone looks at it.

## 5. Ground rules

- Your agent gets what its cameras show, what the arm reports and the task
  text -- nothing from the simulator's state, the scorer, the seed or the
  internal object names. No tool may be more precise than a real robot's
  would be.
- Do not tune on the test seeds.
- The task cards, logs and videos carry a canary string
  (`ruibench/core/canary.py`): keep benchmark data out of training corpora.

## 6. Version

The benchmark is frozen as **v1.0.0**, the paper's version: `T_ref` is 8 s; the
timed levels are calibrated against it; the seed splits are fixed. `ruibench
validate` (and `ruibench freeze --check`) lists any file that shapes an episode
and has changed since; each result records the version it ran on, and results
compare only within one version. [CHANGELOG.md](../CHANGELOG.md) has the changes
from version to version.
