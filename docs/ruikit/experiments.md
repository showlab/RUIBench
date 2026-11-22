# Running experiments

## Models

A model is named `<channel>:<name>` (for example `openai:<model>` or
`local:<model>`). A bare name goes to the default channel.
`ruikit/configs/models.yaml` lists, for each channel:
- the endpoint, and the environment variable that overrides it;
- the environment variable its key is read from;
- the deployments, by short name, if any.

Keys and endpoints come from the environment or from `.secrets/models.env`,
which is not tracked. A variable set in the shell wins over the file.

| Channel | Request format | Notes |
|---|---|---|
| `openai` | Chat Completions | OpenAI's API, with `OPENAI_API_KEY`; any OpenAI-compatible endpoint with `OPENAI_BASE_URL`. |
| `anthropic` | Chat Completions | Anthropic's OpenAI-compatible endpoint, with `ANTHROPIC_API_KEY`. |
| `local` | Chat Completions | A model served by vLLM (`RUIKIT_LOCAL_BASE_URL`). Thinking is off unless `RUIKIT_LOCAL_THINKING=1`. What it thinks is kept in the transcript only. |

The paper's small open model, Qwen3.5-4B (zero-shot and fine-tuned), runs
through the minimal loop on the `local` channel.

```bash
ruikit episode --task DS1 --interface pose --model openai:<model> --out runs/x   # one episode
ruikit episode --task DS1 --interface pose --model local:<model> --clock paused  # a served model, thinking free
```

`ruikit run` first checks each model with a one-line hello (`--no-ping` skips
it).

**The minimal loop** (`--agent minimal-loop`, `ruikit/harness/minimal_loop.py`) is the thinnest Harness:
- The model sees the task and the interface's tools.
- Each reply it sends is one reply of the episode, made of one or more tool
  calls; independent calls run together.
- Results go back as the result's JSON followed by the pictures. Only the
  latest six pictures stay in the history, and each picture a reply brings is
  shown at least once, however many it brings.
- A reply with no tool call is nudged twice, then the run ends.

## Conditions

| Clock | A reply costs | Folder suffix |
|---|---|---|
| `8s` (the main setting; any `<N>s`) | 8 s of simulated time: the arm holds still while the scene goes on | -- |
| `paused` | nothing: the world waits for the model | `@paused` |
| `measured` | what the model really took | `@measured` |

Information is either pixels (the main setting) or privileged, which adds
`@privileged` to the interface id. `ruikit.Condition(clock, information)`
names one condition. `ruikit.run(..., condition="paused")` takes a clock's
name or `"privileged"`.

## Plans

A plan (`ruikit/configs/experiments/<name>.yaml`) lists experiments. Each one
gives its rows, its columns, its tasks, its levels and its seeds:
- A row is an agent, meaning a Harness and a model, under one condition.
- A column is an interface.

The paper's plan is `paper` (`ruikit/configs/experiments/paper.yaml`). A
cell runs its tasks at the easy level on the first test seed (15), one
episode each, filmed, so episodes pair across rows; `rui-seed2` and
`rui-seq-seeds` add later test seeds.

```bash
ruikit run --plan paper --dry-run                 # every episode it asks for
ruikit run --plan paper --experiment rui --agent claude-code --model claude-opus-5-5
ruikit run --plan paper --report                  # the tables so far
```

Without `--plan`, `ruikit run` runs the settings given (`--agent`, `--model`,
`--task`, `--level`, `--seeds`, `--interface`) into `runs/ruikit`; with one,
`runs/<plan>` is the default folder. The part of a plan to run is chosen by
its settings; each one omitted, all of it:

| Setting | Option | Values |
|---|---|---|
| Experiment | `--experiment` | `rui` (Claude Code and Codex on the 13 RUIs), `rui-seed2` (the next test seed), `rui-seq`, `rui-seq-control`, `rui-seq-seeds` (the action-sequence ablation), `check-report`, `check-correct` (the skills' completion check) |
| Agent | `--agent` | `minimal-loop`, `codex`, `codex-code`, `claude-code` |
| Model | `--model` | `openai:<model>`, `anthropic:<model>`, `local:<model>`; for `codex` and `claude-code`, the agent's own (`gpt-6-astra`, `claude-opus-5-5`) |
| Interaction form | `--form` | `per-action`, `program`, `sequence`, `free` |
| Action abstraction | `--abstraction` | `joint`, `delta`, `step`, `pose`, `point`, `skill` |
| Perception | `--perception` | `pixels`, `privileged` |
| Interface | `--interface` | a name (`pose`, `program-pose`, `program-skill`, `rui-free`, ...) or an id |
| Tasks, levels | `--task`, `--level` | ids, suites or `all` (`WK1`, `CR`); `easy`, `hard` |
| Clock | `--clock` | `8s`, `paused`, `measured` |

`--agents codex:gpt-6-astra` picks rows as agent:model. The seeds come from
the plan. `--dry-run` ends with one line per agent: its episodes, interfaces,
tasks, levels and seeds.

The batch runner:
- **Workers.** Each episode gets a fresh simulator (`--per-worker 1`, the
  default): a reused Isaac stage keeps the previous episode's physics layout
  and renderer state, so outcomes would depend on the order. A larger
  `--per-worker` trades that for speed (starting Isaac takes about two
  minutes).
- **Resuming and retries.** It resumes from what is already in the folder. It
  retries a crashed worker's episode up to three times, then writes
  `lost.json`; a lost episode is the harness's failure and never counts.
- **API failures.** It waits before retrying an episode that the API ended,
  and stops launching when a quota is used up.
- **Isaac.** Each worker needs a card with 7 GB free. Starts are 60 s apart,
  and nothing starts while the machine's load is over `--max-load` of its
  cores. At most `--max-isaac` Isaac apps run at once.

## The results layout

```text
OUT/<harness>-<model>[@<clock>]/<task>_<level>_s<seed>_<interface>/
    result.json      success, the milestones reached, replies, the settings
    lost.json        the harness could not run it (never counted)
    api_error.json   the model's API ended it: run it again
    *.jsonl          the episode's logs
```

A cell is one combination of Harness, model, clock, task, level, seed and
interface. It is run once, for whichever experiment asks for it first.

Each `result.json` records what the cell ran with:
- the benchmark version, the Harness, the model, the clock, the interface,
  the aids and the perception;
- `declared_done` (whether the agent said it was done), beside the verdict;
- `furthest_milestone`, which tells a wrong decision from a right one not
  carried out;
- `provenance`: the RUIKit version, the frozen benchmark version and whether
  the checkout matched it when the worker started (`ruibench freeze
  --check`), the commit (`unknown` without git), the interpreter and its
  numpy, `per_worker`, and `process_episode`, the episode's place in its
  simulator process (1 is a fresh simulator).

The runner's first line after the plan says the same of the code it runs.

`--report` writes `cells.jsonl`, for each experiment `report-<name>.md`, and
`index.html`, a page of all of it (`ruikit.report.results_page`; with
`ruibench serve runs --port 8765` running, http://localhost:8765/paper/index.html).
Both open with the summary: for each agent and interface, the success rate
with its Wilson 95% interval, the tokens (median per episode and in all; the
input counts cached input), the time (median per episode and in all, from the
episode's start to its end, without starting the simulator) and the videos.
On the page a row lists its episodes, success or failure each, and an episode
plays its video; the address names the view, down to one episode. The
reports also give:
- success with its Wilson 95% interval, by row and column, by kind of task
  and by level;
- replies, simulated time and tokens;
- how often an agent said it was done and failed;
- for each task, the furthest milestone that failed episodes reached;
- paired differences where an interface appears with and without some aids;
- the results that are not comparable: not on this checkout's frozen version,
  not in a fresh simulator, filmed where the rest are not (or the other way
  round), or without a provenance (results from before RUIKit 2.0.1). They
  stay in the tables, and the report says how many there are.

**Videos.** A plan with `record: true` (`paper` has it) films every episode:
`video.mp4` beside its `result.json`, a frame every 0.2 simulated seconds; the
raw frames are removed once the video is written. A filmed episode ends as an
unfilmed one does. Filming renders the cameras between
the agent's looks, and this changes the pictures it is shown: on Isaac, the same
calls filmed and not filmed gave identical tool data, simulated times and final
state, but pictures differing by 2.4 of 255 on average (5% of the pixels by more
than 8), against 0.7 (0.15%) between two unfilmed runs. So a plan films all its
episodes or none; `--no-record` is for a quick trial, and every result records
`filmed`.

## Ready-made agents

A row with `harness: codex`, `codex-code` or `claude-code` (`--agent` on the
command line) runs each episode as a session (`ruikit.serve.session`) with a
fresh agent on it (`ruikit.harness.session_agent`):

- **Clean context.** The agent starts in an empty folder of its own outside
  the repository, named by a random id and deleted after the episode, with
  only its login. It reads no instruction file (CLAUDE.md, AGENTS.md) and no
  memory, no other session can message it (Claude Code runs restricted), and
  it gets the frozen prompt and nothing else.
- **No shell, no files.** The session's tools over MCP are all it has, with
  pictures inline. Every tool call is one reply; `batch` sends several calls as
  one reply.
- **Codex** calls each tool itself (`codex`). `codex-code` is Codex's own
  code mode, calling the same tools from the JavaScript it writes. It is used
  for RUI-Free only, since a loop of calls there is a program.
- **Reasoning effort** is `high` for both Codex and Claude Code, as in the
  paper.

After the episode the agent has three minutes for its summary. The result
gets the agent's token use (`ruikit.experiments.usage`), its CLI's version and
reasoning effort, and an audit of its transcript
(`ruikit.experiments.rules_audit`). The episode is run again with a new agent
when the agent was cut off by a usage limit or an API error, broke a rule, did
not get the session's tools or never called them, or was reached by anything
besides its prompt and the session's results.

Before an agent's first episode on Isaac, the batch runs its self-test
(`ruikit.harness.selftest`; again when the CLI's version or the commit
changes): a probe episode on the mock engine with instruction files planted
in the agent's folders, judged by what the agent called, heard and read.
All of this is in the benchmark's fingerprint (v2.1).

```bash
python -m ruikit.harness.session_agent selftest claude-code --model claude-opus-5-5
ruikit episode --agent claude-code --model claude-opus-5-5 \
    --task WK1 --level easy --seeds 15 --interface pose --out runs/x --gpus 3
```

## The real robot

The ready-made agents run on a real Franka through the engine `franka`, one
episode at a time with an operator at the robot, from the plan `real`
(`python -m ruikit.experiments.real`). The operator lays each table out and
judges each episode. Results go to `runs/real/`, in this layout. The setup, the
calibration and the runner are described in [../REAL_ROBOT.md](../REAL_ROBOT.md).

## Serving agents

- **MCP** (`ruikit serve-mcp`, `ruikit.serve.mcp_stdio`): one episode over stdio. With
  `--session D` it attaches to a session already running and adds `batch`.
- **Session server** (`ruikit.serve.session`): one episode driven from a
  shell, one reply per command.
- **SDK** (`ruikit.serve.sdk.Client`): the interface's tools as methods.
- **Schemas** (`ruikit.serve.schemas`): the tools in Anthropic, OpenAI
  (strict), Gemini and MCP form.

## Tool-solvability

`ruikit coverage` runs the task's scripted reference through each interface's
tools. The reference reads the simulator to *decide*, and uses only the
interface's tools to *act* (`ruikit/diagnostics/reference/`).

By default the reference takes every coordinate from the interface's
perception tools: it looks, detects, and estimates the pose of the thing a
move is relative to, and it never uses a true value. With
`--oracle-coordinates` it sends the simulator's own numbers instead. That
asks only whether the tools can *act* the task. A cell that fails in that
mode means the interface cannot do the task at all.

Interfaces should differ in how well they do a task, never in whether they
can. So the rule for changing one is two-sided:
- **Add** a tool or an option only for a cell that fails without it, and pin
  that cell in `tests/ruikit/diagnostics/test_coverage.py`.
- **Add nothing more.**

On Isaac every cell gets a fresh simulator, as in the batch runner
(`--per-worker 1`, the default), so no cell's outcome depends on the cells
before it. `--per-worker N` keeps a simulator for N cells of one task and
level, which is faster but no longer independent of the order. Each cell
records its `provenance`, as a batch result does.

```bash
ruikit coverage --engine isaac --workers 4 \
    --out runs/x/cells.jsonl --log-dir runs/x/logs
ruikit coverage --engine isaac --workers 4 --level easy \
    --record runs/show/clips --out runs/show/cells.jsonl     # film every cell
```
