# Extending RUIKit

Each kind of part has a base class and a registry. Adding one means adding a
file and registering it. Run `ruikit check` and the tests afterwards.

## A tool

```python
from dataclasses import dataclass

from ruikit.core.params import param
from ruikit.core.result import ToolResult
from ruikit.core.tool import Cost, tool


@dataclass(frozen=True)
class HeightParams:
    frame_id: str = param("a frame_id returned by camera_look")
    u: float = param("pixel column")
    v: float = param("pixel row")


@tool(name="vision_height", family="perceive", params=HeightParams, reads=("frames",),
      source="vision", cost=Cost("declared", 0.1),
      examples=[{"frame_id": "f3.front", "u": 256, "v": 300}],
      description="""How high above the table the surface at a pixel is, in metres, from that
      frame's depth. Only what the frame shows; a pixel on nothing returns not_visible. Takes
      0.1 s of simulated time and moves nothing.""")
def vision_height(ctx, p: HeightParams) -> ToolResult:
    lifted = ctx.perception.lift(ctx.frame(p.frame_id), p.u, p.v)
    return ToolResult(data={"height_m": lifted["xyz_m"][2]}, text=f"{lifted['xyz_m'][2]:.3f} m")
```

**Naming and description.**
- The name matches `^[a-z][a-z0-9_]{0,63}$`. Providers reject dots; inside a
  program the tool becomes `vision.height`.
- The description gives units and frame, says whether the tool blocks, and
  says what it does not do. It must be at least 120 characters, or
  registration fails.

**What to declare.**
- `source`: where the answer comes from.
  - `vision`: read off the pictures by the perception backend.
  - `model`: an outside model. Such a tool stays out of the main comparison.
  - `oracle`: allowed only on the diagnostic track.
- `controls`: whether the tool moves an arm.
- `cost`: what the call charges to the simulated clock.
- `tracks`: where the tool may be offered.
- `family`: for an MCP client, it decides whether the tool is read-only.

**Touching the world.** A tool touches the world only through `ctx`
(`capture`, `frame`, `perception`, `arm_state`, `move`, `grip`, `stop`,
`task`, ...). `ruikit check` fails a tool module that imports the task cards,
the scorer, the runner or the executor, or that reaches past `ctx`.

**Placing it.** Every registered tool has one place in the design space
(`ruikit/interfaces/roles.py`): an action abstraction, a component or an
extra. A tool with no place fails `ruikit check`. A built-in tool's module
joins `BUILTIN_TOOL_MODULES` in `ruikit/core/tool.py`.

**Long-running tools.** A tool that keeps acting after it returns, such as a
tracker or a policy, is a **session**: an object with `tick(episode, dt)`,
`request_stop(episode)` and `close(episode)`, added with `ctx.add_session`.
`robot_stop` stops it, and nothing it computed before the stop moves an arm
afterwards.

## An action abstraction

1. Write a module in `ruikit/actions/` with its tools, ending with:

   ```python
   ABSTRACTION = ActionAbstraction(
       "wrench", "扳手", "a turn about a held axis: 'a quarter turn clockwise'",
       ("robot_turn_held", "robot_gripper"), "the published work it stands for",
       kind="move", params={"robot_turn_held": {"max_deg": 90.0}})
   ```

   - `kind` is `move` (moves the hand step by step) or `act` (a whole act).
     It orders the tools in the free form.
   - `params` are settings of its tools that every interface over it fixes.
   - `hand_back` says when a per-action reply returns: `when_done`, or
     `each_step`.
2. Add it to `ACTIONS` in `ruikit/actions/__init__.py`, and its module to
   `BUILTIN_TOOL_MODULES`.
3. Name its interfaces in `ruikit/configs/interfaces.yaml`:

   ```yaml
   rui.wrench/v1:
     form: per_action
     action: wrench
     title: "RUIBench wrench"
   rui.code-wrench/v1:
     form: program
     action: wrench
     title: "RUIBench program · over the wrench"
   ```

   The forms generate the tools, what a program may call, the channel and the
   hand-back. RUI-Free picks up the new abstraction's tools by itself. Short
   names (`wrench`, `program-wrench`) come from `ruikit/naming.py`, which lists
   the paper's six abstractions; without one, the interface runs by its id.
4. For `ruikit coverage`, add an expressor that turns the reference's commands
   into the new tools (`ruikit/diagnostics/reference/`).

## An interaction form

Subclass `Interface` (`ruikit/interfaces/base.py`) and register it in `FORMS`
(`ruikit/interfaces/catalog.py`):

```python
class Narrated(Interface):
    """The per-action tools, each reply first saying what it is for."""

    name = "narrated"
    channel = "tools"

    def tools(self, action):
        return SENSING + action.tools + CONTROL + TASK
```

The class can override five methods:
- `tools`: what the model calls.
- `code_tools`: what a program may call.
- `hand_back`: when control comes back.
- `dialect`: what the log calls the abstraction.
- `params`: the tool settings the interface fixes.

Rules the form enforces at run time belong to the episode. The per-action
form's one action per arm per reply is an example (`PerActionRule` in
`ruikit/interfaces/per_action.py`).

## A Harness component

A module in `ruikit/harness/components/` holds the component's tools and its
`COMPONENT`:

```python
COMPONENT = Component(
    "marks", "视觉标记", "numbered marks drawn on a picture (Set-of-Mark)", ("vision_marks",),
    (Implementation("dots", "物体中心画编号圆点", "a numbered dot on each thing",
                    params={"vision_marks": {"style": "dots"}}),
     Implementation("masks", "描出物体轮廓再标号", "each thing outlined and numbered",
                    params={"vision_marks": {"style": "masks"}})),
    default="dots")
```

- **Registering.** Add the component to `COMPONENTS` in
  `ruikit/harness/components/__init__.py`.
- **Attaching.** An id attaches it as `+marks` or `+marks:masks`.
- **Settings.** An implementation's `params` reach its tools as `ctx.params`, the
  settings of the tool being called.
- **Among the aids.** To make the component one of the aids, add it to a group
  in `AID_GROUPS` (`ruikit/harness/components/aids.py`). It must not move the
  arm another way or ask another model.

An **extra** is built the same way, in `ruikit/extras/`, and is registered in
`EXTRAS`.

## A perception backend

Subclass `PerceptionBackend` (`ruikit/perception/backends/base.py`). It
answers `detect`, `segment_at`, `segment_box`, `lift`, `pose`, and the
pixels and points the planners read (`mask_of`, `box_of`, `points_of`,
`scene_points`). Then:
1. Register it in `BACKENDS` (`ruikit/perception/backends/__init__.py`).
2. Make it selectable: add its name to `PERCEPTION` (`ruikit/interfaces/base.py`)
   and a branch to `open_perception`.

A new detector behind the existing pixel backend is a vision-service model
instead (`ruikit/perception/service/`), chosen after `@` in an id.

## A model channel or request format

A channel is an entry in `ruikit/configs/models.yaml`: its endpoint, the
variable that overrides it, and its deployments.

A new request format subclasses `ModelClient` (`ruikit/models/conversation.py`)
and implements four methods:
- `_ask`: send the history and read the reply back into it;
- `_tool_output`;
- `_text`;
- `_image`.

`conversation()` chooses the subclass for a channel.

## A Harness

Subclass `Harness` (`ruikit/harness/base.py`) and add it to `harnesses()`
and to the results layout's labels (`HARNESSES` in
`ruikit/experiments/results.py`). The class sets:
- `name`, its label in the results;
- `worker_module`, what a batch worker runs;
- `in_process`, which says whether a worker keeps one simulator for many
  episodes or starts a session server per episode;
- `preflight()`, which says why it cannot run here;
- `run(job, out, engine)`, which runs one episode into its cell.

## Outside the library

A tool can live in your own file, and an interface in your own YAML that
extends a named one ([interfaces.md](interfaces.md)). `--plugins` imports the
file before the interface is read, both in the session server and in the batch
runner's workers:

```bash
python -m ruikit.serve.session start --dir D --task DS1 --engine isaac \
    --plugins my/tools.py --profile my/interface.yaml
ruikit run --agent minimal-loop --model openai:<model> --task DS1 --level easy --seeds 0 \
    --plugins my/tools.py --interface my/interface.yaml --out runs/mine
```

In Python, call `ruikit.core.plugins.load(["my/tools.py"])`, then
`Episode(profile="my/interface.yaml")`. Packages can also register tools
through the `ruikit.tools` entry-point group.
