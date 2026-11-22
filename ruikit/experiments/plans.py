"""Experiment plans: rows of agents and conditions, columns of interfaces, expanded into jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ruikit import naming
from ruikit.conditions import DEFAULT_CLOCK, clock_settings
from ruikit.experiments import results as cells

#: the paper's 13 RUIs, as run: each with the aids, RUI-Free with them built in
INTERFACES = tuple(naming.id_of(n) for n in naming.RUIS)


label = cells.profile_label

#: the settings a run is chosen by, besides the experiment, the Harness and the model:
#: the interaction forms, the action abstractions and the perception
FORMS = ("per-action", "program", "sequence", "free")
ACTIONS = ("joint", "delta", "step", "pose", "point", "skill")
PERCEPTIONS = ("pixels", "privileged")


def setting_of(profile: str) -> Tuple[str, str, str]:
    """An interface's interaction form (``per-action``, ``program``, ``free``),
    action abstraction (``joint`` ... ``skill``; none for RUI-Free) and
    perception (``pixels``, or ``privileged`` for the control)."""
    from ruikit.interfaces.catalog import form_of

    name = label(profile)
    form, action = form_of(name.split("@")[0].split("+")[0])
    privileged = "privileged" in name.partition("@")[2]
    return form.replace("_", "-"), action or "", "privileged" if privileged else "pixels"


@dataclass(frozen=True)
class Job:
    model: str
    task: str
    level: str
    seed: int
    profile: str
    #: the framework: ``loop`` (ours), ``codex``, ``claude-code``
    harness: str = "loop"
    #: ``8s`` (the main setting), ``<N>s``, ``paused`` or ``measured`` (``ruikit.conditions``)
    clock: str = DEFAULT_CLOCK

    @property
    def stem(self) -> str:
        """As ``ruikit.env.Episode`` names an episode's files."""
        return cells.stem(self.task, self.level, self.seed, self.profile)

    @property
    def agent(self) -> str:
        return cells.agent_dir(self.harness, self.model, self.clock)

    def folder(self, out: Path) -> Path:
        return Path(out) / self.agent / self.stem

    def tag(self) -> str:
        who = self.model if self.harness == "loop" else f"{self.harness}:{self.model}"
        when = "" if self.clock == DEFAULT_CLOCK else f" [{self.clock}]"
        return f"{who}{when} {self.task} {self.level} s{self.seed} {self.profile}"


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------
def parse_seeds(spec: str, task: str, level: str, data: Optional[dict] = None) -> List[int]:
    """``dev``, ``test`` or ``spare`` (the split's seeds), ``dev:3`` (its first
    three), ``0-4``, ``0,3,7`` (``ruibench.options``)."""
    from ruibench.options import parse_seeds as parse

    return parse(spec, task, level, data)


def _seed_lists(tasks: Sequence[str], levels: Sequence[str], seeds: str) -> Dict[tuple, List[int]]:
    from ruibench.tasks import splits

    data = splits.load() if seeds.partition(":")[0] in ("dev", "test", "spare") else None
    return {(t, lv): parse_seeds(seeds, t, lv, data) for t in tasks for lv in levels}


def plan(models: Sequence[str], tasks: Sequence[str], levels: Sequence[str], seeds: str,
         profiles: Sequence[str], harness: str = "loop",
         clock: str = DEFAULT_CLOCK) -> List[Job]:
    """Seed by seed: every task, level, interface and model once before any is
    run again."""
    clock_settings(clock)                      # a clock that is none: said here
    per = _seed_lists(tasks, levels, seeds)
    depth = max((len(v) for v in per.values()), default=0)
    return [Job(m, t, lv, per[(t, lv)][i], p, harness, clock) for i in range(depth)
            for t in tasks for lv in levels if i < len(per[(t, lv)]) for p in profiles
            for m in models]


def all_tasks() -> List[str]:
    """Every implemented task, by id."""
    from ruibench.tasks.loader import load_all

    return sorted(t.id for t in load_all() if str(t.status.value) == "implemented")


def expand_tasks(names: Sequence[str]) -> List[str]:
    """Task ids, a suite standing for its tasks (``CR``: CR1 ... CR6), ``all``."""
    from ruibench.options import expand_tasks as expand

    return expand(names, all_tasks())


def interfaces(spec, sets: Optional[Mapping[str, Sequence[str]]] = None,
               perception: str = "") -> List[str]:
    """A plan's interfaces: a list of interface names (``program-skill``: the
    paper's RUI, with the aids; ``ruikit.naming``), ids and names of the plan's
    ``interface_sets``, or ``{interfaces: [...], aids: [...]}`` -- each named
    interface with each aids setting (``aids`` all of them, ``-memory`` all but
    one group, ``none`` the interface alone); RUI-Free's aids are built in.  The
    older key ``dialects`` reads as ``interfaces``.  ``perception`` sets every
    interface's perception (``privileged``: the control with the simulator's values)."""
    if isinstance(spec, (list, tuple)):
        out = []
        for p in spec:
            for x in (sets or {}).get(str(p), [p]):
                out.append(naming.id_of(str(x)))
    else:
        out = []
        for d in spec.get("interfaces") or spec.get("dialects") or ():
            name = naming.name_of(str(d))
            base = naming.IDS[name]
            if name == naming.FREE:
                out.append(base)
                continue
            for a in spec.get("aids") or ["aids"]:
                a = str(a).lstrip("+")
                pid = base if a == "none" else base + "+" + (
                    "aids" if a in ("", "aids") else a if a.startswith("aids") else "aids" + a)
                out.append(pid)
    if perception:
        if any("@" in p for p in out):
            raise ValueError(f"perception {perception!r} for interfaces that set their own")
        out = [f"{p}@{perception}" for p in out]
    return list(dict.fromkeys(out))


def channel(harness: str, model: str) -> str:
    """How a row reaches its model -- what decides who runs it (the plan's
    ``owners``): ``codex`` (either mode), ``claude-code``, ``local`` (our
    simple loop over the local server) or ``api`` (our simple loop over a
    model API)."""
    if harness.startswith("codex"):
        return "codex"
    if harness == "claude-code":
        return "claude-code"
    return "local" if model.startswith("local:") else "api"


@dataclass(frozen=True)
class Row:
    """One row of an experiment: a framework and a model, on a clock."""
    harness: str
    model: str
    seeds: str
    clock: str = DEFAULT_CLOCK

    @property
    def agent(self) -> str:
        return cells.agent_dir(self.harness, self.model, self.clock)


@dataclass
class Experiment:
    """One experiment of a plan: rows (framework, model, clock, seeds) x columns."""
    name: str
    title: str
    agents: List[Row]
    profiles: List[str]
    tasks: List[str]
    levels: List[str]
    note: str = ""
    #: who runs a row, by its ``channel``
    owners: Mapping[str, str] = field(default_factory=dict)
    #: film every episode (``video.mp4`` in its folder)
    record: bool = False

    def owner(self, row: Row) -> str:
        return self.owners.get(channel(row.harness, row.model), "")

    def jobs(self) -> List[Job]:
        out = []
        for r in self.agents:
            out += plan([r.model], self.tasks, self.levels, r.seeds, self.profiles, r.harness,
                        r.clock)
        return out

    def cells(self) -> List[tuple]:
        """(agent folder, task, level, seed, interface) of every cell it asks for."""
        return [(j.agent, j.task, j.level, j.seed, label(j.profile)) for j in self.jobs()]


def plan_path(plan: str) -> Path:
    """A plan file, or a plan by its name in ``ruikit/configs/experiments`` (``paper``)."""
    from ruikit.core.paths import CONFIGS

    path = Path(plan)
    if path.suffix in (".yaml", ".yml") or path.exists():
        return path
    named = CONFIGS / "experiments" / f"{plan}.yaml"
    if not named.exists():
        have = ", ".join(sorted(p.stem for p in named.parent.glob("*.yaml")))
        raise FileNotFoundError(f"no plan {plan!r}: neither a file nor one of {have}")
    return named


def load_plan(path: str) -> List[Experiment]:
    """The experiments of a plan, a file or a name (``plan_path``), in YAML::

        defaults: {tasks: all, levels: [easy], seeds: test:1, clock: 8s}
        interface_sets:
          twelve: [joint, delta, ..., program-joint, ..., program-skill]
        experiments:
          main:
            title: the main setting
            agents:
              - {harness: loop, model: "openai:<model>"}
              - {harness: claude-code, model: claude-opus-5-5}
            interfaces: [twelve, rui-free]
          paused:
            agents: [{harness: loop, model: "openai:<model>", clock: paused}]
            interfaces: [twelve]
          privileged:
            agents: [{harness: loop, model: "openai:<model>"}]
            interfaces: [twelve]
            perception: privileged

    (``owners`` -- a channel's rows -> who runs them -- may be added.)

    A row's clock is ``8s`` (the main setting; any ``<N>s``), ``paused`` or
    ``measured``; ``think_seconds: N`` is the older way to say ``Ns``.  ``record:
    true`` films every episode (a video in its folder).  Filming changes the
    pictures an agent is shown, so a plan films all its episodes or none."""
    import yaml

    data = yaml.safe_load(plan_path(path).read_text(encoding="utf-8")) or {}
    defaults = data.get("defaults") or {}
    owners = {str(k): str(v) for k, v in (data.get("owners") or {}).items()}
    sets = {str(k): [str(p) for p in v] for k, v in (data.get("interface_sets") or {}).items()}
    out = []
    for name, spec in (data.get("experiments") or {}).items():
        spec = {**defaults, **(spec or {})}
        tasks = spec.get("tasks", "all")
        tasks = all_tasks() if tasks in ("all", None) else [str(t) for t in tasks]
        seeds = str(spec.get("seeds", "test:1"))
        clock = str(spec.get("clock") or _older_clock(spec))
        agents = []
        for a in spec.get("agents") or ():
            harness = str(a.get("harness", "loop"))
            if harness not in cells.HARNESSES:
                raise ValueError(f"{name}: no framework {harness!r} ({', '.join(cells.HARNESSES)})")
            row = Row(harness, str(a["model"]), str(a.get("seeds", seeds)),
                      str(a.get("clock", clock)))
            clock_settings(row.clock)
            agents.append(row)
        if not agents:
            raise ValueError(f"{name}: no agents")
        out.append(Experiment(name=str(name), title=str(spec.get("title") or name), agents=agents,
                              profiles=interfaces(spec.get("interfaces") or list(INTERFACES), sets,
                                                  str(spec.get("perception") or "")),
                              tasks=tasks, levels=[str(x) for x in spec.get("levels",
                                                                            ["easy", "hard"])],
                              note=str(spec.get("note") or ""), owners=owners,
                              record=bool(spec.get("record", False))))
    return out


def _older_clock(spec: Mapping) -> str:
    think = float(spec.get("think_seconds", 8.0))
    return f"{think:g}s" if think else "measured"


def select(experiments: Sequence[Experiment], names: Sequence[str] = (),
           columns: Sequence[str] = (), agents: Sequence[str] = (),
           owners: Sequence[str] = (), harnesses: Sequence[str] = (),
           models: Sequence[str] = (), forms: Sequence[str] = (), actions: Sequence[str] = (),
           perceptions: Sequence[str] = (), tasks: Sequence[str] = (),
           levels: Sequence[str] = (), clocks: Sequence[str] = ()) -> List[Job]:
    """The jobs of a plan with some settings: experiments (``main``), Harnesses
    (``claude-code``), models (``claude-opus-5-5``), interaction forms
    (``per-action``, ``program``, ``free``), action abstractions (``pose``),
    perceptions (``pixels``, ``privileged``), interfaces (a name -- ``pose``,
    ``program-skill``, ``rui-free``; older ones are read too -- or an id),
    tasks (ids or suites), levels and clocks; none given, every one.
    ``agents`` (``codex:gpt-5.6-sol``) and ``owners`` (the plan's ``owners``, or
    a channel) also choose rows.  A cell asked for twice is one job; the order is
    seed by seed across all of them."""
    want_cols = {naming.name_of(c) if naming.is_name(c) else c for c in columns}
    want_forms = {f.replace("_", "-") for f in forms}
    want_perception = {"pixels" if p == "model" else p for p in perceptions}
    want_owners = {o.lower() for o in owners}
    want_tasks = set(expand_tasks(tasks)) if tasks else set()
    out: Dict[Job, int] = {}
    for e in experiments:
        if names and e.name not in names:
            continue
        rows = [r for r in e.agents
                if (not want_owners or {e.owner(r).lower(), channel(r.harness, r.model)}
                    & want_owners)
                and (not harnesses or r.harness in harnesses)
                and (not models or r.model in models) and (not clocks or r.clock in clocks)]
        profiles = [p for p in e.profiles
                    if (not want_cols or cells.column_of(p) in want_cols or label(p) in want_cols)
                    and (not want_forms or setting_of(p)[0] in want_forms)
                    and (not actions or setting_of(p)[1] in actions)
                    and (not want_perception or setting_of(p)[2] in want_perception)]
        e_tasks = [t for t in e.tasks if not want_tasks or t in want_tasks]
        e_levels = [lv for lv in e.levels if not levels or lv in levels]
        rank: Dict[tuple, int] = {}
        for r in rows:
            for j in plan([r.model], e_tasks, e_levels, r.seeds, profiles, r.harness, r.clock):
                if agents and not {f"{j.harness}:{j.model}", j.agent,
                                   cells.agent_dir(j.harness, j.model), j.model} & set(agents):
                    continue
                k = (j.harness, j.model, j.clock, j.task, j.level, j.profile)
                rank[k] = rank.get(k, -1) + 1
                out.setdefault(j, rank[k])
    return [j for j, _ in sorted(out.items(), key=lambda kv: kv[1])]


def describe(jobs: Sequence[Job]) -> List[str]:
    """One line per agent of what ``jobs`` hold: episodes, interfaces, tasks,
    levels and seeds."""
    by: Dict[str, List[Job]] = {}
    for j in jobs:
        by.setdefault(j.agent, []).append(j)
    def n(count: int, what: str) -> str:
        return f"{count} {what}" + ("s" if count != 1 else "")

    out = []
    for agent, js in by.items():
        out.append(f"{agent}: {n(len(js), 'episode')} -- {n(len({j.profile for j in js}), 'interface')}, "
                   f"{n(len({j.task for j in js}), 'task')}, levels "
                   f"{'/'.join(sorted({j.level for j in js}))}, seeds "
                   f"{','.join(str(x) for x in sorted({j.seed for j in js}))}")
    return out


def chunks(jobs: Sequence[Job], size: int) -> List[List[Job]]:
    """Up to ``size`` jobs of one task and level each (a worker builds one scene
    structure), in the order of their first job.  A ready-made agent's job is a
    chunk of its own (its session starts its own simulator)."""
    by: Dict[tuple, List[List[Job]]] = {}
    out: List[List[Job]] = []
    for j in jobs:
        if j.harness != "loop":
            out.append([j])
            continue
        lists = by.setdefault((j.task, j.level), [])
        if not lists or len(lists[-1]) >= size:
            lists.append([])
            out.append(lists[-1])
        lists[-1].append(j)
    return out
