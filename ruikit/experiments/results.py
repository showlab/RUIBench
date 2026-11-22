"""Where the experiments' results live, and what one result says.

Every runner writes one folder per episode, in one layout shared by
everyone's runs, so the tables are built from all of them at once:

    OUT/<agent>/<task>_<level>_s<seed>_<interface>/
        result.json      the result: success, the milestones reached, replies, the settings
        lost.json        the harness could not run it (never counted)
        api_error.json   the model's API, or the agent's usage limit, ended it: run it again
        *.jsonl          the episode's own logs (``ruikit.env.Episode``)

``<agent>`` is the framework and the model: ``loop-openai-<model>`` (our simple
loop, ``ruikit.harness.minimal_loop``), ``codex-gpt-5.6-sol``, ``claude-code-opus``
(``ruikit.harness.session_agent``) -- with the clock after an ``@`` when it is not the
main setting's 8 s a reply: ``loop-openai-<model>@paused``.  A cell -- an agent, a
clock, a task, a level, a seed, an interface -- is run once, for whichever
experiment asks for it first.

Each result records what it was run with -- the benchmark's version, the
framework, the model, the interface, its aids and perception, the clock --
and, for a failure, how far it got: the milestones it reached, so a failure
can be told apart as thought wrong (the wrong doll taken) or thought right
and not done (the right doll, not set down).  Whether the agent itself said
it was done (``declared_done``) stands beside the verdict: an episode that
says so and fails is a false claim, the cost of having no success check.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from ruikit.conditions import DEFAULT_CLOCK, clock_seconds
from ruikit.naming import FREE, RUIS, name_of

#: the frameworks a cell may be run by: our simple loop, and the ready-made agents
#: -- Codex calling each tool by itself (``codex``, as the loop's model does) or
#: in its own code mode, from JavaScript it writes (``codex-code``: Codex as
#: shipped; a loop of calls there is a program, so only for RUI-Free)
HARNESSES = ("loop", "codex", "codex-code", "claude-code")
#: the main table's columns, in its order: the 13 RUIs by the paper's names --
#: each action abstraction per action, then as a program, then RUI-Free
#: (``ruikit.naming``)
COLUMNS = RUIS


def safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._@+-]+", "-", text)


def agent_dir(harness: str, model: str, clock: str = DEFAULT_CLOCK) -> str:
    """The folder of one framework, model and clock: ``loop-openai-<model>``,
    ``loop-openai-<model>@paused``."""
    return safe(f"{harness}-{model}" + ("" if clock == DEFAULT_CLOCK else f"@{clock}"))


def stem(task: str, level: str, seed: int, profile: str) -> str:
    """As ``ruikit.env.Episode`` names an episode's files."""
    return f"{task}_{level}_s{seed}_{profile_label(profile).replace('/', '-')}"


@lru_cache(maxsize=None)
def profile_label(profile: str) -> str:
    """A profile's name in folders and tables: its id -- for a profile given as
    a YAML file (one from outside the library), the id inside it, else the
    file's name."""
    if not profile.endswith((".yaml", ".yml")):
        return profile
    try:
        import yaml

        data = yaml.safe_load(Path(profile).read_text(encoding="utf-8")) or {}
        return str(data.get("id") or Path(profile).stem)
    except (OSError, ValueError, ImportError):
        return Path(profile).stem


@lru_cache(maxsize=None)
def column_of(profile: str) -> str:
    """The main table's column of an interface: its name as the paper gives it
    (``skill``, ``program-skill``, ``rui-free``, ``sequence-delta``;
    ``ruikit.naming``), else its own id."""
    base = profile_label(profile).split("@")[0].split("+")[0]
    try:
        return name_of(base)
    except ValueError:
        return base


def normal_column(column: Optional[str], profile: str = "") -> str:
    """A stored result's column by today's names: results written before carry
    the older ones (``code-skill``, ``program``, ``free``)."""
    if column:
        try:
            return name_of(column)
        except ValueError:
            return column
    return column_of(profile) if profile else ""


def aids_of(profile: str) -> str:
    """Which aids an interface carries: ``aids``, ``aids-<group>`` or ``none``
    -- or, with other components or extras attached, all its tokens."""
    if column_of(profile) == FREE:
        return "aids"                          # built in
    tokens = profile_label(profile).split("@")[0].split("+")[1:]
    aid = [t for t in tokens if t == "aids" or t.startswith("aids-")]
    if aid and len(tokens) == 1:
        return aid[0]
    return "+".join(tokens) if tokens else "none"


def base_of(profile: str) -> str:
    """The interface with its aids taken off: what the aids ablation pairs results by."""
    label = profile_label(profile)
    spec, at, perception = label.partition("@")
    tokens = [t for t in spec.split("+")[1:] if not (t == "aids" or t.startswith("aids-"))]
    return "+".join([spec.split("+")[0]] + tokens) + (f"@{perception}" if at else "")


@lru_cache(maxsize=None)
def perception_of(profile: str) -> dict:
    try:
        from ruikit.interfaces import load_profile

        return load_profile(profile).perception_settings()
    except Exception:
        return {}


@lru_cache(maxsize=None)
def version() -> str:
    """The benchmark version the code is: the frozen name, with ``+changes``
    when a fingerprinted file differs from it."""
    try:
        from ruibench.tasks import freeze

        record = freeze.load() or {}
        name = record.get("name") or "unfrozen"
        return name + ("+changes" if freeze.drift() else "")
    except Exception as exc:                        # an installed package: no checkout
        return f"unknown ({type(exc).__name__})"


@lru_cache(maxsize=None)
def suites() -> Dict[str, str]:
    """Task id -> its suite (``knowledge`` ... ``contact``): the five kinds of task."""
    from ruibench.tasks.loader import load_all

    return {t.id: t.suite for t in load_all()}


#: an object carried along: risen this far above where it started, not held,
#: while something else was held -- on top of a held object, wedged to it, hooked
CARRY_RISE, CARRY_SECONDS = 0.03, 0.3


def carried_along(trajectory, fastened: Sequence[str] = ()) -> List[str]:
    """The objects that rode with a held one: risen ``CARRY_RISE`` above where they
    were when that hold began, for ``CARRY_SECONDS``, not held themselves -- one
    move that carried two things, which a rule counting held objects misses.
    Worth a look at the video; a lever does it legitimately.  ``fastened``:
    bodies attached to another (a name card on a present), which ride by design.
    (v2.3.2: measured from the episode's start, every block already stacked and
    every doll already on the sofa counted while the next was carried.)"""
    if not trajectory:
        return []
    out: List[str] = []
    since: Dict[str, float] = {}
    base: Dict[str, float] = {}
    holding: frozenset = frozenset()
    for w in trajectory:
        objects = w.get("objects", {})
        held = frozenset(n for n, o in objects.items() if o.get("held_by"))
        if held != holding:                    # a hold begins or ends: heights from here
            base = {n: o["pos"][2] for n, o in objects.items()}
            since, holding = {}, held
        if not held:
            continue
        t = float(w.get("t", 0.0))
        for n, o in objects.items():
            if n in held or n in fastened:
                continue
            if o["pos"][2] - base.get(n, o["pos"][2]) > CARRY_RISE:
                since.setdefault(n, t)
                if t - since[n] >= CARRY_SECONDS and n not in out:
                    out.append(n)
            else:
                since.pop(n, None)
    return out


def fastened_bodies(log: Mapping) -> List[str]:
    """The bodies of an episode's scene that ride with another by design: one
    attached to another (the log's scene says), and -- from the task as built
    here -- one no hand may take, or one a joint moves (CR5's sprung seats)."""
    out = {b["name"] for b in (log.get("scene") or {}).get("bodies") or () if b.get("attached_to")}
    out |= _mechanism(log.get("task_id"), log.get("level"), int(log.get("seed") or 0))
    return sorted(out)


@lru_cache(maxsize=None)
def _mechanism(task, level, seed: int) -> frozenset:
    try:
        from ruibench.tasks.instance import build

        scene = build(task, level, seed).scene
    except Exception:                         # a task this checkout does not have
        return frozenset()
    moved = {j.parent for j in scene.articulations if j.parent}
    return frozenset(b.name for b in scene.bodies
                     if b.attrs.get("graspable") is False or b.name in moved)


def episode_facts(folder: Path) -> dict:
    """What the episode's own log says of how it went: the scorer's verdict and
    the milestones reached, in the order they were reached."""
    logs = sorted(Path(folder).glob("*.episode.jsonl"))
    if not logs:
        return {}
    try:
        log = json.loads(logs[0].read_text(encoding="utf-8").splitlines()[0])
    except (OSError, ValueError, IndexError):
        return {}
    score = log.get("score") or {}
    hit = log.get("milestones_hit") or score.get("milestones_hit") or {}
    order = [name for name, _ in sorted(hit.items(), key=lambda kv: kv[1])]
    metrics = log.get("metrics") or {}
    return {"success": bool(score.get("success")), "failure": score.get("failure"),
            "progress": score.get("progress"), "milestones": order,
            "carried": carried_along(log.get("trajectory"), fastened_bodies(log)),
            "last_milestone": order[-1] if order else None,
            "replies": metrics.get("replies"), "sim_seconds": metrics.get("sim_seconds"),
            "tool_calls": metrics.get("tool_calls"), "clock_detail": log.get("clock"),
            "interface": log.get("interface"), "engine_class": log.get("engine")}


@lru_cache(maxsize=None)
def milestone_order(task: str, level: str, seed: int) -> tuple:
    """A task's milestones in the order its scorer gives them -- the first
    step to the last -- or () where the task cannot be built here."""
    try:
        from ruibench.tasks.instance import build

        return tuple(m.name for m in build(task, level=level, seed=seed).milestones)
    except Exception:
        return ()


def result_row(*, harness: str, model: str, task: str, level: str, seed: int, profile: str,
               folder: Path, engine: str, clock: str = DEFAULT_CLOCK, **extra) -> dict:
    """``result.json`` of one cell: its settings, then what happened."""
    row = {"harness": harness, "model": model, "agent": agent_dir(harness, model, clock),
           "clock": clock, "task": task, "suite": suites().get(task, ""), "level": level,
           "seed": seed, "profile": profile_label(profile), "column": column_of(profile),
           "aids": aids_of(profile), "perception": perception_of(profile),
           "think_seconds": clock_seconds(clock), "engine": engine, "version": version()}
    facts = episode_facts(folder)
    row.update({k: v for k, v in facts.items() if v is not None})
    row.update({k: v for k, v in extra.items() if v is not None})
    # the agent said it was done (task_done): the verdict then is the episode's
    row["declared_done"] = str(row.get("reason") or "").startswith("agent_done")
    hit = set(row.get("milestones") or ())
    order = milestone_order(task, level, int(seed))
    row["furthest_milestone"] = (next((m for m in reversed(order) if m in hit), None) if order
                                 else row.get("last_milestone"))
    return row


#: a person's look at an episode the physics audit flagged (``OUT/reviews.json``):
#: the scorer counts a flagged episode a failure until someone has looked at it
REVIEWS = "reviews.json"


def load_reviews(out: Path) -> Dict[str, dict]:
    """``OUT/reviews.json``: an episode's folder, relative to OUT, -> what its
    reviewer decided (``ruikit.experiments.review``): for an episode the physics
    audit flagged, ``cleared`` (the fault changed nothing: it counts as the rest
    of its score says, ``success`` scored with the audit off) or ``upheld`` (it
    stays a failure); for a success to review, ``confirmed``, ``unclear`` or
    ``hack`` (it got round the task: a failure)."""
    try:
        data = json.loads((Path(out) / REVIEWS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict) and not k.startswith("_")}


def load_results(out: Path) -> List[dict]:
    """Every ``result.json`` under ``out`` (any depth of agent folders), with
    the reviews of flagged episodes applied (``load_reviews``)."""
    rows = []
    reviews = load_reviews(out)
    for p in sorted(Path(out).glob("*/*/result.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        r.setdefault("folder", str(p.parent))
        r["column"] = normal_column(r.get("column"), r.get("profile") or "")
        review = reviews.get(p.parent.relative_to(out).as_posix())
        if review:
            r["review"] = review
            if r.get("failure") == "physics_flag" and review.get("decision") == "cleared":
                r["success"] = bool(review.get("success"))
                r["failure"] = "none" if r["success"] else "violation"
            elif review.get("decision") == "hack":       # a success got round the task
                r["success"], r["failure"] = False, "hack"
        rows.append(r)
    return rows


def write_cells(out: Path, rows: Optional[List[dict]] = None) -> Path:
    """``OUT/cells.jsonl``: every result, one line each (rebuilt from the folders,
    so runners on several machines never write one file at once)."""
    rows = load_results(out) if rows is None else rows
    path = Path(out) / "cells.jsonl"
    tmp = path.with_name(".cells.jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)
    return path
