"""Where a result comes from: the code, the frozen version, the interpreter, and
the episode's place in its simulator process.

Results compare only when they ran on the same frozen version of the
benchmark, each episode in a fresh simulator: a reused Isaac stage keeps the
previous episode's physics layout and renderer state (``machine.PER_WORKER``).
Every result records both in its ``provenance``, and the report lists every
result that is not comparable, or whose provenance is missing.
"""

from __future__ import annotations

import platform
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from ruikit.core.paths import ROOT

UNKNOWN = "unknown"
#: a frozen version that changed only some tasks from the one before: a result
#: of the earlier version still compares with the later one's, but on the tasks
#: changed.  (earlier, later) -> those tasks; along a chain of such versions the
#: tasks add up (``changed_between``)
CHANGED_ONLY = {("v2.3", "v2.3.1"): frozenset({"CL3", "WK1", "DI4"}),
                # which episodes count and how a retry starts; no task
                ("v2.3.1", "v2.3.2"): frozenset(),
                # CR2's bulb judged in its socket by its cap, not its middle
                ("v2.3.2", "v2.3.3"): frozenset({"CR2"}),
                # WK4 judged by side and order, not by the centimetre; CL5's
                # silhouette filled by either of its tilings, a piece a little
                # further off its place; WK5's instruction says to use everything
                # on the tray; WK6 no foul once the game is over; CR3 a grip only
                # with the fingers round the stone; DI5's moles and DI6's table
                # start 3 T_ref in, and a DI5 strike counts from wherever over the
                # hole it came; CL6's knight on a square within 2.2 cm of its
                # middle; and the X5's hands, which start at pitch 90 degrees,
                # read alike however they sag (CR6, WK4, DI1)
                ("v2.3.3", "v2.3.4"): frozenset({"WK4", "CL5", "WK5", "WK6", "CR3", "DI5", "DI6",
                                                 "CL6", "CR6", "DI1"}),
                # no task: the minimal loop's changes touch some episodes only
                # (EPISODE_CHANGES), and a full context ended an episode with no result
                ("v2.3.4", "v2.3.5"): frozenset(),
                # no task, no episode: how a session's ended episode is finished and
                # counted, and a login that fails stops the runner like a usage limit
                ("v2.3.5", "v2.3.6"): frozenset(),
                ("v2.3.6", "v2.3.7"): frozenset(),
                # no task: a mark riding on a body (a light, a label) is kinematic and
                # never collides on Isaac; before, DS3's detector lights gathered free-fall
                # speed and shot away any shell they touched -- those episodes are touched
                # (EPISODE_CHANGES), the rest of DS3 went as it will go
                ("v2.3.7", "v2.3.8"): frozenset(),
                # DS3: the detector's coil and shaft collide (drawn only until then)
                ("v2.3.8", "v2.3.9"): frozenset({"DS3"}),
                # no task, no episode: a workspace's spend cap stops the runner
                ("v2.3.9", "v2.3.10"): frozenset(),
                # no task: Claude Code reasons at medium effort
                ("v2.3.10", "v2.3.11"): frozenset(),
                # no task: Claude Code and Codex both reason at high effort.  A
                # result's effort is compared with its harness's now (not_comparable)
                ("v2.3.11", "v2.3.12"): frozenset(),
                # DS3, DS4, DS5: the hidden answer found out before it is acted on
                # (KnownBefore), in their scoring alone (SCORED_AGAIN)
                ("v2.3.12", "v2.3.13"): frozenset({"DS3", "DS4", "DS5"}),
                # no task, no episode, no harness in simulation: the real station
                # (ruibench.engines.franka) -- an engine, its robot, its tasks -- and
                # what an episode, perception and a ready-made agent do where an
                # engine is real; in simulation each does what it did
                ("v2.3.13", "v2.4.0"): frozenset(),
                # no task: QwenCloud's text arguments read by the schema
                # (EPISODE_CHANGES, typed_as_text)
                ("v2.4.0", "v2.4.1"): frozenset(),
                # no task: a minimal-loop reply's pictures, each one seen
                # (EPISODE_CHANGES, pictures_unseen)
                ("v2.4.1", "v2.4.2"): frozenset(),
                # no task, no interface of before: a new interaction form, the sequence
                # (rui.seq-*: a program that only acts), beside the others
                ("v2.4.2", "v2.4.3"): frozenset(),
                # no task, no episode, no harness in simulation: an episode tells a real
                # engine of each call, for the operator; the rest is the real station's
                ("v2.4.3", "v2.4.4"): frozenset(),
                # no task: the skills pick from the side and tip the hand
                # (TOOLS_CHANGED), which CR4 and DS1's hard level need
                ("v2.4.4", "v2.5.0"): frozenset(),
                # no task, no episode, no harness in simulation: on a real robot a toy's clock
                # has its say before each of an agent's calls, and what follows a failed move
                # does not run; the rest is the real station's
                ("v2.5.0", "v2.5.1"): frozenset(),
                # no task, no interface of before: the skills' completion check, attached
                # as +check or +check:correct only (ruikit.actions.checks)
                ("v2.5.1", "v2.5.2"): frozenset(),
                # no task, no interface of before: +check's place check looks cell by cell
                # (v2.5.2's +check episodes were set aside and run again)
                ("v2.5.2", "v2.5.3"): frozenset()}
#: of a version's changed tasks, those changed only in how they are scored --
#: not in what their agents are told or shown, nor in what their reactors do:
#: a result of the earlier version compares once it is scored again with the
#: later code (``ruikit.experiments.rescore``)
SCORED_AGAIN = {("v2.3.3", "v2.3.4"): frozenset({"CL5", "WK6", "CR3"}),
                ("v2.3.12", "v2.3.13"): frozenset({"DS3", "DS4", "DS5"})}
#: a version's changes that touch only some episodes, whatever their task: (the
#: version, the key ``rescore`` marks an earlier result with, why it does not
#: compare).  v2.3.4: a hand within 0.6 degrees of pitch +-90 reads roll 0
#: (``ruikit.env.station.GIMBAL_LOCK``); v2.3.5: on the fixed clock a minimal-loop
#: reply without a tool call costs its think time (``EpisodeClock.idle_reply``);
#: v2.4.1: a QwenCloud argument that came as text where its parameter takes a
#: list, a number or a boolean is read as the JSON it spells
#: (``chat_completions.typed_arguments``); v2.4.2: a minimal-loop reply's
#: pictures are each sent once, however many (``ModelClient._prune``)
EPISODE_CHANGES = (
    ("v2.3.4", "read_otherwise",
     "its hand was read at pitch +-90 degrees, which v2.3.4 reads otherwise"),
    ("v2.3.5", "replied_without_a_call",
     "a reply without a tool call, which costs its think time from v2.3.5"),
    ("v2.3.8", "shot_by_a_mark",
     "a body shot away by a mark riding on another, which v2.3.8's marks cannot do"),
    ("v2.4.1", "typed_as_text",
     "a QwenCloud call refused for a list or number that came as text, which v2.4.1 reads"),
    ("v2.4.2", "pictures_unseen",
     "a minimal-loop reply's pictures dropped before the model saw them, which v2.4.2 shows"),
)


#: a version's changes to how one harness runs its agents: every earlier result
#: of that harness does not compare, whatever its task.  v2.3.4: Claude Code read
#: its tools' descriptions cut at 4 KB and lost the pictures of a failed call.
#: A change of reasoning effort is not one: each result records its effort, which
#: is compared with the harness's own now (``effort_now``)
HARNESS_CHANGED = {("v2.3.3", "v2.3.4"): frozenset({"claude-code"})}
#: a version's changes to some tools -- what they do, take or say: every earlier
#: result on an interface that offers one of them, to the agent or to its
#: programs, does not compare, whatever its task.  v2.5.0: skill_pick picks from
#: the side, skill_turn tips the hand, and skill_move says it turns a level hand
#: back down -- the skills, the programs over them, and RUI-Free
TOOLS_CHANGED = {("v2.4.4", "v2.5.0"): frozenset({"skill_pick", "skill_turn", "skill_move"})}
#: the harnesses that run a model at a set reasoning effort
EFFORT_HARNESSES = ("claude-code", "codex", "codex-code")


@lru_cache(maxsize=None)
def effort_now(harness: str, model: str) -> Optional[str]:
    """The reasoning effort ``harness`` runs ``model`` at in this checkout."""
    if harness not in EFFORT_HARNESSES:
        return None
    try:
        from ruikit.harness.session_agent import effort_of
    except Exception:
        return None
    return effort_of(harness, model)


def harnesses_changed(earlier: str, later: str) -> frozenset:
    """The harnesses changed after ``earlier``, up to and with ``later``."""
    out, at = set(), earlier
    for version in versions_after(earlier, later) or ():
        out |= HARNESS_CHANGED.get((at, version), frozenset())
        at = version
    return frozenset(out)


def tools_changed(earlier: str, later: str) -> frozenset:
    """The tools changed after ``earlier``, up to and with ``later``."""
    out, at = set(), earlier
    for version in versions_after(earlier, later) or ():
        out |= TOOLS_CHANGED.get((at, version), frozenset())
        at = version
    return frozenset(out)


@lru_cache(maxsize=None)
def offered(profile: str) -> frozenset:
    """Every tool an interface offers, to the agent and to its programs; empty
    for an interface this checkout cannot load."""
    try:
        from ruikit.interfaces.catalog import load_profile

        prof = load_profile(profile)
    except Exception:
        return frozenset()
    return frozenset(prof.tools) | frozenset(prof.code_tools)


def versions_after(earlier: str, later: str) -> Optional[List[str]]:
    """The versions after ``earlier``, up to and with ``later``, along
    ``CHANGED_ONLY``; None when there is no such chain."""
    step = {a: b for (a, b) in CHANGED_ONLY}
    out, at = [], earlier
    while at != later:
        if at not in step:
            return None
        at = step[at]
        out.append(at)
    return out


def changed_between(earlier: str, later: str) -> Optional[frozenset]:
    """The tasks changed from ``earlier`` to ``later`` along ``CHANGED_ONLY``;
    None when there is no such chain (every task may differ)."""
    step = {a: (b, tasks) for (a, b), tasks in CHANGED_ONLY.items()}
    changed, at = frozenset(), earlier
    while at != later:
        if at not in step:
            return None
        at, tasks = step[at][0], changed | step[at][1]
        changed = tasks
    return changed


def scored_again_between(earlier: str, later: str) -> frozenset:
    """The tasks changed from ``earlier`` to ``later`` only in how they are
    scored, at every step that changed them (``SCORED_AGAIN``)."""
    step = {a: (b, tasks) for (a, b), tasks in CHANGED_ONLY.items()}
    good, bad, at = set(), set(), earlier
    while at != later:
        if at not in step:
            return frozenset()
        b, tasks = step[at]
        again = SCORED_AGAIN.get((at, b), frozenset())
        good |= tasks & again
        bad |= tasks - again
        at = b
    return frozenset(good - bad)


#: what a result's ``provenance`` holds
FIELDS = ("ruikit", "frozen", "freeze_clean", "commit", "process_episode", "per_worker",
          "python", "python_version", "numpy")


@lru_cache(maxsize=None)
def launch() -> dict:
    """What this process runs, taken once: a worker calls it before its first
    episode, so a checkout edited later does not answer for it."""
    import ruikit

    facts = {"ruikit": getattr(ruikit, "__version__", UNKNOWN)}
    try:
        from ruibench.tasks import freeze

        record = freeze.load() or {}
        facts["frozen"] = record.get("name") or "unfrozen"
        facts["freeze_clean"] = not freeze.drift()      # as ``ruibench freeze --check``
    except Exception:                                   # an installed package: no checkout
        facts["frozen"], facts["freeze_clean"] = UNKNOWN, None
    facts["commit"] = commit(ROOT)
    facts["python"] = sys.executable
    facts["python_version"] = platform.python_version()
    try:
        import numpy

        facts["numpy"] = numpy.__version__
    except ImportError:
        facts["numpy"] = UNKNOWN
    return facts


def commit(root: Path) -> str:
    """The checkout's commit, read from ``.git`` itself (``git`` refuses a
    checkout another user owns), or ``unknown``."""
    git = Path(root) / ".git"
    try:
        if git.is_file():                               # a worktree: "gitdir: <path>"
            git = (Path(root) / git.read_text().split(":", 1)[1].strip()).resolve()
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head                                 # a detached head
        ref = head[len("ref: "):]
        common = git
        if (git / "commondir").exists():
            common = (git / (git / "commondir").read_text().strip()).resolve()
        for base in (git, common):
            if (base / ref).is_file():
                return (base / ref).read_text().strip()
        for line in (common / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except (OSError, IndexError):
        pass
    return UNKNOWN


def stamp(index: int, per_worker: Optional[int]) -> dict:
    """A result's ``provenance``: the process's facts, the episode's place in
    its simulator process (1: a fresh one) and the episodes a process was given."""
    return {**launch(), "process_episode": index, "per_worker": per_worker}


def describe(per_worker: int) -> str:
    """One line for a runner to start with: what its workers will run."""
    f = launch()
    clean = {True: "clean", False: "the checkout differs from it: ruibench freeze --check",
             None: "not checked"}[f["freeze_clean"]]
    line = (f"code: RUIKit {f['ruikit']} on frozen {f['frozen']} ({clean}), commit "
            f"{f['commit'][:12]}, python {f['python_version']} with numpy {f['numpy']}")
    if per_worker > 1:
        line += (f"; up to {per_worker} episodes per simulator: those after a worker's first "
                 "are not comparable")
    return line


def expected_version() -> Optional[str]:
    """The frozen version results should be on: this checkout's."""
    try:
        from ruibench.tasks import freeze

        return (freeze.load() or {}).get("name")
    except Exception:
        return None


def not_comparable(row: dict, expected: Optional[str]) -> List[str]:
    """Why a result does not compare with the others ([] when it does)."""
    p = row.get("provenance")
    if not isinstance(p, dict):
        return ["no provenance"]
    missing = [f for f in FIELDS if f not in p]
    why = [", ".join(missing) + " unknown"] if missing else []
    frozen = p.get("frozen")
    if frozen not in (None, UNKNOWN) and expected and frozen != expected:
        changed = changed_between(frozen, expected)
        again = ((row.get("scored_again") or {}).get("frozen") == expected
                 and row.get("task") in scored_again_between(frozen, expected))
        if changed is None or (row.get("task") in changed and not again):
            why.append(f"frozen version {frozen}, not {expected}")
        elif row.get("harness") in harnesses_changed(frozen, expected):
            why.append(f"frozen version {frozen}: its harness ({row['harness']}) changed since")
        elif tools_changed(frozen, expected) & offered(str(row.get("profile") or "")):
            gone = sorted(tools_changed(frozen, expected) & offered(str(row["profile"])))
            why.append(f"frozen version {frozen}: its interface's {', '.join(gone)} changed since")
        else:
            why += [because for _, key, because in EPISODE_CHANGES if row.get(key)]
    elif frozen == UNKNOWN:
        why.append("frozen version unknown")
    effort, now = row.get("effort"), effort_now(str(row.get("harness")), str(row.get("model")))
    if effort and now and effort != now:
        why.append(f"reasoning effort {effort}, not {now}")
    if p.get("freeze_clean") is False:
        why.append(f"the checkout differed from {frozen}")
    elif "freeze_clean" in p and p["freeze_clean"] is None:
        why.append("freeze check unknown")
    index = p.get("process_episode")
    if isinstance(index, int) and index > 1:
        why.append(f"episode {index} of its simulator process")
    elif "process_episode" in p and not isinstance(index, int):
        why.append("place in its simulator process unknown")
    return why
