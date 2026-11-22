"""Episodes on the real station, one at a time, with an operator at the robot.

    python -m ruikit.experiments.real --plan real --dry-run               # what it would run
    python -m ruikit.experiments.real --plan real --harness claude-code   # run them
    python -m ruikit.experiments.real --plan real --interfaces pose free --tasks R0
    python -m ruikit.experiments.real --plan real --tasks DS2 --seed 16   # another table
    python -m ruikit.experiments.real --plan real --tasks DS2 --test      # a try, which does not count

The batch runner (``ruikit.experiments.batch``) runs simulated episodes
unattended; the real robot's need a person.  For each episode of the plan's
rows -- a ready-made agent on an interface, on a real task:

1. **before** -- the arm goes to its start, and the operator lays the table out
   as the task's card says (``ruibench/tasks/real/tasks.yaml``), then says go,
   or skips the episode, or stops;
2. **the episode** runs as a ready-made agent's runs in simulation
   (``ruikit.harness.session_agent``): a session on the real engine, a fresh
   agent with a clean context reaching it over MCP, filmed, into the shared
   results layout under ``--out``.  The operator follows it here as it happens:
   each of the agent's calls and its answer, a program's steps among them, and
   what the agent says;
3. **after** -- the operator, who watched it, judges it by the card's success:
   nothing else can (``ruibench.tasks.real.OperatorVerdict``).  The verdict is
   ``result.json``'s ``success``, with ``judged_by: operator`` and the
   operator's note; ``scored_success`` keeps what the episode scored itself
   (never a success on a real task);
4. **an episode that does not count** -- the agent cut off by its account, a
   rule broken, the robot faulted -- is offered to the operator to run again.

F in the operator's window fails the episode there and then: the arm stops, the
episode ends, it counts as a failure, and after a word on why the next one starts.

Before the first episode the station is checked (``ruibench.engines.franka.check``:
the robot server answers, the cameras deliver, the station is calibrated), the
vision service must answer, and each agent passes its self-test, as on Isaac.
It resumes: a cell that has a verdict is not run again, and one that ran without
being judged (the runner was stopped) is judged from its video.

A seed is one table: every interface of a seed has it laid out alike, and another
table is another seed (``--seed``, each a line of the task's card), with cells of
its own.  ``--test`` is the operator's try before a suite's run: the pose alone,
apart from the results and from what the next run counts, as often as wanted.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ruibench.engines.franka.terminal import alarm, heading, notice, question
from ruikit.core.paths import ROOT
from ruikit import naming
from ruikit.experiments import plans
from ruikit.experiments import results as cells

ENGINE = "franka"
#: where the real-robot results go: apart from the simulation's
DEFAULT_OUT = ROOT / "runs" / "real"
#: where the operator's tries go, each in a folder of its own: they count for nothing
TEST_OUT = ROOT / "runs" / "real-test"
TEST_INTERFACE = "pose"
#: how often a running episode's files are looked at, and how much of what the agent
#: says is shown at once
FOLLOW_SECONDS = 0.3
SAYS_CHARS = 500
#: F in the operator's window leaves this in the episode's folder, and how long the session
#: has to finish what it is doing and end the episode when told so
FAILED_BY_KEY = "failed_by_operator"
END_IT_SECONDS = 120.0


def todo(jobs: List[plans.Job], out: Path) -> List[plans.Job]:
    """The jobs without a judged result yet."""
    return [j for j in jobs if "judged_by" not in _result(j.folder(out))]


def _result(folder: Path) -> dict:
    try:
        return json.loads((folder / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def in_order_asked(jobs: List[plans.Job], interfaces: List[str]) -> List[plans.Job]:
    """A task's episodes together, and an agent's among them, as the plan has them: the
    operator lays out one table at a time.  Within those, in the order their interfaces
    were named: the operator's, not the plan's.  None named, the plan's."""
    asked = [naming.name_of(i) if naming.is_name(i) else i for i in interfaces]
    tasks = list(dict.fromkeys(j.task for j in jobs))
    agents = list(dict.fromkeys(j.agent for j in jobs))
    return sorted(jobs, key=lambda j: (tasks.index(j.task), agents.index(j.agent), min(
        (n for n, name in enumerate(asked)
         if name in (cells.column_of(j.profile), cells.profile_label(j.profile))), default=0)))


def preflight(jobs: List[plans.Job], out: Path, provisional_ok: bool, say=print) -> List[str]:
    """What keeps these jobs from running here; [] when nothing does."""
    from ruibench.engines.franka import FAKE_ENV
    from ruibench.engines.franka import calibration, check
    from ruikit.harness.base import harnesses
    from ruikit.harness.selftest import ensure

    fake = os.environ.get(FAKE_ENV, "") == "1"
    why: List[str] = []
    if not fake:
        found = check.problems()
        if provisional_ok and calibration.load().provisional:
            found = [p for p in found if "not calibrated" not in p]
        why += found
    why += vision_problems(jobs)
    for h, m in sorted({(j.harness, j.model) for j in jobs}):
        reason = harnesses()[h].preflight() or (None if why else
                                                ensure(h, m, out / "selftest", say=say))
        if reason:
            why.append(f"{h} {m}: {reason}")
    return why


def vision_problems(jobs: List[plans.Job]) -> List[str]:
    """The vision service each job's perception asks for must answer."""
    from ruikit.extras.backends import VISION, get_backend
    from ruikit.interfaces import load_profile

    out = []
    for name in sorted({load_profile(j.profile).perception_settings()["backend"] for j in jobs}):
        try:
            get_backend(VISION, "vision", name).info()
        except Exception as exc:
            out.append(f"the vision service for {name!r} does not answer ({exc}): start it -- "
                       "python -m ruikit.perception.service.server (docs/REAL_ROBOT.md)")
    return out


def ask(asked: str, choices: str, default: str = "") -> str:
    """One letter from the operator, of ``choices``; Enter is ``default``, or --
    with none -- asks again.  No terminal left: 'q' where that stops, else stop."""
    while True:
        try:
            got = input(question(asked)).strip().lower()[:1] or default
        except EOFError:
            if "q" in choices:
                return "q"
            raise SystemExit("the operator's terminal closed: stopping, the episode unjudged")
        if got and got in choices:
            return got


def agent_says(line: str) -> str:
    """What a line of an agent's record (``agent.jsonl``) says in words, if it does:
    Claude Code's stream or Codex's events."""
    try:
        record = json.loads(line)
    except ValueError:
        return ""
    if not isinstance(record, dict):
        return ""
    if record.get("type") == "assistant":
        blocks = (record.get("message") or {}).get("content") or ()
        return " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    item = record.get("item") or {}
    if record.get("type") == "item.completed" and item.get("type") == "agent_message":
        return str(item.get("text") or "")
    return ""


class Follow:
    """A running episode, for the operator, as it happens: the session's account of its
    calls (``engines/franka/steps.py``, in its ``server.log``) and what the agent says
    (its record).  The files may not be there yet, and a new try starts them again."""

    def __init__(self, folder: Path, say=print) -> None:
        self.folder, self.say = Path(folder), say
        self._read_to = {"server.log": 0, "agent.jsonl": 0}
        self._ending = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="follow", daemon=True)

    def __enter__(self) -> "Follow":
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.wait(FOLLOW_SECONDS):
            self.look()
        self.look()

    def look(self) -> None:
        """Say what has been written since the last look."""
        from ruibench.engines.franka.steps import MARK

        for line in self._new_lines("server.log"):
            if line.startswith(MARK):
                self.say("   " + line[len(MARK):])
        for line in self._new_lines("agent.jsonl"):
            words = " ".join(agent_says(line).split())
            if words:
                self.say(f"   agent says: {words[:SAYS_CHARS]}")
        if not self._ending and (self.folder / FAILED_BY_KEY).exists():
            self._ending = True
            self.say(alarm("   F: this interface has failed, ending the episode ..."))
            threading.Thread(target=self._end_it, name="end-it", daemon=True).start()

    def _end_it(self) -> None:
        """End the failed episode as an agent's report would: the session answers it after
        the call it is on, whose moves no longer move the arm."""
        from ruikit.serve.session import _ask

        try:
            said = _ask(self.folder, {"op": "call", "calls": [["task_done", {
                "report": "(the operator pressed F: this interface has failed)"}]]},
                timeout=END_IT_SECONDS)
        except OSError as exc:
            said = {"error": f"{type(exc).__name__}: {exc}"}
        if said.get("error"):
            self.say(alarm(f"   the session did not end ({said['error']}): Ctrl-C ends the run"))

    def _new_lines(self, name: str) -> List[str]:
        at = self._read_to[name]
        try:
            with open(self.folder / name, "rb") as fh:
                if os.fstat(fh.fileno()).st_size < at:
                    at = 0
                fh.seek(at)
                data = fh.read()
        except OSError:
            return []
        whole = data.rfind(b"\n") + 1                # a line still being written waits
        self._read_to[name] = at + whole
        return data[:whole].decode("utf-8", errors="replace").splitlines()


def lay_out(job: plans.Job, n: int, total: int, say=print) -> str:
    """The arm to its start, the table laid out: 'g' go, 's' skip, 'q' stop."""
    from ruibench.engines.franka import FAKE_ENV, check
    from ruibench.engines.franka.engine import home
    from ruibench.tasks.real import get

    card = get(job.task)
    say(heading(f"[{n}/{total}] {job.harness} {job.model} -- {job.task} seed {job.seed} "
                f"({card.name}) through {cells.profile_label(job.profile)}"))
    while os.environ.get(FAKE_ENV, "") != "1":
        alive, said = check.hand()
        if alive:
            break
        say(alarm(f"   {said}"))
        if ask("Enter when it runs again, q to stop:", "gq", "g") == "q":
            return "q"
    say(notice("   the arm goes to its start ..."))
    home()
    say(notice(f"   lay the table out: {card.setup_of(job.seed)}"))
    say(notice("   in the window: P pauses the arm, F fails this interface and goes on to the next"))
    return ask("Enter when it is laid out, s to skip this episode, q to stop:", "gsq", "g")


def judge(job: plans.Job, row: dict, folder: Path, say=print) -> dict:
    """The operator's verdict by the card's success, written into ``result.json``."""
    from ruibench.tasks.real import get

    card = get(job.task)
    done = "said it was done" if row.get("declared_done") else f"ended by {row.get('reason')}"
    say(notice(f"   the agent {done} after {row.get('replies')} replies and "
               f"{float(row.get('sim_seconds') or 0):.0f} s"))
    if row.get("summary"):
        say("   its summary: " + " ".join(str(row["summary"]).split())[:600])
    if card.game:
        say(notice(f"   by the program's own record: {'won' if row.get('success') else 'not won'} "
                   "(where the toys stand it knows only from the card; the verdict is yours)"))
    failed = (folder / FAILED_BY_KEY).exists()
    ok = not failed and ask(f"success -- {card.success}? [y/n]", "yn", "") == "y"
    try:
        note = input(question("why it failed, for the record (Enter for nothing):" if failed
                              else "a note for the record (Enter for none):")).strip()
    except EOFError:
        note = ""
    row.update(scored_success=row.get("success"), success=ok, judged_by="operator",
               layout=card.setup_of(job.seed),
               verdict={"success": ok, "note": note, "criterion": card.success,
                        "failed_by_key": failed,
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    (folder / "result.json").write_text(json.dumps(row, ensure_ascii=False, indent=1,
                                                   default=str))
    return row


def run(jobs: List[plans.Job], out: Path, record: bool = True, say=print) -> List[dict]:
    """Every job, one at a time, with the operator (the module docstring)."""
    from ruibench.engines.franka.engine import FAIL_FILE_ENV
    from ruikit.harness.session_agent import run_episode

    done: List[dict] = []
    for n, job in enumerate(jobs, 1):
        folder = job.folder(out)
        unjudged = _result(folder)
        if unjudged:
            # run before, never judged (the runner was stopped): judge it from its video
            say(heading(f"[{n}/{len(jobs)}] {job.task} through {cells.profile_label(job.profile)} "
                        f"ran but was not judged: its video is {folder / 'video.mp4'}"))
            done.append(judge(job, unjudged, folder, say=say))
            continue
        while True:
            choice = lay_out(job, n, len(jobs), say=say)
            if choice == "q":
                return done
            if choice == "s":
                break
            os.environ[FAIL_FILE_ENV] = str(Path(folder).resolve() / FAILED_BY_KEY)
            try:
                with Follow(folder, say=say):
                    row = run_episode(job.harness, job.model, job.task, job.level, job.seed,
                                      job.profile, str(out), engine=ENGINE, clock=job.clock,
                                      record=record, say=say)
            finally:
                del os.environ[FAIL_FILE_ENV]
            if "api_error" not in row:
                done.append(judge(job, row, folder, say=say))
                break
            say(alarm(f"   this episode does not count: {row['api_error'][:400]}"))
            log = folder / "server.log"
            if log.exists():
                say("   the session's last words: " + log.read_text(errors="replace")[-400:])
            if ask("run it again? [Y/n]", "yn", "y") == "n":
                break
    return done


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", default="real",
                    help="a plan's name or file (ruikit/configs/experiments)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--experiment", "--experiments", dest="experiments", nargs="*", default=[])
    ap.add_argument("--agent", "--harness", dest="harness", nargs="*", default=[],
                    help="claude-code, codex")
    ap.add_argument("--model", "--models", dest="models", nargs="*", default=[])
    ap.add_argument("--interface", "--interfaces", dest="interfaces", nargs="*", default=[],
                    help="interface names (pose, program-skill, rui-free, ...) or ids")
    ap.add_argument("--task", "--tasks", dest="tasks", nargs="*", default=[],
                    help="real task ids (ruibench/tasks/real)")
    ap.add_argument("--seed", type=int, default=None,
                    help="which table: a seed of the task's card (the plan's, if not given)")
    ap.add_argument("--test", action="store_true",
                    help=f"a try: {TEST_INTERFACE} alone, kept in {TEST_OUT.name}/, apart from the "
                         "results; run it as often as wanted")
    ap.add_argument("--dry-run", action="store_true", help="say what would run, and stop")
    ap.add_argument("--no-record", action="store_true", help="do not film the episodes")
    ap.add_argument("--provisional", action="store_true",
                    help="run on the provisional calibration (bring-up only: positions "
                         "are guesses)")
    args = ap.parse_args(argv)

    from ruibench.tasks.real import get

    experiments = plans.load_plan(args.plan)
    interfaces = [TEST_INTERFACE] if args.test else args.interfaces
    jobs = plans.select(experiments, names=args.experiments, columns=interfaces,
                        harnesses=args.harness, models=args.models)
    jobs = in_order_asked([j for j in jobs if not args.tasks or j.task in args.tasks], interfaces)
    if args.seed is not None:
        jobs = [dataclasses.replace(j, seed=args.seed) for j in jobs]
    for job in jobs:
        get(job.task).setup_of(job.seed)    # a task or a seed the station has not: said now
    out = TEST_OUT / datetime.now().strftime("%m%d-%H%M%S") if args.test else Path(args.out)
    left = todo(jobs, out)
    say = print
    say(f"{len(jobs)} episodes in the plan's selection, {len(left)} without a verdict in {out}")
    for line in plans.describe(left):
        say("   " + line)
    if args.dry_run or not left:
        return 0
    why = preflight(left, Path(args.out), args.provisional, say=say)
    if why:
        say(alarm("cannot run here:\n   " + "\n   ".join(why)))
        return 2
    try:
        done = run(left, out, record=not args.no_record, say=say)
    except KeyboardInterrupt:
        say(notice("\nstopped: run it again to go on from the first episode without a verdict"))
        return 130
    wins = sum(1 for r in done if r.get("success"))
    say(notice(f"\n{len(done)} episodes judged, {wins} successes"
               + (f": a try, which counts for nothing, in {out}" if args.test else "")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
