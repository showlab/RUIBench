"""``ruibench`` -- the benchmark: tasks, engines, scoring and reports.  Agents run with ``ruikit``.

    ruibench doctor                 # can this machine run the benchmark
    ruibench setup                  # fetch and prepare what Isaac needs, once
    ruibench tasks                  # the 30 tasks, by suite
    ruibench show DI4               # one task in full
    ruibench seeds DS1 easy         # a split's seeds (--split dev|test|spare)
    ruibench interfaces             # the 13 RUIs (ruikit interfaces)
    ruibench validate               # design rules and the frozen version, checked
    ruibench assets                 # the Isaac asset catalogue
    ruibench run DI4 --seeds dev    # the scripted reference (or a negative control) on Isaac
    ruibench report runs/x.jsonl    # a single self-contained HTML report
    ruibench view runs/x.jsonl      # replay episodes: plan view, cameras, trace
    ruibench video runs/x.jsonl     # an MP4 per recorded episode
    ruibench showcase               # every task: what the agent gets, and a success
    ruibench gallery                # every mission's start, rendered on Isaac
    ruibench serve                  # open those pages from another machine
    ruibench rescore runs/x.jsonl   # re-score stored logs; must match
    ruibench freeze NAME            # freeze for the test split (--check: what changed since)
    ruibench stop                   # stop this checkout's workers (--all: every checkout's)
    ruibench demo                   # end-to-end on the mock engine, no GPU

Options are spelled as in ``ruikit``: --task, --level, --seeds, --clock, --engine,
--gpus, --out (``ruibench.options``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .core.config import BudgetConfig, RunConfig
from .core.errors import RUIBenchError

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def _seeds(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out or [0]


# ---------------------------------------------------------------- commands
def cmd_doctor(args) -> int:
    from . import doctor
    if args.hint:
        from .engines.isaac.app import INSTALL_HINT
        print(INSTALL_HINT)
        return 0
    only = ["asset_catalogue"] if args.assets else None
    from .doctor import Check
    print(f"{BOLD}Can this machine run RUIBench?{RESET}", flush=True)

    tty = sys.stdout.isatty()

    def started(fn):
        doc = (fn.__doc__ or fn.__name__).strip().splitlines()[0]
        if tty:
            print(f"  {DIM}... {doc}{RESET}", end="\r", flush=True)
        else:
            print(f"  ... {doc}", flush=True)

    def finished(c):
        if tty:
            print(" " * 100, end="\r")
        print(c, flush=True)

    report = doctor.run(checks=only, on_result=finished, on_start=started)
    s = report["summary"]
    print(f"\n  {s['pass']} pass · {s['fail']} fail · {s['skipped']} skipped")
    print(f"  {DIM}written to {report['written_to']}{RESET}")
    return 1 if s["fail"] else 0


def cmd_tasks(args) -> int:
    from . import names
    from .tasks import SUITES, by_suite
    for suite, group in by_suite(args.suite).items():
        if not group:
            continue
        prefix, name, where = SUITES[suite]
        print(f"\n{BOLD}{names.suite_name(prefix, name)}{RESET} {DIM}[{prefix}] {where}{RESET}")
        for t in group:
            flag = {"implemented": "●", "planned": "○", "deferred": "·"}[t.status.value]
            role = t.split_role.value
            arms = "2 arms " if t.arms == 2 else ""
            print(f"  {flag} {t.id:5s} {names.task_name(t.id, t.name):30s} {DIM}{role:8s}{arms}"
                  f"{t.knob.easy} → {t.knob.hard} {t.knob.unit}{RESET}")
    print(f"\n{DIM}● implemented  ○ planned  · deferred (held-out template, "
          f"never implemented on purpose){RESET}")
    return 0


def cmd_show(args) -> int:
    from . import names
    from .tasks import get
    t = get(args.task_id)
    if args.json:
        print(json.dumps(t.as_dict(), indent=2, ensure_ascii=False))
        return 0
    print(f"{BOLD}{t.id} · {names.task_name(t.id, t.name)}{RESET}  "
          f"{DIM}({names.suite_name(t.id[:2], t.suite_name)}){RESET}")
    print(f"\n  {t.summary}")
    if t.gap:
        print(f"\n{BOLD}left out{RESET}  {t.gap}")
    if t.abilities:
        order = sorted(t.abilities, key=lambda k: (t.abilities[k] != "primary", k))
        print(f"{BOLD}draws on{RESET}  " + ", ".join(
            f"{k}{'●' if t.abilities[k] == 'primary' else '○'}" for k in order))
    print(f"\n{BOLD}knob{RESET}  {t.knob.field} — {t.knob.description}")
    print(f"        easy {t.knob.easy} → hard {t.knob.hard} {t.knob.unit}"
          + ("" if t.knob.calibrated else f"  {DIM}(NOT calibrated){RESET}"))
    print(f"\n{BOLD}success{RESET}  {t.success}")
    if t.milestones:
        print(f"\n{BOLD}milestones{RESET}")
        for m in t.milestones:
            print(f"  · {m}")
    print(f"\n{BOLD}also changes with the knob{RESET}  {DIM}(why this is not automatically "
          f"a single-factor contrast){RESET}")
    for c in t.confounds:
        print(f"  · {c}")
    if t.gap_filled:
        print(f"\n{BOLD}gap filled{RESET}  {t.gap_filled}")
    if t.blind:
        print(f"{BOLD}blind{RESET}       {t.blind}")
    if t.risks:
        print(f"\n{BOLD}risks{RESET}")
        for r in t.risks:
            print(f"  · {r}")
    print(f"\n{BOLD}build{RESET}  class {t.asset_class.value} · {t.arms} arm(s) · "
          f"{t.status.value} · split {t.split_role.value}")
    if t.base_env:
        print(f"        base: {t.base_env}")
    if t.catalogue_assets:
        print(f"        catalogue assets: {', '.join(t.catalogue_assets)}")
    if t.notes:
        print(f"\n{DIM}{t.notes.strip()}{RESET}")
    return 0


def cmd_interfaces(args) -> int:
    """The interfaces an agent acts through are RUIKit's: the paper's 13 RUIs."""
    try:
        from ruikit.cli import main as ruikit_main
    except ImportError:
        print("the interfaces are RUIKit's (ruikit interfaces)")
        return 1
    return ruikit_main(["interfaces"])


def cmd_validate(args) -> int:
    from .tasks import errors, validate
    findings = validate()
    errs = errors(findings)
    for f in findings:
        colour = "\033[31m" if f.level == "error" else "\033[33m"
        print(f"{colour}{f}{RESET}")
    print(f"\n{len(errs)} error(s), {len(findings) - len(errs)} warning(s)")
    if not errs:
        from .tasks import SUITES, main_board
        two_arm = sum(1 for t in main_board() if t.arms == 2)
        print(f"{DIM}design rules hold: {len(SUITES)} suites x 6 on the board + 1 held out, "
              f"{two_arm} two-arm tasks, every task leaves something out and draws on 3+ "
              f"abilities{RESET}")
    return 1 if errs else 0


def cmd_assets(args) -> int:
    from .engines.isaac import assets
    if args.credits:
        print(assets_credits_markdown())
        return 0
    cat = assets.load_catalogue()
    print(f"{BOLD}Isaac asset catalogue{RESET}  {DIM}(task cards name these, never USD paths){RESET}\n")
    for name, e in sorted(cat.items()):
        mark = "✓" if e.verified else "?"
        print(f"  {mark} {name:14s} {DIM}{e.category:12s} {e.usd}{RESET}")
    built = assets.primitives()
    if built:
        print(f"\n{BOLD}built from primitives{RESET} "
              f"{DIM}(a chosen shape, not a missing mesh){RESET}")
        for name, prim in sorted(built.items()):
            print(f"  ◆ {name:14s} {DIM}{prim.reason}{RESET}")
    n = len(assets.unverified())
    print(f"\n  {n} unverified" + ("" if n == 0 else
          " — run `ruibench doctor --assets` on a machine with Isaac installed"))
    return 0


def cmd_setup(args) -> int:
    """Fetch and prepare what the Isaac engine needs, once per machine: the ARX
    X5 description, the room's textures and light, and the scanned objects the
    tasks put on the table (all of them pinned: ``engines/isaac/assets.yaml``,
    ``assets.lock.json``).  Everything lands under ``RUIBENCH_THIRD_PARTY``
    (default: ``third_party/`` beside the package)."""
    import os
    import subprocess

    from .engines.isaac import app as isaac_app

    if not isaac_app.is_available():
        print(isaac_app.INSTALL_HINT)
        return 1
    if not isaac_app.eula_accepted():
        print(isaac_app.EULA_HINT)
        return 1
    from .engines.isaac import environment
    from .engines.isaac.robots import ARX5_DIR, fetch_arx5

    print(f"{BOLD}1. the ARX X5 description{RESET}", flush=True)
    if (ARX5_DIR / "arx5_obj.urdf").exists():
        print(f"   {DIM}already there: {ARX5_DIR}{RESET}")
    else:
        print(f"   {fetch_arx5()}")
    print(f"{BOLD}2. the room: textures, light, fonts{RESET}", flush=True)
    for name in environment.DOWNLOADS:
        print(f"   {name:24s} {DIM}{environment.fetched(name)}{RESET}", flush=True)
    print(f"{BOLD}3. the scanned objects{RESET}", flush=True)
    from .engines.isaac.assets import load_catalogue
    from .tasks.instance import assets_shown
    catalogue = load_catalogue()
    names = sorted(n for n, e in catalogue.items() if e.source and n == e.name) if args.all \
        else sorted(n for n in assets_shown() if n in catalogue and catalogue[n].source)
    print(f"   {len(names)} objects -- converting them needs Isaac, which starts now", flush=True)
    code = subprocess.call([sys.executable, "-m", "ruibench.engines.isaac.prepare", *names],
                           env=dict(os.environ))
    if code:
        print(f"\033[31merror:\033[0m preparing the objects failed (exit {code})", file=sys.stderr)
        return code
    print(f"\n{BOLD}ready{RESET} -- `ruibench doctor` checks the machine, "
          f"`ruibench run DI4 --engine isaac --seeds 0` runs one task")
    return 0


def assets_credits_markdown() -> str:
    """Who made the scanned and modelled things the built tasks show, as a
    Markdown list -- for a release's credits file."""
    from .engines.isaac.assets import credits
    from .engines.isaac.environment import ambientcg_ids, poly_haven_ids
    from .tasks.instance import assets_shown

    lines = ["# Credits", "", "Objects scanned or modelled by others, used under the licence named.", ""]
    for c in credits(assets_shown()):
        def item(name, url):
            return f"[{name}]({url})" if url else name
        if len(c.items) == 1 and c.items[0][1]:
            lines.append(f"- [{c.source}]({c.items[0][1]}) -- {c.licence}")
        else:
            lines.append(f"- {c.source}: {', '.join(item(n, u) for n, u in c.items)} -- {c.licence}")
    room = ", ".join(f"[{i}](https://polyhaven.com/a/{i})" for i in poly_haven_ids())
    lines.append(f"- Poly Haven, the room light and the materials: {room} -- CC0")
    grounds = ", ".join(f"[{i}](https://ambientcg.com/view?id={i})" for i in ambientcg_ids())
    lines.append(f"- ambientCG, materials: {grounds} -- CC0")
    return "\n".join(lines)


def _check_interface(name: str) -> None:
    """Refuse an interface that is only a contract so far, before any episode."""
    from .core.errors import ConfigError
    from .interfaces import INTERFACES, NotImplementedInterface

    if name not in INTERFACES.names():
        raise ConfigError(f"unknown interface {name!r}; have {', '.join(INTERFACES.names())}")
    if issubclass(INTERFACES.get(name), NotImplementedInterface):
        raise ConfigError(
            f"{name} is not implemented yet: I1-I4 are contracts whose action languages are "
            "written separately (docs/INTERFACES.md).  The scripted reference, the negative "
            "controls and the stress agents run through I0_direct, the default.")


def cmd_run(args) -> int:
    from .policies import build_agent
    from .runner import RunPlan, run_plan

    from . import options as O
    from .core.errors import ConfigError
    from .tasks.instance import implemented

    _check_interface(args.interface)
    gpus = O.gpus_of(args)
    if gpus:
        import os
        os.environ["RUIBENCH_ISAAC_GPU"] = str(gpus[0])
    try:
        args.tasks = O.expand_tasks(O.tasks_of(args) or [], list(implemented()))
        args.levels = O.levels_of(args)
        clock, latency = O.engine_clock(O.normal_clock(args.clock or args.clock_default,
                                                       args.latency))
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    if not args.tasks:
        raise ConfigError("which tasks? ruibench run WK1 CL2, a suite (WK) or all")
    spec = O.seeds_of(args)
    split = spec.partition(":")[0] if spec.partition(":")[0] in O.SPLITS else None
    if args.out.endswith("/") or Path(args.out).is_dir():
        args.out = str(Path(args.out) / "episodes.jsonl")
    record_dir = str(Path(args.out).with_suffix(".frames")) if args.record else None
    cfg = RunConfig(
        run_id=args.run_id, engine=args.engine, interface=args.interface,
        clock=clock, clock_latency_seconds=latency if clock == "controlled" else args.latency,
        budget=BudgetConfig(max_model_calls=args.max_calls),
        record_dir=record_dir, record_every=args.record_every, record_size=args.record_size,
        split=split or "seeds",
    )
    agent = ((args.control or "reference")
             + (f"+misperceive={args.misperceive}" if args.misperceive else "")
             + ("+serial" if args.serial else "")
             + ("+ignore_rules" if args.ignore_rules else "")
             + ("+greedy" if args.greedy else ""))
    missions = None
    if split:
        from .tasks import splits
        data = splits.load()
        missions = [(t, lv, s) for t in args.tasks for lv in args.levels
                    for s in O.parse_seeds(spec, t, lv, data)]
    plan = RunPlan(task_ids=args.tasks, levels=tuple(args.levels),
                   seeds=_seeds(spec) if not split else [0], interface=args.interface,
                   missions=missions)

    isolate = args.isolate if args.isolate is not None else (args.engine == "isaac")
    if isolate:
        # Worker processes that may die.  The Isaac stack can abort with a native
        # heap corruption that does not reproduce; a crashed episode is retried
        # and, if it keeps crashing, recorded as a harness error -- never as a
        # model failure.
        from .runner.isolated import IsolatedRun, run_isolated
        job = IsolatedRun(cells=plan.cells(), config=cfg, agent=agent,
                          max_attempts=args.max_attempts)
        logs = run_isolated(job, args.out, progress=print)
        ok = sum(1 for log in logs if log["score"].get("success"))
        lost = sum(1 for log in logs if log["score"].get("failure") == "harness_error")
        print(f"\n{ok}/{len(logs)} succeeded"
              + (f" · {lost} lost to harness errors (excluded from rates)" if lost else "")
              + f" · logs: {args.out}")
        print(f"{DIM}replay them: ruibench view {args.out}{RESET}")
        return 0

    results = run_plan(plan, lambda inst: build_agent(agent, inst), config=cfg,
                       out=args.out, progress=print)
    ok = sum(1 for r in results if r.success)
    print(f"\n{ok}/{len(results)} succeeded" + (f" · logs: {args.out}" if args.out else ""))
    if args.out:
        print(f"{DIM}replay them: ruibench view {args.out}{RESET}")
    return 0


def cmd_report(args) -> int:
    from . import reports
    from .runner import read_jsonl
    logs = list(read_jsonl(args.logs))
    out = args.out or str(Path(args.logs).with_suffix(".html"))
    path = reports.write(logs, out, title=args.title)
    print(f"wrote {path}  ({path.stat().st_size / 1024:.1f} KB, {len(logs)} episodes)")
    for b in reports.Results.from_logs(logs).leaderboard_blockers():
        print(f"  {DIM}note: {b}{RESET}")
    return 0


def cmd_view(args) -> int:
    from .reports import viewer
    from .runner import read_jsonl

    logs = list(read_jsonl(args.logs))
    out = Path(args.out or Path(args.logs).with_suffix(".viewer.html"))
    video_dir = Path(args.videos) if args.videos else Path(args.logs).with_suffix(".videos")
    videos = viewer.find_videos(logs, video_dir, out.parent) if video_dir.is_dir() else {}
    path = viewer.write(logs, out, title=args.title or Path(args.logs).stem, embed=args.embed,
                        max_frames=args.max_frames, videos=videos)
    if videos:
        print(f"  {len(videos)} episode video(s) from {video_dir}")
    frames = sum(len(log.get("frames") or []) for log in logs)
    print(f"wrote {path}  ({path.stat().st_size / 1024:.0f} KB, {len(logs)} episodes, "
          f"{frames} camera frames{' inlined' if args.embed else ''})")
    if not frames:
        print(f"  {DIM}no camera frames: the plan view is the replay. On Isaac, record "
              f"them with `ruibench run ... --engine isaac --record`.{RESET}")
    return 0


def cmd_video(args) -> int:
    from .reports import video
    from .runner import read_jsonl

    if not video.have_ffmpeg():
        raise RUIBenchError("ffmpeg is not on PATH; it encodes the videos")
    logs = list(read_jsonl(args.logs))
    out_dir = Path(args.out or Path(args.logs).with_suffix(".videos"))
    written = video.write_videos(logs, out_dir, fps=args.fps, speed=args.speed, size=args.size)
    for path in written:
        print(f"  {path}  ({path.stat().st_size / 1024:.0f} KB)")
    missing = len(logs) - len(written)
    print(f"wrote {len(written)} video(s) to {out_dir}"
          + (f"; {missing} episode(s) had no recorded frames" if missing else ""))
    if written:
        print(f"{DIM}they appear in: ruibench view {args.logs}{RESET}")
    return 0


def cmd_gallery(args) -> int:
    from .doctor import isaac_ready, run_isaac_child
    from .reports.gallery import contact_sheet, write_manifest
    from .tasks.instance import implemented

    tasks = args.tasks or list(implemented())
    out_dir = Path(args.out)
    blocked = isaac_ready()
    if blocked:
        raise RUIBenchError(f"cannot render the gallery: {blocked}")
    failed = False
    for task in tasks:
        # one child per level: a task's levels may differ in structure (a rack
        # or not, a rule more), and one simulator process keeps one stage
        for level in args.levels:
            print(f"  rendering {task} {level} {DIM}(seeds {args.seeds}){RESET}", flush=True)
            result = run_isaac_child("ruibench.reports.gallery", task, str(out_dir),
                                     level, args.seeds, str(args.size), timeout=1800)
            if result.get("error"):
                failed = True
                print(f"    \033[31mfailed:\033[0m {str(result.get('error', ''))[-300:]}")
                continue
            unseen = [k for k, v in result.get("gripper_pixels_in_agentview", {}).items() if not v]
            unseen += [f"mark {k}" for k, v in result.get("mark_pixels_in_agentview", {}).items()
                       if not v]
            unseen += [f"one colour: {name}" for name in result.get("blank", [])]
            failed = failed or bool(unseen)
            print(f"    {result.get('saved', 0)} images"
                  + (f" · \033[31mnot in the fixed camera: {', '.join(unseen[:6])}\033[0m"
                     if unseen else " · gripper and every mark in view"))
    write_manifest(out_dir, tasks, args.levels, _seeds(args.seeds), args.size)
    print(f"wrote {contact_sheet(out_dir)}")
    return 1 if failed else 0


def cmd_showcase(args) -> int:
    import shutil

    from .reports import showcase
    from .tasks.loader import load_all
    from .tasks.spec import Status

    tasks = [t for t in load_all() if t.status is Status.IMPLEMENTED
             and (not args.tasks or t.id in args.tasks)]
    out_dir = Path(args.out)
    if not args.page_only:
        if args.gpu is not None:
            import os
            os.environ["RUIBENCH_ISAAC_GPU"] = str(args.gpu)
        if args.engine == "isaac":
            from .doctor import isaac_ready
            blocked = isaac_ready()
            if blocked:
                raise RUIBenchError(f"cannot record the showcase: {blocked}  "
                                    "(--page-only rebuilds the page from what is recorded)")
        roles = tuple(args.roles or showcase.ROLES)
        if args.redo:
            for t in tasks:
                for role in roles:
                    log = showcase.log_path(out_dir, t.id, role)
                    for extra in (log.with_suffix(".frames"), log.with_suffix(".workers")):
                        shutil.rmtree(extra, ignore_errors=True)
                    log.unlink(missing_ok=True)
                    showcase.video_path(out_dir, t.id, role).unlink(missing_ok=True)
        for task, role in showcase.missing(tasks, out_dir, roles):
            print(f"  {task.id} {role}: {DIM}{showcase.agent_for(task, role)}{RESET}", flush=True)
            log = showcase.record(task, role, out_dir, engine=args.engine, progress=print,
                                  seed=args.seed)
            verdict = "as intended" if showcase.shows_its_role(log, role) else \
                "\033[31mNOT as intended\033[0m"
            failure = (log or {}).get("score", {}).get("failure", "no log")
            print(f"    {role}: {failure} · {verdict}")
    page = showcase.write(out_dir)
    print(f"wrote {page}")
    print(f"{DIM}open it through `ruibench serve {out_dir.parent}` on a remote machine{RESET}")
    return 0


def cmd_serve(args) -> int:
    from .reports import serve

    root = Path(args.dir)
    if not root.is_dir():
        raise RUIBenchError(f"{root} is not a directory; generate pages first "
                            "(ruibench demo, ruibench view, ruibench report)")
    server = serve.make_server(root, host=args.host, port=args.port)
    host, port = server.server_address[:2]
    base = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    found = serve.pages(root)
    print(f"{BOLD}serving {root.resolve()}{RESET} at {base}/")
    for page in found[:30]:
        print(f"  {base}/{page.relative_to(root).as_posix()}")
    if not found:
        print(f"  {DIM}(no .html pages yet){RESET}")
    print(f"{DIM}On a remote machine: forward port {port} (VS Code: Ports panel -> Forward a Port; "
          f"or ssh -L {port}:localhost:{port} <host>), then open a link above locally. "
          f"Ctrl-C to stop.{RESET}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_seeds(args) -> int:
    """The seeds of one split for a task and level, space-separated -- for a
    script that runs an agent over them (``ruibench/tasks/splits.json``)."""
    from .tasks import splits

    data = splits.load()
    levels = args.levels or ["easy", "hard"]
    for level in levels:
        seeds = splits.seeds(args.task, level, args.split, data)
        print(" ".join(map(str, seeds)) if len(levels) == 1 else f"{level}: {' '.join(map(str, seeds))}")
    return 0


def cmd_freeze(args) -> int:
    """Freeze the benchmark for the test split -- or, with --check or no name,
    say what has changed since it was frozen (``tasks/freeze.py``)."""
    from .tasks import freeze

    if args.check or not args.name:
        record = freeze.load()
        if record is None:
            print("nothing is frozen yet: `ruibench freeze NAME` freezes the benchmark")
            return 0
        lines = freeze.drift()
        print(f"frozen as {BOLD}{record['name']}{RESET} on {record['date']} "
              f"(T_ref {record['t_ref_seconds']} s, {len(record['files'])} files)")
        for line in lines:
            print(f"  {line}")
        print(f"  {DIM}nothing has changed{RESET}" if not lines else
              f"{len(lines)} difference(s): results on the test split before and after do not compare")
        return 1 if lines else 0
    if freeze.load() is not None and not args.again:
        print(f"already frozen as {freeze.load()['name']}; a new version needs --again", file=sys.stderr)
        return 1
    record = freeze.freeze(args.name)
    print(f"frozen as {BOLD}{record['name']}{RESET}: {len(record['files'])} files and T_ref "
          f"{record['t_ref_seconds']} s fingerprinted in {freeze.PATH.name}; the splits are frozen")
    return 0


def cmd_stop(args) -> int:
    from .runner.isolated import stop_workers

    stopped = stop_workers(Path("/") if args.everywhere else None)
    print(f"stopped {len(stopped)} process(es)" + (f": {', '.join(map(str, stopped))}" if stopped else ""))
    return 0


def cmd_rescore(args) -> int:
    """Score every stored episode again, from its log alone -- the task rebuilt
    from its id, level, seed and config, the trajectory from the snapshots
    (``runner/snapshot``) -- in this process, not the one that ran it, and
    report where the new score differs from the stored one.  A log from before
    v15 (``ruibench-log/1``) kept positions only: for those, check that the
    stored score is complete and consistent."""
    from .runner import read_jsonl
    from .runner.snapshot import is_whole, rescore

    paths = [args.logs] if isinstance(args.logs, str) else list(args.logs)
    logs = [log for path in paths for log in read_jsonl(path)]
    need = ("task_id", "level", "success", "progress", "failure", "milestones_hit")
    bad, rescored, checked = [], 0, 0
    for log in logs:
        score = log.get("score") or {}
        if score.get("failure") == "harness_error":
            continue
        problems = []
        if is_whole(log.get("trajectory") or []):
            rescored += 1
            try:
                _, problems = rescore(log)
            except Exception as exc:                       # a log the task can no longer read
                problems = [f"cannot rescore: {type(exc).__name__}: {exc}"]
        else:
            checked += 1
            missing = [k for k in need if k not in score]
            end = (log.get("trajectory") or [{}])[-1].get("t", 0.0)
            problems = missing and [f"missing {', '.join(missing)}"] or []
            if not missing:
                if bool(score["success"]) != (score["failure"] in ("none", None)):
                    problems.append(f"success {score['success']} but failure {score['failure']!r}")
                late = [m for m, t in dict(score["milestones_hit"]).items()
                        if t is not None and t > end + 1e-6]
                if late:
                    problems.append(f"milestones after the trajectory ends: {', '.join(late)}")
        if problems:
            bad.append((log.get("task_id"), log.get("level"), log.get("seed"), problems))
    for tid, level, seed, problems in bad[:20]:
        print(f"  {tid} {level} seed {seed}: {'; '.join(problems)}")
    print(f"rescored {rescored} episodes from their trajectories, checked {checked} older "
          f"logs' stored scores · {len(bad)} with differences")
    return 1 if bad else 0


def cmd_demo(args) -> int:
    from . import reports
    from .policies import wrap
    from .runner import RunPlan, read_jsonl, run_plan
    from .tasks.instance import implemented

    out = Path(args.out)
    cfg = RunConfig(run_id="demo", engine="mock", interface="I0_direct",
                    clock="zero_latency")
    tasks = list(implemented())
    seeds = tuple(range(args.seeds))
    print(f"{BOLD}RUIBench demo{RESET} {DIM}mock engine, no GPU, no API calls{RESET}")
    print(f"{DIM}tasks: {', '.join(tasks)}{RESET}\n")

    print(f"{BOLD}1. scripted reference{RESET} "
          f"{DIM}(layer 1: is the task physically solvable at all){RESET}")
    ref = run_plan(RunPlan(tasks, levels=("easy", "hard"), seeds=seeds),
                   agent_factory=lambda i: i.reference_factory(i), config=cfg, out=out)
    print(f"   {sum(1 for r in ref if r.success)}/{len(ref)} succeeded")

    print(f"\n{BOLD}2. negative controls{RESET} "
          f"{DIM}(each judged only against the information it removes; only the "
          f"controls a task's card lists){RESET}")
    from .tasks.loader import get
    controls = ("do_nothing", "first_frame_only", "ignore_change", "ignore_feedback",
                "ignore_demonstration")
    header = f"   {'':6s}" + "".join(f"{c:>22s}" for c in controls)
    print(BOLD + header + RESET)
    results = {c: [] for c in controls}
    for task in tasks:
        row = f"   {task:6s}"
        listed = set(get(task).negative_controls)
        for control in controls:
            if control not in listed:
                row += f"{'-':>22s}"
                continue
            res = run_plan(RunPlan([task], levels=("easy", "hard"), seeds=seeds),
                           agent_factory=lambda i, c=control: wrap(c, i.reference_factory(i)),
                           config=cfg)
            results[control] += res
            n = sum(1 for r in res if r.success)
            row += f"{n:>17d}/{len(res):<4d}"
        print(row)
    from .runner import write_jsonl
    for control, res in results.items():
        if res:
            write_jsonl(str(out) + f".{control}", [r.log for r in res])
    print(f"   {DIM}a control passing is not a bug: a static task really is solvable "
          f"from one frame,{RESET}")
    print(f"   {DIM}and ignoring a change, a feedback bit or a demonstration cannot matter "
          f"where there is none.{RESET}")

    logs = list(read_jsonl(out))
    path = reports.write(logs, str(Path(out).with_suffix(".html")),
                         title="RUIBench demo (mock engine)")
    replay = reports.viewer.write(logs, Path(out).with_suffix(".viewer.html"),
                                  title="demo · scripted references, mock engine")
    print(f"\n   report: {path}")
    print(f"   replay: {replay}  {DIM}(plan view and timeline of every episode){RESET}")
    return 0


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ruibench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="can this machine run the benchmark")
    d.add_argument("--assets", action="store_true", help="only verify the asset catalogue")
    d.add_argument("--hint", action="store_true", help="print Isaac install instructions")
    d.set_defaults(fn=cmd_doctor)

    sd = sub.add_parser("seeds", help="the seeds of a split (dev, test, spare) for one task")
    sd.add_argument("task")
    sd.add_argument("levels", nargs="*", help="easy, hard (default: both)")
    sd.add_argument("--split", default="dev", choices=("dev", "test", "spare"))
    sd.set_defaults(fn=cmd_seeds)

    fz = sub.add_parser("freeze", help="freeze the benchmark for the test split, or --check "
                                       "what changed since")
    fz.add_argument("name", nargs="?", help="the version's name, e.g. v1.0")
    fz.add_argument("--check", action="store_true", help="only say what differs from the freeze")
    fz.add_argument("--again", action="store_true", help="freeze a new version over an old one")
    fz.set_defaults(fn=cmd_freeze)

    t = sub.add_parser("tasks", help="list task cards")
    t.add_argument("suite", nargs="?", help="restrict to one suite")
    t.set_defaults(fn=cmd_tasks)

    s = sub.add_parser("show", help="show one task card")
    s.add_argument("task_id")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_show)

    i = sub.add_parser("interfaces", help="the 13 RUIs (as ruikit interfaces)")
    i.set_defaults(fn=cmd_interfaces)

    v = sub.add_parser("validate", help="check the design rules")
    v.set_defaults(fn=cmd_validate)

    a = sub.add_parser("assets", help="the Isaac asset catalogue")
    a.add_argument("--credits", action="store_true",
                   help="who made the things the tasks show (Markdown, for a release)")
    a.set_defaults(fn=cmd_assets)

    r = sub.add_parser("run", help="run the scripted reference, a negative control or a "
                                   "stress agent (models run with `ruikit run`)")
    from . import options as O
    O.add_tasks(r, positional=True)
    O.add_levels(r)
    O.add_seeds(r, "0-4")
    O.add_clock(r, "paused")
    O.add_engine(r, "isaac")
    O.add_gpus(r)
    r.add_argument("--interface", default="I0_direct", help=argparse.SUPPRESS)
    r.add_argument("--latency", type=float, default=20.0, help=argparse.SUPPRESS)
    r.add_argument("--control", choices=["do_nothing", "first_frame_only", "ignore_feedback",
                                         "ignore_change", "ignore_demonstration"],
                   help="run a negative control instead of the reference")
    r.add_argument("--misperceive", type=float, metavar="METRES",
                   help="stress agent: see every object this far off (per axis, fixed per object)")
    r.add_argument("--max-calls", type=int, default=60, dest="max_calls")
    r.add_argument("--isolate", action=argparse.BooleanOptionalAction, default=None,
                   help="run episodes in crash-tolerant worker processes "
                        "(default: on for --engine isaac)")
    r.add_argument("--max-attempts", type=int, default=3, dest="max_attempts",
                   help="retries for an episode whose worker crashed")
    r.add_argument("--run-id", default="dev", dest="run_id")
    r.add_argument("-o", "--out", default="runs/latest.jsonl",
                   help="the episodes' log (JSON lines), or a folder for it (default runs/latest.jsonl)")
    r.add_argument("--serial", action="store_true",
                   help="stress agent for two-arm tasks: one arm at a time")
    r.add_argument("--ignore-rules", action="store_true", dest="ignore_rules",
                   help="stress agent for rules tasks: plays as if the game had no rules")
    r.add_argument("--greedy", action="store_true",
                   help="stress agent for games: plays for its own lines and never blocks")
    r.add_argument("--record", action="store_true",
                   help="save camera frames for `ruibench view` (next to --out)")
    r.add_argument("--record-every", type=float, default=0.5, dest="record_every",
                   help="simulated seconds between recorded frames, besides every observation "
                        "and the final state (0.1 for smooth videos, 0 for none)")
    r.add_argument("--record-size", type=int, default=256, dest="record_size",
                   help="recorded frame size in pixels")
    r.set_defaults(fn=cmd_run)

    vw = sub.add_parser("view", help="replay episodes: plan view, cameras, timeline")
    vw.add_argument("logs")
    vw.add_argument("-o", "--out", help="default: next to the log, .viewer.html")
    vw.add_argument("--title")
    vw.add_argument("--embed", action="store_true",
                    help="inline the camera frames: one file to share")
    vw.add_argument("--max-frames", type=int, dest="max_frames",
                    help="per episode and camera; observations are kept first")
    vw.add_argument("--videos", help="episode videos (default: next to the log, .videos/)")
    vw.set_defaults(fn=cmd_view)

    vd = sub.add_parser("video", help="an MP4 per recorded episode (needs ffmpeg)")
    vd.add_argument("logs")
    vd.add_argument("-o", "--out", help="default: next to the log, .videos/")
    vd.add_argument("--fps", type=int, default=20)
    vd.add_argument("--speed", type=float, default=2.0, help="simulated seconds per video second")
    vd.add_argument("--size", type=int, default=320, help="pixels per camera tile")
    vd.set_defaults(fn=cmd_video)

    sc = sub.add_parser("showcase", help="every task: what the agent is given, and one success")
    sc.add_argument("--task", "--tasks", dest="tasks", nargs="+", help="default: every implemented task")
    sc.add_argument("--engine", default="isaac")
    sc.add_argument("--gpus", "--gpu", dest="gpu", type=int, help="the GPU Isaac renders on")
    sc.add_argument("--redo", action="store_true", help="record the chosen tasks again")
    sc.add_argument("--seed", "--seeds", dest="seed", type=int,
                    help="the mission to film (default: seed 0, or the first seed on which "
                         "the mock shows the clip's role)")
    sc.add_argument("--page-only", action="store_true", dest="page_only",
                    help="only rebuild the page from what is recorded")
    sc.add_argument("--roles", nargs="+", choices=("success", "hard"),
                    help="which clips: the easy level (success), the hard one (default: both)")
    sc.add_argument("-o", "--out", default="runs/showcase")
    sc.set_defaults(fn=cmd_showcase)

    gl = sub.add_parser("gallery", help="render the start of every mission (level x seed) on Isaac")
    gl.add_argument("--task", "--tasks", dest="tasks", nargs="+", help="default: every implemented task")
    gl.add_argument("--level", "--levels", dest="levels", nargs="+", default=["easy", "hard"])
    gl.add_argument("--seeds", default="0-9", help="missions to render per level, e.g. 0-9")
    gl.add_argument("--size", type=int, default=384, help="image side in pixels")
    gl.add_argument("-o", "--out", default="runs/gallery")
    gl.set_defaults(fn=cmd_gallery)

    sv = sub.add_parser("serve", help="serve the generated pages over HTTP (for remote machines)")
    sv.add_argument("dir", nargs="?", default="runs")
    sv.add_argument("--port", type=int, default=None, help="default 8765, or the next free one")
    sv.add_argument("--host", default="127.0.0.1",
                    help="loopback by default: reach it through a forwarded port")
    sv.set_defaults(fn=cmd_serve)

    su = sub.add_parser("setup", help="fetch and prepare the Isaac engine's robots, room and objects")
    su.add_argument("--all", action="store_true",
                    help="every catalogue entry, not only what the tasks put on the table")
    su.set_defaults(fn=cmd_setup)

    st = sub.add_parser("stop", help="stop the RUIBench workers and Isaac checks running from this "
                                     "checkout")
    st.add_argument("--all", action="store_true", dest="everywhere",
                    help="every checkout's, on this account")
    st.set_defaults(fn=cmd_stop)

    rep = sub.add_parser("report", help="render an HTML report")
    rep.add_argument("logs")
    rep.add_argument("-o", "--out")
    rep.add_argument("--title", default="RUIBench results")
    rep.set_defaults(fn=cmd_report)

    rs = sub.add_parser("rescore", help="re-score stored logs")
    rs.add_argument("logs", nargs="+")
    rs.set_defaults(fn=cmd_rescore)

    dm = sub.add_parser("demo", help="end-to-end demo on the mock engine")
    dm.add_argument("-o", "--out", default="runs/demo.jsonl")
    dm.add_argument("--seeds", type=int, default=1,
                    help="seeds per cell (1: about ten minutes on one core)")
    dm.set_defaults(fn=cmd_demo)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except RUIBenchError as exc:
        print(f"\033[31merror:\033[0m {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
