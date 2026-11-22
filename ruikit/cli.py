"""``ruikit``: the interfaces and the tool library, and running Robot Agents on RUIBench.

    ruikit interfaces                      the 13 RUIs by the paper's names, their ids, and the rest
    ruikit interface program-skill [--format anthropic|openai|gemini|mcp] [--strict]
                                           one interface: its tools, or its tool schemas
    ruikit tools | design | check          the tool library; the design space; contract checks
    ruikit episode --task WK1 --interface program-pose --model openai:<name>
                                           one episode of an agent
    ruikit run --plan paper --experiment rui --agent claude-code --model claude-opus-5-5
                                           many episodes: a plan's, or the settings given
    ruikit report --out runs/paper         the tables and the results page of a run
    ruikit rescore runs/paper --task WK1   score a run's episodes again
    ruikit coverage --task WK1 --interface pose
                                           can each interface do each task (the reference acting)
    ruikit serve-mcp --task WK1 --interface rui-free
                                           one episode served over MCP (stdio), for any agent

Every command spells its options the same way: --task, --level, --seeds, --interface,
--agent, --model, --clock, --engine, --gpus, --out (``ruibench.options``,
``ruikit.cli_options``); the interfaces go by the paper's names (``ruikit.naming``).
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence


def _tools(args) -> int:
    from ruikit.core.tool import all_tools
    from ruikit.interfaces.roles import ROLES, role_of, where

    tools = all_tools()
    for role, (en, zh) in list(ROLES.items()) + [("", ("in no place", "无归属"))]:
        names = [n for n in tools if role_of(n) == role]
        if not names:
            continue
        print(f"{en} {zh} ({len(names)})")
        for n in names:
            s = tools[n].spec
            print(f"   {n:22s} {where(n):16s} {s.source:10s} {s.description.split('. ')[0][:64]}")
    return 0


def _design(args) -> int:
    from ruikit.actions import ACTIONS
    from ruikit.extras import EXTRAS
    from ruikit.harness.components import COMPONENTS
    from ruikit.harness.components.aids import AID_ALWAYS, AID_GROUPS, AIDS_LEFT_OUT
    from ruikit.interfaces import FORMS, list_profiles
    from ruikit.interfaces.base import SHARED
    from ruikit.interfaces.catalog import form_of
    from ruikit.perception.settings import DEPTHS, DETECTORS, SCALES

    print(f"Shared by every interface 共享工具: {', '.join(SHARED)}")
    print("   perception, chosen after '@': detector " + " | ".join(DETECTORS)
          + "; depth=" + " | ".join(DEPTHS) + "; noise=, calib= "
          + " | ".join(f"{s:g}" for s in SCALES) + "; privileged (the control condition)")
    print("Action abstractions 动作抽象:")
    for a in ACTIONS.values():
        print(f"   {a.name:8s} {a.zh:4s} {', '.join(a.tools)}")
    print("Interaction forms 交互形式, and the named interfaces of each:")
    for form in FORMS:
        ids = [i for i in list_profiles() if form_of(i)[0] == form]
        print(f"   {form:10s} {', '.join(ids)}")

    def listed(found):
        for c in found.values():
            impls = " | ".join((i.name + ("*" if i.name == c.default else ""))
                               for i in c.implementations)
            state = f"  [not offered: {c.unavailable}]" if c.unavailable else ""
            print(f"   {c.name:8s} {c.zh:6s} {', '.join(c.tools):48s} {impls}{state}")

    print("Harness components Harness 组件, attached with '+name[:implementation]':")
    listed(COMPONENTS)
    print("   guide    操作指南   '+guide[:name]'; pull 不推送图像 '+pull'")
    print("The aids 辅助工具, attached with '+aids'; '+aids-<group>' takes one group away:")
    for group, names in AID_GROUPS.items():
        print(f"   {group:8s} " + ", ".join(names))
    print(f"   always   {', '.join(AID_ALWAYS)}")
    print("   never    " + "; ".join(f"{k} ({v})" for k, v in AIDS_LEFT_OUT.items()))
    print("Extras 设计空间之外, attached the same way, never among the aids:")
    listed(EXTRAS)
    return 0


def _interfaces(args) -> int:
    from ruikit import naming
    from ruikit.interfaces import list_profiles, load_profile

    def line(name, pid):
        p = load_profile(pid)
        return (f"   {name:16s} {naming.label(name):16s} {pid:26s} tools={len(p.tools):2d}"
                f" code_tools={len(p.code_tools):2d}" + (f"  {p.title}" if args.all else ""))

    print("The 13 RUIs of the paper, as run (each with the aids; RUI-Free has them built in):")
    for name in naming.RUIS:
        print(line(name, naming.id_of(name)))
    print("The paper's ablation, the action-sequence variant:")
    for name in naming.SEQUENCE:
        print(line(name, naming.id_of(name)))
    named = {naming.IDS[n] for n in naming.NAMES}
    rest = [pid for pid in list_profiles() if pid not in named]
    if rest:
        print("Beyond the paper:")
        for pid in rest:
            p = load_profile(pid)
            print(f"   {pid:16s} {'':16s} {'':26s} tools={len(p.tools):2d} "
                  f"code_tools={len(p.code_tools):2d}  {p.title}")
    print("Names (program-skill) and ids (rui.code-skill/v1+aids) are the same interfaces; "
          "ruikit interface <name> shows one.")
    return 0


def _interface(args) -> int:
    from ruikit.interfaces import load_profile
    from ruikit.serve import schemas

    from ruikit import naming

    p = load_profile(naming.id_of(args.id))
    if args.format == "yaml":
        import yaml

        print(yaml.safe_dump(p.as_dict(), sort_keys=False, allow_unicode=True), end="")
        return 0
    fn = {"anthropic": lambda: schemas.anthropic_tools(p, strict=args.strict),
          "openai": lambda: schemas.openai_tools(p, strict=args.strict),
          "gemini": lambda: schemas.gemini_tools(p),
          "mcp": lambda: schemas.mcp_tools(p)}[args.format]
    print(json.dumps(fn(), indent=2, ensure_ascii=False))
    return 0


def _check(args) -> int:
    from ruikit.core.check import check_all
    from ruikit.interfaces import list_profiles, load_profile

    bad = 0
    for name, problems in check_all().items():
        if problems:
            bad += 1
            print(f"FAIL {name}: " + "; ".join(problems))
    for pid in list_profiles():
        try:
            load_profile(pid)
        except Exception as exc:
            bad += 1
            print(f"FAIL interface {pid}: {exc}")
    print("all tools and interfaces pass" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


def _run(rest: Sequence[str]) -> int:
    from ruikit.experiments.batch import main as batch_main

    return batch_main(list(rest))


def _report(rest: Sequence[str]) -> int:
    """The tables and the results page of a run's folder (``ruikit run --report``)."""
    from ruikit.experiments.batch import main as batch_main

    rest = list(rest)
    if rest and not rest[0].startswith("-"):              # ruikit report runs/paper
        rest = ["--out", rest[0]] + rest[1:]
    return batch_main(rest + ["--report"])


def _episode(rest: Sequence[str]) -> int:
    """One episode of an agent (``ruikit.run``)."""
    from ruibench import options as O
    from ruikit import cli_options as K
    from ruikit.agent import Agent, run
    from ruikit.conditions import DEFAULT_CLOCK
    from ruikit.experiments.plans import expand_tasks, parse_seeds

    ap = argparse.ArgumentParser(prog="ruikit episode", description=_episode.__doc__)
    O.add_tasks(ap, required=True)
    O.add_levels(ap, default=("easy",))
    O.add_seeds(ap, "dev:1")
    K.add_interfaces(ap)
    K.add_agent(ap, default="minimal-loop")
    K.add_model(ap)
    O.add_clock(ap, DEFAULT_CLOCK)
    O.add_engine(ap, "isaac")
    O.add_gpus(ap)
    O.add_out(ap, "runs/ruikit", "the folder the episode's cell goes into")
    args = ap.parse_args(list(rest))
    try:
        (task,) = expand_tasks(O.tasks_of(args))
        (level,) = O.levels_of(args)
        seed = parse_seeds(O.seeds_of(args), task, level)[0]
        (interface,) = args.interfaces or ["pose"]
        (agent,) = K.agents_of(args)
        clock = O.clock_of(args)
    except ValueError as exc:
        ap.error(f"one task, level, seed, interface and agent: {exc}")
    model = K.models_of(args)
    if not model:
        ap.error("give --model (openai:<name>, anthropic:<name>, local:<name>; for codex and "
                 "claude-code their model)")
    gpus = O.gpus_of(args)
    if gpus:
        import os
        os.environ["RUIBENCH_ISAAC_GPU"] = str(gpus[0])
    name = {"loop": "minimal-loop"}.get(agent, agent)
    row = run(Agent(model=model[0], harness=name, interface=interface), task, level, seed,
              condition=clock, engine=args.engine, out=args.out)
    print(json.dumps({k: row.get(k) for k in ("task", "level", "seed", "column", "success",
                                              "replies", "folder")}, indent=1, default=str))
    return 0


def _rescore(rest: Sequence[str]) -> int:
    from ruikit.experiments.rescore import main as rescore_main

    return rescore_main(list(rest))


def _serve_mcp(rest: Sequence[str]) -> int:
    """One episode served over MCP (stdio), the options spelled as everywhere:
    --interface (a name or an id), --seeds (its first seed), --clock 8s | paused |
    measured, --out (the logs' folder)."""
    from ruibench import options as O
    from ruikit import cli_options as K
    from ruikit.serve.mcp_stdio import main as mcp_main

    ap = argparse.ArgumentParser(prog="ruikit serve-mcp", description=_serve_mcp.__doc__)
    O.add_tasks(ap)
    O.add_levels(ap, default=("easy",))
    O.add_seeds(ap, "dev:1")
    K.add_interfaces(ap, "the interface, by name or id (default rui-free)")
    O.add_engine(ap, "isaac")
    O.add_clock(ap, "8s")
    ap.add_argument("--out", "-o", default=None, help="the folder of the episode's logs")
    args, other = ap.parse_known_args(list(rest))
    from ruikit.experiments.plans import expand_tasks, parse_seeds

    argv = list(other)
    try:
        if O.tasks_of(args):
            (task,) = expand_tasks(O.tasks_of(args))
            (level,) = O.levels_of(args)
            argv += ["--task", task, "--level", level,
                     "--seed", str(parse_seeds(O.seeds_of(args), task, level)[0])]
        (interface,) = K.interface_ids(args.interfaces or ["rui-free"])
        clock = O.clock_of(args)
    except ValueError as exc:
        ap.error(f"one task, level, seed and interface: {exc}")
    argv += ["--profile", interface, "--engine", args.engine]
    if clock == "paused":
        argv += ["--clock", "zero"]
    elif clock == "measured":
        argv += ["--clock", "measured"]
    else:
        argv += ["--clock", "fixed", "--think-seconds", clock[:-1]]
    if args.out:
        argv += ["--log-dir", args.out]
    return mcp_main(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="ruikit", description=__doc__.split("\n\n")[0],
                                 epilog="ruikit <command> --help: the command's options",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")
    sub.add_parser("interfaces", help="the 13 RUIs by the paper's names, and the rest").add_argument(
        "--all", action="store_true", help="with each interface's title (what the agent is told)")
    p = sub.add_parser("interface", help="one interface: its tools, or its tool schemas")
    p.add_argument("id", help="a name (program-skill) or an id (rui.code-skill/v1+aids)")
    p.add_argument("--format", default="yaml", choices=("yaml", "anthropic", "openai", "gemini", "mcp"))
    p.add_argument("--strict", action="store_true")
    sub.add_parser("tools", help="every tool of the library, by its place in the design space")
    sub.add_parser("design", help="the action abstractions, the forms, the components, the aids")
    sub.add_parser("check", help="contract checks of every tool and interface")
    passed = {"episode": ("one episode of an agent", _episode),
              "run": ("many episodes: a plan's, or the settings given", _run),
              "report": ("the tables and the results page of a run", _report),
              "rescore": ("score a run's episodes again", _rescore),
              "coverage": ("can each interface do each task (the reference acting)", None),
              "serve-mcp": ("one episode served over MCP (stdio), for any agent", _serve_mcp)}
    for name, (help_, _) in passed.items():
        sub.add_parser(name, help=help_, add_help=False)
    args, rest = ap.parse_known_args(argv)
    if args.cmd in passed:
        if args.cmd == "coverage":
            from ruikit.diagnostics.coverage import main as coverage_main

            return coverage_main(rest)
        return passed[args.cmd][1](rest)
    return {"tools": _tools, "design": _design, "interfaces": _interfaces,
            "interface": _interface, "check": _check}[args.cmd](args)

if __name__ == "__main__":
    raise SystemExit(main())
