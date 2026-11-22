"""RUIKit's options, spelled one way across its commands (``ruibench.options``
has the tasks, levels, seeds, clock, engine, cards and output):

    --interface program-skill pose rui-free      the paper's names (``ruikit.naming``), ids
                                                 (rui.code-skill/v1+aids) or interface files
    --form per-action program free               with a plan: these interaction forms
    --abstraction joint ... skill                with a plan: these action abstractions
    --agent minimal-loop | codex | codex-code | claude-code
    --model <channel>:<name>                     the model (openai:..., anthropic:..., local:...)

Older spellings (``--interfaces``, ``--columns``, ``--profiles``, ``--profile``,
``--harness``, ``--models``, ``--forms``, ``--actions``) are still read and not shown.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from ruikit import naming

#: the agents (Harnesses) as a person names them -> their labels in the results
AGENTS = {"minimal-loop": "loop", "loop": "loop", "minimal_loop": "loop", "codex": "codex",
          "codex-code": "codex-code", "codex_code": "codex-code",
          "claude-code": "claude-code", "claude_code": "claude-code"}


def add_interfaces(p: argparse.ArgumentParser, help_: str = "") -> None:
    help_ = help_ or ("interfaces by the paper's names -- joint ... skill, program-joint ... "
                      "program-skill, rui-free (each with the aids, as the paper ran them) -- "
                      "or ids (rui.code-skill/v1+aids)")
    p.add_argument("--interface", dest="interfaces", nargs="+", action="extend", default=None,
                   help=help_)
    for old in ("--interfaces", "--columns", "--profiles", "--profile"):
        p.add_argument(old, dest="interfaces", nargs="+", action="extend", help=argparse.SUPPRESS)


def add_agent(p: argparse.ArgumentParser, many: bool = False, default: Optional[str] = None) -> None:
    kw = dict(nargs="+", action="extend") if many else {}
    p.add_argument("--agent", dest="agents_", default=None, **kw,
                   help="minimal-loop (a model API through RUIKit's tool-calling loop), codex, "
                        "codex-code or claude-code" + (f" (default {default})" if default else ""))
    p.add_argument("--harness", dest="agents_", help=argparse.SUPPRESS, **kw)
    p.set_defaults(agent_default=default)


def add_model(p: argparse.ArgumentParser, many: bool = False) -> None:
    kw = dict(nargs="+", action="extend") if many else {}
    p.add_argument("--model", dest="models_", default=None, **kw,
                   help="<channel>:<name> -- openai:..., anthropic:..., local:... "
                        "(ruikit/configs/models.yaml); for codex and claude-code, the agent's own")
    p.add_argument("--models", dest="models_", help=argparse.SUPPRESS, **kw)


def add_forms(p: argparse.ArgumentParser, forms: Sequence[str], actions: Sequence[str]) -> None:
    p.add_argument("--form", dest="forms", nargs="+", action="extend", default=[],
                   choices=tuple(forms) + ("per_action",), help="with a plan: these interaction forms")
    p.add_argument("--forms", dest="forms", nargs="+", action="extend",
                   choices=tuple(forms) + ("per_action",), help=argparse.SUPPRESS)
    p.add_argument("--abstraction", dest="actions", nargs="+", action="extend", default=[],
                   choices=actions, help="with a plan: these action abstractions")
    p.add_argument("--actions", dest="actions", nargs="+", action="extend", choices=actions,
                   help=argparse.SUPPRESS)


# ---------------------------------------------------------------- reading them
def agents_of(args) -> List[str]:
    """The agents asked for, as the results label them (``loop``, ``codex`` ...)."""
    given = args.agents_ if isinstance(args.agents_, list) else (
        [args.agents_] if args.agents_ else [])
    if not given and args.agent_default:
        given = [args.agent_default]
    out = []
    for a in given:
        if a not in AGENTS:
            raise ValueError(f"no agent {a!r}: minimal-loop, codex, codex-code or claude-code")
        out.append(AGENTS[a])
    return list(dict.fromkeys(out))


def models_of(args) -> List[str]:
    m = args.models_
    return list(m) if isinstance(m, list) else ([m] if m else [])


def interface_ids(names: Optional[Sequence[str]]) -> List[str]:
    """The ids to run for interfaces given by name or id (a name brings the aids)."""
    return list(dict.fromkeys(naming.id_of(n) for n in names or ()))


def interface_filter(names: Optional[Sequence[str]]) -> List[str]:
    """Interfaces given to choose among a plan's: names as the paper's (older ones
    read), ids as given."""
    return [naming.name_of(n) if naming.is_name(n) else n for n in names or ()]
