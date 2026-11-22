"""What the agent is given, for every interface the paper ran: the protocol a release must keep.

``snapshot()`` collects, for each of the paper's interfaces (the 13 RUIs, the action-sequence
variant, the action-level checks), the interface as loaded, its tools as each provider and MCP
are sent them, and a program's reference; and the prompts of both kinds of agent (the
ready-made agents' templates, and the source of how the minimal loop words its first message).
``digests()`` is a SHA-256 of each item; ``tests/ruikit/data/protocol_paper.json`` holds them as
the paper's runs had them.

    python tests/ruikit/protocol_snapshot.py OUT.json      # from a checkout on the path
"""

import hashlib
import inspect
import json
import sys

PAPER_IDS = [
    "rui.joint/v1+aids", "rui.delta/v1+aids", "rui.step/v1+aids", "rui.pose/v1+aids",
    "rui.point/v1+aids", "rui.skill/v1+aids",
    "rui.code-joint/v1+aids", "rui.code-delta/v1+aids", "rui.code-step/v1+aids", "rui.code/v1+aids",
    "rui.code-point/v1+aids", "rui.code-skill/v1+aids", "rui.free/v1",
    "rui.seq-delta/v1+aids", "rui.seq-step/v1+aids", "rui.seq-point/v1+aids",
    "rui.skill/v1+aids+check", "rui.code-skill/v1+aids+check",
    "rui.skill/v1+aids+check:correct", "rui.code-skill/v1+aids+check:correct",
]


def snapshot() -> dict:
    from ruikit.harness import minimal_loop, session_agent
    from ruikit.interfaces import load_profile
    from ruikit.serve import schemas

    out = {"interfaces": {}, "prompts": {}}
    for pid in PAPER_IDS:
        p = load_profile(pid)
        out["interfaces"][pid] = {
            "profile": p.as_dict(),
            "anthropic": schemas.anthropic_tools(p),
            "openai": schemas.openai_tools(p),
            "mcp": schemas.mcp_tools(p),
            "code_reference": schemas.code_reference(p) if p.code_tools else "",
        }
    out["prompts"]["session_agent.PROMPT"] = session_agent.PROMPT
    out["prompts"]["session_agent.REAL_PROMPT"] = session_agent.REAL_PROMPT
    for name in ("instructions", "_window", "_guide", "_tools", "_output", "_pictures"):
        out["prompts"][f"minimal_loop.{name}"] = inspect.getsource(getattr(minimal_loop, name))
    return json.loads(json.dumps(out, default=str))


def digests(snap: dict = None) -> dict:
    snap = snapshot() if snap is None else snap

    def h(x):
        return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    return {"interfaces": {pid: {k: h(v) for k, v in item.items()}
                           for pid, item in snap["interfaces"].items()},
            "prompts": {k: h(v) for k, v in snap["prompts"].items()}}


if __name__ == "__main__":
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        json.dump(digests(), fh, indent=1, sort_keys=True)
