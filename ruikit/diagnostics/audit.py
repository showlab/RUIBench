"""Checks for a round of agent tests.

    python -m ruikit.diagnostics.audit rules  TRANSCRIPTS   # did each agent keep the rules?
    python -m ruikit.diagnostics.audit rules  ROLLOUT.jsonl # one transcript: Claude Code or Codex
    python -m ruikit.diagnostics.audit grasps ROUND/clips   # how precisely did it really grasp?
    python -m ruikit.diagnostics.audit where  ROUND/clips   # estimate error vs grasp error vs room

``rules`` reads agent transcripts -- Claude Code's (``*.output`` / ``*.jsonl``,
a subagent's or the stream-json of ``claude -p``) and Codex's session records
(``rollout-*.jsonl``).  Allowed: the MCP tools of the session's own server
(``ruibench``), ``ruikit.serve.session`` commands on the agent's own
session, and opening that session's pictures.  Flagged: anything that names
repository or log paths, the memory folder, imports the packages, or reads /
lists / searches files; any other tool; and more than one session call in one
shell command, or one inside a loop -- in an interface that takes one action
a reply, a loop of calls is a program in disguise.  ``grasps`` and ``where``
read each run's true-state snapshots: a median grasp offset of about 1 mm
means the tools leak exact positions; a real robot's is several
millimetres.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import List, Optional

from ruikit.diagnostics.grasp_audit import audit_grasps, audit_where
from ruikit.experiments.rules_audit import audit_rules


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("check", choices=["rules", "grasps", "where"])
    ap.add_argument("folder")
    args = ap.parse_args(argv)
    if args.check == "rules":
        rows = audit_rules(args.folder)
        for r in rows:
            print(f"{os.path.basename(r['session']):28s} tool uses {r['tool_uses']:4d}  flagged {len(r['flagged'])}")
            for b in r["flagged"]:
                print("     !", json.dumps(b)[:240])
        print(f"{len(rows)} transcripts, {sum(bool(r['flagged']) for r in rows)} with rule breaks")
        return 1 if any(r["flagged"] for r in rows) else 0
    rows = audit_grasps(args.folder) if args.check == "grasps" else audit_where(args.folder)
    for r in rows:
        print(json.dumps(r))
    key = "xy_mm" if args.check == "grasps" else "grasp_mm"
    xs = sorted(r[key] for r in rows if r.get(key) is not None)
    if xs:
        print(json.dumps({"grasps": len(xs), "misses": sum(1 for r in rows if r.get("held", 1) is None),
                          "median_mm": statistics.median(xs), "max_mm": xs[-1]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
