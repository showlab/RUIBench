"""The release changes nothing an agent is given: every interface the paper ran, its tools as
each provider is sent them, and both agents' prompts are as in the paper's runs."""

import json
from pathlib import Path

from protocol_snapshot import digests

FROZEN = Path(__file__).parent / "data" / "protocol_paper.json"


def test_what_the_agent_is_given_is_the_papers():
    want = json.loads(FROZEN.read_text())
    got = digests()
    changed = [f"{pid} {k}" for pid, item in want["interfaces"].items()
               for k, v in item.items() if got["interfaces"].get(pid, {}).get(k) != v]
    changed += [k for k, v in want["prompts"].items() if got["prompts"].get(k) != v]
    assert not changed, "differs from the paper's runs: " + ", ".join(changed)
