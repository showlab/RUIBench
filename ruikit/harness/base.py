"""A Harness: it runs one episode -- asks the model, carries out the calls the
model makes through the interface's tools, hands back what they returned --
until the episode ends.  What it adds to an interface (memory, the aids, an
operation guide, the observation policy) are its components
(``ruikit.harness.components``), attached in the interface id."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional


class Harness:
    #: its label in the results layout (``ruikit.experiments.results.HARNESSES``)
    name = ""
    #: the module a batch worker runs as: ``python -m <module> --worker <spec>``
    worker_module = ""
    #: True: a worker keeps one simulator for many episodes, the model asked from
    #: this process; False: each episode starts its own session server, which the
    #: agent reaches over MCP
    in_process = True

    def preflight(self) -> Optional[str]:
        """Why it cannot run here, or None."""
        return None

    def run(self, job, out: Path, engine: str = "isaac", **options) -> dict:
        """One episode of ``job`` (``ruikit.experiments.plans.Job``) into its
        cell's folder under ``out``; returns what was written there."""
        raise NotImplementedError


def harnesses() -> Mapping[str, Harness]:
    """The Harnesses, by their label: the simple loop and the ready-made agents."""
    from ruikit.harness.minimal_loop import MinimalLoop
    from ruikit.harness.session_agent import SessionAgent

    return {h.name: h for h in (MinimalLoop(), SessionAgent("codex"), SessionAgent("codex-code"),
                                SessionAgent("claude-code"))}
