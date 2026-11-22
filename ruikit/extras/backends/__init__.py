"""Backends: the models and services that tools call -- a VLA policy, a decision model, a
vision-language model, and the detectors and segmenters behind pixel perception.

A tool depends on a backend's *protocol*, never on a vendor SDK: a VLA served
by openpi, GR00T or StarVLA, a decision model like Jev, a pointing VLM.  A
backend is registered under a name and chosen by the profile
(``params.policy_run.backend``).  A tool whose backend is not configured is
not offered in an episode at all (``unavailable``): listed, it cost agents
replies on calls that could only fail.  One that fails while running
says ``backend_unavailable`` -- explicit, never a silent fallback.

Every backend sees only what its tool hands it: frames the agent's cameras
took, proprioception, text.  None is ever given the simulator.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from ruikit.core.result import ToolError


class PolicyBackend(Protocol):
    """A VLA behind ``policy_run`` / ``policy_propose``.

    ``infer`` gets the instruction, the latest frames (``{camera: image}``) and
    proprioception, and returns an action chunk: rows of
    ``[dx, dy, dz, droll, dpitch, dyaw, gripper]`` in the station frame
    (metres, degrees; gripper >= 0.5 closes).  ``rate_hz`` is how often the
    session asks for a new chunk; ``latency_s`` the simulated time one
    inference takes."""

    name: str
    rate_hz: float
    latency_s: float
    cameras: Sequence[str]

    def reset(self) -> None: ...

    def infer(self, instruction: str, frames: Dict[str, Any], proprio: dict) -> List[List[float]]: ...


class DecisionBackend(Protocol):
    """A fast discrete-decision model behind ``model_decide`` (Jev)."""

    name: str

    def decide(self, state: Any, questions: Dict[str, dict]) -> dict: ...


class AskBackend(Protocol):
    """A VLM behind ``model_ask``: images and a question in, text out."""

    name: str

    def ask(self, images: List[bytes], question: str) -> str: ...


class VisionBackend(Protocol):
    """A detector and a segmenter behind ``perception: model``: a picture and
    words in, boxes and masks out.  ``info`` names the models, for the log."""

    name: str

    def info(self) -> dict: ...

    def detect(self, image: Any, phrases: Sequence[str]) -> List[dict]: ...

    def segment(self, image: Any, point: Any = None, box: Any = None) -> Optional[dict]: ...


def _vision(name: str) -> Callable[[], VisionBackend]:
    def make() -> VisionBackend:
        from ruikit.perception.service.client import VisionServer

        return VisionServer(name)

    return make


POLICIES: Dict[str, Callable[[], PolicyBackend]] = {}
DECIDERS: Dict[str, Callable[[], DecisionBackend]] = {}
ASKERS: Dict[str, Callable[[], AskBackend]] = {}
VISION: Dict[str, Callable[[], VisionBackend]] = {
    name: _vision(name) for name in ("sam3", "grounded-sam2", "gdino-tiny-sam2", "owlv2-sam2")}


def unavailable(profile) -> Dict[str, str]:
    """The tools a profile lists that cannot run as it is configured -- tool:
    why.  An episode offers its agent the others only."""
    listed = set(profile.tools) | set(profile.code_tools)
    out: Dict[str, str] = {}

    def needs(tool: str, table: Dict[str, Callable[[], Any]], kind: str,
              default: Optional[str] = None) -> None:
        if tool not in listed:
            return
        name = profile.tool_params(tool).get("backend", default)
        if not name:
            out[tool] = f"no {kind} backend is configured (params.{tool}.backend)"
        elif name not in table:
            out[tool] = f"{kind} backend {name!r} is not registered"
        elif name == "jev" and not os.environ.get("TYPESAFE_API_KEY"):
            out[tool] = "Jev needs TYPESAFE_API_KEY in the tool server's environment"

    needs("policy_run", POLICIES, "policy")
    needs("policy_propose", POLICIES, "policy")
    needs("model_ask", ASKERS, "vision-language")
    needs("model_decide", DECIDERS, "decision", "jev")
    if "model_imagine" in listed:
        out["model_imagine"] = "no world-model backend exists yet"
    return out


def get_backend(table: Dict[str, Callable[[], Any]], kind: str, name: Optional[str]):
    if not name:
        raise ToolError("backend_unavailable", f"no {kind} backend is configured for this profile",
                        hint=f"set params.<tool>.backend; registered: {sorted(table) or 'none'}")
    factory = table.get(name)
    if factory is None:
        raise ToolError("backend_unavailable", f"{kind} backend {name!r} is not registered",
                        hint=f"registered: {sorted(table) or 'none'}")
    return factory()


# ----------------------------------------------------------------------------
# Jev (TypeSafe AI "System One" model): POST /v1/systemone
# ----------------------------------------------------------------------------
class JevBackend:
    """TypeSafe's Jev: text state + typed questions -> answers with probabilities.

    Text only -- it cannot see images.  Needs ``TYPESAFE_API_KEY``; the model
    version is pinned (``jev-1.13.0``) so results stay comparable.
    """

    name = "jev"
    url = "https://api.typesafe.ai/v1/systemone"
    model = "jev-1.13.0"

    def __init__(self) -> None:
        self.key = os.environ.get("TYPESAFE_API_KEY")
        if not self.key:
            raise ToolError("backend_unavailable", "Jev needs TYPESAFE_API_KEY in the tool "
                            "server's environment")

    def decide(self, state: Any, questions: Dict[str, dict]) -> dict:
        import httpx

        t0 = time.time()
        resp = httpx.post(self.url, headers={"Authorization": f"Bearer {self.key}"},
                          json={"model": self.model, "state": state, "questions": questions},
                          timeout=30.0)
        if resp.status_code in (429, 529):
            raise ToolError("rate_limited", "Jev is rate limited; retry later")
        if resp.status_code >= 400:
            raise ToolError("tool_failed", f"Jev returned HTTP {resp.status_code}")
        out = resp.json()
        out["latency_s"] = round(time.time() - t0, 3)
        return out


DECIDERS["jev"] = JevBackend


# ----------------------------------------------------------------------------
# test doubles (never on a leaderboard: they are registered only by tests)
# ----------------------------------------------------------------------------
class HoldPolicy:
    """A policy that proposes standing still: for testing the session plumbing."""

    name = "hold"
    rate_hz = 5.0
    latency_s = 0.05
    cameras = ("front",)

    def reset(self) -> None:
        self.calls = 0

    def infer(self, instruction, frames, proprio):
        self.calls = getattr(self, "calls", 0) + 1
        return [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]


# model_ask and model_decide backends through the model channels (``routes.ASK_ROUTES``)
from ruikit.extras.backends.routes import register as _register_routes

_register_routes()
