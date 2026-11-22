"""One model call and its reply, and the error raised when a channel is down."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ruikit.core.result import ToolError


class GatewayDown(ToolError):
    """A model channel that does not answer; what to do about it is the
    channel's own (``down_hint`` in ``ruikit/configs/models.yaml``)."""

    def __init__(self, url: str, why: str, provider: str = "openai") -> None:
        from ruikit.models.channels import channels

        channel = channels()["channels"].get(provider, {})
        super().__init__("backend_unavailable",
                         f"{channel.get('label', provider)} at {url} is not answering ({why})",
                         hint=channel.get("down_hint", ""))


@dataclass
class Call:
    id: str
    name: str
    arguments: Dict[str, Any]
    bad_arguments: str = ""          # the raw text, when it was not a JSON object


@dataclass
class Reply:
    text: str
    calls: List[Call]
    latency_s: float
    usage: Dict[str, Any] = field(default_factory=dict)
    stop: str = ""
    waited_s: float = 0.0            # of latency_s, before the answering try went out
    retries: int = 0                 # tries beyond the first
    reasoning: str = ""              # a reasoning model's thinking, when it says it apart


def _message(err) -> str:
    return err.get("message", json.dumps(err)) if isinstance(err, dict) else str(err)


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
