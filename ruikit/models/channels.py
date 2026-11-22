"""The model channels: where each model is reached, under which name, and in
which request format (``ruikit/configs/models.yaml``)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import yaml

from ruikit.core.paths import CONFIGS


@lru_cache(maxsize=1)
def channels() -> Mapping[str, Any]:
    return yaml.safe_load((CONFIGS / "models.yaml").read_text(encoding="utf-8"))


def _channel(name: str) -> Mapping[str, Any]:
    return channels()["channels"].get(name) or {}


LOCAL_BASE_URL = _channel("local").get("base_url", "http://127.0.0.1:8000/v1")
#: models served on this machine (vLLM), by short name
LOCAL_MODELS = dict(_channel("local").get("deployments") or {})
#: QwenCloud models, by short name; its endpoint
QWENCLOUD_MODELS = dict(_channel("qwencloud").get("deployments") or {})
QWENCLOUD_BASE_URL = _channel("qwencloud").get("base_url", "")
#: its Token Plan's endpoint (``RUIKIT_QWENCLOUD_PLAN``)
QWENTOKENPLAN_BASE_URL = _channel("qwencloud").get("token_plan_url", "")


def keyed(name: str) -> Mapping[str, Any]:
    """A channel reached with an API key of its own (``api_key_env``): OpenAI,
    Anthropic, or any OpenAI-compatible endpoint written into models.yaml."""
    conf = _channel(name)
    return conf if conf.get("api_key_env") else {}
