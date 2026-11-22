"""The chat-completions request format: a system message first, tool results as
``tool`` messages, pictures as ``image_url`` parts."""

from __future__ import annotations

import json
from typing import List, Optional

from ruikit.models.base import Call, Reply, _data_url
from ruikit.models.conversation import ModelClient, _arguments
from ruikit.models.gateway import Gateway

#: the values a tool argument may take besides text, by its JSON Schema type
_TAKES = {"array": lambda v: isinstance(v, list), "object": lambda v: isinstance(v, dict),
          "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
          "integer": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
          "boolean": lambda v: isinstance(v, bool), "null": lambda v: v is None}


def _types(schema: dict) -> set:
    t = schema.get("type")
    out = set(t) if isinstance(t, list) else {t} if t else set()
    for alt in schema.get("anyOf") or ():
        out |= _types(alt)
    return out


def typed_arguments(args: dict, schema: dict) -> dict:
    """``args`` with each value that came as text where its parameter takes no
    text -- a list, a number, a boolean -- read as the JSON it spells.

    QwenCloud hands on a parameter whose type is a union (``["array", "null"]``,
    an optional list) as the text the model wrote, where an open server (vLLM's
    Qwen parser) reads it by the schema; the tool would refuse it.  Text that
    spells no value the parameter takes stays text, for the tool to refuse."""
    props = schema.get("properties") or {}
    out = dict(args)
    for key, value in args.items():
        types = _types(props.get(key) or {})
        if not isinstance(value, str) or not types or "string" in types:
            continue
        try:
            value = json.loads(value)
        except ValueError:
            continue
        if any(_TAKES[t](value) for t in types if t in _TAKES):
            out[key] = value
    return out


class ChatCompletions(ModelClient):
    def __init__(self, gateway: Gateway, system: str, tools: List[dict],
                 keep_images: int = 6, max_output_tokens: int = 4096,
                 keep_calls: Optional[int] = None, on_overflow: Optional[int] = None) -> None:
        super().__init__(gateway, system, tools, keep_images, max_output_tokens, keep_calls,
                         on_overflow)
        if system:
            self.history.append({"role": "system", "content": system})

    def _ask(self, tries: dict) -> Reply:
        payload: dict = {"model": self.gw.model, "messages": self.outgoing()}
        payload["max_completion_tokens" if self.gw.openai_dialect else "max_tokens"] = \
            self.max_output_tokens
        if self.gw.provider == "local":           # a Qwen-style template: thinking on or off
            payload["chat_template_kwargs"] = {"enable_thinking": bool(self.gw.thinking)}
        if self.tools:
            payload["tools"] = [{"type": "function",
                                 "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["schema"]}} for t in self.tools]
            payload["tool_choice"] = "auto"
        body = self.gw.post("/chat/completions", payload, tries)
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls = []
        schemas = {t["name"]: t["schema"] for t in self.tools}
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args, bad = _arguments(fn.get("arguments"))
            if self.gw.provider == "qwencloud" and not bad:
                args = typed_arguments(args, schemas.get(fn.get("name", "")) or {})
            calls.append(Call(tc.get("id") or f"call_{len(calls)}", fn.get("name", ""), args, bad))
        # a reasoning model's thinking comes apart (vLLM: ``reasoning``); the answer
        # itself can start with blank lines
        content = (msg.get("content") or "").strip()
        thought = msg.get("reasoning") or msg.get("reasoning_content") or ""
        kept = {"role": "assistant", "content": content}
        if msg.get("tool_calls"):
            kept["tool_calls"] = [{"id": c.id, "type": "function",
                                   "function": {"name": c.name,
                                                "arguments": json.dumps(c.arguments)
                                                if not c.bad_arguments else c.bad_arguments}}
                                  for c in calls]
        self.history.append(kept)
        return Reply(content, calls, 0.0, body.get("usage") or {}, choice.get("finish_reason") or "",
                     reasoning=str(thought)[:20000])

    def _tool_output(self, call: Call, out: str) -> dict:
        return {"role": "tool", "tool_call_id": call.id, "content": out}

    def _text(self, text: str) -> dict:
        return {"type": "text", "text": text}

    def _image(self, png: bytes, label: str) -> dict:
        return {"type": "image_url", "image_url": {"url": _data_url(png)}, "_label": label}
