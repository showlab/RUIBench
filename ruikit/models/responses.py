"""The responses request format: the system prompt as ``instructions``, tool
results as ``function_call_output`` items, pictures as ``input_image`` parts."""

from __future__ import annotations

from ruikit.models.base import Call, Reply, _data_url
from ruikit.models.conversation import ModelClient, _arguments


class Responses(ModelClient):
    def _ask(self, tries: dict) -> Reply:
        payload: dict = {"model": self.gw.model, "input": self.outgoing(),
                         "max_output_tokens": self.max_output_tokens, "store": False}
        if self.system:
            payload["instructions"] = self.system
        if self.tools:
            payload["tools"] = [{"type": "function", "name": t["name"],
                                 "description": t["description"], "parameters": t["schema"],
                                 "strict": False} for t in self.tools]
            payload["tool_choice"] = "auto"
        body = self.gw.post("/responses", payload, tries)
        texts, calls = [], []
        for item in body.get("output") or []:
            kind = item.get("type")
            if kind == "message":
                texts += [p.get("text", "") for p in item.get("content") or []
                          if p.get("type") == "output_text"]
                self.history.append({"role": "assistant", "content": "".join(
                    p.get("text", "") for p in item.get("content") or [] if p.get("type") == "output_text")})
            elif kind == "function_call":
                args, bad = _arguments(item.get("arguments"))
                call = Call(item.get("call_id") or item.get("id") or f"call_{len(calls)}",
                            item.get("name", ""), args, bad)
                calls.append(call)
                self.history.append({"type": "function_call", "call_id": call.id, "name": call.name,
                                     "arguments": item.get("arguments") or "{}"})
        return Reply("".join(texts) or body.get("output_text") or "", calls, 0.0,
                     body.get("usage") or {}, body.get("status") or "")

    def _tool_output(self, call: Call, out: str) -> dict:
        return {"type": "function_call_output", "call_id": call.id, "output": out}

    def _text(self, text: str) -> dict:
        return {"type": "input_text", "text": text}

    def _image(self, png: bytes, label: str) -> dict:
        return {"type": "input_image", "image_url": _data_url(png), "_label": label}
