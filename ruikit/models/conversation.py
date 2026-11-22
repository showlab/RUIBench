"""A conversation with a model: the history, the pictures kept, one call at a
time.  Each endpoint's request format is a subclass: ``ChatCompletions``
(``ruikit.models.chat_completions``) and ``Responses`` (``ruikit.models.responses``)."""

from __future__ import annotations

import json
import time
from typing import List, Optional, Sequence

from ruikit.core.result import ToolError
from ruikit.models.base import Call, Reply
from ruikit.models.gateway import Gateway

#: how an endpoint refuses a request longer than the model's context window (vLLM and
#: OpenAI: "This model's maximum context length is 262144 tokens. However, ...")
FULL_CONTEXT = ("maximum context length", "context_length_exceeded", "context window",
                "prompt is too long", "input is too long")
#: what the model is told when its history is first cut for a full context
FULL_CONTEXT_NOTE = (
    "[The conversation has reached the model's context window. From now on only your latest "
    "{keep} tool calls stay whole: an older result keeps only ok, error, text and hint, and an "
    "older call's long arguments (a program, a note) are cut to a line. The text of your "
    "replies stays whole: write there what you will need again.]")

#: what an older call's result keeps (``ToolResult.as_json``): whether it worked and its one line
SHORT_FIELDS = ("ok", "error", "text", "hint")
#: an older result's ``text`` past this many characters is cut
SHORT_TEXT = 300
#: an older call's string argument (a program, a long query) past this many characters is cut
LONG_ARGUMENT = 200
#: the first call to one of these stays whole: the station's conventions (frames, units, cameras)
WHOLE_FIRST = ("task_info",)
#: the key an older result carries once cut (``as_json`` has no such key)
CUT = "cut"


def _arguments(raw) -> tuple:
    if isinstance(raw, dict):
        return raw, ""
    try:
        args = json.loads(raw or "{}")
    except ValueError:
        return {}, str(raw)
    return (args, "") if isinstance(args, dict) else ({}, str(raw))


def _short_result(out: str) -> str:
    """An older call's result: only ``SHORT_FIELDS``, its text cut at
    ``SHORT_TEXT``; the same string back when that would not be shorter."""
    try:
        res = json.loads(out)
    except (TypeError, ValueError):
        res = None
    if isinstance(res, dict):
        if CUT in res:
            return out                        # cut already
        short = {k: res[k] for k in SHORT_FIELDS if res.get(k) is not None}
        if isinstance(short.get("text"), str) and len(short["text"]) > SHORT_TEXT:
            short["text"] = short["text"][:SHORT_TEXT] + "..."
        short[CUT] = "an earlier result: the rest dropped from the history to save space"
        text = json.dumps(short, ensure_ascii=False)
    else:
        mark = " [an earlier result, cut to save space]"
        if out.endswith(mark):
            return out
        text = out[:SHORT_TEXT] + mark
    return text if len(text) < len(out) else out


def _short_value(value):
    if isinstance(value, str) and len(value) > LONG_ARGUMENT:
        lines = value.count("\n") + 1
        what = f"{lines} lines" if lines > 1 else f"{len(value)} characters"
        return f"[{what}, sent earlier: dropped from the history to save space]"
    if isinstance(value, dict):
        return {k: _short_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_short_value(v) for v in value]
    return value


def _short_arguments(raw):
    """An older call's arguments: each string past ``LONG_ARGUMENT`` (a
    program) becomes a line saying what was there.  Still a JSON object, in the
    form it was kept in (a string, or a dict)."""
    args = raw if isinstance(raw, dict) else None
    if args is None:
        try:
            args = json.loads(raw or "{}")
        except (TypeError, ValueError):
            return raw                        # not an object: as it was
        if not isinstance(args, dict):
            return raw
    short = _short_value(args)
    if short == args:
        return raw
    return short if isinstance(raw, dict) else json.dumps(short)


def _full_context(exc: ToolError) -> bool:
    """The endpoint refused the request as longer than the model's context window."""
    text = str(exc).lower()
    return "http 400" in text and any(m in text for m in FULL_CONTEXT)


class ModelClient:
    """A tool-calling conversation with one model, in its endpoint's own shape.

    ``tools``: ``[{"name", "description", "schema"}]``.  Only the latest
    ``keep_images`` pictures stay in the history; older ones become a line
    saying which frame was there (a look every turn would otherwise fill the
    context within a few turns).  A picture the model has not been sent yet
    always goes: a reply that brings more than ``keep_images`` (a
    demonstration's sample is up to nine) has each one seen once.

    ``keep_calls`` does the same for text (``None``: every word stays): the
    latest ``keep_calls`` tool calls stay whole; an older call keeps its
    result's ``SHORT_FIELDS`` (whether it worked, its one line) and its
    arguments with each long string (a program, a note) cut to a line.  A call
    and its result stay paired; the text of the model's replies stays whole, and
    so does the first call to ``WHOLE_FIRST``.  (A note cut here is still in
    the episode's memory: ``memory_recall`` reads it back.)  A call once cut
    stays as cut, so the history before the latest calls does not change from
    one request to the next.

    ``on_overflow`` keeps every word until the endpoint refuses a request as
    longer than the model's context window, and only then turns the window on
    with that many calls, telling the model so (``FULL_CONTEXT_NOTE``); the
    same request is sent again, the window halved while it is still too long.
    A conversation that never fills the context is sent word for word as
    without it.  ``window_from``: the request the window started at."""

    def __init__(self, gateway: Gateway, system: str, tools: List[dict],
                 keep_images: int = 6, max_output_tokens: int = 4096,
                 keep_calls: Optional[int] = None, on_overflow: Optional[int] = None) -> None:
        self.gw = gateway
        self.system = system
        self.tools = tools
        self.keep_images = keep_images
        self.keep_calls = keep_calls
        self.on_overflow = on_overflow
        self.window_from: Optional[int] = None
        self.requests = 0
        self.max_output_tokens = max_output_tokens
        self.history: List[dict] = []
        #: how much of the history the model has been sent: the pictures after it
        #: it has yet to see, and they are not dropped before it does
        self._sent = 0

    # -- what the loop says ------------------------------------------------------
    def user(self, text: str, images: Sequence[bytes] = (), labels: Sequence[str] = ()) -> None:
        if not images:
            self.history.append({"role": "user", "content": text})
            return
        parts: List[dict] = [self._text(text)]
        for k, png in enumerate(images):
            label = labels[k] if k < len(labels) else f"picture {k + 1}"
            parts.append(self._text(label))
            parts.append(self._image(png, label))
        self.history.append({"role": "user", "content": parts})

    def results(self, calls: Sequence[Call], outputs: Sequence[str],
                pictures: Sequence[tuple] = ()) -> None:
        """Each call's output, then the pictures they returned (``(label, png)``):
        a tool result cannot carry an image on either endpoint, so they follow
        as the next user message."""
        for call, out in zip(calls, outputs):
            self.history.append(self._tool_output(call, out))
        if pictures:
            self.user("The pictures those calls returned:", [png for _, png in pictures],
                      [label for label, _ in pictures])

    def ask(self) -> Reply:
        self._prune()
        self._prune_calls()
        t0 = time.time()
        tries: dict = {}
        while True:
            try:
                reply = self._ask(tries)
                break
            except ToolError as exc:
                if not (_full_context(exc) and self._narrow()):
                    raise
        self.requests += 1
        self._sent = len(self.history)
        reply.latency_s = round(time.time() - t0, 3)
        reply.waited_s = tries.get("waited_s", 0.0)
        reply.retries = tries.get("retries", 0)
        return reply

    # -- the endpoint's format ------------------------------------------------------
    def _ask(self, tries: dict) -> Reply:
        """Send the history, read the reply back into it."""
        raise NotImplementedError

    def _tool_output(self, call: Call, out: str) -> dict:
        raise NotImplementedError

    def _text(self, text: str) -> dict:
        raise NotImplementedError

    def _image(self, png: bytes, label: str) -> dict:
        raise NotImplementedError

    # -- pictures -------------------------------------------------------------------
    def _prune(self) -> None:
        """Keep the latest pictures, and every one not sent yet; say which ones
        were dropped.  (The ``_label`` a picture carries is ours: ``outgoing``
        strips it.)"""
        seen = 0
        for i in range(len(self.history) - 1, -1, -1):
            msg = self.history[i]
            parts = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(parts, list):
                continue
            for k in range(len(parts) - 1, -1, -1):
                part = parts[k]
                if part.get("type") not in ("image_url", "input_image"):
                    continue
                seen += 1
                if seen > self.keep_images and i < self._sent:
                    parts[k] = self._text(f"[{part.get('_label', 'an earlier picture')}: dropped "
                                          "from the history to save space]")

    # -- text -----------------------------------------------------------------------
    def _calls(self) -> List[tuple]:
        """Every tool call in the history, in order: ``(holder, name)``, the
        holder carrying its ``arguments`` (either endpoint's shape)."""
        found = []
        for msg in self.history:
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "function_call":                     # responses
                found.append((msg, msg.get("name", "")))
            for tc in msg.get("tool_calls") or ():                     # chat completions
                fn = tc.get("function") or {}
                found.append((fn, fn.get("name", "")))
        return found

    def _outputs(self) -> List[tuple]:
        """Every tool result in the history, in order: ``(message, key)``."""
        found = []
        for msg in self.history:
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "function_call_output":
                found.append((msg, "output"))
            elif msg.get("role") == "tool":
                found.append((msg, "content"))
        return found

    def _narrow(self) -> bool:
        """The context is full: turn the window on (``on_overflow``), or halve
        it, and cut again.  False when there is nothing left to narrow."""
        if self.on_overflow is None:
            return False
        if self.keep_calls is None:
            self.keep_calls = self.on_overflow
            self.window_from = self.requests
            self.user(FULL_CONTEXT_NOTE.format(keep=self.keep_calls))
        elif self.keep_calls > 1:
            self.keep_calls //= 2
        else:
            return False
        self._prune_calls()
        return True

    def _prune_calls(self) -> None:
        """Cut every call before the latest ``keep_calls`` (see the class).  A
        reply's results follow its calls in order, so the n-th call and the
        n-th result are one pair."""
        if self.keep_calls is None:
            return
        calls, outputs = self._calls(), self._outputs()
        pairs = list(zip(calls, outputs))
        whole = set(range(max(0, len(pairs) - self.keep_calls), len(pairs)))
        for name in WHOLE_FIRST:
            first = next((i for i, ((_, n), _) in enumerate(pairs) if n == name), None)
            if first is not None:
                whole.add(first)
        for i, ((holder, _), (msg, key)) in enumerate(pairs):
            if i in whole:
                continue
            if isinstance(msg.get(key), str):
                msg[key] = _short_result(msg[key])
            if "arguments" in holder:
                holder["arguments"] = _short_arguments(holder["arguments"])

    def outgoing(self) -> List[dict]:
        """The history as it is sent: our own ``_label`` keys stripped."""
        out = []
        for msg in self.history:
            parts = msg.get("content")
            if isinstance(parts, list):
                msg = {**msg, "content": [{k: v for k, v in part.items() if k != "_label"}
                                          for part in parts]}
            out.append(msg)
        return out


def conversation(gateway: Gateway, system: str, tools: List[dict], keep_images: int = 6,
                 max_output_tokens: int = 4096, keep_calls: Optional[int] = None,
                 on_overflow: Optional[int] = None) -> ModelClient:
    """A conversation in the request format of ``gateway``'s endpoint."""
    from ruikit.models.chat_completions import ChatCompletions
    from ruikit.models.responses import Responses

    kind = Responses if gateway.responses else ChatCompletions
    return kind(gateway, system, tools, keep_images=keep_images,
                max_output_tokens=max_output_tokens, keep_calls=keep_calls,
                on_overflow=on_overflow)
