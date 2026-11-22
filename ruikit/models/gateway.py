"""A model channel over HTTP: requests with retries and back-off, and one re-authentication."""

from __future__ import annotations

import base64
import email.utils
import hashlib
import http.client
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence

from ruikit.core.result import ToolError
from ruikit.models import settings as _settings
from ruikit.models.base import GatewayDown, _message
from ruikit.models.channels import (
    LOCAL_BASE_URL,
    LOCAL_MODELS,
    QWENCLOUD_BASE_URL,
    QWENCLOUD_MODELS,
    QWENTOKENPLAN_BASE_URL,
    channels,
    keyed,
)
from ruikit.models.settings import load_settings


class Gateway:
    """One model on one channel: OpenAI, Anthropic, a local vLLM server or QwenCloud."""

    def __init__(self, model: str, base_url: Optional[str] = None,
                 timeout_s: Optional[float] = None, retries: int = 5, rate_retries: int = 10,
                 backoff_s: float = 1.0) -> None:
        provider, _, name = model.rpartition(":")
        self.provider = provider or channels()["default_channel"]
        settings = load_settings()
        env = channels()["channels"].get(self.provider, {}).get("base_url_env", "")
        #: sampling sent with each request, over the server's defaults (``RUIKIT_LOCAL_SAMPLING``,
        #: JSON, the local channel only: e.g. the model card's values for thinking)
        self.sampling: dict = {}
        #: a file each answer's whole thinking is added to as it comes (``_keep_thinking``), or None
        self.trace = None
        #: a file each request and its whole answer is added to (``_keep_request``), or None
        self.requests_log = None
        self.key = ""                       # QwenCloud's API key
        #: QwenCloud's Token Plan, asked first when it is on (``RUIKIT_QWENCLOUD_PLAN``):
        #: its endpoint and key; ``plan_state``, a file every worker reads, says until when
        #: it rests (its quota spent, its key refused); the answers per endpoint so far
        self.plan: Optional[dict] = None
        self.plan_state: Optional[Path] = None
        self.endpoints = {"tokenplan": 0, "payg": 0}
        self.stream = False                 # ask for the answer streamed (QwenCloud)
        #: put each answer's thinking back into the history it is sent with
        #: (``RUIKIT_PASSBACK_THINKING``; QwenCloud and the local vLLM), in the field the
        #: endpoint reads; ``_thoughts`` holds it by tool call id.  Once the conversation's
        #: window is on (a full context), only the latest ``passback_window`` answers keep it
        self.passback = settings.get("RUIKIT_PASSBACK_THINKING", "0") not in ("", "0", "false")
        self.thought_field = "reasoning_content"
        self.passback_window = int(settings.get("RUIKIT_PASSBACK_WINDOW") or 8)
        self._thoughts: dict = {}
        self._tools_sha = ""
        if self.provider == "local":
            self.model = LOCAL_MODELS.get(name, name)
            self.base_url = (base_url or settings.get(env) or LOCAL_BASE_URL).rstrip("/")
            self.responses = False
            self.thinking = settings.get("RUIKIT_LOCAL_THINKING", "0") not in ("", "0", "false")
            self.sampling = json.loads(settings.get("RUIKIT_LOCAL_SAMPLING") or "{}")
            # Qwen3.8's chat template renders an earlier answer's thinking from ``reasoning``
            # (not ``reasoning_content``), while ``preserve_thinking`` is on -- its default
            self.thought_field = "reasoning"
        elif self.provider == "qwencloud":
            # the pay-as-you-go key, and the Token Plan's first when it is on (its terms allow
            # interactive use in coding and agent tools only: the user's call, 10-25)
            self.model = QWENCLOUD_MODELS.get(name, name)
            self.base_url = (base_url or settings.get(env) or QWENCLOUD_BASE_URL).rstrip("/")
            self.responses = False
            self.key = settings.get("QWENCLOUD_API_KEY", "")
            if not self.key:
                raise ToolError("backend_unavailable", "no QwenCloud key",
                                hint=f"set QWENCLOUD_API_KEY in {_settings.SECRETS}")
            # thinking is the model's default; the request's own fields (``enable_thinking``,
            # ``reasoning_effort``, sampling) from ``RUIKIT_QWENCLOUD_SAMPLING``, JSON
            self.sampling = {"enable_thinking": True,
                             **json.loads(settings.get("RUIKIT_QWENCLOUD_SAMPLING") or "{}")}
            self.thinking = bool(self.sampling["enable_thinking"])
            self.sampling.setdefault("preserve_thinking", self.passback)
            if settings.get("RUIKIT_QWENCLOUD_PLAN", "0") not in ("", "0", "false"):
                key = settings.get("QWENTOKENPLAN_API_KEY", "")
                if not key:
                    raise ToolError("backend_unavailable", "no Token Plan key",
                                    hint=f"set QWENTOKENPLAN_API_KEY in {_settings.SECRETS}")
                self.plan = {"url": (settings.get("RUIKIT_QWENTOKENPLAN_URL")
                                     or QWENTOKENPLAN_BASE_URL).rstrip("/"), "key": key}
                state = settings.get("RUIKIT_QWENCLOUD_PLAN_STATE", "")
                self.plan_state = Path(state) if state else None
                self._plan_rests_until = 0.0
            # streamed, as QwenCloud advises for thinking: the timeout is then the longest
            # silence, not the longest answer
            self.stream = True
        elif keyed(self.provider):
            # OpenAI, Anthropic (its OpenAI-compatible endpoint) or another such endpoint, with
            # the channel's own key from the environment or the secrets file
            conf = keyed(self.provider)
            self.model = dict(conf.get("deployments") or {}).get(name, name)
            self.base_url = (base_url or settings.get(env) or conf.get("base_url") or "").rstrip("/")
            self.responses = self.model.startswith(tuple(conf.get("responses_prefixes") or ()))
            self.key = settings.get(conf["api_key_env"], "")
            if not self.key:
                raise ToolError("backend_unavailable", f"no key for {conf.get('label', self.provider)}",
                                hint=f"set {conf['api_key_env']} in the environment or in {_settings.SECRETS}")
        else:
            raise ToolError("backend_unavailable", f"unknown model channel {provider!r}",
                            hint="<channel>:<model> with a channel of ruikit/configs/models.yaml: "
                                 + ", ".join(channels()["channels"]))
        self.passback = self.passback and self.provider in ("local", "qwencloud")
        if not self.base_url:
            raise ToolError("backend_unavailable", f"no endpoint for {self.provider}",
                            hint=f"set {env or 'its base_url'} in the environment or in {_settings.SECRETS}")
        # a reply may take this long (``RUIKIT_MODEL_TIMEOUT``): a thinking model's can pass 300 s
        self.timeout_s = (timeout_s if timeout_s is not None
                          else float(settings.get("RUIKIT_MODEL_TIMEOUT") or 300.0))
        self.retries = retries              # a busy server: 408, 409, 425, 5xx
        self.rate_retries = rate_retries    # a rate limit: 429
        self.backoff_s = backoff_s          # the first pause; it doubles up to a minute

    @property
    def openai_dialect(self) -> bool:
        """OpenAI's own API (a channel's ``openai_dialect``): max_completion_tokens, no
        custom temperature."""
        return bool(keyed(self.provider).get("openai_dialect"))

    def _keep_thinking(self, body: dict, seconds: float) -> None:
        """One line for an answer: its whole thinking (the transcript keeps the first
        20,000 characters), why it stopped, its tokens and how long it took -- added
        as it comes, so an episode cut off keeps what was thought until then."""
        choice = (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
        msg = choice.get("message") or {}
        row = {"t": round(time.time(), 3), "seconds": round(seconds, 3),
               "stop": choice.get("finish_reason"), "usage": body.get("usage"),
               "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
               "content": msg.get("content") or "",
               "calls": [(tc.get("function") or {}).get("name") for tc in msg.get("tool_calls") or ()]}
        with open(self.trace, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _keep_request(self, payload: dict, body: dict, seconds: float, headers,
                      endpoint: Optional[str] = None) -> None:
        """One line for a request: what was sent, its pictures as files in ``inputs/`` beside
        the log (one file a picture, named by its hash), the tools only when they change;
        and the whole answer as it came, with the endpoint's request id."""
        folder = Path(self.requests_log).parent / "inputs"
        messages = []
        for msg in payload.get("messages") or ():
            parts = msg.get("content")
            if isinstance(parts, list):
                msg = {**msg, "content": [_stored(part, folder) for part in parts]}
            messages.append(msg)
        sent = {k: v for k, v in payload.items() if k not in ("messages", "tools")}
        tools = json.dumps(payload.get("tools") or [], sort_keys=True)
        sha = hashlib.sha256(tools.encode("utf-8")).hexdigest()[:16]
        sent["tools_sha"] = sha
        if sha != self._tools_sha:
            sent["tools"] = payload.get("tools") or []
            self._tools_sha = sha
        row = {"t": round(time.time(), 3), "seconds": round(seconds, 3),
               **({"endpoint": endpoint} if endpoint else {}),
               "headers": {k.lower(): v for k, v in (headers or {}).items()
                           if k.lower().startswith(("x-", "req-"))},
               "request": {**sent, "messages": messages}, "response": body}
        with open(self.requests_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _on_plan(self) -> bool:
        """Ask the Token Plan now?  Not while it rests (here, or for every worker in
        ``plan_state``)."""
        if self.plan is None or time.time() < self._plan_rests_until:
            return False
        if self.plan_state is not None and self.plan_state.exists():
            try:
                until = float(json.loads(self.plan_state.read_text()).get("rests_until", 0))
            except (OSError, ValueError):
                until = 0.0
            if time.time() < until:
                self._plan_rests_until = until
                return False
        return True

    def _rest_plan(self, seconds: float, why: str) -> None:
        """The Token Plan refused (its quota, its key): the pay-as-you-go endpoint for
        ``seconds``, said in the request log and, for every worker, in ``plan_state``."""
        until = time.time() + seconds
        self._plan_rests_until = until
        note = {"rests_until": round(until, 3), "why": why[:300],
                "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        if self.plan_state is not None:
            tmp = self.plan_state.with_name(f".{self.plan_state.name}.{os.getpid()}")
            tmp.write_text(json.dumps(note))
            os.replace(tmp, self.plan_state)
        if self.requests_log is not None:
            with open(self.requests_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": round(time.time(), 3), "token_plan_rests": note}) + "\n")

    def _flagged(self, said: str, before: int, seconds: float) -> bool:
        """QwenCloud's content filter stopped a sampled answer ("Output data may contain
        inappropriate content"): ask again, up to ``retries`` times, each one noted in the
        request log.  A filtered request (its input) is not asked again."""
        if self.provider != "qwencloud" or before >= self.retries:
            return False
        if "inappropriate" not in said.lower() or "output" not in said.lower():
            return False
        if self.requests_log is not None:
            with open(self.requests_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": round(time.time(), 3), "seconds": round(seconds, 3),
                                     "filtered": said[:300], "asked_again": before + 1}) + "\n")
        return True

    def _remember(self, body: dict) -> None:
        """An answer's thinking, by its tool calls' ids (or its words, with none)."""
        msg = ((body.get("choices") or [{}])[0] or {}).get("message") or {}
        thought = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if not thought:
            return
        ids = [tc.get("id") for tc in msg.get("tool_calls") or () if tc.get("id")]
        for key in ids or ["text:" + (msg.get("content") or "").strip()]:
            self._thoughts[key] = thought

    def _with_thoughts(self, payload: dict) -> dict:
        """The history with each earlier answer's thinking back in it, in
        ``thought_field`` (never inside ``content``); once the window is on, the
        latest ``passback_window`` answers' only."""
        from ruikit.models.conversation import FULL_CONTEXT_NOTE

        note = FULL_CONTEXT_NOTE.split("{keep}")[0]
        history = payload.get("messages") or []
        answers = [k for k, m in enumerate(history) if m.get("role") == "assistant"]
        windowed = any(note in _words(m.get("content")) for m in history if m.get("role") == "user")
        kept = set(answers[-self.passback_window:]) if windowed else set(answers)
        field = self.thought_field
        messages = []
        for k, msg in enumerate(history):
            if k in kept and field not in msg:
                ids = [tc.get("id") for tc in msg.get("tool_calls") or ()]
                key = ids[0] if ids else "text:" + (msg.get("content") or "").strip()
                if key in self._thoughts:
                    msg = {**msg, field: self._thoughts[key]}
            messages.append(msg)
        out = {**payload, "messages": messages}
        if self.provider == "local":           # said, though it is the template's default
            out["chat_template_kwargs"] = {**(payload.get("chat_template_kwargs") or {}),
                                           "preserve_thinking": True}
        return out

    def _auth(self, fresh: bool = False) -> str:
        return f"Bearer {self.key}" if self.key else "Bearer local"

    # -- HTTP ------------------------------------------------------------------
    def post(self, path: str, payload: dict, tries: Optional[dict] = None) -> dict:
        """POST ``payload``, retrying what a retry can fix: a busy server up to
        ``retries`` times; a rate limit up to ``rate_retries`` times.  The pauses
        double from ``backoff_s`` up to a minute, and are never shorter than the
        server asks -- a limit per minute may ask for a second only.
        ``tries``, when given, gets ``waited_s`` -- the seconds before the try
        that answered went out: minting a token, the failed tries, the pauses
        between them (none of it the model's time) -- and ``retries``, the
        tries beyond the first."""
        url = f"{self.base_url}{path}"
        if self.sampling and path == "/chat/completions":
            payload = {**payload, **self.sampling}
        if self.passback and path == "/chat/completions":
            payload = self._with_thoughts(payload)
        stream = self.stream and path == "/chat/completions"
        if stream:
            payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        data = json.dumps(payload).encode("utf-8")
        delay = self.backoff_s
        fresh = False
        attempt = busy = limited = dropped = flagged = 0
        t0 = time.time()
        while True:
            on_plan = self._on_plan()
            req = urllib.request.Request(
                f"{self.plan['url']}{path}" if on_plan else url, data=data, method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.plan['key']}" if on_plan
                         else self._auth(fresh)})
            sent = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    headers = dict(resp.headers.items())
                    body = (_streamed(resp) if stream
                            else json.loads(resp.read().decode("utf-8")))
                if isinstance(body, dict) and body.get("error"):
                    said = _message(body["error"])
                    if on_plan and _plan_spent(said):
                        self._rest_plan(_plan_rest(429, said), said)
                        attempt += 1
                        continue
                    if self._flagged(said, flagged, time.time() - sent):
                        flagged += 1
                        attempt += 1
                        continue
                    raise ToolError("tool_failed", f"{self.model}: {said}")
                if tries is not None:
                    tries.update(waited_s=round(sent - t0, 3), retries=attempt)
                if self.trace is not None:
                    self._keep_thinking(body, time.time() - sent)
                endpoint = "tokenplan" if on_plan else "payg"
                if self.plan is not None:
                    self.endpoints[endpoint] += 1
                if self.requests_log is not None:
                    self._keep_request(payload, body, time.time() - sent, headers,
                                       endpoint if self.plan is not None else None)
                if self.passback:
                    self._remember(body)
                return body
            except (TimeoutError, ConnectionError, http.client.IncompleteRead) as exc:
                if not stream:
                    raise
                busy += 1                         # a stream cut off: the whole answer again
                if busy > self.retries:
                    raise ToolError("tool_failed", f"{self.model}: the answer's stream broke "
                                    f"{busy} times: {exc}")
                time.sleep(delay)
                delay = min(60.0, delay * 2)
                attempt += 1
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:800]
                if on_plan and (exc.code in (401, 402, 403, 404, 429) or exc.code >= 500
                                or (_plan_spent(detail) and not _too_long(detail))):
                    # its quota or its key: the pay-as-you-go endpoint, at once -- for hours
                    # when the key is refused, the model missing or the month's quota spent
                    # ("429 insufficient_quota"), a minute when it is only asked too often
                    # ("429 Requests rate limit exceeded"), a while otherwise
                    self._rest_plan(_plan_rest(exc.code, detail), f"HTTP {exc.code}: {detail}")
                    attempt += 1
                    continue
                if exc.code == 400 and self._flagged(detail, flagged, time.time() - sent):
                    flagged += 1
                    attempt += 1
                    continue
                if exc.code == 400 and self.provider == "qwencloud" and _too_long(detail):
                    # in the words ``conversation._full_context`` knows: the window takes over
                    detail = f"input is too long for the model's context window: {detail}"
                if exc.code == 402 or "quota_exceeded" in detail:
                    raise ToolError("rate_limited", f"{self.model}: the {self.provider} quota is "
                                    "used up for this month", hint=detail[:300])
                # QwenCloud's own fetch of the request's pictures timing out comes as a 400,
                # though the request is sound: sent again as a busy server's would be
                if (exc.code in (408, 409, 425, 429) or exc.code >= 500
                        or (exc.code == 400 and self.provider == "qwencloud"
                            and _fetch_timed_out(detail))):
                    if exc.code == 429:
                        limited += 1
                        again = limited <= self.rate_retries
                    else:
                        busy += 1
                        again = busy <= self.retries
                    if again:
                        wait = min(60.0, max(_retry_after(exc.headers), delay))
                        time.sleep(wait * (1 + 0.2 * random.random()))   # apart from other runs
                        delay = min(60.0, delay * 2)
                        attempt += 1
                        continue
                    raise ToolError("rate_limited" if exc.code == 429 else "tool_failed",
                                    f"{self.model}: HTTP {exc.code} after {attempt + 1} tries: {detail}")
                raise ToolError("tool_failed", f"{self.model}: HTTP {exc.code}: {detail}")
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", exc)
                dropped += 1
                if isinstance(reason, ConnectionRefusedError) or dropped > 1:
                    raise GatewayDown(url, str(reason), self.provider)
                time.sleep(delay)
                delay = min(60.0, delay * 2)
                attempt += 1

    def models(self) -> List[str]:
        """The models the endpoint lists."""
        req = urllib.request.Request(f"{self.base_url}/models",
                                     headers={"Authorization": self._auth()})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ToolError("tool_failed", f"{self.base_url}/models: HTTP {exc.code}: "
                            f"{exc.read().decode('utf-8', 'replace')[:300]}")
        except urllib.error.URLError as exc:
            raise GatewayDown(f"{self.base_url}/models", str(getattr(exc, "reason", exc)),
                              self.provider)
        return sorted(m.get("id", "") for m in body.get("data", []))

    def ask(self, question: str, images: Sequence[bytes] = (), system: str = "") -> str:
        """One question, no tools: the text answer."""
        from ruikit.models.conversation import conversation

        conv = conversation(self, system=system, tools=[])
        conv.user(question, images)
        return conv.ask().text


def _streamed(resp) -> dict:
    """A streamed chat completion put back together as the one body a request
    without streaming gets: the words, the thinking, the tool calls (their parts
    joined by index), why it stopped, the usage (the last chunk's)."""
    msg = {"role": "assistant", "content": "", "reasoning_content": ""}
    calls: dict = {}
    head: dict = {}
    finish = usage = None
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            return chunk
        head = head or {k: chunk[k] for k in ("id", "model", "created") if k in chunk}
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or ():
            delta = choice.get("delta") or {}
            msg["content"] += delta.get("content") or ""
            msg["reasoning_content"] += delta.get("reasoning_content") or delta.get("reasoning") or ""
            for tc in delta.get("tool_calls") or ():
                call = calls.setdefault(tc.get("index", len(calls)),
                                        {"id": "", "type": "function",
                                         "function": {"name": "", "arguments": ""}})
                call["id"] = tc.get("id") or call["id"]
                fn = tc.get("function") or {}
                call["function"]["name"] += fn.get("name") or ""
                call["function"]["arguments"] += fn.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    if calls:
        msg["tool_calls"] = [calls[k] for k in sorted(calls)]
    return {**head, "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": usage}


def _words(content) -> str:
    """A message's text, whether a string or a list of parts."""
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content or () if isinstance(p, dict))


def _stored(part: dict, folder: Path) -> dict:
    """A picture sent inline, as the file it is kept in (``inputs/<hash>.png``)."""
    url = (part.get("image_url") or {}).get("url", "") if part.get("type") == "image_url" else ""
    if not url.startswith("data:"):
        return part
    head, _, b64 = url.partition(",")
    data = base64.b64decode(b64)
    name = f"{hashlib.sha256(data).hexdigest()[:24]}.{'png' if 'png' in head else 'jpg'}"
    if not (folder / name).exists():
        folder.mkdir(exist_ok=True)
        (folder / name).write_bytes(data)
    return {**part, "image_url": {**part["image_url"], "url": f"inputs/{name}"}}


#: how long the Token Plan rests (the pay-as-you-go endpoint answers meanwhile): after a busy
#: server or an unknown refusal; after its month's quota is spent, its key refused or the model
#: missing; after it is only asked too often
PLAN_REST_S, PLAN_KEY_REST_S, PLAN_RATE_REST_S = 15 * 60.0, 6 * 3600.0, 60.0


def _plan_rest(code: int, detail: str) -> float:
    text = detail.lower()
    if code in (401, 402, 403, 404) or "insufficient_quota" in text or "exhausted" in text:
        return PLAN_KEY_REST_S
    if "rate limit" in text:
        return PLAN_RATE_REST_S
    return PLAN_REST_S


def _plan_spent(detail: str) -> bool:
    """The Token Plan saying its credits are spent for now."""
    text = detail.lower()
    return any(w in text for w in ("quota", "credit", "exceeded", "insufficient", "throttl",
                                   "limit reached", "usage limit"))


def _too_long(detail: str) -> bool:
    """QwenCloud refusing a request as longer than the model takes."""
    text = detail.lower()
    return ("input length" in text or "too long" in text
            or ("exceed" in text and "token" in text))


def _fetch_timed_out(detail: str) -> bool:
    """QwenCloud failing to fetch a request's pictures in time ("Download multimodal
    file timed out", an ``invalid_parameter_error``): its own hiccup, not the request's."""
    return "download multimodal file timed out" in detail.lower()


def _retry_after(headers) -> float:
    """The pause a refusal asks for, in seconds: Azure's ``retry-after-ms``,
    else ``Retry-After`` (seconds or an HTTP date); 0 when it asks none."""
    if headers is None:
        return 0.0
    try:
        if headers.get("retry-after-ms"):
            return max(0.0, float(headers.get("retry-after-ms")) / 1000.0)
    except ValueError:
        pass
    value = headers.get("Retry-After")
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, email.utils.parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return 0.0
