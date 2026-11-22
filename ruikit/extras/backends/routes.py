"""Other models behind the model_ask and model_decide tools, reached through the model channels."""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

from ruikit.models.gateway import Gateway


# ----------------------------------------------------------------------------
# model_ask through a model channel
# ----------------------------------------------------------------------------
def route_model(channel: str) -> str:
    """The model a route asks: ``<channel>:$RUIKIT_ASK_MODEL``."""
    import os

    name = os.environ.get("RUIKIT_ASK_MODEL", "")
    if not name:
        from ruikit.core.result import ToolError

        raise ToolError("backend_unavailable", "which model answers model_ask / model_decide?",
                        hint="set RUIKIT_ASK_MODEL to a model of the route's channel")
    return f"{channel}:{name}"


class ModelAsk:
    """``model_ask``'s VLM: the pictures and the question, the text answer, from the
    model ``model`` (``<channel>:<name>``)."""

    def __init__(self, model: str, name: Optional[str] = None) -> None:
        self.name = name or model
        self.gw = Gateway(model)

    def ask(self, images: List[bytes], question: str, system: str = "") -> str:
        return self.gw.ask(question, images, system=system or "Answer briefly and concretely.")


class LLMDecide:
    """``model_decide`` answered by a general model instead of Jev: the same
    typed questions, asked as one prompt, the answers read back as JSON.  It
    sees no image either -- only the state the agent wrote."""

    SYSTEM = ("You answer typed questions about a situation described in text. Reply with JSON "
              "only: {\"<question id>\": {\"answer\": ..., \"confidence\": 0..1}, ...}. For a "
              "'noul' question the answer is true or false; for 'choice' and 'score' it is one of "
              "the options, copied exactly.")

    def __init__(self, model: str, name: str) -> None:
        self.name = name
        self.gw = Gateway(model)

    def decide(self, state, questions: Dict[str, dict]) -> dict:
        t0 = time.time()
        asked = {qid: {"type": q["type"], "question": q["instructions"],
                       **({"options": list(q["criteria"])} if q.get("criteria") else {})}
                 for qid, q in questions.items()}
        text = self.gw.ask("Situation:\n" + str(state) + "\n\nQuestions:\n"
                           + json.dumps(asked, ensure_ascii=False), system=self.SYSTEM)
        got = _json_object(text)
        answers = {}
        for qid, q in questions.items():
            item = got.get(qid) if isinstance(got.get(qid), dict) else {"answer": got.get(qid)}
            value, conf = item.get("answer"), item.get("confidence")
            key = {"noul": "noul", "choice": "choice", "score": "score"}[q["type"]]
            if key == "noul":
                value = bool(value) if not isinstance(value, str) else value.strip().lower() in (
                    "true", "yes", "1")
            elif value is not None:
                options = [str(o) for o in q.get("criteria") or ()]
                value = next((o for o in options if o.lower() == str(value).strip().lower()),
                             str(value))
            answers[qid] = {key: value, "confidence": conf}
        return {"answers": answers, "model": self.gw.model, "latency_s": round(time.time() - t0, 3)}


def _json_object(text: str) -> dict:
    """The first JSON object in a model's answer ({} when there is none)."""
    start, depth = text.find("{"), 0
    if start < 0:
        return {}
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            try:
                out = json.loads(text[start:i + 1])
            except ValueError:
                return {}
            return out if isinstance(out, dict) else {}
    return {}


#: the routes a second model can be asked by: the model channels; the model is RUIKIT_ASK_MODEL
ASK_ROUTES = ("openai", "anthropic", "local")


def register() -> None:
    from ruikit.extras.backends import ASKERS, DECIDERS

    for name in ASK_ROUTES:
        ASKERS[name] = (lambda n=name: ModelAsk(route_model(n), n))
        DECIDERS[name] = (lambda n=name: LLMDecide(route_model(n), n))
