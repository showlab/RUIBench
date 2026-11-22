"""Other models behind tools: Jev for discrete decisions, a VLM for a second look, a world
model for imagining.  Each is a pinned backend, charged its real latency and cost, and gets
only what the agent hands it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from ruikit.core.params import param
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import Cost, tool
from ruikit.extras.backends import ASKERS, DECIDERS, get_backend
from ruikit.harness.components.base import Component, Implementation, backends

#: model_decide's description when a general model answers instead of Jev
DECIDE_BY_MODEL = " ".join("""Quick discrete decisions by a second, general model: describe
    the situation in text and ask up to 20 typed questions -- yes/no, choose one option, or
    score on levels -- and get each answer with the model's own confidence. It cannot see
    images and knows nothing you do not tell it; use the confidence to decide whether to act
    or look again. Its real latency and price are charged.""".split())


@dataclass(frozen=True)
class Question:
    id: str = param("a short name for the question")
    type: Literal["noul", "choice", "score"] = param(
        "'noul': yes/no; 'choice': pick one of options; 'score': rate on the given levels")
    instructions: str = param("the question itself")
    options: Optional[List[str]] = param("choice: the options (up to 255); score: the levels",
                                         default=None)


@dataclass(frozen=True)
class DecideParams:
    state: str = param("the situation in words (Jev cannot see images: describe what you saw)")
    questions: List[Question] = param("up to 20 questions, answered independently",
                                      min_items=1, max_items=20)


@tool(name="model_decide",
      data="answers {question id: its answer}, model, latency_s",
      family="compute", params=DecideParams, source="model",
      cost=Cost("measured"), tracks=("open",),
      examples=[{"state": "Three pots; the left one wobbled when lifted.",
                 "questions": [{"id": "which", "type": "choice",
                                "instructions": "Which pot hides the egg?",
                                "options": ["left", "middle", "right"]}]}],
      description="""Fast discrete decisions by Jev (TypeSafe's 'System One' model, pinned to
      jev-1.13.0): describe the situation in text and ask up to 20 typed questions -- yes/no,
      choose one option, or score on levels -- and get each answer with per-option
      probabilities and a confidence, in about 0.1-0.5 s. It cannot see images and knows
      nothing you do not tell it; use the confidence to decide whether to act or look again.
      Its real latency and price are charged.""")
def model_decide(ctx, p: DecideParams) -> ToolResult:
    backend = get_backend(DECIDERS, "decision", ctx.params.get("backend", "jev"))
    questions = {}
    for q in p.questions:
        spec = {"type": q.type, "instructions": q.instructions}
        if q.type == "choice":
            if not q.options:
                raise ToolError("invalid_args", f"question {q.id!r}: a choice needs options")
            spec["criteria"] = {o: o for o in q.options}
        elif q.type == "score":
            if not q.options or len(q.options) < 2:
                raise ToolError("invalid_args", f"question {q.id!r}: a score needs >= 2 levels")
            spec["criteria"] = list(q.options)
        questions[q.id] = spec
    out = backend.decide(p.state, questions)
    answers = out.get("answers", {})
    return ToolResult(data={"answers": answers, "model": out.get("model"),
                            "latency_s": out.get("latency_s")},
                      text="; ".join(f"{k}: {v.get('choice', v.get('noul', v.get('score')))}"
                                     for k, v in answers.items()))


@dataclass(frozen=True)
class AskParams:
    frame_ids: List[str] = param("frames to show (at most 4): this episode's, or a demonstration's (demo<index>.<frame>.<camera>, as task_demo names them)", min_items=1, max_items=4)
    question: str = param("what to ask about them")


@tool(name="model_ask",
      data="answer",
      family="compute", params=AskParams, reads=("frames",), source="model",
      cost=Cost("measured"), tracks=("open",),
      examples=[{"frame_ids": ["f3.front", "f7.front"], "question": "What changed?"}],
      description="""Ask a second vision-language model about up to four of your frames, e.g.
      'what changed between these?' or 'is the ring on the right peg?' -- a demonstration's
      frames among them, to compare what a film showed with what is on the table. It sees only those
      pictures and your question and answers in text. Its real latency and price are charged;
      it is a second opinion, not a success checker.""")
def model_ask(ctx, p: AskParams) -> ToolResult:
    from ruikit.env.frames import encode_png
    from ruikit.perception.camera import _zoomable

    backend = get_backend(ASKERS, "vision-language", ctx.params.get("backend"))
    # a demonstration's frames too, as camera_zoom takes them: "which doll on
    # the table is the one the film lifted" could not be asked otherwise
    images = [encode_png(_zoomable(ctx, f)[2]) for f in p.frame_ids]
    if any(im is None for im in images):
        raise ToolError("not_supported", "these frames have no pixels (mock engine)")
    answer = backend.ask(images, p.question)
    return ToolResult(data={"answer": answer}, text=answer[:300])


@dataclass(frozen=True)
class ImagineParams:
    frame_id: str = param("the frame to start from")
    actions: List[List[float]] = param("rows of [dx, dy, dz, droll, dpitch, dyaw, gripper]",
                                       min_items=1, max_items=32)


@tool(name="model_imagine", family="compute", params=ImagineParams, reads=("frames",),
      source="model", cost=Cost("measured"), tracks=("open",),
      examples=[{"frame_id": "f3.front", "actions": [[0.01, 0, 0, 0, 0, 0, 0]]}],
      description="""Imagine the future with a world model: from one frame and a sequence of
      gripper moves, predict the frames that would follow. Slow (seconds per step) and only as
      good as the model; it predicts, it does not move the robot. Needs a world-model backend
      (e.g. V-JEPA 2-AC, Cosmos) on the tool server.""")
def model_imagine(ctx, p: ImagineParams) -> ToolResult:
    raise ToolError("backend_unavailable", "no world-model backend is configured")


#: the asking models, by their route in ``ruikit.extras.backends.routes``: (name, zh, en)
ASK_MODELS = (("openai", "OpenAI 的模型（RUIKIT_ASK_MODEL）", "an OpenAI model (RUIKIT_ASK_MODEL)"),
              ("anthropic", "Anthropic 的模型（RUIKIT_ASK_MODEL）", "an Anthropic model (RUIKIT_ASK_MODEL)"),
              ("local", "本地 vLLM 模型（RUIKIT_ASK_MODEL）", "a local vLLM model (RUIKIT_ASK_MODEL)"))

ASK = Component("ask", "问别的模型", "a second vision-language model, for a second look",
                ("model_ask",), backends("model_ask", ASK_MODELS, "the model route it names"),
                default="openai")
DECIDE = Component(
    "decide", "快速判断", "typed quick decisions by a small model",
    ("model_decide",),
    (Implementation("jev", "Jev（TypeSafe）", "Jev, TypeSafe's System One",
                    params={"model_decide": {"backend": "jev"}}, needs="TYPESAFE_API_KEY"),)
    + backends("model_decide", ASK_MODELS, "the model route it names"),
    default="openai")
IMAGINE = Component("imagine", "想象", "predict the next pictures with a world model",
                    ("model_imagine",),
                    unavailable="no world model conditioned on the robot's moves runs here")
