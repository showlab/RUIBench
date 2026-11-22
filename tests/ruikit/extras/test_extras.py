"""The extras: tools outside the design space."""





def test_a_general_model_answers_the_quick_questions(monkeypatch):
    from ruikit.extras.backends.routes import LLMDecide
    from ruikit.models.gateway import Gateway

    asked = {}

    def fake_ask(self, question, images=(), system=""):
        asked["q"], asked["s"] = question, system
        return ('{"which": {"answer": "Middle", "confidence": 0.8}, '
                '"light": {"answer": "yes", "confidence": 0.6}}')

    monkeypatch.setattr(Gateway, "__init__", lambda self, model, **kw: setattr(self, "model", model))
    monkeypatch.setattr(Gateway, "ask", fake_ask)
    decider = LLMDecide("local:m", "local")
    out = decider.decide("three pots", {
        "which": {"type": "choice", "instructions": "which pot?",
                  "criteria": {"left": "left", "middle": "middle", "right": "right"}},
        "light": {"type": "noul", "instructions": "is the left one light?"}})
    assert out["answers"]["which"] == {"choice": "middle", "confidence": 0.8}
    assert out["answers"]["light"] == {"noul": True, "confidence": 0.6}
    assert "which pot?" in asked["q"] and "JSON" in asked["s"]

