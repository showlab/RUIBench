"""The design space: the tools every interface shares, an action abstraction each, an
interaction form deciding how they reach the model, and the components attached in the id."""


import pytest

from ruikit.actions import ACTIONS
from ruikit.core.tool import all_tools
from ruikit.extras import EXTRAS
from ruikit.interfaces import list_profiles, load_profile
from ruikit.interfaces.base import SHARED
from ruikit.interfaces.catalog import form_of
from ruikit.interfaces.roles import layout_problems, role_of

GENERATED = [i for i in list_profiles() if form_of(i)[0] in ("per_action", "program")]


def test_every_tool_has_one_place():
    assert layout_problems(all_tools()) == []
    assert len(SHARED) == 13 and all(role_of(n) == "shared" for n in SHARED)
    assert all(role_of(n) for n in all_tools())


@pytest.mark.parametrize("pid", GENERATED)
def test_an_interface_is_the_shared_tools_and_one_abstraction(pid):
    """One action a reply: the 13 shared tools and the abstraction's own.  A
    program over an abstraction: code_run and the task's tools, its programs
    calling exactly what the per-action interface over it offers."""
    form, action = form_of(pid)
    p = load_profile(pid)
    offered = set(p.tools) | set(p.code_tools)
    if form == "program":
        assert p.dialect == "program" and p.channel == "code"
        per_action = load_profile(f"rui.{action}/v1")
        assert p.code_tools == per_action.tools
        assert set(p.tools) - set(SHARED) == {"code_run"}
    else:
        assert p.dialect == action and set(SHARED) <= set(p.tools)
        assert offered - set(SHARED) == set(ACTIONS[action].tools), pid
    assert all(role_of(n) in ("shared", "action", "form") for n in offered), pid


def test_components_attach_to_any_interface_and_the_id_says_how():
    p = load_profile("rui.step/v1+marks:masks+memory:image@owlv2-sam2,depth=mono")
    assert p.id == "rui.step/v1+marks:masks+memory:image@owlv2-sam2,depth=mono"
    assert p.plugins == ("marks:masks", "memory:image")
    # the older names of the memory variants (until v1.16) still load, under the new ones
    assert load_profile("rui.pose/v1+memory:notes").id == "rui.pose/v1+memory:text"
    assert load_profile("rui.pose/v1+memory:all").id == "rui.pose/v1+memory"
    assert {"vision_marks", "memory_frames"} <= set(p.tools)
    assert "memory_note" not in p.tools                     # frames only
    assert p.tool_params("vision_marks") == {"style": "masks"}
    assert p.perception_settings()["backend"] == "owlv2-sam2"
    assert p.track == "controlled"
    # a program calls them: they go to what code_run may call
    code = load_profile("rui.code/v1+grasps:antipodal+make")
    assert "plan_grasps" in code.code_tools and "plan_grasps" not in code.tools
    assert "toolkit_define" in code.tools                   # never from inside a program
    assert code.tool_params("plan_grasps") == {"impl": "antipodal"}
    # the default implementation is not spelt out
    assert load_profile("rui.pose/v1+marks:dots").id == "rui.pose/v1+marks"
    # an outside model makes it an open-track interface
    ask = load_profile("rui.pose/v1+ask:openai")
    assert ask.track == "open" and ask.tool_params("model_ask") == {"backend": "openai"}
    vlm = load_profile("rui.point/v1+pointer:vlm/anthropic")
    assert vlm.tool_params("vision_point") == {"impl": "vlm", "backend": "anthropic"}


def test_what_cannot_be_offered_says_why():
    for bad in ("rui.pose/v1+nope", "rui.pose/v1+marks:circles", "rui.pose/v1+policy",
                "rui.pose/v1+imagine", "rui.pose/v1+reach:fast"):
        with pytest.raises(ValueError):
            load_profile(bad)
    assert EXTRAS["policy"].unavailable and EXTRAS["imagine"].unavailable

