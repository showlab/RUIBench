"""The tool contract and the profiles, as tests (approach-v2 §4, §5)."""

import re

import pytest

from ruikit.core.check import check_all
from ruikit.core.params import ArgError, parse_args
from ruikit.core.tool import FAMILIES, all_tools
from ruikit.interfaces import list_profiles, load_profile
from ruikit.serve.schemas import anthropic_tools, gemini_tools, mcp_tools, openai_tools

BASE = ("rui.step/v1", "rui.pose/v1", "rui.point/v1", "rui.code/v1")


def test_every_tool_passes_its_contract():
    bad = {k: v for k, v in check_all().items() if v}
    assert not bad, bad


def test_every_family_has_tools():
    families = {t.spec.family for t in all_tools().values()}
    assert families == set(FAMILIES)


def test_names_are_legal_for_every_provider():
    # Anthropic ^[a-zA-Z0-9_-]{1,128}$, OpenAI <= 64 chars: no dots anywhere
    for name in all_tools():
        assert re.match(r"^[a-z][a-z0-9_]{0,63}$", name), name


def test_strict_schemas_are_strict():
    for name, t in all_tools().items():
        schema = t.spec.input_schema(strict=True)
        assert schema["additionalProperties"] is False, name
        assert set(schema.get("required", [])) == set(schema["properties"]), name


def test_bad_arguments_are_named():
    params = all_tools()["robot_move_to"].spec.params
    with pytest.raises(ArgError) as err:
        parse_args(params, {"xyz_m": [0.4, 0.0]})
    assert "xyz_m" in str(err.value)
    with pytest.raises(ArgError) as err:
        parse_args(params, {"xyz_m": [0.4, 0.0, 0.1], "xzy": 1})
    assert "unknown argument" in str(err.value)


def test_an_oracle_tool_cannot_be_offered_on_a_leaderboard_track():
    from dataclasses import replace

    spec = all_tools()["robot_state"].spec
    with pytest.raises(ValueError, match="oracle"):
        replace(spec, source="oracle", tracks=("controlled", "diagnostic"))
    replace(spec, source="oracle", tracks=("diagnostic",))       # fine off the leaderboard


@pytest.mark.parametrize("pid", list_profiles())
def test_profiles_load(pid):
    load_profile(pid)


def test_base_profiles_differ_only_in_the_act_group():
    """I1-I3 offer the same observe / perceive / task tools; only how they act differs."""
    shared = None
    for pid in ("rui.step/v1", "rui.pose/v1", "rui.point/v1"):
        tools = all_tools()
        p = load_profile(pid)
        non_act = {n for n in p.tools if tools[n].spec.family != "act"}
        shared = non_act if shared is None else shared
        assert non_act == shared, pid


def test_pose_and_program_offer_the_same_calls():
    """The cleanest pair: I4 can call exactly what I2 can (v0 §4)."""
    pose, code = load_profile("rui.pose/v1"), load_profile("rui.code/v1")
    assert set(code.code_tools) == set(pose.tools)
    assert code.channel == "code" and pose.channel == "tools"


def test_base_profiles_offer_no_task_solver_and_no_model():
    tools = all_tools()
    for pid in BASE:
        p = load_profile(pid)
        for n in p.tools + p.code_tools:
            assert tools[n].spec.source not in ("oracle", "model"), (pid, n)
            assert not n.startswith(("skill_", "policy_", "model_", "plan_")), (pid, n)


def test_exports_keep_the_profile_order():
    p = load_profile("rui.pose/v1")
    for export in (anthropic_tools(p), openai_tools(p), mcp_tools(p)):
        assert [t["name"] for t in export] == list(p.tools)
    decls = gemini_tools(p)[0]["function_declarations"]
    assert [d["name"] for d in decls] == list(p.tools)
    assert all("additionalProperties" not in d["parameters"] for d in decls)
