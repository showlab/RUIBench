"""The aids: the components every interface of the main table carries,
taken away one group at a time in experiment two -- and RUI-Free, the column
with every action abstraction's tools at once.  The aids help the agent remember, see,
find, follow and plan; none moves the arm another way or asks another model."""

import re

import pytest

from ruikit.actions import ACTIONS
from ruikit.core.tool import all_tools
from ruikit.harness.components.aids import AID_GROUPS, AIDS_LEFT_OUT, aid_tools, parse_aids
from ruikit.interfaces import interface_for, load_profile
from ruikit.interfaces.base import SHARED
from ruikit.interfaces.compose import attachable
from ruikit.serve.schemas import code_reference, tool_entries

INTERFACES = sorted([interface_for("per_action", a) for a in ACTIONS]
                    + [interface_for("program", "pose")])


def _moves(prof):
    """What can move an arm: the tools that command one, a program, a made-up tool."""
    tools = all_tools()
    return {n for n in set(prof.tools) | set(prof.code_tools)
            if tools[n].spec.controls != "none" or n in ("code_run", "toolkit_define")}


@pytest.mark.parametrize("pid", INTERFACES)
def test_the_aids_add_no_way_of_moving_the_arm(pid):
    """With the aids, an interface still moves the arm only as it did alone --
    otherwise the columns of the main table would blur into each other."""
    bare, aided = load_profile(pid), load_profile(pid + "+aids")
    assert _moves(aided) == _moves(bare), (pid, _moves(aided) - _moves(bare))
    assert aided.track == "controlled" and aided.dialect == bare.dialect
    offered = set(aided.tools) | set(aided.code_tools)
    assert set(aid_tools()) <= offered
    tools = all_tools()
    assert not [n for n in offered if tools[n].spec.source == "model"]      # no other model
    left_out = {t for name in AIDS_LEFT_OUT for t in attachable()[name].tools}
    assert not offered & left_out, offered & left_out


def test_a_program_calls_the_aids_and_ends_the_episode_itself():
    p = load_profile("rui.code/v1+aids")
    assert set(p.tools) == {"code_run", "task_info", "task_demo", "task_events", "task_done",
                            "task_give_up"}
    assert {"plan_grasps", "vision_marks", "memory_note"} <= set(p.code_tools)
    assert "task_give_up" not in p.code_tools


@pytest.mark.parametrize("group", sorted(AID_GROUPS))
def test_one_group_taken_away(group):
    """Experiment two: ``+aids-<group>`` is the aids less exactly that group."""
    full, less = load_profile("rui.pose/v1+aids"), load_profile(f"rui.pose/v1+aids-{group}")
    gone = {t for name in AID_GROUPS[group] for t in attachable()[name].tools}
    assert set(full.tools) - set(less.tools) == gone
    assert less.id == f"rui.pose/v1+aids-{group}" and less.plugins == (f"aids-{group}",)
    assert "task_give_up" in less.tools                    # whatever is taken away


def test_the_aids_are_named_one_way():
    assert load_profile("rui.pose/v1+aids@owlv2-sam2").id == "rui.pose/v1+aids@owlv2-sam2"
    assert load_profile("rui.step/v1+aids-plan").tool_params("robot_step")["micro_m"] == 0.0025
    for bad in ("rui.pose/v1+aids-all", "rui.pose/v1+aids-memory-memory", "rui.pose/v1+aids-ask"):
        with pytest.raises(ValueError):
            load_profile(bad)
    assert parse_aids("aids-track") == ("track",)


def test_paths_plans_and_the_interface_moves():
    """The paths component plans; robot_waypoints is the motion extra's."""
    assert set(load_profile("rui.pose/v1+paths").tools) - set(load_profile("rui.pose/v1").tools) \
        == {"plan_motion"}
    assert "robot_waypoints" in load_profile("rui.pose/v1+paths+motion").tools


def test_rui_free_offers_every_abstraction_a_program_and_the_aids():
    p = load_profile("rui.free/v1")
    every = {t for a in ACTIONS.values() for t in a.tools} | {"code_run"}
    assert every <= set(p.tools)
    assert set(SHARED) | set(aid_tools()) <= set(p.tools)
    assert set(p.tools) == set(SHARED) | every | set(aid_tools())   # nothing else
    assert set(p.code_tools) == set(p.tools) - {"code_run", "task_give_up"}
    assert p.track == "controlled" and p.dialect == ""
    step = load_profile("rui.step/v1")
    assert p.tool_params("robot_step") == step.tool_params("robot_step")
    aided = load_profile("rui.pose/v1+aids")
    for name in aid_tools():                     # the aids as everywhere else
        assert p.tool_params(name) == aided.tool_params(name), name


@pytest.mark.parametrize("pid", INTERFACES + [i + "+aids" for i in INTERFACES] + ["rui.free/v1"])
def test_a_description_names_only_tools_on_offer(pid):
    """An aid on an interface without robot_move_to does not tell the agent to use it."""
    p = load_profile(pid)
    offered = set(p.tools) | set(p.code_tools)
    lacking = set(all_tools()) - offered
    texts = [e["description"] for e in tool_entries(p)]
    if p.channel == "code":
        texts.append(code_reference(p))
    named = {m for t in texts for m in lacking if re.search(r"\b" + m + r"\b", t)}
    assert not named, (pid, named)


@pytest.mark.parametrize("pid", ["rui.free/v1", "rui.pose/v1+aids", "rui.code/v1+aids",
                                 "rui.skill/v1"])
def test_every_reply_that_moved_an_arm_shows_the_cameras(pid):
    """v1.17 (the user, 10/11): every interface's reply that moved an arm comes
    back with the cameras' pictures, as a step did in Show-Harness; a reply that
    only looked does not; ``+pull`` leaves the looking to the agent."""
    from ruikit import Episode

    def move(ep):
        tcp = ep.station.describe_pose(ep.engine.state().arm(ep.scene.arm_ids[0]).eef_pose)["xyz_m"]
        to = [tcp[0], tcp[1], tcp[2] + 0.01]
        if pid.startswith("rui.code"):
            return ep.call("code_run", {"source": f"robot.move_to(xyz_m={to})\nresult = 1"})
        if pid.startswith("rui.skill"):
            return ep.call("skill_move", {"target": {"xyz_m": to}})
        return ep.call("robot_move_to", {"xyz_m": to})

    for profile, pushes in ((pid, True), (pid.replace("/v1", "/v1+pull", 1), False)):
        ep = Episode("WK1", level="easy", seed=0, profile=profile, engine="mock")
        try:
            looked = ep.call("task_info", {})
            assert "pushed" not in (looked.data or {})
            r = move(ep)
            assert not r.is_error, r.text
            assert ("pushed" in (r.data or {})) is pushes and bool(r.images) is pushes, profile
        finally:
            ep.close()
    if pid == "rui.free/v1":                   # a step, as at home
        ep = Episode("WK1", level="easy", seed=0, profile=pid, engine="mock")
        try:
            r = ep.call("robot_step", {"move": "up", "size": "fine"})
            assert "pushed" in (r.data or {}) and r.images
        finally:
            ep.close()


def test_a_program_can_show_the_marked_picture():
    """vision_marks' picture can be shown again: by a program (whose tools give
    back no pictures) and with camera_zoom -- under a name of its own, so the
    file a session saves is the marked picture, not the plain one."""
    import painted_mock as pixels

    ep = pixels.episode("DS4", profile="rui.code/v1+aids")
    try:
        r = ep.call("code_run", {"source": (
            "f = camera.look(['front'])['data']['frames'][0]['frame_id']\n"
            "m = vision.marks(f)['data']\n"
            "show(m['marked'])\n"
            "result = m['marked']")})
        assert not r.is_error, r.text
        assert r.data["result"].endswith(".front.marks")
        assert [im.frame_id for im in r.images] == [r.data["result"]] and r.images[0].png
        zoom = ep.call("code_run", {"source": f"show({r.data['result']!r}, box=[0, 0, 200, 150])"})
        assert not zoom.is_error and zoom.images[0].png, zoom.text
    finally:
        ep.close()


@pytest.mark.parametrize("dialect", ["delta", "step", "point", "skill"])
def test_a_program_over_each_dialect(dialect):
    """Experiment three (v1.15): rui.code-<dialect>/v1 is rui.code/v1 with that
    Dialect's moves as its programs' functions -- and the same Core, nothing
    else; the scripted reference does a task through it, every move from a program."""
    from ruikit import Episode
    from ruikit.diagnostics.reference import EXPRESSORS, style_of

    p, own = load_profile(f"rui.code-{dialect}/v1"), load_profile(f"rui.{dialect}/v1")
    assert set(p.tools) == set(load_profile("rui.code/v1").tools)
    assert set(p.code_tools) == set(own.tools) and p.channel == "code"
    assert p.tool_params("robot_step") == own.tool_params("robot_step")
    assert style_of(f"rui.code-{dialect}/v1+aids") == f"code-{dialect}"
    ep = Episode("DS1", level="easy", seed=0, profile=p, engine="mock")
    try:
        out = EXPRESSORS[style_of(p.id)]().run(ep)
        assert out.success, out.reason
        assert ep.replies and all(r["tool"] in ("code_run", "batch", "task_done")
                                  for r in ep.log), {r["tool"] for r in ep.log}
    finally:
        ep.close()
