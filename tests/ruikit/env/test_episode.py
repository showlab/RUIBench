"""The episode: lockstep clock, events, budgets, batches, stops (approach-v2 §4.2, §6.3)."""

import time

import pytest

from ruikit import Budget, Episode
from ruikit.extras.backends import POLICIES, HoldPolicy


def test_thinking_moves_the_world_under_a_fixed_clock():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1", clock="fixed",
                 think_seconds=3.0)
    try:
        ep.model_turn()
        ep.call("robot_state", {})
        assert ep.sim_time == pytest.approx(3.0, abs=0.11)
        ep.call("robot_state", {})              # same turn: no new thinking charged
        assert ep.sim_time == pytest.approx(3.0, abs=0.11)
    finally:
        ep.close()


def test_the_measured_clock_leaves_out_endpoint_waits():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1", clock="measured")
    try:
        ep.call("robot_state", {})
        start = ep.sim_time
        time.sleep(0.6)
        ep.endpoint_wait(0.6)                   # all of it a rate-limit wait: not charged
        ep.call("robot_state", {})
        assert ep.sim_time - start < 0.2
        time.sleep(0.6)                         # thinking: charged
        ep.call("robot_state", {})
        assert ep.sim_time - start > 0.45
    finally:
        ep.close()


def test_zero_clock_charges_nothing_for_thinking():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1", clock="zero")
    try:
        ep.model_turn()
        ep.call("robot_state", {})
        assert ep.sim_time == pytest.approx(0.0, abs=1e-6)
    finally:
        ep.close()


def test_unlisted_tools_are_refused():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        r = ep.call("robot_step", {"move": "down"})
        assert r.error == "not_in_profile"
        assert ep.sim_time == 0.0                  # refused calls do nothing
    finally:
        ep.close()


def test_done_ends_the_episode_and_nothing_runs_after():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        ep.call("task_done", {"report": "nothing done"})
        assert ep.over and ep.outcome.reason == "agent_done"
        assert not ep.outcome.success               # declaring is not succeeding
        assert ep.call("robot_state", {}).error == "episode_over"
    finally:
        ep.close()


def test_budget_ends_the_episode():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1",
                 budget=Budget(max_tool_calls=3))
    try:
        for _ in range(3):
            ep.call("robot_state", {})
        r = ep.call("robot_state", {})
        assert r.error == "budget_exhausted" and ep.outcome.reason == "budget_calls"
    finally:
        ep.close()


def test_results_carry_time_budget_and_events():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        r = ep.call("robot_wait", {"seconds": 1.0})
        assert r.sim_time == pytest.approx(1.0, abs=0.11)
        assert "tool_calls" in r.budget and isinstance(r.events, list)
    finally:
        ep.close()


def test_two_arms_in_one_reply_move_together():
    ep = Episode("WK4", level="easy", seed=0, profile="rui.pose/v1")
    try:
        state = ep.call("robot_state", {}).data["arms"]
        start = {s["arm"]: s["tcp"]["xyz_m"] for s in state}
        calls = [("robot_move_to", {"arm": a, "xyz_m": [p[0] + 0.05, p[1], p[2] - 0.05]})
                 for a, p in start.items()]
        t0 = ep.sim_time
        both = ep.call_batch(calls)
        together = ep.sim_time - t0
        assert all(not r.is_error for r in both), [r.text for r in both]
        # the same two moves one after the other take about twice as long
        back = [("robot_move_to", {"arm": a, "xyz_m": p}) for a, p in start.items()]
        t1 = ep.sim_time
        for c in back:
            ep.call(*c)
        apart = ep.sim_time - t1
        assert together < 0.75 * apart
    finally:
        ep.close()


def test_a_busy_arm_says_so():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        r = ep.call("robot_move_to", {"xyz_m": [0.5, 0.0, 0.2], "wait": False})
        assert r.data["status"] == "running"
        again = ep.call("robot_move_to", {"xyz_m": [0.4, 0.0, 0.2]})
        assert again.error == "resource_busy"
        ep.call("robot_wait", {"handles": [r.data["handle"]]})
        assert not ep.call("robot_move_to", {"xyz_m": [0.4, 0.0, 0.2]}).is_error
    finally:
        ep.close()


def test_nothing_stopped_ever_resumes():
    """After robot_stop neither the rest of a sequence nor a running policy's
    later chunks move the arm again (the late-result hole of v1 §8.2)."""
    POLICIES["hold"] = HoldPolicy
    try:
        ep = Episode("WK1", level="easy", seed=0, profile="open/v1")
        try:
            r = ep.call("robot_move_to", {"xyz_m": [0.55, 0.1, 0.25], "wait": False})
            ep.call("robot_stop", {"arm": "right"})
            here = ep.call("robot_state", {}).data["arms"][0]["tcp"]["xyz_m"]
            ep.call("robot_wait", {"seconds": 3.0})
            after = ep.call("robot_state", {}).data["arms"][0]["tcp"]["xyz_m"]
            assert here == after
            assert ep.handles[r.data["handle"]].status == "cancelled"
        finally:
            ep.close()
    finally:
        POLICIES.pop("hold", None)


def test_a_policy_is_a_tool_with_a_session():
    POLICIES["hold"] = HoldPolicy
    try:
        from dataclasses import replace

        from ruikit.interfaces import load_profile

        prof = load_profile("open/v1")
        prof = replace(prof, params={**prof.params, "policy_run": {"backend": "hold"}})
        ep = Episode("WK1", level="easy", seed=0, profile=prof)
        try:
            r = ep.call("policy_run", {"instruction": "hold still", "max_seconds": 2.0})
            assert not r.is_error, r.text
            assert r.data["chunks"] >= 2 and ep.sim_time >= 2.0
        finally:
            ep.close()
    finally:
        POLICIES.pop("hold", None)


def test_a_tool_with_no_backend_behind_it_is_not_offered():
    """open/v1 lists a VLA, a second vision-language model, a decision model and
    a world model; with none configured the agent is not shown them (they cost
    replies on calls that could only fail: I-014), and a call says why."""
    from ruikit.interfaces import load_profile
    from ruikit.serve.schemas import tool_entries

    listed = load_profile("open/v1")
    assert "policy_run" in listed.tools and "model_ask" in listed.code_tools
    ep = Episode("WK1", level="easy", seed=0, profile="open/v1")
    try:
        gone = {"policy_run", "policy_propose", "model_ask", "model_imagine"}
        assert gone <= set(ep.unavailable)
        assert "plan_motion" in ep.profile.code_tools          # v1.13: planned from depth
        assert not gone & (set(ep.profile.tools) | set(ep.profile.code_tools))
        assert "policy_run" not in {e["name"] for e in tool_entries(ep.profile)}
        assert "skill_pick" in ep.profile.tools                 # the rest is as listed
        r = ep.call("policy_run", {"instruction": "pick up the ring"})
        assert r.error == "not_in_profile" and "no policy backend is configured" in r.text
    finally:
        ep.close()
    plain = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        assert plain.unavailable == {} and plain.profile.tools == load_profile("rui.pose/v1").tools
    finally:
        plain.close()


def test_what_was_decided_before_a_stop_is_turned_away_by_the_executor_too():
    """Belt and braces: a session notes the executor's count of cancels when
    it starts; submitted after a stop, its work comes back stale_handle."""
    from ruibench.core.types import Pose
    from ruibench.interfaces.commands import MoveTo
    from ruikit.core.result import ToolError

    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        arm = ep.station.arm("right")
        before = ep.executor.generation(arm.arm_id)
        ep.call("robot_stop", {"arm": "right"})
        now = ep.engine.state().arm(arm.arm_id).eef_pose
        with pytest.raises(ToolError) as err:
            ep.start_motion("right", [MoveTo(arm.arm_id, Pose(now.pos, now.quat))], "late",
                            generation=before)
        assert err.value.code == "stale_handle"
    finally:
        ep.close()


@pytest.mark.parametrize("task,level", [
    ("WK1", "easy"),
    ("DI5", "hard"),        # the same mole whacked again: an event that repeats
])
def test_a_run_is_logged_so_the_benchmark_can_judge_it_again(tmp_path, task, level):
    """bench-issue #11: the log alone, in another process, gives the same verdict."""
    import json

    from ruibench.runner.snapshot import rescore
    from ruikit.diagnostics.reference.pose import PoseExpressor

    ep = Episode(task, level=level, seed=0, profile="rui.pose/v1", log_dir=str(tmp_path),
                 budget=Budget(max_tool_calls=4000))
    out = PoseExpressor().run(ep)
    ep.close()
    assert out.success
    log = json.loads(next(tmp_path.glob("*.episode.jsonl")).read_text().splitlines()[0])
    assert log["schema"] == "ruibench-log/2" and log["scoring"]["dynamics"] is False
    new, diffs = rescore(log)
    assert new.success and not diffs, diffs


def test_every_reply_can_cost_the_model_s_thinking_time():
    """think_per="reply": each reply (a call or a batch) is one answer from the
    model and costs think_seconds, the world moving on meanwhile."""
    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1", clock="fixed",
                 think_seconds=2.0, think_per="reply")
    try:
        ep.model_turn()                          # announcing a turn charges nothing more
        t0 = ep.sim_time
        ep.call("robot_state", {})
        ep.call_batch([("robot_state", {}), ("task_info", {})])
        assert ep.sim_time - t0 == pytest.approx(4.0, abs=0.3)
    finally:
        ep.close()


def test_a_reply_without_a_call_costs_its_thinking_time_too():
    """Under the fixed clock per reply, a loop's reply that calls no tool costs
    think_seconds all the same: the world goes on now, and the time limit holds.
    The measured clock counts it in the next gap; the zero clock charges nothing."""
    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1", clock="fixed",
                 think_seconds=2.0, think_per="reply", budget=Budget(max_sim_seconds=5.0))
    try:
        ep.idle_reply()
        assert ep.sim_time == pytest.approx(2.0, abs=0.11) and not ep.over
        ep.idle_reply()
        ep.idle_reply()                          # 6 s: past the 5 s the episode has
        assert ep.over and ep.outcome.reason == "timeout"
    finally:
        ep.close()
    for clock in ("measured", "zero"):
        ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1", clock=clock)
        try:
            ep.idle_reply()
            assert ep.sim_time == pytest.approx(0.0, abs=1e-6) and not ep.over
        finally:
            ep.close()


def test_steps_add_up_while_the_hand_settles():
    """A move is done within 2 mm of its target and the hand closes the rest
    later; the next step starts from the target, not from the lag (on Isaac a
    1 cm step moved 0.86 cm).  A hand pushed further off starts from where it is."""
    from ruibench.core.types import Pose
    from ruikit.env.context import ToolContext

    ep = Episode("DS1", level="easy", seed=0, profile="rui.step/v1")
    try:
        ctx = ToolContext(ep, "robot_step")
        arm = ep.station.arm("right").name
        ep.call("robot_step", {"move": "forward", "size": "fine"})
        target = list(ep.handles.values())[-1].target
        for lag, starts_at_target in ((0.0015, True), (0.005, False)):
            ctx.tcp = lambda a, lag=lag: Pose((target.pos[0] - lag, *target.pos[1:]), target.quat)
            assert (ctx.base_pose(arm).pos == target.pos) is starts_at_target
    finally:
        ep.close()


def test_a_demonstrations_frames_can_be_zoomed():
    """The wrist demo was the only way to tell Nono from the decoy (CL2), and
    zoom refused it: demo frames are named by camera and zoom takes them."""
    ep = Episode("CL2", level="easy", seed=0, profile="rui.pose/v1")
    try:
        demo = ep.call("task_demo", {"index": 0, "frames": [0], "camera": "wrist"})
        fid = demo.images[0].frame_id
        assert fid == "demo0.0.wrist"
        zoom = ep.call("camera_zoom", {"frame_id": fid, "box": [100, 100, 200, 180]})
        assert not zoom.is_error and zoom.data["frame_id"] == fid
        assert ep.call("camera_zoom", {"frame_id": "demo0.99.wrist",
                                       "box": [0, 0, 50, 50]}).error == "stale_frame"
    finally:
        ep.close()


def test_a_reply_can_look_and_read_the_picture():
    """camera_look then vision_detect on 'latest.front' in one reply (DS4 via I3
    had to guess the next frame id); an id may carry the alias too."""
    ep = Episode("DS1", level="easy", seed=0, profile="rui.point/v1")
    try:
        look, found = ep.call_batch([("camera_look", {"cameras": ["front"]}),
                                     ("vision_detect", {"frame_id": "latest.front", "query": ""})])
        fid = look.data["frames"][0]["frame_id"]
        assert not found.is_error and found.data["frame_id"] == fid
        first = found.data["objects"][0]["id"]
        pose = ep.call("vision_pose", {"frame_id": "latest.front",
                                       "id": "latest.front#" + first.split("#")[1]})
        assert not pose.is_error
        assert ep.call("vision_detect", {"frame_id": "latest.nowhere",
                                         "query": ""}).error == "stale_frame"
    finally:
        ep.close()


def test_a_tilted_thing_says_which_way_it_leans():
    """A beam's top leans toward the end that went down (DS4 via I2: the tilt
    had no sign, so which pan was down had to be read off zoomed pictures)."""
    import dataclasses
    import math

    from ruibench.core.types import Pose

    ep = Episode("DS1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        look = ep.call("camera_look", {"cameras": ["front"]})
        frame = ep.frames.get(look.data["frames"][0]["frame_id"])
        per = ep.perception
        oid, name, _ = per._ensure_ids(frame)[0]
        obj = frame.objects[name]
        a = math.radians(10)                      # about +x: the +y end rises, -y goes down
        tilted = Pose(obj.pose.pos, (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0))
        frame.objects = {**frame.objects, name: dataclasses.replace(obj, pose=tilted)}
        out = per.pose(frame, oid)
        assert abs(out["tilt_deg"] - 10) < 6          # the object's own tilt error is ~2 deg
        dx, dy = out["tilt_toward_xy"]
        assert dy < -0.5                           # toward -y, the end that went down
        upright = Pose(obj.pose.pos, (1.0, 0.0, 0.0, 0.0))
        frame.objects = {**frame.objects, name: dataclasses.replace(obj, pose=upright)}
        assert per.pose(frame, oid)["tilt_deg"] < 8        # upright reads as a few degrees at most
    finally:
        ep.close()


def test_a_move_that_stalls_short_says_so():
    """Within 5 mm of its target a stalled move counts as arrived; a free hand
    ends within 2 mm, so more than that is the only feel of contact (CL2 I1)."""
    from ruibench.core.types import Pose
    from ruikit.env.context import ToolContext

    ep = Episode("DS1", level="easy", seed=0, profile="rui.step/v1")
    try:
        ctx = ToolContext(ep, "robot_step")
        ep.call("robot_step", {"move": "down", "size": "fine"})
        h = list(ep.handles.values())[-1]
        tcp = h.target
        for lag, flagged in ((0.0015, False), (0.004, True)):
            ctx.tcp = lambda a, lag=lag: Pose((tcp.pos[0], tcp.pos[1], tcp.pos[2] + lag), tcp.quat)
            ep.engine.state().arm(ep.scene.arm_ids[0])        # (the report reads the engine)
            real = ep.engine.state
            class _S:                                          # the engine's view, lagging
                def __init__(self, st):
                    self._st = st
                def arm(self, aid):
                    import dataclasses
                    return dataclasses.replace(self._st.arm(aid), eef_pose=ctx.tcp(None))
                def __getattr__(self, k):
                    return getattr(self._st, k)
            ep.engine.state = lambda real=real: _S(real())
            try:
                r = ctx._report(h)
            finally:
                ep.engine.state = real
            assert ("stopped" in r.text) is flagged and ("short_by_m" in r.data) is flagged
    finally:
        ep.close()


def test_show_latest_pins_the_picture_of_that_moment():
    """show('latest.front') in a program shows the picture newest when show() ran,
    not the one newest when the program ended (WK5 via I4 lost a picture so)."""
    ep = Episode("DS1", level="easy", seed=0, profile="rui.code/v1")
    try:
        src = ("a = camera.look(['front'])['data']['frames'][0]['frame_id']\n"
               "show('latest.front')\n"
               "b = camera.look(['front'])['data']['frames'][0]['frame_id']\n"
               "result = [a, b, math.pi > 3, json.dumps(1)]")
        r = ep.call("code_run", {"source": src})
        assert not r.is_error, r.text
        a, b = r.data["result"][:2]
        assert a != b and [im.frame_id for im in r.images] == [a]
    finally:
        ep.close()


def test_one_reply_can_move_wait_and_look():
    """move (wait=false), robot_wait on ['running'], camera_look: the picture is
    taken after the move, without guessing the move's handle id."""
    ep = Episode("DS1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        before = ep.engine.state().arm(ep.scene.arm_ids[0]).eef_pose.pos
        mv, wt, look = ep.call_batch([
            ("robot_move_to", {"xyz_m": [0.40, 0.05, 0.25], "wait": False}),
            ("robot_wait", {"handles": ["running"]}),
            ("camera_look", {"cameras": ["front"]})])
        assert not any(r.is_error for r in (mv, wt, look)), (mv.text, wt.text, look.text)
        assert wt.data["handles"] and wt.data["handles"][0]["status"] == "done"
        assert wt.data["waited_s"] > 0 and look.data["frames"][0]["t"] >= wt.data["waited_s"]
        assert ep.engine.state().arm(ep.scene.arm_ids[0]).eef_pose.pos != before
    finally:
        ep.close()


def test_a_step_can_repeat_and_the_step_interface_shows_the_result():
    """I1 (9/30): a step may be taken up to 5 times in a row in one reply, and a
    reply that moved the arm brings the cameras' pictures of the result, as
    Show-Harness pushes agentview + wrist after every step."""
    ep = Episode("DS1", level="easy", seed=0, profile="rui.step/v1")
    try:
        before = ep.engine.state().arm(ep.scene.arm_ids[0]).eef_pose.pos
        r = ep.call("robot_step", {"move": "forward", "size": "coarse", "n": 3})
        after = ep.engine.state().arm(ep.scene.arm_ids[0]).eef_pose.pos
        assert not r.is_error and "x3" in r.text
        assert abs((after[0] - before[0]) - 0.12) < 0.002 or abs((after[1] - before[1]) - 0.12) < 0.002 \
            or abs(sum((a - b) ** 2 for a, b in zip(after, before)) ** 0.5 - 0.12) < 0.002
        assert r.data["pushed"]["frames"] and len(r.images) >= 1           # the result, seen
        assert ep.call("robot_step", {"move": "grasp", "n": 2}).error == "invalid_args"
        looked = ep.call("vision_detect", {"frame_id": "latest.front", "query": ""})
        assert not looked.is_error                                          # no push for looking
        assert "pushed" not in (looked.data or {})
        # two different steps of one arm are still two actions: refused
        two = ep.call_batch([("robot_step", {"move": "up"}), ("robot_step", {"move": "left"})])
        assert all(x.is_error for x in two)
    finally:
        ep.close()
    pose = Episode("DS1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        mv = pose.call("robot_move_to", {"xyz_m": [0.4, 0.0, 0.25]})
        assert not mv.is_error and mv.data["pushed"]["frames"]        # every interface (v1.17)
    finally:
        pose.close()


def _open_with(*tools):
    """The open track with some of its code-only tools offered directly."""
    from dataclasses import replace

    from ruikit.interfaces import load_profile

    prof = load_profile("open/v1")
    return replace(prof, tools=type(prof.tools)(list(prof.tools) + list(tools)))


def test_a_pick_keeps_the_fingertips_off_what_a_flat_thing_lies_on(monkeypatch):
    """Half a 1 cm coin's height put an FR3's fingertips (8 mm below the tool
    centre point) into the table: every first grip of DS4's coins on Isaac
    stopped short (I-010).  And a target can be a bare object id (I-006)."""
    from ruikit.env.context import ToolContext

    steps = []
    real = ToolContext.sequence

    def spy(self, arm, groups, label, final):
        steps.extend(pose for group in groups for kind, pose in group if kind == "move")
        return real(self, arm, groups, label, final)

    monkeypatch.setattr(ToolContext, "sequence", spy)
    ep = Episode("DS4", level="easy", seed=0, profile=_open_with("vision_pose"))
    try:
        fid = ep.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
        coin = ep.call("vision_detect", {"frame_id": fid, "query": "coin"}).data["objects"][0]["id"]
        pose = ep.call("vision_pose", {"frame_id": fid, "id": coin}).data
        r = ep.call("skill_pick", {"target": coin})              # the id alone
        assert r.error != "invalid_args", r.text
        station = ToolContext(ep, "skill_pick").station
        lowest = min(station.describe_pose(p)["xyz_m"][2] for p in steps)
        bottom = pose["xyz_m"][2] - pose["extent_m"][2] / 2
        assert lowest >= bottom + 0.008 + 0.002 - 2e-4, (lowest, bottom)   # (reports round to 0.1 mm)
        bad = ep.call("skill_pick", {"target": "the coin"})
        assert bad.error == "invalid_args" and "object id" in bad.text
    finally:
        ep.close()


def test_model_ask_takes_a_demonstrations_frames():
    """CL2: "which doll on the table is the one the film lifted" needs the film's
    frames next to the table's, and model_ask refused them as stale (I-004)."""
    from ruikit.extras.backends import ASKERS

    class Echo:
        def ask(self, images, question):
            return f"{len(images)} pictures"

    ASKERS["echo"] = Echo
    try:
        from dataclasses import replace

        prof = _open_with("model_ask")
        prof = replace(prof, params={**prof.params, "model_ask": {"backend": "echo"}})
        ep = Episode("CL2", level="easy", seed=0, profile=prof)
        try:
            ep.call("task_demo", {"index": 0, "frames": [0], "camera": "wrist"})
            fid = ep.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
            r = ep.call("model_ask", {"frame_ids": ["demo0.0.wrist", fid], "question": "same doll?"})
            assert r.error != "stale_frame", r.text      # the mock has no pixels to send
            bogus = ep.call("model_ask", {"frame_ids": ["demo0.99.wrist"], "question": "?"})
            assert bogus.error == "stale_frame"
        finally:
            ep.close()
    finally:
        ASKERS.pop("echo", None)


def test_a_move_blocked_by_its_turn_says_so():
    """There, but not turned: an FR3 wrist at the end of its travel near yaw
    +-180 read as "0.1 cm short", as if something were in the way (I-007)."""
    import dataclasses
    import math

    from ruibench.core.geometry import quat_mul
    from ruibench.core.types import Pose
    from ruikit.env.context import ToolContext

    ep = Episode("DS1", level="easy", seed=0, profile="rui.step/v1")
    try:
        ctx = ToolContext(ep, "robot_step")
        ep.call("robot_step", {"move": "down", "size": "fine"})
        h = list(ep.handles.values())[-1]
        turned = Pose(h.target.pos, quat_mul((math.cos(0.07), 0.0, 0.0, math.sin(0.07)),
                                             h.target.quat))         # 8 degrees about z
        real = ep.engine.state

        class _S:
            def __init__(self, st):
                self._st = st

            def arm(self, aid):
                return dataclasses.replace(self._st.arm(aid), eef_pose=turned)

            def __getattr__(self, k):
                return getattr(self._st, k)

        from ruibench.executor.handles import HandleStatus

        h.parts[-1].status = HandleStatus.FAILED
        h.parts[-1].detail = "blocked: stopped 0.0 cm and 8 degrees short of the target"
        ep.engine.state = lambda: _S(real())
        try:
            r = ctx._report(h)
        finally:
            ep.engine.state = real
        assert r.error == "blocked" and "could not turn" in r.hint, (r.error, r.hint)
    finally:
        ep.close()


def test_a_pick_closes_across_a_long_things_narrow_side(monkeypatch):
    """A plank 16 cm long and 4 cm wide, its length along its own x: the
    fingers (8 cm) close across the 4 cm.  Until v1.3 skill_pick turned the
    hand a quarter turn whenever the longer extent was too wide, onto the
    length (found when dolls lying down could not be picked by what the
    cameras showed)."""
    import math

    from ruibench.core.geometry import yaw_of
    from ruikit.env.context import ToolContext
    from ruikit.env.station import rpy_deg_from_quat

    grips = []
    real = ToolContext.sequence

    def spy(self, arm, groups, label, final):
        grips.extend(pose for group in groups for kind, pose in group if kind == "move")
        return real(self, arm, groups, label, final)

    monkeypatch.setattr(ToolContext, "sequence", spy)
    ep = Episode("CL3", level="easy", seed=0, profile=_open_with("vision_pose"))
    try:
        plank = next(b for b in ep.scene.bodies if b.name == "plank_ash")
        assert plank.size[0] > 0.04 > plank.size[1]              # long along x, narrow along y
        fid = ep.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
        ids = {name: oid for oid, name, _ in ep.perception._ensure_ids(ep.frames.get(fid))}
        r = ep.call("skill_pick", {"target": ids["plank_ash"]})
        assert r.error != "invalid_args", r.text
        hand = rpy_deg_from_quat(grips[-1].quat)[2]
        body = math.degrees(yaw_of(ep.engine.state().obj("plank_ash").pose.quat))
        off = (hand - body + 90.0) % 180.0 - 90.0               # the hand's yaw from the plank's
        assert abs(off) < 10.0, (hand, body)
    finally:
        ep.close()


def test_a_pick_at_a_pixel_or_a_point_never_puts_the_fingertips_in_the_table(monkeypatch):
    """A pixel or a point says nothing of how thick the thing is: 2 cm below
    it was 6 mm under the table for a doll 1.4 cm thick, and the pick stopped
    short without closing (I-017).  And yaw_deg turns the hand as asked."""
    from ruikit.env.context import ToolContext

    steps = []
    real = ToolContext.sequence

    def spy(self, arm, groups, label, final):
        steps.extend(pose for group in groups for kind, pose in group if kind == "move")
        return real(self, arm, groups, label, final)

    monkeypatch.setattr(ToolContext, "sequence", spy)
    ep = Episode("CL2", level="easy", seed=0, profile="open/v1")
    try:
        ep.call("skill_pick", {"target": {"xyz_m": [0.45, 0.15, 0.014]}, "yaw_deg": 35.0})
        station = ToolContext(ep, "skill_pick").station
        told = [station.describe_pose(p) for p in steps]
        assert min(t["xyz_m"][2] for t in told) >= 0.008 + 0.002 - 2e-4, told
        assert all(abs(t["rpy_deg"][2] - 35.0) < 0.2 for t in told), told
        steps.clear()
        ep.call("skill_pick", {"target": {"xyz_m": [0.45, 0.15, 0.10]}})    # a thing up high
        low = min(station.describe_pose(p)["xyz_m"][2] for p in steps)
        assert abs(low - 0.08) < 2e-4, low                                  # as before: 2 cm below
    finally:
        ep.close()


def test_the_x5_hovers_and_lifts_no_higher_than_it_reaches_there(monkeypatch):
    """The ARX X5 holds its hand straight down lower the farther out it reaches
    (measured on Isaac): the pick hovering 10 cm over a fork 47 cm out on WK4
    stopped 7 cm short (T9).  The skills now hover and lift as high as the arm
    can there, never closer than 2 cm over the target's top; plan_ik says so
    too; the FR3 is as before (v1.15)."""
    import math

    from ruibench.engines import robots
    from ruikit.env.context import ToolContext
    from ruikit.env.station import CEILING_MARGIN_M, CEILINGS

    steps = []
    real = ToolContext.sequence

    def spy(self, arm, groups, label, final):
        steps.extend(pose for group in groups for kind, pose in group if kind == "move")
        return real(self, arm, groups, label, final)

    monkeypatch.setattr(ToolContext, "sequence", spy)
    ep = Episode("WK4", level="easy", seed=0, profile="rui.skill/v1+aids")
    try:
        st = ep.station
        base = st.to_station(robots.arm_base(st.scene.robots[0], "right", st.table_height))
        out = lambda r, z: [base[0] + r, base[1], z]               # straight ahead of it
        for r, z in CEILINGS["arx5"]:
            assert st.ceiling("right", out(r, 0)[:2]) == pytest.approx(z - CEILING_MARGIN_M)
        far = st.ceiling("right", out(0.47, 0)[:2])
        heights = lambda: [st.describe_pose(q)["xyz_m"][2] for q in steps]
        for name, args in (("skill_pick", {}), ("skill_place", {"height_m": 0.0}),
                           ("skill_press", {}), ("skill_push", {"direction_deg": 90.0,
                                                                "distance_m": 0.05})):
            steps.clear()
            ep.call(name, {"target": {"xyz_m": out(0.47, 0.03)}, **args})
            assert max(heights()) == pytest.approx(far, abs=2e-4), (name, heights())
        steps.clear()                                               # near the base: as before
        ep.call("skill_pick", {"target": {"xyz_m": out(0.26, 0.03)}})
        assert max(heights()) == pytest.approx(0.01 + 0.12, abs=2e-4), heights()
        steps.clear()                                               # never under top + 2 cm
        ep.call("skill_pick", {"target": {"xyz_m": out(0.47, 0.07)}})
        assert max(heights()) == pytest.approx(0.07 + 0.02, abs=2e-4), heights()
        r = ep.call("plan_ik", {"xyz_m": out(0.47, 0.10)})
        assert not r.data["reachable"] and r.data["ceiling_m"] == pytest.approx(far, abs=1e-3)
        assert "this far from the base" in r.text, r.text
        assert ep.call("plan_ik", {"xyz_m": out(0.47, far - 0.01)}).data["reachable"]
        for name, h in (("skill_move", 0.10), ("skill_place", 0.05)):  # a height asked for:
            steps.clear()                                           # refused, not lowered
            r = ep.call(name, {"target": {"xyz_m": out(0.47, 0.03)}, "height_m": h})
            assert r.error == "unreachable" and not steps, (name, r.text)
            assert r.text == f"not reachable: higher than {far:.3f} m above the table this " \
                             "far from the base", r.text                 # plan_ik's words
            assert r.hint == f"height_m at most {math.floor((far - 0.03) * 1000) / 1000:.3f} here"
        r = ep.call("skill_move", {"target": {"xyz_m": out(0.47, 0.03)}, "height_m": 0.02})
        assert r.error is None and r.data["status"] == "done", r.text
    finally:
        ep.close()
    steps.clear()
    ep = Episode("CL2", level="easy", seed=0, profile="rui.skill/v1+aids")
    try:                                                            # the FR3: no ceiling
        assert math.isinf(ep.station.ceiling("right", (0.7, 0.0)))
        ep.call("skill_pick", {"target": {"xyz_m": [0.7, 0.0, 0.03]}})
        assert max(ep.station.describe_pose(q)["xyz_m"][2] for q in steps) == \
            pytest.approx(0.01 + 0.12, abs=2e-4)
        assert ep.call("plan_ik", {"xyz_m": [0.7, 0.0, 0.3]}).data["ceiling_m"] is None
        r = ep.call("skill_move", {"target": {"xyz_m": [0.7, 0.0, 0.03]}, "height_m": 0.3})
        assert r.error is None, r.text
    finally:
        ep.close()


def test_a_skill_blocked_on_its_way_down_is_measured_against_that_step():
    """A skill is several moves.  One blocked on its descent reported
    short_by_m against the skill's last pose, 10 cm up: "stopped 1.5 cm short"
    with a remainder pointing upward (I-017)."""
    ep = Episode("WK1", level="easy", seed=0, profile="open/v1")
    try:
        ep.engine.hand_collides = True               # the mock's table stops the fingertips
        r = ep.call("skill_place", {"target": {"xyz_m": [0.40, 0.28, -0.04]}, "height_m": 0.0})
        assert r.error == "blocked", r.text
        short = r.data["short_by_m"]
        assert short[2] < 0 and abs(short[0]) < 0.005 and abs(short[1]) < 0.005, short
        assert r.data["residual_cm"] == pytest.approx(-100 * short[2], abs=0.3), r.data
        assert f"{r.data['residual_cm']:.1f} cm" in r.text, (r.text, r.data)
    finally:
        ep.close()


def test_a_wait_for_an_event_ends_on_one_not_given_yet():
    """An event that came while the agent was thinking, or while its program
    was moving the arm, is the next event: the wait returns it at once.
    Counted from the call, it was waited past (I-017)."""
    ep = Episode("WK1", level="easy", seed=0, profile="open/v1")
    try:
        ep._events.append({"t": 0.0, "type": "message", "text": "the opponent has moved"})
        r = ep.call("robot_wait", {"until_event": True, "max_seconds": 5})
        assert r.data["new_events"] == 1 and r.data["waited_s"] < 0.2, r.data
        assert r.data["events"][0]["text"] == "the opponent has moved"
        again = ep.call("robot_wait", {"until_event": True, "max_seconds": 1})
        assert again.data["new_events"] == 0 and again.data["waited_s"] >= 0.9, again.data
        # inside a program: once for each event, never twice for one
        ep._events.append({"t": 1.0, "type": "message", "text": "your turn"})
        out = ep.call("code_run", {"source": (
            "a = robot.wait(until_event=True, max_seconds=3)['data']\n"
            "b = robot.wait(until_event=True, max_seconds=1)['data']\n"
            "result = [a['new_events'], a['events'][0]['text'], a['waited_s'] < 0.2,\n"
            "          b['new_events'], b['waited_s'] >= 0.9]")})
        assert out.data["result"] == [1, "your turn", True, 0, True], out.data
    finally:
        ep.close()


def test_what_a_function_returns_is_written_with_it():
    """A program reads a result's data by its keys, and had to print one before
    it could use it: each function's keys are listed with it (I-017).  The
    list is checked against what the tools return."""
    import dataclasses
    import re

    from ruikit.core.tool import MOTION_DATA, get_tool
    from ruikit.interfaces import load_profile
    from ruikit.serve.schemas import tool_entries

    code = load_profile("rui.code/v1")
    assert all(get_tool(name).spec.data for name in code.code_tools)
    told = {e["name"]: e["description"] for e in tool_entries(code)}["code_run"]
    assert told.count("-> data:") == len(code.code_tools) and "A motion's result" in told
    assert "4 pictures a run at most" in told

    listed = load_profile("open/v1")
    every = dataclasses.replace(listed, tools=tuple(dict.fromkeys(listed.tools + listed.code_tools)))
    ep = Episode("WK1", level="easy", seed=0, profile=every)
    try:
        fid = ep.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
        thing = ep.call("vision_detect", {"frame_id": fid}).data["objects"][0]
        u, v = thing["center"]
        calls = [("camera_look", {"cameras": ["front"]}), ("robot_state", {}),
                 ("camera_zoom", {"frame_id": fid, "box": [100, 100, 300, 300]}),
                 ("vision_detect", {"frame_id": fid}), ("vision_point", {"frame_id": fid, "query": "ring"}),
                 ("vision_segment", {"frame_id": fid, "prompt": {"point": [u, v]}}),
                 ("vision_lift", {"frame_id": fid, "u": u, "v": v}),
                 ("vision_pose", {"frame_id": fid, "id": thing["id"]}),
                 ("vision_marks", {"frame_id": fid}),
                 ("robot_move_to", {"xyz_m": [0.45, 0.0, 0.25]}),
                 ("robot_move_delta", {"dxyz_m": [0, 0, 0.02]}), ("robot_step", {"move": "up"}),
                 ("robot_move_to_pixel", {"frame_id": fid, "u": u, "v": v, "height_m": 0.1}),
                 ("robot_gripper", {"command": "open"}), ("robot_home", {}),
                 ("robot_wait", {"seconds": 0.2}), ("robot_wait", {"until_event": True, "max_seconds": 0.2}),
                 ("robot_stop", {}), ("skill_pick", {"target": thing["id"]}),
                 ("skill_place", {"target": {"xyz_m": [0.45, 0.1, 0.02]}}),
                 ("plan_ik", {"xyz_m": [0.45, 0.0, 0.2]}),
                 ("plan_grasps", {"target": {"frame_id": fid, "id": thing["id"]}}),
                 ("memory_note", {"key": "a", "text": "b"}), ("memory_recall", {}),
                 ("memory_frames", {"last_n": 1}), ("task_info", {}), ("task_events", {})]
        for name, args in calls:
            r = ep.call(name, args)
            assert not r.is_error, (name, r.text)
            words = set(re.findall(r"[a-z_]+", get_tool(name).spec.data.replace("MOTION", MOTION_DATA)))
            # the pictures a reply that moved an arm brings back (v1.17) are the
            # reply's, on its last result, not one tool's
            assert set(r.data) - {"pushed"} <= words, (name, sorted(set(r.data) - words))
    finally:
        ep.close()



def test_the_agent_is_told_how_far_a_wrist_turns():
    """Measured on Isaac (research/wrist_range.py): the FR3's hand turns to 180
    degrees one way and 100-150 the other, an X5's about 170 in all.  Agents
    could not tell whether half a circle was one move or two (I-017)."""
    one = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1")
    two = Episode("WK4", level="easy", seed=0, profile="rui.pose/v1")
    try:
        told = one.call("task_info", {}).data["conventions"]["orientation"]
        assert "about 345 degrees" in told and "the shorter way" in told
        told = two.call("task_info", {}).data["conventions"]["orientation"]
        assert "about 170 degrees" in told and "left hand" in told
    finally:
        one.close()
        two.close()


def test_a_rule_broken_ends_the_episode_at_once():
    """v2.3.4: every rule of a task is a "never" -- once broken, the episode is
    lost whatever follows -- so it ends then, and nothing runs after; with
    ``end_at_violation`` off it plays on, and is lost all the same."""
    from dataclasses import replace

    from ruibench.scoring import Custom, Never

    always = (("a_rule", Never(Custom(lambda w: w.sim_time > 0.5, "half a second gone"))),)
    for ends in (True, False):
        ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1", end_at_violation=ends)
        try:
            ep.inst = replace(ep.inst, scorer=replace(ep.inst.scorer, violations=always))
            assert ep.broken_rule() is None
            ep.call("robot_wait", {"seconds": 1.0})
            assert ep.broken_rule() == "a_rule"
            assert ep.over is ends
            if ends:
                assert ep.outcome.reason == "rule_broken"
                assert ep.call("robot_state", {}).error == "episode_over"
        finally:
            ep.close()
