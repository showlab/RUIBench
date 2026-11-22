"""``perception: model``: the vision_* tools answered from the picture and the
depth image, and from nothing else (the pure-vision check, 10/3)."""

import dataclasses
import math

import pytest

np = pytest.importorskip("numpy")

from ruikit.core.result import ToolError  # noqa: E402
from ruikit.interfaces import load_profile  # noqa: E402

import painted_mock as pixels  # noqa: E402  (tests/ruikit/painted_mock.py)


def _look(ep, camera="front"):
    return ep.call("camera_look", {"cameras": [camera]}).data["frames"][0]["frame_id"]


def _truth(ep, name):
    return ep.station.to_station(ep.engine.state().obj(name).pose.pos)


def test_pixels_wherever_pictures_are_rendered_and_a_test_double_on_the_mock():
    """v1.13: no tool is handed where a thing is.  Where pictures are rendered
    the perception answers from them, and asking for the synthetic condition
    there is refused; the mock, which renders none, has a test double instead,
    and its log says so."""
    from ruikit import Episode
    from ruikit.interfaces import load_profile

    prof = load_profile("rui.pose/v1")
    assert prof.perception == "model"
    ep = Episode("DS4", profile=prof, engine="mock")
    try:
        assert ep.perception_info["perception"] == "synthetic"
        assert "test double" in ep.perception_info["stand_in"]
    finally:
        ep.close()
    with pytest.raises(ToolError) as err:
        Episode("DS4", profile=dataclasses.replace(prof, perception="synthetic"),
                engine=pixels.PixelMock())
    assert err.value.code == "not_supported" and "simulator's positions" in str(err.value)


def test_things_are_found_and_placed_from_the_picture_and_its_depth():
    """The coins of DS4 through the fixed camera: each found once, under its
    word, within the centimetre or two a far camera and its calibration leave
    -- and the robot's own hand and the table top are not things."""
    ep = pixels.episode("DS4")
    try:
        fid = ep.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
        found = ep.call("vision_detect", {"frame_id": fid, "query": "coin"}).data["objects"]
        assert len(found) == 3 and all(o["label"] == "coin" for o in found)   # the word asked
        assert [o["id"] for o in found] == [f"{fid}#{k}" for k in (1, 2, 3)]     # left to right
        coins = sorted((n for n in ep.engine.state().objects if n.startswith("coin")),
                       key=lambda n: _truth(ep, n)[1])        # the camera faces the robot
        for obj, name in zip(found, coins):
            pose = ep.call("vision_pose", {"frame_id": fid, "id": obj["id"]}).data
            true = _truth(ep, name)
            # the far camera's calibration alone is a centimetre off here (0.3 degrees
            # at 1.4 m), as under the synthetic condition with this seed
            assert math.dist(pose["xyz_m"][:2], true[:2]) < 0.02, (pose["xyz_m"], true)
            assert abs(pose["xyz_m"][2] - true[2]) < 0.015, (pose["xyz_m"], true)
            assert 0.02 < pose["extent_m"][0] < 0.06 and 0.003 < pose["extent_m"][2] < 0.02
        everything = ep.call("vision_detect", {"frame_id": fid}).data["objects"]
        labels = [o["label"] for o in everything]
        assert "robot" not in labels and "table" not in labels
        assert sum(word in ("coin", "gold coin") for word in labels) == 3     # the same three
        # asked for by name, the robot and the table are still turned away
        for word in ("robot", "table"):
            assert ep.call("vision_detect", {"frame_id": fid, "query": word}).data["objects"] == []
    finally:
        ep.close()


def test_a_pixel_is_lifted_by_the_depth_image():
    ep = pixels.episode("DS4")
    try:
        fid = ep.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
        coin = ep.call("vision_detect", {"frame_id": fid, "query": "coin"}).data["objects"][0]
        u, v = coin["center"]
        on = ep.call("vision_lift", {"frame_id": fid, "u": u, "v": v}).data
        assert on["on"] in ("gold coin", "coin") and 0.0 < on["xyz_m"][2] < 0.03
        assert on["normal"][2] > 0.8                          # its top face, facing up
        image = ep.frames.get(fid).image
        bare = (image == np.array(pixels.TABLE, dtype=np.uint8)).all(axis=-1)
        rows, cols = np.nonzero(bare & np.isfinite(ep.perception.view(ep.frames.get(fid)).depth))
        k = int(np.argmin((rows - (v + 25)) ** 2 + (cols - u) ** 2))     # bare table, nearby
        tu, tv = float(cols[k]) + 0.5, float(rows[k]) + 0.5
        off = ep.call("vision_lift", {"frame_id": fid, "u": tu, "v": tv}).data
        assert off["on"] == "table" and abs(off["xyz_m"][2]) < 0.015
        seg = ep.call("vision_segment", {"frame_id": fid, "prompt": {"point": [u, v]}}).data
        assert seg["segments"][0]["id"] == coin["id"]
        assert ep.call("vision_segment", {"frame_id": fid,
                                          "prompt": {"point": [tu, tv]}}).data["segments"] == []
    finally:
        ep.close()


def test_nothing_is_read_from_the_simulator():
    """The answers are the same when what the simulator knows of the frame --
    which bodies are in view, and where they are -- is taken away: the backend
    has only the picture, the depth and the robot's own readings."""
    ep = pixels.episode("DS4")
    try:
        fid = ep.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
        frame = ep.frames.get(fid)

        def answers():
            ep.perception._views.clear()
            ep.perception._found.clear()
            found = ep.perception.detect(frame, "coin")
            return (found, [ep.perception.pose(frame, o["id"]) for o in found],
                    ep.perception.lift(frame, *found[0]["center"]))

        before = answers()
        frame.objects, frame.visible = {}, ()
        assert answers() == before
    finally:
        ep.close()


def test_the_same_look_gives_the_same_answer_and_the_calibration_is_the_synthetic_one():
    from ruikit import Episode

    ep = pixels.episode("DS4")
    syn = Episode("DS4", profile="rui.pose/v1")
    try:
        a = ep.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
        b = ep.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
        ca = ep.call("vision_detect", {"frame_id": a, "query": "coin"}).data["objects"][0]
        cb = ep.call("vision_detect", {"frame_id": b, "query": "coin"}).data["objects"][0]
        pa = ep.call("vision_pose", {"frame_id": a, "id": ca["id"]}).data
        pb = ep.call("vision_pose", {"frame_id": b, "id": cb["id"]}).data
        assert pa["xyz_m"] == pb["xyz_m"]                     # looking twice averages nothing
        sf = syn.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
        one = ep.perception.believed(ep.frames.get(a))
        other = syn.perception.believed(syn.frames.get(sf))
        assert one.pos == pytest.approx(other.pos) and one.quat == pytest.approx(other.quat)
    finally:
        ep.close()
        syn.close()


def test_the_agent_is_told_how_its_estimates_are_made():
    from ruikit.interfaces import load_profile
    from ruikit.serve.schemas import tool_entries

    prof = load_profile("rui.pose/v1")
    said = {e["name"]: e["description"]
            for e in tool_entries(dataclasses.replace(prof, perception="synthetic"))}
    seen = {e["name"]: e["description"] for e in tool_entries(prof)}
    assert "fixed error" in said["vision_pose"] and "fixed error" not in seen["vision_pose"]
    assert "open-vocabulary detector" in seen["vision_detect"]
    assert seen["robot_move_to"] == said["robot_move_to"]
    # a score means what the backend behind the answers makes it mean (v1.4)
    assert "(0.2 to 1: below 0.5 is doubtful" in seen["vision_detect"]
    pair = dataclasses.replace(prof, perception="model", params={
        **prof.params, "perception": {"backend": "grounded-sam2"}})
    told = {e["name"]: e["description"] for e in tool_entries(pair)}
    assert "(0.25 to 1: below 0.4 is doubtful" in told["vision_detect"]
    assert not any("{" in text for text in list(seen.values()) + list(told.values()))
    ep = pixels.episode("DS4")
    try:
        info = ep.call("task_info", {}).data["conventions"]["estimates"]
        assert "depth camera" in info and "fixed error" not in info
        assert ep.perception_info["perception"] == "model" and ep.perception_info["detector"]
    finally:
        ep.close()


def test_an_interface_is_asked_for_under_a_perception_condition_by_its_id():
    """Pixels are the default (v1.13); what answers them is chosen after an
    ``@`` and the id says so, so that no log or table mixes two conditions.
    Never better than a real robot: noise and calibration only grow."""
    from ruikit.interfaces import load_profile
    from ruikit.interfaces.compose import with_perception

    base = load_profile("rui.pose/v1")
    assert (base.id, base.perception) == ("rui.pose/v1", "model")
    seen = load_profile("rui.pose/v1@model")              # the v1.3-v1.12 spelling: the default
    assert (seen.id, seen.perception) == ("rui.pose/v1", "model")
    assert seen.tools == base.tools and seen.params == base.params
    pair = load_profile("rui.pose/v1@grounded-sam2,depth=mono,noise=2")
    assert pair.id == "rui.pose/v1@grounded-sam2,depth=mono,noise=2"
    assert pair.perception_settings() == {"backend": "grounded-sam2", "depth": "mono",
                                          "noise": 2.0, "calib": 1.0}
    assert pair.tools == base.tools
    assert with_perception(base, "synthetic").id == "rui.pose/v1@synthetic"
    for wrong in ("rui.pose/v1@nope", "rui.pose/v1@noise=0.5", "rui.pose/v1@depth=lidar"):
        with pytest.raises(ValueError):
            load_profile(wrong)


def test_the_cameras_deliver_depth_only_to_perception_from_pixels():
    """A depth image is rendered when the run's perception needs one, and not
    otherwise: under the synthetic condition nothing about a run changes."""
    from ruibench.engines.base import CameraSpec
    from ruikit import Episode

    assert CameraSpec("agentview").depth is False
    syn = Episode("DS4", profile="rui.pose/v1")
    ep = pixels.episode("DS4")
    try:
        assert not any(c.depth for c in syn.scene.cameras)
        fid = syn.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
        assert syn.frames.get(fid).depth is None
        assert all(c.depth for c in ep.scene.cameras)
        fid = ep.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
        frame = ep.frames.get(fid)
        assert frame.depth.shape == (frame.height, frame.width)
        assert set(frame.arms) == set(ep.scene.arm_ids)     # and the robot's own readings
    finally:
        syn.close()
        ep.close()


def test_a_thing_is_picked_up_by_what_the_cameras_show():
    """Look, find by a word, go above, look closer, pick: the whole loop on
    tools alone (the painted mock; ``research/vision_pick.py`` does it on Isaac)."""
    ep = pixels.episode("DS4", profile="open/v1")
    try:
        ep.profile = dataclasses.replace(ep.profile, tools=tuple(ep.profile.tools) + ("vision_pose",))
        front = ep.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
        coin = ep.call("vision_detect", {"frame_id": front, "query": "coin"}).data["objects"][0]
        far = ep.call("vision_pose", {"frame_id": front, "id": coin["id"]}).data
        ep.call("robot_move_to", {"xyz_m": [far["xyz_m"][0], far["xyz_m"][1], 0.24],
                                  "rpy_deg": [0, 0, 0]})
        wrist = ep.call("camera_look", {"cameras": ["wrist"]}).data["frames"][0]["frame_id"]
        near = ep.call("vision_detect", {"frame_id": wrist, "query": "coin"}).data["objects"]
        target = min(near, key=lambda o: math.dist(o["center"], (256, 256)))
        r = ep.call("skill_pick", {"target": target["id"]})
        assert not r.is_error and r.data["holding_something"], r.text
    finally:
        ep.close()


def test_a_process_takes_one_of_several_vision_servers(monkeypatch):
    """Copies of the server, comma separated: a process keeps one of them, and
    a server that runs another backend than the profile names is refused."""
    import os

    from ruikit.perception.service.client import VisionServer

    monkeypatch.setenv("RUIKIT_VISION_URL", "http://a:1/, http://b:2")
    monkeypatch.setattr(os, "getpid", lambda: 7)
    assert VisionServer().url == "http://b:2"
    monkeypatch.setattr(os, "getpid", lambda: 8)
    one = VisionServer("sam3")
    assert one.url == "http://a:1"
    monkeypatch.setattr(one, "_call", lambda path, body=None: {"backend": "grounded-sam2"})
    with pytest.raises(ToolError) as err:
        one.info()
    assert err.value.code == "backend_unavailable" and "grounded-sam2" in str(err.value)


def test_holding_is_told_by_the_fingers_own_readings():
    """Whether the hand holds something: commanded closed and stopped apart --
    a real gripper's encoders.  The engine's own bookkeeping of *what* is held
    is never asked, except by the mock, which cannot measure fingers."""
    from ruibench.core.types import ArmState, Pose
    from ruikit.perception.proprioception import HOLD_MIN_GAP_M, holding_something

    pose = Pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    shut = ArmState("a", pose, gripper=1.0, holding=None, opening=0.0)
    on_it = ArmState("a", pose, gripper=1.0, holding=None, opening=0.012)
    known = ArmState("a", pose, gripper=1.0, holding="secret_name", opening=0.0005)
    assert not holding_something(shut) and holding_something(on_it)
    assert not holding_something(known)                     # what the engine knows is not asked
    assert not holding_something(dataclasses.replace(on_it, gripper=0.0))
    mock = ArmState("a", pose, gripper=1.0, holding="x", opening=None)
    assert holding_something(mock)                          # the mock's test double
    assert HOLD_MIN_GAP_M == 0.002


class _MonoVision(pixels.ColourVision):
    """The painted world's detector, and a monocular depth that is the true
    depth 25 % too far: what the scale fit to the table must undo."""

    def info(self):
        return {**super().info(), "depth_model": "test-mono"}

    def depth(self, image):
        return np.nan_to_num(self.engine.last_depth, nan=0.0) * 1.25


def test_depth_from_the_picture_alone_is_scaled_by_the_table():
    ep = pixels.episode("DS4", profile="rui.pose/v1@depth=mono", vision=_MonoVision)
    try:
        assert ep.perception_info["depth"] == "mono"
        assert not any(c.depth for c in ep.scene.cameras)      # no depth camera at all
        fid = _look(ep)
        assert ep.frames.get(fid).depth is None
        coin = ep.call("vision_detect", {"frame_id": fid, "query": "coin"}).data["objects"][0]
        pose = ep.call("vision_pose", {"frame_id": fid, "id": coin["id"]}).data
        assert ep.perception.mono_scales[fid] == pytest.approx(0.8, abs=0.02)
        truth = [ep.station.to_station(o.pose.pos) for n, o in ep.engine.state().objects.items()
                 if n.startswith("coin")]
        assert min(math.dist(pose["xyz_m"][:2], t[:2]) for t in truth) < 0.03
    finally:
        ep.close()


def test_never_better_than_a_real_robot():
    """Depth noise and calibration error can be made worse for an ablation,
    never better: a perception more exact than a real robot's would hand the
    agent positions it could not have."""
    from ruibench.core.config import RunConfig
    from ruibench.tasks.instance import build
    from ruikit.env.station import Station
    from ruikit.perception.backends.pixels import PixelPerception

    scene = build("DS4", level="easy", seed=0, config=RunConfig()).scene
    for params in ({"noise": 0.5}, {"calib": 0.25}, {"error_scale": 0.1}):
        per = PixelPerception(scene, Station(scene), 0, params)
        assert per.noise_scale >= 1.0 and per.calib_scale >= 1.0
    per = PixelPerception(scene, Station(scene), 0, {"noise": 4, "calib": 2})
    assert (per.noise_scale, per.calib_scale) == (4.0, 2.0)
    for wrong in ("rui.pose/v1@calib=0.5", "rui.pose/v1@noise=3"):
        with pytest.raises(ValueError):
            load_profile(wrong)

