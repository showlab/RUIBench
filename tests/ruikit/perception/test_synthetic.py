"""The synthetic perception rules (approach-v2 §3.1, §8.2 rules 4-5)."""

import pytest

from ruikit import Episode


def _look(ep, cam=None):
    r = ep.call("camera_look", {"cameras": [cam] if cam else None})
    return [f["frame_id"] for f in r.data["frames"]]


@pytest.mark.parametrize("task,hidden", [("DS2", "egg"), ("DS3", "gold coin"),
                                         ("DI6", "cake")])      # under its cover (bench-issue #15)
def test_hidden_things_are_not_reported(task, hidden):
    ep = Episode(task, level="easy", seed=0, profile="rui.pose/v1")
    try:
        for fid in _look(ep):
            labels = [o["label"] for o in ep.call("vision_detect", {"frame_id": fid}).data["objects"]]
            assert hidden not in labels, (fid, labels)
            assert not ep.call("vision_detect", {"frame_id": fid, "query": hidden}).data["objects"]
    finally:
        ep.close()


def test_ids_are_frame_local():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        f1, f2 = _look(ep, "front")[0], _look(ep, "front")[0]
        a = ep.call("vision_detect", {"frame_id": f1}).data["objects"]
        b = ep.call("vision_detect", {"frame_id": f2}).data["objects"]
        assert all(o["id"].startswith(f1 + "#") for o in a)
        assert all(o["id"].startswith(f2 + "#") for o in b)
        # an id of one frame means nothing in another
        r = ep.call("vision_pose", {"frame_id": f2, "id": a[0]["id"]})
        assert r.error == "invalid_args"
    finally:
        ep.close()


def test_labels_never_carry_internal_names():
    ep = Episode("DS1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        names = [b.name.lower() for b in ep.scene.bodies]
        for fid in _look(ep):
            for o in ep.call("vision_detect", {"frame_id": fid}).data["objects"]:
                text = f"{o['label']} {o.get('color', '')}".lower()
                for secret in ("tom", "tim", "amy", "leo", "mia"):
                    assert secret not in text
                assert o["id"] not in names
    finally:
        ep.close()


def test_the_same_question_gets_the_same_answer():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        fid = _look(ep, "front")[0]
        obj = ep.call("vision_detect", {"frame_id": fid, "query": "ring"}).data["objects"][0]
        answers = {tuple(ep.call("vision_pose", {"frame_id": fid, "id": obj["id"]}).data["xyz_m"])
                   for _ in range(20)}
        assert len(answers) == 1                      # averaging buys nothing
    finally:
        ep.close()


def test_pixels_outside_the_image_are_refused():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        fid = _look(ep, "front")[0]
        assert ep.call("vision_lift", {"frame_id": fid, "u": -10000, "v": -10000}).error == "not_visible"
        assert ep.call("vision_lift", {"frame_id": "f999.front", "u": 1, "v": 1}).error == "stale_frame"
    finally:
        ep.close()


def test_lift_and_point_agree():
    """robot_move_to_pixel goes where vision_lift says the pixel is."""
    ep = Episode("WK1", level="easy", seed=0, profile="rui.point/v1")
    try:
        fid = _look(ep, "front")[0]
        obj = ep.call("vision_detect", {"frame_id": fid, "query": "stand"}).data["objects"][0]
        u, v = obj["center"]
        lifted = ep.call("vision_lift", {"frame_id": fid, "u": u, "v": v}).data["xyz_m"]
        moved = ep.call("robot_move_to_pixel", {"frame_id": fid, "u": u, "v": v, "height_m": 0.1})
        assert not moved.is_error, moved.text
        assert moved.data["tcp"]["xyz_m"][2] == pytest.approx(lifted[2] + 0.1, abs=0.01)
    finally:
        ep.close()


def test_a_frame_keeps_its_own_pixels():
    """An engine may hand back a view of its camera's buffer (Isaac does), which
    the next render writes over: a frame read again later must still show what
    it was taken of."""
    import numpy as np

    from ruibench.engines.base import CameraFrame

    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        buffer = np.zeros((8, 8, 3), dtype=np.uint8)

        def render(camera):
            return CameraFrame(camera=camera, width=8, height=8, captured_at=ep.sim_time, image=buffer)

        ep.engine.render = render
        buffer[:] = 10
        first = ep.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
        buffer[:] = 200                  # the camera renders again into the same buffer
        ep.call("camera_look", {"cameras": ["front"]})
        assert int(ep.frames.get(first).image.max()) == 10
    finally:
        ep.close()
