"""One episode driven by hand, one tool call at a time, on the mock engine.

    python examples/ruikit/episode_by_hand.py

The mock renders no pictures, so a test double stands in for perception and the
result never counts; pass engine="isaac" (with the vision service running) for one
that does.
"""

from ruikit import Episode

ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1", engine="mock")
try:
    print(ep.call("task_info", {}).text)
    look = ep.call("camera_look", {"cameras": ["front"]})
    frame = look.data["frames"][0]["frame_id"]
    rings = ep.call("vision_detect", {"frame_id": frame, "query": "ring"}).data["objects"]
    print(f"{len(rings)} rings in {frame}")
    pose = ep.call("vision_pose", {"frame_id": frame, "id": rings[0]["id"]}).data
    x, y, z = pose["xyz_m"]
    moved = ep.call("robot_move_to", {"xyz_m": [x, y, z + 0.10]})
    print(moved.text)
    ep.call("task_done", {"report": "looked, found the rings, went above one"})
    print(ep.outcome.reason, "success" if ep.outcome.success else "not done")
finally:
    ep.close()
