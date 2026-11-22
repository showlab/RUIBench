"""Perception backends: what answers the perception tools.

    model        from the cameras' pictures: a detector and depth, chosen after the
                 interface id's ``@`` (``@owlv2-sam2``, ``@depth=mono``) -- the default
    privileged   the control condition: the simulator's own values (``@privileged``)
    synthetic    the mock engine's test double: it renders no pictures

The simulator's positions never reach a tool except under ``privileged``,
which is asked for by name only and recorded in every log.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple, Type

from ruikit.core.result import ToolError
from ruikit.perception.backends.base import PerceptionBackend
from ruikit.perception.backends.privileged import PrivilegedPerception
from ruikit.perception.backends.synthetic import SyntheticPerception


def _pixels() -> Type[PerceptionBackend]:
    from ruikit.perception.backends.pixels import PixelPerception

    return PixelPerception


#: the backends, by the interface's ``perception`` setting
BACKENDS: Mapping[str, Any] = {"model": _pixels, "privileged": lambda: PrivilegedPerception,
                               "synthetic": lambda: SyntheticPerception}


def depth_camera_needed(profile, renders: bool) -> bool:
    """Whether the cameras must deliver depth images: perception from pixels
    with the depth camera's depth, wherever pictures are rendered."""
    return (profile.perception == "model" and renders
            and profile.perception_settings()["depth"] == "sensor")


def open_perception(profile, scene, station, seed: int, renders: bool,
                    real: bool = False) -> Tuple[PerceptionBackend, Dict[str, Any]]:
    """The episode's perception backend, and what the log says of it.  From
    pixels wherever pictures are rendered; on the mock engine, which renders
    none, the test double stands in, and the log says so.  ``real``: the
    pictures come from a real robot's cameras, whose depth and calibration err
    as they err -- no simulated error is added to them."""
    errors = {**profile.tool_params("vision_pose"), **profile.tool_params("vision_lift")}
    if profile.perception == "synthetic" and renders:
        raise ToolError("not_supported", "perception: synthetic reads the simulator's "
                        "positions, which no tool hands an agent (v1.13): where pictures "
                        "are rendered, every vision tool answers from them",
                        hint="use the profile as it is (perception: model), or the mock "
                             "engine for tests")
    if profile.perception == "privileged":
        if real:
            raise ToolError("not_supported", "perception: privileged answers with the "
                            "simulator's own values, which a real robot does not have",
                            hint="use the profile as it is (perception: model)")
        return BACKENDS["privileged"]()(scene, station, seed, errors), {"perception": "privileged"}
    if profile.perception == "model" and renders:
        from ruikit.extras.backends import VISION, get_backend

        settings = profile.perception_settings()
        backend = get_backend(VISION, "vision", settings["backend"])
        #: which models answer, for the log -- asked now, so that a vision server
        #: that is not running stops the episode before it starts
        info = backend.info()
        if settings["depth"] == "mono" and not info.get("depth_model"):
            raise ToolError("backend_unavailable", "this profile measures depth from the "
                            "picture (depth=mono), and the vision server has no depth model",
                            hint="start it with --depth dav2-small")
        said = {"perception": "model", **info,
                **{k: settings[k] for k in ("depth", "noise", "calib")}}
        if real:
            said["sensors"] = "real"
        return BACKENDS["model"]()(scene, station, seed,
                                   {**errors, **settings, **({"sensors": "real"} if real else {})},
                                   backend=backend), said
    said = {"perception": "synthetic"}
    if profile.perception == "model":
        said["stand_in"] = ("the mock engine renders no pictures: a test double stands in "
                            "for perception from pixels")
    return BACKENDS["synthetic"]()(scene, station, seed, errors), said


__all__ = ["BACKENDS", "PerceptionBackend", "depth_camera_needed", "open_perception"]
