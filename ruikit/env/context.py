"""What a tool may touch: :class:`ToolContext`.

A tool never sees the engine, the task instance, the scorer or a
``WorldState``.  It gets this object, whose methods are the only doors:

=================  ===========================================================
``capture``        take a picture now (the engine renders; the frame keeps the
                   camera pose and, server-side, what was in view)
``frame``          look up an earlier frame by id
``arm_state``      proprioception an agent is entitled to: TCP pose, gripper
                   opening, whether the fingers stopped on something (from the
                   fingers' own readings, as a real gripper tells it; not *what*)
``move`` / ...     submit motions through the shared executor, get a handle
``motion_result``  block (or not) and report what the motion achieved
``stop`` / wait    stop an arm, wait for handles, seconds or the next event
``perception``     the configured backend of the ``vision_*`` tools
``task``           instruction, hint, demonstration frames, events, done
``call``           another tool of this interface (code mode, custom tools)
=================  ===========================================================

Each door enforces three rules: information only from sensors, actions only
through the controller, and everything costs time.
"""

from __future__ import annotations

import math
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

from ruibench.core.types import Pose
from ruibench.engines.base import (
    ARRIVAL_POS_TOL,
    ARRIVAL_ROT_TOL,
    arm_camera_model,
    camera_apertures,
    camera_focal_length,
    external_camera_pose,
    wrist_camera_pose,
)
from ruikit.core.result import ToolError, ToolResult
from ruikit.env.frames import Frame, to_array
from ruikit.env.motion import MotionContext
from ruikit.perception.proprioception import holding_something


class TaskView:
    def __init__(self, episode) -> None:
        self._ep = episode

    @property
    def instruction(self) -> str:
        return self._ep.instruction

    @property
    def instruction_version(self) -> int:
        return self._ep.traj[-1].instruction_version

    @property
    def hint(self) -> str:
        return self._ep.inst.hint or ""

    @property
    def task_id(self) -> str:
        return self._ep.inst.task_id

    def demos(self) -> list:
        """Films shown before the episode: frames and a caption, never the
        world states the frames show (those stay with the scripted references)."""
        out = []
        for i, w in enumerate(self._ep._watched):
            out.append({"index": i, "caption": w.caption, "seconds": round(w.seconds, 2),
                        "picture": bool(getattr(w, "picture", False)),
                        "frames": [(t, {self._ep.station.camera_name(c): f
                                        for c, f in cams.items()}) for t, cams in w.frames]})
        return out

    def events(self, since: int = 0) -> List[dict]:
        return self._ep.events(since)

    def done(self, report: str) -> None:
        self._ep.request_done(report)

    def give_up(self, reason: str) -> None:
        self._ep.give_up(reason)


class ToolContext(MotionContext):
    def __init__(self, episode, tool_name: str, from_code: bool = False) -> None:
        self._ep = episode
        self.tool_name = tool_name
        self.from_code = from_code
        self.station = episode.station
        self.params: Dict[str, Any] = dict(episode.profile.tool_params(tool_name))
        self.profile = episode.profile
        self.task = TaskView(episode)

    # -- time ------------------------------------------------------------------
    @property
    def sim_time(self) -> float:
        return self._ep.sim_time

    def advance(self, seconds: float) -> None:
        self._ep.advance(seconds)

    # -- sensing -----------------------------------------------------------------
    def cameras(self) -> List[str]:
        return sorted(self.station.cameras)

    def camera_pose(self, scene_camera: str) -> Pose:
        spec = next(c for c in self._ep.scene.cameras if c.name == scene_camera)
        engine = self._ep.engine
        if hasattr(engine, "camera_pose"):
            return engine.camera_pose(scene_camera)
        if spec.kind == "wrist":
            arm = spec.attached_to or self._ep.scene.arm_ids[0]
            tcp = engine.state().arm(arm).eef_pose
            return wrist_camera_pose(tcp, arm_camera_model(self._ep.scene, arm))
        return spec.pose or external_camera_pose(self._ep.scene)

    def capture(self, camera: str) -> Frame:
        """Render one camera now.  The frame remembers where the camera was and
        (server side) which bodies were in view."""
        scene_camera = self.station.camera(camera)
        engine = self._ep.engine
        spec = next(c for c in self._ep.scene.cameras if c.name == scene_camera)
        raw = engine.render(scene_camera)
        # the frame keeps its own pixels: Isaac hands back a view of the camera's
        # buffer, which the next render writes over -- an older frame read again
        # (zoom, memory_frames, marks) would otherwise show a later picture
        image = to_array(getattr(raw, "image", None))
        if image is not None:
            image = image.copy()
        depth = getattr(raw, "depth", None)
        if depth is not None:
            import numpy as np

            if hasattr(depth, "detach"):
                depth = depth.detach().cpu().numpy()
            depth = np.array(depth, dtype=np.float32)
        # both engines answer this by sight lines / segmentation whatever the
        # task's own visibility tracking is set to
        visible = tuple(engine.visible_objects(scene_camera))
        # the privileged condition's true pixels of each thing, where the engine has them
        masks = (engine.object_masks(scene_camera)
                 if self._ep.profile.perception == "privileged" else {})
        world = engine.state()
        width = int(getattr(raw, "width", 0) or spec.width)
        height = int(getattr(raw, "height", 0) or spec.height)
        frame = Frame(id=self._ep.frames.new_id(camera), camera=camera,
                      scene_camera=scene_camera, sim_time=self.sim_time,
                      width=width, height=height, pose=self.camera_pose(scene_camera),
                      focal_mm=camera_focal_length(self._ep.scene, spec),
                      apertures=camera_apertures(self._ep.scene, spec),
                      image=image, visible=visible, masks=masks,
                      objects=dict(world.objects), depth=depth,
                      arms={arm_id: (world.arm(arm_id).eef_pose,
                                     getattr(world.arm(arm_id), "opening", None))
                            for arm_id in self._ep.scene.arm_ids})
        return self._ep.frames.add(frame)

    def calibration(self, frame: Frame):
        """Where the robot believes the camera was for this frame -- its
        calibration, off by the camera's own fixed error, never the simulator's pose."""
        return self._ep.perception.believed(frame)

    @property
    def camera_size(self) -> tuple:
        """(width, height) the cameras deliver in this episode."""
        return self._ep.cfg.camera_width, self._ep.cfg.camera_height

    def frame(self, frame_id: str) -> Frame:
        return self._ep.frames.get(frame_id)

    def recent_frames(self, n: int) -> List[Frame]:
        return self._ep.frames.last(n)

    @property
    def perception(self):
        return self._ep.perception

    def tcp(self, arm: str) -> Pose:
        info = self.station.arm(arm)
        return self._ep.engine.state().arm(info.arm_id).eef_pose

    def base_pose(self, arm: str) -> Pose:
        """Where a relative move starts: the last move's target while the hand is
        still closing on it, else where the hand is.  A move is done once the
        hand is within 2 mm / 1.15 degrees of its target and the hand closes the
        rest in about half a second; a step taken from the measured pose loses
        that gap every time (on WK5 through the step interface, a 1 cm step moved
        0.86 cm and six 15-degree turns made 83.8)."""
        info = self.station.arm(arm)
        now = self.tcp(info.name)
        last = next((h for h in reversed(list(self._ep.handles.values()))
                     if h.arm == info.name and h.target is not None), None)
        if (last is not None and last.status == "done"
                and now.distance_to(last.target) <= ARRIVAL_POS_TOL
                and now.angle_to(last.target) <= ARRIVAL_ROT_TOL):
            return last.target
        return now

    def arm_state(self, arm: str) -> dict:
        info = self.station.arm(arm)
        st = self._ep.engine.state().arm(info.arm_id)
        joints = self.arm_joints(info.name)
        busy = self._ep.arm_busy(info.name)
        return {"arm": info.name, "tcp": self.station.describe_pose(st.eef_pose),
                # measured, as a real gripper's encoders give it: on something it
                # closed on, that thing's width; engines that cannot tell fall back to
                # the command, and say nothing while holding
                "gripper_opening_m": (round(st.opening, 4) if getattr(st, "opening", None) is not None
                                      else None if st.holding is not None else
                                      round((1.0 - min(1.0, max(0.0, st.gripper))) *
                                            info.max_opening, 4)),
                "gripper_closed": st.gripper >= 0.5,
                "holding_something": holding_something(st),
                "running": busy.id if busy else None,
                # the joint encoders, where the engine has joints (Isaac; the mock
                # computes them from the robot's kinematics)
                **({"joints_deg": [round(math.degrees(q), 1) for q in joints]}
                   if joints is not None else {})}


    def arm_joints(self, arm: str) -> Optional[Tuple[float, ...]]:
        """The arm's joint readings (radians, base to wrist), or None."""
        try:
            return self._ep.engine.arm_joints(self.station.arm(arm).arm_id)
        except NotImplementedError:
            return None


    # -- composition -------------------------------------------------------------
    def call(self, name: str, args: Optional[Mapping[str, Any]] = None,
             allowed: Optional[Iterable[str]] = None) -> ToolResult:
        """Call another tool of this interface from inside a tool (code_run, a
        custom tool).  ``allowed`` narrows what may be called; nested calls
        charge their own costs, never thinking time."""
        prof = self._ep.profile
        t0 = self.sim_time
        if allowed is not None and name not in set(allowed):
            result = ToolResult.fail(ToolError("not_in_profile", f"{name!r} cannot be called here",
                                               hint=f"callable: {', '.join(sorted(allowed))}"))
        elif prof.callable_from_code(name):
            result = self._ep._invoke(name, args, from_code=True)
        elif name in prof.tools or name in self._ep.custom_tools:
            result = self._ep._invoke(name, args, from_code=False)
        else:
            result = ToolResult.fail(ToolError("not_in_profile", f"{name!r} is not available"))
        result.sim_time = self.sim_time
        self._ep.scan(name, result)          # a program sees it even if it never prints it
        self._ep.note_inner(name, args, result, t0)
        return result

    def sandbox(self):
        """The episode's program process (created on first use)."""
        if self._ep.sandbox is None:
            from ruikit.interfaces.sandbox.host import Sandbox

            self._ep.sandbox = Sandbox(self.params.get("isolation", "auto"))
        return self._ep.sandbox

    def custom_tools(self) -> dict:
        return self._ep.custom_tools

    def home_pose(self, arm: str) -> Pose:
        from ruibench.engines.base import arm_start_pose

        info = self.station.arm(arm)
        robot = self._ep.scene.robots[0]
        return arm_start_pose(self._ep.scene, robot, info.name)

    @property
    def memory(self) -> Dict[str, Any]:
        return self._ep.memory

    @property
    def episode_seed(self) -> int:
        return self._ep.inst.seed
