"""Playing a task's demonstrations before its episode.

A demonstration is a scripted policy run in its own scene on the episode's
engine, filmed from the episode's cameras at the episode's resolution: what
the agent receives is that film, a frame a simulated second, and a caption.
Scripted agents -- the references, the controls -- read the world states the
frames show instead of the pixels; a model has only the pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

from ..core.types import CameraFrame, Observation, WorldState
from ..executor import Executor
from ..interfaces.base import TurnContext
from ..interfaces.commands import Done

#: a frame every this many simulated seconds, and one at the end
FRAME_EVERY = 1.0
STEP = 0.1


@dataclass(frozen=True)
class Film:
    """A demonstration as an interface under test receives it: its caption
    and its frames -- never the world states behind them, which hold the
    answer (CL1's tune, which of CL2's dolls is Momo)."""

    caption: str
    #: (sim time, {camera: frame})
    frames: Tuple[Tuple[float, Dict[str, CameraFrame]], ...] = ()
    picture: bool = False

    @property
    def seconds(self) -> float:
        return self.frames[-1][0] if self.frames else 0.0


@dataclass
class Watched:
    caption: str
    #: (sim time, {camera: frame})
    frames: List[Tuple[float, Dict[str, CameraFrame]]] = field(default_factory=list)
    #: the world at each frame -- what the frames show
    states: List[WorldState] = field(default_factory=list)
    #: where it was shot, if not at the episode's table (``Demonstration.look``)
    look: Dict[str, object] = field(default_factory=dict)
    #: a still picture, one frame, not a film (``Demonstration.picture``)
    picture: bool = False

    @property
    def seconds(self) -> float:
        return self.frames[-1][0] if self.frames else 0.0

    def public(self) -> Film:
        """What a model may have of it: the film, not the states."""
        return Film(self.caption, tuple(self.frames), self.picture)

    def as_log(self) -> dict:
        out = {"caption": self.caption, "seconds": round(self.seconds, 2),
               "frames": len(self.frames)}
        if self.look:
            out["look"] = dict(self.look)
        if self.picture:
            out["picture"] = True
        return out


def _still(frame: CameraFrame) -> CameraFrame:
    """A frame that outlives the renderer's buffer."""
    image = frame.image
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy().copy()
    return replace(frame, image=image)


def play(engine, demo, cameras, sized, recorder=None, index: int = 0) -> Watched:
    """Run one demonstration to its end and film it -- at another table, in
    another light, if the demonstration says so (``Demonstration.look``): the
    engine shows it for the film and goes back to its own before the episode.
    An engine that draws nothing (the mock) only records it.  A picture from a file
    (a real task's, ``tasks.real.Drawn``) has nothing to play, on any engine."""
    if hasattr(demo, "drawn"):
        return Watched(demo.caption, frames=[(0.0, demo.drawn(cameras))], picture=True)
    engine.load(sized(demo.scene), 0)
    engine.reset()
    look = dict(getattr(demo, "look", None) or {})
    set_look = getattr(engine, "set_look", None) if look else None
    if set_look is not None:
        set_look(look)
    try:
        return _film(engine, demo, cameras, recorder, index, look)
    finally:
        if set_look is not None:
            set_look(None)


def _film(engine, demo, cameras, recorder, index: int, look) -> Watched:
    out = Watched(demo.caption, look=dict(look), picture=bool(getattr(demo, "picture", False)))

    def film() -> None:
        frames = {name: _still(engine.render(name)) for name in cameras}
        out.frames.append((engine.sim_time, frames))
        out.states.append(engine.state())
        if recorder is not None:
            recorder.demonstration(index, frames)

    if out.picture:
        film()                           # the scene as it stands: one picture
        return out
    executor = Executor(engine, demo.scene.arm_ids)
    policy = demo.demonstrator()
    policy.reset()
    reactor = demo.reactor_factory() if getattr(demo, "reactor_factory", None) else None
    film()
    last = engine.sim_time
    while engine.sim_time < demo.max_seconds:
        world = engine.state()
        obs = Observation(sim_time=world.sim_time, frames={},
                          proprioception={a: world.arm(a) for a in world.arm_ids},
                          instruction=demo.caption)
        ctx = TurnContext(observation=obs, task_id="demonstration", instruction=demo.caption,
                          arm_ids=demo.scene.arm_ids, turn_index=0, sim_time=world.sim_time,
                          budget_left={}, extras={"world": world, "executor_busy": executor.busy()})
        commands = policy.commands(world, ctx)
        finished = any(isinstance(c, Done) for c in commands)
        executor.submit_all([c for c in commands if not isinstance(c, Done)])
        waited = 0.0
        while (executor.busy() or executor.pending_wait > 0 or waited < STEP) \
                and engine.sim_time < demo.max_seconds:
            executor.tick(STEP)
            waited += STEP
            if reactor is not None:
                reactor.tick(engine, engine.state())
            if engine.sim_time - last >= FRAME_EVERY - 1e-9:
                film()
                last = engine.sim_time
        if finished:
            break
    film()
    return out
