"""The episode's logs: one row per call, and the episode record."""

from __future__ import annotations

import json

from ruibench.runner import snapshot
from ruibench.runner.log import EpisodeLog, write_jsonl
from ruikit.core.result import ToolResult


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return json.loads(json.dumps(obj, default=str))


class EpisodeLogging:
    """Writing the tool-call log and the episode record."""

    # -- logging -----------------------------------------------------------------
    def _log(self, name, args, think, t0, results, wall, inner=None) -> None:
        if self.image_dir is not None:
            self._keep_images(results)
        row = {
            "i": len(self.log), "turn": self.model_turns, "tool": name, "args": _jsonable(args),
            "think_s": round(think, 3), "t0": round(t0, 3), "t1": round(self.sim_time, 3),
            "wall_s": round(wall, 4),
            "results": [r.as_json() for r in results],
        }
        if inner:
            row["inner"] = inner
        self.log.append(row)

    def note_inner(self, name: str, args, result: ToolResult, t0: float) -> None:
        """A call made inside a reply (from a program or an agent-made tool)."""
        if self._inner is not None:
            data = result.as_json()
            data.pop("images", None)
            self._inner.append({"tool": name, "args": _jsonable(args), "t0": round(t0, 3),
                                "t1": round(self.sim_time, 3), "result": data})

    def _keep_images(self, results) -> None:
        """Save the pictures a reply shows the agent, once each, as JPEG named
        by frame id, and note the file on the image (``path`` in the log)."""
        import io

        from PIL import Image as PILImage

        for r in results:
            for im in getattr(r, "images", ()) or ():
                if not im.png:
                    continue
                cam = "" if (im.camera or "") in im.frame_id else f"-{im.camera}"
                name = f"{im.frame_id}{cam}{('-' + im.note) if im.note else ''}".replace("/", "_")
                name = "".join(ch if ch.isalnum() or ch in "._-#" else "_" for ch in name) + ".jpg"
                path = self.image_dir / name
                if not path.exists():
                    self.image_dir.mkdir(parents=True, exist_ok=True)
                    with PILImage.open(io.BytesIO(im.png)) as pic:
                        pic.convert("RGB").save(path, quality=85)
                im.path = name

    def _write_log(self) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.inst.task_id}_{self.inst.level}_s{self.inst.seed}_" \
               f"{self.profile.id.replace('/', '-')}"
        header = {"task": self.inst.task_id, "level": self.inst.level, "seed": self.inst.seed,
                  "profile": self.profile.as_dict(), "engine": self.cfg.engine,
                  "clock": self.clock, "think_seconds": self.think_seconds, "think_per": self.think_per,
                  "budget": self.budget.as_dict(), "instruction": self.inst.instruction,
                  "hint": self.inst.hint}
        with open(self.log_dir / f"{stem}.jsonl", "w") as fh:
            fh.write(json.dumps({"header": header}) + "\n")
            for row in self.log:
                fh.write(json.dumps(row) + "\n")
            fh.write(json.dumps({"outcome": self.outcome.__dict__ if self.outcome else None,
                                 "events": self._events, "leaks": self.leaks}) + "\n")
        if self.outcome is not None and getattr(self, "_score", None) is not None:
            write_jsonl(self.log_dir / f"{stem}.episode.jsonl", [self.episode_log(stem)])

    def episode_log(self, stem: str = ""):
        """The run as the benchmark logs its own (``ruibench-log/2``): the whole
        trajectory and what the scorer was told, so ``ruibench rescore`` judges it
        again in another process from this alone -- a leak included, which the
        scorer is told of in ``scoring``."""
        inst = self.inst
        log = EpisodeLog(
            run_id=self.cfg.run_id, task_id=inst.task_id, level=inst.level, seed=inst.seed,
            interface=f"ruikit:{self.profile.id}", agent=self.agent, engine=type(self.engine).__name__,
            clock=(f"{self.clock} {self.think_seconds:g} s/{self.think_per}" if self.clock == "fixed"
                   else self.clock),
            config=self.cfg.as_dict(), instruction=inst.instruction, hint=inst.hint,
            scene=inst.scene.summary(),
            engine_setup=self.engine.describe() if hasattr(self.engine, "describe") else {},
            condition="privileged" if self.profile.perception == "privileged" else "main")
        log.stamp("start")
        log.started_at = self._started_at
        records = list(self.scheduler.records) + list(getattr(self._reactor, "records", ()))
        log.events = [r.as_dict() for r in sorted(records, key=lambda r: r.sim_time)]
        log.handles = [h.as_dict() for h in self.executor.history]
        log.feedback = [m.as_dict() for m in self._said()]
        log.milestones_hit = dict(self.tracker.hit_at)
        log.trajectory = snapshot.dump_all(self.traj, inst.scene)
        log.score = self._score
        log.scoring = self._scoring
        log.metrics = dict(self.outcome.metrics)
        log.task_metrics = {name: fn(self.traj) for name, fn in inst.extra_metrics.items()}
        log.demonstrations = [w.as_log() for w in self._watched]
        log.frames = list(self._recorder.records) if self._recorder is not None else []
        log.provenance = {**log.provenance, "ruikit": {
            "tool_log": f"{stem}.jsonl", "reason": self.outcome.reason, "leaks": self.leaks,
            "perception": self.perception_info, "not_offered": dict(self.unavailable)}}
        log.stamp("end")
        return log
