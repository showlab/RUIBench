"""A Robot Agent -- a model, a Harness, an interface and its components -- and
running it on a task under an evaluation condition.

    import ruikit
    agent = ruikit.Agent(model="openai:<model>", harness="minimal-loop", interface="program-pose")
    ruikit.run(agent, task="CL2", level="easy", seed=15)                      # the main setting
    ruikit.run(agent, task="CL2", level="easy", seed=15, condition="paused")  # thinking is free

``interface`` is a name as the paper gives it (``program-pose``, ``skill``,
``rui-free``; ``ruikit.naming``), an id (``rui.code/v1``) or, as before, an
interaction form and an action abstraction (``("program", "pose")``).  The
paper's RUIs carry the aids, so a name or a form and abstraction comes with
them unless ``components`` says otherwise; an id runs exactly as written.

Each run writes one cell of the shared results layout
(``ruikit.experiments.results``) and returns its result row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from ruikit.conditions import DEFAULT_CLOCK, INFORMATION, Condition

#: the Harnesses' names as a person writes them -> their labels in the results
HARNESS_NAMES = {"minimal-loop": "loop", "minimal_loop": "loop", "loop": "loop", "codex": "codex",
                 "codex_code": "codex-code", "codex-code": "codex-code",
                 "claude_code": "claude-code", "claude-code": "claude-code"}


@dataclass(frozen=True)
class Agent:
    """A model (``openai:<model>``, ``local:<model>``), a Harness
    (``minimal-loop``, ``codex``, ``codex-code``, ``claude-code``), an interface
    -- a name, an id, or an interaction form and an action abstraction -- and the
    Harness components attached to it (``aids``, ``memory:image``, ``guide``)."""
    model: str
    harness: str = "minimal-loop"
    interface: Union[str, Tuple[str, str]] = "pose"
    components: Optional[Sequence[str]] = None

    @property
    def harness_label(self) -> str:
        try:
            return HARNESS_NAMES[self.harness]
        except KeyError:
            raise ValueError(f"no Harness {self.harness!r}; there are "
                             + ", ".join(sorted(HARNESS_NAMES))) from None

    def interface_id(self, condition: Condition = Condition()) -> str:
        """The interface id it runs with, under ``condition``."""
        from ruikit import naming
        from ruikit.interfaces import interface_for

        if isinstance(self.interface, str):
            named = naming.is_name(self.interface)
            base = naming.id_of(self.interface, aids=False)
        else:
            named = True
            form, action = self.interface
            base = interface_for(form.replace("-", "_"), action)
        components = self.components if self.components is not None else (
            ("aids",) if named else ())                            # the paper's RUIs carry the aids
        free = base.split("+")[0] == naming.IDS[naming.FREE]       # its aids are built in
        given = base.split("@")[0].split("+")[1:]                  # components the id already has
        extra = [c for c in components if c not in given and not (free and c == "aids")]
        return condition.interface(base + "".join(f"+{c}" for c in extra))


def as_condition(condition: Union[str, Condition, None]) -> Condition:
    """A condition from its name: a clock (``8s``, ``paused``, ``measured``) or
    ``privileged`` (the main clock with the simulator's values)."""
    if condition is None:
        return Condition()
    if isinstance(condition, Condition):
        return condition
    if condition in INFORMATION:
        return Condition(DEFAULT_CLOCK, condition)
    return Condition(condition)


def run(agent: Agent, task: str, level: str = "easy", seed: int = 0,
        condition: Union[str, Condition, None] = None, engine: str = "isaac",
        out: Union[str, Path] = "runs/ruikit", **options) -> dict:
    """One episode of ``agent`` on ``task`` into ``out``; its result row."""
    from ruikit.experiments.plans import Job
    from ruikit.harness.base import harnesses

    cond = as_condition(condition)
    job = Job(model=agent.model, task=task, level=level, seed=int(seed),
              profile=agent.interface_id(cond), harness=agent.harness_label, clock=cond.clock)
    return harnesses()[job.harness].run(job, Path(out), engine=engine, **options)
