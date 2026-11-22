"""RUIBench -- what makes a good Robot-Use Interface.

30 tasks in five suites -- World knowledge, In-context learning, Discovery,
Dynamic interaction, Contact-rich manipulation -- each at an easy and a hard
level, with one shared controller, clock and scorer per robot.  The agents act
through RUIKit's 13 interfaces (``ruikit``).  ``docs/ARCHITECTURE.md`` explains
the boundaries.

    >>> from ruibench import tasks
    >>> len(tasks.main_board())
    30
"""

#: the package and the frozen benchmark it carries (ruibench/tasks/frozen.json)
__version__ = "1.0.0"

from .core.config import DEFAULT_T_REF_SECONDS, BudgetConfig, RunConfig

__all__ = ["RunConfig", "BudgetConfig", "DEFAULT_T_REF_SECONDS", "__version__"]
