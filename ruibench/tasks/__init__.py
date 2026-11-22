"""Task cards: the suites x (6 on the board + 1 held out).

    >>> from ruibench import tasks
    >>> len(tasks.main_board())
    24
    >>> tasks.get("DI4").knob.field
    'revision_window'
"""

from .loader import CARDS_DIR, by_suite, get, held_out, load_all, load_suite, main_board
from .spec import (
    ABILITIES, SUITE_ORDER, SUITES, AssetClass, Knob, SplitRole, Status, TaskSpec,
)
from .validate import Finding, errors, validate

__all__ = [
    "load_all", "load_suite", "get", "by_suite", "main_board", "held_out", "CARDS_DIR",
    "TaskSpec", "Knob", "ABILITIES", "SUITES", "SUITE_ORDER", "Status", "SplitRole", "AssetClass",
    "validate", "Finding", "errors",
]
