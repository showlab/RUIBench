"""What the operator reads in the terminal, where a run's account scrolls by: what must
be answered and what the runner itself says stand out from it.  In colour on a terminal
only: in a file, or under ``NO_COLOR``, the words alone.
"""

from __future__ import annotations

import os
import sys

_QUESTION = "1;30;43"           # bold black on yellow
_HEADING = "1;36"               # bold cyan
_NOTICE = "1;33"                # bold yellow
_ALARM = "1;31"                 # bold red
RULE = "=" * 78


def _styled(style: str, text: str) -> str:
    if "NO_COLOR" in os.environ or not sys.stdout.isatty():
        return text
    return f"\033[{style}m{text}\033[0m"


def question(text: str) -> str:
    """The prompt of something the operator must answer before anything goes on."""
    return "\n" + _styled(_QUESTION, f" >>> {text} ") + " "


def heading(text: str) -> str:
    """Where an episode starts."""
    return _styled(_HEADING, f"\n{RULE}\n {text}\n{RULE}")


def notice(text: str) -> str:
    """What the runner itself says, among the account of what the agent does."""
    return _styled(_NOTICE, text)


def alarm(text: str) -> str:
    """Something went wrong."""
    return _styled(_ALARM, text)


__all__ = ["alarm", "heading", "notice", "question"]
