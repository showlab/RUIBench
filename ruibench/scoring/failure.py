"""Failure labels.

Four categories, as fixed by the design (v0 §7, TODO §5), plus the two the
harness itself can produce.  The distinction that matters most:
``NOT_EXPRESSIBLE`` is an **interface** limitation and is reported in its own
column -- it is never averaged into a success rate.
"""

from __future__ import annotations

from enum import Enum


class FailureLabel(str, Enum):
    NONE = "none"                       # succeeded
    DECISION_ERROR = "decision_error"   # said the wrong thing
    NOT_EXPRESSIBLE = "not_expressible"  # the action language cannot say it
    EXECUTION_FAILURE = "execution_failure"  # said the right thing, physics did not follow
    TIMEOUT = "timeout"                 # ran out of time
    BUDGET = "budget"                   # ran out of calls or dollars
    VIOLATION = "violation"             # broke a task rule
    PHYSICS_FLAG = "physics_flag"       # the physics audit flagged it: a failure until reviewed
    SECRET_LEAK = "secret_leak"         # a tool showed the agent a secret: void, never a success
    HARNESS_ERROR = "harness_error"     # our bug; never counted as a model result

    @property
    def counts_as_model_failure(self) -> bool:
        return self in (FailureLabel.DECISION_ERROR, FailureLabel.EXECUTION_FAILURE,
                        FailureLabel.TIMEOUT, FailureLabel.BUDGET, FailureLabel.VIOLATION,
                        FailureLabel.PHYSICS_FLAG)
