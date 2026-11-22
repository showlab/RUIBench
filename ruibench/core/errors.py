"""Exception hierarchy.  Every failure the harness can produce has a type."""

from __future__ import annotations


class RUIBenchError(Exception):
    """Base class for everything RUIBench raises."""


class ConfigError(RUIBenchError):
    """A task card, split file or run config is malformed or violates a design rule."""


class EngineNotAvailable(RUIBenchError):
    """An engine backend was requested but its dependencies are not installed."""


class FrozenAssetError(RUIBenchError):
    """An attempt to mutate something that has been frozen for the test split."""


class CoverageGap(RUIBenchError):
    """The interface cannot express what this task requires.

    Raised (or returned) by :meth:`ruibench.interfaces.Interface.coverage_gap`.
    Coverage gaps are reported separately and are **never** scored as a 0 --
    "the interface cannot say it" is not "the model cannot do it"
    (design doc v0 §7, TODO §3).
    """


class BudgetExceeded(RUIBenchError):
    """Wall-clock, sim-time, call-count or dollar budget ran out."""


class SceneRebuildRequired(RUIBenchError):
    """The next episode needs a different scene structure than the one loaded.

    Raised by engines that cannot tear a stage down in-process (Isaac: doing so
    with cameras attached hung for 15+ minutes).  The isolated runner answers it
    by finishing the worker and starting a fresh one.
    """
