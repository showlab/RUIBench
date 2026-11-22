"""Checking a skill's work (v2.5.2): after ``skill_pick`` and ``skill_place``
the interface looks -- the cameras' depth and the fingers, nothing else -- and
says in the result when the thing did not come with the gripper or does not
lie on the target (``report``, the default), or also corrects itself once
(``correct``: picks again; picks up what it dropped and places it again).  No
tool is added: the two skills take the setting (``ruikit.actions.checks``)."""

from __future__ import annotations

from ruikit.harness.components.base import Component, Implementation

_TOOLS = ("skill_pick", "skill_place")

COMPONENT = Component(
    "check", "完成检查", "the skills check their own picks and places", (),
    implementations=(
        Implementation("report", "检查并报告", "a failed check is said in the skill's result",
                       tools=(), params={t: {"check": "report"} for t in _TOOLS}),
        Implementation("correct", "检查并纠正", "a failed check is said, and the skill corrects "
                       "itself once without a new decision", tools=(),
                       params={t: {"check": "correct"} for t in _TOOLS}),
    ),
    default="report")
