"""The alternatives: one module per task, each one way that decides differently from the
reference.  Each reads the simulator as the reference does; they are for training data and for
checking how much of a task's scoring depends on the reference's own choices."""

from __future__ import annotations

from ruikit.diagnostics.alter_reference.alt import (
    cl1, cl2, cl4, cl5, cl6, cr1, cr2, cr4, cr5, cr6, di4, di5, di6, ds3, ds5, ds6, wk2, wk3,
)

WAYS = [m.WAY for m in (cl1, cl2, cl4, cl5, cl6, cr1, cr2, cr4, cr5, cr6, di4, di5, di6, ds3, ds5,
                        ds6, wk2, wk3)]
