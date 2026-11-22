"""Alternatives to the references: found by name, on their own task, and different from the reference."""

import pytest

from ruibench.tasks.instance import build
from ruikit.diagnostics.alter_reference import _all, install, make, ways


@pytest.mark.parametrize("name", sorted(_all()))
def test_each_way_makes_a_policy_of_its_own_on_its_task(name):
    way = _all()[name]
    inst = build(way.task, "easy", 0)
    policy = make(name, inst)
    assert policy.name != inst.reference_factory(inst).name
    assert name in ways(way.task)


def test_a_way_is_refused_on_another_task():
    with pytest.raises(ValueError):
        make("alt:wk2_other_crossing", build("DS5", "easy", 0))


def test_an_unknown_way_is_refused():
    with pytest.raises(KeyError):
        make("alt:nothing", build("WK2", "easy", 0))


def test_install_gives_the_way_in_place_of_the_reference():
    inst = build("WK2", "easy", 0)
    reference = inst.reference_factory(inst).name
    install("alt:wk2_other_crossing", inst)
    assert inst.reference_factory(inst).name != reference


@pytest.mark.parametrize("level", ["easy", "hard"])
def test_the_other_crossing_is_safe_and_not_the_references(level):
    from ruibench.tasks.impl.wk2_river import FAR, NEAR, safe, solve
    from ruikit.diagnostics.alter_reference.alt.wk2 import other_crossing

    inst = build("WK2", level, 0)
    policy = inst.reference_factory(inst)
    plan = other_crossing(policy.names, policy.rules)
    assert plan != solve(policy.names, policy.rules)
    farmer, far = NEAR, frozenset()
    for load in plan:
        if load is not None:
            far = far - {load} if farmer == FAR else far | {load}
        farmer = NEAR if farmer == FAR else FAR
        assert safe((farmer, far), policy.names, policy.rules)
    assert (farmer, far) == (FAR, frozenset(policy.names))


def test_reference_option_runs_a_way_through_an_interface_on_the_mock():
    from ruikit.diagnostics.coverage import run_cell

    cell = run_cell("DS6", "easy", 1, "rui.code-joint/v1", reference="alt:ds6_from_the_right",
                    oracle_coordinates=True)
    assert cell.success, (cell.reason, cell.error)
