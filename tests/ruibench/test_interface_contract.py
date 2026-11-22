"""What any interface implementation must satisfy.

Run these against your own I1-I4 as you write them: they are the acceptance
criteria for "this is a RUIBench interface", not incidental unit tests.
"""

import pytest

from ruibench.interfaces import (
    BENCHMARK_INTERFACES, INTERFACES, Interface, contract_table, make_interface,
)
from ruibench.interfaces.base import Expressivity
from ruibench.interfaces.program import ProgramInterface


@pytest.mark.parametrize("name", INTERFACES.names())
def test_every_interface_declares_a_full_contract(name):
    iface = make_interface(name)
    row = iface.contract.as_row()
    for key in ("position", "orientation", "gripper", "blocking", "cancellable",
                "concurrent_arms", "camera_control", "services", "feedback_delivery",
                "reentry"):
        assert key in row and row[key] is not None, f"{name} leaves {key} undeclared"


@pytest.mark.parametrize("name", INTERFACES.names())
def test_every_interface_offers_the_same_services(name):
    """Perception is part of the fixed half of the design: if one interface got a
    service another did not, the benchmark would be measuring the service."""
    iface = make_interface(name)
    assert {"detect", "segment", "depth_at", "pose_of"} <= set(iface.contract.services)


def test_pose_and_program_expose_the_same_call_surface():
    """I2 and I4 are the cleanest contrast in the benchmark: identical callable
    API, the only difference being step-by-step calls versus a written program.
    Any drift is an unearned advantage, so it fails here rather than in a plot."""
    pose, program = make_interface("I2").contract, make_interface("I4").contract
    for field in ("position", "orientation", "gripper", "services"):
        assert getattr(pose, field) == getattr(program, field), field
    assert set(ProgramInterface.API_SURFACE) >= {
        "move_to", "gripper", "wait", "cancel", "detect", "segment",
        "depth_at", "pose_of", "feedback"}


def test_program_interface_can_read_feedback():
    """The one-bit correct/wrong signal reaches every interface, including this
    one -- otherwise In-Context Learning would be comparing information, not
    action languages."""
    assert make_interface("I4").contract.feedback_delivery != "none"
    assert "feedback" in ProgramInterface.API_SURFACE


def test_pointing_declares_its_orientation_limit():
    """A pixel plus an action type cannot carry a 6-DoF orientation.  Whichever
    way that is resolved, it has to be visible in the contract."""
    c = make_interface("I3").contract
    gap = make_interface("I3").coverage_gap(type("T", (), {"id": "WK3"})())
    assert c.orientation is Expressivity.NONE or gap is None


def test_test_doubles_are_not_benchmark_interfaces():
    for name in INTERFACES.names():
        iface = make_interface(name)
        assert iface.is_benchmark_interface == (name in BENCHMARK_INTERFACES)


@pytest.mark.parametrize("name", BENCHMARK_INTERFACES)
def test_unimplemented_interfaces_fail_loudly(name):
    """Until an interface is written, it must say so clearly rather than fail
    somewhere deep inside a run."""
    iface = make_interface(name)
    with pytest.raises(NotImplementedError) as exc:
        iface.system_prompt(None)
    assert name in str(exc.value)
