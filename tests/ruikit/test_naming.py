"""The interfaces' names follow the paper; the ids, and with them the agent's prompt, stay."""

import pytest

from ruibench import options as O
from ruikit import naming
from ruikit.experiments import results as cells
from ruikit.interfaces import list_profiles


def test_the_thirteen_ruis_have_the_papers_names_in_its_order():
    assert naming.RUIS == ("joint", "delta", "step", "pose", "point", "skill",
                           "program-joint", "program-delta", "program-step", "program-pose",
                           "program-point", "program-skill", "rui-free")
    assert [naming.label(n) for n in ("skill", "program-skill", "rui-free", "sequence-delta")] == [
        "Skill", "Program-Skill", "RUI-Free", "Sequence-Delta"]
    assert cells.COLUMNS == naming.RUIS


def test_every_name_is_an_interface_of_the_catalog_and_back():
    catalog = set(list_profiles())
    for name in naming.NAMES:
        assert naming.IDS[name] in catalog
        assert naming.name_of(naming.IDS[name]) == name
        assert naming.name_of(naming.id_of(name)) == name


def test_the_ids_stay_as_the_papers_runs_had_them():
    # the id is in every prompt the agent reads: renaming it would change the protocol
    assert naming.id_of("program-skill") == "rui.code-skill/v1+aids"
    assert naming.id_of("program-pose") == "rui.code/v1+aids"
    assert naming.id_of("rui-free") == "rui.free/v1"                 # its aids are built in
    assert naming.id_of("pose", aids=False) == "rui.pose/v1"
    assert naming.id_of("rui.code-pose/v1+aids") == "rui.code/v1+aids"
    assert naming.id_of("my/interface.yaml") == "my/interface.yaml"


@pytest.mark.parametrize("old,new", [("code-skill", "program-skill"), ("program", "program-pose"),
                                     ("code", "program-pose"), ("free", "rui-free"),
                                     ("seq-delta", "sequence-delta"), ("Program-Skill", "program-skill"),
                                     ("rui.code-joint/v1+aids@privileged", "program-joint")])
def test_older_names_and_ids_read_as_todays(old, new):
    assert naming.name_of(old) == new
    assert cells.normal_column(old) == new


def test_a_stored_result_without_a_column_takes_its_interfaces():
    assert cells.normal_column(None, "rui.code/v1+aids") == "program-pose"
    assert cells.normal_column("open/v1") == "open/v1"


def test_the_shared_options_read_one_grammar():
    assert O.normal_clock("zero_latency") == "paused" and O.normal_clock("8s") == "8s"
    assert O.normal_clock("controlled", 20.0) == "20s"
    assert O.engine_clock("8s") == ("controlled", 8.0)
    assert O.engine_clock("paused") == ("zero_latency", 0.0)
    with pytest.raises(ValueError):
        O.normal_clock("fast")
    ids = ["CR1", "CR2", "WK1", "WK2"]
    assert O.expand_tasks(["CR", "WK1"], ids) == ["CR1", "CR2", "WK1"]
    assert O.expand_tasks(["all"], ids) == ids
    assert O.parse_seeds("0-2,7", "WK1", "easy") == [0, 1, 2, 7]
