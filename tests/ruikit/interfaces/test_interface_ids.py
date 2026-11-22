"""Every interface id resolves to exactly what v1.17 defined: tools and their
order, what programs may call, the title the agent is told, the parameters, the
log header -- generated from interaction form x action abstraction, not listed
by hand (``data/interfaces_v117.json``)."""
import json
from pathlib import Path

import pytest

from ruikit.interfaces import load_profile

V117 = json.loads((Path(__file__).parents[1] / "data" / "interfaces_v117.json").read_text())


@pytest.mark.parametrize("interface_id", sorted(V117))
def test_an_interface_is_what_v117_defined(interface_id):
    prof = load_profile(interface_id)
    got = prof.as_dict()
    got["_reproduces"], got["_differences"], got["_guide"] = (prof.reproduces, prof.differences,
                                                              prof.guide)
    assert json.dumps(got, ensure_ascii=False) == json.dumps(V117[interface_id], ensure_ascii=False)
