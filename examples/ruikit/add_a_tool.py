"""A tool and an interface from outside the library, used in an episode.

    python examples/ruikit/add_a_tool.py
"""

from pathlib import Path

from ruikit import Episode
from ruikit.core import plugins

here = Path(__file__).parent
plugins.load([str(here / "my_plugin" / "echo_tool.py")])   # registers debug_echo

ep = Episode("WK1", profile=str(here / "my_plugin" / "interface.yaml"), engine="mock")
try:
    print(ep.profile.id, "offers", ep.profile.tools[-1])
    print(ep.call("debug_echo", {"text": "hello from outside the library"}).text)
finally:
    ep.close()
