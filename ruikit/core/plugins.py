"""Tools from outside the library.

A tool is any ``@tool`` function (``ruikit.core.tool``): it is registered
when its module is imported.  ``load`` imports plugin files or modules, so
their tools are there before an interface that lists them is loaded --
without touching the frozen library (the built-in tools and
``ruikit/configs/interfaces.yaml``).  An interface can be a YAML file
anywhere: ``load_profile`` takes a path, and it may ``extends:`` one of the
library's.

    python -m ruikit.experiments.batch ... --plugins my/tools.py --profiles my/interface.yaml
    python -m ruikit.serve.session start ... --plugins my/tools.py --profile my/interface.yaml

Environment variables reach the batch workers and the session server, so a
plugin may read its own settings from them.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable, List, Optional


def load(specs: Iterable[str]) -> List[str]:
    """Import each plugin: a ``.py`` file, or a module name.  A file inside a
    package on ``sys.path`` is imported by its dotted name, so its own imports
    work; any other file by its name, from its folder.  Returns the module
    names, in order; importing one twice does nothing more."""
    names = []
    for spec in specs:
        if not spec:
            continue
        path = Path(spec)
        if path.suffix == ".py" or path.exists():
            if not path.is_file():
                raise FileNotFoundError(f"no plugin file {spec!r}")
            name = _dotted(path)
            if name is None:
                folder = str(path.resolve().parent)
                if folder not in sys.path:
                    sys.path.insert(0, folder)
                name = path.stem
        else:
            name = spec
        importlib.import_module(name)
        names.append(name)
    return names


def _dotted(path: Path) -> Optional[str]:
    """``my/tools.py`` under a ``sys.path`` folder -> ``my.tools``."""
    path = path.resolve()
    for entry in sys.path:
        try:
            rel = path.relative_to(Path(entry or ".").resolve())
        except (ValueError, OSError):
            continue
        parts = rel.with_suffix("").parts
        if parts and all(p.isidentifier() for p in parts):
            return ".".join(parts)
    return None
