"""The asset catalogue.

Task cards name *logical* assets ("duck_small", "basket"); this module resolves them
to USD paths.  One indirection, for one reason: when an asset turns out not to
exist, or a better one appears, exactly one file changes and no task card does.

Nothing here imports Isaac, so ``ruibench assets`` lists the catalogue on any
machine; only :func:`verify` needs a live Isaac install.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import yaml

from ...core.errors import ConfigError

CATALOGUE_PATH = Path(__file__).with_name("assets.yaml")
#: Generated.  The catalogue itself is hand-maintained and never rewritten --
#: an earlier version re-dumped it on every check and erased its comments.
LOCK_PATH = Path(__file__).with_name("assets.lock.json")


@dataclass(frozen=True)
class Primitive:
    """A shape we build rather than load, and the reason why."""

    name: str
    reason: str


@dataclass(frozen=True)
class AssetEntry:
    name: str
    usd: str
    category: str
    verified: bool
    physics: str = "builtin"        # builtin | added
    collision: str = "hull"         # hull | decomposition
    up: str = "z"                   # which axis the USD is authored with pointing up
    footprint: str = "rect"         # rect | round, seen from above
    #: measured, from the lock file; None until `checks geometry` has run
    half_extents: Optional[tuple] = None
    graspable: Optional[bool] = None
    # -- v2: a source outside the Isaac server, made ready by ``prepare`` -----
    #: "polyhaven:<id>", "gso:<name>", "objaverse:<folder>/<uid>", "ycb:<nnn_name>", "nucleus:<path>"
    source: Optional[str] = None
    #: degrees about x, then y, then z, after the source's own axes
    rotate: tuple = (0.0, 0.0, 0.0)
    #: the longest side after turning, in metres (the thing's real size)
    size: Optional[float] = None
    #: or a plain factor on top of the unit conversion
    scale: Optional[float] = None
    #: one prim inside the source, "component:K" for a GSO set, or
    #: "node:A,B" for the named glTF nodes of an Objaverse GLB, or "crop:S"
    #: for an Objaverse GLB without the faces in its lowest share S of height
    part: Optional[str] = None
    #: the source's metres per unit when its stage says otherwise
    units: Optional[float] = None
    #: a disc on a pot's inner bottom over its drainage holes -- ``radius``,
    #: ``above`` the outside bottom (the bottom's thickness), ``color`` (linear
    #: RGB, the clay's): a pot turned over something must not show it
    plug: Optional[dict] = None
    #: which of a Poly Haven model's texture sets to draw it with, by the maps'
    #: file names ("pieces_white"): a set's parts share one USD and one
    #: textures folder, and the first set by name is the default
    maps: Optional[str] = None
    #: with ``part: "node:..."``: the nodes' transforms baked into their
    #: vertices before converting.  The converter applies a kept node's
    #: transform twice: a non-uniform scale on the node came out squared --
    #: DS2's rattan basket 3.7 cm deep, not 8.4 (v10).  Off for the entries
    #: prepared before, which their ``rotate`` was chosen for
    bake: bool = False
    licence: str = ""
    attribution: str = ""

    @property
    def upright_quat(self) -> tuple:
        """Rotation that stands the asset up, in (w, x, y, z)."""
        if self.up == "y":
            return (0.70710678, 0.70710678, 0.0, 0.0)     # +90 degrees about x
        return (1.0, 0.0, 0.0, 0.0)

    @property
    def collision_approximation(self) -> str:
        return {"hull": "convexHull", "decomposition": "convexDecomposition"}[self.collision]

    def resolve(self, nucleus: Optional[str] = None) -> str:
        if self.usd.startswith("{PREPARED}"):
            from .prepare import PREPARED

            path = Path(self.usd.replace("{PREPARED}", str(PREPARED)))
            if not path.exists():
                raise ConfigError(f"asset {self.name!r} is not prepared yet ({path}); run "
                                  f"`python -m ruibench.engines.isaac.prepare {self.name}`")
            from .prepare import missing_sources

            gone = missing_sources(path)
            if gone:
                raise ConfigError(f"asset {self.name!r} is prepared but its scan is gone "
                                  f"({gone[0]}); run `python -m ruibench.engines.isaac.prepare "
                                  f"{self.name}` again")
            return str(path)
        root = nucleus or os.environ.get("RUIBENCH_ISAAC_NUCLEUS") or _nucleus_dir()
        return self.usd.replace("{NUCLEUS}", root)


def nucleus_dir() -> str:
    """The `/Isaac` directory assets resolve against: a local mirror
    (``RUIBENCH_ISAAC_NUCLEUS``) if set, else the asset server."""
    return os.environ.get("RUIBENCH_ISAAC_NUCLEUS") or _nucleus_dir()


def _nucleus_dir() -> str:
    """The `/Isaac` directory on the resolved asset server.

    Deliberately does not import isaaclab: this module is read by
    ``ruibench assets`` and ``ruibench validate`` on machines with no GPU.
    """
    from . import app

    root = app.asset_root() or app.PUBLIC_ASSET_ROOT
    return f"{root}/Isaac"


#: parsed files, by (which, modification time): the catalogue is read on the
#: engine's per-step path (a body's stand-up rotation, ``get``), and parsing
#: its YAML every time made a scene of nine scanned fruit 38 times slower than
#: real time -- a scene of blocks, 3 (Isaac, 2026-09-18)
_PARSED: Dict[tuple, object] = {}


def _mtime(path: Path) -> Optional[int]:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _catalogue_data() -> dict:
    key = ("catalogue", _mtime(CATALOGUE_PATH))
    if key not in _PARSED:
        _PARSED[key] = yaml.safe_load(CATALOGUE_PATH.read_text(encoding="utf-8"))
    return _PARSED[key]


def primitives() -> Dict[str, Primitive]:
    """Shapes deliberately built from primitives, with the reason recorded.

    A primitive is not a missing asset; it is a chosen one, and the choice has
    consequences worth remembering (a cylinder plate has no rim).
    """
    data = _catalogue_data()
    return {name: Primitive(name, reason)
            for name, reason in (data.get("primitives") or {}).items()}


#: A Franka hand opens to 8 cm; leave a little room to approach.
GRIPPER_OPENING = 0.075


def _lock_all() -> dict:
    import json
    try:
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _lock() -> Dict[str, dict]:
    return _lock_all().get("assets", {})


def _lock_section(section: str) -> Dict[str, dict]:
    return _lock_all().get(section, {})


def download_hash(url: str) -> Optional[str]:
    """The SHA-256 the lock file pins for a source download, if any."""
    return _lock_section("downloads").get(url)


def pin_download(url: str, sha256: str) -> None:
    """Pin a source download's SHA-256 in the lock file (a source checkout
    only: an installed package's lock is not ours to edit)."""
    import json

    from ...core.paths import checkout

    if checkout() is None:
        return
    data = _lock_all()
    downloads = data.setdefault("downloads", {})
    if downloads.get(url) == sha256:
        return
    downloads[url] = sha256
    data["downloads"] = dict(sorted(downloads.items()))
    LOCK_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def record_geometry(measured: Mapping[str, Mapping[str, dict]]) -> None:
    """Store measured geometry in the lock file, keeping everything else there
    -- and leave the file alone when nothing measured differs from what it
    holds, so preparing the assets on a new machine does not touch it."""
    import json

    data = _lock_all()
    stored = data.setdefault("geometry", {})
    fresh = {k: json.loads(json.dumps(dict(v))) for k, v in measured.items()}
    if all(stored.get(k) == v for k, v in fresh.items()):
        return
    stored.update(fresh)
    LOCK_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_catalogue() -> Dict[str, AssetEntry]:
    """Every catalogue entry, with what the lock file measured -- read again
    only when either file has changed (``prepare`` writes the lock and reads
    the catalogue back in one process)."""
    key = ("entries", _mtime(CATALOGUE_PATH), _mtime(LOCK_PATH))
    if key not in _PARSED:
        _PARSED[key] = _load_catalogue()
    return dict(_PARSED[key])


def _load_catalogue() -> Dict[str, AssetEntry]:
    data = _catalogue_data()
    lock = _lock()
    lock_geometry = _lock_section("geometry")
    entries = {}
    for name, d in data["assets"].items():
        record = lock.get(name, {})
        # verified only if the lock recorded *this* path resolving
        ok = bool(record.get("ok")) and record.get("usd") == d.get("usd")
        geometry = lock_geometry.get(name, {})
        up = d.get("up", "z")
        half = geometry.get("x+90" if up == "y" else "as_authored", {}).get("half_extents")
        source = d.get("source")
        usd = d.get("usd") or (f"{{PREPARED}}/{name}/{name}.usda" if source else None)
        if usd is None:
            raise ConfigError(f"asset {name!r} needs a `usd` or a `source`")
        if source:
            # a prepared asset counts as verified once it has been measured
            ok = bool(geometry)
        extra = dict(source=source, rotate=tuple(float(v) for v in d.get("rotate", (0, 0, 0))),
                     size=d.get("size"), scale=d.get("scale"), part=d.get("part"),
                     units=d.get("units"), plug=d.get("plug"), maps=d.get("maps"),
                     bake=bool(d.get("bake", False)),
                     licence=d.get("licence", ""), attribution=d.get("attribution", ""))
        entries[name] = AssetEntry(
            name, usd, d.get("category", "object"), ok,
            d.get("physics", "added" if source else "builtin"), d.get("collision", "hull"), up,
            d.get("footprint", "rect"),
            tuple(half) if half else None,
            (min(half[0], half[1]) * 2 <= GRIPPER_OPENING) if half else None, **extra)
    for alias, target in (data.get("aliases") or {}).items():
        if target not in entries:
            raise ConfigError(f"asset alias {alias!r} points at unknown asset {target!r}")
        entries[alias] = entries[target]
    return entries


def get(name: str) -> AssetEntry:
    cat = load_catalogue()
    try:
        return cat[name]
    except KeyError:
        raise ConfigError(
            f"unknown asset {name!r}; catalogue has {', '.join(sorted(cat))}. "
            f"Add it to {CATALOGUE_PATH.name} rather than putting a USD path in a task card."
        ) from None


@dataclass(frozen=True)
class Credit:
    """Where some of the things shown come from: one source, the licence they
    are shared under, and what was taken, as (name, link) pairs."""

    source: str
    licence: str
    items: Tuple[Tuple[str, str], ...]


#: licence identifiers as a credit line writes them
LICENCE_NAMES = {"CC-BY-4.0": "CC BY 4.0", "CC0-1.0": "CC0"}


def credits(names: Iterable[str]) -> List[Credit]:
    """Who made these catalogue assets: one line per source, the licences that
    ask to be credited (CC BY) first.  An entry's ``attribution`` reads
    "<source>, <what>" -- a link, or the source's own name for the thing."""
    cat = load_catalogue()
    grouped: Dict[tuple, Dict[str, str]] = {}
    for name in sorted(set(names)):
        entry = cat[name]
        if not entry.attribution:
            raise ConfigError(f"asset {name!r} has no `attribution` in {CATALOGUE_PATH.name}")
        source, _, what = entry.attribution.partition(", ")
        link = what if what.startswith("http") else ""
        label = link.rstrip("/").rsplit("/", 1)[-1] if link else what
        licence = LICENCE_NAMES.get(entry.licence, entry.licence)
        grouped.setdefault((source, licence), {})[label] = link
    rows = [Credit(source, licence, tuple(sorted(items.items())))
            for (source, licence), items in grouped.items()]
    return sorted(rows, key=lambda c: (c.licence == "CC0", c.source.lower()))


def unverified() -> List[str]:
    return sorted(n for n, e in load_catalogue().items() if not e.verified)


def verify(nucleus: Optional[str] = None, timeout: float = 25.0) -> Dict[str, bool]:
    """Check that every catalogue path actually resolves on the asset server.

    This is Gate A's asset check under the Isaac plan: it replaces "find out what
    LIBERO actually ships" with "find out what the Isaac asset server actually
    serves", and it answers the same question -- which tasks can be built today.

    Over HTTP when the root is a URL, which is the normal case and needs no
    running simulator; through ``omni.client`` for a Nucleus mount.
    """
    results: Dict[str, bool] = {}
    seen: Dict[str, bool] = {}
    for name, entry in load_catalogue().items():
        if entry.source:
            continue                      # checked when it is prepared (``prepare``)
        url = entry.resolve(nucleus)
        if url not in seen:
            seen[url] = (_head_ok(url, timeout) if url.startswith(("http://", "https://"))
                         else _omni_ok(url))
        results[name] = seen[url]
    return results


def _head_ok(url: str, timeout: float) -> bool:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _omni_ok(url: str) -> bool:
    try:
        import omni.client
        status, _ = omni.client.stat(url)
        return status == omni.client.Result.OK
    except Exception:
        return False


def mark_verified(results: Mapping[str, bool], root: Optional[str] = None) -> None:
    """Record verification results in the lock file, never in the catalogue --
    only in a source checkout, and only when some result changed (a check
    that finds what the file already says leaves it alone)."""
    import json
    from datetime import date

    from ...core.paths import checkout

    if checkout() is None:
        return                       # an installed package: the lock is not ours to edit
    data = yaml.safe_load(CATALOGUE_PATH.read_text(encoding="utf-8"))
    payload = _lock_all()
    assets = {name: {"usd": data["assets"][name].get("usd"), "ok": bool(ok)}
              for name, ok in results.items() if name in data["assets"]}
    if assets == payload.get("assets"):
        return
    payload.update({
        "generated_by": "ruibench doctor --assets; python -m ruibench.engines.isaac.checks geometry",
        "checked_on": date.today().isoformat(),
        "asset_root": root or _nucleus_dir(),
        "assets": assets,
    })
    LOCK_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
