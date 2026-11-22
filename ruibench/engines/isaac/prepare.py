"""Catalogue assets from outside the Isaac asset server, made ready to spawn.

Since v2: objects are real
things -- scans and photoreal models from Poly Haven (CC0), Google Scanned
Objects (CC-BY 4.0), Objaverse's CC-BY models, and NVIDIA's ArchVis and SimReady
props.  Each comes in its own units, axes and origin, and a task needs none of
that: RUIBench's convention is metres, z up, standing = identity, and **the
body's pose is the middle of its bounding box** (``BodySpec.size`` is its half
extents).  So every such entry gets a small local USD that references the
source and puts it right::

    /root                      the rigid body (physics is added at spawn)
      /geometry                scale to metres and to the entry's size,
                               turn it (``rotate``), centre it
        /source                -> the source (or one part of it), as authored

Source files are fetched once into ``third_party/assets_raw`` and converted
there (OBJ and GLB through Isaac's asset converter); the wrappers go to
``third_party/assets_usd/catalogue``.  What is measured -- half extents,
centre offset, narrow side -- goes to ``assets.lock.json`` like the Isaac
props' geometry, so the mock engine and ``ruibench validate`` know sizes
without a simulator.

    python -m ruibench.engines.isaac.prepare [names...] [--force]

needs Isaac (the converter and pxr are Kit extensions).
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ...core.errors import ConfigError
from .robots import THIRD_PARTY

RAW = THIRD_PARTY / "assets_raw"
PREPARED = THIRD_PARTY / "assets_usd" / "catalogue"

POLYHAVEN_API = "https://api.polyhaven.com/files/{id}"
GSO_ZIP = "https://fuel.gazebosim.org/1.0/GoogleResearch/models/{id}/1/{id}.zip"
OBJAVERSE_GLB = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/{folder}/{uid}.glb"
YCB_TGZ = "https://ycb-benchmarks.s3.amazonaws.com/data/google/{id}_google_16k.tgz"


#: Poly Haven's API refuses Python's default user agent and asks callers to name themselves
USER_AGENT = "RUIBench-asset-prep/1.0 (research benchmark; github.com/showlab/RUIBench)"


def say(*a) -> None:
    print("PREP", *a, flush=True)


def _open(url: str, timeout: float = 120):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
                                  timeout=timeout)


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, path: Path) -> Path:
    """``url`` saved at ``path`` -- once; checked against the SHA-256 the lock
    file pins for it (``assets.lock.json``, "downloads"), since Poly Haven's
    API serves the latest file and Objaverse is fetched from a branch: a file
    that changed upstream would quietly change the benchmark.  A source the
    lock does not pin yet is pinned, in a source checkout."""
    from .assets import download_hash, pin_download

    fresh = not (path.exists() and path.stat().st_size > 0)
    if fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
        try:
            with _open(url) as response, open(tmp, "wb") as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
        except OSError as exc:
            raise ConfigError(f"could not fetch {url}: {exc}") from None
        os.replace(tmp, path)
    pinned = download_hash(url)
    if pinned is None:
        pin_download(url, _sha256(path))
    elif fresh:
        got = _sha256(path)
        if got != pinned:
            path.unlink()
            raise ConfigError(f"{url} is not the file RUIBench was built with "
                              f"(sha256 {got[:12]}..., pinned {pinned[:12]}...): it changed upstream")
    return path


# -- sources -----------------------------------------------------------------
def fetch_polyhaven(ident: str) -> Path:
    """The model's USD and every texture it names, keeping their relative paths."""
    folder = RAW / "polyhaven" / ident
    with _open(POLYHAVEN_API.format(id=ident)) as response:
        files = json.load(response)
    usd = files["usd"]["2k"]["usd"]
    main = _download(usd["url"], folder / Path(usd["url"]).name)
    for rel, info in (usd.get("include") or {}).items():
        _download(info["url"], folder / rel)
    return main


def fetch_gso(ident: str, part: Optional[str] = None) -> Path:
    """A GSO scan as USD.  ``part`` "component:K" keeps the K-th largest
    connected piece of a set scanned together (a set's pieces that did not
    touch come apart; ones that did stay fused)."""
    folder = RAW / "gso" / ident
    archive = _download(GSO_ZIP.format(id=ident), folder / f"{ident}.zip")
    obj = folder / "meshes" / "model.obj"
    if not obj.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(folder)
    # the scan's material names "texture.png" beside the OBJ, and the converter
    # writes "./textures/texture.png" beside its USD; the zip keeps it in
    # materials/textures/ -- without these copies the scan renders white
    texture = folder / "materials" / "textures" / "texture.png"
    for place in (obj.parent / "texture.png", obj.parent / "textures" / "texture.png"):
        if texture.exists() and not place.exists():
            place.parent.mkdir(parents=True, exist_ok=True)
            place.write_bytes(texture.read_bytes())
    if part and part.startswith("component:"):
        k = int(part.split(":", 1)[1])
        obj = _split_component(obj, k)
    elif part and part.startswith("figure:"):
        k = int(part.split(":", 1)[1])
        obj = _split_figure(obj, k)
    return _convert(obj, obj.with_suffix(".usd"))


def _split_component(obj: Path, k: int) -> Path:
    out = obj.with_name(f"part{k}.obj")
    if out.exists():
        return out
    import trimesh

    mesh = trimesh.load(obj, force="mesh", process=False)
    pieces = sorted(mesh.split(only_watertight=False), key=lambda m: -m.bounding_box.volume)
    if k >= len(pieces):
        raise ConfigError(f"{obj} has {len(pieces)} separate pieces, no piece {k}")
    pieces[k].export(out)
    return out


#: a figure's body is a piece at least this tall; smaller pieces are parts of one
FIGURE_BODY = 0.05


def _split_figure(obj: Path, k: int) -> Path:
    """The K-th largest figure of a group scanned together, whole.  A wooden
    doll's head, hair or hat is its own piece -- it only rests on the body --
    so a figure is a body (a piece at least ``FIGURE_BODY`` tall) with every
    smaller piece that sits over it: over its footprint, and no lower than
    its bottom.  Scraps that sit over no body are left out."""
    out = obj.with_name(f"figure{k}.obj")
    if out.exists():
        return out
    import trimesh

    mesh = trimesh.load(obj, force="mesh", process=False)
    pieces = mesh.split(only_watertight=False)
    bodies = sorted((p for p in pieces if p.extents[2] >= FIGURE_BODY),
                    key=lambda m: -m.bounding_box.volume)
    if k >= len(bodies):
        raise ConfigError(f"{obj} has {len(bodies)} figures, no figure {k}")
    body = bodies[k]
    parts = [body]
    for piece in pieces:
        if any(piece is b for b in bodies):
            continue
        cx, cy, cz = piece.bounds.mean(axis=0)
        over = [b for b in bodies if b.bounds[0][0] <= cx <= b.bounds[1][0]
                and b.bounds[0][1] <= cy <= b.bounds[1][1] and cz >= b.bounds[0][2]]
        if over and min(over, key=lambda b: (b.bounds.mean(axis=0)[0] - cx) ** 2
                        + (b.bounds.mean(axis=0)[1] - cy) ** 2) is body:
            parts.append(piece)
    merged = trimesh.util.concatenate(parts)
    # the pieces share the scan's texture, but concatenating drops the image:
    # put the body's material back, or the figure renders white
    material = getattr(body.visual, "material", None)
    if material is not None and getattr(merged.visual, "uv", None) is not None:
        merged.visual = trimesh.visual.TextureVisuals(uv=merged.visual.uv, material=material)
    merged.export(out)
    return out


def fetch_ycb(ident: str) -> Path:
    """A YCB object's 16k-triangle scan (a textured OBJ) as USD; ``ident`` is
    its numbered name, "056_tennis_ball".  Isaac's asset server carries only
    some of the set -- none of its balls."""
    import tarfile

    folder = RAW / "ycb" / ident
    archive = _download(YCB_TGZ.format(id=ident), folder / f"{ident}_google_16k.tgz")
    obj = folder / ident / "google_16k" / "textured.obj"
    if not obj.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(folder)
    # as for GSO: the converter looks for the texture in textures/ beside its USD
    texture, place = obj.parent / "texture_map.png", obj.parent / "textures" / "texture_map.png"
    if texture.exists() and not place.exists():
        place.parent.mkdir(parents=True, exist_ok=True)
        place.write_bytes(texture.read_bytes())
    return _convert(obj, obj.with_suffix(".usd"))


def fetch_objaverse(ident: str, part: str = "", bake: bool = False) -> Path:
    """``ident`` is "<folder>/<uid>" as in Objaverse's glbs/ tree.  ``part``
    "node:A,B,..." keeps only those glTF nodes (v3: a toy blender's base
    without its jar and lid).  They are picked before converting: the
    converter merges a GLB into one mesh, split only by material, so no node
    can be picked by prim path afterwards."""
    folder, uid = ident.split("/", 1)
    glb = _download(OBJAVERSE_GLB.format(folder=folder, uid=uid), RAW / "objaverse" / f"{uid}.glb")
    if part.startswith("node:"):
        glb = _keep_nodes(glb, [n.strip() for n in part[len("node:"):].split(",") if n.strip()], bake)
    elif part.startswith("crop:"):
        glb = _crop_below(glb, float(part[len("crop:"):]))
    return _convert(glb, glb.with_suffix(".usd"))


def _crop_below(glb: Path, share: float) -> Path:
    """A copy of ``glb`` beside it without the faces wholly in the lowest
    ``share`` of its height (the glTF scene's y): a slab or a patch of floor
    a scan took in with the thing -- DI4's teddy sat on a slate slab (v10).
    Each node keeps its place, material and textures."""
    out = glb.with_name(f"{glb.stem}.crop-{share:g}.glb")
    if out.exists():
        return out
    import trimesh

    scene = trimesh.load(glb, force="scene")
    low, high = scene.bounds[:, 1]
    cut = low + share * (high - low)
    kept = trimesh.Scene()
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        height = trimesh.transform_points(mesh.vertices, transform)[:, 1]
        keep = (height[mesh.faces] > cut).any(axis=1)
        if not keep.any():
            continue
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
        kept.add_geometry(mesh, node_name=node, geom_name=f"{node}_{geometry}", transform=transform)
    kept.export(out)
    return out


def _keep_nodes(glb: Path, nodes: List[str], bake: bool = False) -> Path:
    """A copy of ``glb`` beside it with only the named nodes -- each with its
    geometry, where the scene puts it and its material -- in the same units,
    so the same ``size`` scales a part as it does the whole.  The nodes are
    placed where the glTF scene puts them, which is y-up: a Sketchfab model
    comes out of the converter stood up by its wrapper nodes' turn, which a
    part loses, so its entry says ``rotate: [90, 0, 0]`` (toy_blender_base)."""
    if not nodes:
        raise ConfigError(f"{glb.name}: 'node:' names no node")
    safe = "-".join("".join(c if c.isalnum() or c in "_." else "_" for c in n) for n in nodes)
    out = glb.with_name(f"{glb.stem}.nodes-{safe}{'.baked' if bake else ''}.glb")
    if out.exists():
        return out
    import trimesh

    scene = trimesh.load(glb, force="scene")
    have = set(scene.graph.nodes_geometry)
    missing = [n for n in nodes if n not in have]
    if missing:
        raise ConfigError(f"{glb.name} has no geometry node {missing}; it has {sorted(have)}")
    kept = trimesh.Scene()
    for node in nodes:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        if bake:                                  # in place, the node itself at rest
            mesh.apply_transform(transform)
            kept.add_geometry(mesh, node_name=node, geom_name=f"{node}_{geometry}")
        else:
            kept.add_geometry(mesh, node_name=node, geom_name=f"{node}_{geometry}",
                              transform=transform)
    kept.export(out)
    return out


def _convert(src: Path, out: Path) -> Path:
    if out.exists():
        return out
    import asyncio

    from isaaclab.sim.converters.mesh_converter import MeshConverter

    asyncio.get_event_loop().run_until_complete(
        MeshConverter._convert_mesh_to_usd(str(src), str(out), load_materials=True))
    return out


def polyhaven_maps(src: str, which: Optional[str] = None) -> Dict[str, str]:
    """A Poly Haven model's texture maps, by role, next to its USD --
    ``which`` of its sets, if it has several (the chess set's board, white
    pieces and black pieces)."""
    folder = Path(src).parent / "textures"
    maps = {}
    for role, key in (("diffuse", "_diff_"), ("roughness", "_rough_"), ("normal", "_nor_gl_")):
        found = sorted(f for f in folder.glob(f"*{key}*") if not which or which in f.name)
        if found:
            maps[role] = str(found[0])
    return maps


def source_usd(entry) -> Tuple[str, str]:
    """(USD to reference, prim path inside it or "") for a catalogue entry."""
    kind, _, ident = entry.source.partition(":")
    part = entry.part or ""
    if kind == "polyhaven":
        return str(fetch_polyhaven(ident)), ("" if part.startswith("component:") else part)
    if kind == "gso":
        return str(fetch_gso(ident, part)), ""
    if kind == "objaverse":
        return (str(fetch_objaverse(ident, part, getattr(entry, "bake", False))),
                "" if part.startswith(("node:", "crop:")) else part)
    if kind == "ycb":
        return str(fetch_ycb(ident)), ""
    if kind == "nucleus":
        from .assets import nucleus_dir
        from .environment import nvidia_dir

        return ident.replace("{NUCLEUS}", nucleus_dir()).replace("{NVIDIA}", nvidia_dir()), part
    raise ConfigError(f"{entry.name}: unknown source kind {kind!r} "
                      "(polyhaven, gso, objaverse, ycb, nucleus)")


# -- the wrapper ---------------------------------------------------------------
def _euler_quat(degrees: Iterable[float]):
    """x, then y, then z (extrinsic), as (w, x, y, z)."""
    from ...core.geometry import quat_mul

    q = (1.0, 0.0, 0.0, 0.0)
    for axis, deg in zip(((1, 0, 0), (0, 1, 0), (0, 0, 1)), degrees):
        a = math.radians(deg) / 2
        q = quat_mul((math.cos(a), axis[0] * math.sin(a), axis[1] * math.sin(a), axis[2] * math.sin(a)), q)
    return q


def _bounds(stage, path: str):
    """The world-axis box around the meshes under ``path``, from their points.

    Not from their boxes: a box turned 41 degrees (a toy car lying askew on the
    scanning turntable) bounds the car loosely -- 8.2 cm square for a 5.9 x
    3.3 cm car."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom

    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputePurpose() not in ("default", "render"):
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points:
            continue
        m = np.array(xforms.GetLocalToWorldTransform(prim), dtype=float)   # row vectors
        world = np.c_[np.array(points, dtype=float), np.ones(len(points))] @ m
        lo, hi = np.minimum(lo, world[:, :3].min(0)), np.maximum(hi, world[:, :3].max(0))
    if not np.isfinite(lo).all():
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        return cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return Gf.Range3d(Gf.Vec3d(*lo), Gf.Vec3d(*hi))


def wrap(entry, src: str, prim: str = "") -> Path:
    """Write the entry's local USD: the source in metres, turned, sized, centred."""
    from pxr import Gf, Usd, UsdGeom

    out = PREPARED / entry.name / f"{entry.name}.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    source = Usd.Stage.Open(src)
    unit = UsdGeom.GetStageMetersPerUnit(source) if entry.units is None else entry.units
    stage = Usd.Stage.CreateNew(str(out)) if not out.exists() else Usd.Stage.Open(str(out))
    stage.GetRootLayer().Clear()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/root")
    stage.SetDefaultPrim(root.GetPrim())
    geo = UsdGeom.Xform.Define(stage, "/root/geometry")
    # the source on a child of its own: it may carry its own transform ops
    # (converted scans do), which must stay as they are under ours
    ref = UsdGeom.Xform.Define(stage, "/root/geometry/source").GetPrim()
    if prim:
        ref.GetReferences().AddReference(src, prim)
    else:
        ref.GetReferences().AddReference(src)
    translate, orient, scale = geo.AddTranslateOp(), geo.AddOrientOp(), geo.AddScaleOp()
    w, x, y, z = _euler_quat(entry.rotate)
    orient.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    scale.Set(Gf.Vec3f(unit, unit, unit))
    translate.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    box = _bounds(stage, "/root")
    size = box.GetSize()
    k = entry.scale if entry.scale else 1.0
    if entry.size:
        k = entry.size / max(size)
    scale.Set(Gf.Vec3f(unit * k, unit * k, unit * k))
    box = _bounds(stage, "/root")
    centre = (box.GetMin() + box.GetMax()) / 2
    translate.Set(Gf.Vec3d(-centre[0], -centre[1], -centre[2]))
    if entry.source.startswith("polyhaven:"):
        _own_material(stage, polyhaven_maps(src, entry.maps))
    if entry.plug:
        _plug(stage, entry.plug)
    stage.GetRootLayer().Save()
    return out


def _plug(stage, plug: Dict) -> None:
    """A 2 mm disc lying on the pot's inner bottom, over its drainage holes.

    Turned over an egg (DS2), a real plant pot showed the egg through them to
    a camera above; the disc closes them, in the clay's own colour.  It lies
    inside the pot's bounds, so the measured size is the scan's."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    thick = 0.002
    bottom = _bounds(stage, "/root").GetMin()[2]
    disc = UsdGeom.Cylinder.Define(stage, "/root/plug")
    disc.CreateRadiusAttr(float(plug["radius"]))
    disc.CreateHeightAttr(thick)
    disc.CreateAxisAttr("Z")
    UsdGeom.Xformable(disc).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, bottom + float(plug["above"]) + thick / 2))
    material = UsdShade.Material.Define(stage, "/root/Looks/plug")
    shader = UsdShade.Shader.Define(stage, "/root/Looks/plug/Shader")
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateIdAttr("OmniPBR")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*(float(c) for c in plug["color"])))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.8)
    for output in (material.CreateSurfaceOutput("mdl"), material.CreateVolumeOutput("mdl"),
                   material.CreateDisplacementOutput("mdl")):
        output.ConnectToSource(shader.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(disc.GetPrim()).Bind(material, UsdShade.Tokens.strongerThanDescendants)


def _own_material(stage, maps: Dict[str, str]) -> None:
    """OmniPBR from the source's own texture maps, bound over its material.

    Poly Haven's USD carries a UsdPreviewSurface network with a UsdTransform2d
    and a MaterialX one besides; Isaac 4.5's renderer drew the lemon plain red.
    The same maps on OmniPBR -- as the tabletop has them -- render right."""
    from pxr import Sdf, UsdShade

    if "diffuse" not in maps:
        return
    material = UsdShade.Material.Define(stage, "/root/Looks/scan")
    shader = UsdShade.Shader.Define(stage, "/root/Looks/scan/Shader")
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateIdAttr("OmniPBR")
    for role, name, space in (("diffuse", "diffuse_texture", "sRGB"),
                              ("roughness", "reflectionroughness_texture", "raw"),
                              ("normal", "normalmap_texture", "raw")):
        if role in maps:
            inp = shader.CreateInput(name, Sdf.ValueTypeNames.Asset)
            inp.Set(Sdf.AssetPath(maps[role]))
            inp.GetAttr().SetColorSpace(space)
    if "roughness" in maps:
        shader.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/root/geometry")).Bind(
        material, UsdShade.Tokens.strongerThanDescendants)


def measure(path: Path) -> dict:
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    box = _bounds(stage, "/root")
    lo, hi = box.GetMin(), box.GetMax()
    half = [round((hi[j] - lo[j]) / 2, 4) for j in range(3)]
    centre = [round((hi[j] + lo[j]) / 2, 4) for j in range(3)]
    narrow = 2 * min(half[0], half[1])
    return {"as_authored": {"half_extents": half, "centre_offset": centre},
            "upright": "as_authored", "narrow_width_m": round(narrow, 4),
            "graspable": narrow <= 0.075}


def wrapper_sources(wrapper: Path) -> List[Path]:
    """The files on this machine a prepared wrapper references -- the scan it
    wraps -- leaving out what Kit finds itself (MDL materials) and what is
    fetched over the network."""
    import re

    text = wrapper.read_text(errors="ignore")
    out = []
    for ref in re.findall(r"@([^@]+)@", text):
        if "://" in ref or ref.endswith(".mdl"):
            continue
        path = Path(ref)
        out.append(path if path.is_absolute() else (wrapper.parent / path).resolve())
    return out


def missing_sources(wrapper: Path) -> List[Path]:
    """What a prepared wrapper references and is not there.  A wrapper whose
    scan is gone still opens -- as an empty prim, which on Isaac is a body with
    no mesh and no collider: invisible, and through the table at once (CL2's
    doll2 from 9/19 to 9/24: one of the six dolls, and one of the look-alike
    pair, was never in the picture)."""
    return [p for p in wrapper_sources(wrapper) if not p.exists()]


def prepare(names: Optional[List[str]] = None, force: bool = False) -> Dict[str, dict]:
    from .assets import load_catalogue, record_geometry

    catalogue = load_catalogue()
    todo = [e for n, e in catalogue.items() if e.source and n == e.name and (not names or n in names)]
    measured = {}
    for entry in todo:
        out = PREPARED / entry.name / f"{entry.name}.usda"
        if out.exists() and not force and not missing_sources(out):
            measured[entry.name] = measure(out)
            continue
        src, prim = source_usd(entry)
        out = wrap(entry, src, prim)
        measured[entry.name] = measure(out)
        m = measured[entry.name]
        say(f"{entry.name:24s} {entry.source:48s} half={m['as_authored']['half_extents']} "
            f"narrow {m['narrow_width_m'] * 100:4.1f} cm")
    record_geometry(measured)
    return measured


def main(argv: List[str]) -> None:
    from . import app as isaac_app

    isaac_app.launch(headless=True)
    force = "--force" in argv
    names = [a for a in argv if not a.startswith("--")]
    prepare(names or None, force=force)


if __name__ == "__main__":
    import traceback

    code = 0
    try:
        main(sys.argv[1:])
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)              # a Kit app does not shut down cleanly otherwise
