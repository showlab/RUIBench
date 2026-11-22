"""Spawning mesh assets as physical bodies.

Most of Isaac's YCB props -- the bowl, the mug, the foam brick -- ship as
visual-only USD: no rigid body, no collider.  Isaac Lab's USD spawner only
*modifies* physics properties that are already there, so such a prop spawns as
scenery: nothing can pick it up, and the rigid-object wrapper fails to
initialise.  This spawner loads the USD and then *applies* what is missing.

Collision shape matters more than it looks.  A convex hull of a bowl fills the
bowl in, so a ball placed "inside" rests on an invisible lid -- which would
silently break DI2.  Hollow things are therefore decomposed into convex parts.
"""

from __future__ import annotations

from functools import lru_cache


def spawn_usd_with_physics(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn ``cfg.usd_path`` and make every spawned copy a rigid body with colliders."""
    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.stage as stage_utils
    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from pxr import Usd, UsdGeom, UsdPhysics

    from ...core.errors import ConfigError

    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation)
    stage = stage_utils.get_current_stage()
    for path in sim_utils.find_matching_prim_paths(prim_path):
        root = stage.GetPrimAtPath(path)
        # instance proxies cannot be edited; this asset needs its own copy
        for p in Usd.PrimRange(root):
            if p.IsInstanceable():
                p.SetInstanceable(False)
        # a reference that did not resolve spawns an empty prim: no mesh, no
        # collider -- a body nobody sees, falling through the table (v13)
        if not any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(root)):
            raise ConfigError(f"{cfg.usd_path} put no mesh at {path}: its scan is missing or "
                              "empty; prepare the asset again")
        # Some "visual" assets carry a rigid body on a child prim (Isaac's
        # basic_block did).  Adding another on the root nests two rigid bodies, which PhysX
        # rejects -- so an asset with physics anywhere in it is left alone.
        if any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(root)):
            continue
        schemas.define_rigid_body_properties(path, cfg.rigid_props)
        if not root.HasAPI(UsdPhysics.MassAPI) and cfg.mass_props is not None:
            schemas.define_mass_properties(path, cfg.mass_props)
        for p in Usd.PrimRange(root):
            if not p.IsA(UsdGeom.Mesh) or p.HasAPI(UsdPhysics.CollisionAPI):
                continue
            UsdPhysics.CollisionAPI.Apply(p)
            mesh = UsdPhysics.MeshCollisionAPI.Apply(p)
            mesh.CreateApproximationAttr().Set(cfg.collision_approximation)
    return prim


@lru_cache(maxsize=1)
def physics_usd_cfg_class():
    """``UsdFileCfg`` whose spawner adds rigid-body and collider schemas.

    Built lazily: configclasses can only be defined after Isaac has started.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.utils import configclass

    @configclass
    class PhysicsUsdFileCfg(sim_utils.UsdFileCfg):
        func: object = spawn_usd_with_physics
        #: "convexHull" for solid things, "convexDecomposition" for hollow ones
        collision_approximation: str = "convexHull"

    return PhysicsUsdFileCfg


def bin_boxes(half, wall: float, inverted: bool = False):
    """The five boxes of an open bin (or, inverted, of a cup): name -> (centre, half)."""
    hx, hy, hz = half
    t = wall
    lid_z = hz - t / 2 if inverted else -hz + t / 2
    return {
        "floor": ((0.0, 0.0, lid_z), (hx, hy, t / 2)),
        "wall_px": ((hx - t / 2, 0.0, 0.0), (t / 2, hy, hz)),
        "wall_nx": ((-hx + t / 2, 0.0, 0.0), (t / 2, hy, hz)),
        "wall_py": ((0.0, hy - t / 2, 0.0), (hx, t / 2, hz)),
        "wall_ny": ((0.0, -hy + t / 2, 0.0), (hx, t / 2, hz)),
    }


def lid_boxes(half, plate: float, handle):
    """A lid's plate and the bar on top of it: name -> (centre, half)."""
    hx, hy, hz = half
    return {
        "plate": ((0.0, 0.0, -hz + plate / 2), (hx, hy, plate / 2)),
        "handle": ((0.0, 0.0, plate / 2), (handle[0], handle[1], hz - plate / 2)),
    }


def _spawn_open_bin(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """An open-topped bin built from five boxes: one body, five colliders --
    or, ``inverted``, an open-bottomed cup.

    Built rather than loaded.  Isaac's KLT bin ships a collision scope authored
    about a hundred times too large -- invisible slabs across the whole
    workspace that stopped the fingers short of every object -- and its mesh
    cannot be recoloured, while "put it in the red bin" needs a red bin.
    """
    import isaacsim.core.utils.prims as prim_utils
    import isaacsim.core.utils.stage as stage_utils
    from isaaclab.sim import schemas
    from isaaclab.sim.utils import bind_visual_material
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = stage_utils.get_current_stage()
    prim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    # upside down (a cup put over something): the fifth box is a roof
    parts = bin_boxes(cfg.half_extents, cfg.wall, cfg.inverted)
    material_path = f"{prim_path}/material"
    cfg.visual_material.func(material_path, cfg.visual_material)
    for name, (centre, half) in parts.items():
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        cube.CreateSizeAttr(2.0)
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*centre))
        xform.AddScaleOp().Set(Gf.Vec3f(*half))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        bind_visual_material(f"{prim_path}/{name}", material_path)
    schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    schemas.define_mass_properties(prim_path, cfg.mass_props)
    return stage.GetPrimAtPath(prim_path)


def _spawn_lid(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """A flat lid with a handle bar on top: one body, two colliders.  The
    body's origin is the centre of the pair's bounding box."""
    import isaacsim.core.utils.prims as prim_utils
    import isaacsim.core.utils.stage as stage_utils
    from isaaclab.sim import schemas
    from isaaclab.sim.utils import bind_visual_material
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = stage_utils.get_current_stage()
    prim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    parts = lid_boxes(cfg.half_extents, cfg.plate, cfg.handle)
    material_path = f"{prim_path}/material"
    cfg.visual_material.func(material_path, cfg.visual_material)
    for name, (centre, half) in parts.items():
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        cube.CreateSizeAttr(2.0)
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*centre))
        xform.AddScaleOp().Set(Gf.Vec3f(*half))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        bind_visual_material(f"{prim_path}/{name}", material_path)
    schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    schemas.define_mass_properties(prim_path, cfg.mass_props)
    return stage.GetPrimAtPath(prim_path)


def _spawn_compound(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """A toy built from parts (``engines/parts.py``): one rigid body, one
    collider per part (unless a part is decoration), one material per colour."""
    import isaacsim.core.utils.prims as prim_utils
    import isaacsim.core.utils.stage as stage_utils
    from isaaclab.sim import schemas

    stage = stage_utils.get_current_stage()
    prim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    add_parts(prim_path, cfg.parts, cfg.roughness, cfg.friction, solid=cfg.solid)
    schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    schemas.define_mass_properties(prim_path, cfg.mass_props)
    return stage.GetPrimAtPath(prim_path)


def add_parts(prim_path: str, parts, roughness: float = 0.55, friction=None,
              solid: bool = True) -> None:
    """Geometry for ``engines/parts.py`` parts under ``prim_path``: one prim per
    part, a collider unless it is decoration, one material per colour; a
    ``hidden`` collider is not drawn.  ``friction``: the body's own (its
    ``attrs["friction"]``), on every collider, instead of the stage's 0.5:
    between two bodies the smaller counts."""
    import isaacsim.core.utils.stage as stage_utils
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_visual_material
    from pxr import Gf, UsdGeom, UsdPhysics

    from ..base import color_rgb
    from ..finishes import is_finish, linear
    from .finishes import material as finish_material

    stage = stage_utils.get_current_stage()
    materials = {}
    for i, (kind, centre, dims, rot, color, collides) in enumerate(parts):
        if kind == "asset":
            _add_visual(stage, f"{prim_path}/part{i}", centre, rot, color)
            continue
        hidden = collides == "hidden"
        if not solid:                  # a riding mark: drawn only, whatever its parts say
            if hidden:
                continue
            collides = False
        # a finish ("paint:red", "wood:ash") is one shared material per stage;
        # a plain colour, one flat material per colour under this body
        rgb = color if is_finish(color) else tuple(round(v, 4) for v in color_rgb(color))
        if rgb not in materials and not hidden:
            if is_finish(color) and color.startswith(("flame:", "fire:")):
                materials[rgb] = finish_material(color, f"{prim_path}/material{len(materials)}")
            elif is_finish(color):
                materials[rgb] = finish_material(color)
            else:
                path = f"{prim_path}/material{len(materials)}"
                # the palette is sRGB, as it should look; the renderer is linear
                flat = sim_utils.PreviewSurfaceCfg(diffuse_color=linear(rgb), roughness=roughness)
                flat.func(path, flat)
                materials[rgb] = path
        path = f"{prim_path}/part{i}"
        look = None                    # what is drawn, when the collider is not
        if kind == "box":
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(2.0)
            scale = Gf.Vec3f(*dims)
        elif kind == "cylinder":
            # round parts are drawn as smooth meshes (``parts.SIDES``); a
            # collider is USD's own shape, exact for PhysX, and not drawn
            round_dims = (float(dims[0]), float(dims[0]), float(dims[1]))
            if collides:
                geom = UsdGeom.Cylinder.Define(stage, path)
                geom.CreateRadiusAttr(float(dims[0]))
                geom.CreateHeightAttr(2.0 * float(dims[1]))
                geom.CreateAxisAttr("Z")
                if not hidden:
                    look = lathe_mesh(stage, f"{path}_look", "lathe:cylinder", round_dims)
            else:
                geom = lathe_mesh(stage, path, "lathe:cylinder", round_dims)
            scale = None
        elif kind.startswith("lathe:"):
            geom = lathe_mesh(stage, path, kind, dims)
            scale = None
        elif kind == "print":
            geom = print_mesh(stage, path, dims)
            scale = None
        elif kind == "octagon":
            geom = octagon_mesh(stage, path, dims)
            scale = None
        elif kind == "sphere":
            r = float(dims[0])
            if collides:
                geom = UsdGeom.Sphere.Define(stage, path)
                geom.CreateRadiusAttr(r)
                if not hidden:
                    look = lathe_mesh(stage, f"{path}_look", "lathe:sphere", (r, r, r))
            else:
                geom = lathe_mesh(stage, path, "lathe:sphere", (r, r, r))
            scale = None
        elif kind == "capsule":
            r, h = float(dims[0]), float(dims[1])
            if collides:
                geom = UsdGeom.Capsule.Define(stage, path)
                geom.CreateRadiusAttr(r)
                geom.CreateHeightAttr(2.0 * h)
                geom.CreateAxisAttr("Z")
                if not hidden:
                    look = lathe_mesh(stage, f"{path}_look", "lathe:capsule", (r, r, h))
            else:
                geom = lathe_mesh(stage, path, "lathe:capsule", (r, r, h))
            scale = None
        elif kind == "ellipsoid":
            # a unit sphere, stretched; never a collider (``parts.ellipsoid``)
            geom = lathe_mesh(stage, path, "lathe:sphere", (1.0, 1.0, 1.0))
            scale = Gf.Vec3f(*dims)
        elif kind == "prism":
            # six corners, two triangular ends and three sides, wound outward
            hx, hy, hz = (float(v) for v in dims)
            geom = UsdGeom.Mesh.Define(stage, path)
            geom.CreatePointsAttr([Gf.Vec3f(-hx, -hy, -hz), Gf.Vec3f(hx, -hy, -hz), Gf.Vec3f(0, -hy, hz),
                                   Gf.Vec3f(-hx, hy, -hz), Gf.Vec3f(hx, hy, -hz), Gf.Vec3f(0, hy, hz)])
            geom.CreateFaceVertexCountsAttr([3, 3, 4, 4, 4])
            geom.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 5, 4, 0, 3, 4, 1, 1, 4, 5, 2, 0, 2, 5, 3])
            geom.CreateSubdivisionSchemeAttr("none")
            scale = None
        else:
            raise ValueError(f"unknown part kind {kind!r}")
        for shape in (geom, look) if look is not None else (geom,):
            xform = UsdGeom.Xformable(shape)
            xform.AddTranslateOp().Set(Gf.Vec3d(*centre))
            xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Quatd(float(rot[0]), Gf.Vec3d(*rot[1:])))
            if scale is not None:
                xform.AddScaleOp().Set(scale)
        if collides:
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            if kind in ("prism", "octagon"):
                UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexHull")
        if hidden or look is not None:
            UsdGeom.Imageable(geom).MakeInvisible()
        if not hidden:
            drawn = look if look is not None else geom
            bind_visual_material(str(drawn.GetPath()), materials[rgb])
            if isinstance(color, str) and color.startswith(("glass:", "fire:")):
                casts_no_shadow(drawn.GetPrim())
    if friction is not None:
        # the smaller of this and what it touches -- a slick thing is slick on
        # anything; the fingers' rubber still takes the larger
        own = f"{prim_path}/physics_material"
        sim_utils.spawn_rigid_body_material(own, sim_utils.RigidBodyMaterialCfg(
            static_friction=float(friction), dynamic_friction=float(friction), restitution=0.0,
            friction_combine_mode="min"))
        sim_utils.bind_physics_material(prim_path, own)


def lathe_mesh(stage, path: str, kind: str, dims):
    """A ``lathe:<profile>`` part (``engines/parts.py``) as a USD mesh, its
    normals authored: smooth round the axis and along the profile.  A sheet
    (a vessel's wall) is drawn from both sides."""
    from pxr import Gf, UsdGeom, Vt

    from ..parts import is_sheet, lathe_profile, lathe_sides, revolve, slab_mesh

    if kind == "lathe:slab":                  # a board: its edge carried round a rectangle
        points, normals, counts, indices = slab_mesh(dims)
        sheet = False
        lo, hi = tuple(-float(v) for v in dims), tuple(float(v) for v in dims)
    else:
        profile = lathe_profile(kind, dims)
        points, normals, counts, indices = revolve(profile, lathe_sides(float(dims[0])))
        sheet = is_sheet(profile)
        r = max(abs(p[0]) for p in profile)
        zs = [p[1] for p in profile]
        lo, hi = (-r, -r, min(zs)), (r, r, max(zs))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(sheet)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
    return mesh


def print_mesh(stage, path: str, dims):
    """A ``print`` part: its face facing +z, its image's top towards +x and
    its right towards -y, flat or wrapped (``parts.printed``)."""
    from pxr import Gf, Sdf, UsdGeom, Vt

    from ..parts import print_sheet

    points, normals, st, counts, indices = print_sheet(dims)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                 UsdGeom.Tokens.vertex)
    uv.Set(Vt.Vec2fArray([Gf.Vec2f(*t) for t in st]))
    lo = [min(p[i] for p in points) for i in range(3)]
    hi = [max(p[i] for p in points) for i in range(3)]
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
    return mesh


def octagon_mesh(stage, path: str, dims):
    """An ``octagon`` part: an eight-sided prism, flat-shaded (a collider,
    mostly, hidden under a round look)."""
    from pxr import Gf, UsdGeom, Vt

    from ..parts import octagon_points

    ring = octagon_points(float(dims[0]))
    hz = float(dims[2])
    points = [Gf.Vec3f(x, y, z) for z in (-hz, hz) for x, y in ring]
    counts, indices = [], []
    for k in range(8):
        j = (k + 1) % 8
        counts.append(4)
        indices += [k, j, 8 + j, 8 + k]
    counts += [8, 8]
    indices += list(range(7, -1, -1)) + list(range(8, 16))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    r = float(dims[0]) / 0.9238795
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-r, -r, -hz), Gf.Vec3f(r, r, hz)]))
    return mesh


def casts_no_shadow(prim) -> None:
    """Glass and the liquids in it cast no shadow.  The glass's hid what
    stood in it in the dark: the water in DS5's beakers came out black, and
    lit once the glass cast none (v8, research/v8_beaker_probe.py)."""
    from pxr import Sdf, UsdGeom

    UsdGeom.PrimvarsAPI(prim).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)


def _add_visual(stage, path: str, centre, rot, spec: str) -> None:
    """A catalogue mesh as a toy's look (``parts.visual``): referenced, placed,
    never colliding -- the hidden parts do that."""
    from pxr import Gf, UsdGeom

    from .assets import get

    name, _, scale = spec.partition("@")
    xform = UsdGeom.Xform.Define(stage, path)
    xform.GetPrim().GetReferences().AddReference(get(name).resolve())
    ops = UsdGeom.Xformable(xform)
    ops.ClearXformOpOrder()
    ops.AddTranslateOp().Set(Gf.Vec3d(*centre))
    ops.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(rot[0]), Gf.Vec3d(*rot[1:])))
    s = float(scale or 1.0)
    ops.AddScaleOp().Set(Gf.Vec3f(s, s, s))


@lru_cache(maxsize=1)
def compound_cfg_class():
    """Spawner config for :func:`_spawn_compound`."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    @configclass
    class CompoundCfg(sim_utils.RigidObjectSpawnerCfg):
        func: object = clone(_spawn_compound)
        parts: tuple = ()
        roughness: float = 0.55
        #: the body's friction, or None for the stage's
        friction: object = None
        #: False: no part collides, whatever its own flag -- a mark riding on
        #: another body (``base.rides``)
        solid: bool = True

    return CompoundCfg


@lru_cache(maxsize=1)
def lid_cfg_class():
    """Spawner config for :func:`_spawn_lid`."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    @configclass
    class LidCfg(sim_utils.RigidObjectSpawnerCfg):
        func: object = clone(_spawn_lid)
        half_extents: tuple = (0.05, 0.08, 0.019)
        plate: float = 0.008
        handle: tuple = (0.03, 0.012)
        visual_material: object = None

    return LidCfg


@lru_cache(maxsize=1)
def open_bin_cfg_class():
    """Spawner config for :func:`_spawn_open_bin` (built after Isaac starts)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    @configclass
    class OpenBinCfg(sim_utils.RigidObjectSpawnerCfg):
        func: object = clone(_spawn_open_bin)
        half_extents: tuple = (0.10, 0.15, 0.07)
        wall: float = 0.01
        inverted: bool = False
        visual_material: object = None

    return OpenBinCfg


def _spawn_slide(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """A one-joint articulation: a base pinned to the world and a link that
    slides along one axis -- or turns about it (``cfg.kind == "hinge"``) --
    built from boxes and cylinders.

        <prim_path>             articulation root, at the joint's zero
          base                  a rigid body with no shape, fixed to the world
          link                  the carried body: its shapes, one material
          <joint>               prismatic (or revolute about ``cfg.pivot``),
                                base -> link, with a linear (angular) drive
    """
    import math

    import isaacsim.core.utils.prims as prim_utils
    import isaacsim.core.utils.stage as stage_utils
    from isaaclab.sim.utils import bind_visual_material
    from pxr import Gf, UsdGeom, UsdPhysics

    try:
        import Semantics
    except ImportError:                       # Isaac Sim < 4.2
        from pxr import Semantics

    stage = stage_utils.get_current_stage()
    prim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    base, link = f"{prim_path}/base", f"{prim_path}/link"
    for path in (base, link):
        prim_utils.create_prim(path, "Xform")
    base_prim, link_prim = stage.GetPrimAtPath(base), stage.GetPrimAtPath(link)

    UsdPhysics.RigidBodyAPI.Apply(base_prim)
    base_mass = UsdPhysics.MassAPI.Apply(base_prim)
    base_mass.CreateMassAttr(1.0)
    base_mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-3, 1e-3, 1e-3))
    pin = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/root_joint")
    pin.CreateBody1Rel().SetTargets([base_prim.GetPath()])
    if getattr(cfg, "mount", ""):
        # fixed to what it stands on, where it stands on it now -- unauthored,
        # the joint's frame would be the other body's origin
        pin.CreateBody0Rel().SetTargets([cfg.mount])
        pin.CreateLocalPos0Attr(Gf.Vec3f(*cfg.mount_pos))
        pin.CreateLocalRot0Attr(Gf.Quatf(float(cfg.mount_rot[0]), Gf.Vec3f(*cfg.mount_rot[1:])))
        # a joint of its own, not a link in the other articulation's tree:
        # included, PhysX merged the two and neither could be found by name
        pin.CreateExcludeFromArticulationAttr(True)

    if cfg.parts and len(cfg.parts[0]) == 6:
        add_parts(link, cfg.parts, friction=cfg.friction)   # a toy built from parts
    material = f"{link}/material"
    if not (cfg.parts and len(cfg.parts[0]) == 6):
        cfg.visual_material.func(material, cfg.visual_material)
    for i, part in enumerate(() if cfg.parts and len(cfg.parts[0]) == 6 else cfg.parts):
        kind, centre = part[0], part[1]
        path = f"{link}/part{i}"
        if kind == "cylinder":
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateRadiusAttr(float(part[2]))
            geom.CreateHeightAttr(2.0 * float(part[3]))
            geom.CreateAxisAttr("Z")
            UsdGeom.Xformable(geom).AddTranslateOp().Set(Gf.Vec3d(*centre))
        else:
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(2.0)
            xform = UsdGeom.Xformable(geom)
            xform.AddTranslateOp().Set(Gf.Vec3d(*centre))
            xform.AddScaleOp().Set(Gf.Vec3f(*part[2]))
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        bind_visual_material(path, material)
    UsdPhysics.RigidBodyAPI.Apply(link_prim)
    UsdPhysics.MassAPI.Apply(link_prim).CreateMassAttr(float(cfg.mass))
    sem = Semantics.SemanticsAPI.Apply(link_prim, "class_body")
    sem.CreateSemanticTypeAttr().Set("class")
    sem.CreateSemanticDataAttr().Set(cfg.label)

    if cfg.kind == "hinge":
        # USD gives a revolute joint's limits in degrees; the pivot is the
        # same point in the base's frame and the link's (both sit at the root)
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/{cfg.joint_name}")
        joint.CreateLowerLimitAttr(math.degrees(float(cfg.limits[0])))
        joint.CreateUpperLimitAttr(math.degrees(float(cfg.limits[1])))
        joint.CreateLocalPos0Attr(Gf.Vec3f(*cfg.pivot))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*cfg.pivot))
        drive = "angular"
    else:
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/{cfg.joint_name}")
        joint.CreateLowerLimitAttr(float(cfg.limits[0]))
        joint.CreateUpperLimitAttr(float(cfg.limits[1]))
        drive = "linear"
    joint.CreateAxisAttr(cfg.axis)
    joint.CreateBody0Rel().SetTargets([base_prim.GetPath()])
    joint.CreateBody1Rel().SetTargets([link_prim.GetPath()])
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), drive)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(prim_path))
    return stage.GetPrimAtPath(prim_path)


@lru_cache(maxsize=1)
def slide_cfg_class():
    """Spawner config for :func:`_spawn_slide`."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    @configclass
    class SlideCfg(sim_utils.SpawnerCfg):
        func: object = clone(_spawn_slide)
        #: ("box", centre, half extents) | ("cylinder", centre, radius, half height),
        #: or the six-part tuples of ``engines/parts.py``
        parts: tuple = ()
        joint_name: str = "slide"
        kind: str = "slide"
        axis: str = "X"
        limits: tuple = (0.0, 0.3)
        pivot: tuple = (0.0, 0.0, 0.0)
        mass: float = 0.2
        label: str = ""
        visual_material: object = None
        #: the carried body's friction, or None for the stage's (parts bodies)
        friction: object = None
        #: the body the base is fixed to instead of the world (a mounted
        #: joint): its prim, and where the base sits in its frame
        mount: str = ""
        mount_pos: tuple = (0.0, 0.0, 0.0)
        mount_rot: tuple = (1.0, 0.0, 0.0, 0.0)

    return SlideCfg
