"""Finishes as Isaac materials (``engines/finishes.py`` names them).

One material per finish per stage, under ``/World/Looks/finishes``: parts and
bodies that share a finish share its material.  Everything is OmniPBR --
photographed maps for wood and linen (Poly Haven, CC0, fetched and pinned by
``environment``), constants for paint, plastic and glaze, and the metallic
workflow for metals; clear glass is OmniGlass (with the RTX translucency pass
on), a liquid a glossy OmniPBR.
Textures are projected in world space
(``project_uvw``, ``world_or_object``): USD's cubes and cylinders carry no UVs,
and object space would stretch the grain with each part's scale.
"""

from __future__ import annotations

from typing import Optional

from ..finishes import finish_rgb, is_finish, linear, split

ROOT = "/World/Looks/finishes"

#: finish kind -> (download names for diffuse, normal, roughness; repeats per metre)
TEXTURED = {
    "wood:oak": (("oak_diffuse", "oak_normal", "oak_roughness"), 1.5),
    # the oak photograph, darkened: walnut (DI6's turntable, v8 -- a flat
    # brown until then); reddened: cherry (WK1's discs, v9); stained near
    # black: ebonised (WK1's stand, under the four woods of its rings, v9)
    "wood:walnut": (("oak_diffuse", "oak_normal", "oak_roughness"), 3.0),
    "wood:cherry": (("oak_diffuse", "oak_normal", "oak_roughness"), 3.0),
    "wood:ebony": (("oak_diffuse", "oak_normal", "oak_roughness"), 3.0),
    # the ash photograph, dyed: stained wood, its grain showing through (WK1's
    # rings, v10 -- natural woods were too plain, paint looked a cartoon)
    **{f"wood:stain_{c}": (("ash_diffuse", "ash_normal", "ash_roughness"), 3.0)
       for c in ("red", "yellow", "green", "blue")},
    "wood:ash": (("ash_diffuse", "ash_normal", "ash_roughness"), 3.0),
    "wood:plywood": (("plywood_diffuse", "plywood_normal", "plywood_roughness"), 2.0),
    "sand:beach": (("sand_diffuse", "sand_normal", "sand_roughness"), 3.0),
    # WK2's diorama (v10): damp sand at the water's edge, a dense lawn, and
    # the river's pebbled bed, its pebbles 1-2 cm (a Ganges bed at a third of a metre)
    "sand:wet": (("wetsand_diffuse", "wetsand_normal", "wetsand_roughness"), 2.0),
    "ground:grass": (("grass_diffuse", "grass_normal", "grass_roughness"), 1.0),
    "ground:pebbles": (("pebbles_diffuse", "pebbles_normal", "pebbles_roughness"), 3.0),
    # DI3's model street (v14): photographs of metres of road and pavement,
    # shrunk about ten times -- a 60 cm slab is a 6 cm one, as a model's are
    "ground:asphalt": (("asphalt_diffuse", "asphalt_normal", "asphalt_roughness"), 5.0),
    "ground:paving": (("paving_diffuse", "paving_normal", "paving_roughness"), 5.0),
    "ground:concrete": (("concrete_diffuse", "concrete_normal", "concrete_roughness"), 8.0),
}
#: a textured finish drawn darker than its photograph
TINTS = {"wood:walnut": (0.42, 0.30, 0.24), "wood:cherry": (0.78, 0.46, 0.34),
         "wood:ebony": (0.20, 0.15, 0.13),
         "wood:stain_red": (0.90, 0.16, 0.12), "wood:stain_yellow": (1.00, 0.76, 0.16),
         "wood:stain_green": (0.30, 0.66, 0.26), "wood:stain_blue": (0.16, 0.38, 0.82)}
#: linen for fabric, dyed by the finish's colour, a fine weave
LINEN = (("linen_diffuse", "linen_normal", "linen_roughness"), 8.0)
#: paint over wood: the colour, a little of the ash grain in the normal map
PAINT_GRAIN = ("ash_normal", 3.0, 0.25)
#: a flame's emission (nits-like units of OmniPBR): bright against the lab's light
FLAME_GLOW = 2500.0
#: a flame's (``fire:``) opacity: DS6's burner flame, a solid glowing
#: ellipsoid until v8, looked like a blue leaf
FIRE_OPACITY = 0.55
#: glass: clear glass is a thin wall (a vessel's side is a surface, not a
#: solid block).  A tinted "glass" is a liquid: glossy, its tint, and partly
#: see-through -- clear water, its surface still seen (research/
#: v8_water_probe.py).  Neither casts a shadow (``spawners.casts_no_shadow``):
#: the glass's shadow had drawn every liquid in it black
GLASS_IOR = 1.5
LIQUID_ROUGHNESS, LIQUID_OPACITY = 0.06, 0.4
SOLID_GLASS = (0.74, 0.90, 0.87)
#: a diorama's river: its water a clear slab, as resin poured over the bed
#: is -- a 40% opaque liquid drawn there was a pale grey sheet, the pebbles
#: under it lost (WK2, v10) -- a green-blue tint, and an index low enough
#: that the room's window does not lie on it as a white card (at water's 1.33
#: it did: a slab this flat is a mirror)
STREAM_GLASS, STREAM_IOR = (0.66, 0.86, 0.84), 1.1
#: bottle glass, pale green: a vessel whose wall and rim must read -- CL4's
#: jar, where the ball rising past the rim is the point, clear glass showed
#: two faint lines and no rim at all (v11)
BOTTLE_GLASS = (0.50, 0.78, 0.56)
ROUGHNESS = {"paint": 0.45, "plastic": 0.28, "ceramic": 0.12, "paper": 0.85}
WRAP_ROUGHNESS = 0.4
#: metal -> (reflectance, linear RGB; roughness).  Not NVIDIA's polished
#: Gold MDL: a small mirror shows the room around it, and DS2's golden egg
#: came out black with a few gold highlights.  Foil and brushed metal are
#: rough enough to show their colour
METALS = {"gold": ((1.0, 0.766, 0.336), 0.3), "brass": ((0.91, 0.778, 0.423), 0.32),
          "steel": ((0.56, 0.57, 0.58), 0.28), "aluminium": ((0.913, 0.922, 0.924), 0.35),
          "chrome": ((0.85, 0.86, 0.88), 0.12),
          # satin nickel: a desk bell's dome -- polished, it mirrored the dim
          # room and came out black (WK2, v8)
          "nickel": ((0.80, 0.79, 0.76), 0.26),
          "copper": ((0.955, 0.638, 0.538), 0.3), "black": ((0.03, 0.03, 0.035), 0.45)}


def _safe(finish: str) -> str:
    """A prim name for a finish: ASCII letters and digits -- a print's spec
    may hold any character ("label:CuCl₂"), and then a hash tells it apart."""
    import hashlib

    name = "".join(c if c.isascii() and c.isalnum() else "_" for c in finish)
    if not finish.isascii():
        name += "_" + hashlib.sha1(finish.encode()).hexdigest()[:8]
    return name


def material(finish: str, path: Optional[str] = None) -> str:
    """The material for ``finish`` on the current stage, made the first time
    -- at ``path`` for one body's own (a flame, which is recoloured alone)."""
    import isaacsim.core.utils.stage as stage_utils

    stage = stage_utils.get_current_stage()
    path = path or f"{ROOT}/{_safe(finish)}"
    if stage.GetPrimAtPath(path).IsValid():
        return path
    _omnipbr(stage, path, finish)
    return path


def _omnipbr(stage, path: str, finish: str) -> None:
    from pxr import Gf, Sdf, UsdShade

    from .environment import fetched

    kind, value = split(finish)
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    mdl = "OmniGlass" if finish in ("glass:clear", "glass:solid", "glass:stream", "glass:bottle") else "OmniPBR"
    shader.SetSourceAsset(f"{mdl}.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier(mdl, "mdl")
    shader.CreateIdAttr(mdl)

    def put(name, type_, value_):
        shader.CreateInput(name, type_).Set(value_)

    def texture(name, download, space):
        inp = shader.CreateInput(name, Sdf.ValueTypeNames.Asset)
        inp.Set(Sdf.AssetPath(fetched(download)))
        inp.GetAttr().SetColorSpace(space)

    def project(scale):
        put("project_uvw", Sdf.ValueTypeNames.Bool, True)
        put("world_or_object", Sdf.ValueTypeNames.Bool, True)
        put("texture_scale", Sdf.ValueTypeNames.Float2, Gf.Vec2f(scale, scale))

    rgb = Gf.Vec3f(*linear(finish_rgb(finish)))
    if kind == "metal":
        reflectance, roughness = METALS.get(value, METALS["steel"])
        put("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*reflectance))
        put("metallic_constant", Sdf.ValueTypeNames.Float, 1.0)
        put("reflection_roughness_constant", Sdf.ValueTypeNames.Float, roughness)
    elif finish in TEXTURED or kind == "fabric":
        (diffuse, normal, rough), scale = TEXTURED.get(finish, LINEN)
        texture("diffuse_texture", diffuse, "sRGB")
        texture("normalmap_texture", normal, "raw")
        texture("reflectionroughness_texture", rough, "raw")
        put("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float, 1.0)
        if kind == "fabric":
            # the linen photograph is near white: its weave, in the dye's colour
            put("diffuse_tint", Sdf.ValueTypeNames.Color3f, rgb)
        elif finish in TINTS:
            put("diffuse_tint", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*TINTS[finish]))
        project(scale)
    elif kind in ("flame", "fire"):
        # it glows: ``set_color`` writes the emission with the diffuse colour
        put("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, rgb)
        put("enable_emission", Sdf.ValueTypeNames.Bool, True)
        put("emissive_color", Sdf.ValueTypeNames.Color3f, rgb)
        put("emissive_intensity", Sdf.ValueTypeNames.Float, FLAME_GLOW)
        put("reflection_roughness_constant", Sdf.ValueTypeNames.Float, 0.9)
        if kind == "fire":
            # a flame is seen through, the brighter where it is thicker
            put("enable_opacity", Sdf.ValueTypeNames.Bool, True)
            put("opacity_constant", Sdf.ValueTypeNames.Float, FIRE_OPACITY)
            put("opacity_threshold", Sdf.ValueTypeNames.Float, 0.0)
    elif finish == "glass:clear":
        # glass that looks like glass: NVIDIA's OmniGlass, thin-walled, under
        # the RTX translucency pass (``environment.RTX_SETTINGS``).  Until v8
        # (9/24) an opaque pale OmniPBR stood in -- OmniGlass had drawn
        # nothing with translucency off -- and every test tube, beaker and
        # jar came out white (the advisor: "all the glass renders strangely")
        put("glass_color", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.98, 0.99, 1.0))
        put("glass_ior", Sdf.ValueTypeNames.Float, GLASS_IOR)
        put("thin_walled", Sdf.ValueTypeNames.Bool, True)
        put("frosting_roughness", Sdf.ValueTypeNames.Float, 0.0)
    elif finish == "glass:bottle":
        put("glass_color", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*BOTTLE_GLASS))
        put("glass_ior", Sdf.ValueTypeNames.Float, GLASS_IOR)
        put("thin_walled", Sdf.ValueTypeNames.Bool, True)
        put("frosting_roughness", Sdf.ValueTypeNames.Float, 0.0)
    elif finish == "glass:stream":
        put("glass_color", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*STREAM_GLASS))
        put("glass_ior", Sdf.ValueTypeNames.Float, STREAM_IOR)
        put("thin_walled", Sdf.ValueTypeNames.Bool, False)
        put("frosting_roughness", Sdf.ValueTypeNames.Float, 0.0)
    elif finish == "glass:solid":
        # a rod of glass, drawn as a volume: its edges refract and show, a
        # faint green-blue as soda glass has.  Thin-walled (``glass:clear``)
        # DS5's stirring rod all but vanished (v9)
        put("glass_color", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*SOLID_GLASS))
        put("glass_ior", Sdf.ValueTypeNames.Float, GLASS_IOR)
        put("thin_walled", Sdf.ValueTypeNames.Bool, False)
        put("frosting_roughness", Sdf.ValueTypeNames.Float, 0.0)
    elif kind == "glass":
        # a liquid: its tint, glossy, partly see-through
        put("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, rgb)
        put("reflection_roughness_constant", Sdf.ValueTypeNames.Float, LIQUID_ROUGHNESS)
        put("specular_level", Sdf.ValueTypeNames.Float, 0.8)
        put("enable_opacity", Sdf.ValueTypeNames.Bool, True)
        put("opacity_constant", Sdf.ValueTypeNames.Float, LIQUID_OPACITY)
        put("opacity_threshold", Sdf.ValueTypeNames.Float, 0.0)
    elif kind == "paper":
        # a printed image (``engines/prints.py``) in the face's own coordinates
        # (a print part is a quad with texture coordinates)
        from ..prints import image

        inp = shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
        inp.Set(Sdf.AssetPath(image(value)))
        inp.GetAttr().SetColorSpace("sRGB")
        # wrapping paper has a sheen; a card or a label does not
        put("reflection_roughness_constant", Sdf.ValueTypeNames.Float,
            WRAP_ROUGHNESS if value.startswith("wrap:") else ROUGHNESS["paper"])
    else:
        put("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, rgb)
        put("reflection_roughness_constant", Sdf.ValueTypeNames.Float, ROUGHNESS.get(kind, 0.5))
        if kind == "paint":
            normal, scale, strength = PAINT_GRAIN
            texture("normalmap_texture", normal, "raw")
            put("bump_factor", Sdf.ValueTypeNames.Float, strength)
            project(scale)
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(out)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(out)


def bind(prim_path: str, finish: str) -> None:
    from isaaclab.sim.utils import bind_visual_material

    bind_visual_material(prim_path, material(finish))


def finish_cfg_class():
    """A visual-material config Isaac Lab's shape spawners accept: they call
    ``func(material_path, cfg)`` and bind what it made."""
    import isaaclab.sim as sim_utils
    from isaaclab.utils import configclass

    def spawn(prim_path: str, cfg):
        import isaacsim.core.utils.stage as stage_utils

        stage = stage_utils.get_current_stage()
        # made once, shared: the prim Isaac Lab asks for becomes a reference to it
        shared = material(cfg.finish)
        prim = stage.DefinePrim(prim_path, "Material")
        prim.GetReferences().AddInternalReference(shared)
        return prim

    @configclass
    class FinishCfg(sim_utils.VisualMaterialCfg):
        func: object = spawn
        finish: str = "wood:oak"

    return FinishCfg


def visual_cfg(color, roughness: Optional[float] = None):
    """The visual material for a body's colour: its finish, or a flat colour."""
    import isaaclab.sim as sim_utils

    from ..base import color_rgb

    if is_finish(color):
        return _finish_cfg()(finish=color)
    kwargs = {} if roughness is None else {"roughness": roughness}
    return sim_utils.PreviewSurfaceCfg(diffuse_color=linear(color_rgb(color)), **kwargs)


_cfg_class = None


def _finish_cfg():
    global _cfg_class
    if _cfg_class is None:
        _cfg_class = finish_cfg_class()
    return _cfg_class


__all__ = ["material", "bind", "visual_cfg", "ROOT"]
