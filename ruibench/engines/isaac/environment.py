"""What surrounds the table: the room, its light, the table itself.

Since v2.  The frontier models
under test see nothing but our pictures, so the surroundings must read as a
real place -- v1's white sky, debug grid floor and flat-coloured slab did not.
The arrangement is RoboDojo's (its code, ``env_cfg/scene/default.yml``) with
assets we may use -- RoboDojo's own are non-commercial:

* **room** -- Isaac's ``Simple_Room`` at half scale around the table, with its
  own low table and dome light hidden and its ceiling and ceiling light kept:
  behind the table the camera sees a plain, light wall, not a furnished room;
* **light** -- a photo-studio HDRI (Poly Haven ``brown_photostudio_02``, CC0)
  on the dome for soft, even light, plus the room's ceiling light.  No point
  light: one over the table left a bright patch on it;
* **table** -- an oak veneer top (Poly Haven ``oak_veneer_01`` photographs,
  CC0) on OmniPBR, projected in world space so the slab's scale does not
  stretch the grain, on four aluminium legs.  NVIDIA's maple looked blotchy,
  walnut stripy with a band of reflected light, birch too white for white
  plates (research/realism_probe.py, runs/realism/).

Files not on the Isaac asset server are fetched once into ``third_party/`` and
checked against their SHA-256.
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from ...core.errors import ConfigError
from .robots import THIRD_PARTY

#: Where fetched files live (git-ignored, like the robot descriptions).
CACHE = THIRD_PARTY / "assets_cache"
#: Poly Haven refuses Python's default user agent
USER_AGENT = "RUIBench-asset-prep/1.0 (research benchmark)"


@dataclass(frozen=True)
class Download:
    """A file fetched once and pinned by hash."""

    path: str                 # under CACHE
    url: str
    sha256: str
    licence: str
    #: the file inside the zip at ``url`` (ambientCG ships its sets zipped);
    #: ``sha256`` is the file's
    member: str = ""


_POLY = "https://dl.polyhaven.org/file/ph-assets"
#: google/fonts, pinned to a commit
_FONTS = "https://raw.githubusercontent.com/google/fonts/7380f7cebbf2250460c23fd301b51bcbab162edf/ofl"
DOWNLOADS: Dict[str, Download] = {
    # the script hand of the gift tags (``engines/prints.py``)
    "script_font": Download(
        "fonts/DancingScript-wght.ttf", f"{_FONTS}/dancingscript/DancingScript%5Bwght%5D.ttf",
        "21808625578fe8d8cd10cb684be546dca077b27cd03a53a2f1ec11dc743c924c",
        "SIL OFL 1.1 (The Dancing Script Project Authors)"),
    "studio_hdri": Download(
        "skies/brown_photostudio_02_4k.hdr",
        f"{_POLY}/HDRIs/hdr/4k/brown_photostudio_02_4k.hdr",
        "368251c254a47dc7aaf532a515c0bd9ad3708ce5a3c5bb520b1628bcb9a37bc0", "CC0 (Poly Haven)"),
    "oak_diffuse": Download(
        "textures/oak_veneer_01_diff_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/oak_veneer_01/oak_veneer_01_diff_2k.jpg",
        "0718c2e1a209afe6138ccb41d70e1a18cd825d1e73db33ad7cf80806fa9ce5a4", "CC0 (Poly Haven)"),
    "oak_normal": Download(
        "textures/oak_veneer_01_nor_gl_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/oak_veneer_01/oak_veneer_01_nor_gl_2k.jpg",
        "e74a9d3ddcd86473a21815a28464f4dac4ceb8f86e7f170226dede8ed70d8eb9", "CC0 (Poly Haven)"),
    "oak_roughness": Download(
        "textures/oak_veneer_01_rough_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/oak_veneer_01/oak_veneer_01_rough_2k.jpg",
        "5c81718ee172299c44e0754fadbe652758b6feed05a8fd88be6efff5aa70ca1e", "CC0 (Poly Haven)"),
    "ash_diffuse": Download(
        "textures/ash_veneer_diff_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/ash_veneer/ash_veneer_diff_2k.jpg",
        "00abb70f86bdfe49aa6e09f350af9e8e1605ccf13171f9be46ceb6018846c06b", "CC0 (Poly Haven)"),
    "ash_normal": Download(
        "textures/ash_veneer_nor_gl_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/ash_veneer/ash_veneer_nor_gl_2k.jpg",
        "64bbd775ba2978aa12108ecd58419c333eb0c194b52cbc8e1d2bea02252020af", "CC0 (Poly Haven)"),
    "ash_roughness": Download(
        "textures/ash_veneer_rough_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/ash_veneer/ash_veneer_rough_2k.jpg",
        "3e6b077f6af6ba7725668b438bfc523f1342d0fb4bd497f1c9668663ad189ef0", "CC0 (Poly Haven)"),
    "plywood_diffuse": Download(
        "textures/plywood_diff_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/plywood/plywood_diff_2k.jpg",
        "050a15a796413b91c9cd5ca026dd3c4df83746bb1103ae14ae91503577dead8c", "CC0 (Poly Haven)"),
    "plywood_normal": Download(
        "textures/plywood_nor_gl_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/plywood/plywood_nor_gl_2k.jpg",
        "2391dbe49aaecd6584d12731d4470bf91ba33b5c7f05c78ec5900fcae28f0310", "CC0 (Poly Haven)"),
    "plywood_roughness": Download(
        "textures/plywood_rough_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/plywood/plywood_rough_2k.jpg",
        "bfd6947bca45a306f8c29add227c5eb9c7903d0cf1e72ce114483eb75305aa74", "CC0 (Poly Haven)"),
    "linen_diffuse": Download(
        "textures/rough_linen_diff_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/rough_linen/rough_linen_diff_2k.jpg",
        "e71016e08dfebea1dd4e8a2a675e3b617a889a782af782f1abb795906f4db1b4", "CC0 (Poly Haven)"),
    "linen_normal": Download(
        "textures/rough_linen_nor_gl_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/rough_linen/rough_linen_nor_gl_2k.jpg",
        "c5733ad50cfe0f90c546f3d4076590c1c78a4fd1a83c967cc9c92dc8175b99e6", "CC0 (Poly Haven)"),
    "linen_roughness": Download(
        "textures/rough_linen_rough_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/rough_linen/rough_linen_rough_2k.jpg",
        "48400d48fa203882bf5d984d5beb2f95edb1550309924601c018f2c4f3f80731", "CC0 (Poly Haven)"),
    # DS3's sandbox (v6): clean beige beach sand
    "sand_diffuse": Download(
        "textures/dense_sand_diff_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/dense_sand/dense_sand_diff_2k.jpg",
        "b2e3f157c315b9645d23e55d54b4c2c8fec39e5114001cbbc9a797ed27a06980", "CC0 (Poly Haven)"),
    "sand_normal": Download(
        "textures/dense_sand_nor_gl_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/dense_sand/dense_sand_nor_gl_2k.jpg",
        "22a97f0fb874ab6976c7ad566c09e776d4c191b074ed5fa258cdb29a2e83a7c6", "CC0 (Poly Haven)"),
    "sand_roughness": Download(
        "textures/dense_sand_rough_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/dense_sand/dense_sand_rough_2k.jpg",
        "e8ea428bb774ac990e7f656f18ff1ddb012ecc147b4b479dfd04491654e202c7", "CC0 (Poly Haven)"),
    # WK2's diorama (v10): a dense lawn (ambientCG, zipped), a river's pebbled
    # bed and its damp sand banks (Poly Haven)
    "grass_diffuse": Download(
        "textures/Grass004_2K-JPG_Color.jpg", "https://ambientcg.com/get?file=Grass004_2K-JPG.zip",
        "735b141c8f1972e70a45879424f1238fea9f887324173097785f4ac36e67e9a3", "CC0 (ambientCG)", member="Grass004_2K-JPG_Color.jpg"),
    "grass_normal": Download(
        "textures/Grass004_2K-JPG_NormalGL.jpg", "https://ambientcg.com/get?file=Grass004_2K-JPG.zip",
        "15ccad231764049d6581a52076d02bb6b627df77895cb0508e4017d8a4877fef", "CC0 (ambientCG)", member="Grass004_2K-JPG_NormalGL.jpg"),
    "grass_roughness": Download(
        "textures/Grass004_2K-JPG_Roughness.jpg", "https://ambientcg.com/get?file=Grass004_2K-JPG.zip",
        "c449fdf1e779eb7be5a19c1d58ee28cc155ed0523a141b151f3cbc843c4ae2e3", "CC0 (ambientCG)", member="Grass004_2K-JPG_Roughness.jpg"),
    "pebbles_diffuse": Download(
        "textures/ganges_river_pebbles_diff_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/ganges_river_pebbles/ganges_river_pebbles_diff_2k.jpg",
        "4284c19b6a546c8536a2d5383e5dcdbbae83ab6c2c337c7a0fb25284b0bea419", "CC0 (Poly Haven)"),
    "pebbles_normal": Download(
        "textures/ganges_river_pebbles_nor_gl_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/ganges_river_pebbles/ganges_river_pebbles_nor_gl_2k.jpg",
        "c73e2ea059f50f25d7f8dd719048d26bc0ca355d3009b0ef806898531231d03b", "CC0 (Poly Haven)"),
    "pebbles_roughness": Download(
        "textures/ganges_river_pebbles_rough_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/ganges_river_pebbles/ganges_river_pebbles_rough_2k.jpg",
        "b8ba7e5c7ed6f9c52d537a945d044b0d9e547459087c62829cef56d7fac11fce", "CC0 (Poly Haven)"),
    "wetsand_diffuse": Download(
        "textures/damp_sand_diff_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/damp_sand/damp_sand_diff_2k.jpg",
        "31225a161db73c919a2e6bb780d0e9b16df77a8f4d13cfb9d0ed815093b43483", "CC0 (Poly Haven)"),
    "wetsand_normal": Download(
        "textures/damp_sand_nor_gl_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/damp_sand/damp_sand_nor_gl_2k.jpg",
        "cac8c7c14656a7acafd33ea980ff45b3956adac20420a26429a8b51df9922727", "CC0 (Poly Haven)"),
    "wetsand_roughness": Download(
        "textures/damp_sand_rough_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/damp_sand/damp_sand_rough_2k.jpg",
        "7dabfbd6f9343efe4c8448a3795b4dda889c08ab90f6a1a4f989c397c1e686e6", "CC0 (Poly Haven)"),
    # DI3's model street (v14): a fine asphalt road, a pavement of concrete
    # slabs, and kerbstones of plain concrete (Poly Haven)
    "asphalt_diffuse": Download(
        "textures/asphalt_pit_lane_diff_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/asphalt_pit_lane/asphalt_pit_lane_diff_2k.jpg",
        "35f41f618a4342454b9a46d136b5086e8f1872a866aa1a7db2f813be3c6059d3", "CC0 (Poly Haven)"),
    "asphalt_normal": Download(
        "textures/asphalt_pit_lane_nor_gl_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/asphalt_pit_lane/asphalt_pit_lane_nor_gl_2k.jpg",
        "ff7aea390e34d05cb41afb93b90f043711736c252e9c5766e412efe756bf3660", "CC0 (Poly Haven)"),
    "asphalt_roughness": Download(
        "textures/asphalt_pit_lane_rough_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/asphalt_pit_lane/asphalt_pit_lane_rough_2k.jpg",
        "cf54b6ad76bd641cc5e9fc2fa0378899839c1e2a95630f844e6ed825510d0644", "CC0 (Poly Haven)"),
    "paving_diffuse": Download(
        "textures/concrete_pavement_diff_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/concrete_pavement/concrete_pavement_diff_2k.jpg",
        "e5a1cecfe7765cc0b24a6ab0abf492bd21c9e7af64e97fc46ef6021d0645fbd3", "CC0 (Poly Haven)"),
    "paving_normal": Download(
        "textures/concrete_pavement_nor_gl_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/concrete_pavement/concrete_pavement_nor_gl_2k.jpg",
        "72def937e418ccc9d7736c9b1f51ae22a127fa2a53a54f4d1fe4d3cf828b781c", "CC0 (Poly Haven)"),
    "paving_roughness": Download(
        "textures/concrete_pavement_rough_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/concrete_pavement/concrete_pavement_rough_2k.jpg",
        "e97b7d190dcda5a842813502505bf3d1b2348b4cad57b1d51d1bcc694a92d137", "CC0 (Poly Haven)"),
    "concrete_diffuse": Download(
        "textures/concrete_floor_01_diff_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/concrete_floor_01/concrete_floor_01_diff_2k.jpg",
        "ca24c52dabae67ef49093f91d8f9b9e5328a3b9fcb1327742ff6181ee542e017", "CC0 (Poly Haven)"),
    "concrete_normal": Download(
        "textures/concrete_floor_01_nor_gl_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/concrete_floor_01/concrete_floor_01_nor_gl_2k.jpg",
        "796b010208e346bbeec4c468f286c3892852a454f77cb516d6f4a6e7c240de01", "CC0 (Poly Haven)"),
    "concrete_roughness": Download(
        "textures/concrete_floor_01_rough_2k.jpg",
        f"{_POLY}/Textures/jpg/2k/concrete_floor_01/concrete_floor_01_rough_2k.jpg",
        "e66fa941d6daa33ba334c853bf126c85f1d77e7b72778073237e4ef8c7c97774", "CC0 (Poly Haven)"),
}


def ambientcg_ids() -> Tuple[str, ...]:
    """The ambientCG sets these files come from (all CC0), for the credits."""
    ids = [m.group(1) for d in DOWNLOADS.values()
           for m in [re.search(r"ambientcg\.com/get\?file=([A-Za-z]+\d+)_", d.url)] if m]
    return tuple(dict.fromkeys(ids))


def poly_haven_ids() -> Tuple[str, ...]:
    """The Poly Haven assets these files come from (all CC0), for the credits."""
    ids = []
    for d in DOWNLOADS.values():
        m = re.search(r"/Textures/\w+/\w+/([^/]+)/|/HDRIs/\w+/\w+/(.+?)_\d+k\.", d.url)
        if m:
            ids.append(m.group(1) or m.group(2))
    return tuple(dict.fromkeys(ids))


def fetched(name: str) -> str:
    """The local path of a pinned download, fetching it the first time.

    Written to a temporary name and renamed, so workers starting together
    never read half a file."""
    d = DOWNLOADS[name]
    path = CACHE / d.path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
        request = urllib.request.Request(d.url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = response.read()
        except OSError as exc:
            raise ConfigError(f"could not fetch {d.url} ({exc}); put the file at {path}") from None
        if d.member:
            import io
            import zipfile

            data = zipfile.ZipFile(io.BytesIO(data)).read(d.member)
        with open(tmp, "wb") as out:
            out.write(data)
        os.replace(tmp, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != d.sha256:
        raise ConfigError(f"{path} does not match its pinned SHA-256 ({digest}); delete it to refetch")
    return str(path)


# -- the room ---------------------------------------------------------------
ROOM_USD = "{NUCLEUS}/Environments/Simple_Room/simple_room.usd"
ROOM_SCALE = 0.5
#: Simple_Room's floor surface, unscaled, below its origin
ROOM_FLOOR = -0.77
#: its own table and dome light; the ceiling and its light stay
ROOM_HIDDEN = ("table_low_327", "DomeLight")
ROOM_PRIM = "/World/room"

# -- the light ----------------------------------------------------------------
HDRI_INTENSITY = 1000.0

# -- the table ------------------------------------------------------------------
TABLETOP_TEXTURES = (("diffuse_texture", "oak_diffuse", "sRGB"),
                     ("normalmap_texture", "oak_normal", "raw"),
                     ("reflectionroughness_texture", "oak_roughness", "raw"))
#: texture repeats per metre, in world space
TABLETOP_TEXTURE_SCALE = 1.5
LEG_MATERIAL = "{NVIDIA}/Materials/Base/Metals/Aluminum_Cast.mdl"
LEG_RADIUS = 0.025
#: legs stand this far in from the top's corners
LEG_INSET = 0.08

#: Raw RTX settings, written with carb once the simulation context exists:
#: Isaac Lab 2.1's ``RenderCfg`` refuses keys it does not know, such as the
#: texture-streaming ones.  No automatic exposure (a frame's brightness must
#: not depend on the frames before it); no flat ambient fill (the HDRI lights
#: the scene, as RoboLab and Isaac Lab Arena do); materials and textures loaded
#: before a frame is drawn, never streamed in over later ones.
RTX_SETTINGS: Dict[str, object] = {
    "/rtx/post/histogram/enabled": False,
    "/rtx/sceneDb/ambientLightIntensity": 0.0,
    "/rtx/materialDb/syncLoads": True,
    "/rtx/hydra/materialSyncLoads": True,
    "/rtx-transient/resourcemanager/texturestreaming/async": False,
    # blended transparency: without it a material with opacity draws nothing
    # at all and its body comes out black (research/e3_glass_probe.py)
    "/rtx/raytracing/fractionalCutoutOpacity": True,
    # refraction: glass (OmniGlass) is drawn as glass only with this pass on
    # (research/v8_glass_probe.py, 9/24)
    "/rtx/translucency/enabled": True,
}


def nvidia_dir() -> str:
    """The asset server's ``NVIDIA`` directory (materials, skies)."""
    from .assets import nucleus_dir

    isaac = nucleus_dir()
    return isaac[: -len("/Isaac")] + "/NVIDIA" if isaac.endswith("/Isaac") else isaac + "/../NVIDIA"


def room_cfgs(centre: Tuple[float, float], floor_z: float) -> Dict[str, object]:
    """The room and the studio light, as scene entities."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    from .assets import nucleus_dir

    s = ROOM_SCALE
    return {
        "room": AssetBaseCfg(
            prim_path=ROOM_PRIM,
            spawn=sim_utils.UsdFileCfg(usd_path=ROOM_USD.replace("{NUCLEUS}", nucleus_dir()),
                                       scale=(s, s, s)),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(centre[0], centre[1], floor_z - ROOM_FLOOR * s))),
        "dome_light": AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=HDRI_INTENSITY, texture_file=fetched("studio_hdri"),
                                         texture_format="latlong", visible_in_primary_ray=True)),
    }


def tabletop_material():
    """OmniPBR with world-space projection; its textures are asset inputs, which
    ``MdlFileCfg`` cannot write -- ``finish`` sets them."""
    import isaaclab.sim as sim_utils
    from isaaclab.utils import configclass

    @configclass
    class OmniPBRCfg(sim_utils.MdlFileCfg):
        world_or_object: bool | None = None

    return OmniPBRCfg(mdl_path="OmniPBR.mdl", project_uvw=True, world_or_object=True,
                      texture_scale=(TABLETOP_TEXTURE_SCALE, TABLETOP_TEXTURE_SCALE))


def leg_cfgs(prefix: str, centre: Tuple[float, float], size: Tuple[float, float, float],
             top_z: float, floor_z: float) -> Dict[str, object]:
    """Four aluminium legs from the floor to the top's underside."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    sx, sy, sz = size
    height = top_z - sz - floor_z
    material = sim_utils.MdlFileCfg(mdl_path=LEG_MATERIAL.replace("{NVIDIA}", nvidia_dir()))
    out = {}
    for i, (lx, ly) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        out[f"table_leg{i}"] = AssetBaseCfg(
            prim_path=f"{prefix}/table_leg{i}",
            spawn=sim_utils.CylinderCfg(radius=LEG_RADIUS, height=height, visual_material=material),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(centre[0] + lx * (sx / 2 - LEG_INSET), centre[1] + ly * (sy / 2 - LEG_INSET),
                     floor_z + height / 2)))
    return out


def finish(stage, table_prim: str) -> None:
    """After the scene is built: hide the room's own table and dome light and
    the debug ground's grid, and give the tabletop its photographs."""
    from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdShade

    for name in ROOM_HIDDEN:
        prim = stage.GetPrimAtPath(f"{ROOM_PRIM}/{name}")
        if not prim.IsValid():
            continue
        if prim.HasAPI(UsdLux.LightAPI):
            UsdLux.LightAPI(prim).GetIntensityAttr().Set(0.0)
        UsdGeom.Imageable(prim).MakeInvisible()
    ground = stage.GetPrimAtPath("/World/ground")
    if ground.IsValid():
        # the collision plane stays; only the grid is not drawn
        UsdGeom.Imageable(ground).MakeInvisible()
    root = stage.GetPrimAtPath(table_prim)
    for prim in (Usd.PrimRange(root) if root.IsValid() else ()):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        for name, download, colour_space in TABLETOP_TEXTURES:
            inp = shader.CreateInput(name, Sdf.ValueTypeNames.Asset)
            inp.Set(Sdf.AssetPath(fetched(download)))
            inp.GetAttr().SetColorSpace(colour_space)
        shader.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(1.0)


# -- a film's look (``Demonstration.look``) -------------------------------------
#: A demonstration may be shot at another table, in another light (CL2, v3):
#: what the agent learns from it must be the thing, not the picture.  The keys:
#: ``table`` -- a finish (the benchmark's own is wood:oak; wood:ash,
#: wood:plywood, or fabric:<colour>, a dyed linen cloth); ``light`` -- the
#: studio light's strength, times the usual; ``light_colour`` -- its tint
#: (0-1 RGB); ``light_turn`` -- degrees the studio light is turned about the
#: vertical.  Only textures and light attributes change, so the stage stays.
LOOK_KEYS = ("table", "light", "light_colour", "light_turn")
#: table finish -> (diffuse, normal, roughness downloads; repeats per metre)
TABLE_LOOKS = {"wood:oak": (("oak_diffuse", "oak_normal", "oak_roughness"), TABLETOP_TEXTURE_SCALE),
               "wood:ash": (("ash_diffuse", "ash_normal", "ash_roughness"), 3.0),
               "wood:plywood": (("plywood_diffuse", "plywood_normal", "plywood_roughness"), 2.0),
               "fabric": (("linen_diffuse", "linen_normal", "linen_roughness"), 8.0)}
LIGHT_PRIM = "/World/light"


def check_look(look) -> Dict[str, object]:
    """``look`` checked and filled in with the benchmark's own values."""
    look = dict(look or {})
    unknown = sorted(set(look) - set(LOOK_KEYS))
    if unknown:
        raise ConfigError(f"a film's look has no {unknown}; it has {list(LOOK_KEYS)}")
    table = str(look.get("table", "wood:oak"))
    if table not in TABLE_LOOKS and not table.startswith("fabric:"):
        raise ConfigError(f"no table finish {table!r}; have {sorted(TABLE_LOOKS)} or fabric:<colour>")
    colour = tuple(float(v) for v in look.get("light_colour", (1.0, 1.0, 1.0)))
    if len(colour) != 3 or not all(0.0 <= v <= 1.0 for v in colour):
        raise ConfigError(f"light_colour {colour} is not an RGB triple in 0-1")
    light = float(look.get("light", 1.0))
    if not 0.1 <= light <= 4.0:
        raise ConfigError(f"light {light} is outside 0.1-4 times the usual")
    return {"table": table, "light": light, "light_colour": colour,
            "light_turn": float(look.get("light_turn", 0.0))}


def apply_look(stage, table_prim: str, look=None) -> Dict[str, object]:
    """Dress the table and set the studio light for ``look`` (None: the
    benchmark's own).  Returns the look as applied."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    from ..finishes import finish_rgb, linear

    look = check_look(look)
    table = look["table"]
    maps, scale = TABLE_LOOKS["fabric" if table.startswith("fabric:") else table]
    tint = linear(finish_rgb(table)) if table.startswith("fabric:") else (1.0, 1.0, 1.0)
    root = stage.GetPrimAtPath(table_prim)
    for prim in (Usd.PrimRange(root) if root.IsValid() else ()):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        for (name, _, colour_space), download in zip(TABLETOP_TEXTURES, maps):
            inp = shader.CreateInput(name, Sdf.ValueTypeNames.Asset)
            inp.Set(Sdf.AssetPath(fetched(download)))
            inp.GetAttr().SetColorSpace(colour_space)
        shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(scale, scale))
        shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*tint))
    prim = stage.GetPrimAtPath(LIGHT_PRIM)
    if prim.IsValid():
        dome = UsdLux.DomeLight(prim)
        dome.GetIntensityAttr().Set(HDRI_INTENSITY * look["light"])
        dome.GetColorAttr().Set(Gf.Vec3f(*look["light_colour"]))
        xform = UsdGeom.Xformable(prim)
        turn = next((op for op in xform.GetOrderedXformOps()
                     if op.GetOpName() == "xformOp:rotateZ:look"), None)
        if turn is None:
            turn = xform.AddRotateZOp(opSuffix="look")
        turn.Set(look["light_turn"])
    return look


def apply_rtx_settings() -> None:
    import carb

    settings = carb.settings.get_settings()
    for key, value in RTX_SETTINGS.items():
        settings.set(key, value)


def wait_for_assets(sim, limit: int = 600) -> int:
    """Render until every material and texture has loaded: a frame drawn
    before that shows grey or blank surfaces."""
    from isaacsim.core.simulation_manager import SimulationManager

    n = 0
    while SimulationManager.assets_loading() and n < limit:
        sim.render()
        n += 1
    return n


__all__ = ["CACHE", "DOWNLOADS", "fetched", "room_cfgs", "tabletop_material", "leg_cfgs",
           "finish", "apply_rtx_settings", "wait_for_assets", "RTX_SETTINGS", "LOOK_KEYS",
           "check_look", "apply_look"]
