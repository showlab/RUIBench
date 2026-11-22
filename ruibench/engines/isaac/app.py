"""Isaac Sim application lifecycle.

Isaac Sim has one hard rule that shapes this whole package: ``SimulationApp``
must be constructed **before** any ``omni.*``, ``pxr`` or ``isaaclab`` import,
and only once per process.  Everything Isaac-related therefore imports lazily,
inside functions, after :func:`launch` has run.

That is also why the rest of RUIBench never imports this module at top level:
``ruibench validate``, ``ruibench tasks`` and ``ruibench report`` must keep
working on a laptop with no GPU.
"""

from __future__ import annotations

import os
from typing import Optional

from ...core.errors import EngineNotAvailable

_APP = None
_LAUNCHER = None
_CONFIG: dict = {}

#: The setting Isaac Lab reads at import time to build every USD path it owns.
CLOUD_ROOT_SETTING = "/persistent/isaac/asset_root/cloud"

#: Where the public Isaac assets live.  Used only as a last resort, and only to
#: verify the catalogue without launching the app.
PUBLIC_ASSET_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"
)

INSTALL_HINT = """Isaac Sim is not importable in this interpreter.

    pip install "isaacsim[all,extscache]==4.5.0" \\
        --extra-index-url https://pypi.nvidia.com
    # Isaac Lab 2.1 from source (isaaclab_assets is not on PyPI):
    git clone --depth 1 --branch v2.1.0 https://github.com/isaac-sim/IsaacLab
    pip install "setuptools<81" flatdict==4.0.1 --no-build-isolation
    pip install -e IsaacLab/source/isaaclab -e IsaacLab/source/isaaclab_assets
    pip install "warp-lang==1.5.0" h5py requests      # see below

**Pin warp-lang to 1.5.0.**  Isaac Lab leaves it unpinned, so pip picks the
newest release, while Kit ships its own warp 1.5 as an extension.  Two warp
native libraries in one process corrupted the heap intermittently here: a
two-arm scene with a table and a camera aborted in 3 of 5 runs with warp 1.17,
and in 0 of 8 after pinning 1.5.0.

RUIBench is built and checked on Isaac Sim 4.5.0 with Isaac Lab 2.1, the
matching task/asset layer, which supplies the robot configs, camera sensors
and IK controller this engine builds on.  Needs an RTX GPU, a driver new enough for
the shipped PhysX/RTX build, and Python 3.10 or 3.11.  Install it in its own
virtualenv -- the dependency set is large and pins hard.

Then, in that virtualenv:

    pip install -e <this repository>[isaac]
    export OMNI_KIT_ACCEPT_EULA=YES     # after reading NVIDIA's terms
    ruibench setup                      # the ARX X5 description, the room's
                                        # textures, the scanned objects
    ruibench doctor                     # re-checks all of this

`ruibench doctor` prints what is still missing.
"""


EULA_HINT = """Isaac Sim will not start until NVIDIA's licence terms are accepted.

Importing isaacsim prompts for the Omniverse EULA, which cannot be answered in a
non-interactive process.  Read the terms, and if you accept them, set:

    export OMNI_KIT_ACCEPT_EULA=YES

RUIBench does not set this for you: accepting a licence agreement is the
operator's decision, not the harness's.
"""


def launch(headless: bool = True, enable_cameras: bool = True, gpu: Optional[int] = None,
           rendering_mode: Optional[str] = None, **_ignored):
    """Start (or return) the single simulation app for this process.

    Launched through Isaac Lab's ``AppLauncher`` rather than a bare
    ``SimulationApp``.  The difference matters: rendering cameras only work under
    the ``headless.rendering`` experience, and Isaac Lab refuses to spawn a camera
    unless ``/isaaclab/cameras_enabled`` is set -- which is exactly what
    ``AppLauncher(enable_cameras=True)`` arranges.  RUIBench always renders (the
    rendered image *is* the observation), so cameras are on by default.

    ``gpu`` pins rendering and physics to one device.  Worth setting on a shared
    machine: left alone, Kit opens every visible GPU and compiles shaders on all
    of them, which is slow and rude to whoever else is on the box.  It also
    reads ``RUIBENCH_ISAAC_GPU``.

    The first launch on a new machine takes a long time -- about ten minutes
    here -- while the RTX shader cache is built.  Later launches take seconds.
    """
    global _APP, _CONFIG, _LAUNCHER
    if _APP is not None:
        return _APP
    if not os.environ.get("OMNI_KIT_ACCEPT_EULA"):
        raise EngineNotAvailable(EULA_HINT)

    if gpu is None and os.environ.get("RUIBENCH_ISAAC_GPU"):
        gpu = int(os.environ["RUIBENCH_ISAAC_GPU"])
    if gpu is not None:
        # pin before anything CUDA-aware is imported; cuda:0 is then that GPU
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))

    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise EngineNotAvailable(INSTALL_HINT) from exc

    _CONFIG = {"headless": headless, "enable_cameras": enable_cameras, "device": "cuda:0"}
    if rendering_mode:
        _CONFIG["rendering_mode"] = rendering_mode
    _LAUNCHER = AppLauncher(**_CONFIG)
    _APP = _LAUNCHER.app
    _fix_asset_root()
    return _APP


def _fix_asset_root() -> None:
    """Point Isaac Lab at the asset server.

    Isaac Lab reads ``/persistent/isaac/asset_root/cloud`` **at import time**,
    and a pip install of Isaac Sim leaves that setting empty -- only
    ``/persistent/isaac/asset_root/default`` is populated.  The result is an
    asset root of the literal string ``None``, so every USD path Isaac Lab
    builds (the Franka robot included) is nonsense and nothing loads.

    So: after the app starts and before anything imports ``isaaclab``, copy the
    resolved root across.  This is the concrete reason for the ordering rule
    that shapes this whole package.
    """
    import carb

    settings = carb.settings.get_settings()
    if settings.get(CLOUD_ROOT_SETTING):
        return
    root = os.environ.get("RUIBENCH_ISAAC_NUCLEUS")
    if not root:
        try:
            from isaacsim.storage.native import get_assets_root_path
            root = get_assets_root_path()
        except Exception:
            root = None
    if not root:
        root = settings.get("/persistent/isaac/asset_root/default")
    if root:
        settings.set(CLOUD_ROOT_SETTING, root)


def asset_root() -> Optional[str]:
    """The resolved asset-server root, or None if the app has not started."""
    try:
        import carb
        return carb.settings.get_settings().get(CLOUD_ROOT_SETTING)
    except Exception:
        return os.environ.get("RUIBENCH_ISAAC_NUCLEUS")


def app():
    if _APP is None:
        raise EngineNotAvailable("call ruibench.engines.isaac.app.launch() first")
    return _APP


def shutdown() -> None:
    global _APP
    if _APP is not None:
        _APP.close()
        _APP = None


def is_available() -> bool:
    """True if Isaac Sim could be started in this interpreter.  Does not start it."""
    import importlib.util
    return importlib.util.find_spec("isaacsim") is not None


def eula_accepted() -> bool:
    return bool(os.environ.get("OMNI_KIT_ACCEPT_EULA"))
