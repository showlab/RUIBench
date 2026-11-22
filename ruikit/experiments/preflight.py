"""Before anything starts: the vision services the jobs need, and the sandbox
their programs run in."""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

from ruikit.experiments.plans import (
    label,
)


def vision_check(profiles: Sequence[str]) -> Optional[str]:
    """Why ``profiles`` cannot run on Isaac, or None.  On Isaac the vision
    tools answer from pixels, through a vision server; checked here, a
    server that is not running shows before any simulator starts, not after
    each one has (every server ``RUIKIT_VISION_URL`` lists: a worker gets one)."""
    from ruikit.extras.backends import VISION, get_backend
    from ruikit.interfaces import load_profile
    from ruikit.perception.service.client import DEFAULT_URL, VisionServer

    urls = [u.strip() for u in (os.environ.get("RUIKIT_VISION_URL") or DEFAULT_URL).split(",")
            if u.strip()]
    told: Dict[tuple, dict] = {}
    for p in profiles:
        prof = load_profile(p)
        if prof.perception == "synthetic":
            return (f"{label(p)}: synthetic perception is refused where pictures are rendered "
                    "(v1.13); use the profile without @synthetic, or --engine mock")
        if prof.perception != "model":
            continue                          # the privileged control asks no vision server
        settings = prof.perception_settings()
        name = settings["backend"]
        try:
            backend = get_backend(VISION, "vision", name)
            servers = ([VisionServer(name, url=u) for u in urls]
                       if isinstance(backend, VisionServer) else [backend])
            for server in servers:
                key = (name, getattr(server, "url", None))
                if key not in told:
                    told[key] = server.info()
                if settings["depth"] == "mono" and not told[key].get("depth_model"):
                    return (f"{label(p)} measures depth from the picture (depth=mono), and the "
                            f"vision server at {key[1]} has no depth model "
                            "(start it with --depth dav2-small)")
        except Exception as exc:              # a ToolError: not running, or another backend
            hint = getattr(exc, "hint", "")
            return f"{label(p)}: {exc}" + (f" -- {hint}" if hint else "")
    return None


def sandbox_check(profiles: Sequence[str]) -> Optional[str]:
    """Why the programs of ``profiles`` would not run in the sandbox here, or None.
    A program runs in bubblewrap, apart from the repository, the task cards, the
    logs, the home folder and the network (``ruikit.interfaces.sandbox.host``);
    where bwrap is missing or cannot start, a program would run as a plain process
    that can read them, or not at all."""
    from ruikit.interfaces import load_profile
    from ruikit.interfaces.sandbox.host import Sandbox

    programs = [p for p in profiles if load_profile(p).code_tools]
    if not programs:
        return None
    box = Sandbox(isolation="bwrap")
    try:
        done = box.run("result = 1", {}, lambda name, args: {})
    except Exception as exc:                  # a SandboxError: bwrap is not installed
        done = {"error": f"{type(exc).__name__}: {exc}", "isolation": "none"}
    finally:
        box.close()
    if done.get("isolation") == "bwrap" and done.get("value") == 1 and not done.get("error"):
        return None
    return (f"{label(programs[0])} runs programs in a sandbox, and bubblewrap does not work here "
            f"({(done.get('error') or done.get('isolation') or '').strip()[:200]}): install bwrap "
            "and allow unprivileged user namespaces")
