"""Settings of perception from pixels: the detector, the depth, and how much worse than a
real robot's the errors may be made."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

#: the detectors of the ``model`` perception backend: what answers the perception
#: tools (``rui.pose/v1@owlv2-sam2,depth=mono``)
DETECTORS: Mapping[str, Tuple[str, str]] = {
    "sam3": ("SAM 3：一个模型按词找出并分割每一个", "SAM 3: one model finds and outlines each instance"),
    "grounded-sam2": ("Grounding DINO 找框 + SAM 2.1 分割", "Grounding DINO boxes, SAM 2.1 masks"),
    "gdino-tiny-sam2": ("Grounding DINO tiny 找框 + SAM 2.1 分割（检测器小一号）",
                        "Grounding DINO tiny boxes, SAM 2.1 masks (a smaller detector)"),
    "owlv2-sam2": ("OWLv2 找框 + SAM 2.1 分割", "OWLv2 boxes, SAM 2.1 masks"),
}


DEPTHS: Mapping[str, Tuple[str, str]] = {
    "sensor": ("深度相机（带真实噪声和边缘空洞）", "the depth camera, with its noise and holes"),
    "mono": ("只用 RGB：Depth Anything V2 预测深度，尺度按已知桌面校准",
             "RGB only: depth predicted by Depth Anything V2, its scale fitted to the known table top"),
}


#: how much worse than a real robot's the depth noise and the camera calibration
#: may be made (never better: 1 is a real robot's)
SCALES: Tuple[float, ...] = (1.0, 2.0, 4.0)


PERCEPTION_DEFAULTS: Mapping[str, Any] = {"backend": "sam3", "depth": "sensor",
                                          "noise": 1.0, "calib": 1.0}


def parse_perception(text: str) -> Dict[str, Any]:
    """'owlv2-sam2,depth=mono,noise=2' -> {backend, depth, noise}.  A bare word
    is the detector; 'model', the backend's own name, means the default one."""
    out: Dict[str, Any] = {}
    for part in (p.strip() for p in text.split(",") if p.strip()):
        key, eq, value = part.partition("=")
        if not eq:
            if key == "model":
                continue
            key, value = "backend", part
        key = {"detector": "backend"}.get(key, key)
        if key == "backend":
            if value not in DETECTORS:
                raise ValueError(f"no perception backend {value!r}; there are "
                                 + ", ".join(DETECTORS))
        elif key == "depth":
            if value not in DEPTHS:
                raise ValueError(f"depth must be one of {', '.join(DEPTHS)}, not {value!r}")
        elif key in ("noise", "calib"):
            try:
                scale = float(value)
            except ValueError:
                raise ValueError(f"{key} must be a number, not {value!r}") from None
            if scale not in SCALES:
                raise ValueError(f"{key} is a multiple of a real robot's error: one of "
                                 f"{', '.join(f'{s:g}' for s in SCALES)} (never better)")
            value = scale
        else:
            raise ValueError(f"unknown perception setting {key!r} (backend, depth, noise, calib)")
        out[key] = value
    return out


def perception_label(settings: Mapping[str, Any]) -> str:
    """The canonical text of perception settings, defaults left out ('' for all
    defaults): it goes into the profile id and so into every log."""
    parts = []
    for key in ("backend", "depth", "noise", "calib"):
        value = settings.get(key, PERCEPTION_DEFAULTS[key])
        if value == PERCEPTION_DEFAULTS[key]:
            continue
        if key == "backend":
            parts.insert(0, str(value))
        else:
            parts.append(f"{key}={value:g}" if isinstance(value, float) else f"{key}={value}")
    return ",".join(parts)
