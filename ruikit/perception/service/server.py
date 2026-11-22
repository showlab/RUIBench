"""The vision server behind ``perception: model``: open-vocabulary detection
and segmentation, over HTTP.

    python -m ruikit.perception.service.server [--backend sam3] [--port 18300] [--device cuda:0]

It runs in its own process (its own GPU memory, torch and transformers), apart
from the simulator, and sees only what a camera would hand it: an RGB picture
and words.  Four backends, chosen when it starts:

* ``sam3`` (the default) -- **SAM 3**: a noun phrase in, every instance of it
  out, each with its mask; the same weights take a point or a box.  The
  weights are gated (``facebook/sam3``: ask for access on Hugging Face, then
  download once with a token; after that the server starts from the cache).
* ``grounded-sam2`` -- **Grounding DINO** finds boxes for a phrase and
  **SAM 2.1** turns a box, or a point, into a mask: open weights.
* ``gdino-tiny-sam2`` -- the same with **Grounding DINO tiny**: the detector's
  size, on its own.
* ``owlv2-sam2`` -- **OWLv2** (large, ensemble) finds the boxes, SAM 2.1
  outlines them.

The last two are for the ablations: the same pictures, words and depth
filtering, another detector.  Every one is open weights except SAM 3.
(MM-Grounding-DINO and LLMDet were tried too: under transformers 5.17 their
boxes come out squashed into a band across the picture, weights and all
loaded, so they are left out.)

``--depth dav2-small`` also loads a monocular depth model, **Depth Anything
V2** (metric, indoor, small; Apache-2.0), for profiles that measure from the
RGB picture alone (``depth=mono``): ``POST /depth`` answers with its depth.

``POST /detect``   {image, phrases} -> {detections: [{phrase, score, box, mask, mask_score}], size}
``POST /segment``  {image, point? | box?} -> {mask, score, box} or {}
``POST /depth``    {image} -> {depth, size}: metres along the optical axis, float16, base64
``GET  /info``     the backend, its models and their revisions, the thresholds

One phrase a pass, so two phrases never merge into one label; every phrase's
answers are returned, and what to make of two phrases firing on one thing is
the caller's.  Images are PNG, base64; a mask is its bits packed row by row
(``numpy.packbits``), base64; boxes are [u0, v0, u1, v1] in pixels.  The same
picture gives the same answer: nothing is sampled.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from ruikit.perception.service.models import BACKENDS, DEPTHS

#: a box over more of the picture than this is the scene, not a thing in it
MAX_BOX_FRACTION = 0.5
#: two boxes of one phrase this much the same (IoU) are one detection: the
#: better score stays.  Boxes of different phrases are all returned
SAME_BOX_IOU = 0.6
DEFAULT_PORT = 18300
DEFAULT_BACKEND = "sam3"


def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class Models:
    def __init__(self, device: str, backend: str = DEFAULT_BACKEND,
                 depth: Optional[str] = None) -> None:
        self.engine = BACKENDS[backend](device)
        self.depth_engine = DEPTHS[depth](device) if depth else None
        self.device = device
        self.lock = threading.Lock()
        self._results: "OrderedDict[str, dict]" = OrderedDict()

    def info(self) -> dict:
        extra = self.depth_engine.info() if self.depth_engine is not None else {}
        return {**self.engine.info(), **extra, "device": self.device}

    def depth(self, png: bytes) -> dict:
        import numpy as np
        from PIL import Image

        if self.depth_engine is None:
            raise RuntimeError("this server has no depth model: start it with --depth dav2-small")
        with self.lock:
            image = Image.open(io.BytesIO(png)).convert("RGB")
            z = self.depth_engine.depth(image)
        return {"size": list(z.shape),
                "depth": base64.b64encode(np.asarray(z, dtype=np.float16).tobytes()).decode()}

    def detect(self, png: bytes, phrases) -> dict:
        from PIL import Image

        key = hashlib.sha1(png + json.dumps(list(phrases)).encode()).hexdigest()
        with self.lock:
            hit = self._results.get(key)
            if hit is not None:
                return hit
            image = Image.open(io.BytesIO(png)).convert("RGB")
            width, height = image.size
            found = []
            for phrase, score, box, mask, mask_score in (self.engine.detect(image, list(phrases))
                                                         if phrases else []):
                box = [max(0.0, box[0]), max(0.0, box[1]), min(width, box[2]), min(height, box[3])]
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area < 4 or area > MAX_BOX_FRACTION * width * height or not mask.any():
                    continue
                found.append({"phrase": phrase, "score": score, "box": box, "_mask": mask,
                              "mask_score": mask_score})
            found.sort(key=lambda d: -d["score"])
            kept = []
            for d in found:
                if all(d["phrase"] != k["phrase"] or _iou(d["box"], k["box"]) < SAME_BOX_IOU
                       for k in kept):
                    kept.append(d)
            result = {"size": [height, width],
                      "detections": [{"phrase": o["phrase"], "score": round(o["score"], 4),
                                      "box": [round(v, 1) for v in o["box"]],
                                      "mask": _pack(o["_mask"]),
                                      "mask_score": round(o["mask_score"], 4)} for o in kept]}
            self._results[key] = result
            while len(self._results) > 64:
                self._results.popitem(last=False)
            return result

    def segment(self, png: bytes, point=None, box=None) -> dict:
        import numpy as np
        from PIL import Image

        with self.lock:
            image = Image.open(io.BytesIO(png)).convert("RGB")
            mask, score = self.engine.segment(image, point=point, box=box)
        if not mask.any():
            return {}
        vs, us = np.nonzero(mask)
        return {"size": list(mask.shape), "mask": _pack(mask), "score": round(score, 4),
                "box": [int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1]}


def _pack(mask) -> str:
    import numpy as np

    return base64.b64encode(np.packbits(mask.astype(bool), axis=None).tobytes()).decode()


def serve(models: Models, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:        # quiet
            pass

        def _reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/info":
                self._reply(200, models.info())
            else:
                self._reply(404, {"error": "unknown path"})

        def do_POST(self) -> None:
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                png = base64.b64decode(req["image"])
                if self.path.rstrip("/") == "/detect":
                    self._reply(200, models.detect(png, [str(p) for p in req.get("phrases") or []]))
                elif self.path.rstrip("/") == "/segment":
                    self._reply(200, models.segment(png, point=req.get("point"),
                                                    box=req.get("box")))
                elif self.path.rstrip("/") == "/depth":
                    self._reply(200, models.depth(png))
                else:
                    self._reply(404, {"error": "unknown path"})
            except Exception as exc:                 # the client gets the reason, not a hang
                self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ruikit.perception.service.server",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--backend", default=DEFAULT_BACKEND, choices=sorted(BACKENDS))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--device", default="cuda:0", help="cuda:N or cpu (choose the card with "
                    "CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--threshold", type=float, help="for trying another score threshold than "
                    "the backend's own (it is reported by /info and so in every log)")
    ap.add_argument("--depth", choices=sorted(DEPTHS), default=None,
                    help="also load a monocular depth model, for profiles with depth=mono")
    args = ap.parse_args(argv)
    models = Models(args.device, args.backend, args.depth)
    if args.threshold is not None:
        if hasattr(models.engine, "THRESHOLD"):
            models.engine.THRESHOLD = args.threshold
        elif hasattr(models.engine, "BOX_THRESHOLD"):
            models.engine.BOX_THRESHOLD = args.threshold         # the box's; a phrase's stays
        else:
            ap.error(f"--threshold: {args.backend} has no threshold to set")
    server = serve(models, args.host, args.port)
    print(f"vision server on http://{args.host}:{args.port}  {json.dumps(models.info())}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
