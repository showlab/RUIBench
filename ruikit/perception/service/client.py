"""The vision backend of ``perception: model``: the client of the vision
server (``python -m ruikit.perception.service.server``).

The server holds the models in its own process -- SAM 3 (``sam3``, the
default), or a box detector with SAM 2.1 (``grounded-sam2``, ``gdino-tiny-sam2``,
``owlv2-sam2``) -- and this side sends it a picture and
words and gets boxes and masks back; or a picture alone, and gets the depth a
monocular model sees in it (``depth``, a server started with ``--depth``).  It is handed
pixels and nothing else.  Where the server is: ``RUIKIT_VISION_URL`` (default
``http://127.0.0.1:18300``).  Several, comma separated, are copies of one
server: a process takes one of them (by its process id) and keeps it, so many
episodes at once spread over them.  A profile names the backend it is to be
answered by (``params.perception.backend``); a server that runs another is
refused, so two conditions never mix.  Standard library and numpy only.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import List, Optional, Sequence

from ruikit.core.result import ToolError

DEFAULT_URL = "http://127.0.0.1:18300"
#: what answers ``perception: model`` unless the profile names another
DEFAULT_BACKEND = "sam3"
#: per backend: the lowest score it reports, and the score under which an
#: answer is doubtful -- what ``vision_detect`` tells the agent.  SAM 3's own
#: default cut is 0.5; on the marked frames (docs/RUIKIT.md) a phrase that
#: fits ("red cross", "blue ring") scores its things above it, and one that
#: fits loosely ("x piece") leaves most of them below
SCORES = {"sam3": (0.2, 0.5), "grounded-sam2": (0.25, 0.4), "gdino-tiny-sam2": (0.25, 0.4),
          "owlv2-sam2": (0.1, 0.2)}


class VisionServer:
    def __init__(self, name: str = DEFAULT_BACKEND, url: Optional[str] = None,
                 timeout: float = 120.0) -> None:
        self.name = name
        urls = [u.strip().rstrip("/") for u in (
            url or os.environ.get("RUIKIT_VISION_URL") or DEFAULT_URL).split(",") if u.strip()]
        self.url = urls[os.getpid() % len(urls)]
        self.timeout = timeout
        self._info: Optional[dict] = None

    def _call(self, path: str, body: Optional[dict] = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data,
                                     {"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                why = json.loads(exc.read()).get("error", "")
            except Exception:
                why = ""
            raise ToolError("tool_failed", f"the vision server returned HTTP {exc.code}"
                            + (f": {why}" if why else "")) from None
        except (urllib.error.URLError, OSError) as exc:
            raise ToolError("backend_unavailable",
                            f"the vision server at {self.url} is not answering ({exc})",
                            hint="start it: python -m ruikit.perception.service.server (its own "
                                 "process and GPU), or set RUIKIT_VISION_URL") from None
        return out

    def info(self) -> dict:
        """The models behind the answers and their revisions: for the log.  The
        server must run the backend this one stands for."""
        if self._info is None:
            said = self._call("/info")
            runs = said.get("backend", "grounded-sam2")       # an older server does not say
            if runs != self.name:
                raise ToolError("backend_unavailable",
                                f"the vision server at {self.url} runs {runs!r}, and this "
                                f"profile is answered by {self.name!r}",
                                hint=f"start one with --backend {self.name} (another port and "
                                     "RUIKIT_VISION_URL, if both are needed)")
            self._info = {**said, "backend": self.name}
        return self._info

    @staticmethod
    def _png(image) -> str:
        from ruikit.env.frames import encode_png

        png = encode_png(image)
        if png is None:
            raise ToolError("not_supported", "this frame has no pixels")
        return base64.b64encode(png).decode()

    @staticmethod
    def _mask(packed: str, size):
        import numpy as np

        rows, cols = int(size[0]), int(size[1])
        bits = np.unpackbits(np.frombuffer(base64.b64decode(packed), dtype=np.uint8))
        return bits[:rows * cols].reshape(rows, cols).astype(bool)

    def detect(self, image, phrases: Sequence[str]) -> List[dict]:
        """Boxes and masks for ``phrases`` in ``image`` (HxWx3 uint8): a list of
        {phrase, score, box, mask (bool HxW), mask_score}."""
        out = self._call("/detect", {"image": self._png(image), "phrases": list(phrases)})
        return [{**d, "mask": self._mask(d["mask"], out["size"])} for d in out["detections"]]

    def segment(self, image, point=None, box=None) -> Optional[dict]:
        """The mask at a pixel, or in a box: {mask, score, box}, or None."""
        out = self._call("/segment", {"image": self._png(image), "point": point, "box": box})
        if not out:
            return None
        return {**out, "mask": self._mask(out["mask"], out["size"])}

    def depth(self, image):
        """The depth a monocular model sees in ``image`` (HxWx3 uint8): HxW
        float32 metres along the optical axis, at the model's own scale."""
        import numpy as np

        if not self.info().get("depth_model"):
            raise ToolError("backend_unavailable",
                            f"the vision server at {self.url} has no depth model, and this "
                            "profile measures depth from the picture (depth=mono)",
                            hint="start it with --depth dav2-small")
        out = self._call("/depth", {"image": self._png(image)})
        rows, cols = int(out["size"][0]), int(out["size"][1])
        z = np.frombuffer(base64.b64decode(out["depth"]), dtype=np.float16)
        return z[:rows * cols].reshape(rows, cols).astype(np.float32)
