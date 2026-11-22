"""Serve the generated pages over HTTP -- for machines you reach over SSH.

    ruibench serve                 # runs/ on http://127.0.0.1:8765/

A report or viewer on a remote GPU machine cannot be opened by clicking its
path.  This serves the directory (pages and the frames they link to) on the
loopback interface only; forward the port -- VS Code's Ports panel, or
``ssh -L 8765:localhost:8765 host`` -- and open the link on your own machine.
"""

from __future__ import annotations

import functools
import http.server
import os
import re
import socket
from pathlib import Path
from typing import List, Optional

DEFAULT_PORT = 8765


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Static files, quietly, with byte ranges.

    Ranges matter for the episode videos: Safari will not play a video from a
    server that cannot answer ``Range: bytes=...``, and no browser can seek in
    one without it.
    """

    def log_message(self, format, *args):        # noqa: A002 - signature is the base class's
        pass

    def send_head(self):
        spec = self.headers.get("Range", "")
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", spec.strip())
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        first, last = m.groups()
        if first == "":                              # "-N": the last N bytes
            start, end = max(0, size - int(last or 0)), size - 1
        else:
            start, end = int(first), min(size - 1, int(last) if last else size - 1)
        if start > end or start >= size:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None
        f = open(path, "rb")
        f.seek(start)
        self._remaining = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self._remaining))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        self._remaining = None
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def end_headers(self):
        if not self.headers.get("Range"):
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()


def pages(root: str | Path) -> List[Path]:
    """Every HTML page under ``root``, newest first."""
    root = Path(root)
    return sorted(root.rglob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)


def free_port(host: str, start: int, tries: int = 20) -> int:
    for port in range(start, start + tries):
        with socket.socket() as s:
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"no free port in {start}-{start + tries - 1}")


def make_server(root: str | Path, host: str = "127.0.0.1",
                port: Optional[int] = None) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(_Handler, directory=str(Path(root).resolve()))
    chosen = free_port(host, port if port is not None else DEFAULT_PORT)
    return http.server.ThreadingHTTPServer((host, chosen), handler)
