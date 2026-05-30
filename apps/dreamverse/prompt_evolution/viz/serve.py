"""
Zero-dependency web server for the evolution visualization.

Serves the dashboard (static/index.html), the run data (runs/<id>/evolution.json),
and the rollout videos with HTTP Range support (so mp4 seeking works). Symlinked
sample clips are followed. Intended to be exposed via ngrok (alexzms3.ngrok.app).

Usage:
    python serve.py [--port 8088]
    # then:  ngrok http --domain=alexzms3.ngrok.app 8088
"""

import argparse
import json
import mimetypes
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE  # web root = viz/
RUNS_DIR = os.path.join(HERE, "runs")


class Handler(BaseHTTPRequestHandler):

    def log_message(self, *a):  # quiet
        pass

    def _send_bytes(self, data: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            return self._serve_file(os.path.join(ROOT, "static", "index.html"))
        if path == "/api/runs":
            runs = []
            if os.path.isdir(RUNS_DIR):
                runs = sorted(d for d in os.listdir(RUNS_DIR)
                              if os.path.isfile(os.path.join(RUNS_DIR, d, "evolution.json")))
            return self._send_bytes(json.dumps({"runs": runs}).encode(), "application/json")
        # static assets + run files + videos
        if path.startswith("/static/") or path.startswith("/runs/"):
            return self._serve_file(os.path.join(ROOT, path.lstrip("/")))
        self._send_bytes(b"not found", "text/plain", 404)

    do_HEAD = do_GET

    def _serve_file(self, full: str):
        full = os.path.realpath(full)
        if not os.path.isfile(full):
            return self._send_bytes(b"not found", "text/plain", 404)
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        size = os.path.getsize(full)
        rng = self.headers.get("Range")
        if rng and (m := re.match(r"bytes=(\d+)-(\d*)", rng)):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
            end = min(end, size - 1)
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if self.command != "HEAD":
                with open(full, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if self.command != "HEAD":
            with open(full, "rb") as f:
                self.wfile.write(f.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"evolve-viz serving on http://{args.host}:{args.port}  (ngrok -> alexzms3.ngrok.app)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
