#!/usr/bin/env python3
"""Serve a browser-based MP4 preview for the local results directory."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


class PreviewHandler(BaseHTTPRequestHandler):
    server_version = "AMIVideoPreview/1.0"

    @property
    def root(self) -> Path:
        return self.server.results_root  # type: ignore[attr-defined]

    @property
    def web_root(self) -> Path:
        return self.server.web_root  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/videos":
            self._api_videos()
        elif parsed.path.startswith("/video/"):
            self._serve_video(unquote(parsed.path[len("/video/"):]))
        elif parsed.path in {"/", "/index.html"}:
            self._serve_bytes(self.web_root / "index.html", "text/html; charset=utf-8")
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/video/"):
            self._serve_video(unquote(parsed.path[len("/video/"):]), head_only=True)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _api_videos(self) -> None:
        videos = [{"name": path.relative_to(self.root).as_posix(), "size": path.stat().st_size} for path in sorted(self.root.rglob("*.mp4")) if path.is_file()]
        payload = json.dumps({"videos": videos}, ensure_ascii=False).encode("utf-8")
        self._serve_payload(payload, "application/json; charset=utf-8")

    def _safe_video_path(self, name: str) -> Path | None:
        candidate = (self.root / name).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() and candidate.suffix.lower() == ".mp4" else None

    def _serve_video(self, name: str, head_only: bool = False) -> None:
        path = self._safe_video_path(name)
        if path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Video not found")
            return
        total = path.stat().st_size
        start, end = 0, total - 1
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            value = range_header[6:].split(",", 1)[0].strip()
            left, _, right = value.partition("-")
            try:
                if left:
                    start = int(left)
                    end = int(right) if right else end
                else:
                    start = max(0, total - int(right))
                if start < 0 or start >= total or end < start:
                    raise ValueError
                end = min(end, total - 1)
            except ValueError:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            status = HTTPStatus.PARTIAL_CONTENT
        else:
            status = HTTPStatus.OK
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.end_headers()
        if not head_only:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    def _serve_bytes(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        self._serve_payload(path.read_bytes(), content_type)

    def _serve_payload(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"), help="Directory containing MP4 files.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the preview page in the default browser.")
    args = parser.parse_args()
    results_root = args.results.resolve()
    if not results_root.is_dir():
        raise SystemExit(f"Results directory does not exist: {results_root}")
    web_root = (Path(__file__).resolve().parents[1] / "web" / "video_preview").resolve()
    server = ThreadingHTTPServer((args.host, args.port), PreviewHandler)
    server.results_root = results_root  # type: ignore[attr-defined]
    server.web_root = web_root  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}/"
    print(f"MP4 preview: {url}")
    print(f"Serving: {results_root}")
    if args.open:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
