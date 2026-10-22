"""The mix server: live stream + phone remote over the local network.

A small HTTP server (standard library) serves:

  /              the phone remote (now playing, decks, queue, seed search,
                 transition markers, transport)
  /live.mp3      the live mix as an endless MP3 stream (192 kbit/s)
  /status        JSON: decks, BPM, keys, auto-mix status (listen-only)
  /api/state     full remote state            (PIN)
  /api/search    library search               (PIN)
  /api/wave      overview waveform of a deck  (PIN)
  /api/cmd       POST {"action": ...}         (PIN)
  /art           album art thumbnails         (PIN)

The PIN comes in the X-CTA-Key header or a ``k`` query parameter; wrong
PINs are rate limited per address.

Encoding runs in its own thread; every listener gets its own bounded queue
so a slow phone can never stall the audio engine or other listeners.
"""

from __future__ import annotations

import hmac
import json
import logging
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import av
import numpy as np

from .engine import SR
from .remote_page import PAGE

log = logging.getLogger(__name__)


ICONS = {"/apple-touch-icon.png": "app-icon-180.png", "/apple-touch-icon-precomposed.png": "app-icon-180.png",
         "/icon-512.png": "app-icon-512.png", "/favicon.ico": "app-icon-180.png"}

MANIFEST = json.dumps({
    "name": "CTA live remote", "short_name": "CTA", "display": "standalone", "orientation": "portrait",
    "background_color": "#121316", "theme_color": "#121316",
    "icons": [{"src": "/apple-touch-icon.png", "sizes": "180x180", "type": "image/png"},
              {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}],
})


def _icon(name: str) -> bytes | None:
    try:
        from importlib import resources

        return resources.files("cta").joinpath(name).read_bytes()
    except (OSError, FileNotFoundError):
        return None


def lan_addresses() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))       # no packets are sent
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


class StreamServer:
    """backend (optional) is the remote's link to the app, with methods
    state(), command(action, args) -> dict, search(q) -> list,
    art(idx) -> bytes | None and wave(deck) -> dict | None. Without it the
    server is listen-only. pin=None disables the PIN check."""

    MAX_FAILS = 10          # wrong PINs per address ...
    FAIL_WINDOW = 60.0      # ... per minute, then a lock-out of the same length

    def __init__(self, engine, port: int = 8000, bitrate: int = 192000, status_fn=None, backend=None,
                 pin: str | None = None):
        self.engine = engine
        self.port = port
        self.bitrate = bitrate
        self.status_fn = status_fn or (lambda: {})
        self.backend = backend
        self.pin = pin
        self._urls: list[str] | None = None
        self._fails: dict[str, list[float]] = {}
        self._in: queue.Queue = queue.Queue(maxsize=200)
        self._clients: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._httpd = None
        self._threads: list[threading.Thread] = []
        self._running = False

    # engine tap (audio thread): must never block
    def write(self, block: np.ndarray):
        if not self._clients:
            return
        try:
            self._in.put_nowait(block.copy())
        except queue.Full:
            pass

    @property
    def listeners(self) -> int:
        return len(self._clients)

    @property
    def urls(self) -> list[str]:
        # looked up once: name resolution can block for seconds on some networks
        if self._urls is None:
            self._urls = [f"http://{ip}:{self.port}/" for ip in lan_addresses()] or [f"http://localhost:{self.port}/"]
        return self._urls

    def pair_url(self) -> str:
        """Address with the PIN (for the QR code: scanning it connects at once and a
        home-screen link made from it keeps connecting)."""
        u = self.urls[0]
        return u + (f"?pin={self.pin}" if self.pin else "")

    # ---- auth -------------------------------------------------------------------------
    def _locked_out(self, ip: str) -> bool:
        now = time.monotonic()
        fails = [t for t in self._fails.get(ip, []) if now - t < self.FAIL_WINDOW]
        self._fails[ip] = fails
        return len(fails) >= self.MAX_FAILS

    def check_key(self, ip: str, key: str | None) -> int:
        """200 ok, 401 wrong/missing PIN, 429 too many attempts."""
        if not self.pin:
            return 200
        if self._locked_out(ip):
            return 429
        if key is not None and hmac.compare_digest(str(key).encode(), str(self.pin).encode()):
            return 200
        if key:
            self._fails.setdefault(ip, []).append(time.monotonic())
        return 401

    def start(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", cache)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj, code: int = 200):
                self._send(code, json.dumps(obj, separators=(",", ":")).encode(), "application/json")

            def _authed(self, qs) -> bool:
                key = self.headers.get("X-CTA-Key") or (qs.get("k") or [None])[0]
                code = server.check_key(self.client_address[0], key)
                if code != 200:
                    self._json({"error": "PIN required" if code == 401 else "too many attempts"}, code)
                    return False
                return True

            def do_GET(self):
                u = urlsplit(self.path)
                path, qs = u.path, parse_qs(u.query)
                try:
                    if path == "/":
                        self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                    elif path in ICONS:
                        data = _icon(ICONS[path])
                        if data:
                            self._send(200, data, "image/png", cache="public, max-age=604800")
                        else:
                            self._json({"error": "not found"}, 404)
                    elif path == "/manifest.webmanifest":
                        self._send(200, MANIFEST.encode(), "application/manifest+json")
                    elif path == "/status":
                        st = dict(server.status_fn())
                        st["listeners"] = server.listeners
                        self._json(st)
                    elif path == "/live.mp3":
                        if self._authed(qs):
                            self._stream()
                    elif path.startswith("/api/") or path == "/art":
                        if not self._authed(qs):
                            return
                        be = server.backend
                        if be is None:
                            self._json({"error": "remote control is not available"}, 404)
                        elif path == "/api/state":
                            st = be.state()
                            st["listeners"] = server.listeners
                            self._json(st)
                        elif path == "/api/search":
                            self._json({"results": be.search((qs.get("q") or [""])[0])})
                        elif path == "/api/wave":
                            w = be.wave((qs.get("deck") or [""])[0])
                            self._json(w or {"error": "no track"}, 200 if w else 404)
                        elif path == "/art":
                            data = be.art(int((qs.get("i") or ["-1"])[0]))
                            if data:
                                ctype = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
                                self._send(200, data, ctype, cache="private, max-age=86400")
                            else:
                                self._send(404, b"", "text/plain")
                        else:
                            self._json({"error": "not found"}, 404)
                    else:
                        self._json({"error": "not found"}, 404)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:  # noqa: BLE001
                    log.exception("remote GET %s failed", path)
                    try:
                        self._json({"error": str(e)}, 500)
                    except OSError:
                        pass

            def do_POST(self):
                u = urlsplit(self.path)
                if u.path != "/api/cmd":
                    self._json({"error": "not found"}, 404)
                    return
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(min(n, 65536)) if n > 0 else b""
                if not self._authed(parse_qs(u.query)):
                    return
                if server.backend is None:
                    self._json({"error": "remote control is not available"}, 404)
                    return
                try:
                    args = json.loads(raw or b"{}")
                    if not isinstance(args, dict):
                        raise ValueError("expected a JSON object")
                    action = str(args.pop("action", ""))
                    res = server.backend.command(action, args) or {}
                    self._json({"ok": True, **res})
                except (ValueError, KeyError, TypeError) as e:
                    self._json({"error": str(e)}, 400)
                except Exception as e:  # noqa: BLE001
                    log.exception("remote command failed")
                    self._json({"error": str(e)}, 500)

            def _stream(self):
                q: queue.Queue = queue.Queue(maxsize=64)
                with server._lock:
                    server._clients.append(q)
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    while server._running:
                        try:
                            data = q.get(timeout=5)
                        except queue.Empty:
                            continue
                        self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with server._lock:
                        if q in server._clients:
                            server._clients.remove(q)

        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        self._httpd.daemon_threads = True
        self._running = True
        t1 = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="stream-http")
        t2 = threading.Thread(target=self._encode_loop, daemon=True, name="stream-encoder")
        self._threads = [t1, t2]
        t1.start()
        t2.start()
        if self not in self.engine.taps:
            self.engine.taps.append(self)
        log.info("mix server on %s", ", ".join(self.urls))

    def stop(self):
        self._running = False
        if self in self.engine.taps:
            self.engine.taps.remove(self)
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        try:
            self._in.put_nowait(None)
        except queue.Full:
            pass

    @property
    def running(self) -> bool:
        return self._running

    def _encode_loop(self):
        ctx = av.CodecContext.create("libmp3lame", "w")
        ctx.sample_rate = SR
        ctx.layout = "stereo"
        ctx.format = "fltp"
        ctx.bit_rate = self.bitrate
        ctx.open()
        fs = ctx.frame_size or 1152
        pending = np.zeros((2, 0), np.float32)
        while self._running:
            try:
                b = self._in.get(timeout=1)
            except queue.Empty:
                continue
            if b is None:
                break
            pending = np.concatenate([pending, b.astype(np.float32)], axis=1)
            out = bytearray()
            while pending.shape[1] >= fs:
                chunk = np.ascontiguousarray(np.clip(pending[:, :fs], -1, 1))
                pending = pending[:, fs:]
                fr = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo")
                fr.sample_rate = SR
                for pkt in ctx.encode(fr):
                    out += bytes(pkt)
            if out:
                with self._lock:
                    clients = list(self._clients)
                for q in clients:
                    try:
                        q.put_nowait(bytes(out))
                    except queue.Full:
                        pass                         # slow listener: skip, never block
