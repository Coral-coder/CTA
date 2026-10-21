"""Live stream of the master output to phones/tablets over the network.

A tiny HTTP server (standard library) serves:

  /            a mobile-friendly player page with now-playing info
  /live.mp3    the live mix as an endless MP3 stream (192 kbit/s)
  /status      JSON: decks, BPM, keys, auto-mix status

Encoding runs in its own thread; every listener gets its own bounded queue
so a slow phone can never stall the audio engine or other listeners.
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import av
import numpy as np

from .engine import SR

log = logging.getLogger(__name__)

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CTA live</title><style>
body{font-family:system-ui,sans-serif;background:#16171a;color:#e6e6ea;margin:0;padding:20px}
h1{font-size:20px;margin:0 0 12px}.deck{background:#24252a;border-radius:10px;padding:12px;margin:10px 0}
.deck.on{border-left:4px solid #4caf50}.t{font-weight:600}.m{color:#9a9aa2;font-size:14px}
button{font-size:18px;padding:14px 22px;border-radius:10px;border:0;background:#3478dc;color:#fff;width:100%}
#st{color:#9a9aa2;margin:12px 0}</style></head><body>
<h1>CTA live mix</h1>
<button id="b" onclick="go()">▶ Listen</button>
<audio id="a" preload="none"></audio>
<div id="st">Tap Listen to start the stream.</div><div id="d"></div>
<script>
function go(){var a=document.getElementById('a');
 if(!a.paused){a.pause();a.removeAttribute('src');a.load();b.textContent='▶ Listen';return}
 a.src='/live.mp3?'+Date.now();a.play();b.textContent='■ Stop'}
async function poll(){try{var r=await fetch('/status');var s=await r.json();
 document.getElementById('st').textContent=s.automix+' · '+s.listeners+' listening';
 document.getElementById('d').innerHTML=s.decks.map(d=>'<div class="deck '+(d.playing?'on':'')+'"><div class="t">'+
 d.deck+': '+(d.title||'—')+'</div><div class="m">'+(d.artist||'')+' · '+d.bpm+' BPM · '+d.key+
 (d.playing?' · playing':'')+'</div></div>').join('')}catch(e){}}
setInterval(poll,2000);poll();</script></body></html>"""


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
    def __init__(self, engine, port: int = 8000, bitrate: int = 192000, status_fn=None):
        self.engine = engine
        self.port = port
        self.bitrate = bitrate
        self.status_fn = status_fn or (lambda: {})
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
        return [f"http://{ip}:{self.port}/" for ip in lan_addresses()] or [f"http://localhost:{self.port}/"]

    def start(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    body = PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/status":
                    st = dict(server.status_fn())
                    st["listeners"] = server.listeners
                    body = json.dumps(st).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/live.mp3":
                    q: queue.Queue = queue.Queue(maxsize=64)
                    with server._lock:
                        server._clients.append(q)
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/mpeg")
                    self.send_header("Cache-Control", "no-cache, no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
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
                else:
                    self.send_error(404)

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
        log.info("live stream on %s", ", ".join(self.urls))

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
