"""Library bridge: play the collection on the computer at home from anywhere.

The home Tempest **shares** its library: a small HTTPS server hands out the
catalogue (every track's tags, analysis - BPM, key, AI style, sound
vectors -, beat grid, cues and mix markers) and the audio files themselves,
to anyone holding the **pairing key**. The Tempest you take along **mirrors**
that catalogue into its own library - the tracks show up, can be searched,
suggested and queued like local ones - and downloads a file the moment a
deck or the auto-mix needs it (or ahead of time, for a whole setlist).

Security: a 32-byte random key, sent in a header; a self-signed TLS
certificate whose fingerprint is part of the pairing code, pinned by the
away machine (no certificate authority needed). Over the internet the home
machine needs its port reachable (a port forward on the router, or a tunnel
such as Tailscale, which also makes the address stable).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import json
import logging
import os
import secrets
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlsplit

from .db import TRACK_FIELDS, Database

log = logging.getLogger(__name__)

DEFAULT_PORT = 8008
PAGE = 500                     # tracks per catalogue page
CHUNK = 1 << 18                # file streaming chunk


# --------------------------------------------------------------------------
# pairing code
# --------------------------------------------------------------------------

def make_code(host: str, port: int, key: str, fingerprint: str | None) -> str:
    """TEMPEST-SHARE|host:port|key|fingerprint  (the away machine may edit the host)."""
    return f"TEMPEST-SHARE|{host}:{port}|{key}|{fingerprint or ''}"


def parse_code(code: str) -> dict:
    parts = [p.strip() for p in code.strip().split("|")]
    if len(parts) < 3 or parts[0] not in ("TEMPEST-SHARE", "CTA-SHARE"):
        raise ValueError("That is not a Tempest sharing code (it starts with TEMPEST-SHARE|)")
    hostport = parts[1]
    if hostport.startswith("["):                     # [ipv6]:port
        host, _, port = hostport[1:].partition("]:")
    else:
        host, _, port = hostport.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError("The address must be host:port")
    return {"host": host, "port": int(port), "key": parts[2], "fingerprint": parts[3] if len(parts) > 3 else ""}


# --------------------------------------------------------------------------
# TLS: a self-signed certificate, pinned by its fingerprint
# --------------------------------------------------------------------------

def ensure_cert(data_dir: str) -> tuple[str, str, str] | None:
    """(cert path, key path, SHA-256 fingerprint) - created once; None when
    the cryptography package is missing (plain HTTP then, key still required)."""
    cert_p, key_p = os.path.join(data_dir, "share-cert.pem"), os.path.join(data_dir, "share-key.pem")
    try:
        if not (os.path.exists(cert_p) and os.path.exists(key_p)):
            import datetime

            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.x509.oid import NameOID

            key = ec.generate_private_key(ec.SECP256R1())
            name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Tempest library share")])
            now = datetime.datetime.now(datetime.timezone.utc)
            cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                    .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
                    .not_valid_after(now + datetime.timedelta(days=3650)).sign(key, hashes.SHA256()))
            with open(key_p, "wb") as f:
                f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()))
            with open(cert_p, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            try:
                os.chmod(key_p, 0o600)
            except OSError:
                pass
        return cert_p, key_p, fingerprint_of(cert_p)
    except Exception as e:  # noqa: BLE001
        log.warning("no TLS for the library share (%s): plain HTTP, key still required", e)
        return None


def fingerprint_of(cert_pem_path: str) -> str:
    with open(cert_pem_path, "rb") as f:
        der = ssl.PEM_cert_to_DER_cert(f.read().decode())
    return hashlib.sha256(der).hexdigest()


# --------------------------------------------------------------------------
# the home side: serving the library
# --------------------------------------------------------------------------

def _row_payload(r) -> dict:
    d = {k: r[k] for k in TRACK_FIELDS if k not in ("path", "style_vec", "embedding")}
    d["id"] = r["id"]
    d["name"] = os.path.basename(r["path"])
    d["ext"] = os.path.splitext(r["path"])[1].lower()
    for k in ("style_vec", "embedding"):
        v = r[k]
        d[k] = base64.b64encode(v).decode() if v else None
    for k in ("first_beat", "grid_bpm", "grid_ver", "phrase_origin", "mix_in", "mix_out", "mix_out_end"):
        d[k] = r[k] if k in r.keys() else None
    return d


class ShareServer:
    """Serves the catalogue and the files to holders of the key."""

    MAX_FAILS, FAIL_WINDOW = 8, 600.0

    def __init__(self, db: Database, data_dir: str, port: int = DEFAULT_PORT, key: str | None = None,
                 name: str = "", tls: bool = True):
        self.db = db
        self.port = port
        self.key = key or secrets.token_urlsafe(32)
        self.name = name or socket.gethostname()
        self.library_id = self._library_id(data_dir)
        self.tls = ensure_cert(data_dir) if tls else None
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._fails: dict[str, list[float]] = {}
        self.served_bytes = 0
        self.requests = 0

    @staticmethod
    def _library_id(data_dir: str) -> str:
        p = os.path.join(data_dir, "share-id.txt")
        try:
            with open(p, encoding="utf-8") as f:
                v = f.read().strip()
            if v:
                return v
        except OSError:
            pass
        v = secrets.token_hex(8)
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(v)
        except OSError:
            pass
        return v

    @property
    def fingerprint(self) -> str | None:
        return self.tls[2] if self.tls else None

    def code(self, host: str) -> str:
        return make_code(host, self.port, self.key, self.fingerprint)

    def check_key(self, ip: str, key: str | None) -> int:
        now = time.monotonic()
        fails = [t for t in self._fails.get(ip, []) if now - t < self.FAIL_WINDOW]
        self._fails[ip] = fails
        if len(fails) >= self.MAX_FAILS:
            return 429
        if key is not None and hmac.compare_digest(str(key).encode(), self.key.encode()):
            return 200
        self._fails.setdefault(ip, []).append(now)
        return 401

    # ---- catalogue ---------------------------------------------------------------------
    def info(self) -> dict:
        from . import __version__

        return {"name": self.name, "library": self.library_id, "version": __version__, "count": self.db.count(),
                "tls": bool(self.tls)}

    def catalog(self, page: int, since: float = 0.0) -> dict:
        rows = self.db.conn().execute(
            "SELECT * FROM tracks WHERE missing=0 AND status='ok' AND COALESCE(local, 1)=1 "
            "AND COALESCE(analyzed_at, 0) >= ? ORDER BY id LIMIT ? OFFSET ?",
            (since, PAGE, page * PAGE)).fetchall()
        out = []
        for r in rows:
            d = _row_payload(r)
            try:
                d["cues"] = self.db.get_cues(r["path"])
            except Exception:  # noqa: BLE001
                d["cues"] = None
            out.append(d)
        return {"page": page, "tracks": out, "more": len(rows) == PAGE}

    def ids(self) -> list[int]:
        return [r[0] for r in self.db.conn().execute(
            "SELECT id FROM tracks WHERE missing=0 AND status='ok' AND COALESCE(local, 1)=1")]

    def path_of(self, track_id: int) -> str | None:
        r = self.db.conn().execute("SELECT path FROM tracks WHERE id=? AND missing=0 AND COALESCE(local, 1)=1",
                                   (track_id,)).fetchone()
        return r[0] if r else None

    # ---- server -------------------------------------------------------------------------
    def start(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, obj, code: int = 200):
                body = json.dumps(obj, separators=(",", ":")).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                server.requests += 1
                u = urlsplit(self.path)
                path, qs = u.path, parse_qs(u.query)
                try:
                    if path == "/share/ping":
                        self._json({"cta": "share", "library": server.library_id, "name": server.name})
                        return
                    code = server.check_key(self.client_address[0], self.headers.get("X-CTA-Key"))
                    if code != 200:
                        self._json({"error": "key required" if code == 401 else "too many attempts"}, code)
                        return
                    if path == "/share/info":
                        self._json(server.info())
                    elif path == "/share/catalog":
                        self._json(server.catalog(int((qs.get("page") or ["0"])[0]),
                                                  float((qs.get("since") or ["0"])[0])))
                    elif path == "/share/ids":
                        self._json({"ids": server.ids()})
                    elif path == "/share/file":
                        self._file(int((qs.get("id") or ["-1"])[0]))
                    else:
                        self._json({"error": "not found"}, 404)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    pass
                except Exception as e:  # noqa: BLE001
                    log.exception("share GET %s failed", path)
                    try:
                        self._json({"error": str(e)}, 500)
                    except OSError:
                        pass

            def _file(self, track_id: int):
                p = server.path_of(track_id)
                if p is None or not os.path.isfile(p):
                    self._json({"error": "no such track"}, 404)
                    return
                size = os.path.getsize(p)
                start, end = 0, size - 1
                rng = self.headers.get("Range")
                partial = False
                if rng and rng.startswith("bytes="):
                    a, _, b = rng[6:].partition("-")
                    start = int(a) if a else max(0, size - int(b))
                    end = int(b) if (b and a) else size - 1
                    partial = True
                    if start > end or start >= size:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                length = end - start + 1
                self.send_response(206 if partial else 200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("X-CTA-Size", str(size))
                self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(os.path.basename(p))}")
                if partial:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                with open(p, "rb") as f:
                    f.seek(start)
                    left = length
                    while left > 0:
                        chunk = f.read(min(CHUNK, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                        server.served_bytes += len(chunk)

        httpd = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        httpd.daemon_threads = True
        if self.tls:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(self.tls[0], self.tls[1])
            httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        self._httpd = httpd
        self.port = httpd.server_address[1]               # port 0 asked for a free one
        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="library-share")
        self._thread.start()
        log.info("library share on port %d (%s)", self.port, "https" if self.tls else "http")

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    @property
    def running(self) -> bool:
        return self._httpd is not None


# --------------------------------------------------------------------------
# the away side: a client for one shared library
# --------------------------------------------------------------------------

class ShareClient:
    """Talks to one home library: catalogue pages and file downloads (with
    resume), the certificate pinned by its fingerprint."""

    def __init__(self, host: str, port: int, key: str, fingerprint: str = "", timeout: float = 20.0):
        self.host, self.port, self.key, self.fingerprint = host, int(port), key, (fingerprint or "").lower()
        self.timeout = timeout
        self.tls = bool(self.fingerprint)

    def _conn(self, timeout: float | None = None):
        if self.tls:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE                    # pinned below instead of a CA
            c = http.client.HTTPSConnection(self.host, self.port, timeout=timeout or self.timeout, context=ctx)
            c.connect()
            der = c.sock.getpeercert(binary_form=True)
            if hashlib.sha256(der).hexdigest() != self.fingerprint:
                c.close()
                raise ssl.SSLError("the home library's certificate does not match the pairing code")
            return c
        c = http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)
        c.connect()
        return c

    def _get(self, path: str, timeout: float | None = None, headers: dict | None = None):
        c = self._conn(timeout)
        h = {"X-CTA-Key": self.key}
        h.update(headers or {})
        c.request("GET", path, headers=h)
        return c, c.getresponse()

    def _json(self, path: str) -> dict:
        c, r = self._get(path)
        try:
            body = r.read()
        finally:
            c.close()
        if r.status != 200:
            try:
                msg = json.loads(body).get("error", "")
            except ValueError:
                msg = body[:200].decode(errors="replace")
            raise ConnectionError(f"HTTP {r.status}: {msg}")
        return json.loads(body)

    def info(self) -> dict:
        return self._json("/share/info")

    def catalog(self, since: float = 0.0):
        """Every track, page by page (a generator of track dicts)."""
        page = 0
        while True:
            d = self._json(f"/share/catalog?page={page}&since={since}")
            yield from d["tracks"]
            if not d.get("more"):
                return
            page += 1

    def ids(self) -> set[int]:
        return set(self._json("/share/ids")["ids"])

    def download(self, track_id: int, dest: str, progress=None, stop: threading.Event | None = None) -> str:
        """Fetch one file to dest (resumes a partial .part); returns dest."""
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        part = dest + ".part"
        have = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        c, r = self._get(f"/share/file?id={track_id}", timeout=60.0, headers=headers)
        try:
            if r.status == 416:                                   # we already have it all
                r.read()
                os.replace(part, dest)
                return dest
            if r.status not in (200, 206):
                body = r.read()
                raise ConnectionError(f"HTTP {r.status}: {body[:200].decode(errors='replace')}")
            if r.status == 200:
                have = 0
            total = int(r.getheader("X-CTA-Size") or r.getheader("Content-Length") or 0) if r.status == 200 \
                else int(r.getheader("X-CTA-Size") or 0)
            with open(part, "ab" if r.status == 206 else "wb") as f:
                done = have
                while True:
                    if stop is not None and stop.is_set():
                        raise InterruptedError("download stopped")
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
        finally:
            c.close()
        os.replace(part, dest)
        return dest


# --------------------------------------------------------------------------
# mirroring a shared library into the local database
# --------------------------------------------------------------------------

def mirror_dir(data_dir: str, library_id: str) -> str:
    return os.path.join(data_dir, "remote", library_id)


def _local_path(data_dir: str, library_id: str, track: dict) -> str:
    name = track.get("name") or f"{track['id']}{track.get('ext') or ''}"
    name = "".join(ch for ch in name if ch not in '<>:"/\\|?*')
    return os.path.join(mirror_dir(data_dir, library_id), f"{track['id']}_{name}")


def mirror_catalog(db: Database, data_dir: str, library_id: str, tracks, progress=None) -> dict:
    """Write the home library's tracks into the local database as mirrored
    tracks (analysis, grid, cues, markers included). A track already
    downloaded keeps its file; one no longer at home is marked missing.
    Returns {"added": n, "updated": n, "removed": n}."""
    from .scanner import ANALYSIS_VERSION

    seen: set[str] = set()
    rows, remotes, grids, cues, marks = [], [], [], [], []
    n_new = n_upd = 0
    known = {r[0] for r in db.conn().execute("SELECT path FROM tracks WHERE remote IS NOT NULL")}
    for k, t in enumerate(tracks):
        path = _local_path(data_dir, library_id, t)
        seen.add(path)
        row = {f: t.get(f) for f in TRACK_FIELDS if f not in ("path", "style_vec", "embedding")}
        row["path"] = path
        row["status"] = "ok"
        row["error"] = None
        row["version"] = ANALYSIS_VERSION           # analysed at home: never re-analysed here
        for f in ("style_vec", "embedding"):
            v = t.get(f)
            row[f] = base64.b64decode(v) if v else None
        rows.append(row)
        remotes.append((json.dumps({"lib": library_id, "id": t["id"], "size": t.get("size")}),
                        int(os.path.isfile(path)), path))
        if path in known:
            n_upd += 1
        else:
            n_new += 1
        if t.get("first_beat") is not None and t.get("grid_bpm"):
            grids.append((t["first_beat"], t["grid_bpm"], t.get("grid_ver") or 1, t.get("phrase_origin") or 0, path))
        if t.get("cues"):
            cues.append((path, t["cues"]))
        marks.append((t.get("mix_in"), t.get("mix_out"), t.get("mix_out_end"), path))
        if progress is not None and k % 200 == 0:
            progress(k)
    db.upsert_many(rows)
    c = db.conn()
    with c:
        c.executemany("UPDATE tracks SET remote=?, local=? WHERE path=?", remotes)
        c.executemany("UPDATE tracks SET first_beat=?, grid_bpm=?, grid_ver=?, phrase_origin=? WHERE path=?", grids)
        c.executemany("UPDATE tracks SET mix_in=?, mix_out=?, mix_out_end=? WHERE path=?", marks)
    for path, cs in cues:
        try:
            db.set_cues(path, cs)
        except Exception:  # noqa: BLE001
            pass
    prefix = mirror_dir(data_dir, library_id) + os.sep
    gone = [p for p in known if p.startswith(prefix) and p not in seen]
    if gone:
        db.set_missing(gone, True)
    return {"added": n_new, "updated": n_upd, "removed": len(gone)}


def forget_library(db: Database, data_dir: str, library_id: str, delete_files: bool = False) -> int:
    """Drop a mirrored library from the database (and its downloads if asked)."""
    prefix = mirror_dir(data_dir, library_id) + os.sep
    c = db.conn()
    paths = [r[0] for r in c.execute("SELECT path FROM tracks WHERE path LIKE ?", (prefix + "%",))]
    with c:
        c.execute("DELETE FROM tracks WHERE path LIKE ?", (prefix + "%",))
        c.execute("DELETE FROM cues WHERE path LIKE ?", (prefix + "%",))
    if delete_files:
        for p in paths:
            for q in (p, p + ".part"):
                try:
                    os.remove(q)
                except OSError:
                    pass
    return len(paths)


class Downloader:
    """Fetches mirrored tracks in the background, one at a time, in the order
    asked; a deck that needs a file can wait for it."""

    def __init__(self, db: Database, client_of, data_dir: str):
        self.db = db
        self.client_of = client_of            # library id -> ShareClient
        self.data_dir = data_dir
        self._queue: list[str] = []            # paths, first = next
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._done: dict[str, threading.Event] = {}
        self._errors: dict[str, str] = {}
        self.progress: dict[str, tuple[int, int]] = {}   # path -> (done, total) while downloading
        self.on_change = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="library-download")
        self._thread.start()

    def close(self):
        self._stop.set()
        self._wake.set()

    def is_local(self, path: str) -> bool:
        return os.path.isfile(path)

    def request(self, path: str, first: bool = False) -> threading.Event:
        """Queue a download (or move it to the front); returns an event set when it is there."""
        with self._lock:
            ev = self._done.get(path)
            if ev is None:
                ev = self._done[path] = threading.Event()
                if self.is_local(path):
                    ev.set()
                    return ev
            if path in self._queue:
                if first:
                    self._queue.remove(path)
                    self._queue.insert(0, path)
            elif not ev.is_set():
                self._queue.insert(0, path) if first else self._queue.append(path)
            self._errors.pop(path, None)
        self._wake.set()
        return ev

    def ensure(self, path: str, timeout: float = 600.0) -> str:
        """Block until the file is here (a deck load, the auto-mix loader)."""
        ev = self.request(path, first=True)
        if not ev.wait(timeout):
            raise TimeoutError("the download did not finish in time")
        err = self._errors.get(path)
        if err:
            raise ConnectionError(err)
        return path

    def error(self, path: str) -> str | None:
        return self._errors.get(path)

    def queued(self) -> list[str]:
        with self._lock:
            return list(self._queue)

    def _remote_of(self, path: str) -> dict | None:
        r = self.db.conn().execute("SELECT remote FROM tracks WHERE path=?", (path,)).fetchone()
        try:
            return json.loads(r[0]) if r and r[0] else None
        except ValueError:
            return None

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                path = self._queue.pop(0) if self._queue else None
            if path is None:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            ev = self._done.get(path)
            try:
                rem = self._remote_of(path)
                if rem is None:
                    raise LookupError("not a mirrored track")
                client = self.client_of(rem["lib"])
                if client is None:
                    raise ConnectionError("the home library is not set up here any more")
                self.progress[path] = (0, int(rem.get("size") or 0))

                def prog(done, total, p=path):
                    self.progress[p] = (done, total)
                client.download(int(rem["id"]), path, progress=prog, stop=self._stop)
                c = self.db.conn()
                with c:
                    c.execute("UPDATE tracks SET local=1, mtime=?, size=? WHERE path=?",
                              (os.path.getmtime(path), os.path.getsize(path), path))
            except Exception as e:  # noqa: BLE001
                log.warning("download of %s failed: %s", os.path.basename(path), e)
                self._errors[path] = str(e)
            finally:
                self.progress.pop(path, None)
                if ev is not None:
                    ev.set()
                    with self._lock:
                        if self._errors.get(path):
                            self._done.pop(path, None)       # a later request tries again
                if self.on_change is not None:
                    try:
                        self.on_change(path)
                    except Exception:  # noqa: BLE001
                        pass
