"""SQLite storage. One connection per thread; WAL mode for concurrent reads."""

from __future__ import annotations

import json
import sqlite3
import threading

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id            INTEGER PRIMARY KEY,
    path          TEXT NOT NULL UNIQUE,
    mtime         REAL,
    size          INTEGER,
    title         TEXT,
    artist        TEXT,
    album         TEXT,
    tag_genre     TEXT,
    tag_bpm       REAL,
    tag_key       INTEGER DEFAULT -1,
    duration      REAL,
    bpm           REAL,
    bpm_conf      REAL,
    key           INTEGER DEFAULT -1,
    key_conf      REAL,
    loudness      REAL,
    centroid      REAL,
    onset_rate    REAL,
    genre         TEXT,
    styles        TEXT,
    style_vec     BLOB,
    embedding     BLOB,
    bpm_override  REAL,
    key_override  INTEGER,
    status        TEXT NOT NULL DEFAULT 'ok',
    error         TEXT,
    analyzed_at   REAL,
    version       INTEGER,
    missing       INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS folders (
    path TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS cues (
    path   TEXT NOT NULL,
    idx    INTEGER NOT NULL,
    pos    REAL NOT NULL,
    PRIMARY KEY (path, idx)
);
CREATE TABLE IF NOT EXISTS history (
    id     INTEGER PRIMARY KEY,
    ts     REAL NOT NULL,
    path   TEXT NOT NULL,
    deck   TEXT,
    session TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);
"""

# columns added after the first release (applied on start-up)
MIGRATIONS = [
    ("first_beat", "REAL"),       # beat grid: seconds of the first downbeat
    ("grid_bpm", "REAL"),         # tempo the grid was fitted for
]

TRACK_FIELDS = [
    "path", "mtime", "size", "title", "artist", "album", "tag_genre", "tag_bpm", "tag_key",
    "duration", "bpm", "bpm_conf", "key", "key_conf", "loudness", "centroid", "onset_rate",
    "genre", "styles", "style_vec", "embedding", "status", "error", "analyzed_at", "version",
]


def vec_to_blob(v: np.ndarray | None) -> bytes | None:
    if v is None:
        return None
    return np.asarray(v, dtype=np.float16).tobytes()


def blob_to_vec(b: bytes | None, dim: int) -> np.ndarray | None:
    if not b or len(b) != dim * 2:
        return None
    return np.frombuffer(b, dtype=np.float16).astype(np.float32)


class Database:
    def __init__(self, path: str):
        self.path = path
        self._local = threading.local()
        c = self.conn()
        c.executescript(SCHEMA)
        cols = {r[1] for r in c.execute("PRAGMA table_info(tracks)")}
        for col, decl in MIGRATIONS:
            if col not in cols:
                c.execute(f"ALTER TABLE tracks ADD COLUMN {col} {decl}")
        c.commit()

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA temp_store=MEMORY")
            c.execute("PRAGMA cache_size=-200000")
            c.row_factory = sqlite3.Row
            self._local.conn = c
        return c

    def close(self):
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    # ---- folders ---------------------------------------------------------
    def folders(self) -> list[str]:
        return [r[0] for r in self.conn().execute("SELECT path FROM folders ORDER BY path")]

    def add_folder(self, path: str):
        c = self.conn()
        c.execute("INSERT OR IGNORE INTO folders(path) VALUES (?)", (path,))
        c.commit()

    def remove_folder(self, path: str):
        c = self.conn()
        c.execute("DELETE FROM folders WHERE path=?", (path,))
        c.commit()

    # ---- tracks ----------------------------------------------------------
    def known_files(self) -> dict[str, tuple[float, int, int, str, bool]]:
        """path -> (mtime, size, version, status, has_ai)"""
        out = {}
        for r in self.conn().execute(
                "SELECT path, mtime, size, version, status, embedding IS NOT NULL FROM tracks"):
            out[r[0]] = (r[1], r[2], r[3], r[4], bool(r[5]))
        return out

    def upsert_many(self, rows: list[dict]):
        if not rows:
            return
        cols = ",".join(TRACK_FIELDS)
        ph = ",".join("?" for _ in TRACK_FIELDS)
        upd = ",".join(f"{f}=excluded.{f}" for f in TRACK_FIELDS if f != "path")
        sql = (f"INSERT INTO tracks({cols},missing) VALUES ({ph},0) "
               f"ON CONFLICT(path) DO UPDATE SET {upd}, missing=0")
        c = self.conn()
        with c:
            c.executemany(sql, [tuple(r.get(f) for f in TRACK_FIELDS) for r in rows])

    def set_missing(self, paths: list[str], missing: bool = True):
        c = self.conn()
        with c:
            c.executemany("UPDATE tracks SET missing=? WHERE path=?", [(int(missing), p) for p in paths])

    def set_override(self, track_id: int, bpm: float | None = None, key: int | None = None,
                     clear_bpm: bool = False, clear_key: bool = False):
        c = self.conn()
        with c:
            if bpm is not None or clear_bpm:
                c.execute("UPDATE tracks SET bpm_override=? WHERE id=?", (bpm, track_id))
            if key is not None or clear_key:
                c.execute("UPDATE tracks SET key_override=? WHERE id=?", (key, track_id))

    def get_grid(self, path: str) -> tuple[float, float] | None:
        r = self.conn().execute("SELECT first_beat, grid_bpm FROM tracks WHERE path=?", (path,)).fetchone()
        return (r[0], r[1]) if r and r[0] is not None else None

    def set_grid(self, path: str, first_beat: float, bpm: float):
        c = self.conn()
        with c:
            c.execute("UPDATE tracks SET first_beat=?, grid_bpm=? WHERE path=?", (first_beat, bpm, path))

    def get_cues(self, path: str) -> list[float | None]:
        out: list[float | None] = [None] * 8
        for r in self.conn().execute("SELECT idx, pos FROM cues WHERE path=?", (path,)):
            if 0 <= r[0] < 8:
                out[r[0]] = r[1]
        return out

    def set_cues(self, path: str, cues: list[float | None]):
        c = self.conn()
        with c:
            c.execute("DELETE FROM cues WHERE path=?", (path,))
            c.executemany("INSERT INTO cues(path, idx, pos) VALUES (?,?,?)",
                          [(path, i, float(p)) for i, p in enumerate(cues) if p is not None])

    def add_history(self, ts: float, path: str, deck: str, session: str):
        c = self.conn()
        with c:
            c.execute("INSERT INTO history(ts, path, deck, session) VALUES (?,?,?,?)", (ts, path, deck, session))

    def history(self, session: str | None = None, limit: int = 2000):
        if session:
            q = "SELECT ts, path, deck, session FROM history WHERE session=? ORDER BY ts DESC LIMIT ?"
            return self.conn().execute(q, (session, limit)).fetchall()
        return self.conn().execute("SELECT ts, path, deck, session FROM history ORDER BY ts DESC LIMIT ?",
                                   (limit,)).fetchall()

    def purge_missing(self) -> int:
        c = self.conn()
        with c:
            return c.execute("DELETE FROM tracks WHERE missing=1").rowcount

    def all_tracks(self):
        return self.conn().execute("SELECT * FROM tracks WHERE missing=0 ORDER BY id").fetchall()

    def count(self) -> int:
        return self.conn().execute("SELECT COUNT(*) FROM tracks WHERE missing=0").fetchone()[0]

    def get_meta(self, k: str, default=None):
        r = self.conn().execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_meta(self, k: str, v):
        c = self.conn()
        with c:
            c.execute("INSERT OR REPLACE INTO meta(k,v) VALUES (?,?)", (k, json.dumps(v)))
