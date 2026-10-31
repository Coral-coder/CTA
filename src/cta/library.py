"""In-memory, vectorised view of the analysed library for instant queries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from .db import Database, blob_to_vec
from .keys import camelot_str, key_name
from .model import EMBED_DIM, N_STYLES

SIM_DIM = 128   # embeddings are PCA-reduced for fast similarity
STYLE_DIM = 64


@dataclass
class TrackView:
    idx: int
    id: int
    path: str
    title: str
    artist: str
    album: str
    bpm: float
    key: int
    energy: float
    genre: str
    tag_genre: str
    duration: float

    @property
    def key_label(self) -> str:
        return f"{camelot_str(self.key)} {key_name(self.key)}".strip()


def _norm_rows(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-9)


def _to_percentile(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    out = np.interp(np.nan_to_num(v, nan=0.0), q, np.linspace(0.0, 1.0, q.size))
    return np.where(np.isnan(v), 0.5, out)


def _dedup_key(artist: str, title: str) -> str:
    t = re.sub(r"\s*[\(\[].*?[\)\]]", "", (title or "").lower())   # drop (Original Mix) etc
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    a = re.sub(r"[^a-z0-9]+", " ", (artist or "").lower()).strip()
    return f"{a}|{t}"


class Library:
    def __init__(self, db: Database):
        self.db = db
        self.reload()

    def reload(self):
        all_rows = self.db.all_tracks()
        rows = [r for r in all_rows if r["status"] == "ok"]
        self.errors = [r for r in all_rows if r["status"] != "ok"]
        n = len(rows)
        self.n = n
        self.ids = np.array([r["id"] for r in rows], dtype=np.int64)
        self.id_to_idx = {int(i): k for k, i in enumerate(self.ids)}
        self.paths = [r["path"] for r in rows]
        self.titles = [r["title"] or "" for r in rows]
        self.artists = [r["artist"] or "" for r in rows]
        self.albums = [r["album"] or "" for r in rows]
        self.tag_genres = [r["tag_genre"] or "" for r in rows]
        self.genres = [r["genre"] or "" for r in rows]
        # library bridge: mirrored from a home library (remote JSON) - local once downloaded
        keys = rows[0].keys() if rows else ()
        self.remote = [(r["remote"] if "remote" in keys else None) or "" for r in rows]
        self.local = np.array([bool(r["local"]) if "local" in keys and r["local"] is not None else True
                               for r in rows], bool) if n else np.zeros(0, bool)
        self.styles = []
        self.parent_genres = []
        for r in rows:
            try:
                s = json.loads(r["styles"]) if r["styles"] else None
            except ValueError:
                s = None
            self.styles.append(s["top"] if s else [])
            self.parent_genres.append(s["parent"] if s else "")
        self.duration = np.array([r["duration"] or 0 for r in rows], dtype=np.float64)
        self.bpm_detected = np.array([r["bpm"] or 0 for r in rows], dtype=np.float64)
        ov = np.array([r["bpm_override"] if r["bpm_override"] is not None else np.nan for r in rows],
                      dtype=np.float64)
        self.bpm = np.where(np.isnan(ov), self.bpm_detected, ov) if n else self.bpm_detected
        self.bpm_overridden = ~np.isnan(ov) if n else np.zeros(0, bool)
        self.key_detected = np.array([r["key"] if r["key"] is not None else -1 for r in rows], dtype=np.int64)
        kov = np.array([r["key_override"] if r["key_override"] is not None else -99 for r in rows],
                       dtype=np.int64)
        self.key = np.where(kov != -99, kov, self.key_detected) if n else self.key_detected
        self.key_overridden = (kov != -99) if n else np.zeros(0, bool)
        self.key_conf = np.array([r["key_conf"] or 0 for r in rows])
        self.bpm_conf = np.array([r["bpm_conf"] or 0 for r in rows])
        self.tag_bpm = np.array([r["tag_bpm"] or 0 for r in rows], dtype=np.float64)
        self.tag_key = np.array([r["tag_key"] if r["tag_key"] is not None else -1 for r in rows])
        self.dedup = [_dedup_key(a, t) for a, t in zip(self.artists, self.titles)]
        self.artist_norm = [a.lower().strip() for a in self.artists]
        loud = np.array([r["loudness"] if r["loudness"] is not None else -20 for r in rows], dtype=np.float64)
        onset = np.array([r["onset_rate"] or 0 for r in rows], dtype=np.float64)
        cent = np.array([r["centroid"] or 0 for r in rows], dtype=np.float64)
        self.energy = self._energy(loud, onset, cent)

        emb = np.zeros((n, EMBED_DIM), np.float32)
        sv = np.zeros((n, N_STYLES), np.float32)
        has = np.zeros(n, bool)
        for i, r in enumerate(rows):
            e = blob_to_vec(r["embedding"], EMBED_DIM)
            s = blob_to_vec(r["style_vec"], N_STYLES)
            if e is not None and s is not None:
                emb[i] = e
                sv[i] = s
                has[i] = True
        self.has_ai = has
        self.emb = self._reduce(emb, has, SIM_DIM)
        self.style_vec = self._reduce(np.sqrt(np.maximum(sv, 0)), has, STYLE_DIM)   # Hellinger-style
        self._sim_cache: dict[int, np.ndarray] = {}
        self._style_cache: dict[int, np.ndarray] = {}
        self.sound_q = self._similarity_quantiles(self.emb, has)
        self.style_q = self._similarity_quantiles(self.style_vec, has)

        # integer codes make artist / duplicate / genre comparisons vectorised
        self.artist_code = self._codes(self.artist_norm, empty_unique=True)
        self.dedup_code = self._codes([d if d != "|" else "" for d in self.dedup], empty_unique=True)
        self.parent_code = self._codes(self.parent_genres, empty_unique=False)
        self.parent_names = sorted(set(self.parent_genres))
        # AI sub-genre (the top style) and its family: what blends with what
        from .styles import family_of

        # a style outside the dance map (hip hop, rock, ...) is its own family: its parent genre
        self.families = [family_of(g) or pg or "" for g, pg in zip(self.genres, self.parent_genres)]
        self.style_code = self._codes([g.lower() for g in self.genres], empty_unique=False)
        self.family_code = self._codes(self.families, empty_unique=False)
        counts: dict[str, int] = {}
        for g in self.genres:
            if g:
                counts[g] = counts.get(g, 0) + 1
        self.style_names = sorted(counts, key=lambda g: (-counts[g], g))
        self.style_counts = counts
        self._hay: list[str] | None = None
        try:
            stats = self.db.play_stats()
        except Exception:  # noqa: BLE001
            stats = {}
        self.play_count = np.array([stats.get(p, (0, 0))[0] for p in self.paths], dtype=np.int64)
        self.last_played = np.array([stats.get(p, (0, 0.0))[1] or 0.0 for p in self.paths], dtype=np.float64)
        self.session_played = np.zeros(self.n, bool)

    @staticmethod
    def _codes(values: list[str], empty_unique: bool) -> np.ndarray:
        """Integer code per value; empty strings get unique negative codes
        (never equal to anything) when ``empty_unique`` else -1."""
        table: dict[str, int] = {}
        out = np.empty(len(values), np.int64)
        for i, v in enumerate(values):
            if not v:
                out[i] = -(i + 2) if empty_unique else -1
            else:
                out[i] = table.setdefault(v, len(table))
        return out

    @staticmethod
    def _similarity_quantiles(m: np.ndarray, has: np.ndarray) -> np.ndarray:
        """Library-wide distribution of pairwise cosine similarity, used to
        turn raw cosine values into comparable 0..1 percentiles."""
        idx = np.nonzero(has)[0]
        if idx.size < 3:
            return np.linspace(-1, 1, 101)
        rng = np.random.default_rng(1)
        q_rows = idx if idx.size <= 64 else rng.choice(idx, 64, replace=False)
        others = idx if idx.size <= 20000 else rng.choice(idx, 20000, replace=False)
        sims = (m[others] @ m[q_rows].T).ravel()
        q = np.quantile(sims, np.linspace(0, 1, 101))
        return np.maximum.accumulate(q + np.arange(101) * 1e-9)

    @staticmethod
    def _energy(loud, onset, cent) -> np.ndarray:
        """1-10 energy rating relative to this library."""
        if loud.size == 0:
            return loud

        def z(x):
            sd = x.std()
            return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)

        comp = 0.5 * z(loud) + 0.3 * z(onset) + 0.2 * z(cent)
        ranks = comp.argsort().argsort() / max(comp.size - 1, 1)
        return np.round(1 + 9 * ranks, 1)

    @staticmethod
    def _reduce(x_all: np.ndarray, has: np.ndarray, dim: int) -> np.ndarray:
        """Mean-centred PCA projection to ``dim`` dims, rows L2-normalised."""
        n = x_all.shape[0]
        out = np.zeros((n, dim), np.float32)
        if has.sum() < 3:
            return out
        x = x_all[has].astype(np.float32)
        mu = x.mean(axis=0)
        sample = x if x.shape[0] <= 8000 else x[np.random.default_rng(0).choice(x.shape[0], 8000, False)]
        _, _, vt = np.linalg.svd(sample - mu, full_matrices=False)
        k = min(dim, vt.shape[0])
        out[has, :k] = _norm_rows((x - mu) @ vt[:k].T)
        return out

    # ------------------------------------------------------------------
    def track(self, idx: int) -> TrackView:
        return TrackView(idx, int(self.ids[idx]), self.paths[idx], self.titles[idx], self.artists[idx],
                         self.albums[idx], float(self.bpm[idx]), int(self.key[idx]),
                         float(self.energy[idx]), self.genres[idx], self.tag_genres[idx],
                         float(self.duration[idx]))

    def sound_similarity(self, idx: int) -> np.ndarray:
        """Cosine similarity of track idx to all tracks (cached)."""
        v = self._sim_cache.get(idx)
        if v is None:
            if not self.has_ai[idx]:
                v = np.full(self.n, np.nan, np.float32)
            else:
                v = self.emb @ self.emb[idx]
                v[~self.has_ai] = np.nan
            if len(self._sim_cache) > 4096:
                self._sim_cache.clear()
            self._sim_cache[idx] = v
        return v

    def style_similarity(self, idx: int) -> np.ndarray:
        v = self._style_cache.get(idx)
        if v is None:
            if not self.has_ai[idx]:
                v = np.full(self.n, np.nan, np.float32)
            else:
                v = self.style_vec @ self.style_vec[idx]
                v[~self.has_ai] = np.nan
            if len(self._style_cache) > 4096:
                self._style_cache.clear()
            self._style_cache[idx] = v
        return v

    def sound_score(self, idx: int) -> np.ndarray:
        """Sound similarity as a 0..1 library percentile (0.5 when unknown)."""
        return _to_percentile(self.sound_similarity(idx), self.sound_q)

    def style_score(self, idx: int) -> np.ndarray:
        return _to_percentile(self.style_similarity(idx), self.style_q)

    def blend_vector(self, idx: int) -> np.ndarray:
        """How well every track's style sits next to track idx's (styles.blend_score, vectorised)."""
        from .styles import NEIGHBOURS

        out = np.full(self.n, 0.5)
        fam = self.families[idx]
        if not fam:
            return out
        known = self.family_code >= 0
        same_parent = (self.parent_code == self.parent_code[idx]) & (self.parent_code >= 0)
        out[known] = np.where(same_parent[known], 0.2, 0.0)
        neigh = np.array([f in NEIGHBOURS.get(fam, ()) for f in self.families], bool)
        out[neigh] = 0.55
        out[self.family_code == self.family_code[idx]] = 0.85
        out[self.style_code == self.style_code[idx]] = 1.0
        return out

    def search(self, text: str) -> np.ndarray:
        if not text:
            return np.arange(self.n)
        terms = text.lower().split()
        if self._hay is None:
            self._hay = [f"{t} {a} {al} {g} {tg} {pg}".lower() for t, a, al, g, tg, pg in zip(
                self.titles, self.artists, self.albums, self.genres, self.tag_genres, self.parent_genres)]
        return np.array([i for i, h in enumerate(self._hay) if all(t in h for t in terms)], dtype=np.int64)
