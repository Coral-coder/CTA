"""Decode a track for a deck (stereo, 44.1 kHz, fully in RAM) and build its
beat grid and waveform overview."""

from __future__ import annotations

from dataclasses import dataclass

import av
import numpy as np

from .beatgrid import BeatGrid, fit_grid
from .engine import SR, TrackInfo


@dataclass
class LoadedTrack:
    audio: np.ndarray            # (2, n) float32
    grid: BeatGrid
    peaks: np.ndarray            # (W, 3) low/mid/high-ish envelope for drawing
    info: TrackInfo


def decode_stereo(path: str) -> np.ndarray:
    with av.open(path) as c:
        st = next(s for s in c.streams if s.type == "audio")
        st.thread_type = "AUTO"
        rs = av.AudioResampler(format="fltp", layout="stereo", rate=SR, frame_size=1 << 16)
        parts = []
        for fr in c.decode(st):
            fr.pts = None
            for o in rs.resample(fr):
                parts.append(o.to_ndarray())
        for o in rs.resample(None):
            parts.append(o.to_ndarray())
    if not parts:
        raise ValueError("no audio")
    a = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    return np.nan_to_num(a)


def waveform_peaks(audio: np.ndarray, width: int = 2000) -> np.ndarray:
    """Per-column peak of the full signal and of a crude low band (for a
    two-colour waveform)."""
    mono = audio.mean(axis=0)
    n = mono.size
    width = max(1, min(width, n))
    edges = np.linspace(0, n, width + 1).astype(np.int64)
    k = 64
    low = np.convolve(np.abs(mono), np.ones(k) / k, mode="same")  # smoothed: mostly bass
    full = np.maximum.reduceat(np.abs(mono), edges[:-1])
    lowp = np.maximum.reduceat(low, edges[:-1])
    m = max(full.max(), 1e-9)
    return np.stack([full / m, lowp / m], axis=1).astype(np.float32)


def load_track(path: str, bpm: float, info: TrackInfo, db=None) -> LoadedTrack:
    audio = decode_stereo(path)
    grid = None
    if db is not None and bpm > 0:
        cached = db.get_grid(path)
        if cached and cached[1] and abs(cached[1] - bpm) < 1e-3:
            grid = BeatGrid(bpm=bpm, first_beat=cached[0], duration=audio.shape[1] / SR)
    if grid is None:
        # analyse at most 4 minutes from the start for the grid: fast and the
        # phase does not drift with a fixed-tempo grid
        mono = audio[:, : SR * 240].mean(axis=0)
        grid = fit_grid(mono, SR, bpm if bpm > 0 else 120.0)
        grid.duration = audio.shape[1] / SR
        if db is not None and bpm > 0:
            try:
                db.set_grid(path, grid.first_beat, bpm)
            except Exception:  # noqa: BLE001
                pass
    return LoadedTrack(audio=audio, grid=grid, peaks=waveform_peaks(audio), info=info)
