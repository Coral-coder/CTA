"""Decode a track for a deck (stereo, 44.1 kHz, fully in RAM) and build its
beat grid and waveform overview."""

from __future__ import annotations

from dataclasses import dataclass

import av
import numpy as np

from .beatgrid import BeatGrid, beat_activity, fit_grid, refine_downbeat
from .engine import SR, TrackInfo


@dataclass
class LoadedTrack:
    audio: np.ndarray            # (2, n) float32
    grid: BeatGrid
    peaks: np.ndarray            # (W, 3) low/mid/high envelope for the overview
    info: TrackInfo
    bands: np.ndarray | None = None   # (frames, 3) per 256 samples for the zoomed view
    cues: list | None = None     # saved hot cues (source frames)
    mix_in: float | None = None  # user intro marker (seconds)
    mix_out: float | None = None # user outro marker (seconds)
    mix_out_end: float | None = None  # user outro end: the blend lasts from mix_out to here
    first_beat_t: float = 0.0    # first audible beat (seconds) -> default cue point
    phrase_origin: int = 0       # bar (mod 8) where the track's phrases start


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


def waveform_bands(audio: np.ndarray, hop: int = 256) -> np.ndarray:
    """Per-hop peak amplitude of the low / mid / high bands, (frames, 3),
    for frequency-coloured waveforms (red = bass, green = mids, blue = highs)."""
    mono = audio.mean(axis=0)
    n_fft = 1024
    if mono.size < n_fft:
        return np.zeros((1, 3), np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(mono, n_fft)[::hop]
    win = np.hanning(n_fft).astype(np.float32)
    out = np.empty((frames.shape[0], 3), np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    bands = [(freqs < 250), (freqs >= 250) & (freqs < 2500), (freqs >= 2500)]
    for s in range(0, frames.shape[0], 8192):
        mag = np.abs(np.fft.rfft(frames[s:s + 8192] * win, axis=1))
        for j, b in enumerate(bands):
            out[s:s + 8192, j] = np.sqrt((mag[:, b] ** 2).sum(axis=1))
    m = out.max(axis=0, keepdims=True)
    return out / np.maximum(m, 1e-9)


def load_track(path: str, bpm: float, info: TrackInfo, db=None) -> LoadedTrack:
    audio = decode_stereo(path)
    grid = None
    grid_ver, phrase_origin = 1, 0
    if db is not None and bpm > 0:
        cached = db.get_grid(path)
        if cached and cached[1] and abs(cached[1] - bpm) < 1e-3:
            grid = BeatGrid(bpm=bpm, first_beat=cached[0], duration=audio.shape[1] / SR)
            grid_ver, phrase_origin = cached[2], cached[3]
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
    cues = None
    if db is not None:
        try:
            cues = [None if c is None else c * SR for c in db.get_cues(path)]
        except Exception:  # noqa: BLE001
            cues = None
    mix_in = mix_out = mix_out_end = None
    if db is not None:
        try:
            mix_in, mix_out = db.get_markers(path)
            mix_out_end = db.get_out_end(path)
        except Exception:  # noqa: BLE001
            pass
    fine = waveform_bands(audio, hop=256)
    if grid_ver < 2 and bpm > 0:
        # correct "beat 1" from the track's section changes (once, then cached)
        grid, phrase_origin, conf = refine_downbeat(fine, grid)
        if db is not None:
            try:
                db.set_grid(path, grid.first_beat, bpm, version=2, phrase_origin=phrase_origin)
            except Exception:  # noqa: BLE001
                pass
    edges = np.linspace(0, fine.shape[0], min(2000, fine.shape[0]) + 1).astype(np.int64)
    over = np.maximum.reduceat(fine, edges[:-1], axis=0).astype(np.float32)
    _prof, first, _last = beat_activity(fine, grid)
    return LoadedTrack(audio=audio, grid=grid, peaks=over, info=info, bands=fine, cues=cues,
                       mix_in=mix_in, mix_out=mix_out, mix_out_end=mix_out_end,
                       first_beat_t=max(0.0, grid.time_of_beat(first)),
                       phrase_origin=phrase_origin)
