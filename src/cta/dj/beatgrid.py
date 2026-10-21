"""Beat grid: tempo + the position of the first beat (and the bar phase).

The library stores the BPM; for beat/phase matching the engine also needs
*where* the beats fall. The grid is fitted on the decoded audio when a track
is loaded onto a deck (~0.1 s) and cached in the database.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import dsp

# window length minus hop of the onset STFT (1024-256 @ 22050 Hz); verified by
# tests on synthetic click tracks to within ~1 ms
ONSET_LEAD = (1024 - 256) / 22050


@dataclass
class BeatGrid:
    bpm: float
    first_beat: float          # seconds of a beat that starts a bar (downbeat)
    duration: float

    @property
    def beat_len(self) -> float:
        return 60.0 / self.bpm if self.bpm > 0 else 0.5

    def beat_at(self, t: float) -> float:
        """Fractional beat index at time t (beat 0 = first_beat, a downbeat)."""
        return (t - self.first_beat) / self.beat_len

    def time_of_beat(self, beat: float) -> float:
        return self.first_beat + beat * self.beat_len

    def phase(self, t: float) -> float:
        """Position inside the current beat, 0..1."""
        return self.beat_at(t) % 1.0

    @property
    def n_beats(self) -> int:
        return int(max(0.0, (self.duration - self.first_beat)) / self.beat_len)


def fit_grid(y: np.ndarray, sr: int, bpm: float) -> BeatGrid:
    """Find the beat phase (and the downbeat) for a known tempo.

    y: mono float32 audio at ``sr``.
    """
    dur = y.size / sr
    if bpm <= 0 or y.size < sr * 2:
        return BeatGrid(bpm=bpm if bpm > 0 else 120.0, first_beat=0.0, duration=dur)
    if sr != dsp.SR:
        # cheap decimation is fine for an onset envelope
        idx = np.arange(0, y.size, sr / dsp.SR).astype(np.int64)
        y = y[idx[idx < y.size]]
    full = dsp.onset_envelope(y)
    low = _low_band_envelope(y)
    m = min(full.size, low.size)
    # beats are where the kick (and snare) hit; off-beat hats must not win
    env = 2.0 * low[:m] + 0.5 * full[:m]
    fps = dsp.TEMPO_FPS
    period = 60.0 * fps / bpm

    # 1) beat phase: comb over the whole track, sub-frame resolution
    phases = np.linspace(0, period, 96, endpoint=False)
    n = int((env.size - 2) / period) - 1
    if n < 4:
        return BeatGrid(bpm=bpm, first_beat=0.0, duration=dur)
    k = np.arange(n) * period
    strength = np.empty(phases.size)
    for i, ph in enumerate(phases):
        pos = ph + k
        i0 = np.floor(pos).astype(int)
        frac = pos - i0
        strength[i] = np.mean(env[i0] * (1 - frac) + env[i0 + 1] * frac)
    best = phases[int(np.argmax(strength))]
    # STFT frames are left-aligned windows: an onset first shows up when it
    # enters the window, i.e. ONSET_LEAD seconds before it actually sounds
    beat0 = best / fps + ONSET_LEAD

    # 2) downbeat: which of the 4 beats in a bar carries the most low-end
    # (kick) energy at bar level -> beat 1
    bar_scores = np.zeros(4)
    nb = int((env.size / fps - beat0) / (60.0 / bpm))
    for b in range(nb):
        t = beat0 + b * 60.0 / bpm
        f = int(round(t * fps))
        if 0 <= f < low.size:
            bar_scores[b % 4] += low[max(0, f - 1):f + 2].max()
    down = int(np.argmax(bar_scores)) if bar_scores.max() > 0 else 0
    first = beat0 + down * 60.0 / bpm
    beat = 60.0 / bpm
    while first - 4 * beat >= -beat * 0.25:   # earliest downbeat in the track
        first -= 4 * beat
    return BeatGrid(bpm=bpm, first_beat=float(first), duration=dur)


def _low_band_envelope(y: np.ndarray) -> np.ndarray:
    mag = dsp._stft_mag(y, 1024, dsp.TEMPO_HOP)
    freqs = np.arange(mag.shape[1]) * dsp.SR / 1024
    low = np.log1p(100 * mag[:, (freqs > 30) & (freqs < 150)].sum(axis=1))
    d = np.maximum(0.0, np.diff(low, prepend=low[:1]))
    return d / (d.max() + 1e-9)


def energy_profile(y: np.ndarray, sr: int, grid: BeatGrid) -> np.ndarray:
    """RMS energy per bar (4 beats), used to find intros/outros."""
    bar = grid.beat_len * 4
    n = max(1, int((y.size / sr - max(grid.first_beat, 0)) / bar))
    out = np.zeros(n)
    for i in range(n):
        s = int((grid.first_beat + i * bar) * sr)
        e = int((grid.first_beat + (i + 1) * bar) * sr)
        seg = y[max(0, s):max(0, e)]
        out[i] = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if seg.size else 0.0
    return out


def beat_activity(bands: np.ndarray | None, grid: BeatGrid, hop: int = 256, sr: int = 44100):
    """Per-beat 'is there a beat here' profile from the waveform bands.

    Returns (energy per beat, index of the first real beat, index of the last
    real beat). A beat counts when its kick-band (or, for tracks without a
    kick, full-band) energy reaches 25 % of the track's typical beat level.
    """
    if bands is None or grid is None or grid.bpm <= 0:
        return np.zeros(0), 0, 0
    n_beats = max(1, int((grid.duration - grid.first_beat) / grid.beat_len))
    fps = sr / hop
    t = grid.first_beat + np.arange(n_beats) * grid.beat_len
    idx = np.clip(np.round(t * fps).astype(np.int64), 0, bands.shape[0] - 1)
    w = max(1, int(0.04 * fps))                 # +-40 ms around each beat
    low = np.array([bands[max(0, i - w):i + w + 1, 0].max() for i in idx])
    full = np.array([bands[max(0, i - w):i + w + 1].max() for i in idx])
    prof = low if np.median(low) > 0.1 * np.median(full) else full
    active = prof[prof > 0.02]
    ref = float(np.median(active)) if active.size else 0.0
    on = np.nonzero(prof >= 0.25 * ref)[0] if ref > 0 else np.array([0])
    first = int(on[0]) if on.size else 0
    last = int(on[-1]) if on.size else n_beats - 1
    return prof, first, last


def refine_downbeat(bands: np.ndarray | None, grid: BeatGrid, hop: int = 256, sr: int = 44100,
                    min_votes: int = 2) -> tuple[BeatGrid, int, float]:
    """Fix which beat is "beat 1" using the music's own structure.

    In dance music the kick drops out / comes back, breakdowns start and
    drops hit on the downbeat of a phrase. Every such change point votes for
    its beat position modulo 4 (the downbeat) - far more reliable than kick
    loudness, which is identical on all four beats of a 4/4 groove.

    Returns (grid with corrected first_beat, phrase origin in bars modulo 8,
    confidence 0..1).
    """
    if bands is None or grid is None or grid.bpm <= 0:
        return grid, 0, 0.0
    fps = sr / hop
    n_beats = int((grid.duration - grid.first_beat) / grid.beat_len)
    if n_beats < 64:
        return grid, 0, 0.0
    t = grid.first_beat + np.arange(n_beats) * grid.beat_len
    idx = np.clip(np.round(t * fps).astype(np.int64), 0, bands.shape[0] - 1)
    w = max(1, int(0.035 * fps))
    half = np.clip(np.round((t + grid.beat_len / 2) * fps).astype(np.int64), 0, bands.shape[0] - 1)
    kick = np.array([bands[max(0, i - w):i + w + 1, 0].max() - bands[max(0, j - w):j + w + 1, 0].mean()
                     for i, j in zip(idx, half)])
    energy = np.array([np.sqrt((bands[i:i + max(1, int(grid.beat_len * fps))] ** 2).mean()) for i in idx])
    bl = max(1, int(grid.beat_len * fps))
    low_e = np.array([np.sqrt((bands[i:i + bl, 0] ** 2).mean()) for i in idx])
    votes = np.zeros(4)
    phrase_votes = np.zeros(8)
    eps = 1e-4
    best_steps = []
    for e in (energy, low_e, np.maximum(kick, 0)):
        le = np.log(e + eps)
        step = np.abs(np.diff(le, prepend=le[0]))            # change from beat b-1 to beat b
        for bb in range(4, n_beats - 4):
            before = le[bb - 4:bb].mean()
            after = le[bb:bb + 4].mean()
            if abs(after - before) < np.log(1.35):
                continue
            if step[bb] < step[max(0, bb - 2):bb + 3].max() - 1e-12:
                continue                                       # not the sharpest beat of this change
            best_steps.append((bb, float(abs(after - before))))
    for bb, wgt in best_steps:
        votes[bb % 4] += wgt
    total = votes.sum()
    if total < min_votes * 0.4:
        return grid, 0, 0.0
    best = int(np.argmax(votes))
    conf = float(votes[best] / total)
    if conf < 0.45 or best == 0:
        g = grid
    else:
        g = BeatGrid(bpm=grid.bpm, first_beat=grid.first_beat + best * grid.beat_len, duration=grid.duration)
        while g.first_beat - 4 * g.beat_len >= -g.beat_len * 0.25:
            g.first_beat -= 4 * g.beat_len
    # phrase origin (bars mod 8) relative to the corrected grid: section
    # changes also land on the first bar of a phrase
    shift_bars = 0
    if best_steps and conf >= 0.45:
        for bb, wgt in best_steps:
            if bb % 4 == best:
                t_change = grid.first_beat + bb * grid.beat_len
                bar = int(round((t_change - g.first_beat) / (4 * g.beat_len)))
                phrase_votes[bar % 8] += wgt
        shift_bars = int(np.argmax(phrase_votes))
    return g, shift_bars, conf
