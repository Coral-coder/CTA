"""Signal processing: tempo, key, loudness/energy features and model input.

Everything here is pure numpy so it runs identically on Windows, macOS and
Linux and packages cleanly into a frozen executable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 22050          # analysis sample rate for tempo / key / energy
MODEL_SR = 16000    # sample rate expected by the Discogs-EffNet model

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def frame_signal(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Return a strided (n_frames, frame) view; only full frames are kept."""
    if y.size < frame:
        y = np.pad(y, (0, frame - y.size))
    n = 1 + (y.size - frame) // hop
    return np.lib.stride_tricks.as_strided(
        y, shape=(n, frame), strides=(y.strides[0] * hop, y.strides[0]), writeable=False
    )


def _stft_mag(y: np.ndarray, n_fft: int, hop: int, block: int = 4096) -> np.ndarray:
    """Magnitude STFT, computed in blocks to bound peak memory."""
    frames = frame_signal(y, n_fft, hop)
    win = np.hanning(n_fft).astype(np.float32)
    out = np.empty((frames.shape[0], n_fft // 2 + 1), dtype=np.float32)
    for s in range(0, frames.shape[0], block):
        out[s:s + block] = np.abs(np.fft.rfft(frames[s:s + block] * win, axis=1))
    return out


# --------------------------------------------------------------------------
# mel filterbanks (Slaney style, matches Essentia's TensorflowInputMusiCNN)
# --------------------------------------------------------------------------

def _hz_to_slaney_mel(f):
    f = np.asarray(f, dtype=np.float64)
    fsp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / fsp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz,
                    min_log_mel + np.log(np.maximum(f, 1e-9) / min_log_hz) / logstep,
                    f / fsp)


def _slaney_mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    fsp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / fsp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), fsp * m)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular, unit-area ('unit_tri') Slaney mel filterbank: (n_mels, n_fft//2+1)."""
    mels = np.linspace(_hz_to_slaney_mel(fmin), _hz_to_slaney_mel(fmax), n_mels + 2)
    pts = _slaney_mel_to_hz(mels)
    bins = np.arange(n_fft // 2 + 1) * sr / n_fft
    fb = np.zeros((n_mels, bins.size), dtype=np.float64)
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (bins - lo) / (c - lo)
        down = (hi - bins) / (hi - c)
        fb[i] = np.maximum(0.0, np.minimum(up, down)) * (2.0 / (hi - lo))
    return fb.astype(np.float32)


_MODEL_FB = None


def model_melspectrogram(y16: np.ndarray) -> np.ndarray:
    """Log-mel frames (n, 96) identical to Essentia TensorflowInputMusiCNN.

    512-sample frames, hop 256, symmetric Hann, power spectrum, 96 Slaney mel
    bands up to 8 kHz, then log10(1 + 10000 * x).
    """
    global _MODEL_FB
    if _MODEL_FB is None:
        _MODEL_FB = mel_filterbank(MODEL_SR, 512, 96, 0.0, 8000.0)
    frames = frame_signal(y16.astype(np.float32, copy=False), 512, 256)
    win = np.hanning(512).astype(np.float32)
    out = np.empty((frames.shape[0], 96), dtype=np.float32)
    for s in range(0, frames.shape[0], 8192):
        spec = np.abs(np.fft.rfft(frames[s:s + 8192] * win, axis=1)) ** 2
        out[s:s + 8192] = np.log10(1.0 + 10000.0 * (spec @ _MODEL_FB.T))
    return out


def model_patches(y16: np.ndarray, max_patches: int = 32, patch: int = 128) -> np.ndarray:
    """Pick up to ``max_patches`` evenly spread 128x96 patches for the model.

    The first and last 5% of the track (silence, fades, DJ-tool intros) are
    skipped when the track is long enough.
    """
    mel = model_melspectrogram(y16)
    n = mel.shape[0]
    if n < patch:
        mel = np.pad(mel, ((0, patch - n), (0, 0)))
        n = patch
    lo, hi = 0, n - patch
    if n > patch * 8:
        lo, hi = int(n * 0.05), int(n * 0.95) - patch
    count = int(min(max_patches, max(1, (hi - lo) // 62 + 1)))
    starts = np.linspace(lo, hi, count).astype(int)
    return np.stack([mel[s:s + patch] for s in starts]).astype(np.float32)


# --------------------------------------------------------------------------
# tempo
# --------------------------------------------------------------------------

TEMPO_HOP = 256
TEMPO_FPS = SR / TEMPO_HOP


def onset_envelope(y: np.ndarray) -> np.ndarray:
    """Log-mel spectral flux onset strength at ~86 frames/second."""
    mag = _stft_mag(y, 1024, TEMPO_HOP)
    fb = mel_filterbank(SR, 1024, 64, 30.0, 11000.0)
    mel = np.log1p(100.0 * (mag @ fb.T))
    flux = np.maximum(0.0, np.diff(mel, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    # remove slowly varying trend (~1 s) so only transients remain
    k = int(TEMPO_FPS)
    if flux.size > k:
        trend = np.convolve(flux, np.ones(k) / k, mode="same")
        flux = np.maximum(0.0, flux - trend)
    m = flux.max()
    return (flux / m).astype(np.float64) if m > 0 else flux.astype(np.float64)


class _Spectrum:
    """Magnitude spectrum of the onset envelope, zero-padded so that it can be
    sampled at ~0.005 BPM resolution with linear interpolation."""

    def __init__(self, env: np.ndarray, fps: float):
        e = env - env.mean()
        nfft = max(1 << 20, 1 << int(np.ceil(np.log2(e.size * 4))))
        self.mag = np.abs(np.fft.rfft(e, nfft)) / max(e.size, 1)
        self.hz_per_bin = fps / nfft

    def at(self, freqs_hz: np.ndarray) -> np.ndarray:
        idx = freqs_hz / self.hz_per_bin
        return np.interp(idx, np.arange(self.mag.size), self.mag)


def _harmonic_strength(spec: _Spectrum, bpms: np.ndarray) -> np.ndarray:
    """Pulse-train salience of each tempo: fundamental plus 2x/4x harmonics."""
    f = bpms / 60.0
    return spec.at(f) + 0.5 * spec.at(2 * f) + 0.25 * spec.at(4 * f)


def _alternation(env: np.ndarray, bpm: float, fps: float) -> float:
    """Ratio between the stronger and weaker of alternating beats at ``bpm``."""
    period = 60.0 * fps / bpm
    if env.size < period * 8:
        return 1.0
    # local max so slightly early/late onsets still count
    pad = np.pad(env, 2)
    m = np.max(np.stack([pad[i:i + env.size] for i in range(5)]), axis=0)
    n = int((env.size - 1) / period) - 1
    k = np.arange(n)
    best, best_pos = -1.0, None
    for phase in np.linspace(0, period, 16, endpoint=False):
        pos = np.round(phase + k * period).astype(int)
        v = m[pos].mean()
        if v > best:
            best, best_pos = v, pos
    vals = m[best_pos]
    even, odd = vals[0::2].mean(), vals[1::2].mean()
    return float(max(even, odd) / max(min(even, odd), 1e-9))


@dataclass
class TempoResult:
    bpm: float
    confidence: float


def estimate_tempo(y: np.ndarray, min_bpm: float = 70.0, max_bpm: float = 180.0,
                   env: np.ndarray | None = None) -> TempoResult:
    """Estimate a single global tempo.

    1. autocorrelation of the onset envelope gives periodicity salience,
    2. a log-normal prior around 122 BPM plus the allowed DJ range resolves
       octave ambiguity,
    3. a harmonic DFT search refines the result to ~0.01 BPM.
    """
    if env is None:
        env = onset_envelope(y)
    fps = TEMPO_FPS
    if env.size < fps * 4 or env.max() <= 0:
        return TempoResult(0.0, 0.0)

    # analyse at most ~6 minutes from the middle of the track: plenty for a
    # stable estimate and keeps very long files fast
    max_len = int(fps * 360)
    if env.size > max_len:
        s = (env.size - max_len) // 2
        env = env[s:s + max_len]

    e = env - env.mean()
    nfft = 1 << int(np.ceil(np.log2(2 * e.size)))
    spec = np.fft.rfft(e, nfft)
    ac = np.fft.irfft(spec * np.conj(spec), nfft)[: e.size]
    if ac[0] <= 0:
        return TempoResult(0.0, 0.0)
    ac = ac / ac[0]

    # coarse candidate grid (0.5 BPM) between 40 and 240 BPM
    grid = np.arange(40.0, 240.0, 0.5)
    lags = 60.0 * fps / grid
    ac_val = np.interp(lags, np.arange(ac.size), ac)
    # metrical support: a real beat period also has energy at 2x and 4x the lag
    ac2 = np.interp(np.minimum(lags * 2, ac.size - 1), np.arange(ac.size), ac)
    sal = np.maximum(ac_val, 0) + 0.5 * np.maximum(ac2, 0)
    spec_env = _Spectrum(env, fps)
    sal_dft = _harmonic_strength(spec_env, grid)
    sal = sal / (sal.max() + 1e-12) + sal_dft / (sal_dft.max() + 1e-12)

    prior = np.exp(-0.5 * (np.log2(grid / 122.0) / 0.9) ** 2)
    in_range = (grid >= min_bpm) & (grid <= max_bpm)
    weighted = sal * prior * np.where(in_range, 1.0, 0.05)
    best = grid[int(np.argmax(weighted))]

    # explicit octave check: compare best with its half/double inside the range
    cands = [c for c in (best, best / 2, best * 2, best * 1.5, best / 1.5) if min_bpm <= c <= max_bpm]
    if cands:
        def score(c):
            i = int(np.argmin(np.abs(grid - c)))
            lo, hi = max(0, i - 2), i + 3
            v = sal[lo:hi].max() * np.exp(-0.5 * (np.log2(c / 122.0) / 0.9) ** 2)
            if any(abs(o - 2 * c) < 1 for o in cands):
                # strong/weak alternation at double tempo means the faster
                # pulse is a subdivision (e.g. kick + off-beat hats)
                alt = _alternation(env, 2 * c, fps)
                if alt > 1.2:
                    v *= 1.0 + min(0.35, 0.35 * (alt - 1.2) / 0.8)
            return v
        best = max(cands, key=score)

    # fine refinement +-1.5 BPM at 0.005 resolution on the harmonic DFT
    fine = np.arange(best - 1.5, best + 1.5, 0.005)
    fine = fine[fine > 1]
    fs = _harmonic_strength(spec_env, fine)
    bpm = float(fine[int(np.argmax(fs))])

    # confidence: peak prominence of the salience function
    srt = np.sort(sal)
    conf = float(np.clip((sal.max() - np.median(srt)) / (sal.max() + 1e-12), 0, 1))
    return TempoResult(round_bpm(bpm), conf)


def round_bpm(bpm: float) -> float:
    """Snap to the integer when very close (most dance music is produced on
    an integer grid), otherwise keep two decimals."""
    r = round(bpm)
    if abs(bpm - r) < 0.06:
        return float(r)
    return round(bpm, 2)


# --------------------------------------------------------------------------
# key
# --------------------------------------------------------------------------

# Shaath (KeyFinder) profiles, tuned on electronic / popular music.
MAJOR_PROFILE = np.array([7.23900502618145, 3.50351166725158, 3.58445177536649, 2.84511816478676,
                          5.81898892118006, 4.55865057415532, 2.44778850545506, 6.99473192146152,
                          3.39106842095157, 4.55614256274638, 4.07419990108264, 4.45130343666788])
MINOR_PROFILE = np.array([7.00255045060127, 3.14360928092613, 4.35904801316005, 5.40418360993528,
                          3.67234319908218, 4.08971943474578, 3.90874105101362, 6.20023281082467,
                          3.63390832183476, 2.87580787045416, 5.39919018745637, 3.67096101616418])

KEY_NFFT = 4096
KEY_HOP = 4096
KEY_FMIN, KEY_FMAX = 25.0, 3500.0
KEY_PEAKS = 60
KEY_HARMONICS = 4


def _estimate_tuning(spec: np.ndarray) -> float:
    """Tuning offset in cents (-50..50) from magnitude-weighted spectral peaks."""
    freqs = np.arange(spec.size) * SR / KEY_NFFT
    sel = (freqs > 100) & (freqs < 2000)
    if not np.any(spec[sel] > 0):
        return 0.0
    s = spec.copy()
    s[~sel] = 0
    peaks = (s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]) & (s[1:-1] > np.percentile(s[sel], 90))
    idx = np.nonzero(peaks)[0] + 1
    if idx.size < 8:
        return 0.0
    a, b, c = s[idx - 1], s[idx], s[idx + 1]
    denom = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / denom, 0.0)
    f = (idx + off) * SR / KEY_NFFT
    cents = 1200.0 * np.log2(f / 440.0)
    ang = (((cents + 50.0) % 100.0) - 50.0) / 100.0 * 2 * np.pi
    mean = np.arctan2((b * np.sin(ang)).sum(), (b * np.cos(ang)).sum())
    return float(mean / (2 * np.pi) * 100.0)


@dataclass
class KeyResult:
    key: int          # 0-11 major (pitch class of tonic), 12-23 minor; -1 unknown
    confidence: float


def chroma_profile(y: np.ndarray) -> np.ndarray:
    """Harmonic pitch class profile (HPCP-style).

    Per frame: the strongest spectral peaks (parabolically interpolated) and
    their first sub-harmonics vote for pitch classes with a cos^2 window of
    one semitone; frames are max-normalised and summed.
    """
    mag = _stft_mag(y, KEY_NFFT, KEY_HOP, block=1024)
    if mag.shape[0] == 0 or mag.max() <= 0:
        return np.zeros(12)
    nb = mag.shape[1]
    freqs = np.arange(nb) * SR / KEY_NFFT
    tuning = _estimate_tuning(np.log1p(mag / mag.max() * 1000.0).sum(axis=0))

    top = int(np.searchsorted(freqs, KEY_FMAX)) + 2
    mag = mag[:, :top]
    nb = top
    m = mag.copy()
    m[:, freqs[:top] < KEY_FMIN] = 0
    m[:, freqs[:top] > KEY_FMAX] = 0
    lm = np.zeros_like(m, dtype=bool)
    lm[:, 1:-1] = (m[:, 1:-1] > m[:, :-2]) & (m[:, 1:-1] >= m[:, 2:])
    lm &= m > m.max(axis=1, keepdims=True) * 1e-3
    vals = np.where(lm, m, 0.0)
    k = min(KEY_PEAKS, nb - 1)
    idx = np.argpartition(-vals, k, axis=1)[:, :k]
    pm = np.take_along_axis(vals, idx, 1)
    a = np.take_along_axis(mag, np.maximum(idx - 1, 0), 1)
    c = np.take_along_axis(mag, np.minimum(idx + 1, nb - 1), 1)
    den = a - 2 * pm + c
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    pf = (idx + off) * SR / KEY_NFFT
    valid = pm > 0

    n_frames = mag.shape[0]
    out = np.zeros(n_frames * 12)
    frame_idx = np.repeat(np.arange(n_frames), k)
    for h in range(1, KEY_HARMONICS + 1):
        f0 = pf / h
        semi = 12.0 * np.log2(np.maximum(f0, 1e-6) / 440.0) + 69.0 - tuning / 100.0
        nearest = np.rint(semi)
        # a one-semitone cos^2 window only ever touches the nearest pitch class
        win = np.cos(np.pi * (semi - nearest)) ** 2
        w = pm * (0.6 ** (h - 1)) * (valid & (f0 >= KEY_FMIN)) * win
        pc = nearest.astype(np.int64) % 12
        out += np.bincount((frame_idx * 12 + pc.ravel()), weights=w.ravel(), minlength=n_frames * 12)
    out = out.reshape(n_frames, 12)
    mx = out.max(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(mx > 0, out / mx, 0.0)
    return out.sum(axis=0)


def estimate_key(y: np.ndarray) -> KeyResult:
    pcp = chroma_profile(y)
    if not np.any(pcp > 0):
        return KeyResult(-1, 0.0)
    pcp = pcp - pcp.mean()
    scores = np.empty(24)
    for mode, prof in ((0, MAJOR_PROFILE), (1, MINOR_PROFILE)):
        p = prof - prof.mean()
        for tonic in range(12):
            r = np.roll(p, tonic)
            scores[mode * 12 + tonic] = float(pcp @ r / (np.linalg.norm(pcp) * np.linalg.norm(r) + 1e-12))
    order = np.argsort(scores)[::-1]
    conf = float(np.clip(scores[order[0]] - scores[order[1]], 0, 1) * 5)
    return KeyResult(int(order[0]), min(conf, 1.0))


# --------------------------------------------------------------------------
# loudness / energy descriptors
# --------------------------------------------------------------------------

def energy_features(y: np.ndarray, env: np.ndarray | None = None) -> dict:
    frames = frame_signal(y, 2048, 1024)
    ms = (frames.astype(np.float64) ** 2).mean(axis=1)
    db = 10 * np.log10(ms + 1e-12)
    gated = db[db > max(-70.0, db.max() - 30.0)] if db.size else db
    loudness = float(10 * np.log10(np.mean(10 ** (gated / 10)) + 1e-12)) if gated.size else -70.0

    mag = _stft_mag(y, 2048, 2048)
    freqs = np.arange(mag.shape[1]) * SR / 2048
    tot = mag.sum(axis=1)
    keep = tot > tot.max() * 1e-3 if tot.size else tot
    centroid = float(((mag[keep] @ freqs) / tot[keep]).mean()) if np.any(keep) else 0.0

    onset_rate = 0.0
    if env is not None and env.size > 3:
        thr = np.percentile(env, 90)
        peaks = (env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > max(thr, 0.05))
        onset_rate = float(peaks.sum() / (env.size / TEMPO_FPS))
    return {"loudness": loudness, "centroid": centroid, "onset_rate": onset_rate}
