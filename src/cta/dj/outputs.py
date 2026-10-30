"""Output plumbing: a second sound card for the headphones, delay lines to
line outputs up, device rescans and latency calibration.

Bluetooth speakers and headphones play audio 150-300 ms after it was
rendered. CTA keeps one "delay" per output (what the driver reports, or
what the DJ measured with the tap test) and uses it to

* delay the faster output so master and headphones stay in time, and
* draw waveforms that late, so what you see matches what you hear.
"""

from __future__ import annotations

import re
import threading

import numpy as np

BLUETOOTH_HINTS = re.compile(r"bluetooth|airpods|beats|\bbt\b|bose|jbl|sony wh|sony wf|soundcore|"
                             r"hands-?free|a2dp|galaxy buds|pixel buds|marshall|ue boom|sonos", re.I)
BLUETOOTH_GUESS_MS = 200.0
TAP_ANTICIPATION_S = 0.02      # people tap a little ahead of a steady beat


def is_bluetooth(name: str) -> bool:
    return bool(BLUETOOTH_HINTS.search(name or ""))


def rescan_devices() -> None:
    """Make PortAudio see devices plugged in (or paired) since start-up.
    Only call it while no stream is open."""
    import sounddevice as sd

    try:
        sd._terminate()
    finally:
        sd._initialize()


class Ring:
    """Thread-safe stereo FIFO between the main audio callback and a second
    device's callback. The two sound cards run on their own clocks, so it
    keeps its fill near `target` frames: overfull -> skip ahead, empty ->
    silence (a tiny click at worst, never growing delay)."""

    def __init__(self, target: int = 2048, capacity: int = 32768):
        self.buf = np.zeros((capacity, 2), np.float32)
        self.cap = capacity
        self.target = target
        self.r = self.w = 0
        self.lock = threading.Lock()
        self.underruns = 0
        self.skips = 0

    @property
    def fill(self) -> int:
        return (self.w - self.r) % self.cap

    def write(self, x: np.ndarray):
        """x: (n, 2) float32."""
        n = len(x)
        with self.lock:
            if n >= self.cap:
                x = x[-(self.cap - 1):]
                n = len(x)
            end = self.w + n
            if end <= self.cap:
                self.buf[self.w:end] = x
            else:
                k = self.cap - self.w
                self.buf[self.w:] = x[:k]
                self.buf[:n - k] = x[k:]
            self.w = end % self.cap
            if self.fill > self.target * 3:              # the reader runs slow: drop to the target
                self.r = (self.w - self.target) % self.cap
                self.skips += 1

    def read(self, n: int) -> np.ndarray:
        out = np.zeros((n, 2), np.float32)
        with self.lock:
            avail = min(self.fill, n)
            if avail < n:
                self.underruns += 1
            end = self.r + avail
            if end <= self.cap:
                out[:avail] = self.buf[self.r:end]
            else:
                k = self.cap - self.r
                out[:k] = self.buf[self.r:]
                out[k:avail] = self.buf[:avail - k]
            self.r = end % self.cap
        return out


class Delay:
    """Fixed delay line for a (2, n) signal (up to `max_s` seconds)."""

    def __init__(self, sr: int, max_s: float = 1.0):
        self.cap = int(sr * max_s) + 1
        self.buf = np.zeros((2, self.cap), np.float32)
        self.pos = 0
        self.frames = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        d = int(min(max(self.frames, 0), self.cap - 1))
        if d == 0:
            return x
        n = x.shape[1]
        out = np.empty_like(x)
        for i in range(0, n, self.cap - d):          # works for any block size
            seg = x[:, i:i + self.cap - d]
            m = seg.shape[1]
            idx_w = (self.pos + np.arange(m)) % self.cap
            self.buf[:, idx_w] = seg
            idx_r = (self.pos - d + np.arange(m)) % self.cap
            out[:, i:i + m] = self.buf[:, idx_r]
            self.pos = (self.pos + m) % self.cap
        return out


def estimate_latency(taps: list[float], clicks: list[float], min_taps: int = 8) -> float | None:
    """Tap-along calibration: each tap is matched to the click it answers
    (the latest click before it, within one interval). Returns the output
    delay in seconds (median, first taps skipped while the DJ finds the
    beat), or None when the taps are too few or too scattered."""
    if len(clicks) < 2 or len(taps) < min_taps:
        return None
    c = np.asarray(sorted(clicks))
    period = float(np.median(np.diff(c)))
    offs = []
    for t in taps[3:]:
        j = np.searchsorted(c, t) - 1
        if j < 0:
            continue
        d = t - c[j]
        if 0 <= d < period:
            offs.append(d)
    if len(offs) < min_taps - 3:
        return None
    offs = np.asarray(offs)
    med = float(np.median(offs))
    if float(np.median(np.abs(offs - med))) > 0.06:        # all over the place: try again
        return None
    return max(0.0, med + TAP_ANTICIPATION_S)
