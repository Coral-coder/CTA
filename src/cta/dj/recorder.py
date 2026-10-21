"""Record the master output to a 24-bit WAV file (written by a background
thread so the audio callback never touches the disk)."""

from __future__ import annotations

import queue
import threading
import wave

import numpy as np

from .engine import SR


class Recorder:
    def __init__(self, path: str):
        self.path = path
        self.q: queue.SimpleQueue = queue.SimpleQueue()
        self.frames = 0
        self._wav = wave.open(path, "wb")
        self._wav.setnchannels(2)
        self._wav.setsampwidth(3)
        self._wav.setframerate(SR)
        self._t = threading.Thread(target=self._run, daemon=True, name="recorder")
        self._t.start()

    def write(self, block: np.ndarray):
        """Called from the audio thread with (2, n) float32."""
        self.q.put(block.copy())
        self.frames += block.shape[1]

    @property
    def seconds(self) -> float:
        return self.frames / SR

    def _run(self):
        while True:
            b = self.q.get()
            if b is None:
                break
            x = np.clip(b.T, -1.0, 1.0)
            i = (x * 8388607.0).astype("<i4").reshape(-1)
            raw = i.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()   # little-endian 24-bit
            self._wav.writeframes(raw)
        self._wav.close()

    def close(self):
        self.q.put(None)
        self._t.join(timeout=10)
