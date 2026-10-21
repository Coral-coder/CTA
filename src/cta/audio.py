"""Audio decoding through PyAV (bundled FFmpeg): every common format works
(mp3, flac, wav, aiff, m4a/aac/alac, ogg, opus, wma...)."""

from __future__ import annotations

from dataclasses import dataclass

import av
import numpy as np

from .dsp import MODEL_SR, SR

AUDIO_EXTENSIONS = {
    ".mp3", ".flac", ".wav", ".wave", ".aif", ".aiff", ".aifc", ".m4a", ".mp4", ".aac",
    ".alac", ".ogg", ".oga", ".opus", ".wma", ".wv", ".ape", ".mka",
}

# Long DJ mixes / podcasts: analysing the first 20 minutes is plenty.
MAX_SECONDS = 20 * 60


@dataclass
class Decoded:
    y: np.ndarray        # mono float32 @ SR
    y16: np.ndarray      # mono float32 @ MODEL_SR
    duration: float      # full file duration (s), from container when available


class DecodeError(Exception):
    pass


def _to_mono(frame: av.AudioFrame) -> np.ndarray:
    a = frame.to_ndarray()                 # planar float: (channels, n)
    if a.ndim == 2 and a.shape[0] > 1:
        return a.mean(axis=0, dtype=np.float32)
    return a.reshape(-1).astype(np.float32, copy=False)


def decode(path: str, max_seconds: float = MAX_SECONDS, want_model: bool = True) -> Decoded:
    """Decode once and resample to both analysis rates in the same pass."""
    try:
        container = av.open(path)
    except Exception as e:  # noqa: BLE001 - FFmpeg raises many types
        raise DecodeError(f"cannot open: {e}") from e
    try:
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            raise DecodeError("no audio stream")
        stream = streams[0]
        stream.thread_type = "AUTO"
        r_main = av.AudioResampler(format="fltp", layout=None, rate=SR, frame_size=1 << 16)
        r_model = (av.AudioResampler(format="fltp", layout=None, rate=MODEL_SR, frame_size=1 << 16)
                   if want_model else None)
        main, model = [], []
        got = 0
        limit = int(max_seconds * SR)
        try:
            for frame in container.decode(stream):
                frame.pts = None
                for f in r_main.resample(frame):
                    m = _to_mono(f)
                    main.append(m)
                    got += m.size
                if r_model is not None:
                    for f in r_model.resample(frame):
                        model.append(_to_mono(f))
                if got >= limit:
                    break
        except av.error.InvalidDataError:
            # truncated / partially corrupt file: keep what decoded fine
            if got < SR * 5:
                raise
        for f in r_main.resample(None):
            main.append(_to_mono(f))
        if r_model is not None:
            for f in r_model.resample(None):
                model.append(_to_mono(f))

        duration = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = container.duration / 1_000_000
    except DecodeError:
        raise
    except Exception as e:  # noqa: BLE001
        raise DecodeError(f"decode failed: {e}") from e
    finally:
        container.close()

    y = np.concatenate(main) if main else np.zeros(0, np.float32)
    y16 = np.concatenate(model) if model else np.zeros(0, np.float32)
    if y.size < SR:
        raise DecodeError("audio too short or empty")
    if not np.all(np.isfinite(y[:: max(1, y.size // 10000)])):
        y = np.nan_to_num(y)
        y16 = np.nan_to_num(y16)
    if duration <= 0 or duration < y.size / SR - 1:
        duration = y.size / SR
    return Decoded(y=y[:limit], y16=y16[: int(max_seconds * MODEL_SR)], duration=duration)
