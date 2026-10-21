"""Album art: embedded pictures (ID3 / MP4 / FLAC / Ogg / ASF) or a cover
image next to the file, scaled to small JPEG thumbnails and cached on disk.

Thread-safe; returns None when a track has no art.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import threading
from collections import OrderedDict

from . import paths

log = logging.getLogger(__name__)

COVER_NAMES = ("cover", "folder", "front", "album", "albumart", "albumartsmall")
COVER_EXTS = (".jpg", ".jpeg", ".png", ".webp")

_mem: OrderedDict = OrderedDict()
_lock = threading.Lock()
_MEM_MAX = 400


def _embedded(path: str) -> bytes | None:
    import mutagen

    f = mutagen.File(path)
    if f is None:
        return None
    pics = []
    # FLAC / some others: .pictures
    for p in getattr(f, "pictures", None) or []:
        pics.append((getattr(p, "type", 0), p.data))
    tags = f.tags
    if tags is not None:
        try:
            keys = list(tags.keys())
        except Exception:  # noqa: BLE001
            keys = []
        for k in keys:
            ks = str(k)
            try:
                v = tags[k]
            except Exception:  # noqa: BLE001
                continue
            if ks.startswith("APIC"):                            # ID3
                pics.append((getattr(v, "type", 0), v.data))
            elif ks == "covr":                                   # MP4
                for c in v:
                    pics.append((3, bytes(c)))
            elif ks.lower() == "metadata_block_picture":         # Ogg Vorbis / Opus
                from mutagen.flac import Picture

                for b64 in v:
                    try:
                        pic = Picture(base64.b64decode(b64))
                        pics.append((pic.type, pic.data))
                    except Exception:  # noqa: BLE001
                        pass
            elif ks == "WM/Picture":                             # ASF / WMA
                for item in v:
                    raw = bytes(getattr(item, "value", b""))
                    # type(1) + size(4) + mime utf16z + desc utf16z + data
                    try:
                        i = 5
                        for _ in range(2):
                            while raw[i:i + 2] != b"\x00\x00":
                                i += 2
                            i += 2
                        pics.append((raw[0], raw[i:]))
                    except Exception:  # noqa: BLE001
                        pass
    if not pics:
        return None
    pics.sort(key=lambda p: 0 if p[0] == 3 else 1)    # 3 = front cover
    return pics[0][1] or None


def _folder_cover(path: str) -> bytes | None:
    folder = os.path.dirname(path)
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    best = None
    for n in names:
        stem, ext = os.path.splitext(n.lower())
        if ext in COVER_EXTS and stem in COVER_NAMES:
            rank = COVER_NAMES.index(stem)
            if best is None or rank < best[0]:
                best = (rank, n)
    if best is None:
        return None
    try:
        with open(os.path.join(folder, best[1]), "rb") as fh:
            return fh.read()
    except OSError:
        return None


def raw_art(path: str) -> bytes | None:
    try:
        data = _embedded(path)
    except Exception:  # noqa: BLE001 - broken tags never break playback
        data = None
    return data or _folder_cover(path)


def _thumb(data: bytes, size: int) -> bytes:
    """Scale to a JPEG thumbnail with Qt when it is available."""
    try:
        from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
        from PySide6.QtGui import QImage

        img = QImage()
        if not img.loadFromData(data):
            return data
        if max(img.width(), img.height()) > size:
            img = img.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.WriteOnly)
        img.convertToFormat(QImage.Format_RGB888).save(buf, "JPEG", 86)
        buf.close()
        return bytes(ba.data()) or data
    except Exception:  # noqa: BLE001
        return data


def _cache_dir() -> str:
    d = os.path.join(paths.data_dir(), "art")
    os.makedirs(d, exist_ok=True)
    return d


def get_art(path: str, size: int = 300) -> bytes | None:
    """Thumbnail bytes (JPEG, or the original image if it cannot be scaled)."""
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = hashlib.sha1(f"{path}|{st.st_mtime_ns}|{st.st_size}|{size}".encode("utf-8", "replace")).hexdigest()
    with _lock:
        if key in _mem:
            _mem.move_to_end(key)
            return _mem[key]
    fn = os.path.join(_cache_dir(), key[:2], key + ".jpg")
    data = None
    if os.path.exists(fn):
        try:
            with open(fn, "rb") as fh:
                data = fh.read()
        except OSError:
            data = None
    elif os.path.exists(fn + ".none"):
        data = b""
    else:
        raw = raw_art(path)
        try:
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            if raw:
                data = _thumb(raw, size)
                with open(fn, "wb") as fh:
                    fh.write(data)
            else:
                data = b""
                open(fn + ".none", "wb").close()
        except OSError:
            data = _thumb(raw, size) if raw else b""
    with _lock:
        _mem[key] = data
        while len(_mem) > _MEM_MAX:
            _mem.popitem(last=False)
    return data or None
