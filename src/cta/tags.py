"""Read basic metadata (title/artist/album/genre and any existing BPM/key tags)."""

from __future__ import annotations

import os

from .keys import parse_key


def _first(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if v is None:
        return None
    s = str(v).strip().replace("\x00", "")
    return s or None


def read_tags(path: str) -> dict:
    out = {"title": None, "artist": None, "album": None, "tag_genre": None,
           "tag_bpm": None, "tag_key": -1, "comment": None}
    try:
        import mutagen

        f = mutagen.File(path, easy=True)
        if f is not None and f.tags is not None:
            t = f.tags
            g = lambda k: _first(t.get(k)) if hasattr(t, "get") else None  # noqa: E731
            out["title"] = g("title")
            out["artist"] = g("artist") or g("albumartist")
            out["album"] = g("album")
            out["tag_genre"] = g("genre")
            bpm = g("bpm")
            try:
                out["tag_bpm"] = float(bpm) if bpm else None
            except ValueError:
                pass
        # key is not in the "easy" mappings for every format: read it raw
        f = mutagen.File(path)
        if f is not None and f.tags is not None:
            raw = None
            for k in ("TKEY", "initialkey", "INITIALKEY", "----:com.apple.iTunes:initialkey",
                      "----:com.apple.iTunes:KEY", "KEY", "key"):
                try:
                    v = f.tags.get(k)
                except Exception:  # noqa: BLE001
                    v = None
                if v is not None:
                    if hasattr(v, "text"):
                        v = v.text
                    if isinstance(v, (list, tuple)) and v and isinstance(v[0], bytes):
                        v = [v[0].decode("utf-8", "ignore")]
                    raw = _first(v)
                    if raw:
                        break
            out["tag_key"] = parse_key(raw)
    except Exception:  # noqa: BLE001 - broken tags must never fail a scan
        pass
    if not out["title"]:
        out["title"] = os.path.splitext(os.path.basename(path))[0]
    return out
