"""The phone remote's link into the running app (see dj.stream).

Reads (state, search, waveforms, art) run on the HTTP threads and only read;
every command is executed on the GUI thread and waited for.
"""

from __future__ import annotations

import logging
import secrets
import threading

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal, Slot

from ..dj.engine import SR
from ..keys import camelot_str, key_name

log = logging.getLogger(__name__)

BLEND_CHOICES = (4, 8, 16, 32)


def new_pin() -> str:
    return f"{secrets.randbelow(10 ** 6):06d}"


class _Invoker(QObject):
    """Runs callables on the thread that owns it (the GUI thread)."""

    _sig = Signal(object)

    def __init__(self):
        super().__init__()
        self._sig.connect(self._run, Qt.QueuedConnection)

    @Slot(object)
    def _run(self, job):
        fn, box, done = job
        try:
            box["result"] = fn()
        except Exception as e:  # noqa: BLE001
            box["error"] = e
        done.set()

    def call(self, fn, timeout: float = 10.0):
        if threading.current_thread() is threading.main_thread():
            return fn()
        box: dict = {}
        done = threading.Event()
        self._sig.emit((fn, box, done))
        if not done.wait(timeout):
            raise TimeoutError("CTA is busy - try again")
        if "error" in box:
            raise box["error"]
        return box.get("result")


class RemoteBackend:
    def __init__(self, dj):
        self.dj = dj
        self.eng = dj.engine
        self.am = dj.automix
        self._inv = _Invoker()
        self._hay: tuple[int, list[str]] | None = None

    # ---- helpers -----------------------------------------------------------------------
    @property
    def lib(self):
        return self.dj.main.lib

    def _item(self, idx: int, pos: int | None = None) -> dict:
        lib = self.lib
        if lib is None or not (0 <= idx < lib.n):
            return {"idx": idx, "title": "?", "artist": "", "bpm": 0, "key": "", "art": None, "played": False}
        k = int(lib.key[idx])
        d = {"idx": int(idx), "title": lib.titles[idx], "artist": lib.artists[idx],
             "bpm": round(float(lib.bpm[idx]), 1),
             "key": f"{camelot_str(k)} {key_name(k)}".strip() if k >= 0 else "",
             "art": f"/art?i={int(idx)}", "played": bool(lib.play_count[idx] > 0 or lib.session_played[idx])}
        if pos is not None:
            d["pos"] = pos
        return d

    # ---- reads (HTTP threads) ----------------------------------------------------------
    def state(self) -> dict:
        e, am = self.eng, self.am
        # while paused the decks are stopped underneath: still show them as on air
        held = set(e._paused_decks) if e.paused else set()
        on_air = {n for n, d in e.decks.items() if d.playing} | held
        m = e._master()
        if m is None and held:
            m = e.decks[e.master_deck] if e.master_deck in held else e.decks[sorted(held)[0]]
        tr = am.tr
        nxt = {d for d in am.meta if d != (m.name if m else None) and d not in on_air}
        decks = []
        for n, d in e.decks.items():
            if not d.loaded and n in "CD":
                continue
            ek = d.effective_key if d.loaded else -1
            decks.append({
                "deck": n, "loaded": d.loaded, "title": d.info.title, "artist": d.info.artist,
                "idx": d.info.lib_index, "wid": d.info.path and abs(hash(d.info.path)) % 10 ** 9,
                "playing": n in on_air, "master": m is not None and m.name == n,
                "next": n in nxt or (tr is not None and tr.in_deck == n),
                "bpm": round(float(d.bpm), 1) if d.loaded else 0.0,
                "key": f"{camelot_str(ek)} {key_name(ek)}".strip() if ek >= 0 else "",
                "time": round(d.time, 2) if d.loaded else 0.0,
                "duration": round(d.length / SR, 2) if d.loaded else 0.0,
                "rate": round(float(d.rate), 4) if d.loaded else 1.0,
                "volume": round(float(d.volume), 3), "level": round(float(d.level), 3),
                "mix_in": round(d.mix_in / SR, 3) if d.mix_in is not None else None,
                "mix_out": round(d.mix_out / SR, 3) if d.mix_out is not None else None,
                "art": f"/art?i={d.info.lib_index}" if d.loaded and d.info.lib_index >= 0 else None,
            })
        trans = None
        if tr is not None:
            dout, din = e.decks[tr.out_deck], e.decks[tr.in_deck]
            trans = {"out_deck": tr.out_deck, "in_deck": tr.in_deck, "beats": tr.beats, "style": tr.style,
                     "started": tr.started,
                     "start_t": dout.grid.time_of_beat(tr.start_beat) if dout.grid else None,
                     "in_start_t": din.grid.time_of_beat(tr.in_start_beat) if din.grid else None}
        with am._lock:
            queue = list(am.queue)
        hist = []
        for i in am.aired[-30:]:
            if m is None or i != m.info.lib_index:
                hist.append(self._item(int(i)))
        return {
            "automix": {"active": am.active, "status": am.status, "blend_bars": int(am.s.mix_bars)},
            "paused": bool(e.paused),
            "crossfader": float(e.crossfader),
            "decks": decks,
            "transition": trans,
            "queue": [self._item(int(i), p) for p, i in enumerate(queue[:50]) if isinstance(i, (int, np.integer))],
            "history": hist,
        }

    def search(self, q: str, limit: int = 60) -> list[dict]:
        lib = self.lib
        words = [w for w in (q or "").lower().split() if w]
        if lib is None or not words:
            return []
        if self._hay is None or self._hay[0] != id(lib):
            hay = [f"{a} {t} {al} {g} {tg}".lower() for a, t, al, g, tg in
                   zip(lib.artists, lib.titles, lib.albums, lib.parent_genres, lib.tag_genres)]
            self._hay = (id(lib), hay)
        hay = self._hay[1]
        out = []
        for i, h in enumerate(hay):
            if all(w in h for w in words):
                out.append(i)
                if len(out) >= limit:
                    break
        return [self._item(i) for i in out]

    def art(self, idx: int) -> bytes | None:
        lib = self.lib
        if lib is None or not (0 <= idx < lib.n):
            return None
        from ..art import get_art

        return get_art(lib.paths[idx])

    def wave(self, deck: str, bins: int = 480) -> dict | None:
        d = self.eng.decks.get(deck)
        if d is None or not d.loaded:
            return None
        bands = getattr(d, "bands", None)
        dur = d.length / SR
        if bands is None or len(bands) == 0:
            pk = np.abs(np.asarray(d.peaks if d.peaks is not None else np.zeros(1), dtype=np.float32))
            bands = np.stack([pk, pk, pk], axis=1) if pk.ndim == 1 else pk
        b = np.asarray(bands, dtype=np.float32)
        if b.ndim == 1:
            b = np.stack([b, b, b], axis=1)
        edges = np.linspace(0, len(b), bins + 1).astype(int)
        out = np.zeros((bins, 3), np.float32)
        for i in range(bins):
            seg = b[edges[i]:max(edges[i + 1], edges[i] + 1)]
            if len(seg):
                out[i] = seg[:, :3].max(axis=0)
        out /= max(float(out.max()), 1e-9)
        q = np.round(np.sqrt(out) * 99).astype(int)
        g = d.grid
        return {"deck": deck, "duration": dur, "low": q[:, 0].tolist(), "mid": q[:, 1].tolist(),
                "high": q[:, 2].tolist(), "bpm": float(g.bpm) if g else 0.0,
                "first_beat": float(g.first_beat) if g else 0.0}

    # ---- commands (run on the GUI thread) ------------------------------------------------
    def command(self, action: str, args: dict) -> dict:
        fn = getattr(self, "_cmd_" + action, None)
        if fn is None:
            raise ValueError(f"unknown action {action!r}")
        return self._inv.call(lambda: fn(**args)) or {}

    def _idx(self, idx) -> int:
        lib = self.lib
        i = int(idx)
        if lib is None or not (0 <= i < lib.n):
            raise ValueError("unknown track")
        return i

    def _anything_playing(self) -> bool:
        return any(d.playing for d in self.eng.decks.values()) or self.eng.paused

    def _ensure_automix(self):
        if not self.am.active:
            self.am.start()

    def _start_from_queue(self):
        """Nothing is playing: start the mix with the first queued track."""
        self.am.start(from_queue=True)
        self.dj._show_queue()

    def _cmd_toggle_pause(self):
        if not self._anything_playing():
            return self._cmd_skip("mix")
        on = not self.eng.paused
        self.eng.set_paused(on)
        return {"message": "Paused" if on else "Playing"}

    def _cmd_pause(self):
        self.eng.set_paused(True)
        return {"message": "Paused"}

    def _cmd_resume(self):
        self.eng.set_paused(False)
        return {}

    def _cmd_skip(self, mode: str = "mix"):
        if self.eng.paused:
            self.eng.set_paused(False)
        if not self._anything_playing():
            if self.am.queue:
                self._start_from_queue()
            elif any(d.loaded for d in self.eng.decks.values()):
                self.am.stop()
                self.am.start()
            else:
                raise ValueError("Pick a seed song first")
            return {"message": "Starting the mix"}
        self.am.skip(mode)
        self._ensure_automix()
        return {"message": "Cutting to the next track" if mode == "cut" else "Mixing into the next track"}

    def _cmd_back(self):
        prev = self.am.previous_item()
        if prev is None:
            raise ValueError("Nothing to go back to yet")
        self.am.play_next(int(prev), skip="mix")
        self.dj._show_queue()
        return {"message": "Going back to " + self._item(int(prev))["title"]}

    def _cmd_automix(self, on: bool = True):
        if not on:
            self.am.stop()
        elif self._anything_playing():
            self._ensure_automix()
        elif self.am.queue:
            self._start_from_queue()
        elif any(d.loaded for d in self.eng.decks.values()):
            self.am.stop()
            self.am.start()
        else:
            raise ValueError("Pick a seed song first")
        return {"message": "Auto-mix on" if on else "Auto-mix off (the current track keeps playing)"}

    def _cmd_blend(self, bars: int = 16):
        bars = int(bars)
        if not 1 <= bars <= 64:
            raise ValueError("blend must be 1-64 bars")
        self.dj.mix_bars.setValue(bars)
        return {"message": f"Blends over {bars} bars"}

    def _cmd_seed(self, idx):
        i = self._idx(idx)
        main = self.dj.main
        length = int(main.settings.value("remote_set_len", 30))
        try:
            opts = main._setlist_options(length)
        except Exception:  # noqa: BLE001
            from ..mixer import SetlistOptions

            opts = SetlistOptions(length=length, mix=main.mix_settings())
        from ..mixer import build_setlist

        exclude = self.dj.played_indices() - {i}
        sl = build_setlist(self.lib, i, opts, exclude=exclude)
        rest = [x for x in sl if x != i]
        main.setlist = list(sl)
        main._refresh_setlist()
        if self._anything_playing():
            self.am.set_queue(rest)
            self.am.play_next(i, skip="mix", requeue=False)
            msg = f"New set from {self.lib.titles[i]}: mixing it in now"
        else:
            self.am.set_queue([i] + rest)
            self._start_from_queue()
            msg = f"Starting a set from {self.lib.titles[i]}"
        self.dj._show_queue()
        return {"message": msg + f" ({len(sl)} tracks)"}

    def _cmd_play_next(self, idx):
        i = self._idx(idx)
        if not self._anything_playing():
            self.am.set_queue([i] + list(self.am.queue))
            self._start_from_queue()
        else:
            self.am.play_next(i)
            self._ensure_automix()
        self.dj._show_queue()
        return {"message": f"Next up: {self.lib.titles[i]}"}

    def _cmd_queue_add(self, idx):
        i = self._idx(idx)
        with self.am._lock:
            self.am.queue.append(i)
        self.dj._show_queue()
        return {"message": f"Added {self.lib.titles[i]}"}

    def _queue_pop(self, pos) -> int:
        with self.am._lock:
            q = list(self.am.queue)
            p = int(pos)
            if not 0 <= p < len(q):
                raise ValueError("that track is no longer in the queue")
            item = q.pop(p)
            self.am.queue.clear()
            self.am.queue.extend(q)
        return item

    def _cmd_queue_remove(self, pos):
        item = self._queue_pop(pos)
        self.dj._show_queue()
        return {"message": f"Removed {self._item(int(item))['title']}"}

    def _cmd_queue_play(self, pos):
        item = self._queue_pop(pos)
        return self._cmd_play_next(item)

    def _cmd_queue_move(self, pos, to):
        with self.am._lock:
            q = list(self.am.queue)
            p, t = int(pos), int(to)
            if not (0 <= p < len(q) and 0 <= t < len(q)):
                raise ValueError("that track is no longer in the queue")
            q.insert(t, q.pop(p))
            self.am.queue.clear()
            self.am.queue.extend(q)
        self.dj._show_queue()
        return {}

    def _cmd_markers(self, deck: str, **kw):
        mi, mo = kw.get("in"), kw.get("out")
        d = self.eng.decks.get(str(deck))
        if d is None or not d.loaded:
            raise ValueError("that deck is empty")
        msg = self.dj.set_markers(str(deck), None if mi is None else float(mi), None if mo is None else float(mo))
        return {"message": msg}
