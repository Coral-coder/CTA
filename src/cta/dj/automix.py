"""Auto-mix: plays a queue of tracks as a continuous, beat-matched mix.

For each transition:

1. The mix-out point of the playing track is chosen from its energy
   profile: the transition ends where the music (not the fade/tail) ends,
   aligned to an 8-bar phrase.
2. The next track starts on its first strong downbeat, tempo-synced and
   phase-locked to the master, at exactly the same position in the bar.
3. Over ``mix_bars`` the incoming track fades in (equal power), the bass is
   swapped half way (never two kick drums/basslines at full level), then the
   outgoing track fades out.
4. Afterwards the tempo glides to the new track's original BPM over
   ``glide_bars`` (key lock keeps the pitch constant), so a set can travel
   from 122 to 128 BPM without anyone hearing a jump.

It runs as an engine hook (inside the audio thread, sample-accurate) and
never decodes audio there: tracks are prepared by a background loader.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from .beatgrid import energy_profile
from .engine import SR, Engine

log = logging.getLogger(__name__)


@dataclass
class AutoMixSettings:
    mix_bars: int = 16
    phrase_bars: int = 8
    glide: bool = True
    glide_bars: int = 16
    bass_swap: bool = True


@dataclass
class _Meta:
    bar_energy: np.ndarray
    mix_out_beat: float
    mix_in_beat: float
    title: str


@dataclass
class _Transition:
    out_deck: str
    in_deck: str
    start_beat: float        # master beat at which the incoming track starts
    beats: float
    started: bool = False


class AutoMix:
    def __init__(self, engine: Engine, load_fn, settings: AutoMixSettings | None = None,
                 decks=("A", "B"), synchronous_loads: bool = False):
        """load_fn(item) -> LoadedTrack (runs in a background thread)."""
        self.eng = engine
        self.load_fn = load_fn
        self.s = settings or AutoMixSettings()
        self.decks = decks
        self.queue: deque = deque()
        self.active = False
        self.meta: dict[str, _Meta] = {}
        self.tr: _Transition | None = None
        self.glide: tuple[str, float, float, float, float] | None = None  # deck, b0, b1, r0, r1
        self.status = "Auto-mix off"
        self.history: list[str] = []
        self._loading: set[str] = set()
        self._sync_loads = synchronous_loads
        self.on_change = None        # optional callback for the UI (called from audio/loader thread)
        self._lock = threading.Lock()

    # ---- control (UI thread) -----------------------------------------------------
    def set_queue(self, items):
        with self._lock:
            self.queue = deque(items)

    def start(self):
        if self.active:
            return
        self.active = True
        if self._hook not in self.eng.hooks:
            self.eng.hooks.append(self._hook)
        self.status = "Auto-mix starting…"
        self.eng.cmd(lambda e: self._start_in_engine())

    def _start_in_engine(self):
        e = self.eng
        playing = [d for d in self.decks if e.decks[d].playing]
        for d in playing:
            self._register(d)
        if not playing:
            first = self.decks[0]
            if e.decks[first].loaded:
                self._register(first)
                d = e.decks[first]
                d.volume = 1.0
                d.playing = True
                e.master_deck = first
            else:
                self._load_next(first, autoplay=True)
        self._fill_free_deck()

    def stop(self):
        self.active = False
        self.tr = None
        self.glide = None
        self.status = "Auto-mix off"
        self._changed()

    def user_took_over(self):
        """The DJ touched a fader/EQ during a transition: hand control back."""
        if self.active:
            self.stop()
            self.status = "Manual control - auto-mix paused"
            self._changed()

    def transition_now(self, bars: int | None = None):
        """Assist: mix into the other (loaded) deck starting at the next bar."""
        self.eng.cmd(lambda e: self._plan_now(bars))
        if not self.active:
            self.active = True
            if self._hook not in self.eng.hooks:
                self.eng.hooks.append(self._hook)

    # ---- loading -----------------------------------------------------------------
    def _changed(self):
        cb = self.on_change
        if cb:
            try:
                cb()
            except Exception:  # noqa: BLE001
                pass

    def _free_deck(self) -> str | None:
        """A deck that is neither playing, loading, nor holding the prepared
        next track."""
        for d in self.decks:
            dk = self.eng.decks[d]
            if dk.playing or d in self._loading or d in self.meta:
                continue
            if self.tr is not None and d == self.tr.in_deck:
                continue
            return d
        return None

    def _fill_free_deck(self):
        d = self._free_deck()
        if d is not None and self.queue:
            self._load_next(d, autoplay=False)

    def _load_next(self, deck: str, autoplay: bool):
        with self._lock:
            if not self.queue:
                return
            item = self.queue.popleft()
        self._loading.add(deck)

        def job():
            try:
                lt = self.load_fn(item)
            except Exception as e:  # noqa: BLE001
                log.warning("auto-mix could not load %s: %s", item, e)
                self._loading.discard(deck)
                self.eng.cmd(lambda e: self._fill_free_deck())
                return

            def install(e):
                e._load(e, deck, lt.audio, lt.grid, lt.info, lt.peaks)
                d = e.decks[deck]
                d.keylock = True
                d.volume = 1.0 if autoplay else 0.0
                d.eq = np.ones(3)
                self.meta[deck] = self._analyse(lt)
                self._loading.discard(deck)
                if autoplay:
                    e._seek(d, max(lt.grid.time_of_beat(self.meta[deck].mix_in_beat), 0) * SR)
                    d.playing = True
                    e.master_deck = deck
                    self.history.append(lt.info.title)
                self._changed()
            self.eng.cmd(install)

        if self._sync_loads:
            job()
        else:
            threading.Thread(target=job, daemon=True, name=f"automix-load-{deck}").start()

    def _register(self, deck: str):
        """Analyse a deck that was loaded by the user."""
        d = self.eng.decks[deck]
        if d.loaded and deck not in self.meta and d.grid is not None:
            class _LT:
                pass
            lt = _LT()
            lt.audio, lt.grid, lt.info = d.audio, d.grid, d.info
            self.meta[deck] = self._analyse(lt)

    def _analyse(self, lt) -> _Meta:
        mono = lt.audio.mean(axis=0)
        bars = energy_profile(mono, SR, lt.grid)
        med = float(np.median(bars)) if bars.size else 0.0
        # mix-in: first bar with real energy, floored to a 4-bar boundary
        loud_in = np.nonzero(bars > 0.25 * med)[0] if med > 0 else np.array([0])
        in_bar = int(loud_in[0]) // 4 * 4 if loud_in.size else 0
        # mix-out: where the music ends (ignore fade-out tails), phrase aligned
        loud = np.nonzero(bars > 0.5 * med)[0] if med > 0 else np.array([bars.size - 1])
        end_bar = int(loud[-1]) + 1 if loud.size else bars.size
        out_bar = end_bar - self.s.mix_bars
        out_bar = max(out_bar // self.s.phrase_bars * self.s.phrase_bars, in_bar + self.s.phrase_bars)
        out_bar = min(out_bar, max(bars.size - self.s.mix_bars, 0))
        return _Meta(bar_energy=bars, mix_out_beat=out_bar * 4.0, mix_in_beat=in_bar * 4.0,
                     title=lt.info.title)

    # ---- audio-thread logic ------------------------------------------------------
    def _plan_now(self, bars: int | None):
        e = self.eng
        m = e._master()
        if m is None:
            return
        others = [d for d in self.decks if d != m.name and e.decks[d].loaded and not e.decks[d].playing]
        if not others:
            return
        self._register(m.name)
        self._register(others[0])
        mb = m.grid.beat_at(m.time)
        start = np.ceil((mb + 1.0) / 4.0) * 4.0          # next bar line
        beats = 4.0 * (bars or self.s.mix_bars)
        self.tr = _Transition(m.name, others[0], start, beats)
        self.status = f"Transition to {e.decks[others[0]].info.title} at next bar"

    def _hook(self, e: Engine, n: int):
        if not self.active:
            return
        m = e._master()
        if m is None:
            return
        if m.name not in self.meta:
            self._register(m.name)
        mb = m.grid.beat_at(m.time)

        if self.glide is not None:
            self._do_glide(e, mb)

        if self.tr is None:
            # plan the next transition once the other deck is ready
            nxt = [d for d in self.decks if d != m.name and e.decks[d].loaded and not e.decks[d].playing
                   and d in self.meta and d not in self._loading]
            if nxt and m.name in self.meta:
                out_beat = self.meta[m.name].mix_out_beat
                if out_beat <= mb + 1:
                    out_beat = np.ceil((mb + 1) / 4) * 4      # late: mix at the next bar
                self.tr = _Transition(m.name, nxt[0], out_beat, 4.0 * self.s.mix_bars)
            elif not nxt and not self._loading and self.queue:
                self._fill_free_deck()
            bars_left = (self.tr.start_beat - mb) / 4 if self.tr else None
            if bars_left is not None:
                self.status = f"Next: {e.decks[self.tr.in_deck].info.title} in {int(bars_left)} bars"
            elif not self.queue and not self._loading:
                self.status = "Auto-mix: last track playing"
            return

        tr = self.tr
        din, dout = e.decks[tr.in_deck], e.decks[tr.out_deck]
        if not tr.started:
            if mb < tr.start_beat - 1e-6:
                return
            meta_in = self.meta.get(tr.in_deck)
            in_beat0 = meta_in.mix_in_beat if meta_in else 0.0
            # exact bar/phase alignment: incoming beat = in_beat0 + beats since start
            target_beat = in_beat0 + (mb - tr.start_beat)
            din.keylock = True
            din.sync = True
            din.rate_override = e.sync_ratio(m.bpm, din.grid.bpm)
            e._seek(din, din.grid.time_of_beat(target_beat) * SR)
            din.volume = 0.0
            din.eq = np.array([0.0 if self.s.bass_swap else 1.0, 1.0, 1.0])
            din.playing = True
            tr.started = True
            self.history.append(din.info.title)
            self.status = f"Mixing in {din.info.title}"
            self._changed()
            return

        p = (mb - tr.start_beat) / tr.beats
        p = float(np.clip(p, 0.0, 1.0))
        din.volume = float(np.sin(min(p * 2.0, 1.0) * np.pi / 2))
        dout.volume = float(np.cos(max(0.0, (p - 0.5) * 2.0) * np.pi / 2))
        if self.s.bass_swap:
            # swap the low end over two beats around the middle of the mix
            w = 2.0 / tr.beats
            q = float(np.clip((p - 0.5) / w + 0.5, 0.0, 1.0))
            din.eq = np.array([q, 1.0, 1.0])
            dout.eq = np.array([1.0 - q, 1.0, 1.0])
        if p >= 1.0 or not dout.playing:
            self._finish(e, tr, mb)

    def _finish(self, e: Engine, tr: _Transition, mb: float):
        din, dout = e.decks[tr.in_deck], e.decks[tr.out_deck]
        dout.playing = False
        dout.volume = 1.0
        dout.eq = np.ones(3)
        dout.sync = False
        dout.rate_override = None
        din.volume = 1.0
        din.eq = np.ones(3)
        din.sync = False
        din.phase_corr = 0.0
        e.master_deck = din.name
        self.meta.pop(tr.out_deck, None)
        self.tr = None
        r0 = din.rate_override if din.rate_override is not None else din.base_rate
        din.rate_override = r0
        if self.s.glide and abs(r0 - 1.0) > 1e-4:
            b0 = din.grid.beat_at(din.time)
            self.glide = (din.name, b0, b0 + 4.0 * self.s.glide_bars, r0, 1.0)
        self.status = f"Playing {din.info.title}"
        self._fill_free_deck()
        self._changed()

    def _do_glide(self, e: Engine, _mb: float):
        name, b0, b1, r0, r1 = self.glide
        d = e.decks[name]
        if not d.playing or d.grid is None:
            self.glide = None
            return
        b = d.grid.beat_at(d.time)
        p = float(np.clip((b - b0) / max(b1 - b0, 1e-9), 0.0, 1.0))
        p = p * p * (3 - 2 * p)                       # smoothstep: no audible corners
        d.rate_override = r0 + (r1 - r0) * p
        if p >= 1.0:
            d.rate_override = None
            d.tempo = 0.0
            self.glide = None
