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

from .engine import SR, Engine
from .structure import Structure, analyse, plan_transition

# hand moves that make a transition in progress the DJ's own
TAKEOVER_ACTIONS = {"volume", "eq_low", "eq_mid", "eq_high", "crossfader", "filter", "gain", "trim", "play",
                    "cue", "sync", "tempo", "jog", "jog_touch", "scratch", "bend", "jump", "beatjump_back",
                    "beatjump_fwd", "loop", "loop_in", "loop_out", "loop_double", "loop_halve", "fader", "eq"}

log = logging.getLogger(__name__)


@dataclass
class AutoMixSettings:
    mix_bars: int = 16           # blend length (the longest one when auto_length)
    auto_length: bool = True     # pick the length from the music, never hanging on dead air
    phrase_bars: int = 8
    glide: bool = True
    glide_bars: int = 16
    bass_swap: bool = True


@dataclass
class _Meta:
    structure: Structure
    in_marker_bar: float | None      # user intro marker (bar)
    out_marker_bar: float | None     # user outro marker (bar)
    title: str
    genre: str = ""
    long_form: bool = False          # trance / progressive: 16-bar phrases, long blends
    out_end_bar: float | None = None   # user outro end marker (bar): blend lasts until here

    @property
    def mix_in_beat(self) -> float:
        if self.in_marker_bar is not None:
            return 4.0 * self.in_marker_bar
        fb = self.structure.first_beat_bar
        return 4.0 * (fb - fb % 4)


@dataclass
class _Transition:
    out_deck: str
    in_deck: str
    start_beat: float        # master beat at which the incoming track starts
    beats: float             # blend length
    in_start_beat: float = 0.0
    swap_beat: float = 0.0   # beats into the blend where the basses swap (downbeat)
    style: str = "blend"     # blend | loop_blend | echo_out
    loop: tuple[float, float] | None = None   # outgoing loop (beats) to hold the groove
    reason: str = ""
    started: bool = False
    fixed_length: bool = False       # set by "transition now" with an explicit length
    loop_set: bool = False
    in_loop: tuple[float, float] | None = None   # incoming loop (its beats) stretching a short intro
    in_release: float = 0.0          # beats into the blend when the incoming loop is let go
    in_loop_on: bool = False
    in_wraps: float = 0.0            # beats the incoming loop has wrapped back (keeps x moving forward)
    last_raw: float = 0.0
    echo_on: bool = False
    rehearsal: object = None         # a Rehearsal: replay the DJ's own pre-mixed moves

    @property
    def tail_beats(self) -> float:
        """Extra time after the swap for the outgoing track to leave: a
        quarter of the blend, 2 to 8 bars, so it never just vanishes."""
        if self.style == "echo_out":
            return 8.0
        if self.style in ("cut", "rehearsed"):
            return 0.0
        if self.swap_beat >= self.beats - 1e-6:
            return float(np.clip(0.25 * self.beats, 8.0, 32.0))
        return 0.0


SHORT_BLEND_GENRES = {"Hip Hop", "Pop", "Rock", "Funk / Soul", "Latin", "Reggae", "Folk, World, & Country",
                      "Blues", "Jazz", "Brass & Military", "Children's", "Stage & Screen"}


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
        self.used_items: list = []           # every queue item ever loaded (never auto-pick these again)
        self.aired: list = []                # library indices in the order they went on air
        self.pending_skip: str | None = None # "mix" | "cut": requested by the remote / skip button
        self.skip_bars = 4
        self._load_gen: dict[str, int] = {}
        self.learner = None                  # MixLearner: the DJ's own transition habits
        self.resume_after_manual = False     # the DJ took over a blend: carry on after it
        self.rehearsals = None               # RehearsalStore: transitions the DJ mixed ahead of time
        self._resume_t = None
        self._sync_loads = synchronous_loads
        self._loading_item: dict = {}
        self.on_change = None        # optional callback for the UI (called from audio/loader thread)
        self._lock = threading.Lock()

    # ---- control (UI thread) -----------------------------------------------------
    def set_queue(self, items):
        with self._lock:
            self.queue = deque(items)

    def start(self, from_queue: bool = False):
        """from_queue: when nothing is playing, start with the first queued
        track even if a deck still holds an older one."""
        if self.active and not from_queue:
            return
        self.active = True
        if self._hook not in self.eng.hooks:
            self.eng.hooks.append(self._hook)
        self.status = "Auto-mix starting…"
        self.eng.cmd(lambda e: self._start_in_engine(from_queue))

    def _start_in_engine(self, from_queue: bool = False):
        e = self.eng
        playing = [d for d in self.decks if e.decks[d].playing]
        for d in playing:
            self._register(d)
        if not playing and from_queue and self.queue and not e.paused:
            # whatever the decks hold is stale: the queue decides what plays
            for dn in self.decks:
                self.meta.pop(dn, None)
                self._load_gen[dn] = self._load_gen.get(dn, 0) + 1
                self._loading.discard(dn)
                self._loading_item.pop(dn, None)
            self.tr = None
            self._load_next(self.decks[0], autoplay=True)
        elif not playing:
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

    def stop(self, settle: bool = False):
        """settle: a blend in progress is completed at once (the incoming
        track carries on alone) instead of leaving both tracks playing."""
        tr = self.tr
        if settle and tr is not None and tr.started:
            def f(e):
                self._finish(e, tr, 0.0)
                self.glide = None
                self.status = "Auto-mix off"
            self.eng.cmd(f)
        self.active = False
        self.tr = None
        self.glide = None
        self.resume_after_manual = False
        self.status = "Auto-mix off"
        self._changed()

    def user_took_over(self):
        """The DJ took over a transition: auto-mix lets go at once (no more
        fader / EQ moves), the learner keeps the blend as a lesson, and
        auto-mix carries on by itself once the DJ's blend is done."""
        if self.active:
            self.stop()
            self.resume_after_manual = True
            self._resume_t = None
            if self.learner is not None:
                self.learner.took_over()
            self.status = "You're mixing - auto-mix carries on after your blend"
            self._changed()

    def manual(self, deck: str | None, action: str) -> bool:
        """The DJ did something by hand (UI, keyboard or controller). If it
        touches a transition in progress - or starts the other deck - the
        transition is theirs. Returns True when auto-mix stepped back."""
        if not self.active or action not in TAKEOVER_ACTIONS and not action.startswith("hotcue"):
            return False
        tr = self.tr
        if tr is not None and tr.started:
            if deck is None or deck in (tr.in_deck, tr.out_deck):
                self.user_took_over()
                return True
        elif action == "play" and deck in self.decks and not self.eng.decks[deck].playing \
                and any(self.eng.decks[d].playing for d in self.decks if d != deck):
            self.user_took_over()               # starting the other deck: a manual mix begins
            return True
        return False

    def maybe_resume(self):
        """Called ~30x/s by the UI: back to auto-mix once one track plays alone."""
        if not self.resume_after_manual or self.active:
            return
        import time as _t

        playing = [d for d in self.decks if self.eng.decks[d].playing]
        if len(playing) != 1 or self.eng.paused:
            self._resume_t = None
            return
        now = _t.monotonic()
        if self._resume_t is None:
            self._resume_t = now
        elif now - self._resume_t >= 3.0:
            self.resume_after_manual = False
            self.start()

    def transition_now(self, bars: int | None = None):
        """Assist: mix into the other (loaded) deck starting at the next bar."""
        self.eng.cmd(lambda e: self._plan_now(bars))
        if not self.active:
            self.active = True
            if self._hook not in self.eng.hooks:
                self.eng.hooks.append(self._hook)

    def skip(self, mode: str = "mix"):
        """Go to the next track now: "mix" blends over a few bars from the next
        bar line, "cut" switches on the next beat. Works whether or not the
        next track is loaded yet (it is mixed in as soon as it is ready)."""
        self.pending_skip = "cut" if mode == "cut" else "mix"
        if not self.active:
            self.start()

    def play_next(self, item, skip: str | None = None, requeue: bool = True):
        """Make item the very next track (replacing the prepared one, which
        goes back to the front of the queue unless requeue is False)."""
        def f(e):
            back = None
            for dn in self.decks:
                d = e.decks[dn]
                if d.playing:
                    continue
                if self.tr is not None and self.tr.started and dn == self.tr.in_deck:
                    continue
                if dn in self._loading:
                    self._load_gen[dn] = self._load_gen.get(dn, 0) + 1   # discard that load
                    self._loading.discard(dn)
                    back = self._loading_item.pop(dn, None)
                elif dn in self.meta and d.loaded:
                    back = d.info.lib_index if d.info.lib_index >= 0 else None
                    self.meta.pop(dn, None)          # free: the new track loads over it
                else:
                    continue
                if self.tr is not None and not self.tr.started and self.tr.in_deck == dn:
                    self.tr = None
                break
            with self._lock:
                if requeue and back is not None and back != item:
                    if back in self.used_items:
                        self.used_items.remove(back)
                    self.queue.appendleft(back)
                self.queue.appendleft(item)
            if skip:
                self.pending_skip = skip
            if not self.active:
                self.start()
            else:
                self._fill_free_deck()
            self._changed()
        self.eng.cmd(f)

    def previous_item(self):
        """The track that was on air before the current one (for "back")."""
        m = self.eng._master()
        cur = m.info.lib_index if m is not None else None
        for it in reversed(self.aired):
            if it != cur:
                return it
        return None

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
        on_air = {d.info.lib_index for d in self.eng.decks.values() if d.playing and d.info.lib_index >= 0}
        with self._lock:
            # never the song that is on air again (back to back)
            while self.queue and isinstance(self.queue[0], (int, np.integer)) and int(self.queue[0]) in on_air:
                self.queue.popleft()
            if not self.queue:
                return
            item = self.queue.popleft()
        self.used_items.append(item)
        self._loading.add(deck)
        self._loading_item[deck] = item
        gen = self._load_gen.get(deck, 0)

        def job():
            try:
                lt = self.load_fn(item)
            except Exception as e:  # noqa: BLE001
                log.warning("auto-mix could not load %s: %s", item, e)
                if self._load_gen.get(deck, 0) != gen:
                    return
                self._loading.discard(deck)
                self._loading_item.pop(deck, None)
                self.eng.cmd(lambda e: self._fill_free_deck())
                return

            def install(e):
                if self._load_gen.get(deck, 0) != gen or e.decks[deck].playing:
                    return                      # replaced meanwhile ("play next")
                e._load(e, deck, lt.audio, lt.grid, lt.info, lt.peaks, cue=getattr(lt, "first_beat_t", None),
                        mix_in=getattr(lt, "mix_in", None), mix_out=getattr(lt, "mix_out", None),
                        bands=getattr(lt, "bands", None), mix_out_end=getattr(lt, "mix_out_end", None))
                d = e.decks[deck]
                if getattr(lt, "cues", None):
                    d.hotcues = (list(lt.cues) + [None] * 8)[:8]
                d.keylock = True
                d.volume = 1.0 if autoplay else 0.0
                d.eq = np.ones(3)
                d.phrase_origin = getattr(lt, "phrase_origin", 0)
                self.meta[deck] = self._analyse_deck(d)
                self._loading.discard(deck)
                self._loading_item.pop(deck, None)
                if autoplay:
                    e._seek(d, max(lt.grid.time_of_beat(self.meta[deck].mix_in_beat), 0) * SR)
                    d.playing = True
                    e.master_deck = deck
                    self.history.append(lt.info.title)
                    self._aired(d)
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
            self.meta[deck] = self._analyse_deck(d)

    def refresh_markers(self, deck: str):
        """Markers changed in the UI: re-plan with them (from any thread)."""
        def f(e):
            d = e.decks[deck]
            if deck in self.meta and d.loaded:
                self.meta[deck] = self._analyse_deck(d)
                if self.tr is not None and not self.tr.started and deck in (self.tr.in_deck, self.tr.out_deck):
                    self.tr = None          # plan again with the new markers
        self.eng.cmd(f)

    def _analyse_deck(self, d) -> _Meta:
        g = d.grid
        bands = getattr(d, "bands", None)
        if bands is None:
            from .loader import waveform_bands
            bands = waveform_bands(d.audio, hop=256)
            d.bands = bands
        style = (getattr(d.info, "style", "") or "").lower()
        long_form = any(w in style for w in ("trance", "progressive", "psy"))
        st = analyse(bands, g, phrase_origin=getattr(d, "phrase_origin", None),
                     phrase_bars=16 if long_form else 8)
        in_m = out_m = None
        if d.mix_in is not None:
            in_m = float(np.floor(g.beat_at(d.mix_in / SR) / 4.0 + 1e-6))
        out_end = None
        if d.mix_out is not None:
            out_m = float(np.floor(g.beat_at(d.mix_out / SR) / 4.0 + 1e-6))
            if getattr(d, "mix_out_end", None) is not None:
                out_end = float(np.floor(g.beat_at(d.mix_out_end / SR) / 4.0 + 1e-6))
                if out_end <= out_m:
                    out_end = None
        return _Meta(structure=st, in_marker_bar=in_m, out_marker_bar=out_m, out_end_bar=out_end,
                     title=d.info.title,
                     genre=getattr(d.info, "genre", "") or "", long_form=long_form)

    # ---- audio-thread logic ------------------------------------------------------
    def _make_plan(self, e: Engine, m, din_name: str, now_beat: float, force_bars: int | None = None,
                   immediate: bool = False) -> _Transition:
        from ..keys import compatibility

        mo, mi = self.meta[m.name], self.meta[din_name]
        din = e.decks[din_name]
        r = None
        if self.rehearsals is not None and force_bars is None and not immediate:
            try:
                r = self.rehearsals.get(m.info.path, din.info.path)
            except Exception:  # noqa: BLE001
                r = None
        if r is not None and r.out_start_beat > now_beat + 1.0:
            # the DJ mixed this one ahead of time: same beat, same moves
            return _Transition(m.name, din_name, r.out_start_beat, r.beats, in_start_beat=r.in_start_beat,
                               swap_beat=r.beats, style="rehearsed", reason="your time-shifted mix",
                               fixed_length=True, rehearsal=r)
        ratio = e.sync_ratio(m.bpm, din.grid.bpm)
        gap = abs(ratio - 1.0) * 100
        harmonic = compatibility(m.effective_key, din.info.key) if din.info.key >= 0 else 1.0
        short = any(g in SHORT_BLEND_GENRES for g in (mo.genre, mi.genre))
        learned = None
        if self.learner is not None and force_bars is None:
            from .learn import style_group
            learned = self.learner.preferences(style_group(m.info.style, m.info.genre))
        plan = plan_transition(mo.structure, mi.structure, max_bars=force_bars or self.s.mix_bars,
                               harmonic=harmonic, tempo_gap_pct=gap,
                               out_marker_bar=mo.out_marker_bar, in_marker_bar=mi.in_marker_bar,
                               genre_short=short and force_bars is None, learned=learned,
                               out_end_bar=mo.out_end_bar if force_bars is None else None,
                               auto=force_bars is None and self.s.auto_length)
        start = plan.out_start_bar * 4.0
        if immediate or start <= now_beat + 1.0:
            # too late for the planned spot (or "transition now"): next bar line
            start = float(np.ceil((now_beat + 1.0) / 4.0) * 4.0)
            loop = None
        else:
            loop = (plan.loop[0] * 4.0, plan.loop[1] * 4.0) if plan.loop else None
        style = plan.style if (loop or plan.style != "loop_blend") else "blend"
        in_loop = (plan.in_loop[0] * 4.0, plan.in_loop[1] * 4.0) if plan.in_loop else None
        return _Transition(m.name, din_name, start, plan.bars * 4.0, in_start_beat=plan.in_start_bar * 4.0,
                           swap_beat=plan.swap_bar * 4.0, style=style, loop=loop, reason=plan.reason,
                           fixed_length=force_bars is not None or plan.exact, in_loop=in_loop,
                           in_release=(plan.in_release or 0.0) * 4.0)

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
        self.tr = self._make_plan(e, m, others[0], mb, force_bars=bars, immediate=True)
        self.status = f"Transition to {e.decks[others[0]].info.title} at the next bar"

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

        if self.pending_skip is not None and self._do_skip(e, m, mb):
            return

        if self.tr is None:
            # plan the next transition once the other deck is ready
            nxt = [d for d in self.decks if d != m.name and e.decks[d].loaded and not e.decks[d].playing
                   and d in self.meta and d not in self._loading]
            if nxt and m.name in self.meta:
                self.tr = self._make_plan(e, m, nxt[0], mb)
                log.info("auto-mix plan: %s (%s)", self.tr.style, self.tr.reason)
            elif not nxt and not self._loading and self.queue:
                self._fill_free_deck()
            if self.tr is None and not self.queue and not self._loading:
                self.status = "Auto-mix: last track playing"
            return

        tr = self.tr
        din, dout = e.decks[tr.in_deck], e.decks[tr.out_deck]
        if not tr.started:
            # hold the groove: engage the outgoing loop when we reach it
            if tr.loop and not tr.loop_set and mb >= tr.loop[0] - 0.05:
                g = dout.grid
                dout.loop = (g.time_of_beat(tr.loop[0]) * SR, g.time_of_beat(tr.loop[1]) * SR)
                tr.loop_set = True
            bars_left = (tr.start_beat - mb) / 4
            how = {"blend": "blend", "loop_blend": "loop + blend", "echo_out": "echo out",
                   "cut": "cut", "rehearsed": "your time-shifted mix"}[tr.style]
            self.status = (f"Next: {din.info.title} in {max(0, int(np.ceil(bars_left)))} bars "
                           f"({how}, {int(tr.beats / 4)} bars)")
            if mb < tr.start_beat - 1e-6:
                return
            if not tr.fixed_length and tr.style != "echo_out":
                tr.beats = min(tr.beats, 4.0 * self.s.mix_bars)      # honour a later change
                tr.swap_beat = min(tr.swap_beat, tr.beats)
            # exact bar/phase alignment: incoming beat = its start + beats since start
            target_beat = tr.in_start_beat + (mb - tr.start_beat)
            din.keylock = True
            din.sync = True
            din.rate_override = e.sync_ratio(m.bpm, din.grid.bpm)
            e._seek(din, din.grid.time_of_beat(target_beat) * SR)
            din.volume = 0.0
            din.eq = np.array([0.0, 0.5, 1.0]) if self.s.bass_swap else np.ones(3)
            if tr.rehearsal is not None:
                from .rehearsal import apply

                apply(tr.rehearsal, 0.0, din, dout)
            if tr.in_loop is not None:
                g = din.grid
                din.loop = (g.time_of_beat(tr.in_loop[0]) * SR, g.time_of_beat(tr.in_loop[1]) * SR)
                tr.in_loop_on = True
            din.playing = True
            tr.started = True
            self.history.append(din.info.title)
            self._aired(din)
            if tr.style == "cut":
                din.volume = 1.0
                din.eq = np.ones(3)
                self._finish(e, tr, mb)
                self.status = f"Cut to {din.info.title}"
                return
            self.status = f"Mixing in {din.info.title}"
            self._changed()
            return

        # beats into the transition, counted on the incoming track: the outgoing
        # may be looping, so its position is not a clock
        raw = din.grid.beat_at(din.time) - tr.in_start_beat
        if tr.in_loop_on and raw < tr.last_raw - 1.0:
            tr.in_wraps += tr.in_loop[1] - tr.in_loop[0]         # the incoming loop jumped back
        tr.last_raw = raw
        x = raw + tr.in_wraps
        if tr.in_loop_on and x >= tr.in_release - 1e-6:
            din.loop = None                    # let the incoming run on into its drop
            tr.in_loop_on = False
        if tr.style == "rehearsed":
            from .rehearsal import apply

            apply(tr.rehearsal, x, din, dout)
        elif tr.style == "echo_out":
            self._echo_out(e, tr, din, dout, x)
        else:
            self._blend(tr, din, dout, x)
        if x >= tr.beats + tr.tail_beats or not dout.playing and x >= tr.swap_beat:
            self._finish(e, tr, mb)

    def _aired(self, d):
        if d.info.lib_index >= 0:
            self.aired.append(d.info.lib_index)
            del self.aired[:-200]

    def _do_skip(self, e: Engine, m, mb: float) -> bool:
        mode = self.pending_skip
        tr = self.tr
        if tr is not None and tr.started:
            self.pending_skip = None
            if mode == "cut":
                din = e.decks[tr.in_deck]
                din.volume = 1.0
                din.eq = np.ones(3)
                self._finish(e, tr, mb)
            return False
        nxt = [d for d in self.decks if d != m.name and e.decks[d].loaded and not e.decks[d].playing
               and d in self.meta and d not in self._loading]
        if not nxt:
            if not self._loading and self.queue:
                self._fill_free_deck()
            self.status = "Skip: preparing the next track…"
            return False
        self.pending_skip = None
        self._register(m.name)
        if mode == "cut":
            t = self._make_plan(e, m, nxt[0], mb, force_bars=1, immediate=True)
            t.start_beat = float(np.ceil(mb + 0.02))          # the next beat
            t.in_start_beat = self.meta[nxt[0]].mix_in_beat
            t.beats, t.swap_beat, t.style, t.loop = 1.0, 0.0, "cut", None
        else:
            bars = self.skip_bars * (2 if self.meta[nxt[0]].long_form else 1)
            t = self._make_plan(e, m, nxt[0], mb, force_bars=bars, immediate=True)
        self.tr = t
        self.status = f"Skipping to {e.decks[nxt[0]].info.title}"
        self._changed()
        return True

    @staticmethod
    def _ramp(v: float) -> float:
        return float(np.clip(v, 0.0, 1.0))

    def _blend(self, tr: _Transition, din, dout, x: float):
        """EQ mixing like a DJ: the incoming comes in highs first with mids
        under and bass cut; the outgoing eases off (highs, mids, a little
        level) through the second half of the build-up; the basses swap on
        the downbeat; then the outgoing fades away over the tail - a real
        crossfade, never a sudden drop."""
        beats = max(tr.beats, 1.0)
        sw = min(tr.swap_beat, beats)
        fade_in = min(float(np.clip(0.4 * beats, 4.0, 32.0)), max(sw, 4.0))
        din.volume = float(np.sin(self._ramp(x / fade_in) * np.pi / 2))
        swap = self._ramp((x - sw) / 0.5 + 0.5) if self.s.bass_swap else self._ramp(x / beats)
        mid_in = 0.5 + 0.5 * self._ramp(x / max(sw, 1.0))
        din.eq = np.array([swap, mid_in, 1.0])
        pre = self._ramp((x - 0.5 * sw) / max(0.5 * sw, 1.0))        # second half before the swap
        post = self._ramp((x - sw) / max(beats + tr.tail_beats - sw, 4.0))
        dout.eq = np.array([1.0 - swap, float(np.clip(1.0 - 0.3 * pre - 0.5 * post, 0, 1)),
                            float(np.clip(1.0 - 0.35 * pre - 0.5 * post, 0, 1))])
        dout.volume = float((1.0 - 0.2 * pre) * np.cos(post * np.pi / 2))

    def _echo_out(self, e: Engine, tr: _Transition, din, dout, x: float):
        """Clash-safe: the outgoing track echoes away on the last bar, the next
        track drops in on the one."""
        sw = tr.swap_beat
        unit = e.fx_units[-1]                          # auto-mix's own delay unit
        if not tr.echo_on and x >= sw - 4.0:
            unit.kind, unit.wet, unit.amount, unit.div_index = "Delay", 0.6, 0.55, 2
            unit.on = True
            dout.fx = set(dout.fx) | {len(e.fx_units) - 1}
            tr.echo_on = True
        if x < sw:
            din.volume = 0.0
            dout.volume = 1.0
            dout.eq = np.array([1.0, 1.0, 1.0]) if x < sw - 4.0 else np.array([self._ramp((sw - x) / 4.0), 1, 1])
        else:
            din.volume = 1.0
            din.eq = np.ones(3)
            dout.trim = 0.0                            # dry signal off, echoes ring out
            unit.on = False
            dout.volume = float(np.cos(self._ramp((x - sw) / tr.tail_beats) * np.pi / 2))

    def _finish(self, e: Engine, tr: _Transition, mb: float):
        din, dout = e.decks[tr.in_deck], e.decks[tr.out_deck]
        if tr.in_loop_on:
            din.loop = None
            tr.in_loop_on = False
        dout.playing = False
        dout.volume = 1.0
        dout.eq = np.ones(3)
        dout.loop = None
        dout.trim = 1.0
        dout.fx = set(dout.fx) - {len(e.fx_units) - 1}
        dout.sync = False
        dout.rate_override = None
        dout.filter = 0.0
        if tr.style != "rehearsed":            # a rehearsed mix ends where the DJ left the incoming
            din.volume = 1.0
            din.eq = np.ones(3)
            din.filter = 0.0
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
