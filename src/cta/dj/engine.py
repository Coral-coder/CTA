"""Real-time DJ audio engine.

* 4 decks, each with: key lock (Rubber Band R3 real-time time-stretching, so
  tempo changes never change the pitch), vinyl mode (resampling) for jog
  scratching, tempo fader, nudge, cue, beat-quantised loops, 3-band kill EQ,
  gain, channel fader, headphone cue.
* Sync = tempo match + continuous phase lock to the master deck.
* Mixer with crossfader, master and headphone (cue) outputs; on a 4-channel
  interface such as the DDJ-T1's built-in sound card the headphones get
  their own output pair.

The audio callback is self-contained and never raises: on any internal error
it outputs silence for that block and logs once. Commands from the GUI/MIDI
threads go through a queue and are applied at the start of each block, so
the callback never waits on a lock.
"""

from __future__ import annotations

import logging
import queue
import threading
import weakref
from dataclasses import dataclass, field

import numpy as np

from .beatgrid import BeatGrid

log = logging.getLogger(__name__)

SR = 44100
DECKS = ("A", "B", "C", "D")
EQ_LOW_HZ = 220.0
EQ_HIGH_HZ = 2500.0


# --------------------------------------------------------------------------
# filters (scipy's sosfilt keeps state between blocks)
# --------------------------------------------------------------------------

def sos_filter(sos: np.ndarray, x: np.ndarray, zi: np.ndarray) -> np.ndarray:
    """Filter a (channels, n) block with second-order sections, keeping the
    state zi (channels, sections, 2) in place. scipy's own sosfilt checks and
    reshapes its arguments on every call - most of the time for a 512-frame
    block - so the audio thread calls its compiled core directly."""
    y = np.array(x, dtype=np.float64, order="C")
    if _SOS_CORE is not None:
        _SOS_CORE(sos, y, zi)
        return y
    from scipy.signal import sosfilt

    out, z = sosfilt(sos, y, axis=1, zi=np.moveaxis(zi, 0, 1))
    zi[...] = np.moveaxis(z, 1, 0)
    return out


try:                                    # scipy's compiled core (same maths, no argument checks)
    from scipy.signal._sosfilt import _sosfilt as _SOS_CORE
except Exception:  # noqa: BLE001
    _SOS_CORE = None


def _lr4(freq: float, kind: str):
    """4th-order Linkwitz-Riley = two cascaded 2nd-order Butterworths."""
    from scipy.signal import butter

    sos = butter(2, freq, kind, fs=SR, output="sos")
    return np.vstack([sos, sos])


class ThreeBandEQ:
    """Two cascaded LR4 crossovers (220 Hz, 2.5 kHz). LR4 low+high sum to a
    flat all-pass, so at unity the EQ is transparent in level, and a kill
    removes its band completely."""

    _filters = None

    def __init__(self):
        if ThreeBandEQ._filters is None:
            ThreeBandEQ._filters = tuple(np.ascontiguousarray(f) for f in (
                _lr4(EQ_LOW_HZ, "low"), _lr4(EQ_LOW_HZ, "high"), _lr4(EQ_HIGH_HZ, "low"), _lr4(EQ_HIGH_HZ, "high")))
        self.states = [np.zeros((2, f.shape[0], 2)) for f in ThreeBandEQ._filters]
        self.gains = np.ones(3)
        self._ramp = np.zeros(0)

    def _f(self, i: int, x: np.ndarray) -> np.ndarray:
        return sos_filter(ThreeBandEQ._filters[i], x, self.states[i])

    def process(self, x: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """x: (2, n) float. gains: (low, mid, high) linear."""
        low = self._f(0, x)
        rest = self._f(1, x)
        mid = self._f(2, rest)
        high = self._f(3, rest)
        n = x.shape[1]
        gains = np.asarray(gains, dtype=np.float64)
        if np.array_equal(gains, self.gains):             # steady: no ramp to build
            g0, g1, g2 = gains
            out = low * g0 + mid * g1 + high * g2
        else:
            if self._ramp.shape[0] != n:
                self._ramp = np.linspace(0.0, 1.0, n)
            r = self._ramp
            out = (low * (self.gains[0] + (gains[0] - self.gains[0]) * r)
                   + mid * (self.gains[1] + (gains[1] - self.gains[1]) * r)
                   + high * (self.gains[2] + (gains[2] - self.gains[2]) * r))
            self.gains = gains.copy()
        return out.astype(np.float32)


# --------------------------------------------------------------------------
# deck
# --------------------------------------------------------------------------

@dataclass
class TrackInfo:
    path: str = ""
    title: str = ""
    artist: str = ""
    key: int = -1
    bpm: float = 0.0
    loudness: float = -12.0
    lib_index: int = -1
    genre: str = ""
    style: str = ""          # AI style, e.g. "Progressive Trance"


@dataclass
class Deck:
    name: str
    audio: np.ndarray | None = None          # (2, n) float32 @ SR
    grid: BeatGrid | None = None
    info: TrackInfo = field(default_factory=TrackInfo)
    peaks: np.ndarray | None = None          # waveform overview for the UI
    cue_by_dj: bool = False                  # the DJ set the cue (else LOOP IN moves it)
    gliding: bool = False                    # auto-mix is gliding the tempo fader home
    play_pos: float = 0.0                    # source frame being heard now
    feed_pos: float = 0.0                    # source frame fed to the stretcher
    playing: bool = False
    tempo: float = 0.0                       # tempo fader, -1..1 of range
    tempo_range: float = 0.08                # +-8 %
    rate_override: float | None = None       # set by sync/automix (absolute rate)
    nudge: float = 0.0                       # temporary rate offset (jog/nudge)
    keylock: bool = True
    sync: bool = False
    volume: float = 1.0                      # channel fader 0..1
    gain: float = 1.0                        # auto-gain (from the track's loudness), linear
    trim: float = 1.0                        # user trim knob, linear
    eq: np.ndarray = field(default_factory=lambda: np.ones(3))  # low, mid, high
    cue_phones: bool = False
    xf_side: int = 0                         # -1 left, +1 right, 0 thru
    cue_point: float = 0.0
    cue_preview: bool = False
    loop: tuple[float, float] | None = None  # (start, end) source frames
    phase_offset: float = 0.0                # beats: where the DJ's jog nudge left the sync lock
    nudged: bool = False                     # the jog is (or was just) nudging this deck
    start_pos: float = 0.0                   # where the last start began (frames): the display holds there
    auto_phase: bool = False                 # auto-mix holds it on the beat even with tempo-only sync
    loop_size: float = 4.0                   # the chosen loop length (beats): OUT with no IN uses it
    scope: np.ndarray | None = None          # the deck's newest output block (XY scope)
    scratching: bool = False
    scratch_rate: float = 0.0
    phase_corr: float = 0.0
    end_reached: bool = False
    hotcues: list = field(default_factory=lambda: [None] * 8)   # source frames
    key_shift: float = 0.0                   # semitones (key lock path only)
    filter: float = 0.0                      # -1 low-pass .. 0 off .. +1 high-pass
    fx: set = field(default_factory=set)     # assigned FX units (0, 1)
    flux: bool = False                       # loops/jumps keep the timeline running
    flux_pos: float = 0.0
    level: float = 0.0                       # post-fader peak for the channel meter
    mix_in: float | None = None              # intro marker (source frames)
    mix_out: float | None = None             # outro marker (source frames)
    mix_out_end: float | None = None         # outro end marker (source frames): blend ends here
    bands: object = None                     # waveform bands (frames, 3) for analysis/drawing
    phrase_origin: int = 0                   # bar (mod 8) where phrases start
    _filter: object = None
    _pitch: float = 1.0
    _rb: object = None
    _rb_ratio: float = 1.0
    _rb_drop: int = 0
    _rb_reset: bool = False                  # jump / cue / loop: start the stretcher afresh (reused)
    _mode: str = "vinyl"
    _calm_frames: int = 0                    # how long the speed has been back at normal (stretcher exit)
    _eq: ThreeBandEQ | None = None
    _last_gain: float = 0.0
    _tail: object = None                     # last output frame: a stop fades from it instead of cutting
    _scr_gain: float = 0.0                   # a held record is silent: level follows the hand's speed

    # ---- derived ----------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self.audio is not None

    @property
    def length(self) -> int:
        return 0 if self.audio is None else self.audio.shape[1]

    @property
    def base_rate(self) -> float:
        if self.rate_override is not None:
            return self.rate_override
        return 1.0 + self.tempo * self.tempo_range

    @property
    def rate(self) -> float:
        if self.scratching:
            return self.scratch_rate
        return self.base_rate * (1.0 + self.nudge) + self.phase_corr

    @property
    def bpm(self) -> float:
        """Current playing tempo."""
        return (self.grid.bpm if self.grid else 0.0) * self.base_rate

    @property
    def time(self) -> float:
        return self.play_pos / SR

    def beat(self) -> float:
        return self.grid.beat_at(self.time) if self.grid else 0.0

    @property
    def effective_key(self) -> int:
        """Musical key heard now (key shift, or vinyl pitch when key lock is off)."""
        k = self.info.key
        if k is None or k < 0:
            return -1
        semis = self.key_shift if self.keylock else 12 * np.log2(max(self.base_rate, 1e-3))
        s = int(round(semis)) % 12
        return (k % 12 + s) % 12 + (12 if k >= 12 else 0)


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------

def _rb_new(rate: float):
    from pylibrb import Option, RubberBandStretcher

    rb = RubberBandStretcher(SR, 2, Option.PROCESS_REALTIME | Option.ENGINE_FINER,
                             initial_time_ratio=1.0 / max(rate, 0.05))
    rb.set_max_process_size(16384)
    pad = rb.get_preferred_start_pad()
    if pad:
        rb.process(np.zeros((2, pad), np.float32))
    return rb, rb.get_start_delay()


class Engine:
    _instances = weakref.WeakSet()          # every engine, so all can be stopped at exit

    def __init__(self, block: int = 512):
        self.decks = {n: Deck(n) for n in DECKS}
        for d in self.decks.values():
            d.xf_side = -1 if d.name in ("A", "C") else 1
        self.crossfader = 0.0          # -1 (left) .. +1 (right)
        self.master_volume = 0.9
        self.phones_volume = 0.8
        self.cue_mix = 0.0             # 0 = cue only, 1 = master only
        self.master_deck: str | None = None
        self.block = block
        self.frames_rendered = 0
        self.underruns = 0
        self.peak_level = 0.0
        self.out_channels = 2
        self.master_chans = (0, 1)
        self.phones_chans: tuple[int, int] | None = None
        self._cmds: queue.SimpleQueue = queue.SimpleQueue()
        Engine._instances.add(self)
        # time-shift preview: a second engine rendered inside this one's audio
        # callback (its own decks; heard on the headphones, or the main output)
        # output clock: which output keeps time ("auto" = the slowest), which
        # outputs follow it (None = all) and the DJ's fine adjustment
        self.clock = "auto"
        self.sync_outputs: set[str] | None = None
        self.clock_adjust_s = 0.0
        self.output_delay_s: dict[str, float] = {}
        self.scope: np.ndarray | None = None
        self.scope_prev: np.ndarray | None = None
        self.reset_on_load = True              # gain / EQ / filter back to neutral when a track loads
        # the sound card's cushion: "low" = the least delay; seconds = more headroom
        # against the system pausing Tempest (background, busy machine) - no glitches live
        self.out_latency: str | float = 0.05
        self.phase_lock = True                 # SYNC = tempo + beat phase; False: tempo only (phase by hand)
        self._lead_deck: str | None = None     # the deck the synced ones follow right now (sticky)
        self._boosted = False                  # the audio thread's priority is raised on its first block
        self._boost_ok = False
        self._lead_master_seen = None
        self.cpu_load = 0.0                    # share of each block's time spent rendering (peak, decaying)
        self.slow_blocks = 0                   # blocks that nearly missed their deadline
        self.preview: Engine | None = None
        self.preview_route = "phones"          # phones | master
        self.preview_level = 1.0
        self._driven = False                   # True: rendered by another engine (commands queue)
        # live update (one version hands the running mix to the next): where every deck
        # was at the start of the last block, on the perf_counter clock both processes share,
        # and a scheduled fade of the whole output (start time, seconds, fading in?)
        self.ho_clock: tuple | None = None
        self._ho: tuple | None = None
        self._stream = None
        self._error_logged = False
        self.hooks: list = []          # callables(engine, n) run before each block (automix)
        from .fx import FXUnit
        # two user FX units + one reserved for auto-mix (echo out)
        self.fx_units = [FXUnit("Delay"), FXUnit("Reverb"), FXUnit("Delay")]
        self._lim_gain = 1.0
        self.quantize = True
        self.taps: list = []           # extra consumers of the master (live stream)
        self.recorder = None
        self.callback_errors = 0       # audio callback failures (watchdog/diagnostics)
        self.deck_errors: list[str] = []   # decks stopped because they failed to render (for the UI)
        self._errors_seen: set[str] = set()
        # outputs: optional second sound card for the headphones, and per-output
        # delay (seconds, None = what the driver reports) used to line them up
        self._pstream = None
        self._pring = None
        self.master_delay_s: float | None = None
        self.phones_delay_s: float | None = None
        from .outputs import Delay

        self._mdelay = Delay(SR)
        self._pdelay = Delay(SR)
        self.display_latency = 0.0     # seconds: draw positions this late so they match the sound
        self.extra_outputs: list = []  # more master outputs (Bluetooth speakers...), each in a helper process
        # tap calibration clicks
        self.click_target: str | None = None     # "master" | "phones"
        self.click_period = 0.6
        self.click_times: list[float] = []
        self._next_click = 0.0
        self.paused = False            # master pause (remote / server): fade out, then stop the decks
        self._pause_gain = 1.0
        self._pause_applied = False
        self._paused_decks: list[str] = []
        self.lock = threading.RLock()  # only for non-realtime state queries

    # ---- command API (any thread) -------------------------------------------
    def cmd(self, fn, *args, **kw):
        """Run fn(engine, *args) at the start of the next audio block."""
        self._cmds.put((fn, args, kw))
        if self._stream is None and not self._driven:   # offline / not started: apply immediately
            self._drain()

    def _drain(self):
        while True:
            try:
                fn, args, kw = self._cmds.get_nowait()
            except queue.Empty:
                return
            try:
                fn(self, *args, **kw)
            except Exception:  # noqa: BLE001
                log.exception("engine command failed")

    # ---- deck operations (called inside the audio thread via cmd) -------------
    @staticmethod
    def _load(eng, deck: str, audio, grid, info, peaks, cue: float | None = None,
              mix_in: float | None = None, mix_out: float | None = None, bands=None,
              mix_out_end: float | None = None):
        """cue / mix_in / mix_out in seconds. The cue point defaults to the
        intro marker, else the first audible beat, else the first grid beat."""
        d = eng.decks[deck]
        d.audio, d.grid, d.info, d.peaks = audio, grid, info, peaks
        d.bands = bands
        d.playing = False
        d.loop = None
        d.hotcues = [None] * 8
        d.key_shift = 0.0
        d._loop_in = None
        d.phase_offset, d.nudged = 0.0, False
        # a new track starts with the channel's levels back at neutral - unless a
        # controller is attached: its knobs don't move, so they keep the say
        if eng.reset_on_load:
            d.trim = 1.0
            d.eq = np.ones(3)
            d.filter = 0.0
        d.end_reached = False
        d.mix_in = mix_in * SR if mix_in is not None else None
        d.mix_out = mix_out * SR if mix_out is not None else None
        d.mix_out_end = mix_out_end * SR if mix_out_end is not None and mix_out is not None else None
        if d.mix_in is not None:
            d.cue_point = d.mix_in
        elif cue is not None:
            d.cue_point = max(cue, 0.0) * SR
        else:
            d.cue_point = max(grid.first_beat, 0.0) * SR if grid else 0.0
        d.cue_by_dj = False                  # the automatic cue: LOOP IN moves it until the DJ sets one
        eng._seek(d, d.cue_point)
        target_lufs = -10.0
        d.gain = float(np.clip(10 ** ((target_lufs - info.loudness) / 20), 0.25, 4.0))
        if eng.master_deck == deck:
            eng.master_deck = None

    def load(self, deck: str, audio, grid, info, peaks=None, **kw):
        self.cmd(Engine._load, deck, audio, grid, info, peaks, **kw)

    def load_track(self, deck: str, lt, only_if_stopped: bool = True):
        """Install a LoadedTrack (from dj.loader) on a deck."""
        rb = None
        if self.decks[deck]._rb is None:
            try:
                rb = _rb_new(1.0)              # built here, not inside an audio block (it takes ~12 ms)
            except Exception:  # noqa: BLE001
                rb = None

        def f(e):
            d = e.decks[deck]
            if rb is not None and d._rb is None:
                d._rb, d._rb_drop = rb
                d._rb_ratio, d._pitch, d._rb_reset = 1.0, 1.0, True
            if only_if_stopped and d.playing and not d.cue_preview:
                return                         # never replaces a track that is on air
            # holding CUE (a preview) or scrubbing a stopped deck is not "on air": the load wins
            d.cue_preview = False
            d.scratching = False
            d.scratch_rate = 0.0
            e._load(e, deck, lt.audio, lt.grid, lt.info, lt.peaks, cue=getattr(lt, "first_beat_t", None),
                    mix_in=getattr(lt, "mix_in", None), mix_out=getattr(lt, "mix_out", None),
                    bands=getattr(lt, "bands", None), mix_out_end=getattr(lt, "mix_out_end", None))
            e.decks[deck].phrase_origin = getattr(lt, "phrase_origin", 0)
            if getattr(lt, "cues", None):
                e.decks[deck].hotcues = (list(lt.cues) + [None] * 8)[:8]
        self.cmd(f)

    def set_markers(self, deck: str, mix_in: float | None, mix_out: float | None,
                    mix_out_end: float | None = None):
        """Intro/outro(/outro end) markers in seconds (None clears)."""
        def f(e):
            d = e.decks[deck]
            d.mix_in = mix_in * SR if mix_in is not None else None
            d.mix_out = mix_out * SR if mix_out is not None else None
            d.mix_out_end = mix_out_end * SR if mix_out_end is not None and mix_out is not None else None
            if d.mix_in is not None and not d.playing:
                d.cue_point = d.mix_in
        self.cmd(f)

    def unload(self, deck: str, even_if_playing: bool = False):
        """Empty a deck (ready for the next set). A playing deck is left alone
        unless even_if_playing."""
        def f(e):
            d = e.decks[deck]
            if d.playing and not even_if_playing:
                return
            d.playing = False
            d.cue_preview = False
            d.audio, d.grid, d.peaks, d.bands = None, None, None, None
            d.info = TrackInfo()
            d.loop = None
            d.hotcues = [None] * 8
            d.mix_in = d.mix_out = d.mix_out_end = None
            d.play_pos = d.feed_pos = d.cue_point = 0.0
            d._rb = None
            d.scope = None
            if e.master_deck == deck:
                e.master_deck = None
        self.cmd(f)

    def set_cue(self, deck: str, seconds: float):
        def f(e):
            d = e.decks[deck]
            d.cue_point = max(0.0, seconds) * SR
            d.cue_by_dj = True
            if not d.playing:
                e._seek(d, d.cue_point)
        self.cmd(f)

    def _seek(self, d: Deck, frame: float):
        frame = float(np.clip(frame, 0, max(d.length - 1, 0)))
        d.play_pos = d.feed_pos = frame
        d.start_pos = frame
        d._rb_reset = True
        d.end_reached = False

    def seek(self, deck: str, seconds: float):
        self.cmd(lambda e: e._seek(e.decks[deck], seconds * SR))

    def seek_by(self, deck: str, seconds: float):
        """Move relative to where the deck is now (jog search): computed here,
        from the engine's own position, so a stream of jog ticks never jitters."""
        def f(e):
            d = e.decks[deck]
            if d.loaded:
                e._seek(d, float(np.clip(d.play_pos + seconds * SR, 0.0, max(0.0, d.length - 1))))
        self.cmd(f)

    def _play_now(e, deck: str, on: bool, quantize: bool = True):
        d = e.decks[deck]
        if not d.loaded:
            return
        if on and d.cue_preview:
            d.cue_preview = False          # PLAY while holding CUE: keep playing
            return
        if on and not d.playing:
            if quantize and d.sync and e.phase_lock and e._master(exclude=deck) is not None:
                e._align_phase(d, e._master(exclude=deck))
            d.playing = True
            d.start_pos = d.play_pos
            if e.master_deck is None:
                # a nominal master only - not a MASTER press: the deck already on air keeps the lead
                e.master_deck = deck
                e._lead_master_seen = deck
        elif not on:
            d.cue_preview = False
            d.playing = False
            if e.master_deck == deck:
                e.master_deck = None

    def set_playing(self, deck: str, on: bool, quantize: bool = True):
        self.cmd(Engine._play_now, deck, on, quantize)

    def toggle_play(self, deck: str):
        # decided in the audio thread: a CUE preview counts as "not playing yet"
        def f(e):
            d = e.decks[deck]
            Engine._play_now(e, deck, d.cue_preview or not d.playing)
        self.cmd(f)

    def cue_press(self, deck: str):
        def f(e):
            d = e.decks[deck]
            if not d.loaded:
                return
            if d.playing and not d.cue_preview:
                d.playing = False
                e._seek(d, d.cue_point)
            elif abs(d.play_pos - d.cue_point) < SR * 0.01:
                d.cue_preview = True
                d.playing = True
                d.start_pos = d.play_pos
            else:
                # set the cue where you are - where you *hear* you are (snapped to the beat
                # only with quantise on)
                p = e._quant(d, e.heard_pos(d))
                d.cue_point = max(0.0, p)
                d.cue_by_dj = True
                e._seek(d, d.cue_point)
        self.cmd(f)

    def cue_release(self, deck: str, keep_playing: bool = False):
        """keep_playing: the CUE button was let go somewhere else (dragged off
        it): the preview carries on as normal play."""
        def f(e):
            d = e.decks[deck]
            if d.cue_preview:
                d.cue_preview = False
                if not keep_playing:
                    d.playing = False
                    e._seek(d, d.cue_point)
        self.cmd(f)

    def set(self, deck: str, **kw):
        """Set plain deck attributes (volume, eq, tempo, keylock, sync, ...)."""
        def f(e):
            d = e.decks[deck]
            for k, v in kw.items():
                if k == "eq":
                    d.eq = np.asarray(v, dtype=np.float64)
                elif k == "sync":
                    d.sync = bool(v)
                    d.phase_offset, d.nudged = 0.0, False     # (re)pressing SYNC: on the beat again
                    if not v:
                        e._bake_rate(d)             # keep the current tempo when leaving sync
                        d.phase_corr = 0.0
                    else:
                        m = e._master(exclude=deck)
                        if m is not None and not d.playing and e.phase_lock:
                            e._align_phase(d, m)
                else:
                    setattr(d, k, v)
        self.cmd(f)

    def set_paused(self, on: bool):
        """Pause/resume the whole mix with a short fade (nothing is cut off)."""
        def f(e):
            if on and not e.paused:
                e.paused = True
                e._pause_applied = False
            elif not on and e.paused:
                e.paused = False
                if e._pause_applied:
                    for n in e._paused_decks:
                        if e.decks[n].loaded:
                            e.decks[n].playing = True
                e._paused_decks = []
                e._pause_applied = False
        self.cmd(f)

    def _apply_pause(self, master: np.ndarray, n: int):
        if self.paused and self._pause_applied and any(d.playing for d in self.decks.values()):
            # a deck was started by hand while paused: that ends the pause
            self.paused = False
            self._pause_applied = False
            self._paused_decks = []
        target = 0.0 if self.paused else 1.0
        g0 = self._pause_gain
        if g0 != target:
            step = n / (0.35 * SR)
            g1 = max(target, g0 - step) if target < g0 else min(target, g0 + step)
            master *= np.linspace(g0, g1, n, dtype=np.float32)
            self._pause_gain = g1
        elif target == 0.0:
            master *= 0.0
        if self.paused and self._pause_gain <= 0.0 and not self._pause_applied:
            self._paused_decks = [d.name for d in self.decks.values() if d.playing]
            for d in self.decks.values():
                d.playing = False
            self._pause_applied = True

    def set_master(self, deck: str | None):
        """The MASTER button: that deck leads the synced ones from now on (even
        when it was the nominal master already and another deck had the lead)."""
        def f(e):
            e.master_deck = deck
            e._lead_master_seen = deck
            e._lead_deck = deck
        self.cmd(f)

    def jog(self, deck: str, delta: float, touched: bool):
        """delta: jog movement in 'ticks' (signed). touched: platter touched
        (scratch) vs outer ring (nudge)."""
        def f(e):
            d = e.decks[deck]
            if not d.loaded:
                return
            if touched or not d.playing:
                # scratch / search: move the position directly
                if not d.playing:
                    e._seek(d, d.play_pos + delta * SR / 400.0)
                else:
                    d.scratching = True
                    d.scratch_rate = float(np.clip(delta / 6.0, -8, 8))
            else:
                d.nudge = float(np.clip(d.nudge + delta * 0.002, -0.2, 0.2))
        self.cmd(f)

    def scratch(self, deck: str, rate: float, on: bool = True):
        """Platter held/moved: the audio follows the hand (vinyl mode)."""
        def f(e):
            d = e.decks[deck]
            if not d.loaded:
                return
            d.scratching = on
            d.scratch_rate = float(np.clip(rate, -10, 10))
        self.cmd(f)

    def jog_release(self, deck: str):
        def f(e):
            d = e.decks[deck]
            if d.scratching:
                d.scratching = False
                d.scratch_rate = 0.0
                d._rb_reset = True
                d.feed_pos = d.play_pos
        self.cmd(f)

    def set_hotcues(self, deck: str, cues: list):
        self.cmd(lambda e: setattr(e.decks[deck], "hotcues", (list(cues) + [None] * 8)[:8]))

    def set_loop(self, deck: str, loop):
        """An exact loop (start, end) in source frames, or None."""
        self.cmd(lambda e: setattr(e.decks[deck], "loop", tuple(loop) if loop else None))

    def loop_beats(self, deck: str, beats: float | None):
        def f(e):
            d = e.decks[deck]
            if beats is None or not d.grid:
                d.loop = None
                return
            b = np.floor(d.grid.beat_at(d.time))
            s = d.grid.time_of_beat(b) * SR
            d.loop = (s, s + beats * d.grid.beat_len * SR)
        self.cmd(f)

    # ---- hot cues / loops / jumps -------------------------------------------------
    def heard_pos(self, d: Deck) -> float:
        """The frame coming out of the speakers right now: the render runs
        display_latency ahead of the sound, and a DJ presses CUE / IN / a hot
        cue on what they hear, not on what is being rendered."""
        if d.playing and self.running and self.display_latency > 0:
            return max(0.0, d.play_pos - self.display_latency * SR * d.rate)
        return d.play_pos

    def _quant(self, d: Deck, frame: float) -> float:
        if self.quantize and d.grid:
            b = round(d.grid.beat_at(frame / SR))
            return max(0.0, d.grid.time_of_beat(b) * SR)
        return frame

    def _jump(self, d: Deck, target: float):
        """Jump keeping the beat phase when playing and quantize is on."""
        if d.playing and self.quantize and d.grid:
            cur = d.grid.beat_at(d.time)
            tb = d.grid.beat_at(target / SR)
            target = d.grid.time_of_beat(np.round(tb) + (cur - np.floor(cur))) * SR
        if d._mode == "stretch" and d.playing:
            # keep the stretcher running (no glitch): just move the feed head
            off = target - d.play_pos
            d.play_pos = float(np.clip(target, 0, d.length - 1))
            d.feed_pos = float(np.clip(d.feed_pos + off, 0, d.length - 1))
        else:
            self._seek(d, target)

    def hotcue(self, deck: str, i: int):
        """Press: set the hot cue if empty, otherwise jump to it."""
        def f(e):
            d = e.decks[deck]
            if not d.loaded:
                return
            if d.hotcues[i] is None:
                d.hotcues[i] = e._quant(d, e.heard_pos(d))
            else:
                e._jump(d, d.hotcues[i])
                if not d.playing:
                    d.cue_point = d.hotcues[i]
                    d.cue_by_dj = True
        self.cmd(f)

    def hotcue_delete(self, deck: str, i: int):
        self.cmd(lambda e: e.decks[deck].hotcues.__setitem__(i, None))

    def loop_in(self, deck: str):
        def f(e):
            d = e.decks[deck]
            p = e._quant(d, e.heard_pos(d))
            d._loop_in = p
            if d.loop is not None:
                d.loop = None
            if not getattr(d, "cue_by_dj", False):
                d.cue_point = p                 # no cue of the DJ's own yet: LOOP IN is the cue
        self.cmd(f)

    def loop_out(self, deck: str, beats: float | None = None):
        """OUT: loop from the IN point to here - or, with no IN point set, loop
        the chosen loop length (beats) at once, from the current beat."""
        def f(e):
            d = e.decks[deck]
            start = getattr(d, "_loop_in", None)
            end = e._quant(d, e.heard_pos(d))
            if start is not None and end - start > SR * 0.01:
                d.loop = (start, end)
                d._loop_in = None                 # used: the next OUT loops the chosen length again
            elif d.grid is not None:
                n = float(beats if beats else d.loop_size)
                b = np.floor(d.grid.beat_at(d.time) + 1e-6)
                s = d.grid.time_of_beat(b) * SR
                d.loop = (s, s + n * d.grid.beat_len * SR)
            else:
                return
            if d.flux:
                d.flux_pos = d.play_pos
        self.cmd(f)

    def loop_resize(self, deck: str, factor: float):
        def f(e):
            d = e.decks[deck]
            d.loop_size = float(np.clip(d.loop_size * factor, 0.25, 64.0))   # the chosen length follows
            if d.loop is None:
                return
            s, t = d.loop
            new_len = max((t - s) * factor, SR * 0.01)
            d.loop = (s, s + new_len)
            if d.play_pos >= s + new_len:
                e._jump(d, s + (d.play_pos - s) % new_len)
        self.cmd(f)

    def beatjump(self, deck: str, beats: float):
        def f(e):
            d = e.decks[deck]
            if not (d.loaded and d.grid):
                return
            off = beats * d.grid.beat_len * SR
            if d.loop is not None:
                d.loop = (d.loop[0] + off, d.loop[1] + off)
            e._jump(d, d.play_pos + off)
        self.cmd(f)

    def key_sync(self, deck: str):
        """Shift this deck's key (key lock) by the smallest interval that makes
        it harmonically compatible with the master deck."""
        from ..keys import compatibility

        def f(e):
            d = e.decks[deck]
            m = e._master(exclude=deck)
            if m is None or d.info.key < 0 or m.info.key < 0:
                return
            mk = m.effective_key

            def shifted(s):
                return (d.info.key % 12 + s) % 12 + (12 if d.info.key >= 12 else 0)

            # smallest transposition that mixes well; otherwise the best available
            shifts = sorted(range(-6, 7), key=abs)
            good = [s for s in shifts if compatibility(shifted(s), mk) >= 0.85]
            best = good[0] if good else max(shifts, key=lambda s: compatibility(shifted(s), mk))
            d.key_shift = float(best)
            d.keylock = True
        self.cmd(f)

    def set_grid(self, deck: str, first_beat: float | None = None, bpm: float | None = None):
        def f(e):
            d = e.decks[deck]
            if d.grid is None:
                return
            if bpm:
                d.grid.bpm = float(bpm)
            if first_beat is not None:
                d.grid.first_beat = float(first_beat)
        self.cmd(f)

    def grid_downbeat_here(self, deck: str):
        """Declare the current position (snapped to the nearest beat) as beat 1."""
        def f(e):
            d = e.decks[deck]
            if d.grid is None:
                return
            b = round(d.grid.beat_at(e.heard_pos(d) / SR))
            t = d.grid.time_of_beat(b)
            while t - 4 * d.grid.beat_len >= 0:
                t -= 4 * d.grid.beat_len
            d.grid.first_beat = t
        self.cmd(f)

    # ---- sync helpers ---------------------------------------------------------
    def _master(self, exclude: str | None = None) -> Deck | None:
        """The deck the synced ones follow. A deck only cueing (CUE held) is
        never it: nobody follows a cue preview."""
        def ok(d):
            return d.name != exclude and d.playing and d.grid and not d.cue_preview
        m = self.master_deck
        if m and ok(self.decks[m]):
            return self.decks[m]
        for d in self.decks.values():
            if ok(d) and not d.sync:
                return d
        for d in self.decks.values():
            if ok(d):
                return d
        return None

    @staticmethod
    def sync_ratio(master_bpm: float, deck_bpm: float) -> float:
        """Rate for deck to match master, allowing half/double time."""
        if deck_bpm <= 0 or master_bpm <= 0:
            return 1.0
        r = master_bpm / deck_bpm
        return min((r, r * 2, r / 2), key=lambda x: abs(np.log(x)))

    def _align_phase(self, d: Deck, m: Deck):
        """Jump d so its beat phase equals the master's (used when starting)."""
        if not (d.grid and m.grid):
            return
        mphase = m.grid.beat_at(m.time) % 1.0
        db = d.grid.beat_at(d.time)
        target = np.floor(db) + mphase
        if target - db > 0.5:
            target -= 1
        elif db - target > 0.5:
            target += 1
        self._seek(d, d.grid.time_of_beat(target) * SR)

    def phase_snap(self, deck: str):
        """Put the deck's beats in phase with the master (a jump of under a beat)."""
        def f(e):
            d = e.decks[deck]
            m = e._master(exclude=deck)
            if d.loaded and m is not None:
                e._align_phase(d, m)
                d.phase_offset, d.nudged = 0.0, False
        self.cmd(f)

    def phase_error(self, deck: str) -> float | None:
        """Beat phase difference to the master in beats (-0.5..0.5)."""
        d = self.decks[deck]
        m = self._master(exclude=deck)
        if not (d.grid and m and m.grid and d.loaded):
            return None
        mb = m.grid.beat_at(m.time)
        db = d.grid.beat_at(d.time)
        # half/double time: compare at the master's beat resolution
        k = d.base_rate * d.grid.bpm / max(m.bpm, 1e-9)
        if abs(k - 2) < 0.2:
            db = db / 2
        elif abs(k - 0.5) < 0.1:
            db = db * 2
        return float(((mb - db) + 0.5) % 1.0 - 0.5)

    def _lead(self) -> Deck | None:
        """The deck the synced ones follow - sticky: whichever is leading keeps
        leading while it plays. A deck that starts (CUE, PLAY) never takes the
        lead by itself, so the deck already playing is never bent to re-align
        to the newcomer; pressing MASTER (or auto-mix handing over) moves it."""
        def ok(d):
            return d is not None and d.playing and d.grid is not None and not d.cue_preview
        cur = self.decks.get(self._lead_deck) if self._lead_deck else None
        m = self.master_deck
        if m != self._lead_master_seen:            # MASTER changed: the new one leads once it plays
            self._lead_master_seen = m
            if m and ok(self.decks.get(m)):
                cur = self.decks[m]
        if not ok(cur):
            cur = self._master()
        self._lead_deck = cur.name if cur is not None else None
        return cur

    @staticmethod
    def _bake_rate(d: Deck):
        """A deck that stops following keeps the tempo it was playing at: the
        followed rate becomes its own tempo fader value (the fader range is
        widened when the offset does not fit), never a snap back to its own
        BPM. The DJ moves the fader when they want it moved."""
        r = d.rate_override
        d.rate_override = None
        if r is None or abs(r - 1.0) < 1e-6:
            return
        for rng in (0.06, 0.08, 0.16, 0.5):
            if rng >= d.tempo_range and abs(r - 1.0) <= rng + 1e-9:
                d.tempo_range = rng
                break
        d.tempo = float(np.clip((r - 1.0) / d.tempo_range, -1.0, 1.0))

    def _apply_sync(self):
        lead = self._lead()                        # the deck everyone follows: its tempo fader rules
        hand = any(x.nudge != 0.0 for x in self.decks.values() if x.loaded and x.playing and not x.cue_preview)
        lead_name = lead.name if lead is not None else None
        if lead_name != getattr(self, "_sync_lead_prev", None):
            # a new master (MASTER pressed, a handover): it is in charge of the
            # tempo from now on and nothing else - the beats stay exactly where
            # they are, so every follower re-bases its lock on its current phase
            self._sync_lead_prev = lead_name
            for x in self.decks.values():
                if x.sync and x.playing:
                    x.nudged = True
                    x.phase_corr = 0.0
        for d in self.decks.values():
            if not (d.sync and d.loaded and d.grid) or d is lead:
                # (with SYNC lit on both decks, the master would otherwise follow its follower
                # and its own tempo fader would do nothing)
                d.phase_corr = 0.0
                if d.rate_override is not None and not getattr(d, "gliding", False):
                    self._bake_rate(d)             # stopped following (lead now / sync off): keep the tempo
                continue
            m = lead if lead is not None else self._master(exclude=d.name)
            if m is None:
                d.phase_corr = 0.0
                continue
            d.rate_override = self.sync_ratio(m.bpm, d.grid.bpm)
            if not d.playing or d.scratching or d.cue_preview:
                continue                           # CUE preview: plays from the cue point, untouched
            if not (self.phase_lock or d.auto_phase):
                d.phase_corr = 0.0                 # tempo sync only: the phase is the DJ's
                continue
            err = self.phase_error(d.name)
            if err is None:
                d.phase_corr = 0.0
                continue
            if d.nudge != 0.0 or hand:
                # the DJ is nudging with a jog (this deck or the one it follows):
                # no pulling back while the hand moves - the DJ sets the phase
                d.phase_corr = 0.0
                d.nudged = True
                continue
            if d.nudged:
                # ... and where the nudge leaves it is the new lock point
                d.phase_offset = err
                d.nudged = False
            err = ((err - d.phase_offset) + 0.5) % 1.0 - 0.5
            if abs(err) < 0.004:
                d.phase_corr = 0.0
                continue
            # close the gap over ~half a second, at most +-4 % of speed
            src_seconds = err * d.grid.beat_len
            d.phase_corr = float(np.clip(src_seconds / 0.5, -0.04, 0.04))

    # ---- rendering --------------------------------------------------------------
    def _read(self, d: Deck, n: int) -> np.ndarray:
        """Read n source frames from feed_pos at rate 1 (for the stretcher)."""
        out = np.zeros((2, n), np.float32)
        pos = int(d.feed_pos)
        got = 0
        while got < n:
            if pos < 0:                      # scratched back past the start: silence there
                k = min(n - got, -pos)
                got += k
                pos += k
                continue
            # whole samples on both sides: a fractional loop end must still wrap
            # (comparing int pos to a float end left the stretcher starved - silence)
            if d.loop is not None and int(d.loop[1]) - int(d.loop[0]) > 0 and pos >= int(d.loop[1]):
                pos = int(d.loop[0])
            end = min(int(d.loop[1]), d.length) if d.loop is not None else d.length
            take = min(n - got, end - pos)
            if take <= 0 or pos >= d.length:
                break
            chunk = d.audio[:, pos:pos + take]
            take = min(take, chunk.shape[1])    # the track may have been swapped under us (load / release)
            if take <= 0:
                break
            out[:, got:got + take] = chunk[:, :take]
            got += take
            pos += take
        d.feed_pos = float(pos) + (d.feed_pos - int(d.feed_pos))
        return out

    def _render_vinyl(self, d: Deck, n: int, rate: float) -> np.ndarray:
        idx = d.play_pos + rate * np.arange(n, dtype=np.float64)
        if d.loop is not None and rate > 0:
            ls, le = d.loop
            span = le - ls
            if span > 1:
                over = idx >= le
                idx[over] = ls + (idx[over] - ls) % span
        i0 = np.floor(idx).astype(np.int64)
        frac = (idx - i0).astype(np.float32)
        valid = (i0 >= 0) & (i0 + 1 < d.length)
        i0c = np.clip(i0, 0, max(d.length - 2, 0))
        a = d.audio[:, i0c]
        b = d.audio[:, i0c + 1]
        y = a + (b - a) * frac
        y[:, ~valid] = 0.0
        newpos = float(idx[-1] + rate) if n else d.play_pos
        if d.loop is not None and newpos >= d.loop[1]:
            newpos = d.loop[0] + (newpos - d.loop[0]) % max(d.loop[1] - d.loop[0], 1)
        newpos = min(max(newpos, 0.0), max(float(d.length - 1), 0.0))   # the record stops at its ends
        d.play_pos = d.feed_pos = newpos
        return y

    def _render_stretched(self, d: Deck, n: int, rate: float) -> np.ndarray:
        if d._rb is None:
            # built once per deck (12 ms - too long for inside an audio block, so
            # never again: a cue press, jump or loop only resets it, 0.1 ms)
            d._rb, d._rb_drop = _rb_new(rate)
            d._rb_ratio = rate
            d._pitch = 1.0
            d.feed_pos = d.play_pos
            d._rb_reset = False
        elif d._rb_reset:
            rb = d._rb
            rb.reset()
            pad = rb.get_preferred_start_pad()
            if pad:
                rb.process(np.zeros((2, pad), np.float32))
            d._rb_drop = rb.get_start_delay()
            d.feed_pos = d.play_pos
            d._rb_reset = False
        rb = d._rb
        if abs(rate - d._rb_ratio) > 1e-6:
            rb.time_ratio = 1.0 / rate
            d._rb_ratio = rate
        pitch = 2.0 ** (d.key_shift / 12.0)
        if abs(pitch - d._pitch) > 1e-6:
            rb.pitch_scale = pitch
            d._pitch = pitch
        out = np.zeros((2, n), np.float32)
        have = 0
        guard = 0
        while have < n and guard < 64:
            guard += 1
            avail = rb.available()
            if avail > 0:
                chunk = rb.retrieve(min(avail, n - have + d._rb_drop))
                if d._rb_drop:
                    k = min(d._rb_drop, chunk.shape[1])
                    chunk = chunk[:, k:]
                    d._rb_drop -= k
                m = chunk.shape[1]
                out[:, have:have + m] = chunk
                have += m
                continue
            req = max(int(rb.get_samples_required()), 256)
            rb.process(self._read(d, req))
        d.play_pos += rate * n
        if d.loop is not None and d.play_pos >= d.loop[1]:
            d.play_pos = d.loop[0] + (d.play_pos - d.loop[0]) % max(d.loop[1] - d.loop[0], 1)
        return out

    def _render_deck(self, d: Deck, n: int) -> np.ndarray | None:
        if not d.loaded:
            return None
        if not d.playing and not d.scratching:     # a hand on a stopped record still scrubs audibly
            d._scr_gain = 0.0
            d._mode = "vinyl"                      # the next start (a CUE tap) is instant again
            tail, d._tail = d._tail, None
            if tail is not None and np.any(tail):
                # just stopped: fade from where the waveform was instead of a hard cut (no click)
                return (tail[:, None] * np.linspace(1.0, 0.0, n, dtype=np.float32)[None, :]).astype(np.float32)
            if d._last_gain > 0:
                d._last_gain = 0.0
            return None
        rate = d.rate
        # with key lock on, always stay in the stretcher (even at 1.0x) so the
        # phase lock's tiny rate corrections never toggle modes and click
        # the stretcher only when it changes something: at the original tempo (and
        # key) the plain path starts on the very first sample - instant CUE taps
        # key lock: the stretcher only when its job is audible - the tempo fader moved, or a
        # jog nudge / a big sync correction (more than ~20 cents). Sync's small, short phase
        # corrections play through the clean path: the stretcher smears transients (a
        # faint "reverb" on everything), and a few cents for half a second is inaudible.
        # Once in, it stays until the speed has been back at normal for ~1.5 s (no flip-flop).
        moved = abs(d.base_rate - 1.0) > 1e-3
        bent = abs(d.nudge) > 0.012 or abs(d.phase_corr) > 0.012
        calm = abs(d.nudge) < 0.003 and abs(d.phase_corr) < 0.003
        if d._mode == "stretch" and not moved and calm:
            d._calm_frames += n
        else:
            d._calm_frames = 0
        stay = d._mode == "stretch" and d._calm_frames < 1.5 * SR
        locked = d.keylock and (moved or bent or stay)
        use_stretch = (locked or abs(d.key_shift) > 1e-6) and not d.scratching and rate > 0.25
        if use_stretch:
            if d._mode != "stretch":
                d._rb_reset = True             # coming from the plain path: start the stretcher here
            d._mode = "stretch"
            y = self._render_stretched(d, n, rate)
        else:
            if d._mode == "stretch":
                d._rb_reset = True
                d._mode = "vinyl"
            y = self._render_vinyl(d, n, rate)
        if d.scratching:
            # a record held still is silent (no DC thump); the sound follows the hand's speed
            g = float(min(1.0, abs(rate) / 0.08))
            if g != d._scr_gain or g < 1.0:
                y = y * np.linspace(d._scr_gain, g, n, dtype=np.float32)[None, :]
            d._scr_gain = g
        else:
            d._scr_gain = 1.0
        if d.play_pos >= d.length - 1 and d.loop is None and d.playing:
            d.playing = False
            d.end_reached = True
            if self.master_deck == d.name:
                self.master_deck = None
        if d.nudge != 0.0:
            d.nudge *= 0.85            # nudges decay quickly once the jog stops
            if abs(d.nudge) < 1e-4:
                d.nudge = 0.0
        if d.flux:
            if d.loop is None and getattr(d, "_flux_armed", False):
                # loop released in flux mode: continue where the track would be
                d._flux_armed = False
                self._jump(d, d.flux_pos)
            elif d.loop is not None:
                d._flux_armed = True
            d.flux_pos += rate * n if d.loop is not None else 0.0
            if d.loop is None:
                d.flux_pos = d.play_pos
        if d._eq is None:
            d._eq = ThreeBandEQ()
        y = d._eq.process(y, d.eq) * (d.gain * d.trim)
        if d._filter is None:
            from .fx import DeckFilter
            d._filter = DeckFilter()
        y = d._filter.process(y, d.filter)
        if d.fx and d.grid is not None:
            beat_s = d.grid.beat_len / max(d.base_rate, 1e-3)
            bpos = d.grid.beat_at(d.time)
            binc = rate / SR / d.grid.beat_len
            for u in sorted(d.fx):
                y = self.fx_units[u].process(d.name, y, beat_s, bpos, binc)
        d._tail = np.array(y[:, -1], dtype=np.float32) if y.shape[1] else None
        return y

    def _xf_gain(self, side: int) -> float:
        """Club-style curve: full level up to the centre, then a smooth
        (constant-power) fade to silence at the far side."""
        if side == 0:
            return 1.0
        x = float(np.clip((self.crossfader * side + 1.0) / 2.0, 0.0, 1.0))  # 1 = towards this side
        return 1.0 if x >= 0.5 else float(np.sin(x * np.pi))

    # ---- live update ----------------------------------------------------------------
    def handover_clock(self):
        """(t, {deck: (frame at t, frames per second)}) - where the decks are, for the next version."""
        return self.ho_clock

    def handover_fade(self, t_start: float, seconds: float, fade_in: bool):
        """Fade the whole output in or out starting at perf_counter time t_start
        (inf: hold - muted when fading in, as is when fading out)."""
        self._ho = (float(t_start), max(float(seconds), 1e-3), bool(fade_in))

    def handover_align(self, clock):
        """Put every deck where the other version's decks are right now (from its
        handover_clock): positions run on at each deck's speed since that clock."""
        def f(e):
            import time as _t

            if not clock:
                return
            t_ref, pos = clock
            now = _t.perf_counter()
            for name, (frame, fps) in pos.items():
                d = e.decks.get(name)
                if d is not None and d.loaded:
                    e._seek(d, float(frame) + (now - float(t_ref)) * float(fps))
        self.cmd(f)

    def _ho_gain(self, t0: float, n: int):
        if self._ho is None:
            return None
        t_start, dur, fade_in = self._ho
        t = t0 + np.arange(n, dtype=np.float64) / SR
        k = np.clip((t - t_start) / dur, 0.0, 1.0) if np.isfinite(t_start) else np.zeros(n)
        g = np.sin(k * np.pi / 2) if fade_in else np.cos(k * np.pi / 2)      # equal power
        g[k >= 1.0] = 1.0 if fade_in else 0.0                                 # exact ends: truly silent
        g[k <= 0.0] = 0.0 if fade_in else 1.0
        if fade_in and np.isfinite(t_start) and t0 > t_start + dur:
            self._ho = None                                                  # done: full level from now
        return g.astype(np.float32)

    def render(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Mix n frames; returns (master (2,n), phones (2,n))."""
        import time as _t

        self._drain()
        t_block = _t.perf_counter()
        self.ho_clock = (t_block, {name: (float(d.play_pos), float(d.rate) * SR if d.playing else 0.0)
                                   for name, d in self.decks.items() if d.loaded})
        for h in list(self.hooks):
            try:
                h(self, n)
            except Exception:  # noqa: BLE001
                log.exception("engine hook failed")
                self.hooks.remove(h)
        self._apply_sync()
        master = np.zeros((2, n), np.float32)
        cue = np.zeros((2, n), np.float32)
        for d in self.decks.values():
            try:
                y = self._render_deck(d, n)
            except Exception as ex:  # noqa: BLE001 - one bad deck must never silence the mix
                log.exception("deck %s failed to render %s - stopping it", d.name, d.info.path)
                d.playing = False
                self.deck_errors.append(f"Deck {d.name} stopped: {type(ex).__name__}: {ex}")
                y = None
            if y is None:
                d.level *= 0.85
                continue
            g = d.volume ** 2 * self._xf_gain(d.xf_side)   # squared: audio-taper fader
            if g == d._last_gain:                          # fader still: no ramp to build
                master += y * np.float32(g)
            else:
                master += y * np.linspace(d._last_gain, g, n, dtype=np.float32)
            d._last_gain = g
            d.scope = y                                    # newest block, for the XY scope (no copy)
            pre = float(np.abs(y).max()) if n else 0.0
            d.level = max(pre * d.volume ** 2, d.level * 0.85)   # channel meter (post-fader)
            if d.cue_phones:
                cue += y
        self._apply_pause(master, n)
        pv = None
        if self.preview is not None:
            try:
                pv = self.preview.render(n)[0] * self.preview_level
            except Exception:  # noqa: BLE001 - the preview must never stop the live mix
                log.exception("preview render failed")
                self.preview = None
            if pv is not None and self.preview_route == "master":
                master += pv
                pv = None
        master *= self.master_volume
        # brick-wall peak limiter: instant attack, ~150 ms release; transparent
        # below -1 dBFS and never distorts when two loud tracks overlap
        peak = float(np.abs(master).max()) if n else 0.0
        ceiling = 0.89
        target = min(1.0, ceiling / peak) if peak > 1e-9 else 1.0
        if target < self._lim_gain:
            g1 = target                                # attack: at once
        else:
            g1 = self._lim_gain + (target - self._lim_gain) * (1 - np.exp(-n / (0.15 * SR)))
        if self._lim_gain != 1.0 or g1 != 1.0:
            master *= np.linspace(min(self._lim_gain, g1), g1, n, dtype=np.float32)
        self._lim_gain = g1
        self.peak_level = max(min(peak * g1, 1.0), self.peak_level * 0.9)
        np.clip(master, -0.99, 0.99, out=master)
        self.scope_prev, self.scope = self.scope, master     # the oscilloscope shows the last two blocks
        phones = (cue * (1 - self.cue_mix) + master * self.cue_mix) * self.phones_volume
        if pv is not None:
            phones = phones + pv * self.phones_volume
        if self.recorder is not None:
            self.recorder.write(master)
        for tap in self.taps:
            tap.write(master)
        self.frames_rendered += n
        phones = np.tanh(phones)
        g = self._ho_gain(t_block, n)
        if g is not None:
            master = master * g
            phones = phones * g
        return master, phones

    # ---- device -------------------------------------------------------------------
    def _clicks(self, master: np.ndarray, phones: np.ndarray, n: int, t0: float):
        """Calibration clicks (1 kHz blips) mixed into one output; the time
        each one was generated is kept to compare with the DJ's taps."""
        import time as _t

        now = t0 if t0 else _t.monotonic()
        if self._next_click < now:
            self._next_click = now + 0.3
        target = phones if self.click_target == "phones" else master
        while self._next_click < now + n / SR:
            k = int((self._next_click - now) * SR)
            m = min(int(0.03 * SR), n - k)
            if m > 0:
                tt = np.arange(m) / SR
                blip = (0.6 * np.sin(2 * np.pi * 1000 * tt) * np.exp(-tt * 120)).astype(np.float32)
                target[:, k:k + m] += blip
                self.click_times.append(self._next_click)
                del self.click_times[:-200]
            self._next_click += self.click_period

    def _route(self, master: np.ndarray, phones: np.ndarray, frames: int):
        """Line the outputs up (delay the faster one) and send the headphones
        to their own sound card when they have one."""
        import time as _t

        extra_click = None
        if self.click_target and self.click_target.startswith("extra:"):
            extra_click = self.click_target[6:]          # clicks only on that speaker
            clicked = master.copy()
            self._clicks(clicked, phones.copy(), frames, _t.monotonic())
        elif self.click_target:
            self._clicks(master, phones, frames, _t.monotonic())
        for eo in tuple(self.extra_outputs):
            eo.feed(clicked if extra_click == eo.label else master)
        master = self._mdelay.process(master)
        phones = self._pdelay.process(phones)
        if self._pring is not None:
            self._pring.write(np.ascontiguousarray(phones.T))
        return master, phones

    def _callback(self, outdata, frames, time_info, status):
        import time as _t

        if not self._boosted:
            # the first block: this thread is the one that must never wait
            from .outputs import boost_audio_thread

            self._boosted = True
            self._boost_ok = boost_audio_thread(frames / SR)
        if status and status.output_underflow:
            self.underruns += 1
        t0 = _t.perf_counter()
        try:
            master, phones = self.render(frames)
            master, phones = self._route(master, phones, frames)
            outdata.fill(0)
            outdata[:, self.master_chans[0]] = master[0]
            outdata[:, self.master_chans[1]] = master[1]
            if self.phones_chans is not None and self._pring is None:
                outdata[:, self.phones_chans[0]] = phones[0]
                outdata[:, self.phones_chans[1]] = phones[1]
            # how close each block came to its deadline (stutter warning)
            load = (_t.perf_counter() - t0) / (frames / SR)
            self.cpu_load = max(load, self.cpu_load * 0.98)
            if load > 0.85:
                self.slow_blocks += 1
        except Exception as ex:  # noqa: BLE001
            outdata.fill(0)
            self.callback_errors += 1
            key = f"{type(ex).__name__}: {ex}"
            if key not in self._errors_seen and len(self._errors_seen) < 50:
                self._errors_seen.add(key)
                log.exception("audio callback error (#%d)", self.callback_errors)

    def _pcallback(self, outdata, frames, time_info, status):
        """Headphones on their own sound card: play what the main callback queued."""
        try:
            x = self._pring.read(frames)
            outdata.fill(0)
            c = self.phones_chans or (0, 1)
            outdata[:, c[0]] = x[:, 0]
            outdata[:, c[1]] = x[:, 1]
        except Exception:  # noqa: BLE001
            outdata.fill(0)

    def start(self, device=None, block: int | None = None, master_chans=(0, 1), phones_chans=None,
              phones_device=None):
        """Open the output(s); falls back to safer settings rather than failing.
        phones_device: a different sound card for the headphones (None = the
        master's card, channels phones_chans)."""
        import sounddevice as sd

        self.stop()
        self._boosted = False
        if block:
            self.block = block
        attempts = [(device, self.block), (device, max(self.block, 1024))]
        if device is not None:
            attempts.append((None, max(self.block, 1024)))      # system default as last resort
        last = None
        separate = phones_device is not None and phones_device != device
        for dev, blk in attempts:
            try:
                self._open(sd, dev, blk, master_chans, None if separate else phones_chans)
                break
            except Exception as e:  # noqa: BLE001
                last = e
                log.warning("audio open failed on %s (block %d): %s", dev, blk, e)
        else:
            raise RuntimeError(f"no usable audio output: {last}")
        if separate:
            try:
                self._open_phones(sd, phones_device, phones_chans or (0, 1))
            except Exception as e:  # noqa: BLE001
                log.warning("headphone device %s failed: %s - headphones off", phones_device, e)
                self.phones_chans = None
        self.update_alignment()

    def _open_phones(self, sd, device, chans):
        from .outputs import Ring

        info = sd.query_devices(device, "output")
        n = int(min(info["max_output_channels"], max(2, max(chans) + 1)))
        if max(chans) >= n:
            chans = (0, 1)
        self._pring = Ring(target=max(2 * self.block, 1024))
        extra = None
        if "WASAPI" in sd.query_hostapis(info["hostapi"])["name"]:
            extra = sd.WasapiSettings(auto_convert=True)
        st = sd.OutputStream(samplerate=SR, blocksize=self.block, channels=n, dtype="float32", device=device,
                             latency=self.out_latency, callback=self._pcallback, extra_settings=extra)
        st.start()
        self._pstream = st
        self.phones_chans = chans
        log.info("headphones on %s (%d ch)", info["name"], n)

    # ---- output delays -------------------------------------------------------------------
    def reported_latency(self, phones: bool = False) -> float:
        st = self._pstream if phones and self._pstream is not None else self._stream
        try:
            return float(st.latency) if st is not None and not isinstance(st, _VirtualOutput) else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def add_output(self, eo, delay_s=None, chans=(0, 1)) -> None:
        """Another master output: an ExtraOutput, or a device label (one is
        started for it); everything is re-lined up."""
        if isinstance(eo, str):
            from . import outputs

            eo = outputs.ExtraOutput(eo, chans=tuple(chans), delay_s=delay_s, block=self.block)
        self.extra_outputs = [o for o in self.extra_outputs if o.label != eo.label] + [eo]
        self.update_alignment()

    def set_output_delay(self, label: str, delay_s) -> None:
        for o in self.extra_outputs:
            if o.label == label:
                o.delay_s = delay_s
        self.update_alignment()

    def calibrate(self, which: str | None) -> None:
        """Tap test: clicks on one output with the alignment delays off
        (which = "master" | "phones" | "extra:LABEL"); None ends it."""
        if which:
            self._cal_saved = (self._mdelay.frames, self._pdelay.frames)
            self._mdelay.frames = self._pdelay.frames = 0
            self.click_times.clear()
            self.click_target = which
        else:
            self.click_target = None
            saved = getattr(self, "_cal_saved", None)
            if saved:
                self._mdelay.frames, self._pdelay.frames = saved
                self._cal_saved = None

    def remove_output(self, label: str) -> None:
        keep, gone = [], []
        for o in self.extra_outputs:
            (gone if o.label == label else keep).append(o)
        self.extra_outputs = keep
        for o in gone:
            o.close()
        self.update_alignment()

    def output_delays(self) -> tuple[float, float]:
        """(master, headphones) delay in seconds: measured/entered, else reported."""
        m = self.master_delay_s if self.master_delay_s is not None else self.reported_latency()
        if self._pstream is None and self.phones_delay_s is None:
            return m, m                       # same sound card: same delay
        p = self.phones_delay_s if self.phones_delay_s is not None else self.reported_latency(phones=True)
        return m, p

    def output_latencies(self) -> dict[str, float]:
        """How late each output plays (seconds): "master", "phones" (when used)
        and "extra:<device>" for every extra speaker."""
        m, p = self.output_delays()
        lat = {"master": m}
        if self.phones_chans is not None:
            lat["phones"] = p
        for o in self.extra_outputs:
            lat[f"extra:{o.label}"] = o.effective_delay
        return lat

    def update_alignment(self):
        """Line the outputs up with the clock output: every output set to
        follow the clock is held back until it plays the same beat at the same
        moment as the clock (+ the DJ's adjustment); the others play at once.
        Waveforms are drawn as late as the main output plays."""
        lat = self.output_latencies()
        synced = [k for k in lat if self.sync_outputs is None or k in self.sync_outputs]
        if self.clock in lat:
            base = lat[self.clock]
        else:                                   # auto: the slowest output that follows the clock
            base = max((lat[k] for k in synced), default=lat["master"])
        target = max(0.0, base + self.clock_adjust_s)
        delay = {k: (max(0.0, target - v) if k in synced else 0.0) for k, v in lat.items()}
        self._mdelay.frames = int(round(delay["master"] * SR))
        self._pdelay.frames = int(round(delay.get("phones", 0.0) * SR))
        for o in self.extra_outputs:
            o.align.frames = int(round(delay[f"extra:{o.label}"] * SR))
        self.output_delay_s = delay
        self.display_latency = lat["master"] + delay["master"]

    def _open(self, sd, device, block, master_chans, phones_chans):
        info = sd.query_devices(device, "output")
        api = sd.query_hostapis(info["hostapi"])["name"]
        need = max(max(master_chans), max(phones_chans) if phones_chans else 0) + 1
        chans = int(min(info["max_output_channels"], max(2, need)))
        if max(master_chans) >= chans:
            master_chans = (0, 1)
        if phones_chans and max(phones_chans) >= chans:
            phones_chans = None
        extra = None
        if "WASAPI" in api:
            # shared mode only accepts the device's own rate unless Windows converts
            extra = sd.WasapiSettings(auto_convert=True)
        stream = sd.OutputStream(samplerate=SR, blocksize=block, channels=chans, dtype="float32",
                                 device=device, latency=self.out_latency, callback=self._callback, extra_settings=extra)
        stream.start()
        self._stream = stream
        self.block = block
        self.master_chans = master_chans
        self.phones_chans = phones_chans
        self.out_channels = chans
        log.info("audio started on %s via %s (%d ch, block %d)", info["name"], api, chans, block)

    def start_virtual(self, block: int = 1024):
        """No sound card: render in real time anyway so the mix keeps
        streaming to phones (and recording)."""
        self.stop()
        self.block = block
        self.out_channels = 2
        self.master_chans = (0, 1)
        self.phones_chans = None
        self._stream = _VirtualOutput(self, block)
        self._stream.start()
        log.info("no audio device: virtual output (stream/record only)")

    @property
    def virtual(self) -> bool:
        return isinstance(self._stream, _VirtualOutput)

    def close_outputs(self):
        for o in tuple(self.extra_outputs):
            o.close()
        self.extra_outputs = []

    @classmethod
    def stop_all(cls):
        """Stop every engine's audio (used when the process is about to exit)."""
        for e in list(cls._instances):
            try:
                e.stop()
            except Exception:  # noqa: BLE001
                pass

    def stop(self):
        if self._pstream is not None:
            try:
                self._pstream.abort()
                self._pstream.close()
            except Exception:  # noqa: BLE001
                pass
            self._pstream = None
            self._pring = None
        if self._stream is not None:
            try:
                self._stream.abort()          # never waits on a hung device
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None

    @property
    def running(self) -> bool:
        return self._stream is not None

class _VirtualOutput:
    """Clock-paced render thread standing in for a sound card."""

    latency = 0.0

    def __init__(self, eng: Engine, block: int):
        self.eng = eng
        self.block = block
        self._run = False
        self._t: threading.Thread | None = None

    def start(self):
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True, name="virtual-audio")
        self._t.start()

    def _loop(self):
        import time

        period = self.block / SR
        nxt = time.monotonic()
        while self._run:
            try:
                self.eng.render(self.block)
            except Exception:  # noqa: BLE001
                log.exception("virtual render failed")
            nxt += period
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.025:
                self.eng.underruns += 1           # later than a sound card's cushion: a real card would glitch
                if delay < -1.0:
                    nxt = time.monotonic()        # fell far behind: do not race to catch up

    def stop(self):
        self._run = False
        if self._t is not None and self._t is not threading.current_thread():
            self._t.join(2)

    abort = stop          # Engine.stop() aborts outputs: the render thread must end too

    def close(self):
        pass


def list_output_devices():
    """[(index, label, channels)], best host API first (WASAPI/ASIO/CoreAudio)."""
    import sounddevice as sd

    rank = {"ASIO": 0, "Windows WASAPI": 1, "Core Audio": 1, "Windows DirectSound": 2, "MME": 3}
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            api = sd.query_hostapis(d["hostapi"])["name"]
            out.append((rank.get(api, 2), i, f"{d['name']} ({api}, {d['max_output_channels']} ch)",
                        d["max_output_channels"]))
    out.sort(key=lambda r: (r[0], r[1]))
    return [(i, label, ch) for _r, i, label, ch in out]
