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
        from scipy.signal import sosfilt

        if ThreeBandEQ._filters is None:
            ThreeBandEQ._filters = (_lr4(EQ_LOW_HZ, "low"), _lr4(EQ_LOW_HZ, "high"),
                                    _lr4(EQ_HIGH_HZ, "low"), _lr4(EQ_HIGH_HZ, "high"))
        self._sosfilt = sosfilt
        self.states = [np.zeros((f.shape[0], 2, 2)) for f in ThreeBandEQ._filters]
        self.gains = np.ones(3)

    def _f(self, i: int, x: np.ndarray) -> np.ndarray:
        y, self.states[i] = self._sosfilt(ThreeBandEQ._filters[i], x, axis=1, zi=self.states[i])
        return y

    def process(self, x: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """x: (2, n) float. gains: (low, mid, high) linear."""
        low = self._f(0, x)
        rest = self._f(1, x)
        mid = self._f(2, rest)
        high = self._f(3, rest)
        n = x.shape[1]
        ramp = np.linspace(0.0, 1.0, n)
        g = [self.gains[i] + (gains[i] - self.gains[i]) * ramp for i in range(3)]
        self.gains = np.array(gains, dtype=np.float64)
        return (low * g[0] + mid * g[1] + high * g[2]).astype(np.float32)


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


@dataclass
class Deck:
    name: str
    audio: np.ndarray | None = None          # (2, n) float32 @ SR
    grid: BeatGrid | None = None
    info: TrackInfo = field(default_factory=TrackInfo)
    peaks: np.ndarray | None = None          # waveform overview for the UI
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
    bands: object = None                     # waveform bands (frames, 3) for analysis/drawing
    _filter: object = None
    _pitch: float = 1.0
    _rb: object = None
    _rb_ratio: float = 1.0
    _rb_drop: int = 0
    _mode: str = "vinyl"
    _eq: ThreeBandEQ | None = None
    _last_gain: float = 0.0

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
        self._stream = None
        self._error_logged = False
        self.hooks: list = []          # callables(engine, n) run before each block (automix)
        from .fx import FXUnit
        self.fx_units = [FXUnit("Delay"), FXUnit("Reverb")]
        self.quantize = True
        self.taps: list = []           # extra consumers of the master (live stream)
        self.recorder = None
        self.lock = threading.RLock()  # only for non-realtime state queries

    # ---- command API (any thread) -------------------------------------------
    def cmd(self, fn, *args, **kw):
        """Run fn(engine, *args) at the start of the next audio block."""
        self._cmds.put((fn, args, kw))
        if self._stream is None:      # offline / not started: apply immediately
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
              mix_in: float | None = None, mix_out: float | None = None, bands=None):
        """cue / mix_in / mix_out in seconds. The cue point defaults to the
        intro marker, else the first audible beat, else the first grid beat."""
        d = eng.decks[deck]
        d.audio, d.grid, d.info, d.peaks = audio, grid, info, peaks
        d.bands = bands
        d.playing = False
        d.loop = None
        d.hotcues = [None] * 8
        d.key_shift = 0.0
        d.end_reached = False
        d.mix_in = mix_in * SR if mix_in is not None else None
        d.mix_out = mix_out * SR if mix_out is not None else None
        if d.mix_in is not None:
            d.cue_point = d.mix_in
        elif cue is not None:
            d.cue_point = max(cue, 0.0) * SR
        else:
            d.cue_point = max(grid.first_beat, 0.0) * SR if grid else 0.0
        eng._seek(d, d.cue_point)
        target_lufs = -10.0
        d.gain = float(np.clip(10 ** ((target_lufs - info.loudness) / 20), 0.25, 4.0))
        if eng.master_deck == deck:
            eng.master_deck = None

    def load(self, deck: str, audio, grid, info, peaks=None, **kw):
        self.cmd(Engine._load, deck, audio, grid, info, peaks, **kw)

    def load_track(self, deck: str, lt, only_if_stopped: bool = True):
        """Install a LoadedTrack (from dj.loader) on a deck."""
        def f(e):
            if only_if_stopped and e.decks[deck].playing:
                return
            e._load(e, deck, lt.audio, lt.grid, lt.info, lt.peaks, cue=getattr(lt, "first_beat_t", None),
                    mix_in=getattr(lt, "mix_in", None), mix_out=getattr(lt, "mix_out", None),
                    bands=getattr(lt, "bands", None))
            if getattr(lt, "cues", None):
                e.decks[deck].hotcues = (list(lt.cues) + [None] * 8)[:8]
        self.cmd(f)

    def set_markers(self, deck: str, mix_in: float | None, mix_out: float | None):
        """Intro/outro markers in seconds (None clears)."""
        def f(e):
            d = e.decks[deck]
            d.mix_in = mix_in * SR if mix_in is not None else None
            d.mix_out = mix_out * SR if mix_out is not None else None
            if d.mix_in is not None and not d.playing:
                d.cue_point = d.mix_in
        self.cmd(f)

    def set_cue(self, deck: str, seconds: float):
        def f(e):
            d = e.decks[deck]
            d.cue_point = max(0.0, seconds) * SR
            if not d.playing:
                e._seek(d, d.cue_point)
        self.cmd(f)

    def _seek(self, d: Deck, frame: float):
        frame = float(np.clip(frame, 0, max(d.length - 1, 0)))
        d.play_pos = d.feed_pos = frame
        d._rb = None
        d.end_reached = False

    def seek(self, deck: str, seconds: float):
        self.cmd(lambda e: e._seek(e.decks[deck], seconds * SR))

    def set_playing(self, deck: str, on: bool, quantize: bool = True):
        def f(e):
            d = e.decks[deck]
            if not d.loaded:
                return
            if on and not d.playing:
                if quantize and d.sync and e._master(exclude=deck) is not None:
                    e._align_phase(d, e._master(exclude=deck))
                d.playing = True
                if e.master_deck is None:
                    e.master_deck = deck
            elif not on:
                d.playing = False
                if e.master_deck == deck:
                    e.master_deck = None
        self.cmd(f)

    def toggle_play(self, deck: str):
        self.set_playing(deck, not self.decks[deck].playing)

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
            else:
                # set cue at the nearest beat (quantised)
                p = d.play_pos
                if d.grid:
                    b = round(d.grid.beat_at(p / SR))
                    p = d.grid.time_of_beat(b) * SR
                d.cue_point = max(0.0, p)
                e._seek(d, d.cue_point)
        self.cmd(f)

    def cue_release(self, deck: str):
        def f(e):
            d = e.decks[deck]
            if d.cue_preview:
                d.cue_preview = False
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
                    if not v:
                        # keep the current tempo when leaving sync
                        if d.rate_override is not None and d.tempo_range > 0:
                            d.tempo = float(np.clip((d.rate_override - 1.0) / d.tempo_range, -1, 1))
                        d.rate_override = None
                        d.phase_corr = 0.0
                    else:
                        m = e._master(exclude=deck)
                        if m is not None and not d.playing:
                            e._align_phase(d, m)
                else:
                    setattr(d, k, v)
        self.cmd(f)

    def set_master(self, deck: str | None):
        self.cmd(lambda e: setattr(e, "master_deck", deck))

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
                d._rb = None
                d.feed_pos = d.play_pos
        self.cmd(f)

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
                d.hotcues[i] = e._quant(d, d.play_pos)
            else:
                e._jump(d, d.hotcues[i])
                if not d.playing:
                    d.cue_point = d.hotcues[i]
        self.cmd(f)

    def hotcue_delete(self, deck: str, i: int):
        self.cmd(lambda e: e.decks[deck].hotcues.__setitem__(i, None))

    def set_hotcues(self, deck: str, frames: list):
        def f(e):
            d = e.decks[deck]
            d.hotcues = (list(frames) + [None] * 8)[:8]
        self.cmd(f)

    def loop_in(self, deck: str):
        def f(e):
            d = e.decks[deck]
            p = e._quant(d, d.play_pos)
            d._loop_in = p
            if d.loop is not None:
                d.loop = None
        self.cmd(f)

    def loop_out(self, deck: str):
        def f(e):
            d = e.decks[deck]
            start = getattr(d, "_loop_in", None)
            if start is None:
                return
            end = e._quant(d, d.play_pos)
            if end - start > SR * 0.01:
                d.loop = (start, end)
                if d.flux:
                    d.flux_pos = d.play_pos
        self.cmd(f)

    def loop_resize(self, deck: str, factor: float):
        def f(e):
            d = e.decks[deck]
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
            b = round(d.grid.beat_at(d.time))
            t = d.grid.time_of_beat(b)
            while t - 4 * d.grid.beat_len >= 0:
                t -= 4 * d.grid.beat_len
            d.grid.first_beat = t
        self.cmd(f)

    # ---- sync helpers ---------------------------------------------------------
    def _master(self, exclude: str | None = None) -> Deck | None:
        m = self.master_deck
        if m and m != exclude and self.decks[m].playing and self.decks[m].grid:
            return self.decks[m]
        for d in self.decks.values():
            if d.name != exclude and d.playing and d.grid and not d.sync:
                return d
        for d in self.decks.values():
            if d.name != exclude and d.playing and d.grid:
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

    def _apply_sync(self):
        for d in self.decks.values():
            if not (d.sync and d.loaded and d.grid):
                d.phase_corr = 0.0
                continue
            m = self._master(exclude=d.name)
            if m is None:
                d.phase_corr = 0.0
                continue
            d.rate_override = self.sync_ratio(m.bpm, d.grid.bpm)
            if not d.playing or d.scratching:
                continue
            err = self.phase_error(d.name)
            if err is None or abs(err) < 0.004:
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
            if d.loop is not None and pos >= d.loop[1]:
                pos = int(d.loop[0])
            end = int(d.loop[1]) if d.loop is not None else d.length
            take = min(n - got, end - pos)
            if take <= 0 or pos >= d.length:
                break
            out[:, got:got + take] = d.audio[:, pos:pos + take]
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
        d.play_pos = d.feed_pos = newpos
        return y

    def _render_stretched(self, d: Deck, n: int, rate: float) -> np.ndarray:
        if d._rb is None:
            d._rb, d._rb_drop = _rb_new(rate)
            d._rb_ratio = rate
            d._pitch = 1.0
            d.feed_pos = d.play_pos
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
        if not d.playing:
            if d._last_gain > 0:
                d._last_gain = 0.0
            return None
        rate = d.rate
        # with key lock on, always stay in the stretcher (even at 1.0x) so the
        # phase lock's tiny rate corrections never toggle modes and click
        use_stretch = (d.keylock or abs(d.key_shift) > 1e-6) and not d.scratching and rate > 0.25
        if use_stretch:
            d._mode = "stretch"
            y = self._render_stretched(d, n, rate)
        else:
            if d._mode == "stretch":
                d._rb = None
                d._mode = "vinyl"
            y = self._render_vinyl(d, n, rate)
        if d.play_pos >= d.length - 1 and d.loop is None:
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
        return y

    def _xf_gain(self, side: int) -> float:
        """Club-style curve: full level up to the centre, then a smooth
        (constant-power) fade to silence at the far side."""
        if side == 0:
            return 1.0
        x = float(np.clip((self.crossfader * side + 1.0) / 2.0, 0.0, 1.0))  # 1 = towards this side
        return 1.0 if x >= 0.5 else float(np.sin(x * np.pi))

    def render(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Mix n frames; returns (master (2,n), phones (2,n))."""
        self._drain()
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
            y = self._render_deck(d, n)
            if y is None:
                d.level *= 0.85
                continue
            g = d.volume ** 2 * self._xf_gain(d.xf_side)   # squared: audio-taper fader
            ramp = np.linspace(d._last_gain, g, n, dtype=np.float32)
            d._last_gain = g
            pre = float(np.abs(y).max()) if n else 0.0
            d.level = max(pre * d.volume ** 2, d.level * 0.85)   # channel meter (post-fader)
            master += y * ramp
            if d.cue_phones:
                cue += y
        master *= self.master_volume
        # soft limiter: transparent below -1 dBFS
        peak = float(np.abs(master).max()) if n else 0.0
        self.peak_level = max(peak, self.peak_level * 0.9)
        if peak > 0.89:
            master = np.tanh(master)
        phones = (cue * (1 - self.cue_mix) + master * self.cue_mix) * self.phones_volume
        if self.recorder is not None:
            self.recorder.write(master)
        for tap in self.taps:
            tap.write(master)
        self.frames_rendered += n
        return master, np.tanh(phones)

    # ---- device -------------------------------------------------------------------
    def _callback(self, outdata, frames, time_info, status):
        if status and status.output_underflow:
            self.underruns += 1
        try:
            master, phones = self.render(frames)
            outdata.fill(0)
            outdata[:, self.master_chans[0]] = master[0]
            outdata[:, self.master_chans[1]] = master[1]
            if self.phones_chans is not None:
                outdata[:, self.phones_chans[0]] = phones[0]
                outdata[:, self.phones_chans[1]] = phones[1]
        except Exception:  # noqa: BLE001
            outdata.fill(0)
            if not self._error_logged:
                log.exception("audio callback error")
                self._error_logged = True

    def start(self, device=None, block: int | None = None, master_chans=(0, 1), phones_chans=None):
        """Open the output; falls back to safer settings rather than failing."""
        import sounddevice as sd

        self.stop()
        if block:
            self.block = block
        attempts = [(device, self.block), (device, max(self.block, 1024))]
        if device is not None:
            attempts.append((None, max(self.block, 1024)))      # system default as last resort
        last = None
        for dev, blk in attempts:
            try:
                self._open(sd, dev, blk, master_chans, phones_chans)
                return
            except Exception as e:  # noqa: BLE001
                last = e
                log.warning("audio open failed on %s (block %d): %s", dev, blk, e)
        raise RuntimeError(f"no usable audio output: {last}")

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
                                 device=device, latency="low", callback=self._callback, extra_settings=extra)
        stream.start()
        self._stream = stream
        self.block = block
        self.master_chans = master_chans
        self.phones_chans = phones_chans
        self.out_channels = chans
        log.info("audio started on %s via %s (%d ch, block %d)", info["name"], api, chans, block)

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None

    @property
    def running(self) -> bool:
        return self._stream is not None

    @property
    def latency_ms(self) -> float:
        if self._stream is None:
            return 0.0
        try:
            return float(self._stream.latency) * 1000
        except Exception:  # noqa: BLE001
            return 0.0


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
