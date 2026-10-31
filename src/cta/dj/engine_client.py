"""The GUI's handle on the audio engine running in its own process.

EngineClient has the engine's public API (set, seek, cue_press, load_track,
start...) and the same shaped state (decks[...].playing, .volume, .loop...,
master_deck, crossfader, automix.queue...), so the rest of the program
drives it exactly like the in-process Engine. Reads come from state
snapshots the server sends ~30 times a second; writes are messages. Track
audio goes over in shared memory; the master output comes back through a
shared ring that feeds the phone stream, the recorder and the scopes.

inproc=True runs the same server inside this process (tests, or when a
child process cannot be started): identical behaviour, no isolation.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
from dataclasses import asdict, fields
from types import SimpleNamespace

import numpy as np

from .audio_server import DECK_SCALARS, AudioRing, Server, publish_track, serve
from .beatgrid import BeatGrid
from .engine import SR, Engine, TrackInfo

log = logging.getLogger(__name__)

_PHONES_SEPARATE = SimpleNamespace(separate=True)       # stands in for engine._pstream


class DeckView:
    """A deck as the GUI sees it: the server's numbers plus what the GUI
    itself loaded (grid, waveform data, track info)."""

    def __init__(self, name: str, scope: str = "deck"):
        self.name = name
        self._scope = scope
        for k in DECK_SCALARS:
            setattr(self, k, 0.0)
        self.playing = self.cue_preview = self.sync = self.flux = self.cue_phones = self.scratching = False
        self.end_reached = self.nudged = False
        self.keylock = True
        self.volume = self.gain = self.trim = 1.0
        self.tempo_range = 0.08
        self.rate_override = None
        self.loop = None
        self.hotcues = [None] * 8
        self.eq = np.ones(3)
        self.fx: set = set()
        self.loop_size = 4.0
        self.mix_in = self.mix_out = self.mix_out_end = None
        self.phrase_origin = 0
        self.length = 0
        self.loaded = False
        self.base_rate = self.rate = 1.0
        self.bpm = 0.0
        self.effective_key = -1
        self.info = TrackInfo()
        self.grid: BeatGrid | None = None
        self.peaks = None
        self.bands = None
        self.audio = None
        self.scope = None
        self.track: str | None = None
        self._info_key = None

    @property
    def time(self) -> float:
        return self.play_pos / SR

    def _apply(self, s: dict, registry: dict, held=()):
        for k in DECK_SCALARS:
            if (self._scope, self.name, k) not in held:
                setattr(self, k, s[k])
        self.loop = tuple(s["loop"]) if s["loop"] else None
        self.hotcues = list(s["hotcues"])
        if (self._scope, self.name, "eq") not in held:
            self.eq = np.asarray(s["eq"], np.float64)
        self.fx = set(s["fx"])
        self.length, self.loaded = s["length"], s["loaded"]
        self.base_rate, self.rate, self.bpm = s["base_rate"], s["rate"], s["bpm"]
        self.effective_key = s["effective_key"]
        self.scope = s.get("scope")
        info = s["info"]
        key = (info.get("path"), info.get("title"), info.get("key"), info.get("bpm"))
        if key != self._info_key:
            self._info_key = key
            self.info = TrackInfo(**info)
        track = s.get("track")
        if track != self.track:
            self.track = track
            lt = registry.get(track) if track else None
            self.grid = getattr(lt, "grid", None)
            self.peaks = getattr(lt, "peaks", None)
            self.bands = getattr(lt, "bands", None)
            self.audio = getattr(lt, "audio", None)
        elif track and self.grid is None:
            lt = registry.get(track)
            if lt is not None:
                self.grid, self.peaks, self.bands, self.audio = lt.grid, lt.peaks, getattr(lt, "bands", None), lt.audio
        if not self.loaded:
            self.grid = self.peaks = self.bands = self.audio = None


class FXView:
    _ATTRS = ("kind", "on", "wet", "amount", "div_index")

    def __init__(self, client: EngineClient, index: int):
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_i", index)
        object.__setattr__(self, "_v", {"kind": "Delay", "on": False, "wet": 0.5, "amount": 0.5, "div_index": 2})

    def __getattr__(self, name):
        if name in self._ATTRS:
            return self._v[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self._ATTRS:
            self._client._send("fx", self._i, name, value, hold=(("fx", self._i, name),))
            self._v[name] = value
        else:
            object.__setattr__(self, name, value)

    def _apply(self, s: dict, held=()):
        self._v.update({k: v for k, v in s.items() if ("fx", self._i, k) not in held})


class _SettingsView:
    """automix.s - a dataclass mirror whose writes go to the server."""

    def __init__(self, client: EngineClient, target: str = "ams"):
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_target", target)
        from .automix import AutoMixSettings

        object.__setattr__(self, "_v", asdict(AutoMixSettings()))

    def __getattr__(self, name):
        try:
            return self._v[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self._client._send("set", self._target, name, value, hold=((self._target, name),))
        self._v[name] = value

    def _apply(self, s: dict, held=()):
        self._v.update({k: v for k, v in s.items() if (self._target, k) not in held})


class AutoMixView:
    """Auto-mix as the GUI sees it. Commands are forwarded; everything else
    comes from the snapshots."""

    def __init__(self, client: EngineClient):
        self._client = client
        self.active = False
        self.status = "Auto-mix off"
        self.queue: list = []
        self.history: list = []
        self.used_items: list = []
        self.aired: list = []
        self.meta: set = set()
        self._loading: set = set()
        self.tr = None
        self.pending_skip = None
        self.resume_after_manual = False
        self._resume_after_takeover = True
        self._rehearsal_fill = True
        self.s = _SettingsView(client)
        self.learner = None              # the GUI's MixLearner (its numbers are pushed to the server)
        self.rehearsals = None           # the GUI's RehearsalStore (pushed likewise)
        self._lock = threading.Lock()    # kept for old callers; the queue is read-only here
        self.decks = ("A", "B")

    # ---- state ---------------------------------------------------------------------
    def _apply(self, s: dict, held=()):
        for k in ("active", "status", "queue", "history", "used_items", "aired", "pending_skip",
                  "resume_after_manual"):
            if ("am", k) not in held:
                setattr(self, k, s[k])
        self.meta = set(s["meta"])
        self._loading = set(s["loading"])
        if ("am", "resume_after_takeover") not in held:
            self._resume_after_takeover = s["resume_after_takeover"]
        if ("am", "rehearsal_fill") not in held:
            self._rehearsal_fill = s["rehearsal_fill"]
        self.s._apply(s["s"], held)
        tr = s["tr"]
        self.tr = SimpleNamespace(**tr) if tr else None

    @property
    def resume_after_takeover(self) -> bool:
        return self._resume_after_takeover

    @resume_after_takeover.setter
    def resume_after_takeover(self, v: bool):
        self._client._send("set", "am", "resume_after_takeover", bool(v), hold=(("am", "resume_after_takeover"),))
        self._resume_after_takeover = bool(v)

    @property
    def rehearsal_fill(self) -> bool:
        return self._rehearsal_fill

    @rehearsal_fill.setter
    def rehearsal_fill(self, v: bool):
        self._client._send("set", "am", "rehearsal_fill", bool(v), hold=(("am", "rehearsal_fill"),))
        self._rehearsal_fill = bool(v)

    @staticmethod
    def _analyse_deck(d):
        from .automix import analyse_deck

        return analyse_deck(d)

    # ---- commands ---------------------------------------------------------------------
    def _call(self, name, *args, **kw):
        self._client.call("am", name, *args, **kw)

    def set_queue(self, items):
        self.queue = list(items)
        self._call("set_queue", list(items))

    def queue_clear(self):
        self.queue = []
        self._call("queue_clear")

    def queue_remove(self, pos: int):
        if 0 <= pos < len(self.queue):
            self.queue.pop(pos)
        self._call("queue_remove", int(pos))

    def queue_insert(self, pos: int, item):
        self.queue.insert(max(0, min(pos, len(self.queue))), item)
        self._call("queue_insert", int(pos), item)

    def queue_append(self, item):
        self.queue.append(item)
        self._call("queue_append", item)

    def start(self, from_queue: bool = False):
        self.active = True
        self._call("start", from_queue)

    def stop(self, settle: bool = False):
        self.active = False
        self._call("stop", settle)

    def skip(self, mode: str = "mix"):
        self._call("skip", mode)

    def play_next(self, item, skip=None, requeue: bool = True):
        self._call("play_next", item, skip, requeue)

    def previous_item(self):
        """The track that was on air before the current one (mirrored here)."""
        m = self._client._master()
        cur = m.info.lib_index if m is not None else None
        for it in reversed(list(self.aired)):
            if it != cur:
                return it
        return None

    def forget_deck(self, deck: str):
        self.meta.discard(deck)
        self._call("forget_deck", deck)

    def replan(self):
        if self.tr is not None and not self.tr.started:
            self.tr = None
        self._call("replan")

    def transition_now(self, bars=None):
        self.active = True
        self._call("transition_now", bars)

    def manual(self, deck, action: str) -> bool:
        """Forwarded; the answer (did auto-mix step back?) is worked out here
        from the mirrored state, the same way the server does."""
        from .automix import LEVEL_ACTIONS, TAKEOVER_ACTIONS

        self._call("manual", deck, action)
        if not self.active or action not in TAKEOVER_ACTIONS and not action.startswith("hotcue"):
            return False
        tr = self.tr
        decks = self._client.decks
        if tr is not None and tr.started:
            took = deck is None or deck in (tr.in_deck, tr.out_deck)
        elif tr is not None and deck == tr.in_deck and action in LEVEL_ACTIONS:
            took = True
        elif action == "play" and deck in decks and not decks[deck].playing \
                and any(decks[d].playing for d in self.decks if d != deck):
            took = True
        else:
            took = False
        if took:
            self.active = False
            self.resume_after_manual = True
        return took

    def user_took_over(self):
        self.active = False
        self._call("user_took_over")

    def maybe_resume(self):
        pass                                 # the server does this on its own clock

    def reset(self):
        self.queue, self.history, self.used_items, self.aired = [], [], [], []
        self.active = False
        self._call("reset")

    def refresh_markers(self, deck: str):
        self._call("refresh_markers", deck)


class SessionView:
    """The Mix Ahead recorder (rehearsal.Session) in the preview engine."""

    def __init__(self, client: EngineClient):
        self._client = client
        self.mode = "idle"
        self.x = 0.0
        self.drop_beat = None
        self.replaying = None
        self.lanes: dict = {}
        self.result = None
        self._auto_record = True
        self._fill = True

    def _apply(self, s: dict, held=()):
        self.mode, self.x, self.drop_beat = s["mode"], s["x"], s["drop_beat"]
        self.replaying = True if s["replaying"] else None
        if ("pvs", "auto_record") not in held:
            self._auto_record = s["auto_record"]
        if ("pvs", "fill") not in held:
            self._fill = s["fill"]
        if s.get("lanes") is not None:
            self.lanes = s["lanes"]
        r = s.get("result")
        if r is not None and (self.result is None or self.result.created != r.get("created")):
            from .rehearsal import Rehearsal

            self.result = Rehearsal.from_dict(r)
        elif r is None and self.mode != "done":
            self.result = None

    @property
    def auto_record(self) -> bool:
        return self._auto_record

    @auto_record.setter
    def auto_record(self, v: bool):
        self._client._send("set", "pvs", "auto_record", bool(v), hold=(("pvs", "auto_record"),))
        self._auto_record = bool(v)

    @property
    def fill(self) -> bool:
        return self._fill

    @fill.setter
    def fill(self, v: bool):
        self._client._send("set", "pvs", "fill", bool(v), hold=(("pvs", "fill"),))
        self._fill = bool(v)

    def rehearse(self, out_beat: float, in_beat: float, lead_beats=None):
        self.result = None
        self.mode = "lead"
        self._client.call("pvs", "rehearse", float(out_beat), float(in_beat), lead_beats)

    def drop_now(self):
        self._client.call("pvs", "drop_now")

    def replay(self, r, lead_beats: float = 16.0):
        self.mode = "lead"
        self._client.call("srv", "session_replay", r.to_dict(), lead_beats)

    def stop(self):
        self._client.call("pvs", "stop")

    def close(self):
        pass


class PreviewClient:
    """The Mix Ahead preview engine (decks A/B of their own) in the server."""

    def __init__(self, client: EngineClient):
        self._client = client
        self.decks = {n: DeckView(n, "pvdeck") for n in "ABCD"}
        self._crossfader = 0.0
        self.master_deck = None
        self.session = SessionView(client)
        self.block = client.block

    def _apply(self, s: dict, held=()):
        for n, d in s["decks"].items():
            self.decks[n]._apply(d, self._client._registry, held)
        if ("pv", "crossfader") not in held:
            self._crossfader = s["crossfader"]
        self.master_deck = s["master_deck"]
        self.session._apply(s["session"], held)

    @property
    def crossfader(self) -> float:
        return self._crossfader

    @crossfader.setter
    def crossfader(self, v: float):
        self._client._send("set", "pv", "crossfader", float(v), hold=(("pv", "crossfader"),))
        self._crossfader = float(v)

    def load_track(self, deck: str, lt, only_if_stopped: bool = True):
        self._client._load("pv", deck, lt, only_if_stopped)

    _FWD = ("set", "seek", "set_playing", "toggle_play", "set_markers", "set_cue", "cue_press", "cue_release",
            "loop_beats", "hotcue", "hotcue_delete", "beatjump", "set_master", "unload", "scratch", "jog",
            "jog_release", "loop_in", "loop_out", "loop_resize", "key_sync", "phase_snap", "set_paused")

    def __getattr__(self, name):
        if name in self._FWD:
            return lambda *a, **k: self._client.call("pv", name, *a, **k)
        if name in ("master_volume", "cue_mix", "phones_volume", "quantize", "paused"):
            return 1.0 if name != "paused" else False
        if name == "fx_units":
            return []
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in ("master_volume", "cue_mix", "phones_volume", "quantize"):
            return                          # a controller aimed at the preview: no such knobs there
        object.__setattr__(self, name, value)


class EngineClient:
    DECKS = "ABCD"

    def __init__(self, block: int = 512, inproc: bool = False, automix_settings: dict | None = None,
                 load_fn=None):
        self.block = block
        self.load_fn = load_fn               # item -> LoadedTrack (auto-mix's loader, runs in the GUI)
        self.decks = {n: DeckView(n) for n in self.DECKS}
        self.fx_units = [FXView(self, 0), FXView(self, 1)]
        self.automix = AutoMixView(self)
        self.taps: list = []
        self.hooks: list = []
        self.recorder = None
        self.preview: PreviewClient | None = None
        self._registry: dict[str, object] = {}        # shm name -> LoadedTrack kept alive here
        self._handles: dict[str, list] = {}
        self._rpc: dict[int, tuple[threading.Event, list]] = {}
        self._req = 0
        self._lock = threading.Lock()
        self._snap: dict = {}
        # every command is numbered; a value written here is held against snapshots
        # until one that already includes that command (ack) arrives, and a snapshot
        # older than the last applied one is dropped - so nothing ever snaps back
        self._seq = 0
        self._holds: dict[tuple, int] = {}
        self._snap_seq = 0
        self._apply_lock = threading.Lock()
        self.scope = None
        self.scope_prev = None
        self._scope_buf = np.zeros((2, 2048), np.float32)
        self.ring: AudioRing | None = None
        self.server_errors: list[str] = []
        self.deck_errors: list[str] = []
        self.click_times: list[float] = []
        self.extra_outputs: list = []
        self.inproc = inproc
        self.cfg = {"block": block, "automix": automix_settings or {}}
        self._local = {"crossfader": 0.0, "master_volume": 1.0, "phones_volume": 1.0, "cue_mix": 0.0,
                       "quantize": True, "phase_lock": True, "reset_on_load": True, "out_latency": 0.025,
                       "clock": "auto", "sync_outputs": None, "clock_adjust_s": 0.0, "master_delay_s": None,
                       "phones_delay_s": None, "click_target": None, "preview_route": "phones",
                       "preview_level": 1.0}
        self.alive = False
        self._spawn()

    # ---- process --------------------------------------------------------------------------
    def _spawn(self):
        parent, child = mp.Pipe()
        self._conn = parent
        self.server = None
        self._proc = None
        if self.inproc:
            self.server = Server(child, self.cfg)
            self._thread = threading.Thread(target=self.server.run, daemon=True, name="audio-server")
            self._thread.start()
        else:
            ctx = mp.get_context("spawn")
            self._proc = ctx.Process(target=serve, args=(child, self.cfg), daemon=True, name="cta-audio")
            self._proc.start()
            child.close()
        self.alive = True
        self._hello = threading.Event()
        self._reader = threading.Thread(target=self._read, daemon=True, name="audio-client")
        self._reader.start()
        if not self._hello.wait(30):
            raise RuntimeError("the audio process did not start")

    @property
    def process_alive(self) -> bool:
        if self.inproc:
            return self._thread.is_alive()
        return self._proc is not None and self._proc.is_alive()

    def respawn(self):
        """The audio process died: start a new one and put every deck's track
        back (the GUI still holds them). The caller restarts the sound card."""
        old_tracks = {n: d.track for n, d in self.decks.items()}
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass
        if self.ring is not None:
            self.ring.close()
            self.ring = None
        self._spawn()
        for name, v in self._local.items():
            self._send("set", "eng", name, v)
        for n, track in old_tracks.items():
            lt = self._registry.get(track) if track else None
            if lt is not None:
                self._load("eng", n, lt, False)

    def close(self):
        if not self.alive:
            return
        try:
            self._send("quit")
        except Exception:  # noqa: BLE001
            pass
        self.alive = False
        if self._proc is not None:
            self._proc.join(5)
            if self._proc.is_alive():
                self._proc.kill()
        elif self.inproc:
            self._thread.join(5)
        for handles in self._handles.values():
            self._unlink(handles)
        self._handles.clear()
        if self.ring is not None:
            self.ring.close()

    # ---- messages -------------------------------------------------------------------------
    def _send(self, *msg, hold=()):
        if not self.alive:
            return                              # closed: nothing to talk to any more
        with self._lock:
            self._seq += 1
            seq = self._seq
            if hold:
                with self._apply_lock:
                    for key in hold:
                        self._holds[key] = seq
            if self.server is not None:
                # in-process: dispatch right here and refresh the views, so what was
                # just asked for is visible at once (tests rely on it)
                try:
                    self.server._dispatch(msg, seq)
                except Exception:  # noqa: BLE001
                    log.exception("audio command failed: %r", msg[:3])
                    return
            else:
                try:
                    self._conn.send(("m", seq, msg))
                except (OSError, ValueError, EOFError, BrokenPipeError) as e:
                    log.warning("audio process unreachable: %s", e)
                    self.alive = False
                return
        if msg[0] != "quit":
            try:
                self._apply_snapshot(self.server._snapshot())
            except Exception:  # noqa: BLE001
                pass

    def call(self, target: str, name: str, *args, **kwargs):
        self._send("call", target, name, args, kwargs)

    def rpc(self, target: str, name: str, *args, timeout: float = 20.0, **kwargs):
        if not self.alive:
            raise RuntimeError("the audio engine is closed")
        if self.server is not None:
            with self._lock, self.server._state_lock:
                value = getattr(self.server._target(target), name)(*args, **kwargs)
            self._apply_snapshot(self.server._snapshot())
            return value
        with self._lock:
            self._req += 1
            req_id = self._req
            ev = (threading.Event(), [])
            self._rpc[req_id] = ev
            self._conn.send(("rpc", req_id, target, name, args, kwargs))
        if not ev[0].wait(timeout):
            self._rpc.pop(req_id, None)
            raise TimeoutError(f"audio process did not answer {name}")
        ok, value = ev[1][0]
        if not ok:
            raise RuntimeError(value)
        return value

    def _read(self):
        conn = self._conn
        while self.alive:
            try:
                if not conn.poll(0.5):
                    continue
                msg = conn.recv()
            except (EOFError, OSError):
                break
            try:
                self._handle(msg)
            except Exception:  # noqa: BLE001
                log.exception("audio client: bad message %r", msg[:2] if isinstance(msg, tuple) else msg)

    def _handle(self, msg):
        kind = msg[0]
        if kind == "snap":
            self._apply_snapshot(msg[1])
        elif kind == "hello":
            self.ring = AudioRing(create=False, names=msg[1]["ring"])
            self._hello.set()
        elif kind == "need_load":
            _, req_id, item = msg
            threading.Thread(target=self._serve_load, args=(req_id, item), daemon=True,
                             name=f"load-{item}").start()
        elif kind == "rpc":
            _, req_id, ok, value = msg
            ev = self._rpc.pop(req_id, None)
            if ev is not None:
                ev[1].append((ok, value))
                ev[0].set()
        elif kind == "event":
            _, what, data = msg
            if what == "released":
                for name in data:
                    self._registry.pop(name, None)
                    handles = self._handles.pop(name, None)
                    if handles:
                        self._unlink(handles)
            elif what == "took_over":
                lr = self.automix.learner
                if lr is not None:
                    lr.took_over()
            elif what == "rehearsal_done":
                from .rehearsal import Rehearsal

                if self.preview is not None:
                    self.preview.session.result = Rehearsal.from_dict(data)
                    self.preview.session.mode = "done"
            elif what == "error":
                self.server_errors.append(data)
                del self.server_errors[:-20]

    def _apply_snapshot(self, snap: dict):
        if not snap:
            return
        with self._apply_lock:
            seq = snap.get("seq", 0)
            if seq < self._snap_seq:
                return                          # older than what is shown already (two threads deliver)
            self._snap_seq = seq
            ack = snap.get("ack", 0)
            for key in [k for k, n in self._holds.items() if n <= ack]:
                del self._holds[key]
            held = set(self._holds)
            old = self._snap
            for key in held:
                if key[0] == "eng" and key[1] in old:
                    snap[key[1]] = old[key[1]]
            self._snap = snap
            for n, d in snap["decks"].items():
                self.decks[n]._apply(d, self._registry, held)
            for i, s in enumerate(snap["fx"]):
                if i < len(self.fx_units):
                    self.fx_units[i]._apply(s, held)
            self.automix._apply(snap["automix"], held)
            self.deck_errors = snap["deck_errors"]
            self.click_times = snap["click_times"]
            self.extra_outputs = [SimpleNamespace(**o) for o in snap["extra_outputs"]]
            for k in ("crossfader", "master_volume", "phones_volume", "cue_mix", "quantize", "phase_lock",
                      "reset_on_load", "out_latency", "clock", "clock_adjust_s", "master_delay_s",
                      "phones_delay_s", "click_target", "preview_route", "preview_level", "sync_outputs"):
                if ("eng", k) not in held:
                    v = snap[k]
                    self._local[k] = set(v) if k == "sync_outputs" and v is not None else v
            pv = snap.get("preview")
            if pv is not None and self.preview is not None:
                self.preview._apply(pv, held)
        self._drain_ring()

    def _drain_ring(self):
        if self.ring is None:
            return
        x = self.ring.read_new()
        if x is None:
            return
        block = np.ascontiguousarray(x.T)             # (2, n) like the engine's taps get
        for tap in list(self.taps):
            try:
                tap.write(block)
            except Exception:  # noqa: BLE001
                log.exception("tap failed")
        rec = self.recorder
        if rec is not None:
            try:
                rec.write(block)
            except Exception:  # noqa: BLE001
                log.exception("recorder failed")
        n = block.shape[1]
        if n >= self._scope_buf.shape[1]:
            self._scope_buf[:] = block[:, -self._scope_buf.shape[1]:]
        else:
            self._scope_buf = np.roll(self._scope_buf, -n, axis=1)
            self._scope_buf[:, -n:] = block
        self.scope = self._scope_buf[:, -1024:]
        self.scope_prev = self._scope_buf[:, :-1024]

    def _serve_load(self, req_id: int, item):
        try:
            if self.load_fn is None:
                raise RuntimeError("no loader")
            lt = self.load_fn(item)
            payload, handles = publish_track(lt)
            name = payload["audio"]["name"]
            self._registry[name] = lt
            self._handles[name] = handles
            self._send("loaded", req_id, payload, None)
        except Exception as e:  # noqa: BLE001
            log.exception("auto-mix load failed")
            self._send("loaded", req_id, None, f"{type(e).__name__}: {e}")

    @staticmethod
    def _unlink(handles):
        for shm in handles:
            try:
                shm.close()
                shm.unlink()
            except Exception:  # noqa: BLE001
                pass

    # ---- tracks ------------------------------------------------------------------------------
    def _load(self, target: str, deck: str, lt, only_if_stopped: bool):
        payload, handles = publish_track(lt)
        name = payload["audio"]["name"]
        self._registry[name] = lt
        self._handles[name] = handles
        self._send("load", target, deck, payload, only_if_stopped)

    def load_track(self, deck: str, lt, only_if_stopped: bool = True):
        self._load("eng", deck, lt, only_if_stopped)

    def load(self, deck: str, audio, grid, info, peaks=None, **kw):
        """Engine.load's shape (tests): wrap the pieces as a LoadedTrack."""
        lt = SimpleNamespace(audio=audio, grid=grid, info=info, peaks=peaks, bands=kw.get("bands"),
                             cues=None, mix_in=kw.get("mix_in"), mix_out=kw.get("mix_out"),
                             mix_out_end=kw.get("mix_out_end"), first_beat_t=kw.get("cue") or 0.0,
                             phrase_origin=0)
        self._load("eng", deck, lt, False)

    # ---- the engine API ------------------------------------------------------------------------
    def __getattr__(self, name):
        if name in ("set_playing", "toggle_play", "cue_press", "cue_release", "seek", "set_cue",
                    "set_markers", "loop_beats", "hotcue", "hotcue_delete", "set_hotcues", "loop_in", "loop_out",
                    "loop_resize", "beatjump", "key_sync", "set_grid", "grid_downbeat_here", "phase_snap",
                    "set_master", "jog", "scratch", "jog_release", "set_paused", "unload", "update_alignment",
                    "remove_output", "set_output_delay", "close_outputs", "stop"):
            return lambda *a, **k: self.call("eng", name, *a, **k)
        if name in ("start_virtual",):
            return lambda *a, **k: self.call("eng", name, *a, **k)
        raise AttributeError(name)

    def set(self, deck: str, **kw):
        """Set plain deck attributes (volume, eq, tempo, keylock, sync, ...)."""
        self._send("call", "eng", "set", (deck,), kw, hold=tuple(("deck", deck, k) for k in kw))

    def start(self, device_label=None, block=None, master_chans=(0, 1), phones_chans=None, phones_label=None):
        """Open the sound card(s) in the audio process; raises when it cannot."""
        if block:
            self.block = int(block)
        out = self.rpc("srv", "start", device_label, block, tuple(master_chans),
                       tuple(phones_chans) if phones_chans else None, phones_label, timeout=30)
        return out

    def add_output(self, label: str, delay_s=None, chans=(0, 1)):
        self.call("srv", "add_output", label, delay_s, tuple(chans))

    def calibrate(self, which):
        self.call("srv", "calibrate", which)

    def set_want_deck_scope(self, on: bool):
        self.call("srv", "set_want_deck_scope", bool(on))

    def push_learned(self, prefs: dict):
        self.call("srv", "set_learned", prefs)

    def push_rehearsals(self, items: list[dict]):
        self.call("srv", "set_rehearsals", items)

    def create_preview(self) -> PreviewClient:
        if self.preview is None:
            self.preview = PreviewClient(self)
        self.rpc("srv", "create_preview")
        return self.preview

    def close_preview(self):
        self.call("srv", "close_preview")
        self.preview = None

    def render(self, n: int):
        """In-process only (tests): render a block here and refresh the views at once."""
        if self.server is None:
            raise RuntimeError("render() is for the in-process engine")
        from .engine import _VirtualOutput

        eng = self.server.engine
        if isinstance(eng._stream, _VirtualOutput):
            eng._stream.stop()                    # the test is the clock now: one renderer per stretcher
            eng._stream = None
        out = eng.render(n)
        self._apply_snapshot(self.server._snapshot())
        return out

    def ping(self) -> bool:
        try:
            return self.rpc("srv", "ping", timeout=5) == "pong"
        except Exception:  # noqa: BLE001
            return False

    # ---- state ------------------------------------------------------------------------------
    def _g(self, key, default=None):
        return self._snap.get(key, default)

    @property
    def master_deck(self):
        return self._g("master_deck")

    @property
    def paused(self) -> bool:
        return bool(self._g("paused", False))

    @property
    def frames_rendered(self) -> int:
        return int(self._g("frames_rendered", 0))

    @property
    def underruns(self) -> int:
        return int(self._g("underruns", 0))

    @property
    def peak_level(self) -> float:
        return float(self._g("peak_level", 0.0))

    @property
    def _lim_gain(self) -> float:
        return float(self._g("_lim_gain", 1.0))

    @property
    def out_channels(self) -> int:
        return int(self._g("out_channels", 2))

    @property
    def master_chans(self):
        return self._g("master_chans", (0, 1))

    @property
    def phones_chans(self):
        return self._g("phones_chans")

    @property
    def _pstream(self):
        return _PHONES_SEPARATE if self._g("phones_separate") else None

    @property
    def output_delay_s(self) -> dict:
        return self._g("output_delay_s", {})

    @property
    def display_latency(self) -> float:
        return float(self._g("display_latency", 0.0))

    @property
    def cpu_load(self) -> float:
        return float(self._g("cpu_load", 0.0))

    @property
    def slow_blocks(self) -> int:
        return int(self._g("slow_blocks", 0))

    @property
    def callback_errors(self) -> int:
        return int(self._g("callback_errors", 0))

    @property
    def running(self) -> bool:
        return bool(self._g("running", False))

    @property
    def virtual(self) -> bool:
        return bool(self._g("virtual", False))

    @property
    def latency_ms(self) -> float:
        return float(self._g("reported_latency", 0.0)) * 1000.0

    def reported_latency(self, phones: bool = False) -> float:
        return float(self._g("reported_latency", 0.0))

    def output_delays(self):
        return tuple(self._g("output_delays", (0.0, 0.0)))

    def output_latencies(self) -> dict:
        return dict(self._g("output_latencies", {}))

    def _master(self, exclude: str | None = None):
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

    sync_ratio = staticmethod(Engine.sync_ratio)

    def phase_error(self, deck: str):
        return self._g("phase_err", {}).get(deck)

    def _xf_gain(self, side: int) -> float:
        if side == 0:
            return 1.0
        x = float(np.clip((self.crossfader * side + 1.0) / 2.0, 0.0, 1.0))
        return 1.0 if x >= 0.5 else float(np.sin(x * np.pi))


def _prop(name):
    def get(self):
        return self._local[name]

    def set_(self, v):
        self._send("set", "eng", name, v if not isinstance(v, set) else set(v), hold=(("eng", name),))
        self._local[name] = v
    return property(get, set_)


for _name in ("crossfader", "master_volume", "phones_volume", "cue_mix", "quantize", "phase_lock", "reset_on_load",
              "out_latency", "clock", "sync_outputs", "clock_adjust_s", "master_delay_s", "phones_delay_s",
              "click_target", "preview_route", "preview_level"):
    setattr(EngineClient, _name, _prop(_name))

__all__ = ["EngineClient", "DeckView", "fields"]
