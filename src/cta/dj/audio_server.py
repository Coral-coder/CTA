"""The audio engine in its own process.

The GUI (Python + Qt) can hold the interpreter for tens of milliseconds
while it paints; an audio callback in the same process then misses its
deadline and the room hears a glitch. So the engine - decks, mixing,
auto-mix, the sound card, extra speakers, the Mix Ahead preview - runs
here, in a child process with nothing else to do, and the GUI talks to it
over a pipe (commands one way, state snapshots the other).

Track audio never travels over the pipe: the GUI decodes a track into
shared memory and sends the name; the master output goes back the same
way (a shared ring the GUI reads for the phone stream, the recorder and
the oscilloscope).

Messages GUI -> server (tuples):
  ("call", target, method, args, kwargs)   target: eng | am | pv | pvs | srv
  ("set", target, attribute, value)
  ("fx", unit, attribute, value)
  ("load", target, deck, payload, only_if_stopped)
  ("loaded", req_id, payload | None, error | None)   answer to need_load
  ("rpc", req_id, target, method, args, kwargs)      answered with ("rpc", req_id, ok, value)
  ("quit",)
Server -> GUI:
  ("hello", info)   ("snap", dict)   ("need_load", req_id, item)
  ("rpc", req_id, ok, value)   ("event", kind, data)
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import asdict
from multiprocessing import shared_memory
from types import SimpleNamespace

import numpy as np

from .beatgrid import BeatGrid
from .engine import SR, Engine, TrackInfo

log = logging.getLogger(__name__)

RING_SECONDS = 8.0
SNAP_HZ = 30.0
DECK_SCALARS = ("play_pos", "playing", "cue_point", "cue_preview", "volume", "gain", "trim", "filter", "sync",
                "keylock", "key_shift", "tempo", "tempo_range", "rate_override", "nudge", "level", "flux",
                "cue_phones", "scratching", "end_reached", "start_pos", "xf_side", "loop_size", "phase_offset",
                "mix_in", "mix_out", "mix_out_end", "phrase_origin", "phase_corr")
ENGINE_SCALARS = ("crossfader", "master_volume", "phones_volume", "cue_mix", "master_deck", "block",
                  "frames_rendered", "underruns", "peak_level", "_lim_gain", "out_channels", "master_chans",
                  "phones_chans", "clock", "clock_adjust_s", "output_delay_s", "display_latency", "cpu_load",
                  "slow_blocks", "paused", "quantize", "callback_errors", "phase_lock", "reset_on_load",
                  "preview_route", "preview_level", "master_delay_s", "phones_delay_s", "out_latency",
                  "click_target")


# --------------------------------------------------------------------------
# track audio in shared memory
# --------------------------------------------------------------------------

def _put(arr: np.ndarray) -> tuple[shared_memory.SharedMemory, dict]:
    arr = np.ascontiguousarray(arr)
    shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
    view = np.ndarray(arr.shape, arr.dtype, buffer=shm.buf)
    view[...] = arr
    return shm, {"name": shm.name, "shape": arr.shape, "dtype": str(arr.dtype)}


def publish_track(lt) -> tuple[dict, list[shared_memory.SharedMemory]]:
    """GUI side: a LoadedTrack -> (payload for the server, shm handles to keep
    open until the server says it released them)."""
    handles = []
    shm, audio = _put(np.asarray(lt.audio, np.float32))
    handles.append(shm)
    bands = None
    if getattr(lt, "bands", None) is not None:
        shm_b, bands = _put(np.asarray(lt.bands, np.float32))
        handles.append(shm_b)
    g = lt.grid
    payload = {
        "audio": audio, "bands": bands,
        "grid": None if g is None else {"bpm": g.bpm, "first_beat": g.first_beat, "duration": g.duration},
        "info": asdict(lt.info), "peaks": None if lt.peaks is None else np.asarray(lt.peaks, np.float32),
        "cues": list(getattr(lt, "cues", None) or []) or None,
        "mix_in": getattr(lt, "mix_in", None), "mix_out": getattr(lt, "mix_out", None),
        "mix_out_end": getattr(lt, "mix_out_end", None),
        "first_beat_t": float(getattr(lt, "first_beat_t", 0.0) or 0.0),
        "phrase_origin": int(getattr(lt, "phrase_origin", 0) or 0),
    }
    return payload, handles


class _Attached:
    """A track's shared memory as seen by the server."""

    def __init__(self, payload: dict):
        self.names: list[str] = []
        self.shms: list[shared_memory.SharedMemory] = []
        self.audio = self._view(payload["audio"])
        self.bands = self._view(payload["bands"]) if payload.get("bands") else None

    def _view(self, spec: dict) -> np.ndarray:
        shm = shared_memory.SharedMemory(name=spec["name"])
        self.shms.append(shm)
        self.names.append(spec["name"])
        return np.ndarray(tuple(spec["shape"]), np.dtype(spec["dtype"]), buffer=shm.buf)

    def close(self):
        for shm in self.shms:
            try:
                shm.close()
            except Exception:  # noqa: BLE001
                pass


def attach_track(payload: dict):
    """Server side: payload -> (LoadedTrack-like object, its shared memory)."""
    att = _Attached(payload)
    g = payload["grid"]
    info = TrackInfo(**payload["info"])
    lt = SimpleNamespace(audio=att.audio, bands=att.bands, info=info,
                         grid=BeatGrid(**g) if g else None, peaks=payload["peaks"], cues=payload["cues"],
                         mix_in=payload["mix_in"], mix_out=payload["mix_out"], mix_out_end=payload["mix_out_end"],
                         first_beat_t=payload["first_beat_t"], phrase_origin=payload["phrase_origin"],
                         shm_name=payload["audio"]["name"])
    return lt, att


# --------------------------------------------------------------------------
# the master output back to the GUI
# --------------------------------------------------------------------------

class AudioRing:
    """A shared stereo ring the engine writes every block (as a tap) and the
    GUI reads at its own pace: header[0] = frames written so far."""

    def __init__(self, create: bool, names: dict | None = None, frames: int = int(RING_SECONDS * SR)):
        if create:
            self.buf_shm = shared_memory.SharedMemory(create=True, size=frames * 2 * 4)
            self.hdr_shm = shared_memory.SharedMemory(create=True, size=16)
            self.names = {"buf": self.buf_shm.name, "hdr": self.hdr_shm.name, "frames": frames}
        else:
            self.buf_shm = shared_memory.SharedMemory(name=names["buf"])
            self.hdr_shm = shared_memory.SharedMemory(name=names["hdr"])
            frames = names["frames"]
            self.names = names
        self.frames = frames
        self.buf = np.ndarray((frames, 2), np.float32, buffer=self.buf_shm.buf)
        self.hdr = np.ndarray((2,), np.int64, buffer=self.hdr_shm.buf)
        if create:
            self.hdr[:] = 0
        self.rpos = int(self.hdr[0])

    def write(self, block: np.ndarray):
        """block: (2, n) - the engine's tap interface."""
        n = block.shape[1]
        w = int(self.hdr[0]) % self.frames
        x = block.T
        end = w + n
        if end <= self.frames:
            self.buf[w:end] = x
        else:
            k = self.frames - w
            self.buf[w:] = x[:k]
            self.buf[:n - k] = x[k:]
        self.hdr[0] += n

    def read_new(self) -> np.ndarray | None:
        """Everything written since the last read, (n, 2) - or None."""
        wpos = int(self.hdr[0])
        if wpos == self.rpos:
            return None
        if wpos - self.rpos > self.frames:
            self.rpos = wpos - self.frames                 # the reader fell too far behind
        n = wpos - self.rpos
        r = self.rpos % self.frames
        out = np.empty((n, 2), np.float32)
        end = r + n
        if end <= self.frames:
            out[:] = self.buf[r:end]
        else:
            k = self.frames - r
            out[:k] = self.buf[r:]
            out[k:] = self.buf[:n - k]
        self.rpos = wpos
        return out

    def close(self, unlink: bool = False):
        for shm in (self.buf_shm, self.hdr_shm):
            try:
                shm.close()
                if unlink:
                    shm.unlink()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------
# the server
# --------------------------------------------------------------------------

class _LearnedShim:
    """What auto-mix learned lives in the GUI's database; the server only
    needs the numbers, pushed over when they change."""

    def __init__(self, server):
        self.server = server
        self.prefs: dict[str, dict] = {}

    def preferences(self, group: str):
        return self.prefs.get(group) or self.prefs.get("*")

    def took_over(self):
        self.server.event("took_over", None)


class _RehearsalShim:
    """Mix Ahead recordings, pushed over by the GUI (only enabled, usable ones)."""

    def __init__(self):
        self.items: dict[str, object] = {}

    def get(self, out_path: str, in_path: str):
        r = self.items.get(f"{out_path}\n{in_path}")
        return r if r is not None and r.enabled else None

    find = get


class Server:
    def __init__(self, conn, cfg: dict | None = None):
        from .automix import AutoMix, AutoMixSettings

        cfg = cfg or {}
        self.conn = conn
        self.engine = Engine(block=int(cfg.get("block", 512)))
        self.ring = AudioRing(create=True)
        self.engine.taps.append(self.ring)
        s = cfg.get("automix") or {}
        self.automix = AutoMix(self.engine, self._load_fn, AutoMixSettings(**s))
        self.learned = _LearnedShim(self)
        self.automix.learner = self.learned
        self.rehearsals = _RehearsalShim()
        self.automix.rehearsals = self.rehearsals
        self.preview: Engine | None = None
        self.session = None
        self._session_sent = None
        self.tracks: dict[str, _Attached] = {}          # shm name -> attached (installed or pending)
        self.deck_track: dict[tuple[str, str], str] = {}  # (target, deck) -> shm name
        self._pending_loads: dict[int, dict] = {}
        self._load_events: dict[int, tuple[threading.Event, list]] = {}
        self._req = 0
        self._lock = threading.Lock()
        self.want_deck_scope = False
        self.alive = True
        self._snap_thread = threading.Thread(target=self._snapshots, daemon=True, name="audio-snap")

    # ---- plumbing ---------------------------------------------------------------------
    def send(self, *msg):
        with self._lock:
            try:
                self.conn.send(msg)
            except (OSError, ValueError, EOFError):
                self.alive = False

    def event(self, kind: str, data):
        self.send("event", kind, data)

    def _target(self, name: str):
        if name == "eng":
            return self.engine
        if name == "am":
            return self.automix
        if name == "pv":
            return self.preview
        if name == "pvs":
            return self.session
        if name == "ams":
            return self.automix.s
        if name == "srv":
            return self
        raise KeyError(name)

    def run(self):
        self.send("hello", {"ring": self.ring.names, "pid": __import__("os").getpid()})
        self._snap_thread.start()
        while self.alive:
            try:
                if not self.conn.poll(0.5):
                    continue
                msg = self.conn.recv()
            except (EOFError, OSError):
                break
            try:
                self._dispatch(msg)
            except Exception:  # noqa: BLE001
                log.exception("audio server: %r failed", msg[:3] if isinstance(msg, tuple) else msg)
                self.event("error", traceback.format_exc()[-800:])
        self.shutdown()

    def _dispatch(self, msg):
        kind = msg[0]
        if kind == "call":
            _, target, name, args, kwargs = msg
            obj = self._target(target)
            if obj is None:
                return
            getattr(obj, name)(*args, **kwargs)
        elif kind == "set":
            _, target, name, value = msg
            obj = self._target(target)
            if obj is not None:
                setattr(obj, name, value)
                if target == "eng" and name in ("clock", "sync_outputs", "clock_adjust_s", "master_delay_s",
                                                "phones_delay_s"):
                    self.engine.update_alignment()
        elif kind == "fx":
            _, unit, name, value = msg
            setattr(self.engine.fx_units[unit], name, value)
        elif kind == "load":
            _, target, deck, payload, only_if_stopped = msg
            self.load(target, deck, payload, only_if_stopped)
        elif kind == "loaded":
            _, req_id, payload, error = msg
            ev = self._load_events.get(req_id)
            if ev is not None:
                ev[1].append((payload, error))
                ev[0].set()
        elif kind == "rpc":
            _, req_id, target, name, args, kwargs = msg
            try:
                obj = self._target(target)
                value = getattr(obj, name)(*args, **kwargs)
                self.send("rpc", req_id, True, value)
            except Exception as e:  # noqa: BLE001
                log.exception("rpc %s failed", name)
                self.send("rpc", req_id, False, f"{type(e).__name__}: {e}")
        elif kind == "quit":
            self.alive = False

    # ---- tracks --------------------------------------------------------------------------
    GRACE_S = 30.0          # a track just attached is never freed before auto-mix could install it

    def _attach(self, payload: dict):
        lt, att = attach_track(payload)
        att.attached_at = time.monotonic()
        att.installed = False
        self.tracks[lt.shm_name] = att
        return lt

    def load(self, target: str, deck: str, payload: dict, only_if_stopped: bool):
        eng = self._target(target)
        if eng is None:
            return
        lt = self._attach(payload)
        name = lt.shm_name
        eng.load_track(deck, lt, only_if_stopped=only_if_stopped)

        def after(e):
            att = self.tracks.get(name)
            if att is not None and not any(d.audio is not None and self._owns(d.audio, att)
                                           for d in e.decks.values()):
                att.installed = True          # skipped (deck playing): free it on this pass
            self._release_unused()
        eng.cmd(after)

    @staticmethod
    def _owns(arr: np.ndarray, att: _Attached) -> bool:
        try:
            return arr.__array_interface__["data"][0] == att.audio.__array_interface__["data"][0]
        except Exception:  # noqa: BLE001
            return False

    def _release_unused(self):
        """Runs on the audio thread after a load: drop shared memory no deck
        holds any more (the GUI unlinks it on the "released" event)."""
        in_use = set()
        for target, eng in (("eng", self.engine), ("pv", self.preview)):
            if eng is None:
                continue
            for dn, d in eng.decks.items():
                if d.audio is None:
                    continue
                for name, att in self.tracks.items():
                    if self._owns(d.audio, att):
                        in_use.add(name)
                        att.installed = True
                        self.deck_track[(target, dn)] = name
        pending = {p.get("audio", {}).get("name") for p in self._pending_loads.values()}
        now = time.monotonic()
        gone = [n for n, att in self.tracks.items() if n not in in_use and n not in pending
                and (att.installed or now - att.attached_at > self.GRACE_S)]
        if gone:
            self._to_release = getattr(self, "_to_release", []) + gone
            for n in gone:
                self.tracks.pop(n, None)

    def _flush_releases(self):
        gone = getattr(self, "_to_release", [])
        if gone:
            self._to_release = []
            self.event("released", gone)

    def _load_fn(self, item):
        """Auto-mix needs a track: the GUI (library, decoder) loads it into
        shared memory and answers; this waits in auto-mix's loader thread."""
        with self._lock:
            self._req += 1
            req_id = self._req
            ev = (threading.Event(), [])
            self._load_events[req_id] = ev
        self.send("need_load", req_id, item)
        if not ev[0].wait(180):
            self._load_events.pop(req_id, None)
            raise TimeoutError("the track did not load in time")
        self._load_events.pop(req_id, None)
        payload, error = ev[1][0]
        if error or payload is None:
            raise RuntimeError(error or "load failed")
        return self._attach(payload)

    # ---- learned mixes / recordings pushed by the GUI ------------------------------------
    def set_learned(self, prefs: dict):
        self.learned.prefs = dict(prefs or {})

    def set_rehearsals(self, items: list[dict]):
        from .rehearsal import Rehearsal, RehearsalStore

        self.rehearsals.items = {RehearsalStore.key(r["out_path"], r["in_path"]): Rehearsal.from_dict(r)
                                 for r in items}

    # ---- preview (Mix Ahead) ---------------------------------------------------------------
    def create_preview(self):
        from .rehearsal import Session

        if self.preview is None:
            self.preview = Engine(block=self.engine.block)
            self.preview._driven = True
            self.session = Session(self.preview)
            self._session_sent = None
        self.engine.preview = self.preview
        return True

    def session_replay(self, r: dict, lead_beats: float = 16.0):
        from .rehearsal import Rehearsal

        if self.session is not None:
            self.session.replay(Rehearsal.from_dict(r), lead_beats)

    def close_preview(self):
        if self.session is not None:
            self.session.stop()
            self.session.close()
        self.engine.preview = None
        pv, self.preview, self.session = self.preview, None, None
        if pv is not None:
            for d in pv.decks.values():
                d.playing = False
                d.audio = None
            self.engine.cmd(lambda e: self._release_unused())

    # ---- device ----------------------------------------------------------------------------
    def start(self, device_label=None, block=None, master_chans=(0, 1), phones_chans=None, phones_label=None):
        """Open the sound card(s) by label (this process has its own PortAudio)."""
        from .engine import list_output_devices
        from .outputs import base_name, raise_priority

        raise_priority()
        devs = list_output_devices()

        def resolve(label):
            if not label:
                return None
            for i, lb, _c in devs:
                if lb == label:
                    return i
            for i, lb, _c in devs:
                if base_name(lb) == base_name(label):
                    return i
            return None
        dev = resolve(device_label)
        pdev = resolve(phones_label)
        if device_label and dev is None:
            raise RuntimeError(f"audio device not found: {device_label}")
        self.engine.start(dev, block, tuple(master_chans), tuple(phones_chans) if phones_chans else None,
                          phones_device=pdev)
        return self.engine.out_channels

    def add_output(self, label: str, delay_s=None, chans=(0, 1)):
        self.engine.add_output(label, delay_s=delay_s, chans=tuple(chans))

    def calibrate(self, which: str | None):
        self.engine.calibrate(which)

    def set_want_deck_scope(self, on: bool):
        self.want_deck_scope = bool(on)

    def ping(self):
        return "pong"

    def shutdown(self):
        self.alive = False
        try:
            self.close_preview()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.automix.stop()
            self.engine.close_outputs()
            self.engine.stop()
        except Exception:  # noqa: BLE001
            pass
        for att in self.tracks.values():
            att.close()
        self.ring.close(unlink=True)

    # ---- snapshots --------------------------------------------------------------------------
    def _track_of(self, d) -> str | None:
        if d.audio is None:
            return None
        for name, att in self.tracks.items():
            if self._owns(d.audio, att):
                att.installed = True          # seen on a deck: may be freed once it leaves it
                return name
        return None

    def _deck_snap(self, target: str, d) -> dict:
        s = {k: getattr(d, k) for k in DECK_SCALARS}
        s["loop"] = tuple(d.loop) if d.loop else None
        s["hotcues"] = list(d.hotcues)
        s["eq"] = [float(x) for x in d.eq]
        s["fx"] = sorted(d.fx)
        s["length"] = d.length
        s["loaded"] = d.loaded
        s["base_rate"] = d.base_rate
        s["rate"] = d.rate
        s["bpm"] = d.bpm
        s["effective_key"] = d.effective_key
        s["track"] = self._track_of(d)
        s["info"] = asdict(d.info)
        if self.want_deck_scope and d.scope is not None and d.playing:
            s["scope"] = d.scope
        return s

    def _snapshot(self) -> dict:
        if not self.alive:
            return {}
        e, am = self.engine, self.automix
        snap = {k: getattr(e, k) for k in ENGINE_SCALARS}
        snap["decks"] = {n: self._deck_snap("eng", d) for n, d in e.decks.items()}
        snap["phase_err"] = {n: e.phase_error(n) if d.playing else None for n, d in e.decks.items()}
        snap["running"] = e.running
        snap["phones_separate"] = e._pstream is not None
        snap["virtual"] = getattr(e, "virtual", False)
        snap["deck_errors"] = list(e.deck_errors[-10:])
        snap["click_times"] = list(e.click_times[-60:])
        snap["output_delays"] = e.output_delays()
        snap["output_latencies"] = e.output_latencies()
        snap["reported_latency"] = e.reported_latency()
        snap["extra_outputs"] = [{"label": o.label, "alive": o.alive, "effective_delay": o.effective_delay,
                                  "delay_s": o.delay_s} for o in e.extra_outputs]
        snap["fx"] = [{"kind": u.kind, "on": u.on, "wet": u.wet, "amount": u.amount, "div_index": u.div_index}
                      for u in e.fx_units]
        snap["sync_outputs"] = None if e.sync_outputs is None else sorted(e.sync_outputs)
        snap["ring_pos"] = int(self.ring.hdr[0])
        tr = am.tr
        snap["automix"] = {
            "active": am.active, "status": am.status, "queue": list(am.queue), "history": list(am.history),
            "used_items": list(am.used_items), "aired": list(am.aired), "meta": sorted(am.meta),
            "loading": sorted(am._loading), "pending_skip": am.pending_skip,
            "resume_after_manual": am.resume_after_manual, "resume_after_takeover": am.resume_after_takeover,
            "rehearsal_fill": am.rehearsal_fill, "s": asdict(am.s),
            "tr": None if tr is None else {"started": tr.started, "in_deck": tr.in_deck, "out_deck": tr.out_deck,
                                           "beats": tr.beats, "start_beat": tr.start_beat,
                                           "in_start_beat": tr.in_start_beat, "style": tr.style},
        }
        pv, ss = self.preview, self.session
        if pv is not None and ss is not None:
            snap["preview"] = {
                "decks": {n: self._deck_snap("pv", d) for n, d in pv.decks.items()},
                "crossfader": pv.crossfader, "master_deck": pv.master_deck,
                "session": {"mode": ss.mode, "x": ss.x, "drop_beat": ss.drop_beat,
                            "replaying": ss.replaying is not None, "auto_record": ss.auto_record,
                            "fill": ss.fill, "lanes": ss.lanes if ss.mode == "recording" else None,
                            "result": ss.result.to_dict() if ss.mode == "done" and ss.result is not None
                            else None},
            }
        return snap

    def _snapshots(self):
        period = 1.0 / SNAP_HZ
        n = 0
        while self.alive:
            t0 = time.perf_counter()
            n += 1
            try:
                self.automix.maybe_resume()
                if n % 60 == 0:                              # every 2 s: free tracks no deck holds
                    self.engine.cmd(lambda e: self._release_unused())
                snap = self._snapshot()
                self.send("snap", snap)
                self._flush_releases()
                ss = self.session
                if ss is not None and ss.mode == "done" and ss.result is not None \
                        and ss.result is not self._session_sent:
                    self._session_sent = ss.result
                    self.event("rehearsal_done", ss.result.to_dict())
            except Exception:  # noqa: BLE001
                log.exception("snapshot failed")
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))


def serve(conn, cfg: dict | None = None):
    """Entry point of the child process."""
    import gc
    import sys

    sys.setswitchinterval(0.0005)
    gc.collect()
    gc.freeze()
    gc.set_threshold(50000, 50, 1000)
    # its own log next to the app's (a windowed build has no stderr): what happened in here
    handlers: list = []
    try:
        from .. import paths

        handlers.append(logging.FileHandler(__import__("os").path.join(paths.data_dir(), "audio.log"), mode="w",
                                            encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s audio %(levelname)s %(name)s: %(message)s",
                        handlers=handlers or None)
    try:
        from .outputs import raise_priority

        raise_priority()
    except Exception:  # noqa: BLE001
        pass
    Server(conn, cfg).run()
