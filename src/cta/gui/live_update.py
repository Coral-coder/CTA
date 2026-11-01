"""Live update: the running version hands the mix to the new one without
stopping the music (Windows and macOS; Linux and stopped decks restart the
old way).

  old version (host)                        new version (guest)
  ------------------                        -------------------
  copies the new build aside, starts it --> starts muted, loads the same tracks
  answers "state" and "clock"          <--> sets every deck up the same way, plays
                                            muted, lines up on the old one's clock
  fades out at T                       <--  "switch at T": fades in at T
  lets go of the controller, "bye"     -->  takes the controller
  installs the update on exit, quits

Both processes use perf_counter, which is the same clock in every process on
Windows (QPC) and macOS (mach_absolute_time). The old version keeps playing
until the new one has said "switch": if anything goes wrong before that, the
new one quits silently and the update waits until the decks stop.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time

from PySide6.QtCore import QObject, QTimer, Signal

from .. import __version__, paths
from ..dj.engine import SR

log = logging.getLogger(__name__)

ENV = "TEMPEST_HANDOVER"          # host:port:token - set for the new version only
FADE_S = 0.15                     # the crossfade between the two versions
LEAD_S = 0.6                      # how far ahead the switch is scheduled
GIVE_UP_S = 180.0                 # the new version must be ready within this


def supported() -> bool:
    return sys.platform in ("win32", "darwin")


def playing(dj) -> bool:
    return dj is not None and any(d.playing and not d.cue_preview for d in dj.engine.decks.values())


# --------------------------------------------------------------------------
# wire format: one JSON object per line
# --------------------------------------------------------------------------

def _send(sock, obj):
    sock.sendall((json.dumps(obj) + "\n").encode())


def _lines(sock):
    buf = b""
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip():
                yield json.loads(line)


def _clock(dj):
    c = dj.engine.handover_clock()
    if not c:
        return None
    t, pos = c
    return [float(t), {k: [float(v[0]), float(v[1])] for k, v in pos.items()}]


def capture_state(main) -> dict:
    """Everything the new version needs to carry on the same mix."""
    dj, lib = main.dj, main.lib
    eng = dj.engine
    decks = {}
    for n, d in eng.decks.items():
        if not d.loaded or not d.info.path:
            continue
        decks[n] = dict(path=d.info.path, playing=bool(d.playing and not d.cue_preview), tempo=float(d.tempo),
                        tempo_range=float(d.tempo_range), keylock=bool(d.keylock), sync=bool(d.sync),
                        volume=float(d.volume), eq=[float(x) for x in d.eq], trim=float(d.trim), gain=float(d.gain),
                        filter=float(d.filter), key_shift=float(d.key_shift), cue_phones=bool(d.cue_phones),
                        loop_size=float(d.loop_size), cue=float(d.cue_point) / SR,
                        phase_offset=float(d.phase_offset),
                        loop=[float(x) for x in d.loop] if d.loop else None,
                        hotcues=[None if c is None else float(c) for c in d.hotcues])
    queue = []
    for i in list(dj.automix.queue):
        try:
            queue.append(lib.paths[int(i)])
        except (TypeError, ValueError, IndexError):
            pass
    return dict(version=__version__, decks=decks, master=eng.master_deck,
                crossfader=float(eng.crossfader), master_volume=float(eng.master_volume),
                phones_volume=float(eng.phones_volume), cue_mix=float(eng.cue_mix),
                automix=bool(dj.automix.active), queue=queue, show_cd=bool(dj.show_cd.isChecked()),
                set_started=getattr(dj, "_set_started", None))


# --------------------------------------------------------------------------
# the running version
# --------------------------------------------------------------------------

class LiveUpdateHost(QObject):
    """In the old version: start the new one and hand it the mix."""

    _finish = Signal()
    _failed = Signal(str)
    _released = Signal()

    def __init__(self, main, new_root: str, on_fail):
        super().__init__(main)
        self.main = main
        self.new_root = new_root
        self.on_fail = on_fail                 # called (GUI thread) with a reason: install the old way later
        self.token = secrets.token_hex(16)
        self.proc = None
        self.switched = False
        self.done = False
        self._finish.connect(self._quit_into_new)
        self._failed.connect(self._fail)
        self._released.connect(self._device_released)

    # ---- start ---------------------------------------------------------------------
    def start(self):
        self.main.status_label.setText("Live update: starting the new version in the background - keep playing")
        threading.Thread(target=self._run, daemon=True, name="live-update").start()
        QTimer.singleShot(int(GIVE_UP_S * 1000), self._timeout)

    def _run(self):
        try:
            live = self._copy_aside()
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            srv.settimeout(GIVE_UP_S)
            port = srv.getsockname()[1]
            self.proc = _launch(live, f"127.0.0.1:{port}:{self.token}")
            conn, _ = srv.accept()
            srv.close()
            conn.settimeout(None)
            self._serve(conn)
        except Exception as e:  # noqa: BLE001
            log.exception("live update failed")
            if not self.done:
                self._failed.emit(str(e))          # (brings this version's audio back if it had let go)

    def _copy_aside(self) -> str:
        """The new version runs from its own copy: the installer moves the staged
        one into place when this version exits, and must not touch running files."""
        base = os.path.join(paths.data_dir(), "live", f"{int(time.time())}")
        os.makedirs(base, exist_ok=True)
        dest = os.path.join(base, os.path.basename(self.new_root.rstrip("/\\")))
        if sys.platform == "darwin":
            subprocess.run(["ditto", self.new_root, dest], check=True)     # keeps the bundle's links and signature
        else:
            shutil.copytree(self.new_root, dest)
        return dest

    def _serve(self, conn):
        dj = self.main.dj
        for msg in _lines(conn):
            op = msg.get("op")
            if op == "hello":
                if msg.get("token") != self.token:
                    raise RuntimeError("wrong token")
                _send(conn, {"op": "state", "state": capture_state(self.main)})
            elif op == "state":
                _send(conn, {"op": "state", "state": capture_state(self.main)})
            elif op == "clock":
                _send(conn, {"op": "clock", "clock": _clock(dj)})
            elif op == "need_device":
                # the sound card cannot be shared: a short gap - fade, close, hand over the clock
                dj.engine.handover_fade(time.perf_counter(), 0.08, False)
                time.sleep(0.12)
                clock = _clock(dj)
                self.switched = True
                self._released.emit()          # closes this version's sound card (GUI thread)
                time.sleep(0.4)
                _send(conn, {"op": "released", "clock": clock})
            elif op == "started":
                self._finish.emit()            # the new version plays: this one leaves
                time.sleep(1.0)
                _send(conn, {"op": "bye"})
                return
            elif op == "switch":
                t = float(msg["t"])
                dj.engine.handover_fade(t, float(msg.get("fade", FADE_S)), False)
                self.switched = True
                _send(conn, {"op": "ok"})
                wait = t + float(msg.get("fade", FADE_S)) + 0.25 - time.perf_counter()
                time.sleep(max(0.0, wait))
                self._finish.emit()
                time.sleep(1.0)                # the controller is closed on the GUI thread
                _send(conn, {"op": "bye"})
                return
            elif op == "abort":
                raise RuntimeError(msg.get("why") or "the new version gave up")
        raise RuntimeError("the new version closed the connection")

    # ---- end ---------------------------------------------------------------------------
    def _device_released(self):
        try:
            self.main.dj.engine.stop()
        except Exception:  # noqa: BLE001
            log.exception("closing the sound card for the new version")

    def _quit_into_new(self):
        if self.done:
            return
        self.done = True
        from .. import updater

        dj = self.main.dj
        try:
            dj._handing_over = True
            dj.keepawake.off(restore=False)    # the new version keeps the computer awake from here
        except Exception:  # noqa: BLE001
            pass
        try:
            if dj.controller is not None:
                dj.controller.close()          # the new version opens the controller next
                dj.controller = None
        except Exception:  # noqa: BLE001
            pass
        try:
            updater.launch_swap(self.new_root, relaunch=False)    # installed for the next start
        except Exception:  # noqa: BLE001
            log.exception("could not schedule the install - the new version still runs from its copy")
        log.info("live update: handed the mix to the new version")
        try:
            self.main.hide()                   # gone from the screen at once: one Tempest, the new one
            if getattr(self.main, "tray", None) is not None:
                self.main.tray.hide()
        except Exception:  # noqa: BLE001
            pass
        # last resort: whatever holds this process up (a stuck close, a thread that will
        # not end), it goes - it is silent and has let go of everything
        t = threading.Timer(8.0, self._hard_exit)
        t.daemon = True
        t.start()
        QTimer.singleShot(1500, self._exit)

    def _exit(self):
        from PySide6.QtWidgets import QApplication

        try:
            self.main.quit_app()
        except Exception:  # noqa: BLE001
            log.exception("closing the old version after the live update")
        QApplication.instance().quit()

    def _hard_exit(self):
        log.warning("live update: the old version did not close by itself - ending it")
        try:
            self.main.dj.engine.close()        # the audio process goes with it
        except Exception:  # noqa: BLE001
            pass
        try:
            logging.shutdown()
        finally:
            os._exit(0)

    def _timeout(self):
        if not self.switched and not self.done:
            self._fail("the new version took too long to get ready")

    def _fail(self, why: str):
        if self.done:
            return
        self.done = True
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass
        if self.switched:
            # the sound card was already released: bring this version's audio back
            try:
                self.main.dj.engine.handover_fade(float("inf"), FADE_S, False)
                self.main.dj.start_audio(quiet=True)
            except Exception:  # noqa: BLE001
                log.exception("restarting audio after a failed live update")
        log.warning("live update failed: %s", why)
        self.on_fail(why)


def _launch(app_root: str, env_value: str):
    """Start the copied new version with the handover address."""
    from .. import updater

    env = dict(os.environ)
    env[ENV] = env_value
    target = updater.update_target()
    if target:
        env["TEMPEST_INSTALL_TARGET"] = target     # the copy is temporary: the real install stays the target
    if sys.platform == "darwin":
        name = os.path.splitext(os.path.basename(app_root))[0]
        exe = os.path.join(app_root, "Contents", "MacOS", name)
        return subprocess.Popen([exe], env=env, start_new_session=True, close_fds=True,
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    exe = os.path.join(app_root, "Tempest.exe")
    flags = 0x00000008 | 0x00000200 | 0x01000000   # DETACHED, NEW_PROCESS_GROUP, BREAKAWAY_FROM_JOB
    try:
        return subprocess.Popen([exe], env=env, creationflags=flags, close_fds=True,
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return subprocess.Popen([exe], env=env, creationflags=0x00000008 | 0x00000200, close_fds=True,
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _on_card(eng) -> bool:
    """Really playing through the sound card (not the silent stand-in it falls back to)."""
    return bool(eng.running) and not bool(getattr(eng, "virtual", False))


def cleanup_old_copies():
    """Copies of earlier live updates that are no longer running."""
    base = os.path.join(paths.data_dir(), "live")
    if not os.path.isdir(base):
        return
    here = os.path.abspath(sys.executable)
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if here.startswith(os.path.abspath(p) + os.sep):
            continue                            # this version runs from it
        shutil.rmtree(p, ignore_errors=True)


# --------------------------------------------------------------------------
# the new version
# --------------------------------------------------------------------------

class LiveUpdateGuest(QObject):
    """In the new version: take the running mix over, muted until the switch."""

    _bye = Signal()

    def __init__(self, main, env_value: str):
        super().__init__(main)
        self.main = main
        host, port, self.token = env_value.split(":", 2)
        self.addr = (host, int(port))
        self.sock = None
        self.state = None
        self.t0 = time.monotonic()
        self._bye.connect(self._take_controller)
        self._replies: list = []
        self._ev = threading.Event()

    # ---- talking to the old version -------------------------------------------------------
    def start(self):
        try:
            self.sock = socket.create_connection(self.addr, timeout=10)
            self.sock.settimeout(None)
        except OSError as e:
            log.error("live update: cannot reach the running version: %s", e)
            self._give_up(str(e), tell=False)
            return
        threading.Thread(target=self._reader, daemon=True, name="live-update-guest").start()
        _send(self.sock, {"op": "hello", "token": self.token})
        QTimer.singleShot(250, self._step)

    def _reader(self):
        try:
            for msg in _lines(self.sock):
                if msg.get("op") == "bye":
                    self._bye.emit()
                    return
                self._replies.append(msg)
                self._ev.set()
        except OSError:
            pass

    def _ask(self, op: str, want: str, timeout: float = 10.0, **kw):
        self._replies.clear()
        self._ev.clear()
        _send(self.sock, {"op": op, **kw})
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for m in list(self._replies):
                if m.get("op") == want:
                    return m
            self._ev.wait(0.05)
            self._ev.clear()
        raise TimeoutError(f"no {want} from the running version")

    # ---- the steps (GUI thread, polled so nothing blocks the window) --------------------------
    def _step(self):
        try:
            self._advance()
        except Exception as e:  # noqa: BLE001
            log.exception("live update (new version) failed")
            self._give_up(str(e))

    def _later(self, ms=200):
        if time.monotonic() - self.t0 > GIVE_UP_S - 10:
            self._give_up("not ready in time")
            return
        QTimer.singleShot(ms, self._step)

    def _advance(self):
        main = self.main
        if self.state is None:
            for m in list(self._replies):
                if m.get("op") == "state":
                    self.state = m["state"]
            if self.state is None:
                return self._later()
        if main.lib is None:
            return self._later()
        if getattr(main, "dj", None) is None:
            if not getattr(main, "_dj_opening", False):
                main.open_dj()
            return self._later(500)
        dj = main.dj
        if not getattr(self, "_loads_sent", False):
            self._loads_sent = True
            self._index = {os.path.normcase(os.path.normpath(p)): i for i, p in enumerate(main.lib.paths)}
            if self.state.get("show_cd"):
                dj.show_cd.setChecked(True)
            for n, st in self.state["decks"].items():
                idx = self._index.get(os.path.normcase(os.path.normpath(st["path"])))
                if idx is None:
                    if st["playing"]:
                        raise RuntimeError(f"deck {n}'s track is not in this library")
                    continue
                dj.load_index(n, idx)
            return self._later()
        eng = dj.engine
        for n, st in self.state["decks"].items():
            if st["playing"] and eng.decks[n].info.path != st["path"]:
                return self._later()                       # still loading
        if not _on_card(eng):
            if not getattr(self, "_asked_device", False) and time.monotonic() - self.t0 > 8:
                return self._take_device()
            return self._later()
        if not getattr(self, "_set_up", False):
            self._set_up = True
            self._apply_state(dj, self.state)
            return self._later(300)
        # fresh state: the DJ may have moved something while this version loaded
        now_state = self._ask("state", "state")["state"]
        for n, st in now_state["decks"].items():
            mine = self.state["decks"].get(n)
            if mine is None or mine["path"] != st["path"] or mine["playing"] != st["playing"]:
                raise RuntimeError("the DJ changed tracks meanwhile - trying again when the decks stop")
        self._apply_state(dj, now_state, levels_only=True)
        eng.handover_align(self._ask("clock", "clock")["clock"])
        QTimer.singleShot(250, self._switch_step)

    def _switch_step(self):
        try:
            self._switch()
        except Exception as e:  # noqa: BLE001
            log.exception("live update (new version) failed at the switch")
            self._give_up(str(e))

    def _switch(self):
        eng = self.main.dj.engine
        eng.handover_align(self._ask("clock", "clock")["clock"])   # once more, right before the switch
        t = time.perf_counter() + LEAD_S
        eng.handover_fade(t, FADE_S, True)
        self._ask("switch", "ok", t=t, fade=FADE_S)
        self.main.status_label.setText(f"Updated live to Tempest {__version__} - the music never stopped")
        log.info("live update: took over the mix")

    def _apply_state(self, dj, st, levels_only: bool = False):
        eng = dj.engine
        for k in ("crossfader", "master_volume", "phones_volume", "cue_mix"):
            try:
                setattr(eng, k, float(st[k]))
            except Exception:  # noqa: BLE001
                pass
        for n, ds in st["decks"].items():
            if eng.decks[n].info.path != ds["path"]:
                continue
            kw = dict(volume=ds["volume"], eq=tuple(ds["eq"]), trim=ds["trim"], filter=ds["filter"],
                      cue_phones=ds["cue_phones"])
            if not levels_only:
                kw.update(tempo_range=ds["tempo_range"], tempo=ds["tempo"], keylock=ds["keylock"],
                          key_shift=ds["key_shift"], gain=ds["gain"], loop_size=ds["loop_size"])
            eng.set(n, **kw)
            if not levels_only:
                eng.set_cue(n, ds["cue"])
                eng.set_loop(n, ds["loop"])
                eng.set_hotcues(n, ds["hotcues"])
                if ds["sync"]:
                    eng.set(n, sync=True)
                    eng.set(n, phase_offset=ds.get("phase_offset", 0.0))   # where the DJ's nudges left it
                if ds["playing"] and not eng.decks[n].playing:
                    eng.set_playing(n, True)
        if not levels_only:
            if st.get("master"):
                eng.set_master(st["master"])
            if st.get("set_started"):
                dj._set_started = float(st["set_started"])

    def _take_device(self):
        """The sound card is not shared between two programs here: a short gap."""
        self._asked_device = True
        m = self._ask("need_device", "released", timeout=15)
        dj = self.main.dj
        self._apply_state(dj, self.state)
        dj.start_audio(quiet=True)
        if not _on_card(dj.engine):
            raise RuntimeError("the new version could not open the sound card")
        dj.engine.handover_align(m.get("clock"))
        dj.engine.handover_fade(time.perf_counter(), 0.05, True)
        _send(self.sock, {"op": "started"})
        self.main.status_label.setText(f"Updated live to Tempest {__version__}")

    # ---- after the switch -------------------------------------------------------------------
    def _take_controller(self):
        dj = self.main.dj
        try:
            dj._auto_connect_midi()
        except Exception:  # noqa: BLE001
            log.exception("connecting the controller after the live update")
        st = self.state or {}
        if st.get("queue"):
            idx = [self._index[k] for k in (os.path.normcase(os.path.normpath(p)) for p in st["queue"])
                   if k in self._index]
            dj.automix.set_queue(idx)
            dj._show_queue()
        if st.get("automix"):
            dj.b_auto.setChecked(True)
            dj.toggle_automix(True)
        QTimer.singleShot(3000, self.main.claim_single_instance)
        try:
            self.sock.close()
        except OSError:
            pass

    def _give_up(self, why: str, tell: bool = True):
        log.warning("live update (new version) gave up: %s", why)
        if tell and self.sock is not None:
            try:
                _send(self.sock, {"op": "abort", "why": why})
            except OSError:
                pass
        from PySide6.QtWidgets import QApplication

        self.main.quit_app()
        QApplication.instance().quit()
