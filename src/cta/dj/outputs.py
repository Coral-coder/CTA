"""Output plumbing: a second sound card for the headphones, delay lines to
line outputs up, device rescans and latency calibration.

Bluetooth speakers and headphones play audio 150-300 ms after it was
rendered. Tempest keeps one "delay" per output (what the driver reports, or
what the DJ measured with the tap test) and uses it to

* delay the faster output so master and headphones stay in time, and
* draw waveforms that late, so what you see matches what you hear.
"""

from __future__ import annotations

import re
import threading

import numpy as np

BLUETOOTH_HINTS = re.compile(r"bluetooth|airpods|beats|\bbt\b|bose|jbl|sony wh|sony wf|soundcore|"
                             r"hands-?free|a2dp|galaxy buds|pixel buds|marshall|ue boom|sonos", re.I)
BLUETOOTH_GUESS_MS = 200.0
TAP_ANTICIPATION_S = 0.02      # people tap a little ahead of a steady beat


def is_bluetooth(name: str) -> bool:
    return bool(BLUETOOTH_HINTS.search(name or ""))


def rescan_devices() -> None:
    """Make PortAudio see devices plugged in (or paired) since start-up.
    Only call it while no stream is open."""
    import sounddevice as sd

    try:
        sd._terminate()
    finally:
        sd._initialize()


class Ring:
    """Thread-safe stereo FIFO between the main audio callback and a second
    device's callback. The two sound cards run on their own clocks, so it
    keeps its fill near `target` frames: overfull -> skip ahead, empty ->
    silence (a tiny click at worst, never growing delay)."""

    def __init__(self, target: int = 2048, capacity: int = 32768):
        self.buf = np.zeros((capacity, 2), np.float32)
        self.cap = capacity
        self.target = target
        self.r = self.w = 0
        self.lock = threading.Lock()
        self.underruns = 0
        self.skips = 0

    @property
    def fill(self) -> int:
        return (self.w - self.r) % self.cap

    def write(self, x: np.ndarray):
        """x: (n, 2) float32."""
        n = len(x)
        with self.lock:
            if n >= self.cap:
                x = x[-(self.cap - 1):]
                n = len(x)
            end = self.w + n
            if end <= self.cap:
                self.buf[self.w:end] = x
            else:
                k = self.cap - self.w
                self.buf[self.w:] = x[:k]
                self.buf[:n - k] = x[k:]
            self.w = end % self.cap
            if self.fill > self.target * 3:              # the reader runs slow: drop to the target
                self.r = (self.w - self.target) % self.cap
                self.skips += 1

    def read(self, n: int) -> np.ndarray:
        out = np.zeros((n, 2), np.float32)
        with self.lock:
            avail = min(self.fill, n)
            if avail < n:
                self.underruns += 1
            end = self.r + avail
            if end <= self.cap:
                out[:avail] = self.buf[self.r:end]
            else:
                k = self.cap - self.r
                out[:k] = self.buf[self.r:]
                out[k:avail] = self.buf[:avail - k]
            self.r = end % self.cap
        return out


class Delay:
    """Fixed delay line for a (2, n) signal (up to `max_s` seconds)."""

    def __init__(self, sr: int, max_s: float = 1.0):
        self.cap = int(sr * max_s) + 1
        self.buf = np.zeros((2, self.cap), np.float32)
        self.pos = 0
        self.frames = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        d = int(min(max(self.frames, 0), self.cap - 1))
        if d == 0:
            return x
        n = x.shape[1]
        out = np.empty_like(x)
        for i in range(0, n, self.cap - d):          # works for any block size
            seg = x[:, i:i + self.cap - d]
            m = seg.shape[1]
            idx_w = (self.pos + np.arange(m)) % self.cap
            self.buf[:, idx_w] = seg
            idx_r = (self.pos - d + np.arange(m)) % self.cap
            out[:, i:i + m] = self.buf[:, idx_r]
            self.pos = (self.pos + m) % self.cap
        return out


def estimate_latency(taps: list[float], clicks: list[float], min_taps: int = 8) -> float | None:
    """Tap-along calibration: each tap is matched to the click it answers
    (the latest click before it, within one interval). Returns the output
    delay in seconds (median, first taps skipped while the DJ finds the
    beat), or None when the taps are too few or too scattered."""
    if len(clicks) < 2 or len(taps) < min_taps:
        return None
    c = np.asarray(sorted(clicks))
    period = float(np.median(np.diff(c)))
    offs = []
    for t in taps[3:]:
        j = np.searchsorted(c, t) - 1
        if j < 0:
            continue
        d = t - c[j]
        if 0 <= d < period:
            offs.append(d)
    if len(offs) < min_taps - 3:
        return None
    offs = np.asarray(offs)
    med = float(np.median(offs))
    if float(np.median(np.abs(offs - med))) > 0.06:        # all over the place: try again
        return None
    return max(0.0, med + TAP_ANTICIPATION_S)


# --------------------------------------------------------------------------
# Live device scans and extra outputs in helper processes
# --------------------------------------------------------------------------
# PortAudio only sees the devices that existed when it started, and re-reading
# them means closing every stream. So a scan runs in a short-lived helper
# process, and every extra output (a second / third Bluetooth speaker...) is
# played by its own helper process that has a fresh PortAudio: devices come and
# go without the music in the main process ever stopping.

_keep_awake = []          # macOS activity token: must stay referenced for the activity to last


def raise_priority(audio_process: bool = False) -> None:
    """Keep the music steady when Tempest is minimised or in the background.

    Windows: above-normal priority (HIGH for the audio process), no power
    throttling ("efficiency mode" slows background apps on Windows 11) and
    1 ms timer precision. macOS: no App Nap - a latency-critical,
    user-initiated activity (also set in the app's Info.plist). Linux: the
    audio process asks for a better nice level (granted when allowed)."""
    import sys

    if sys.platform == "win32":
        _windows_no_throttle(high=audio_process)
    elif sys.platform == "darwin":
        _mac_no_app_nap()
    elif audio_process:
        try:
            import os

            os.nice(-10)
        except (OSError, AttributeError):
            pass


def boost_audio_thread(period_s: float) -> bool:
    """Called once from inside the audio callback: make *this* thread the
    most important one on the machine. Windows: time-critical priority and
    the MMCSS "Pro Audio" class (what every DAW uses). macOS: a real-time
    time-constraint policy sized to the block (CoreAudio's own thread
    already has one; a PortAudio worker may not). Linux: SCHED_FIFO when the
    user may (rtprio limits), silently not otherwise."""
    import sys

    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.windll.kernel32
            k32.GetCurrentThread.restype = wintypes.HANDLE
            k32.SetThreadPriority(k32.GetCurrentThread(), 15)          # THREAD_PRIORITY_TIME_CRITICAL
            try:
                avrt = ctypes.windll.avrt
                idx = wintypes.DWORD(0)
                avrt.AvSetMmThreadCharacteristicsW.restype = wintypes.HANDLE
                h = avrt.AvSetMmThreadCharacteristicsW("Pro Audio", ctypes.byref(idx))
                if h:
                    avrt.AvSetMmThreadPriority(h, 2)                     # AVRT_PRIORITY_CRITICAL
            except Exception:  # noqa: BLE001
                pass
            return True
        if sys.platform == "darwin":
            import ctypes

            libc = ctypes.CDLL(None)

            class TimebaseInfo(ctypes.Structure):
                _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]

            tb = TimebaseInfo()
            libc.mach_timebase_info(ctypes.byref(tb))
            ns_to_abs = tb.denom / tb.numer if tb.numer else 1.0
            period = int(period_s * 1e9 * ns_to_abs)

            class TimeConstraint(ctypes.Structure):
                _fields_ = [("period", ctypes.c_uint32), ("computation", ctypes.c_uint32),
                            ("constraint", ctypes.c_uint32), ("preemptible", ctypes.c_int)]

            pol = TimeConstraint(period, max(period // 5, 1), max(period // 2, 1), 1)
            libc.mach_thread_self.restype = ctypes.c_uint32
            thread = libc.mach_thread_self()
            THREAD_TIME_CONSTRAINT_POLICY, COUNT = 2, 4
            r = libc.thread_policy_set(thread, THREAD_TIME_CONSTRAINT_POLICY, ctypes.byref(pol), COUNT)
            return r == 0
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)

        class SchedParam(ctypes.Structure):
            _fields_ = [("sched_priority", ctypes.c_int)]

        libc.pthread_self.restype = ctypes.c_void_p
        SCHED_FIFO = 1
        r = libc.pthread_setschedparam(ctypes.c_void_p(libc.pthread_self()), SCHED_FIFO,
                                       ctypes.byref(SchedParam(60)))
        return r == 0
    except Exception:  # noqa: BLE001
        return False


def _windows_no_throttle(high: bool = False):
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        proc = k32.GetCurrentProcess()
        ABOVE_NORMAL_PRIORITY_CLASS, HIGH_PRIORITY_CLASS = 0x8000, 0x80
        k32.SetPriorityClass(proc, HIGH_PRIORITY_CLASS if high else ABOVE_NORMAL_PRIORITY_CLASS)

        class PowerThrottling(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]
        # ProcessPowerThrottling = 4; execution speed + timer resolution: controlled, and OFF
        pt = PowerThrottling(1, 0x1 | 0x4, 0)
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation(proc, 4, ctypes.byref(pt), ctypes.sizeof(pt))
    except Exception:  # noqa: BLE001 - older Windows: no such setting
        pass
    try:
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:  # noqa: BLE001
        pass


def _mac_no_app_nap():
    if _keep_awake:
        return
    try:
        import ctypes
        import ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        send = objc.objc_msgSend
        send.restype = ctypes.c_void_p
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        info = send(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
        nsstring = objc.objc_getClass(b"NSString")
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
        reason = send(nsstring, objc.sel_registerName(b"stringWithUTF8String:"), b"Playing music (DJ mode)")
        # NSActivityUserInitiated | NSActivityLatencyCritical
        opts = 0x00FFFFFF | 0xFF00000000
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        token = send(info, objc.sel_registerName(b"beginActivityWithOptions:reason:"), opts, reason)
        if token:
            send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            send(token, objc.sel_registerName(b"retain"))
            _keep_awake.append(token)
    except Exception:  # noqa: BLE001
        pass


def cli_command() -> list[str]:
    """How to run Tempest's command line (frozen builds or from source)."""
    import os
    import sys

    if getattr(sys, "frozen", False):
        if sys.platform == "win32":
            d = os.path.dirname(sys.executable)
            cli = next((p for p in (os.path.join(d, "tempest-cli.exe"), os.path.join(d, "cta-cli.exe"))
                        if os.path.exists(p)), os.path.join(d, "tempest-cli.exe"))
            if os.path.exists(cli):
                return [cli]
        return [sys.executable]
    return [sys.executable, "-m", "cta"]


def _popen_flags() -> dict:
    import subprocess
    import sys

    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def scan_devices(timeout: float = 15.0) -> list[dict]:
    """Every output device connected right now, read by a helper process so
    the running audio is untouched. [{label, name, channels, bluetooth}]."""
    import json
    import os
    import subprocess
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".json", prefix="cta-devices-")
    os.close(fd)
    try:
        subprocess.run(cli_command() + ["devices", "--out", path], timeout=timeout, check=False,
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       **_popen_flags())
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def device_records() -> list[dict]:
    """The device list as the helper writes it (runs inside the helper)."""
    from .engine import list_output_devices

    out = []
    for _i, label, ch in list_output_devices():
        name = label.rsplit(" (", 1)[0]
        out.append({"label": label, "name": name, "channels": ch, "bluetooth": is_bluetooth(label)})
    return out


def base_name(label: str) -> str:
    """'JBL Flip 6 (Windows WASAPI, 2 ch)' -> 'JBL Flip 6' (one device, several driver APIs)."""
    return (label or "").rsplit(" (", 1)[0].strip()


class ExtraOutput:
    """One extra output device played by a helper process. The main audio
    callback hands it blocks (never waiting); a writer thread pipes them on."""

    def __init__(self, label: str, chans=(0, 1), delay_s: float | None = None, block: int = 512,
                 sr: int = 44100, cmd: list[str] | None = None, null: bool = False, stats: str | None = None):
        import queue as _q
        import subprocess

        self.label = label
        self.chans = chans
        self.delay_s = delay_s                 # None: driver latency + Bluetooth guess
        self.align = Delay(sr)                 # holds this output back to line up with the slowest
        self.q: _q.Queue = _q.Queue(maxsize=64)
        self.alive = True
        args = (cmd or cli_command()) + ["output-helper", "--device", label, "--chans", f"{chans[0]},{chans[1]}",
                                         "--block", str(block)]
        if null:
            args.append("--null")
        if stats:
            args += ["--stats", stats]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, **_popen_flags())
        self.t = threading.Thread(target=self._pump, daemon=True, name=f"out-{base_name(label)}")
        self.t.start()

    @property
    def effective_delay(self) -> float:
        if self.delay_s is not None:
            return self.delay_s
        return (BLUETOOTH_GUESS_MS / 1000.0) if is_bluetooth(self.label) else 0.03

    def feed(self, block: np.ndarray):
        """block: (2, n) float32 from the audio thread - never blocks."""
        if not self.alive:
            return
        try:
            self.q.put_nowait(np.ascontiguousarray(self.align.process(block).T, dtype=np.float32).tobytes())
        except Exception:  # noqa: BLE001 - full: the helper fell behind, drop
            pass

    def _pump(self):
        import queue as _q

        try:
            while self.proc.poll() is None:
                try:
                    data = self.q.get(timeout=0.5)
                except _q.Empty:
                    continue
                if data is None:
                    break
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        self.alive = False

    def close(self):
        """Stop after sending what is already queued (a second at most)."""
        try:
            self.q.put(None, timeout=1)
        except Exception:  # noqa: BLE001
            pass
        self.t.join(timeout=2)
        self.alive = False
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def run_output_helper(device: str, chans=(0, 1), block: int = 512, null: bool = False,
                      stats: str | None = None, sr: int = 44100) -> int:
    """Body of `cta output-helper`: play raw float32 stereo from stdin on a
    device until stdin closes (null: just consume, for tests)."""
    import json
    import os
    import time

    # Bluetooth stacks pull audio in bursts: a deeper cushion than a wired card
    # needs, so they never run dry (stutter). Its delay is part of the output's
    # measured delay, which the clock lines up.
    ring = Ring(target=max(4 * block, 4096), capacity=131072)
    got = {"frames": 0}
    done = threading.Event()

    def reader():
        pending = b""
        while True:
            try:
                chunk = os.read(0, 65536)
            except OSError:
                break
            if not chunk:
                break
            pending += chunk
            n = len(pending) // 8 * 8
            if n:
                x = np.frombuffer(pending[:n], dtype=np.float32).reshape(-1, 2)
                pending = pending[n:]
                got["frames"] += len(x)
                if not null:
                    ring.write(x)
        done.set()

    threading.Thread(target=reader, daemon=True).start()
    stream = None
    if not null:
        import sounddevice as sd

        from .engine import list_output_devices

        idx = next((i for i, label, _c in list_output_devices() if label == device), None)
        if idx is None:
            idx = next((i for i, label, _c in list_output_devices() if base_name(label) == base_name(device)), None)
        if idx is None:
            return 2
        info = sd.query_devices(idx, "output")
        n = int(min(info["max_output_channels"], max(2, max(chans) + 1)))
        c = chans if max(chans) < n else (0, 1)

        def cb(outdata, frames, _ti, _st):
            x = ring.read(frames)
            outdata.fill(0)
            outdata[:, c[0]] = x[:, 0]
            outdata[:, c[1]] = x[:, 1]
        extra = None
        if "WASAPI" in sd.query_hostapis(info["hostapi"])["name"]:
            extra = sd.WasapiSettings(auto_convert=True)
        stream = sd.OutputStream(samplerate=sr, blocksize=block, channels=n, dtype="float32", device=idx,
                                 latency="low", callback=cb, extra_settings=extra)
        stream.start()
    done.wait()
    if stream is not None:
        time.sleep(0.2)
        stream.stop()
        stream.close()
    if stats:
        with open(stats, "w", encoding="utf-8") as fh:
            json.dump(got, fh)
    return 0
