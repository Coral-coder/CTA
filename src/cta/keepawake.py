"""Keep the computer playing with the lid closed while music is playing.

  macOS    caffeinate stops idle sleep; closing the lid needs "pmset disablesleep",
           which only root may change - a one-time setup (the user's password once)
           adds a sudoers rule allowing exactly that command, nothing else.
  Windows  SetThreadExecutionState stops idle sleep; the power plan's
           "when I close the lid" action is set to "do nothing" (AC and battery)
           and put back afterwards.
  Linux    systemd-inhibit blocks sleep and the lid switch for this session.

Whatever is changed is put back when the music stops, when Tempest quits - and,
if Tempest crashes or is killed, by a small watcher process that waits for
Tempest to exit and restores the setting (a laptop must never stay unable to
sleep in a closed bag).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys

log = logging.getLogger(__name__)

SUDOERS = "/etc/sudoers.d/tempest-lid"
PMSET = "/usr/bin/pmset"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, timeout=15, creationflags=_NO_WINDOW, **kw)


def _detach(args):
    """Start a helper that outlives Tempest (the restore watcher)."""
    if sys.platform == "win32":
        flags = 0x00000008 | 0x00000200 | _NO_WINDOW          # DETACHED, NEW_PROCESS_GROUP
        try:
            return subprocess.Popen(args, creationflags=flags | 0x01000000, close_fds=True,   # + BREAKAWAY
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            return subprocess.Popen(args, creationflags=flags, close_fds=True, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return subprocess.Popen(args, start_new_session=True, close_fds=True, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --------------------------------------------------------------------------
# macOS
# --------------------------------------------------------------------------

def mac_lid_ready() -> bool:
    """The one-time setup is done: pmset disablesleep can be changed without a password."""
    if sys.platform != "darwin":
        return False
    try:
        return _run(["/usr/bin/sudo", "-n", "-l", PMSET, "-a", "disablesleep", "1"]).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def mac_lid_setup() -> tuple[bool, str]:
    """Ask for the user's password once (macOS dialog) and allow exactly
    "pmset -a disablesleep 0|1" without a password from then on."""
    if sys.platform != "darwin":
        return False, "only on macOS"
    rule = f"%admin ALL=(root) NOPASSWD: {PMSET} -a disablesleep 0, {PMSET} -a disablesleep 1"
    sh = (f"tmp=$(mktemp) && printf '%s\\n' '{rule}' > \"$tmp\" && /usr/sbin/visudo -cf \"$tmp\" "
          f"&& /usr/sbin/chown root:wheel \"$tmp\" && /bin/chmod 0440 \"$tmp\" && /bin/mv \"$tmp\" {SUDOERS}")
    quoted = sh.replace("\\", "\\\\").replace('"', '\\"')        # as an AppleScript string
    script = (f'do shell script "{quoted}" with administrator privileges '
              'with prompt "Tempest wants to keep the music playing when you close the lid."')
    try:
        r = _run(["/usr/bin/osascript", "-e", script])
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout).strip() or "cancelled"
    return mac_lid_ready(), ""


def _mac_disablesleep(on: bool) -> bool:
    try:
        return _run(["/usr/bin/sudo", "-n", PMSET, "-a", "disablesleep", "1" if on else "0"]).returncode == 0
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------

def _win_lid_values() -> tuple[int, int] | None:
    """(AC, battery) "lid close" action of the active power plan: 0 nothing, 1 sleep..."""
    try:
        r = _run(["powercfg", "/q", "SCHEME_CURRENT", "SUB_BUTTONS", "LIDACTION"])
    except Exception:  # noqa: BLE001
        return None
    vals = re.findall(r":\s*0x([0-9a-fA-F]+)\s*$", r.stdout, re.M)
    if len(vals) < 2:
        return None
    return int(vals[-2], 16), int(vals[-1], 16)          # the last two: current AC, current DC


def _win_set_lid(ac: int, dc: int) -> bool:
    ok = True
    for which, v in (("/setacvalueindex", ac), ("/setdcvalueindex", dc)):
        try:
            ok &= _run(["powercfg", which, "SCHEME_CURRENT", "SUB_BUTTONS", "LIDACTION", str(v)]).returncode == 0
        except Exception:  # noqa: BLE001
            ok = False
    try:
        _run(["powercfg", "/setactive", "SCHEME_CURRENT"])
    except Exception:  # noqa: BLE001
        pass
    return ok


def _win_exec_state(on: bool):
    try:
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_AWAYMODE_REQUIRED = 0x80000000, 0x00000001, 0x00000040
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------
# the switch
# --------------------------------------------------------------------------

class KeepAwake:
    """on() while music plays, off() when it has stopped (and on quit)."""

    def __init__(self, settings=None):
        self.settings = settings
        self.active = False
        self.lid_ok = False                   # the lid really is covered (not just idle sleep)
        self._procs: list = []
        self._win_saved: tuple[int, int] | None = None

    def on(self) -> bool:
        if self.active:
            return self.lid_ok
        self.active = True
        if os.environ.get("CTA_NO_KEEPAWAKE"):
            return True                      # (tests: never touch the machine's power settings)
        pid = os.getpid()
        try:
            if sys.platform == "darwin":
                self._procs.append(_detach(["/usr/bin/caffeinate", "-i", "-s", "-w", str(pid)]))
                if mac_lid_ready() and _mac_disablesleep(True):
                    self.lid_ok = True
                    # put it back when Tempest exits, however it exits
                    self._procs.append(_detach(["/bin/sh", "-c",
                                                f"while kill -0 {pid} 2>/dev/null; do sleep 3; done; "
                                                f"/usr/bin/sudo -n {PMSET} -a disablesleep 0"]))
            elif sys.platform == "win32":
                _win_exec_state(True)
                cur = _win_lid_values()
                if cur is not None:
                    if cur != (0, 0):
                        self._win_saved = cur
                        if self.settings is not None:
                            self.settings.setValue("lid_saved", f"{cur[0]},{cur[1]}")
                    elif self.settings is not None and self.settings.value("lid_saved"):
                        # already "do nothing" from the version this one took over from: its original
                        a_, d_ = (int(x) for x in str(self.settings.value("lid_saved")).split(","))
                        self._win_saved = (a_, d_)
                    if _win_set_lid(0, 0):
                        self.lid_ok = True
                        if self._win_saved:
                            ac, dc = self._win_saved
                            ps = (f"Wait-Process -Id {pid} -ErrorAction SilentlyContinue; "
                                  f"powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION {ac}; "
                                  f"powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION {dc}; "
                                  "powercfg /setactive SCHEME_CURRENT")
                            self._procs.append(_detach(["powershell.exe", "-NoProfile", "-NonInteractive",
                                                        "-WindowStyle", "Hidden", "-Command", ps]))
            elif shutil.which("systemd-inhibit"):
                self._procs.append(_detach(["systemd-inhibit", "--what=sleep:idle:handle-lid-switch",
                                            "--who=Tempest", "--why=DJ set playing", "--mode=block",
                                            "/bin/sh", "-c", f"while kill -0 {pid} 2>/dev/null; do sleep 3; done"]))
                self.lid_ok = True
        except Exception:  # noqa: BLE001
            log.exception("keeping the computer awake")
        log.info("keep awake: on (lid %s)", "covered" if self.lid_ok else "not covered - one-time setup needed")
        return self.lid_ok

    def reassert(self):
        """Every few seconds while on: nothing else (an old version going away after a
        live update, a settings window) may have turned lid sleep back on."""
        if not self.active or not self.lid_ok:
            return
        try:
            if sys.platform == "darwin":
                _mac_disablesleep(True)
            elif sys.platform == "win32":
                _win_exec_state(True)
                if _win_lid_values() not in (None, (0, 0)):
                    _win_set_lid(0, 0)
        except Exception:  # noqa: BLE001
            pass

    def off(self, restore: bool = True):
        """restore=False: a live update hands the setting to the new version (which
        keeps it and puts it back itself) - only this version's helpers go."""
        if not self.active:
            return
        self.active = False
        if not restore:
            for p in self._procs:
                try:
                    if p.poll() is None:
                        p.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._procs.clear()
            return
        try:
            if sys.platform == "darwin" and self.lid_ok:
                _mac_disablesleep(False)
            elif sys.platform == "win32":
                _win_exec_state(False)
                if self._win_saved:
                    _win_set_lid(*self._win_saved)
                    if self.settings is not None:
                        self.settings.remove("lid_saved")
        except Exception:  # noqa: BLE001
            log.exception("letting the computer sleep again")
        for p in self._procs:
            try:
                if p.poll() is None:
                    p.kill()                  # (the watcher's job is done: restored here)
            except Exception:  # noqa: BLE001
                pass
        self._procs.clear()
        self._win_saved = None
        self.lid_ok = False
        log.info("keep awake: off")


def restore_after_crash(settings):
    """Start-up: a lid setting left changed by a Tempest that could not put it
    back (power cut, forced shutdown) is put back now."""
    if os.environ.get("CTA_NO_KEEPAWAKE"):
        return
    try:
        if sys.platform == "win32":
            saved = settings.value("lid_saved")
            if saved:
                ac, dc = (int(x) for x in str(saved).split(","))
                _win_set_lid(ac, dc)
                settings.remove("lid_saved")
        elif sys.platform == "darwin" and mac_lid_ready():
            _mac_disablesleep(False)
    except Exception:  # noqa: BLE001
        log.exception("restoring the lid setting")
