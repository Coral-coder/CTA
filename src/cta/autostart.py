"""Start CTA (as a background mix server) when the user logs in.

Windows: HKCU\\...\\Run registry value. macOS: a LaunchAgent. Linux: an XDG
autostart entry. Everything is per user; nothing needs admin rights.
"""

from __future__ import annotations

import os
import shlex
import sys

NAME = "CTA"
MAC_LABEL = "com.cta.mixserver"


def command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--background"]
    exe = sys.executable
    if sys.platform == "win32":                     # no console window at login
        w = os.path.join(os.path.dirname(exe), "pythonw.exe")
        exe = w if os.path.exists(w) else exe
    return [exe, "-m", "cta", "--background"]


def _win_cmdline(args: list[str]) -> str:
    import subprocess

    return subprocess.list2cmdline(args)


def _mac_plist() -> str:
    return os.path.expanduser(f"~/Library/LaunchAgents/{MAC_LABEL}.plist")


def _xdg_file() -> str:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "autostart", "cta.desktop")


def is_enabled() -> bool:
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as k:
                winreg.QueryValueEx(k, NAME)
            return True
        if sys.platform == "darwin":
            return os.path.exists(_mac_plist())
        return os.path.exists(_xdg_file())
    except OSError:
        return False


def set_enabled(on: bool) -> None:
    """Raises OSError when the setting cannot be written."""
    args = command()
    if sys.platform == "win32":
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                            winreg.KEY_SET_VALUE) as k:
            if on:
                winreg.SetValueEx(k, NAME, 0, winreg.REG_SZ, _win_cmdline(args))
            else:
                try:
                    winreg.DeleteValue(k, NAME)
                except FileNotFoundError:
                    pass
        return
    if sys.platform == "darwin":
        fn = _mac_plist()
        if not on:
            if os.path.exists(fn):
                os.remove(fn)
            return
        from xml.sax.saxutils import escape

        items = "".join(f"<string>{escape(a)}</string>" for a in args)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, "w", encoding="utf-8") as fh:
            fh.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                     '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                     '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
                     f'<plist version="1.0"><dict><key>Label</key><string>{MAC_LABEL}</string>'
                     f"<key>ProgramArguments</key><array>{items}</array>"
                     "<key>RunAtLoad</key><true/><key>ProcessType</key><string>Interactive</string>"
                     "</dict></plist>\n")
        return
    fn = _xdg_file()
    if not on:
        if os.path.exists(fn):
            os.remove(fn)
        return
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w", encoding="utf-8") as fh:
        fh.write("[Desktop Entry]\nType=Application\nName=CTA mix server\n"
                 f"Exec={shlex.join(args)}\nX-GNOME-Autostart-enabled=true\nNoDisplay=false\n")


def refresh() -> None:
    """Keep an enabled entry pointing at the current executable (e.g. after
    the app folder was moved)."""
    if is_enabled():
        try:
            set_enabled(True)
        except OSError:
            pass
