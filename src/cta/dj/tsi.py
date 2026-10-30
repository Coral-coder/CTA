"""Reading Traktor settings / mapping files (.tsi).

A .tsi is XML; the controller mapping is the base64 entry
"DeviceIO.Config.Controller", a tree of frames (4-letter tag, big-endian
size, body). Each device (DEVI) has its name and its list of inputs (DDCI)
and outputs (DDCO), every one a DCDT frame that starts with the control's
name. Traktor's built-in controllers (the Pioneer DDJ-T1 among them) are
listed by control name, not MIDI message: CTA supplies the MIDI messages
for those devices from the maker's MIDI list.
"""

from __future__ import annotations

import base64
import re
import struct


class TsiError(ValueError):
    pass


def _frames(b: bytes, off: int, end: int):
    while off + 8 <= end:
        tag = b[off:off + 4]
        size = struct.unpack(">I", b[off + 4:off + 8])[0]
        if not tag.isalpha() or off + 8 + size > end:
            return
        yield tag.decode("ascii"), off + 8, off + 8 + size
        off += 8 + size


def _wstr(b: bytes, off: int) -> tuple[str, int]:
    n = struct.unpack(">I", b[off:off + 4])[0]
    return b[off + 4:off + 4 + 2 * n].decode("utf-16-be", "replace"), off + 4 + 2 * n


def _child(b: bytes, s: int, e: int, tag: str):
    for t, cs, ce in _frames(b, s, e):
        if t == tag:
            return cs, ce
    return None


def _controls(b: bytes, s: int, e: int) -> list[str]:
    """A DDCI / DDCO body: a count, then DCDT frames starting with the name."""
    names = []
    for t, cs, _ce in _frames(b, s + 4, e):
        if t == "DCDT":
            names.append(_wstr(b, cs)[0])
    return names


def read_tsi(path: str) -> list[dict]:
    """[{name, inputs: [...], outputs: [...]}] for every controller in the file."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as ex:
        raise TsiError(f"Could not read the file: {ex}") from ex
    m = re.search(r'<Entry Name="DeviceIO\.Config\.Controller"[^>]*Value="([^"]*)"', text)
    if not m:
        raise TsiError("This .tsi has no controller mapping in it (export it from Traktor's "
                       "Controller Manager, or export all settings).")
    try:
        b = base64.b64decode(m.group(1))
    except ValueError as ex:
        raise TsiError("The controller mapping in this file is damaged.") from ex
    devices = []
    root = _child(b, 0, len(b), "DIOM")
    if root is None:
        raise TsiError("Unknown controller-mapping format.")
    devs = _child(b, *root, "DEVS")
    if devs is None:
        return []
    for t, cs, ce in _frames(b, devs[0] + 4, devs[1]):
        if t != "DEVI":
            continue
        name, p = _wstr(b, cs)
        dev = {"name": name, "inputs": [], "outputs": []}
        dd = _child(b, p, ce, "DDAT")
        if dd is not None:
            dc = _child(b, *dd, "DDDC")
            if dc is not None:
                for key, tag in (("inputs", "DDCI"), ("outputs", "DDCO")):
                    f = _child(b, *dc, tag)
                    if f is not None:
                        dev[key] = _controls(b, *f)
        devices.append(dev)
    return devices


# Traktor's DDJ-T1 control names that CTA plays (others are listed as not used)
_COVERED = re.compile(r"\.(Loop Active|Auto Loop|Key Lock|Cue Delete|HotCue|Loop Size|EQ\.|Vol$|FX[12] Assign|"
                      r"Cue$|Sync|Master|Load)|X Fader|Dry/Wet|Parameter 1|Browse$|Browse Push|Favorites|"
                      r"Main Vol|HP Mix|HP Vol|^Shift$|Needle\.Search")


def tsi_bindings(path: str):
    """(device name, bindings, controls CTA does not use) from a Traktor .tsi."""
    from .midi import ddj_t1_default

    devices = read_tsi(path)
    for dev in devices:
        if "DDJ-T1" in dev["name"].upper():
            unused = sorted({re.sub(r"\b[A-D]\.", "", n) for n in dev["inputs"] if not _COVERED.search(n)})
            return dev["name"], ddj_t1_default(), unused
    names = ", ".join(d["name"] for d in devices) or "none"
    raise TsiError(f"Controllers in this file: {names}. CTA can import the Pioneer DDJ-T1 from a Traktor "
                   "mapping so far; for other controllers use the guided setup.")
