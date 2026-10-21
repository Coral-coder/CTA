"""Musical key helpers: names, Camelot wheel codes and harmonic compatibility.

Keys are stored as integers: 0-11 = major with that tonic pitch class,
12-23 = minor (tonic = key - 12), -1 = unknown.
"""

from __future__ import annotations

import re

import numpy as np

NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def key_name(key: int) -> str:
    if key is None or key < 0:
        return ""
    return NOTE_NAMES[key % 12] + ("m" if key >= 12 else "")


def camelot(key: int) -> tuple[int, str] | None:
    """(number 1-12, 'A' minor / 'B' major)."""
    if key is None or key < 0:
        return None
    if key < 12:
        return ((key * 7) % 12 + 7) % 12 + 1, "B"
    rel_major = (key - 12 + 3) % 12
    return ((rel_major * 7) % 12 + 7) % 12 + 1, "A"


def camelot_str(key: int) -> str:
    c = camelot(key)
    return f"{c[0]}{c[1]}" if c else ""


def from_camelot(num: int, letter: str) -> int:
    letter = letter.upper()
    pc_major = ((num - 8) * 7) % 12   # inverse of the formula above
    if letter == "B":
        return pc_major
    return 12 + (pc_major - 3) % 12


_NOTE_ALIASES = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4, "F": 5, "E#": 5,
    "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11, "CB": 11,
}


def parse_key(text: str | None) -> int:
    """Parse tag key notations: 'Am', 'A minor', 'F#m', '8A', '11B', 'Ebmaj', 'o' (OpenKey e.g. '1m')."""
    if not text:
        return -1
    t = text.strip().upper().replace("♯", "#").replace("♭", "B")
    m = re.fullmatch(r"(1[0-2]|0?[1-9])\s*([AB])", t)
    if m:
        return from_camelot(int(m.group(1)), m.group(2))
    m = re.fullmatch(r"(1[0-2]|0?[1-9])\s*([MD])", t)   # Open Key: 1d = C major, 1m = Am
    if m:
        n = int(m.group(1))
        cam = (n + 6) % 12 + 1
        return from_camelot(cam, "B" if m.group(2) == "D" else "A")
    m = re.fullmatch(r"([A-G])([#B]?)\s*(MAJOR|MAJ|MINOR|MIN|M)?", t)
    if not m:
        return -1
    pc = _NOTE_ALIASES.get(m.group(1) + m.group(2))
    if pc is None:
        return -1
    mode = m.group(3) or ""
    # A bare 'M' after upper-casing is ambiguous; tags almost always mean minor ('Am').
    minor = mode in ("MINOR", "MIN", "M")
    return pc + 12 if minor else pc


def compatibility(a: int, b: int) -> float:
    """Harmonic mixing score in [0, 1] for going from key a to key b."""
    ca, cb = camelot(a), camelot(b)
    if ca is None or cb is None:
        return 0.5
    na, la = ca
    nb, lb = cb
    step = (nb - na) % 12            # clockwise steps on the wheel
    if la == lb:
        if step == 0:
            return 1.0
        if step in (1, 11):
            return 0.9               # adjacent: classic smooth mix
        if step in (2, 10):
            return 0.55              # +2 'energy boost'
        if step == 7:
            return 0.5               # +1 semitone energy boost
        if step == 5:
            return 0.35
        return 0.1
    if step == 0:
        return 0.85                  # relative major/minor
    if (la == "A" and step == 1) or (la == "B" and step == 11):
        return 0.6                   # diagonal mix
    if (la == "A" and step == 3) or (la == "B" and step == 9):
        return 0.4                   # parallel major/minor (same tonic)
    return 0.1


# Precomputed 25x25 matrix (index key+1 so unknown=-1 maps to row 0) for
# vectorised scoring over the whole library.
COMPAT = np.zeros((25, 25), dtype=np.float32)
for _a in range(-1, 24):
    for _b in range(-1, 24):
        COMPAT[_a + 1, _b + 1] = compatibility(_a, _b)
