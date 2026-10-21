"""Mix compatibility scoring, next-track suggestions and setlist generation.

All scoring is vectorised over the whole library so a suggestion list or a
40-track setlist from a 50k-track library is computed in well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .keys import COMPAT
from .library import Library

ENERGY_CURVES = ("steady", "build", "peak", "wave", "cooldown")


@dataclass
class MixSettings:
    max_pitch_pct: float = 6.0        # max tempo change (%) you are willing to ride
    allow_half_double: bool = True    # 87 <-> 174 style transitions
    w_tempo: float = 0.30
    w_key: float = 0.25
    w_sound: float = 0.25
    w_genre: float = 0.10
    w_energy: float = 0.10
    strict_key: bool = False          # only harmonically compatible keys (score >= 0.5)
    strict_tempo: bool = True         # exclude tracks outside the pitch range
    avoid_same_artist: bool = True
    energy_step: float = 0.0          # preferred energy change per transition (-2..+2)
    genre_lock: bool = False          # stay within the seed's AI parent genre

    def weights(self) -> np.ndarray:
        w = np.array([self.w_tempo, self.w_key, self.w_sound, self.w_genre, self.w_energy], np.float64)
        s = w.sum()
        return w / s if s > 0 else np.full(5, 0.2)


@dataclass
class Scored:
    total: np.ndarray
    tempo: np.ndarray
    key: np.ndarray
    sound: np.ndarray
    genre: np.ndarray
    energy: np.ndarray
    pitch_pct: np.ndarray           # signed tempo change needed (%)
    valid: np.ndarray


def tempo_match(lib: Library, a: int, s: MixSettings, ref_bpm: float | None = None):
    """Return (score 0..1, signed pitch % to match, valid mask)."""
    ref = ref_bpm if ref_bpm else lib.bpm[a]
    b = lib.bpm
    if ref <= 0:
        z = np.zeros(lib.n)
        return np.full(lib.n, 0.5), z, np.ones(lib.n, bool)
    mults = [1.0, 2.0, 0.5] if s.allow_half_double else [1.0]
    best_pct = np.full(lib.n, np.inf)
    best_pen = np.ones(lib.n)
    for m in mults:
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = (ref / (b * m) - 1.0) * 100.0
        pct = np.where(b > 0, pct, np.inf)
        pen = 1.0 if m == 1.0 else 0.85
        better = np.abs(pct) < np.abs(best_pct)
        best_pct = np.where(better, pct, best_pct)
        best_pen = np.where(better, pen, best_pen)
    lim = max(s.max_pitch_pct, 0.1)
    score = np.clip(1.0 - (np.abs(best_pct) / lim) ** 2, 0.0, 1.0) * best_pen
    valid = np.abs(best_pct) <= lim
    score = np.where(np.isfinite(best_pct), score, 0.0)
    return score, best_pct, valid


def score_from(lib: Library, a: int, s: MixSettings, target_energy: float | None = None,
               ref_bpm: float | None = None) -> Scored:
    tempo, pct, valid = tempo_match(lib, a, s, ref_bpm)
    key = COMPAT[lib.key[a] + 1, lib.key + 1].astype(np.float64)
    sound = lib.sound_score(a)
    genre = lib.style_score(a)
    te = lib.energy[a] + s.energy_step if target_energy is None else target_energy
    energy = np.clip(1.0 - np.abs(lib.energy - te) / 5.0, 0.0, 1.0)
    w = s.weights()
    total = w[0] * tempo + w[1] * key + w[2] * sound + w[3] * genre + w[4] * energy

    if s.strict_tempo:
        valid = valid.copy()
    else:
        valid = np.ones(lib.n, bool)
    if s.strict_key:
        valid &= key >= 0.5
    if s.genre_lock and lib.parent_code[a] >= 0:
        valid &= (lib.parent_code == lib.parent_code[a]) | (lib.parent_code < 0)
    if s.avoid_same_artist:
        total = np.where(lib.artist_code == lib.artist_code[a], total * 0.85, total)
    # never suggest the track itself or another copy of it
    dup = lib.dedup_code == lib.dedup_code[a]
    dup[a] = True
    valid &= ~dup
    return Scored(total, tempo, key, sound, genre, energy, pct, valid)


@dataclass
class Suggestion:
    idx: int
    score: float
    tempo: float
    key: float
    sound: float
    genre: float
    energy: float
    pitch_pct: float


def suggest_next(lib: Library, a: int, s: MixSettings, n: int = 50,
                 exclude: set[int] | None = None) -> list[Suggestion]:
    if lib.n == 0:
        return []
    sc = score_from(lib, a, s)
    total = np.where(sc.valid, sc.total, -np.inf)
    banned = np.zeros(lib.n, bool)
    banned[a] = True
    if exclude:
        ex = [int(i) for i in exclude if 0 <= int(i) < lib.n]
        banned[ex] = True
    total[banned] = -np.inf
    if not np.isfinite(total).any():
        # nothing inside the hard limits: fall back to the best overall matches
        total = np.where(banned, -np.inf, sc.total)
    k = min(n, lib.n)
    idx = np.argpartition(-total, k - 1)[:k]
    idx = idx[np.argsort(-total[idx])]
    return [Suggestion(int(i), float(total[i]), float(sc.tempo[i]), float(sc.key[i]), float(sc.sound[i]),
                       float(sc.genre[i]), float(sc.energy[i]), float(sc.pitch_pct[i]))
            for i in idx if np.isfinite(total[i])]


def energy_target(curve: str, start: float, frac: float) -> float:
    """Target energy (1-10) at position frac (0..1) of the set."""
    if curve == "build":
        t = start + (min(10.0, start + 4) - start) * frac
    elif curve == "peak":
        top = min(10.0, start + 4)
        t = start + (top - start) * (frac / 0.65) if frac <= 0.65 else top - (top - start) * ((frac - 0.65) / 0.35)
    elif curve == "wave":
        t = start + 2.0 * np.sin(frac * 2 * np.pi * 2)
    elif curve == "cooldown":
        t = start - (start - max(1.0, start - 4)) * frac
    else:
        t = start
    return float(np.clip(t, 1.0, 10.0))


@dataclass
class SetlistOptions:
    length: int = 20
    end_bpm: float | None = None      # ramp tempo from the seed BPM towards this
    energy_curve: str = "steady"
    variety: float = 0.0              # 0 = deterministic best, 1 = lots of randomness
    beam_width: int = 8
    seed: int | None = None
    mix: MixSettings = field(default_factory=MixSettings)


def build_setlist(lib: Library, seed_idx: int, opt: SetlistOptions, exclude=None) -> list[int]:
    """Beam search for a sequence that maximises transition quality while
    following the requested tempo ramp and energy curve. Tracks in exclude
    (e.g. already played tonight) are never picked after the seed."""
    if lib.n == 0:
        return []
    banned = np.zeros(lib.n, bool)
    if exclude:
        ex = [int(i) for i in exclude if 0 <= int(i) < lib.n]
        if ex:
            banned[ex] = True
            banned |= np.isin(lib.dedup_code, lib.dedup_code[ex])
        banned[seed_idx] = False
    length = max(1, min(opt.length, lib.n))
    rng = np.random.default_rng(opt.seed)
    s = opt.mix
    seed_bpm = float(lib.bpm[seed_idx])
    seed_energy = float(lib.energy[seed_idx])
    beams: list[tuple[float, list[int]]] = [(0.0, [seed_idx])]
    width = max(1, opt.beam_width)
    per_beam = max(3, width)

    for step in range(1, length):
        frac = step / max(length - 1, 1)
        te = energy_target(opt.energy_curve, seed_energy, frac)
        target_bpm = None
        if opt.end_bpm and seed_bpm > 0:
            target_bpm = seed_bpm + (opt.end_bpm - seed_bpm) * frac
        cands: list[tuple[float, list[int]]] = []
        for cum, seq in beams:
            last = seq[-1]
            sc = score_from(lib, last, s, target_energy=te)
            total = sc.total.copy()
            if target_bpm:
                # pull towards the ramp: tempo distance from the planned BPM
                t_score, _, _ = tempo_match(lib, last, s, ref_bpm=target_bpm)
                total = 0.7 * total + 0.3 * t_score
            if opt.variety > 0:
                total = total + rng.gumbel(size=lib.n) * 0.08 * opt.variety
            ok = sc.valid & ~banned
            used = np.isin(lib.dedup_code, lib.dedup_code[seq])
            used[seq] = True
            ok &= ~used
            if not ok.any():
                continue      # dead end: nothing mixable from here
            total = np.where(ok, total, -np.inf)
            k = min(per_beam, int(ok.sum()))
            top = np.argpartition(-total, k - 1)[:k]
            for i in top:
                if np.isfinite(total[i]):
                    cands.append((cum + float(total[i]), seq + [int(i)]))
        if not cands:
            break
        cands.sort(key=lambda c: -c[0])
        # keep beams diverse: at most 2 beams sharing the same last track
        seen: dict[int, int] = {}
        beams = []
        for c in cands:
            lt = c[1][-1]
            if seen.get(lt, 0) >= 2:
                continue
            seen[lt] = seen.get(lt, 0) + 1
            beams.append(c)
            if len(beams) >= width:
                break
    return max(beams, key=lambda b: b[0])[1]


def transition_report(lib: Library, seq: list[int], s: MixSettings) -> list[dict]:
    """Per-transition details for display (pitch change, key relation, score)."""
    out = []
    for a, b in zip(seq, seq[1:]):
        sc = score_from(lib, a, s)
        out.append({"from": a, "to": b, "score": float(sc.total[b]), "pitch_pct": float(sc.pitch_pct[b]),
                    "key": float(sc.key[b]), "sound": float(sc.sound[b])})
    return out
