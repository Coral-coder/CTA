"""Track structure: phrases and sections (intro, groove, build, drop,
breakdown, outro), the information a DJ uses to decide *where* to mix.

Everything is derived from the waveform band envelopes (bass / mid / high
per 256 samples) and the beat grid, so it costs a few milliseconds per track.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .beatgrid import BeatGrid

PHRASE_BARS = 8


@dataclass
class Section:
    start_bar: int
    end_bar: int                 # exclusive
    kind: str                    # intro | groove | build | drop | breakdown | outro
    energy: float                # 0..1 relative to the track
    kick: float                  # 0..1 share of bars with a kick

    @property
    def bars(self) -> int:
        return self.end_bar - self.start_bar


@dataclass
class Structure:
    bars: int
    bar_energy: np.ndarray       # (bars,) 0..1
    bar_kick: np.ndarray         # (bars,) bool: has a steady kick
    bar_bass: np.ndarray         # (bars,) 0..1 bass energy
    sections: list[Section] = field(default_factory=list)
    first_beat_bar: int = 0      # first bar with beats
    last_beat_bar: int = 0       # last bar with beats (inclusive)
    first_drop_bar: int | None = None
    last_drop_end: int | None = None   # bar where the final full-energy section ends

    def section_at(self, bar: float) -> Section | None:
        for s in self.sections:
            if s.start_bar <= bar < s.end_bar:
                return s
        return None

    def describe(self) -> str:
        return " | ".join(f"{s.kind} {s.start_bar}-{s.end_bar}" for s in self.sections)


def _bar_matrix(bands: np.ndarray, grid: BeatGrid, hop: int, sr: int):
    fps = sr / hop
    beat = grid.beat_len
    n_bars = max(1, int((grid.duration - grid.first_beat) / (4 * beat)))
    feats = np.zeros((n_bars, 4))            # low, mid, high, kick-presence
    w = max(1, int(0.035 * fps))
    for b in range(n_bars):
        s = int((grid.first_beat + b * 4 * beat) * fps)
        e = int((grid.first_beat + (b + 1) * 4 * beat) * fps)
        s, e = max(0, s), min(bands.shape[0], max(e, s + 1))
        if s >= bands.shape[0]:
            break
        seg = bands[s:e]
        feats[b, :3] = np.sqrt((seg ** 2).mean(axis=0))
        # kick presence: low-band peak on each beat vs between beats
        hits = []
        for k in range(4):
            c = int((grid.first_beat + (b * 4 + k) * beat) * fps)
            m = int((grid.first_beat + (b * 4 + k + 0.5) * beat) * fps)
            if 0 <= c < bands.shape[0] and 0 <= m < bands.shape[0]:
                on = bands[max(0, c - w):c + w + 1, 0].max()
                off = bands[max(0, m - w):m + w + 1, 0].mean()
                hits.append(on - off)
        feats[b, 3] = float(np.mean(hits)) if hits else 0.0
    return feats


def analyse(bands: np.ndarray | None, grid: BeatGrid | None, hop: int = 256, sr: int = 44100,
            phrase_origin: int | None = None, phrase_bars: int = PHRASE_BARS) -> Structure:
    if bands is None or grid is None or grid.bpm <= 0:
        return Structure(0, np.zeros(0), np.zeros(0, bool), np.zeros(0))
    f = _bar_matrix(bands, grid, hop, sr)
    n = f.shape[0]

    def norm(x):
        hi = np.percentile(x, 95) if x.size else 1.0
        return np.clip(x / max(hi, 1e-9), 0, 1.2)

    low, mid, high = norm(f[:, 0]), norm(f[:, 1]), norm(f[:, 2])
    kick_strength = norm(np.maximum(f[:, 3], 0))
    energy = np.clip(0.45 * low + 0.35 * mid + 0.2 * high, 0, 1.2)
    # a bar "has a kick" when the beat-vs-offbeat bass contrast is clear
    has_kick = kick_strength > 0.3
    # smooth single-bar glitches (fills) out of the kick mask
    if n >= 3:
        k = has_kick.astype(float)
        k = np.convolve(np.pad(k, 1, mode="edge"), np.ones(3) / 3, "valid")
        has_kick = k >= 0.5

    beats = np.nonzero(has_kick)[0]
    first_beat_bar = int(beats[0]) if beats.size else 0
    last_beat_bar = int(beats[-1]) if beats.size else n - 1

    # phrase blocks aligned to the first beat bar (phrases count from there)
    origin = (phrase_origin if phrase_origin is not None else first_beat_bar) % phrase_bars
    edges = [0] + ([origin] if origin else []) + list(range(origin + phrase_bars, n, phrase_bars))
    edges = sorted(set(e for e in edges if e < n)) + [n]

    hi_e = np.percentile(energy[has_kick], 60) if has_kick.any() else np.percentile(energy, 60)
    blocks = []
    for s, e in zip(edges[:-1], edges[1:]):
        if e <= s:
            continue
        en = float(energy[s:e].mean())
        kk = float(has_kick[s:e].mean())
        bass = float(low[s:e].mean())
        blocks.append([s, e, en, kk, bass])

    # classify blocks
    labels = []
    for _s, _e, en, kk, bass in blocks:
        if kk < 0.4:
            labels.append("breakdown")
        elif en >= hi_e * 0.92 and bass > 0.45:
            labels.append("drop")
        else:
            labels.append("groove")
    # intro / outro: groove/breakdown before the first drop and after the last
    drops = [i for i, lab in enumerate(labels) if lab == "drop"]
    if drops:
        for i in range(drops[0]):
            labels[i] = "intro"
        for i in range(drops[-1] + 1, len(labels)):
            labels[i] = "outro"
        # a groove right before a drop that rises in energy is a build
        for i in drops:
            j = i - 1
            if j >= 0 and labels[j] in ("groove", "breakdown") and blocks[j][2] < blocks[i][2]:
                if labels[j] == "groove":
                    labels[j] = "build"
    else:
        # no clear drop: the loudest beat phrase plays that role
        if blocks:
            j = int(np.argmax([bl[2] * (bl[3] > 0.4) for bl in blocks]))
            labels[j] = "drop"
            for i in range(j):
                labels[i] = "intro"
            for i in range(j + 1, len(labels)):
                labels[i] = "outro"

    # merge neighbours with the same label
    sections: list[Section] = []
    for (s, e, en, kk, _bass), lab in zip(blocks, labels):
        if sections and sections[-1].kind == lab:
            p = sections[-1]
            w0, w1 = p.bars, e - s
            p.energy = (p.energy * w0 + en * w1) / (w0 + w1)
            p.kick = (p.kick * w0 + kk * w1) / (w0 + w1)
            p.end_bar = e
        else:
            sections.append(Section(s, e, lab, en, kk))

    drop_secs = [s for s in sections if s.kind == "drop"]
    return Structure(
        bars=n, bar_energy=energy, bar_kick=has_kick, bar_bass=low, sections=sections,
        first_beat_bar=first_beat_bar, last_beat_bar=last_beat_bar,
        first_drop_bar=drop_secs[0].start_bar if drop_secs else None,
        last_drop_end=drop_secs[-1].end_bar if drop_secs else None,
    )


# --------------------------------------------------------------------------
# transition planning
# --------------------------------------------------------------------------

@dataclass
class Plan:
    style: str                  # blend | loop_blend | echo_out | cut
    out_start_bar: float        # outgoing bar where the incoming track starts
    in_start_bar: float         # incoming bar it starts from
    bars: int                   # blend length
    swap_bar: float             # bars into the blend where the basses swap (on a downbeat)
    loop: tuple[float, float] | None = None   # outgoing loop (start_bar, end_bar) to hold the groove
    reason: str = ""


def plan_transition(out_s: Structure, in_s: Structure, max_bars: int = 16, min_bars: int = 4,
                    harmonic: float = 1.0, tempo_gap_pct: float = 0.0,
                    out_marker_bar: float | None = None, in_marker_bar: float | None = None,
                    genre_short: bool = False, learned: dict | None = None) -> Plan:
    """Decide how and where to go from the outgoing to the incoming track.

    harmonic: key compatibility 0..1; tempo_gap_pct: tempo change needed.
    Bars are counted from each track's grid downbeat.
    """
    max_bars = max(1, int(max_bars))
    if genre_short:
        max_bars = min(max_bars, 8)

    # ---- learned from the DJ's own transitions (markers still win)
    if learned and out_marker_bar is None and in_marker_bar is None and harmonic >= 0.5 and tempo_gap_pct <= 8:
        bars = int(min(max_bars, max(1, learned["bars"])))
        drop_end = out_s.last_drop_end if out_s.last_drop_end is not None else out_s.last_beat_bar + 1
        drop_in = in_s.first_drop_bar if in_s.first_drop_bar is not None else in_s.first_beat_bar
        start = float(max(out_s.first_beat_bar + 4, np.round((drop_end + learned["out_offset"]) / 4) * 4))
        in_start = float(max(0, np.round((drop_in - learned["in_lead"]) / 4) * 4))
        loop = None
        last_beat_end = out_s.last_beat_bar + 1
        if start + bars > last_beat_end:                # would run out of beats: hold the groove
            lb = 8 if last_beat_end - start >= 8 else 4
            loop = (float(last_beat_end - lb), float(last_beat_end))
        swap = float(min(bars, max(1.0, drop_in - in_start))) if drop_in > in_start else float(max(1, bars // 2))
        return Plan("loop_blend" if loop else "blend", start, in_start, bars, swap_bar=swap, loop=loop,
                    reason="your style (learned from your mixes)")

    # ---- incoming: where to start, and how long until its first drop
    if in_marker_bar is not None:
        in_start = float(in_marker_bar)
    else:
        in_start = float(in_s.first_beat_bar - (in_s.first_beat_bar % 4))
    drop_in = in_s.first_drop_bar
    intro_len = (drop_in - in_start) if drop_in is not None and drop_in > in_start else None

    # ---- clash -> short echo-out style transition
    if harmonic < 0.5 or tempo_gap_pct > 8:
        bars = min(4, max_bars)
        if out_marker_bar is not None:
            start = out_marker_bar
        else:
            end = (out_s.last_drop_end if out_s.last_drop_end is not None else out_s.last_beat_bar + 1)
            start = max(float(end - bars), 0.0)
        # bring the incoming in right at its drop if it has one close by
        if in_marker_bar is None and drop_in is not None:
            in_start = float(max(drop_in - bars, 0))
        return Plan("echo_out", start, in_start, bars, swap_bar=float(bars),
                    reason="keys/tempo clash: echo out and drop the next track on the one")

    # ---- blend length: let the incoming intro decide (drop lands at the end)
    if intro_len is not None and intro_len >= min_bars:
        bars = int(min(max_bars, intro_len))
        if intro_len > max_bars and in_marker_bar is None:
            in_start = float(drop_in - bars)            # skip the part of a long intro
    else:
        bars = max_bars
    bars = max(min_bars if max_bars >= min_bars else max_bars, bars)

    # ---- outgoing: start the blend in its outro, after the last drop
    loop = None
    if out_marker_bar is not None:
        start = float(out_marker_bar)
        reason = "your mix-out marker"
    else:
        last_beat_end = out_s.last_beat_bar + 1
        drop_end = out_s.last_drop_end if out_s.last_drop_end is not None else last_beat_end
        outro_beats = max(0, last_beat_end - drop_end)          # beat bars after the last drop
        if outro_beats >= bars:
            # enough groove after the last drop: start blending right as it ends
            start = float(drop_end)
            reason = "blend over the outgoing outro"
        elif outro_beats >= 8:
            # outro a bit shorter than the blend: fit the blend to it, keep the
            # incoming drop landing at the end
            bars = int(outro_beats // 4 * 4)
            if in_marker_bar is None and drop_in is not None and drop_in - bars >= 0:
                in_start = float(drop_in - bars)
            start = float(drop_end)
            reason = "blend fitted to the outgoing outro"
        elif outro_beats >= 4:
            start = float(drop_end)
            # outro shorter than the blend: loop its last 4 bars to hold the beat
            lb = 4 if outro_beats < 8 else 8
            loop = (float(last_beat_end - lb), float(last_beat_end))
            reason = "short outro: loop the last bars to keep the beat"
        else:
            # no beat outro (track ends after the drop / fades): loop the end
            # of the last drop and blend over it
            lb = 8 if drop_end - (out_s.first_beat_bar) >= 16 else 4
            start = float(max(drop_end - lb, out_s.first_beat_bar))
            loop = (start, start + lb)
            reason = "no outro: loop the final groove and blend over it"
    style = "loop_blend" if loop else "blend"
    # basses swap on the incoming drop (end of the blend) or half way
    lands = drop_in is not None and abs((drop_in - in_start) - bars) < 1e-6
    swap = float(bars) if lands else float(max(1, bars // 2))
    return Plan(style, start, in_start, bars, swap_bar=swap, loop=loop, reason=reason)
