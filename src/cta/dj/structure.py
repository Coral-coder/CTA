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
    in_loop: tuple[float, float] | None = None   # incoming loop (bars) to stretch a short intro
    in_release: float | None = None              # bars into the blend when the incoming loop is let go
    exact: bool = False                          # length set by the DJ (markers / fixed length)


# How each style family is mixed (see cta.styles): the shortest blend that sounds
# right, and where the bass swap lands - "end": the incoming drop lands on the last
# bar of the blend and the basses swap right there (trance, drum & bass, dubstep,
# hard dance: the drop is the moment), "half": a classic EQ blend with the swap on
# a phrase in the middle (house, techno, disco). The DJ's blend length is the
# ceiling; a family's floor raises a too-short setting.
FAMILY_RULES = {
    "Trance": (16, "end"), "Drum & Bass": (16, "end"), "Bass": (8, "end"), "Hard Dance": (8, "end"),
    "House": (8, "half"), "Techno": (8, "half"), "Disco": (8, "half"), "Downtempo": (8, "half"),
    "Breaks": (8, "half"), "Dance / Pop": (4, "half"),
}

AUDIBLE = 0.08          # bar energy (0..1 of the track's loud parts) that is more than silence
DEAD = 0.2              # combined level below which a blend bar sounds empty
PHRASES = (4, 8, 16, 32, 64, 128)


def _phrase_floor(x: float, lo: int = 4) -> int:
    """Largest phrase length (4, 8, 16, ...) not longer than x (at least lo)."""
    best = lo
    for p in PHRASES:
        if p <= x:
            best = p
    return best


def _energy(st: Structure, bar: float) -> float:
    b = int(np.floor(bar))
    return float(st.bar_energy[b]) if 0 <= b < len(st.bar_energy) else 0.0


def _kick(st: Structure, bar: float) -> bool:
    b = int(np.floor(bar))
    return bool(st.bar_kick[b]) if 0 <= b < len(st.bar_kick) else False


def _audible_span(st: Structure) -> tuple[int, int]:
    """First audible bar and end (exclusive) of the last audible bar."""
    a = np.nonzero(st.bar_energy > AUDIBLE)[0]
    return (int(a[0]), int(a[-1]) + 1) if a.size else (0, st.bars)


def blend_profile(out_s: Structure, in_s: Structure, start: float, in_start: float, bars: int,
                  swap: float, loop: tuple[float, float] | None = None,
                  in_loop: tuple[float, float] | None = None, in_release: float | None = None):
    """Per bar of the blend: (combined level, any kick) as the listener hears it."""
    out = []
    for k in range(int(bars)):
        ob = start + k
        if loop is not None and ob >= loop[1]:
            ln = loop[1] - loop[0]
            ob = loop[0] + (ob - loop[0]) % ln
        ib = in_start + k
        if in_loop is not None:
            ln = in_loop[1] - in_loop[0]
            rel = in_release if in_release is not None else bars
            if k < rel:
                ib = in_loop[0] + k % ln
            else:
                ib = in_loop[0] + (rel % ln) + (k - rel)
        fade_out = 1.0 if k < swap else max(0.0, 1.0 - (k - swap) / max(bars - swap, 1))
        fade_in = min(1.0, (k + 1) / max(1.0, min(4.0, bars * 0.25)))
        e = max(_energy(out_s, ob) * fade_out, _energy(in_s, ib) * fade_in)
        out.append((e, _kick(out_s, ob) and fade_out > 0.3 or _kick(in_s, ib)))
    return out


def dead_bars(profile) -> int:
    """Bars that would sound empty: very quiet, or no beat at all for more than a bar."""
    dead = sum(1 for e, _k in profile if e < DEAD)
    run = longest = 0
    for _e, k in profile:
        run = 0 if k else run + 1
        longest = max(longest, run)
    return dead + max(0, longest - 2)


def plan_transition(out_s: Structure, in_s: Structure, max_bars: int = 16, min_bars: int = 4,
                    harmonic: float = 1.0, tempo_gap_pct: float = 0.0,
                    out_marker_bar: float | None = None, in_marker_bar: float | None = None,
                    genre_short: bool = False, learned: dict | None = None,
                    out_end_bar: float | None = None, auto: bool = True, family: str | None = None) -> Plan:
    """Decide how and where to go from the outgoing to the incoming track.

    max_bars: the longest blend (auto) or the blend length (auto=False).
    harmonic: key compatibility 0..1; tempo_gap_pct: tempo change needed.
    out_end_bar: the DJ's outro-end marker: the blend lasts exactly from the
    mix-out marker to there. Bars are counted from each track's grid downbeat.
    """
    max_bars = max(1, int(max_bars))
    if genre_short:
        max_bars = min(max_bars, 8)
    rules = FAMILY_RULES.get(family or "")
    swap_end = False
    if rules and auto and not genre_short:
        fmin, where = rules
        min_bars = max(min_bars, fmin)
        max_bars = max(max_bars, fmin)
        swap_end = where == "end"
    last_beat_end = out_s.last_beat_bar + 1
    drop_end = out_s.last_drop_end if out_s.last_drop_end is not None else last_beat_end
    drop_in = in_s.first_drop_bar
    in_first = float(in_s.first_beat_bar - (in_s.first_beat_bar % 4))

    def in_start_for(bars: int) -> float:
        """Start so the incoming drop lands at the end of the blend."""
        if in_marker_bar is not None:
            return float(in_marker_bar)
        if harmonic < 0.7 and drop_in is not None and drop_in - bars < in_s.first_beat_bar - 1:
            return in_first                  # never start in an out-of-key beatless fill
        if drop_in is not None and drop_in - bars >= 0 and drop_in - bars >= _audible_span(in_s)[0] - 4:
            return float(drop_in - bars)
        return in_first

    def out_loop_for(start: float, bars: int):
        """Loop the outgoing's last groove when it runs out of beats during the blend."""
        if start + bars <= last_beat_end + 2:          # the outgoing is fading out by then anyway
            return None
        groove = last_beat_end - max(out_s.first_beat_bar, 0)
        if groove < 4:
            return None
        le = last_beat_end + (1 if (last_beat_end + 1) % 4 == 0 else 0)   # on the phrase grid
        le = le - le % 4 if le % 4 else le
        le = max(le, 4)
        lb = 8 if le - start >= 8 and groove >= 8 else 4
        return (float(le - lb), float(le))

    # ---- the DJ's own outro start + end: exactly that long
    if out_marker_bar is not None and out_end_bar is not None and out_end_bar > out_marker_bar:
        bars = int(max(1, round(out_end_bar - out_marker_bar)))
        start = float(out_marker_bar)
        in_start = in_start_for(bars)
        loop = out_loop_for(start, bars)
        lands = drop_in is not None and abs((drop_in - in_start) - bars) < 1e-6
        swap = float(bars) if lands else float(max(1, bars // 2))
        return Plan("loop_blend" if loop else "blend", start, in_start, bars, swap_bar=swap, loop=loop,
                    reason=f"your outro markers ({bars} bars)", exact=True)

    # ---- learned from the DJ's own transitions (markers still win)
    if learned and out_marker_bar is None and in_marker_bar is None and harmonic >= 0.3 and tempo_gap_pct <= 10:
        bars = int(min(max_bars, max(1, learned["bars"])))
        drop_in2 = drop_in if drop_in is not None else in_s.first_beat_bar
        start = float(max(out_s.first_beat_bar + 4, np.round((drop_end + learned["out_offset"]) / 4) * 4))
        in_start = float(max(0, np.round((drop_in2 - learned["in_lead"]) / 4) * 4))
        loop = None
        if start + bars > last_beat_end:                # would run out of beats: hold the groove
            lb = 8 if last_beat_end - start >= 8 else 4
            loop = (float(last_beat_end - lb), float(last_beat_end))
        swap = float(min(bars, max(1.0, drop_in2 - in_start))) if drop_in2 > in_start else float(max(1, bars // 2))
        return Plan("loop_blend" if loop else "blend", start, in_start, bars, swap_bar=swap, loop=loop,
                    reason="your style (learned from your mixes)")

    # ---- real clash -> short echo-out style transition (only when a blend
    #      would sound wrong: keys far apart or a big tempo jump)
    if harmonic < 0.3 or tempo_gap_pct > 10:
        bars = min(4, max_bars)
        if out_marker_bar is not None:
            start = out_marker_bar
        else:
            start = max(float(drop_end - bars), 0.0)
        in_start = float(in_marker_bar) if in_marker_bar is not None else in_first
        if in_marker_bar is None and drop_in is not None:
            in_start = float(max(drop_in - bars, 0))      # drop the next track on the one
        return Plan("echo_out", start, in_start, bars, swap_bar=float(bars),
                    reason="keys/tempo clash: echo out and drop the next track on the one")

    fill_cap = max(max_bars, 8) if auto else max_bars
    in_aud0, _in_aud1 = _audible_span(in_s)
    in_beat4 = int(round(in_s.first_beat_bar / 4.0) * 4)   # the beat starts on a phrase line
    in_fill = in_beat4 - in_aud0                      # audible intro bars before the beat starts
    _out_aud0, out_aud1 = _audible_span(out_s)
    out_fill = out_aud1 - last_beat_end               # audible bars after the beat stops
    out_has_groove = last_beat_end - out_s.first_beat_bar >= 4
    in_groove = (drop_in if drop_in is not None and drop_in > in_first else in_s.last_beat_bar + 1) - in_first

    # ---- the next track opens with a long beatless fill: play it over the
    #      outgoing groove (looped if the outro is short), its beat lands as
    #      the outgoing leaves
    if in_marker_bar is None and in_fill >= 8 and harmonic >= 0.7 and out_has_groove and not genre_short:
        bars = _phrase_floor(min(in_fill, fill_cap))
        in_start = float(max(0, in_beat4 - bars))
        start = float(out_marker_bar) if out_marker_bar is not None else float(min(drop_end, last_beat_end))
        loop = out_loop_for(start, bars)
        if loop is not None and start > loop[0]:
            start = loop[0]                         # beats would stop inside the blend: loop from the last groove
        return Plan("loop_blend" if loop else "blend", start, in_start, bars, swap_bar=float(bars), loop=loop,
                    reason="the next track's intro fill plays over this groove"
                    + (" (looped outro)" if loop else ""), exact=True)

    # ---- the outgoing ends in a long beatless fill: bring the next beat in
    #      under it, looping the next intro if it is too short
    if out_marker_bar is None and out_fill >= 8 and in_groove >= 4 and in_fill < 8 and not genre_short:
        bars = _phrase_floor(min(out_fill, fill_cap if harmonic >= 0.7 else 8))
        start = float(last_beat_end)
        in_start = float(in_marker_bar) if in_marker_bar is not None else in_first
        in_loop = release = None
        if in_groove < bars:
            lb = 8 if in_groove >= 8 else 4
            in_loop = (in_start, in_start + lb)
            release = float(np.ceil((bars - in_groove) / lb) * lb)
            bars = int(release + in_groove)                # the incoming drop lands at the end
        return Plan("blend", start, in_start, bars, swap_bar=1.0, reason="the outro fill plays over the next "
                    "track's beat" + (" (looped intro)" if in_loop else ""), in_loop=in_loop,
                    in_release=release, exact=True)

    # ---- blend length: let the incoming intro decide (drop lands at the end)
    intro_len = (drop_in - in_first) if drop_in is not None and drop_in > in_first else None
    if in_marker_bar is not None and drop_in is not None and drop_in > in_marker_bar:
        intro_len = drop_in - in_marker_bar
    if not auto:
        bars = max_bars
    elif intro_len is not None and intro_len >= min_bars:
        bars = int(min(max_bars, intro_len))
    else:
        bars = max_bars
    if auto:
        bars = max(min_bars if max_bars >= min_bars else max_bars, bars)
    in_start = in_start_for(bars) if (intro_len is None or intro_len >= bars or not auto) else in_first

    # ---- outgoing: start the blend in its outro, after the last drop
    loop = None
    if out_marker_bar is not None:
        start = float(out_marker_bar)
        reason = "your mix-out marker"
        loop = out_loop_for(start, bars)
    else:
        outro_beats = max(0, last_beat_end - drop_end)          # beat bars after the last drop
        if outro_beats >= bars or not auto:
            start = float(drop_end)
            loop = out_loop_for(start, bars)
            reason = "blend over the outgoing outro"
        elif outro_beats >= 8 and outro_beats // 4 * 4 >= min_bars:
            # outro a bit shorter than the blend: fit the blend to it, keep the
            # incoming drop landing at the end
            bars = int(outro_beats // 4 * 4)
            in_start = in_start_for(bars)
            start = float(drop_end)
            reason = "blend fitted to the outgoing outro"
        elif outro_beats >= 4:
            start = float(drop_end)
            lb = 4 if outro_beats < 8 else 8
            loop = (float(last_beat_end - lb), float(last_beat_end))
            reason = "short outro: loop the last bars to keep the beat"
        else:
            # no beat outro: loop the end of the last drop and blend over it
            lb = 8 if drop_end - out_s.first_beat_bar >= 16 else 4
            start = float(max(drop_end - lb, out_s.first_beat_bar))
            loop = (start, start + lb)
            reason = "no outro: loop the final groove and blend over it"

    if loop is not None and start >= loop[1] - 1e-6:
        start = loop[0]                     # the beats end where the blend would begin: blend from the loop

    def swap_for(b, ist):
        if drop_in is not None and 1.0 <= drop_in - ist <= b + 1e-6:
            return float(drop_in - ist)     # the incoming drop inside the blend: the basses swap right there
        if swap_end:
            return float(b)                 # trance / d&b / bass: at the very end, on the drop
        return float(max(1, b // 2))

    # ---- never hang on dead air: find a length (and a start a few bars either
    #      side) where every bar of the blend has music and a beat
    if auto and out_marker_bar is None and loop is None:
        def ok(st, b, ist):
            return dead_bars(blend_profile(out_s, in_s, st, ist, b, swap_for(b, ist))) == 0

        if not ok(start, bars, in_start):
            found = None
            lengths = [p for p in PHRASES if min_bars <= p <= bars][::-1] or [bars]
            for nb in lengths:
                ist = in_start_for(nb)
                cands = [start + 4 * k for k in range(0, 9)] + [start - 4 * k for k in range(1, 3)]
                for st in cands:
                    if st < out_s.first_beat_bar or (st != start and st + nb > last_beat_end):
                        continue
                    if ok(st, nb, ist):
                        found = (st, nb, ist)
                        break
                if found:
                    break
            if found and (found[0], found[1]) != (start, bars):
                start, bars, in_start = found
                reason += f" ({bars} bars, placed to avoid dead air)"
    style = "loop_blend" if loop else "blend"
    if rules and auto:
        reason += f" · {family} style"
    return Plan(style, start, in_start, bars, swap_bar=swap_for(bars, in_start), loop=loop, reason=reason,
                exact=not auto)
