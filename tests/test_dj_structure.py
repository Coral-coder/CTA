"""Auto-mix musicianship: structure detection and DJ-style transition plans."""
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import structured_track

pytest.importorskip("pylibrb")

from cta.dj.automix import AutoMix, AutoMixSettings  # noqa: E402
from cta.dj.beatgrid import fit_grid  # noqa: E402
from cta.dj.engine import SR, Engine, TrackInfo  # noqa: E402
from cta.dj.loader import waveform_bands  # noqa: E402
from cta.dj.structure import analyse, plan_transition  # noqa: E402

FULL = (("intro", 16), ("drop", 32), ("breakdown", 16), ("drop", 32), ("outro", 16))
NO_OUTRO = (("intro", 16), ("drop", 32), ("breakdown", 8), ("drop", 16))
SHORT = (("intro", 8), ("drop", 32), ("outro", 8))


def _track(name, bpm, layout, key=21, chord=(57, 60, 64)):
    y = structured_track(bpm, layout, chord=chord)
    a = np.stack([y, y])
    g = fit_grid(y, SR, bpm)
    return SimpleNamespace(audio=a, grid=g, peaks=None, bands=waveform_bands(a), first_beat_t=g.first_beat,
                           info=TrackInfo(title=name, bpm=bpm, loudness=-10, key=key))


def _struct(layout):
    t = _track("x", 126.0, layout)
    return analyse(t.bands, t.grid)


def test_sections_detected():
    st = _struct(FULL)
    kinds = [(s.kind, s.start_bar) for s in st.sections]
    assert kinds[:4] == [("intro", 0), ("drop", 16), ("breakdown", 48), ("drop", 64)]
    assert st.sections[-1].kind == "outro" and st.sections[-1].start_bar == 96
    assert st.first_drop_bar == 16 and st.last_drop_end == 96


def test_plans_like_a_dj():
    full, no_outro, nxt = _struct(FULL), _struct(NO_OUTRO), _struct(SHORT)
    p = plan_transition(full, nxt, max_bars=16)
    assert p.out_start_bar == 96                      # after the last drop, never inside it
    assert p.in_start_bar + p.bars == 8               # incoming drop lands at the end of the blend
    assert p.swap_bar == p.bars                       # bass swap on the incoming drop
    p = plan_transition(no_outro, nxt, max_bars=16)
    assert p.style == "loop_blend" and p.loop is not None   # hold the groove with a loop
    p = plan_transition(full, nxt, max_bars=16, harmonic=0.1)
    assert p.style == "echo_out" and p.bars <= 4
    p = plan_transition(full, nxt, max_bars=16, out_marker_bar=40, in_marker_bar=0)
    assert p.out_start_bar == 40 and p.in_start_bar == 0     # markers always win


def _run_mix(tracks, order, seconds=600):
    eng = Engine()
    am = AutoMix(eng, lambda i: tracks[i], AutoMixSettings(mix_bars=16), synchronous_loads=True)
    am.set_queue(order)
    am.start()
    events = []
    t = 0.0
    peak = 0.0
    seen_start = False
    while t < seconds:
        m, _ = eng.render(1024)
        t += 1024 / SR
        peak = max(peak, float(np.abs(m).max()))
        tr = am.tr
        if tr is not None and tr.started and not seen_start:
            seen_start = True
            events.append(("start", tr.style, eng.decks[tr.out_deck].grid.beat_at(eng.decks[tr.out_deck].time) / 4,
                           eng.decks[tr.in_deck].grid.beat_at(eng.decks[tr.in_deck].time) / 4))
        if tr is None:
            seen_start = False
        if am.history == order and not any(d.playing for d in eng.decks.values()):
            break
    return am, events, peak, t


def test_automix_end_to_end_structure_aware():
    tr = {"a": _track("a", 126.0, FULL), "b": _track("b", 126.0, NO_OUTRO, key=16, chord=(52, 55, 59)),
          "c": _track("c", 127.0, SHORT)}
    am, events, peak, t = _run_mix(tr, ["a", "b", "c"])
    assert am.history == ["a", "b", "c"]
    assert peak <= 0.99                                           # limiter: no clipping
    style1, out_bar1, in_bar1 = events[0][1], events[0][2], events[0][3]
    assert style1 == "blend" and abs(out_bar1 - 96) < 0.5          # mixed out right after a's last drop
    assert abs(in_bar1 - 4) < 0.5            # b enters 12 bars before its drop (bar 16): drop lands on the swap
    style2 = events[1][1]
    assert style2 == "loop_blend"                                 # b has no outro -> loop to keep the beat
    assert t < 600                                                # finished (no stuck loop)


def test_echo_out_on_key_clash():
    tr = {"a": _track("a", 126.0, SHORT, key=21), "b": _track("b", 126.0, SHORT, key=1, chord=(61, 65, 68))}
    am, events, _peak, _t = _run_mix(tr, ["a", "b"], seconds=300)
    assert events[0][1] == "echo_out"
    assert am.history == ["a", "b"]
