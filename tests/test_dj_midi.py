import pytest

pytest.importorskip("pylibrb")

import numpy as np  # noqa: E402

from cta.dj.beatgrid import BeatGrid  # noqa: E402
from cta.dj.engine import SR, Engine, TrackInfo  # noqa: E402
from cta.dj.midi import Controller, Decoder, Learner, eq_curve, rel7, rel14  # noqa: E402


def test_decoder_and_relative():
    d = Decoder({(0, 1)})
    assert d.feed([0xB0, 1, 0]) is None                  # MSB waits for LSB
    ev = d.feed([0xB0, 33, 5])
    assert ev.type == "cc14" and ev.num == 1 and ev.value == 5
    ev = d.feed([0x90, 11, 127])
    assert ev.type == "note" and ev.value == 127
    assert d.feed([0x90, 11, 0]).value == 0
    assert rel7(1) == 1 and rel7(127) == -1 and rel14(16383) == -1 and rel14(3) == 3
    assert eq_curve(0.0) == 0.0 and eq_curve(0.5) == 1.0 and eq_curve(1.0) == 2.0


def _engine():
    e = Engine()
    a = (np.random.default_rng(0).standard_normal((2, SR * 30)) * 0.1).astype(np.float32)
    e.load("A", a, BeatGrid(120.0, 0.0, 30.0), TrackInfo(loudness=-10))
    return e


def test_ddj_t1_default_play_tempo_jog(tmp_data_dir):
    e = _engine()
    c = Controller(e, device="PIONEER DDJ-T1")
    c.handle_raw([0x90, 11, 127])                       # deck A play
    assert e.decks["A"].playing
    c.handle_raw([0xB0, 0, 127])                        # tempo MSB
    c.handle_raw([0xB0, 32, 127])                       # tempo LSB -> max
    assert abs(e.decks["A"].tempo - 1.0) < 1e-3
    c.handle_raw([0x90, 11, 127])                       # pause
    pos = e.decks["A"].play_pos
    c.handle_raw([0xB0, 1, 0])
    c.handle_raw([0xB0, 33, 100])                       # jog while paused: search forwards
    assert e.decks["A"].play_pos > pos


def test_learner_detects_14bit_and_buttons():
    from cta.dj.midi import Event

    L = Learner()
    assert L.offer("play", Event("note", 2, 40, 127, 0)).num == 40
    b = None
    for i in range(4):
        b = b or L.offer("volume", Event("cc", 6, 3, 10 + i, 0))
        b = b or L.offer("volume", Event("cc", 6, 35, 5, 0))
    assert b is not None and b.type == "cc14" and b.num == 3 and b.ch == 6
