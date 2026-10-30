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


def test_learn_wizard_waits_for_space_and_keeps_other_settings(tmp_path, monkeypatch):
    pytest.importorskip("PySide6.QtWidgets")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication

    from cta.dj.midi import Binding, Event, import_mapping
    from cta.gui.dj_window import LearnWizard

    app = QApplication.instance() or QApplication([])
    saved = {}

    class FakeCtl:
        device = "Test Controller"
        bindings = [Binding("note", 0, 11, "play", "A"), Binding("cc", 0, 19, "volume", "A")]
        learn_cb = None

        def set_bindings(self, bs, learned=True):
            saved["b"] = list(bs)

    ctl = FakeCtl()
    wz = LearnWizard(ctl, ("A",))
    steps = [s[0] for s in wz.steps]
    i_play = steps.index("play")
    wz.i = i_play
    wz._show()

    def ev(t, ch, num, v):
        wz._on_event(Event(t, ch, num, v, 0.0))

    def space():
        app.sendEvent(wz, QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier, " "))

    # a fader streaming while a button step is up is not taken for a button
    for v in range(0, 60, 4):
        ev("cc", 0, 5, v)
    assert wz.candidate is None and wz.i == i_play
    # a press is shown but nothing changes until SPACE
    ev("note", 0, 42, 127)
    assert wz.candidate is not None and wz.result[("play", "A")].num == 11 and wz.i == i_play
    ev("note", 0, 43, 127)                      # another button meanwhile: still the first one
    assert wz.candidate.num == 42
    space()
    assert wz.result[("play", "A")].num == 42 and wz.i == i_play + 1
    # the control just confirmed, still sending, is not learned again for the next step
    ev("note", 0, 42, 127)
    assert wz.candidate is None
    # Space with nothing heard skips, keeping what was set
    wz.i = steps.index("volume")
    wz._show()
    space()
    assert wz.result[("volume", "A")].num == 19
    # export, then import puts it to use
    p = str(tmp_path / "map.json")
    import cta.gui.dj_window as DW

    monkeypatch.setattr(DW.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (p, "")))
    wz._export()
    assert {(b.action, b.num) for b in import_mapping(p)} >= {("play", 42), ("volume", 19)}
    monkeypatch.setattr(DW.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (p, "")))
    wz._import()
    assert any(b.action == "play" and b.num == 42 for b in saved["b"])
    wz.done(0)
