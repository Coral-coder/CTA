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
    c.handle_raw([0xB0, 0, 64])                         # tempo fader meets the app's value (centre) first
    c.handle_raw([0xB0, 32, 0])
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


def _tsi_file(path, device="Pioneer.DDJ-T1", inputs=("Deck.A.HotCue 1/5", "Mixer.A.Vol", "Deck.A.Needle.Function 1")):
    """A minimal Traktor .tsi with a controller mapping, built frame by frame."""
    import base64
    import struct

    def fr(tag, body):
        return tag.encode() + struct.pack(">I", len(body)) + body

    def ws(s_):
        return struct.pack(">I", len(s_)) + s_.encode("utf-16-be")
    dcdt = b"".join(fr("DCDT", ws(n) + struct.pack(">IIfII", 1, 0, 1.0, 3, 8)) for n in inputs)
    ddci = fr("DDCI", struct.pack(">I", len(inputs)) + dcdt)
    ddco = fr("DDCO", struct.pack(">I", 0))
    ddat = fr("DDAT", fr("DDIF", b"\0" * 4) + fr("DDDC", ddci + ddco))
    devi = fr("DEVI", ws(device) + ddat)
    blob = fr("DIOM", fr("DIOI", struct.pack(">I", 1)) + fr("DEVS", struct.pack(">I", 1) + devi))
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n<NIXML><TraktorSettings>'
                '<Entry Name="DeviceIO.Config.Controller" Type="3" Value="'
                + base64.b64encode(blob).decode() + '"></Entry></TraktorSettings></NIXML>')


def test_traktor_tsi_import_gives_the_full_ddj_t1(tmp_path):
    from cta.dj.tsi import TsiError, read_tsi, tsi_bindings

    p = str(tmp_path / "Settings.tsi")
    _tsi_file(p)
    assert read_tsi(p) == [{"name": "Pioneer.DDJ-T1",
                            "inputs": ["Deck.A.HotCue 1/5", "Mixer.A.Vol", "Deck.A.Needle.Function 1"],
                            "outputs": []}]
    name, bs, unused = tsi_bindings(p)
    assert name == "Pioneer.DDJ-T1" and unused == ["Deck.Needle.Function 1"]
    have = {(b.type, b.ch, b.num, b.action, b.deck) for b in bs}
    # spot checks against Pioneer's MIDI list
    assert ("note", 0, 11, "play", "A") in have and ("note", 1, 12, "cue", "B") in have
    assert ("cc", 6, 19, "volume", "A") in have and ("cc", 6, 21, "volume", "B") in have
    assert ("cc", 6, 31, "crossfader", None) in have and ("cc", 6, 16, "eq_low", "B") in have
    assert ("note", 6, 88, "sync", "A") in have and ("note", 6, 85, "phones_cue", "B") in have
    _tsi_file(p, device="Generic MIDI")
    with pytest.raises(TsiError):
        tsi_bindings(p)


def test_ddj_t1_performance_buttons_and_mixer():
    from cta.dj.midi import Controller, ddj_t1_default

    eng = Engine()
    t = np.arange(SR * 20) / SR
    y = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    eng.load("A", np.stack([y, y]), BeatGrid(120.0, 0.0, 20.0), TrackInfo(loudness=-10))
    c = Controller(eng, device="Test DDJ-T1")
    c.set_bindings(ddj_t1_default(), learned=False)
    c.handle_raw([0xB6, 19, 127])                     # deck A channel fader: meets the app's value (full)...
    c.handle_raw([0xB6, 19, 64])                      # ... then half way
    eng.render(256)
    assert abs(eng.decks["A"].volume - 64 / 127) < 1e-6
    c.handle_raw([0x90, 46, 127])                     # pad 1 in HOT CUE mode: sets hot cue 1
    eng.render(256)
    assert eng.decks["A"].hotcues[0] is not None
    c.handle_raw([0x90, 45, 127])                     # hold DELETE + pad 1: removes it
    c.handle_raw([0x90, 46, 127])
    c.handle_raw([0x90, 45, 0])
    eng.render(256)
    assert eng.decks["A"].hotcues[0] is None
    c.handle_raw([0x90, 42, 127])                     # MOVE mode: pad 4 jumps forward
    t0 = eng.decks["A"].time
    c.handle_raw([0x90, 49, 127])
    eng.render(256)
    assert eng.decks["A"].time > t0 + 1.0
    c.handle_raw([0xB0, 3, 64])                       # needle strip: half way into the track
    eng.render(256)
    assert abs(eng.decks["A"].time - 10.0) < 0.3
    c.handle_raw([0x96, 92, 127])                     # MASTER A
    eng.render(256)
    assert eng.master_deck == "A"


def test_saved_ddj_t1_mapping_is_completed_and_scratch_can_be_switched_off(tmp_path, monkeypatch):
    import cta.dj.midi as M

    monkeypatch.setattr(M.paths, "data_dir", lambda: str(tmp_path))
    # an old saved map: jog without its touch sensor, play on a learned note
    M.save_mapping("PIONEER DDJ-T1", [M.Binding("cc14", 0, 1, "jog", "A", relative=True),
                                      M.Binding("note", 0, 11, "play", "A")])
    bs, _ = M.load_mapping("PIONEER DDJ-T1")
    have = {(b.type, b.ch, b.num, b.action, b.deck) for b in bs}
    assert ("note", 0, 54, "jog_touch", "A") in have and ("cc", 6, 19, "volume", "A") in have
    assert sum(1 for b in bs if (b.action, b.deck) == ("play", "A")) == 1

    eng = Engine()
    t = np.arange(SR * 20) / SR
    y = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    eng.load("A", np.stack([y, y]), BeatGrid(120.0, 0.0, 20.0), TrackInfo(loudness=-10))
    c = M.Controller(eng, device="PIONEER DDJ-T1")
    eng.set_playing("A", True)
    eng.render(256)
    c.handle_raw([0x90, 54, 127])                 # touch the top
    c.handle_raw([0xB0, 1, 127])
    c.handle_raw([0xB0, 33, 107])                 # turn back
    eng.render(256)
    assert eng.decks["A"].scratching
    c.handle_raw([0x90, 54, 0])
    eng.render(256)
    c.scratch_enabled = False                     # scratch switched off: the jog only nudges
    c.handle_raw([0x90, 54, 127])
    c.handle_raw([0xB0, 1, 127])
    c.handle_raw([0xB0, 33, 107])
    eng.render(256)
    assert not eng.decks["A"].scratching and eng.decks["A"].playing


def test_fader_jitter_is_not_a_takeover():
    from types import SimpleNamespace

    from cta.dj.midi import Binding, Controller

    eng = Engine()
    calls = []
    am = SimpleNamespace(manual=lambda deck, a: calls.append((deck, a)))
    c = Controller(eng, am, device="x")
    c.set_bindings([Binding("cc", 6, 19, "volume", "A"), Binding("note", 0, 11, "play", "A")], learned=False)
    c.handle_raw([0xB6, 19, 127])                   # picked up (the app's fader is at full)
    for v in (64, 65, 64, 63, 64, 65):             # a fader's electrical noise
        c.handle_raw([0xB6, 19, v])
    assert calls == [("A", "volume")]              # the first real move down counted, the jitter did not
    c.handle_raw([0xB6, 19, 75])                    # a real move
    assert calls == [("A", "volume"), ("A", "volume")]
    c.handle_raw([0x90, 11, 127])                   # a button press always counts
    assert calls[-1] == ("A", "play")


def test_controller_picks_up_a_control_only_where_the_app_is():
    """Soft takeover: a fader at a different position than the app's value
    does nothing until it reaches (or crosses) it; after the app moves a
    control by itself the hand has to meet it again."""
    from cta.dj.midi import Binding, Controller

    eng = Engine()
    c = Controller(eng, device="x")
    c.set_bindings([Binding("cc", 6, 19, "volume", "A"), Binding("cc", 6, 24, "crossfader", None)], learned=False)
    d = eng.decks["A"]
    assert d.volume == 1.0
    c.handle_raw([0xB6, 19, 40])                     # far from the app's value: ignored
    assert d.volume == 1.0
    c.handle_raw([0xB6, 19, 90])                     # still below it: ignored
    assert d.volume == 1.0
    c.handle_raw([0xB6, 19, 126])                    # within reach: picked up
    assert abs(d.volume - 126 / 127) < 1e-6
    c.handle_raw([0xB6, 19, 60])                     # follows the hand from now on
    assert abs(d.volume - 60 / 127) < 1e-6
    eng.set("A", volume=0.2)                         # the app moved it (auto-mix, the mouse)
    c.handle_raw([0xB6, 19, 70])                     # no jump: not picked up again yet
    assert abs(d.volume - 0.2) < 1e-6
    c.handle_raw([0xB6, 19, 10])                     # crossed the app's value on the way down: picked up
    assert abs(d.volume - 10 / 127) < 1e-6
    # the crossfader (an engine-level control) behaves the same
    eng.crossfader = -0.5
    c.handle_raw([0xB6, 24, 127])
    assert eng.crossfader == -0.5
    c.handle_raw([0xB6, 24, 32])                     # ~ -0.5: met
    assert abs(eng.crossfader - (32 / 127 * 2 - 1)) < 1e-6
