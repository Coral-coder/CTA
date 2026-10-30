"""Time-shift mixing: rehearse a transition on the preview engine, then
auto-mix replays it live on the same beat with the same moves."""
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import synth_track

pytest.importorskip("pylibrb")

from cta.dj.beatgrid import fit_grid  # noqa: E402
from cta.dj.engine import SR, Engine, TrackInfo  # noqa: E402
from cta.dj.rehearsal import Rehearsal, RehearsalStore, Session, apply, xf_gain  # noqa: E402


def _track(name, bpm, chord):
    y = synth_track(bpm, 60, chord=chord, sr=SR)
    a = np.stack([y, y])
    return SimpleNamespace(audio=a, grid=fit_grid(a[0], SR, bpm), peaks=None,
                           info=TrackInfo(path=f"/m/{name}.flac", title=name, bpm=bpm, loudness=-10))


def _load(eng, deck, t):
    eng.load(deck, t.audio, t.grid, t.info)


def _rehearse(t1, t2):
    pv = Engine()
    _load(pv, "A", t1)
    _load(pv, "B", t2)
    s = Session(pv)
    s.rehearse(out_beat=32.0, in_beat=0.0, lead_beats=8.0)
    a, b = pv.decks["A"], pv.decks["B"]
    for _ in range(int(20 * SR / 512)):            # lead-in: 8 beats at 124 bpm
        pv.render(512)
        if s.mode == "recording":
            break
    assert s.mode == "recording"
    assert abs(a.grid.beat_at(a.time) - 32.0) < 0.1
    # the DJ mixes: bring B up over 8 beats, swap the bass at beat 8, take A out by beat 16
    b.volume = 0.0
    for _ in range(int(30 * SR / 512)):
        pv.render(512)
        x = s.x
        b.volume = float(np.clip(x / 8.0, 0, 1))
        b.eq = np.array([1.0 if x >= 8 else 0.0, 1.0, 1.0])
        a.eq = np.array([0.0 if x >= 8 else 1.0, 1.0, 1.0])
        a.volume = float(np.clip(1 - (x - 8) / 8.0, 0, 1))
        if s.mode == "done":
            break
    assert s.mode == "done"
    r = s.result
    s.close()
    return r


def test_rehearse_records_the_moves_on_the_beat():
    t1, t2 = _track("one", 124.0, (57, 60, 64)), _track("two", 124.0, (52, 55, 59))
    r = _rehearse(t1, t2)
    assert r.out_path == "/m/one.flac" and r.in_path == "/m/two.flac"
    assert r.out_start_beat == 32.0 and r.in_start_beat == 0.0
    assert 16.0 <= r.beats <= 24.0                     # ends a bar after A went silent
    assert abs(r.value("in_vol", 4.0) - 0.5) < 0.1
    assert r.value("in_low", 4.0) < 0.1 and r.value("in_low", 12.0) > 0.9
    assert r.value("out_vol", 15.9) < 0.05
    # round-trips through storage
    r2 = Rehearsal.from_dict(r.to_dict())
    assert r2.value("in_vol", 4.0) == r.value("in_vol", 4.0)


def test_apply_folds_the_crossfader_into_the_faders():
    r = Rehearsal("a", "b", 0, 0, 8, {"xf": [[0, -1.0], [8, 1.0]]})
    din, dout = SimpleNamespace(), SimpleNamespace()
    apply(r, 0.0, din, dout)
    assert dout.volume == 1.0 and din.volume == 0.0          # fully on the outgoing side
    apply(r, 8.0, din, dout)
    assert din.volume == 1.0 and dout.volume == 0.0
    assert xf_gain(0.0, 1) == 1.0 and xf_gain(0.0, -1) == 1.0


def test_automix_replays_the_rehearsal_live(tmp_path):
    from cta.db import Database
    from cta.dj.automix import AutoMix, AutoMixSettings

    t1, t2 = _track("one", 124.0, (57, 60, 64)), _track("two", 124.0, (52, 55, 59))
    r = _rehearse(t1, t2)
    store = RehearsalStore(Database(str(tmp_path / "x.db")))
    store.put(r)
    store = RehearsalStore(store.db)                    # read back from the library database
    assert store.get(r.out_path, r.in_path) is not None

    eng = Engine()
    tracks = {"one": t1, "two": t2}
    am = AutoMix(eng, lambda item: tracks[item], AutoMixSettings(mix_bars=4), synchronous_loads=True)
    am.rehearsals = store
    am.set_queue(["one", "two"])
    am.start()
    seen = []
    for _ in range(int(40 * SR / 1024)):
        eng.render(1024)
        tr = am.tr
        if tr is not None and tr.started:
            din = eng.decks[tr.in_deck]
            x = din.grid.beat_at(din.time) - tr.in_start_beat
            seen.append((x, din.volume, eng.decks[tr.out_deck].grid.beat_at(eng.decks[tr.out_deck].time)))
        if seen and am.tr is None:
            break
    assert seen, "the rehearsed transition never started"
    assert am.history[-1] == "two" and eng.master_deck in ("A", "B")
    # it started on the rehearsed beat of the outgoing track
    assert abs(seen[0][2] - 32.0) < 0.2
    # and the incoming fader followed the rehearsal (half way up after 4 beats)
    mid = min(seen, key=lambda s: abs(s[0] - 4.0))
    assert abs(mid[1] - r.value("in_vol", mid[0])) < 0.05


def test_mixing_the_preview_records_itself_and_can_be_switched_off():
    t1, t2 = _track("one", 124.0, (57, 60, 64)), _track("two", 124.0, (52, 55, 59))
    pv = Engine()
    _load(pv, "A", t1)
    _load(pv, "B", t2)
    s = Session(pv)
    a = pv.decks["A"]
    pv.set("B", sync=True, keylock=True)
    pv.set_playing("A", True)
    for _ in range(int(2 * SR / 512)):
        pv.render(512)
    s.auto_record = False                          # "just mix": nothing is recorded
    pv.set_playing("B", True)
    pv.render(512)
    assert s.mode == "idle"
    pv.set_playing("B", False)
    pv.render(512)
    s.auto_record = True
    pv.set_playing("B", True)                      # bring the incoming in: recording starts by itself
    pv.render(512)
    assert s.mode == "recording"
    for _ in range(int(20 * SR / 512)):
        pv.render(512)
        pv.crossfader = float(np.clip(-1 + s.x / 4.0, -1, 1))   # the DJ swings the crossfader over 8 beats
        a.volume = 1.0 if s.x < 8 else 0.0
        if s.mode == "done":
            break
    assert s.mode == "done" and s.result is not None
    r = s.result
    assert r.touched("xf") and not r.touched("in_low")
    s.close()

    # replay: the crossfader really moves; the untouched EQ follows auto-mix's blend
    from cta.dj.automix import blend_values

    din, dout = SimpleNamespace(xf_side=1), SimpleNamespace(xf_side=-1)
    eng = SimpleNamespace(crossfader=0.0)
    base = blend_values(1.0, r.beats, 8.0, 0.0)
    apply(r, 1.0, din, dout, eng, base)
    assert eng.crossfader < -0.5 and din.eq[0] == base["in_eq"][0]
    # the outgoing on the right-hand deck: the move is mirrored
    din2, dout2 = SimpleNamespace(xf_side=-1), SimpleNamespace(xf_side=1)
    apply(r, 1.0, din2, dout2, eng, base)
    assert eng.crossfader > 0.5
    # fill off: exactly what was recorded
    apply(r, 1.0, din, dout, eng, None)
    assert din.eq[0] == r.value("in_low", 1.0)


def test_a_mix_recorded_after_auto_mix_planned_is_still_used(tmp_path):
    from cta.db import Database
    from cta.dj.automix import AutoMix, AutoMixSettings

    t1, t2 = _track("one", 124.0, (57, 60, 64)), _track("two", 124.0, (52, 55, 59))
    store = RehearsalStore(Database(str(tmp_path / "x.db")))
    eng = Engine()
    tracks = {"one": t1, "two": t2}
    am = AutoMix(eng, lambda item: tracks[item], AutoMixSettings(mix_bars=4), synchronous_loads=True)
    am.rehearsals = store
    am.set_queue(["one", "two"])
    am.start()
    for _ in range(int(2 * SR / 1024)):
        eng.render(1024)
    assert am.tr is not None and am.tr.style != "rehearsed"      # planned before any recording
    store.put(_rehearse(t1, t2))                                  # the DJ mixes it ahead now
    eng.render(1024)
    assert am.tr.style == "rehearsed"


def test_blips_are_not_recorded_and_learned_blends_stay_long(tmp_path):
    from types import SimpleNamespace as NS

    from cta.db import Database
    from cta.dj.learn import MixLearner, TransitionExample

    t1, t2 = _track("one", 124.0, (57, 60, 64)), _track("two", 124.0, (52, 55, 59))
    pv = Engine()
    _load(pv, "A", t1)
    _load(pv, "B", t2)
    s = Session(pv)
    pv.set_playing("A", True)
    pv.render(512)
    pv.set_playing("B", True)                      # in and straight out again
    pv.render(512)
    assert s.mode == "recording"
    pv.set_playing("A", False)
    for _ in range(int(3 * SR / 512)):
        pv.render(512)
    assert s.mode == "done" and s.result is None
    s.close()
    store = RehearsalStore(Database(str(tmp_path / "x.db")))
    store.put(Rehearsal("/a", "/b", 64.0, 0.0, 3.0, {}))          # saved by an older version
    assert store.get("/a", "/b") is None
    lr = MixLearner(Database(str(tmp_path / "y.db")), NS())
    mk = lambda bars: TransitionExample(0, "club", "a", "b", bars, 0.0, 8.0, "drop", "intro")  # noqa: E731
    lr.examples = [mk(0.5), mk(1.0), mk(1.5), mk(3.0), mk(3.0), mk(3.0)]
    p = lr.preferences("club")
    assert p is not None and p["bars"] >= 4 and p["n"] == 3
