"""Offline tests of the DJ engine: sync/phase lock, key lock, EQ, loops."""
import numpy as np
import pytest
from conftest import synth_track

pytest.importorskip("pylibrb")
pytest.importorskip("scipy")

from cta.dj.beatgrid import BeatGrid, fit_grid  # noqa: E402
from cta.dj.engine import SR, Engine, TrackInfo  # noqa: E402


def _stereo(bpm, seconds=40, chord=(57, 60, 64), offset=0.0):
    y = synth_track(bpm, seconds, chord=chord, sr=SR)
    y = np.concatenate([np.zeros(int(offset * SR), np.float32), y])
    return np.stack([y, y])


def _load(eng, deck, bpm, offset=0.0, chord=(57, 60, 64)):
    a = _stereo(bpm, offset=offset, chord=chord)
    g = fit_grid(a[0], SR, bpm)
    eng.load(deck, a, g, TrackInfo(title=deck, bpm=bpm, loudness=-10))
    return a, g


def _run(eng, seconds, block=512):
    out = []
    for _ in range(int(seconds * SR / block)):
        m, _ = eng.render(block)
        out.append(m)
    return np.concatenate(out, axis=1)


def test_grid_phase_on_synthetic():
    for bpm, off in ((128, 0.37), (174, 0.05), (124.5, 1.234)):
        a = _stereo(bpm, 30, offset=off)
        g = fit_grid(a[0], SR, bpm)
        beat = 60 / bpm
        err = ((g.first_beat - off + beat / 2) % beat) - beat / 2
        assert abs(err) < 0.004, (bpm, err)


def test_sync_locks_tempo_and_phase():
    eng = Engine()
    _load(eng, "A", 128.0)
    _load(eng, "B", 124.0, offset=0.21)       # different tempo and phase
    eng.set_playing("A", True)
    _run(eng, 3)
    eng.set("B", sync=True, keylock=True)
    eng.set_playing("B", True)
    _run(eng, 6)
    assert abs(eng.decks["B"].bpm - 128.0) < 0.01
    errs = []
    for _ in range(40):
        _run(eng, 0.1)
        errs.append(eng.phase_error("B"))
    assert max(abs(e) for e in errs) < 0.02    # < 2 % of a beat (~9 ms)


def test_keylock_keeps_pitch():
    eng = Engine()
    tone = np.sin(2 * np.pi * 440 * np.arange(SR * 20) / SR).astype(np.float32) * 0.5
    eng.load("A", np.stack([tone, tone]), BeatGrid(120.0, 0.0, 20.0), TrackInfo(loudness=-10))
    eng.set("A", keylock=True, tempo=1.0, tempo_range=0.08)    # +8 %
    eng.set_playing("A", True)
    y = _run(eng, 4)[0, SR:]
    f = np.fft.rfftfreq(y.size, 1 / SR)[np.argmax(np.abs(np.fft.rfft(y * np.hanning(y.size))))]
    assert abs(f - 440) < 2
    assert abs(eng.decks["A"].time - 4 * 1.08) < 0.05

    eng.set("A", keylock=False)                 # vinyl: pitch follows tempo
    y = _run(eng, 3)[0, SR:]
    f = np.fft.rfftfreq(y.size, 1 / SR)[np.argmax(np.abs(np.fft.rfft(y * np.hanning(y.size))))]
    assert abs(f - 440 * 1.08) < 3


def test_eq_kill_low():
    eng = Engine()
    t = np.arange(SR * 6) / SR
    sig = (0.4 * np.sin(2 * np.pi * 60 * t) + 0.4 * np.sin(2 * np.pi * 5000 * t)).astype(np.float32)
    eng.load("A", np.stack([sig, sig]), BeatGrid(120.0, 0.0, 6.0), TrackInfo(loudness=-10))
    eng.set("A", keylock=False)
    eng.set_playing("A", True)
    _run(eng, 1)
    eng.set("A", eq=(0.0, 1.0, 1.0))
    y = _run(eng, 2)[0, SR // 2:]
    spec = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(y.size, 1 / SR)
    low = spec[(f > 50) & (f < 70)].max()
    high = spec[(f > 4900) & (f < 5100)].max()
    assert low < high * 0.05


def test_loop_and_cue():
    eng = Engine()
    _load(eng, "A", 120.0)
    eng.set("A", keylock=False)
    eng.set_playing("A", True)
    _run(eng, 2)
    eng.loop_beats("A", 4)
    _run(eng, 6)                              # 4 beats @120 = 2 s loop
    d = eng.decks["A"]
    assert d.loop is not None and d.loop[0] <= d.play_pos <= d.loop[1]
    eng.loop_beats("A", None)
    eng.cue_press("A")                        # playing -> back to cue, stop
    assert not d.playing and abs(d.play_pos - d.cue_point) < 1


def test_render_never_raises_on_empty():
    eng = Engine()
    m, p = eng.render(512)
    assert m.shape == (2, 512) and not np.any(m)


def test_automix_plays_queue_beatmatched():
    from types import SimpleNamespace

    from cta.dj.automix import AutoMix, AutoMixSettings

    tracks = {}
    for name, bpm, chord in (("t1", 124.0, (57, 60, 64)), ("t2", 128.0, (52, 55, 59)), ("t3", 126.0, (60, 64, 67))):
        a = _stereo(bpm, 70, chord=chord)
        tracks[name] = SimpleNamespace(audio=a, grid=fit_grid(a[0], SR, bpm), peaks=None,
                                       info=TrackInfo(title=name, bpm=bpm, loudness=-10))

    eng = Engine()
    am = AutoMix(eng, lambda item: tracks[item], AutoMixSettings(mix_bars=4, phrase_bars=4, glide_bars=4),
                 synchronous_loads=True)
    am.set_queue(["t1", "t2", "t3"])
    am.start()

    worst_phase = 0.0
    rms = []
    masters = []
    block = 1024
    for i in range(int(200 * SR / block)):
        m, _ = eng.render(block)
        if i % 43 == 0:
            rms.append(float(np.sqrt(np.mean(m ** 2))))
            masters.append(eng.master_deck)
        if am.tr is not None and am.tr.started:
            err = eng.phase_error(am.tr.in_deck)
            if err is not None:
                worst_phase = max(worst_phase, abs(err))
        if am.history == ["t1", "t2", "t3"] and not any(d.playing for d in eng.decks.values()):
            break
    assert am.history == ["t1", "t2", "t3"]
    assert worst_phase < 0.03, worst_phase              # beats stay locked during every blend
    # continuous music: no silent second until the last track ends
    first_silence = next((k for k, r in enumerate(rms) if r < 1e-3), len(rms))
    assert first_silence > 150 * SR / block / 43 * 0.9 or first_silence == len(rms)
