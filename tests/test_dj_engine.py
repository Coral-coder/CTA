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


@pytest.mark.parametrize("phase_lock", [True, False])
def test_automix_plays_queue_beatmatched(phase_lock):
    """Auto-mix blends stay on the beat - also when SYNC is set to tempo only."""
    from types import SimpleNamespace

    from cta.dj.automix import AutoMix, AutoMixSettings

    tracks = {}
    for name, bpm, chord in (("t1", 124.0, (57, 60, 64)), ("t2", 128.0, (52, 55, 59)), ("t3", 126.0, (60, 64, 67))):
        a = _stereo(bpm, 70, chord=chord)
        tracks[name] = SimpleNamespace(audio=a, grid=fit_grid(a[0], SR, bpm), peaks=None,
                                       info=TrackInfo(title=name, bpm=bpm, loudness=-10))

    eng = Engine()
    eng.phase_lock = phase_lock
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


def _tone_deck(eng, freq=440.0, seconds=30, bpm=120.0):
    t = np.arange(SR * seconds) / SR
    y = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    eng.load("A", np.stack([y, y]), BeatGrid(bpm, 0.0, float(seconds)), TrackInfo(loudness=-10, key=21))


def _peak_hz(y):
    return np.fft.rfftfreq(y.size, 1 / SR)[np.argmax(np.abs(np.fft.rfft(y * np.hanning(y.size))))]


def test_key_shift_changes_pitch_not_tempo():
    eng = Engine()
    _tone_deck(eng)
    eng.set("A", keylock=True, key_shift=2.0)
    eng.set_playing("A", True)
    y = _run(eng, 3)[0, SR:]
    assert abs(_peak_hz(y) - 440 * 2 ** (2 / 12)) < 3
    assert abs(eng.decks["A"].time - 3.0) < 0.05
    assert eng.decks["A"].effective_key == 23        # Am + 2 semitones = Bm


def test_key_sync_picks_compatible_key():
    from cta.keys import compatibility

    eng = Engine()
    _load(eng, "A", 128.0)
    _load(eng, "B", 128.0)
    eng.decks["A"].info.key = 21        # Am (8A)
    eng.decks["B"].info.key = 15        # Ebm (2A) - clashes
    eng.set_playing("A", True)
    eng.key_sync("B")
    d = eng.decks["B"]
    assert compatibility(d.effective_key, 21) >= 0.85
    assert abs(d.key_shift) <= 6


def test_hotcues_loops_beatjump():
    eng = Engine()
    _tone_deck(eng)
    d = eng.decks["A"]
    eng.set("A", keylock=False)
    eng.set_playing("A", True)
    _run(eng, 1.1)
    eng.hotcue("A", 0)                    # set, quantised to the beat at 1.0 s
    assert abs(d.hotcues[0] - SR * 1.0) < 2
    _run(eng, 2)
    eng.hotcue("A", 0)                    # jump back, keeping beat phase
    assert abs(d.time - 1.0) < 0.6
    eng.beatjump("A", 4)                  # 4 beats @120 = +2 s
    t = d.time
    assert t > 2.4
    eng.loop_in("A")
    _run(eng, 1.0)
    eng.loop_out("A")
    assert d.loop is not None
    L = d.loop[1] - d.loop[0]
    eng.loop_resize("A", 0.5)
    assert abs((d.loop[1] - d.loop[0]) - L / 2) < 1


def test_flux_mode_resumes_timeline():
    eng = Engine()
    _tone_deck(eng)
    d = eng.decks["A"]
    eng.set("A", keylock=False, flux=True)
    eng.set_playing("A", True)
    _run(eng, 1.0)
    eng.loop_beats("A", 1)                # 0.5 s loop
    _run(eng, 3.0)
    eng.loop_beats("A", None)
    _run(eng, 0.2)
    assert abs(d.time - 4.2) < 0.15       # continued as if the loop never happened


def test_filter_and_fx_are_sane():
    eng = Engine()
    t = np.arange(SR * 8) / SR
    sig = (0.3 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    eng.load("A", np.stack([sig, sig]), BeatGrid(120.0, 0.0, 8.0), TrackInfo(loudness=-10))
    eng.set("A", keylock=False, filter=-0.7)          # low-pass well below 6 kHz
    eng.set_playing("A", True)
    y = _run(eng, 1.5)[0, SR // 2:]
    spec = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(y.size, 1 / SR)
    assert spec[(f > 5900) & (f < 6100)].max() < spec[(f > 90) & (f < 110)].max() * 0.05
    eng.set("A", filter=0.0)
    for kind in ("Delay", "Reverb", "Flanger", "Gater", "Crush"):
        u = eng.fx_units[0]
        u.kind, u.on, u.wet, u.amount = kind, True, 0.8, 0.6
        eng.set("A", fx={0})
        y = _run(eng, 0.5)
        assert np.all(np.isfinite(y)) and np.abs(y).max() < 1.5, kind
    u.on = False
    _run(eng, 0.3)
    assert eng.decks["A"].level > 0


def test_live_stream_serves_mp3_and_status():
    import socket
    import threading
    import time as _t
    import urllib.request

    from cta.dj.stream import StreamServer

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    eng = Engine()
    _load(eng, "A", 128.0)
    eng.set_playing("A", True)
    srv = StreamServer(eng, port=port, status_fn=lambda: {"decks": [], "automix": "off"})
    srv.start()
    try:
        assert b"CTA live" in urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
        got = bytearray()
        done = threading.Event()

        def listen():
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/live.mp3", timeout=10) as r:
                while len(got) < 20000:
                    got.extend(r.read(4096))
            done.set()

        threading.Thread(target=listen, daemon=True).start()
        t0 = _t.time()
        while not done.is_set() and _t.time() - t0 < 20:
            eng.render(1024)
            _t.sleep(0.002)
        assert done.is_set()
        # MP3 frame sync word
        assert any(got[i] == 0xFF and (got[i + 1] & 0xE0) == 0xE0 for i in range(min(len(got) - 1, 4000)))
        import json
        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5).read())
        assert st["automix"] == "off"
    finally:
        srv.stop()


def test_cue_on_first_beat_and_markers_drive_automix():
    from types import SimpleNamespace

    from cta.dj.automix import AutoMix, AutoMixSettings
    from cta.dj.beatgrid import beat_activity
    from cta.dj.loader import waveform_bands

    # 6 s of near-silence before the beats start
    a = _stereo(126.0, 60, offset=6.0)
    g = fit_grid(a[0], SR, 126.0)
    bands = waveform_bands(a)
    _p, first, last = beat_activity(bands, g)
    assert abs(g.time_of_beat(first) - 6.0) < 0.1          # first audible beat found

    eng = Engine()
    eng.load("A", a, g, TrackInfo(loudness=-10), cue=g.time_of_beat(first), bands=bands)
    assert abs(eng.decks["A"].cue_point / SR - 6.0) < 0.1  # cue on the first beat, not 0:00

    # markers: outro marker decides exactly where the blend starts
    tracks = {n: SimpleNamespace(audio=_stereo(126.0, 60), grid=None, peaks=None,
                                 info=TrackInfo(title=n, bpm=126.0, loudness=-10)) for n in ("x", "y")}
    for t in tracks.values():
        t.grid = fit_grid(t.audio[0], SR, 126.0)
        t.bands = waveform_bands(t.audio)
        t.first_beat_t = t.grid.first_beat
    beat = 60 / 126
    tracks["x"].mix_out = tracks["x"].grid.time_of_beat(32)          # bar 8
    tracks["y"].mix_in = tracks["y"].grid.time_of_beat(16)           # bar 4
    eng = Engine()
    am = AutoMix(eng, lambda i: tracks[i], AutoMixSettings(mix_bars=4), synchronous_loads=True)
    am.set_queue(["x", "y"])
    am.start()
    started_at = None
    for _ in range(int(40 * SR / 1024)):
        eng.render(1024)
        if am.tr is not None and am.tr.started and started_at is None:
            started_at = eng.decks[am.tr.out_deck].time
            in_pos = eng.decks[am.tr.in_deck].time
    assert started_at is not None
    assert abs(started_at - tracks["x"].grid.time_of_beat(32)) < beat * 0.6
    assert abs(in_pos - tracks["y"].grid.time_of_beat(16)) < beat * 0.6


def test_cue_drag_off_and_play_latch_keep_playing():
    eng = Engine()
    _load(eng, "A", 124.0)
    d = eng.decks["A"]
    eng.cue_press("A")                  # at the cue point: preview
    _run(eng, 0.3)
    assert d.playing and d.cue_preview
    eng.cue_release("A")                # let go on the button: back to the cue
    _run(eng, 0.05)
    assert not d.playing and abs(d.play_pos - d.cue_point) < 1
    eng.cue_press("A")
    _run(eng, 0.3)
    eng.cue_release("A", keep_playing=True)     # dragged off the button
    _run(eng, 0.2)
    assert d.playing and not d.cue_preview and d.play_pos > d.cue_point + 0.3 * SR
    eng.cue_press("A")                  # playing: CUE stops and returns
    _run(eng, 0.05)
    assert not d.playing
    eng.cue_press("A")
    eng.toggle_play("A")                # PLAY while holding CUE: latch
    eng.cue_release("A")
    _run(eng, 0.3)
    assert d.playing and not d.cue_preview


def test_levels_reset_when_a_track_loads():
    eng = Engine()
    _load(eng, "A", 124.0)
    eng.set("A", eq=(0.0, 1.5, 0.2), trim=2.0, filter=0.6, key_shift=2.0)
    _run(eng, 0.05)
    _load(eng, "A", 126.0)
    d = eng.decks["A"]
    assert d.trim == 1.0 and d.filter == 0.0 and d.key_shift == 0.0 and np.allclose(d.eq, 1.0)
    eng.reset_on_load = False                     # a controller is attached: its knobs keep the say
    eng.set("A", eq=(0.0, 1.5, 0.2), trim=2.0, filter=0.6)
    _run(eng, 0.05)
    _load(eng, "A", 124.0)
    assert d.trim == 2.0 and d.filter == 0.6 and d.eq[0] == 0.0


def test_taking_over_a_transition_hands_control_back_and_resumes():
    from types import SimpleNamespace

    from cta.dj.automix import AutoMix, AutoMixSettings

    tracks = {}
    for name, bpm in (("t1", 124.0), ("t2", 124.0), ("t3", 124.0)):
        a = _stereo(bpm, 60)
        tracks[name] = SimpleNamespace(audio=a, grid=fit_grid(a[0], SR, bpm), peaks=None,
                                       info=TrackInfo(title=name, bpm=bpm, loudness=-10))
    eng = Engine()
    am = AutoMix(eng, lambda item: tracks[item], AutoMixSettings(mix_bars=8, phrase_bars=4),
                 synchronous_loads=True)
    took = []
    am.learner = SimpleNamespace(took_over=lambda: took.append(1), preferences=lambda g: None)
    am.set_queue(["t1", "t2", "t3"])
    am.start()
    am.transition_now(8)
    for _ in range(int(4 * SR / 1024)):
        eng.render(1024)
    assert am.tr is not None and am.tr.started
    tr = am.tr
    # touching a deck that is not in the blend changes nothing
    assert not am.manual("C", "volume") and am.active
    assert am.manual(tr.in_deck, "volume")
    assert not am.active and am.resume_after_manual and took
    # the DJ finishes the blend: auto-mix no longer moves the faders
    eng.decks[tr.in_deck].volume = 0.7
    for _ in range(int(2 * SR / 1024)):
        eng.render(1024)
    assert eng.decks[tr.in_deck].volume == 0.7
    eng.set_playing(tr.out_deck, False)
    eng.render(1024)
    am._resume_t = -100.0                  # pretend one track has played alone for a while
    am.maybe_resume()
    assert am.active and not am.resume_after_manual


@pytest.mark.parametrize("keylock", [True, False])
def test_loops_keep_playing_audio(keylock):
    """A loop must sound every time round, with key lock (time-stretcher) too."""
    eng = Engine()
    _load(eng, "A", 124.0)
    d = eng.decks["A"]
    eng.set("A", keylock=keylock)
    eng.set_playing("A", True)
    _run(eng, 3)
    for beats in (4, 1, 8):
        eng.loop_beats("A", beats)
        _run(eng, 0.05)
        out = _run(eng, 6)
        blocks = out[0, : out.shape[1] // 512 * 512].reshape(-1, 512)
        rms = np.sqrt((blocks ** 2).mean(axis=1))
        assert (rms < 1e-4).mean() < 0.02, (beats, (rms < 1e-4).sum())   # no silence
        assert d.loop[0] - 1 <= d.play_pos <= d.loop[1] + 1
        eng.loop_beats("A", None)
        _run(eng, 0.5)


def test_scratching_back_past_the_start_never_stops_the_deck():
    eng = Engine()
    _load(eng, "A", 124.0)
    d = eng.decks["A"]
    eng.set("A", keylock=True)
    eng.set_playing("A", True)
    _run(eng, 0.3)
    eng.scratch("A", -4.0, True)                  # pull the record back hard
    _run(eng, 0.5)
    assert d.play_pos == 0.0                      # held at the start, like a record
    eng.jog_release("A")
    _run(eng, 1.0)
    assert d.playing and not eng.deck_errors and d.play_pos > 0


def test_loop_out_without_in_loops_the_chosen_length_at_once():
    eng = Engine()
    _, g = _load(eng, "A", 124.0)
    d = eng.decks["A"]
    eng.set_playing("A", True)
    _run(eng, 3)
    eng.set("A", loop_size=8.0)
    eng.loop_out("A")                              # no IN: an 8-beat loop right away
    _run(eng, 0.05)
    assert d.loop is not None and abs((d.loop[1] - d.loop[0]) / SR - 8 * g.beat_len) < 1e-3
    eng.loop_beats("A", None)
    eng.loop_in("A")                               # IN tagged earlier in the track ...
    _run(eng, 2.0)
    eng.loop_out("A")                              # ... OUT loops from it to here
    _run(eng, 0.05)
    assert abs((d.loop[1] - d.loop[0]) / SR - 2.0) < g.beat_len
    eng.loop_beats("A", None)
    _run(eng, 0.5)
    eng.loop_out("A", 2)                           # the IN was used: the chosen length again
    _run(eng, 0.05)
    assert abs((d.loop[1] - d.loop[0]) / SR - 2 * g.beat_len) < 1e-3


def test_jog_nudge_on_a_synced_deck_stays_where_the_dj_put_it():
    eng = Engine()
    _load(eng, "A", 124.0)
    _load(eng, "B", 124.0, offset=0.1)
    eng.set_playing("A", True)
    _run(eng, 1)
    eng.set("B", sync=True)
    eng.set_playing("B", True)
    _run(eng, 4)
    assert abs(eng.phase_error("B")) < 0.02           # locked on the beat
    for _ in range(40):                                # the DJ pushes it ahead with the jog
        eng.jog("B", 40.0, False)
        _run(eng, 0.02)
    _run(eng, 1)
    moved = eng.phase_error("B")
    assert abs(moved) > 0.03
    _run(eng, 5)                                       # the lock keeps the nudge, never pulls it back
    assert abs(eng.phase_error("B") - moved) < 0.01
    eng.set("B", sync=True)                            # pressing SYNC again: back on the beat
    _run(eng, 4)
    assert abs(eng.phase_error("B")) < 0.02


def test_tempo_only_sync_leaves_the_phase_to_the_dj():
    eng = Engine()
    _load(eng, "A", 128.0)
    _load(eng, "B", 124.0, offset=0.21)
    eng.phase_lock = False
    eng.set_playing("A", True)
    _run(eng, 2)
    before = eng.phase_error("B")
    eng.set("B", sync=True)
    eng.set_playing("B", True)                    # no jump onto the beat when it starts
    _run(eng, 0.02)
    start = eng.phase_error("B")
    assert abs(start - before) < 0.05 or abs(abs(start - before) - 1) < 0.05
    _run(eng, 6)
    assert abs(eng.decks["B"].bpm - 128.0) < 0.01  # the tempo is matched ...
    assert abs(eng.phase_error("B") - start) < 0.01   # ... and the phase stays where it was
    eng.phase_lock = True                          # beat sync again: it is pulled onto the beat
    _run(eng, 4)
    assert abs(eng.phase_error("B")) < 0.02


def test_cue_preview_plays_straight_from_the_cue_point_even_when_synced():
    eng = Engine()
    _load(eng, "A", 128.0)
    _load(eng, "B", 128.0, offset=0.2)
    eng.set_playing("A", True)
    _run(eng, 1)
    eng.set("B", sync=True)
    d = eng.decks["B"]
    eng.set_cue("B", 5.0)
    _run(eng, 0.05)
    eng.cue_press("B")                              # hold CUE: preview from the cue point
    _run(eng, 1.0)
    assert d.cue_preview and d.phase_corr == 0.0   # sync does not pull it while you preview
    assert abs(d.time - (5.0 + 1.0 * d.rate)) < 0.03
    assert abs(d.start_pos / SR - 5.0) < 1e-6      # the display holds here, never earlier
    eng.cue_release("B")


def test_setting_up_the_incoming_deck_before_the_blend_makes_it_the_djs():
    from types import SimpleNamespace

    from cta.dj.automix import AutoMix, AutoMixSettings

    tracks = {}
    for name in ("t1", "t2"):
        a = _stereo(124.0, 60)
        tracks[name] = SimpleNamespace(audio=a, grid=fit_grid(a[0], SR, 124.0), peaks=None,
                                       info=TrackInfo(title=name, bpm=124.0, loudness=-10, lib_index=len(tracks)))
    eng = Engine()
    am = AutoMix(eng, lambda item: tracks[item], AutoMixSettings(mix_bars=8), synchronous_loads=True)
    am.set_queue(["t1", "t2"])
    am.start()
    for _ in range(int(3 * SR / 1024)):
        eng.render(1024)
    assert am.tr is not None and not am.tr.started
    din = am.tr.in_deck
    eng.set(din, volume=0.3)                        # the DJ sets the incoming fader before the blend
    assert am.manual(din, "volume") and not am.active   # ... so the blend is theirs: auto-mix lets go
    for _ in range(int(2 * SR / 1024)):
        eng.render(1024)
    assert abs(eng.decks[din].volume - 0.3) < 1e-6  # never overridden
    # with "carry on after I take over" off, auto-mix stays off
    am.resume_after_takeover = False
    am._resume_t = -100.0
    am.maybe_resume()
    assert not am.active and not am.resume_after_manual
    # the DJ mixes by hand into the incoming and stops the old deck: auto-mix, started again,
    # never plans a blend back into the track that already played
    out = am.decks[0] if din != am.decks[0] else am.decks[1]
    eng.set_playing(din, True)
    eng.render(1024)
    eng.set_playing(out, False)
    eng.render(1024)
    am.start()
    for _ in range(int(2 * SR / 1024)):
        eng.render(1024)
    assert am.tr is None or am.tr.in_deck != out or eng.decks[out].info.title != "t1"


def test_master_tempo_fader_still_works_with_sync_on_both_decks():
    """SYNC lit on both decks: the master leads, the other follows - the
    master's tempo fader must move the whole mix, not be ignored."""
    eng = Engine()
    _load(eng, "A", 128.0)
    _load(eng, "B", 124.0)
    eng.set("A", sync=True)
    eng.set("B", sync=True)
    eng.set_playing("A", True)
    _run(eng, 1)
    eng.set_playing("B", True)
    _run(eng, 2)
    assert abs(eng.decks["B"].bpm - 128.0) < 0.01
    eng.set("A", tempo=0.5, tempo_range=0.08)             # +4 % on the master
    _run(eng, 3)
    assert abs(eng.decks["A"].bpm - 128.0 * 1.04) < 0.05
    assert abs(eng.decks["B"].bpm - 128.0 * 1.04) < 0.05  # the follower came along


def test_cueing_a_synced_deck_never_bends_the_master():
    """Both decks synced, B the master and playing: holding CUE on A (a cue
    preview) must leave B's tempo alone - and A plays the preview at B's tempo."""
    eng = Engine()
    _load(eng, "A", 128.0)
    _load(eng, "B", 124.0, offset=0.21)
    eng.set("A", sync=True)
    eng.set("B", sync=True)
    eng.set_master("B")
    eng.set_playing("B", True)
    _run(eng, 2)
    rates = []
    eng.cue_press("A")
    for _ in range(30):
        _run(eng, 0.05)
        rates.append(eng.decks["B"].rate)
    eng.cue_release("A")
    assert all(abs(r - 1.0) < 1e-9 for r in rates), rates      # B: untouched, no bend at all
    assert abs(eng.decks["A"].bpm - 124.0) < 0.01               # A previewed at the master's tempo


def test_cue_and_loops_land_where_the_dj_hears_them():
    """The render runs display_latency ahead of the speakers: CUE, hot cues,
    loop IN/OUT and 'beat 1 here' pressed while playing land on what was
    heard at that moment, not on what was about to be rendered."""
    eng = Engine()
    _load(eng, "A", 120.0)
    eng.quantize = False
    eng.display_latency = 0.1                          # 100 ms late output
    eng._stream = object()                             # "running": a real output is open
    try:
        eng.set_playing("A", True)
        _run(eng, 2)
        d = eng.decks["A"]
        heard = d.play_pos - 0.1 * SR
        eng.hotcue("A", 3)
        _run(eng, 0.05)
        assert abs(d.hotcues[3] - heard) < 0.03 * SR   # within a block of the heard position
        p0 = d.play_pos
        eng.loop_in("A")
        _run(eng, 0.5)
        eng.loop_out("A")
        _run(eng, 0.05)
        assert d.loop is not None and abs(d.loop[0] - (p0 - 0.1 * SR)) < 0.03 * SR
        assert abs((d.loop[1] - d.loop[0]) / SR - 0.5) < 0.05
        eng.loop_beats("A", None)
        eng.set_playing("A", False)                    # stopped: what you see is what is there (no latency)
        _run(eng, 0.05)
        p = d.play_pos
        eng.cue_press("A")                             # stopped, away from the cue: sets it right here
        _run(eng, 0.05)
        assert abs(d.cue_point - p) < 1.0
        eng.cue_release("A")
    finally:
        eng._stream = None
