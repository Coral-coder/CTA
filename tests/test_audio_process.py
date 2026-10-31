"""The audio engine in its own process: the client drives it like the
in-process engine, tracks go over in shared memory, the master output
comes back through the ring, auto-mix loads through the GUI's loader."""
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import synth_track

pytest.importorskip("pylibrb")

from cta.dj.beatgrid import fit_grid  # noqa: E402
from cta.dj.engine import SR, TrackInfo  # noqa: E402
from cta.dj.engine_client import EngineClient  # noqa: E402


def _track(name, bpm, seconds=40, idx=0):
    y = synth_track(bpm, seconds, chord=(57, 60, 64), sr=SR)
    a = np.stack([y, y])
    from cta.dj.loader import waveform_bands

    return SimpleNamespace(audio=a, grid=fit_grid(a[0], SR, bpm), peaks=np.ones((100, 3), np.float32),
                           bands=waveform_bands(a), info=TrackInfo(path=f"/m/{name}.flac", title=name, bpm=bpm,
                                                                  loudness=-10, lib_index=idx),
                           cues=None, mix_in=None, mix_out=None, mix_out_end=None, first_beat_t=0.0,
                           phrase_origin=0)


def _wait(cond, timeout=10.0, step=0.02):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if cond():
            return True
        time.sleep(step)
    return False


@pytest.mark.parametrize("inproc", [True, False])
def test_client_drives_the_engine_in_the_audio_process(inproc):
    tracks = {0: _track("one", 124.0, idx=0), 1: _track("two", 126.0, idx=1)}
    eng = EngineClient(block=512, inproc=inproc, load_fn=lambda item: tracks[int(item)])
    try:
        assert eng.ping()
        eng.start_virtual()                                   # paced render thread, no sound card
        eng.load_track("A", tracks[0], only_if_stopped=False)
        assert _wait(lambda: eng.decks["A"].loaded and eng.decks["A"].grid is not None)
        d = eng.decks["A"]
        assert d.info.title == "one" and d.bands is not None and d.length == tracks[0].audio.shape[1]
        eng.set_playing("A", True)
        assert _wait(lambda: d.playing and d.time > 0.5)
        t0 = d.time
        time.sleep(0.5)
        assert 0.3 < d.time - t0 < 0.8                        # real time, from the other process
        # mixer writes land, and come back in the snapshot
        eng.set("A", volume=0.5, eq=(0.2, 1.0, 1.0))
        eng.crossfader = -0.5
        assert _wait(lambda: abs(d.volume - 0.5) < 1e-6 and abs(d.eq[0] - 0.2) < 1e-6)
        assert _wait(lambda: abs(eng.crossfader + 0.5) < 1e-6)
        # the master output reaches the GUI side: a tap and the scope get audio
        got = []
        eng.taps.append(SimpleNamespace(write=lambda b: got.append(b.shape[1])))
        assert _wait(lambda: sum(got) > SR * 0.5)
        assert eng.scope is not None and float(np.abs(eng.scope).max()) > 0.01
        # loops, cue, hot cues through the same API
        eng.loop_beats("A", 4)
        assert _wait(lambda: d.loop is not None)
        eng.hotcue("A", 2)
        assert _wait(lambda: d.hotcues[2] is not None)
        eng.loop_beats("A", None)
        # auto-mix: the queue is loaded through the GUI's loader (shared memory) and mixed
        am = eng.automix
        am.set_queue([1])
        am.start()
        assert _wait(lambda: am.active and "B" in am.meta and eng.decks["B"].loaded, timeout=30)
        assert eng.decks["B"].info.title == "two" and eng.decks["B"].grid is not None
        am.stop()
        eng.set_playing("A", False)
        assert _wait(lambda: not d.playing)
        # unloading frees the shared memory (the server says so, the client unlinks)
        n_before = len(eng._handles)
        eng.unload("B")
        assert _wait(lambda: not eng.decks["B"].loaded and len(eng._handles) < n_before, timeout=10)
    finally:
        eng.close()
    assert not eng.process_alive


def test_mix_ahead_preview_records_in_the_audio_process():
    tracks = {0: _track("one", 124.0, idx=0), 1: _track("two", 124.0, idx=1)}
    eng = EngineClient(block=512, inproc=False, load_fn=lambda item: tracks[int(item)])
    try:
        eng.start_virtual()
        pv = eng.create_preview()
        pv.load_track("A", tracks[0], only_if_stopped=False)
        pv.load_track("B", tracks[1], only_if_stopped=False)
        assert _wait(lambda: pv.decks["A"].loaded and pv.decks["B"].loaded)
        pv.set("B", sync=True, keylock=True)
        pv.set_playing("A", True)
        time.sleep(0.5)
        pv.set_playing("B", True)                               # mixing the two decks records
        s = pv.session
        assert _wait(lambda: s.mode == "recording", timeout=5)
        pv.set("B", volume=1.0)
        time.sleep(4.5)                                          # over 2 bars at 124 bpm
        pv.set("A", volume=0.0)
        assert _wait(lambda: s.mode == "done" and s.result is not None, timeout=15)
        r = s.result
        assert r.out_path == "/m/one.flac" and r.in_path == "/m/two.flac" and r.beats >= 8
        # the live decks never moved
        assert not any(d.playing for d in eng.decks.values())
        eng.close_preview()
    finally:
        eng.close()


def test_gui_stalls_do_not_reach_the_audio_process():
    """The point of the separate process: hold this (GUI) process's
    interpreter for 150 ms at a time - the audio keeps flowing evenly."""
    tracks = {0: _track("one", 124.0, 60, idx=0)}
    eng = EngineClient(block=512, inproc=False, load_fn=lambda item: tracks[int(item)])
    try:
        eng.start_virtual()
        eng.load_track("A", tracks[0], only_if_stopped=False)
        assert _wait(lambda: eng.decks["A"].loaded)
        eng.set("A", keylock=True, tempo=0.3)
        eng.set_playing("A", True)
        stamps = []
        eng.taps.append(SimpleNamespace(write=lambda b: stamps.append((time.perf_counter(), b.shape[1]))))
        stop = threading.Event()

        def hog():
            # pure-Python busy loops hold the GIL of *this* process
            while not stop.is_set():
                t = time.perf_counter()
                while time.perf_counter() - t < 0.15:
                    sum(range(2000))
                time.sleep(0.05)
        threading.Thread(target=hog, daemon=True).start()
        time.sleep(4.0)
        stop.set()
        frames = sum(n for _t, n in stamps)
        assert frames >= SR * 3.5                              # nothing lost: the ring holds it all
        # Lateness is only meaningful on a real machine: a shared CI runner (2-3 cores,
        # noisy neighbours) starves the child of CPU outright, which no process split can
        # fix - the macOS runner has shown anything from 2% to 25% late blocks for the
        # same code. In-process, the same hog makes ~75% of blocks late on a desktop.
        if not os.environ.get("CI"):
            assert eng.underruns <= frames // 512 // 100, (eng.underruns, frames // 512)
    finally:
        eng.close()


def test_positions_run_on_between_snapshots():
    """The GUI side draws the playhead where it is *now*, not where the last
    30 Hz snapshot said: the time runs on from the snapshot's stamp, wraps in
    a loop, and stands still when the deck is stopped."""
    tracks = {0: _track("one", 120.0, 30, idx=0)}
    eng = EngineClient(block=512, inproc=False, load_fn=lambda item: tracks[int(item)])
    try:
        eng.start_virtual()
        eng.load_track("A", tracks[0], only_if_stopped=False)
        assert _wait(lambda: eng.decks["A"].loaded)
        d = eng.decks["A"]
        eng.set_playing("A", True)
        assert _wait(lambda: d.playing and d.time > 0.2)
        # between two snapshots the time keeps moving smoothly (never backwards, never a 33 ms stall)
        ts = [d.time for _ in range(200) if not time.sleep(0.002)]
        steps = np.diff(ts)
        assert (steps >= -1e-6).all(), steps.min()           # never a step back (a block's quantisation)
        assert (steps > 0).mean() > 0.8 and steps.max() < 0.06   # moves on nearly every read; no 33 ms stalls
        eng.loop_beats("A", 4)
        assert _wait(lambda: d.loop is not None)
        time.sleep(1.0)
        lp = d.loop
        assert all(lp[0] / SR - 0.01 <= d.time <= lp[1] / SR + 0.01 for _ in range(50) if not time.sleep(0.01))
        eng.set_playing("A", False)
        assert _wait(lambda: not d.playing)
        t0 = d.time
        time.sleep(0.2)
        assert d.time == t0
    finally:
        eng.close()


def test_a_crashed_audio_process_comes_back_where_the_mix_was():
    """The watchdog's respawn puts the decks back exactly: track, position,
    playing, cue, hot cues, loop, levels, master deck, auto-mix queue."""
    tracks = {0: _track("one", 124.0, 40, idx=0), 1: _track("two", 124.0, 40, idx=1)}
    eng = EngineClient(block=512, inproc=False, load_fn=lambda item: tracks[int(item)])
    try:
        eng.start_virtual()
        eng.load_track("A", tracks[0], only_if_stopped=False)
        assert _wait(lambda: eng.decks["A"].loaded)
        eng.seek("A", 5.0)
        eng.set("A", volume=0.5, eq=(0.3, 1.0, 1.0), tempo=0.25, keylock=True)
        eng.set_cue("A", 4.0)
        eng.hotcue("A", 2)
        eng.loop_beats("A", 8)
        eng.set_playing("A", True)
        eng.set_master("A")
        eng.automix.set_queue([1])
        d = eng.decks["A"]
        assert _wait(lambda: d.playing and d.loop is not None and d.hotcues[2] is not None and d.time > 5.2)
        t_before, loop, hc = d.time, d.loop, d.hotcues[2]
        eng._proc.kill()                                  # the crash
        assert _wait(lambda: not eng.process_alive, timeout=10)
        eng.respawn()
        eng.start_virtual()
        assert _wait(lambda: d.loaded and d.playing and d.time > t_before, timeout=15)
        assert d.time - t_before < 3.0                    # back where it was, not from the start
        assert abs(d.volume - 0.5) < 1e-6 and abs(d.eq[0] - 0.3) < 1e-6 and abs(d.tempo - 0.25) < 1e-6
        assert abs(d.cue_point / SR - 4.0) < 1e-6 and d.hotcues[2] == hc and d.loop == loop
        assert eng.master_deck == "A" and list(eng.automix.queue) == [1]
    finally:
        eng.close()


def test_auto_buffer_walks_the_ladder_on_glitches_and_back_down_when_quiet(monkeypatch):
    """"Auto" output safety: start at the card's minimum; glitches raise the
    reserve one step at once; ten quiet minutes with nothing playing bring it
    back down. (The sound card is stubbed: start() only records the latency.)"""
    from cta.dj import audio_server as AS
    from cta.dj.engine import Engine

    opened = []

    def fake_start(self, device=None, block=None, master_chans=(0, 1), phones_chans=None, phones_device=None):
        opened.append(self.out_latency)
        self._stream = object()                       # "running", not virtual
    monkeypatch.setattr(Engine, "start", fake_start)
    monkeypatch.setattr("cta.dj.engine.list_output_devices", lambda: [])   # no PortAudio on the Linux runner
    import multiprocessing as mp

    a, b = mp.Pipe()
    srv = AS.Server(b, {"block": 512})
    try:
        srv.engine.out_latency = "auto"
        srv.start(None, 512, (0, 1), None, None)
        assert opened == ["low"] and srv._auto_latency
        t = 1000.0
        srv._lat_changed = t - 10.0                    # the test's own clock
        srv._check_latency(t)                          # nothing happened: stays
        assert srv._lat_idx == 0
        srv.engine.underruns += 1                      # a glitch: up one step at once
        srv._check_latency(t + 6.0)
        assert srv._lat_idx == 1 and opened[-1] == 0.012 and srv.engine.out_latency == 0.012
        srv.engine.slow_blocks += 3                    # near misses count as trouble too, but one
        srv._check_latency(t + 8.0)                    # burst needs a second within 30 s (or an underrun)
        assert srv._lat_idx == 1
        srv.engine.slow_blocks += 3
        srv._check_latency(t + 12.0)
        assert srv._lat_idx == 2 and opened[-1] == 0.025
        # quiet for ten minutes with no deck playing: one step back down
        srv._check_latency(t + 12.0 + 601.0)
        assert srv._lat_idx == 1 and opened[-1] == 0.012
        # the event reached the pipe
        msgs = []
        while a.poll(0.1):
            msgs.append(a.recv())
        assert any(m[0] == "event" and m[1] == "latency" for m in msgs)
    finally:
        srv.alive = False
        srv.engine.stop()
