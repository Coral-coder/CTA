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
        # a shared CI runner (2-3 cores) also starves the child of CPU, which no process split
        # can fix; a desktop never does. In-process, the same hog makes ~75% of blocks late.
        allowed = len(stamps) // 8 if os.environ.get("CI") else 0
        assert eng.underruns <= allowed, (eng.underruns, len(stamps))
    finally:
        eng.close()
