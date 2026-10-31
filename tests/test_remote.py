"""Mix server: phone remote API, PIN, pause, skip / cut / back, album art,
start at login."""
import json
import os
import socket
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import structured_track, synth_track, write_audio

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pylibrb")

from cta.dj.automix import AutoMix, AutoMixSettings  # noqa: E402
from cta.dj.beatgrid import fit_grid  # noqa: E402
from cta.dj.engine import SR, Engine, TrackInfo  # noqa: E402
from cta.dj.loader import waveform_bands  # noqa: E402

SHORT = (("intro", 8), ("drop", 24), ("outro", 8))


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _get(url, key=None):
    req = urllib.request.Request(url, headers={"X-CTA-Key": key} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), ""


def _post(url, body, key=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **({"X-CTA-Key": key} if key else {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class FakeBackend:
    def __init__(self):
        self.calls = []

    def state(self):
        return {"decks": [], "automix": {"active": False, "status": "off", "blend_bars": 16}, "queue": []}

    def command(self, action, args):
        if action == "bad":
            raise ValueError("nope")
        self.calls.append((action, args))
        return {"message": "done"}

    def search(self, q):
        return [{"idx": 1, "title": q}]

    def art(self, idx):
        return b"\x89PNG fake" if idx == 1 else None

    def wave(self, deck, bins=4000):
        return {"deck": deck} if deck == "A" else None


def test_server_pin_routes_and_rate_limit():
    from cta.dj.stream import StreamServer

    port = _free_port()
    be = FakeBackend()
    srv = StreamServer(Engine(), port=port, backend=be, pin="123456")
    srv.start()
    base = f"http://127.0.0.1:{port}"
    try:
        code, body, _ = _get(base + "/")
        assert code == 200 and b"Tempest live" in body and b"apple-touch-icon" in body
        code, body, ctype = _get(base + "/apple-touch-icon.png")
        assert code == 200 and body[:4] == b"\x89PNG" and ctype == "image/png"
        assert json.loads(_get(base + "/manifest.webmanifest")[1])["display"] == "standalone"
        assert _get(base + "/api/state")[0] == 401                      # no PIN
        code, body, _ = _get(base + "/api/state", "123456")
        assert code == 200 and "decks" in json.loads(body)
        assert _get(base + "/api/state?k=123456")[0] == 200            # PIN as a parameter (audio/img tags)
        assert json.loads(_get(base + "/api/search?q=hello", "123456")[1])["results"][0]["title"] == "hello"
        assert _get(base + "/api/wave?deck=A", "123456")[0] == 200
        assert _get(base + "/api/wave?deck=B", "123456")[0] == 404
        code, body, ctype = _get(base + "/art?i=1&k=123456")
        assert code == 200 and ctype == "image/png"
        assert _get(base + "/art?i=2&k=123456")[0] == 404
        assert _post(base + "/api/cmd", {"action": "skip", "mode": "cut"})[0] == 401
        code, res = _post(base + "/api/cmd", {"action": "skip", "mode": "cut"}, "123456")
        assert code == 200 and res["message"] == "done" and be.calls == [("skip", {"mode": "cut"})]
        assert _post(base + "/api/cmd", {"action": "bad"}, "123456")[0] == 400
        assert "?pin=123456" in srv.pair_url()
        # guessing PINs: locked out after 10 wrong tries, even the right PIN
        for _ in range(10):
            assert _get(base + "/api/state", "000000")[0] == 401
        assert _get(base + "/api/state", "123456")[0] == 429
    finally:
        srv.stop()


def test_master_pause_fades_and_resumes():
    eng = Engine()
    y = synth_track(126.0, 20)
    a = np.stack([y, y])
    eng.load("A", a, fit_grid(y, SR, 126.0), TrackInfo(title="a", bpm=126.0, loudness=-10))
    eng.set_playing("A", True, quantize=False)
    for _ in range(20):
        eng.render(1024)
    t0 = eng.decks["A"].time
    eng.set_paused(True)
    peaks = [float(np.abs(eng.render(1024)[0]).max()) for _ in range(30)]   # ~0.7 s
    assert not eng.decks["A"].playing and peaks[-1] == 0.0
    assert peaks[0] > 0.0                      # faded, not cut
    t1 = eng.decks["A"].time
    assert t1 - t0 < 0.6
    for _ in range(10):
        eng.render(1024)
    assert eng.decks["A"].time == t1           # the track waits for us
    eng.set_paused(False)
    for _ in range(30):
        m, _ = eng.render(1024)
    assert eng.decks["A"].playing and float(np.abs(m).max()) > 0.01 and not eng.paused


def _tracks(n):
    out = {}
    for i in range(n):
        y = structured_track(126.0, SHORT, seed=i)
        a = np.stack([y, y])
        g = fit_grid(y, SR, 126.0)
        out[i] = SimpleNamespace(audio=a, grid=g, peaks=None, bands=waveform_bands(a), first_beat_t=g.first_beat,
                                 info=TrackInfo(title=f"t{i}", bpm=126.0, loudness=-10, key=21, lib_index=i))
    return out


def _master_title(eng):
    m = eng._master()
    return m.info.title if m else None


def _run(eng, seconds, until=None):
    for _ in range(int(seconds * SR / 1024)):
        eng.render(1024)
        if until is not None and until():
            return True
    return until is None


def test_skip_cut_back_play_next_and_start_from_queue():
    tracks = _tracks(4)
    eng = Engine()
    am = AutoMix(eng, lambda i: tracks[i], AutoMixSettings(mix_bars=16), synchronous_loads=True)
    am.set_queue([0, 1, 2, 3])
    am.start()
    _run(eng, 1.0)
    assert _master_title(eng) == "t0"
    am.skip("mix")                                    # blend over a few bars from the next bar
    assert _run(eng, 20, lambda: _master_title(eng) == "t1" and am.tr is None)
    assert sum(d.playing for d in eng.decks.values()) == 1
    am.skip("cut")                                    # on the next beat
    assert _run(eng, 2.0, lambda: _master_title(eng) == "t2")
    assert am.aired[-3:] == [0, 1, 2]
    assert am.previous_item() == 1
    _run(eng, 0.5)
    am.play_next(1, skip="mix")                       # "back": the prepared track (3) goes back in the queue
    assert list(am.queue)[:1] == [3]
    assert _run(eng, 20, lambda: _master_title(eng) == "t1" and am.tr is None)
    # nothing playing, decks still loaded: a new seed must win over the stale decks
    am.stop()
    for d in eng.decks.values():
        d.playing = False
    am.set_queue([2, 0])
    am.start(from_queue=True)
    _run(eng, 0.5)
    assert _master_title(eng) == "t2"


def _png(w=600, h=600) -> bytes:
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QColor, QImage

    img = QImage(w, h, QImage.Format_RGB32)
    img.fill(QColor(200, 40, 90))
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(ba.data())


def test_album_art_embedded_and_folder(tmp_data_dir):
    from mutagen.flac import FLAC, Picture
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from cta.art import get_art

    QApplication.instance() or QApplication([])      # a full QApplication: later GUI tests need one
    d1 = tmp_data_dir / "album1"
    d1.mkdir()
    f1 = str(d1 / "a.flac")
    write_audio(f1, synth_track(124.0, 2))
    f = FLAC(f1)
    pic = Picture()
    pic.type, pic.mime, pic.data = 3, "image/png", _png()
    f.add_picture(pic)
    f.save()
    data = get_art(f1)
    img = QImage()
    assert data and img.loadFromData(data) and max(img.width(), img.height()) <= 300   # thumbnail
    assert get_art(f1) == data                                                          # cached
    d2 = tmp_data_dir / "album2"
    d2.mkdir()
    f2 = str(d2 / "b.flac")
    write_audio(f2, synth_track(124.0, 2))
    assert get_art(f2) is None
    (d2 / "Cover.png").write_bytes(_png(100, 100))
    os.utime(f2, None)                                                                  # new cache key
    st = os.stat(f2)
    os.utime(f2, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    assert get_art(f2)


def test_start_at_login_linux(tmp_path, monkeypatch):
    import sys

    if sys.platform != "linux":
        pytest.skip("XDG autostart is Linux only")
    from cta import autostart

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert not autostart.is_enabled()
    autostart.set_enabled(True)
    assert autostart.is_enabled()
    txt = (tmp_path / "autostart" / "cta.desktop").read_text()
    assert "--background" in txt and "Exec=" in txt
    autostart.set_enabled(False)
    assert not autostart.is_enabled()


def test_one_broken_deck_never_silences_the_mix():
    eng = Engine()
    y = synth_track(126.0, 20)
    a = np.stack([y, y])
    for n in "AB":
        eng.load(n, a, fit_grid(y, SR, 126.0), TrackInfo(title=n, bpm=126.0, loudness=-10))
        eng.set_playing(n, True, quantize=False)
    real = eng._render_deck

    def broken(d, n):
        if d.name == "B":
            raise RuntimeError("decoder blew up")
        return real(d, n)
    eng._render_deck = broken
    for _ in range(10):
        m, _ = eng.render(1024)
    assert eng.decks["A"].playing and not eng.decks["B"].playing
    assert float(np.abs(m).max()) > 0.01                 # A still on air
    assert eng.deck_errors and "Deck B stopped" in eng.deck_errors[0]


def test_automix_off_mid_blend_settles_on_one_track():
    tracks = _tracks(2)
    eng = Engine()
    am = AutoMix(eng, lambda i: tracks[i], AutoMixSettings(mix_bars=16), synchronous_loads=True)
    am.set_queue([0, 1])
    am.start()
    _run(eng, 0.5)
    am.skip("mix")
    assert _run(eng, 10, lambda: am.tr is not None and am.tr.started)
    am.stop(settle=True)
    _run(eng, 0.2)
    playing = [d for d in eng.decks.values() if d.playing]
    assert [d.info.title for d in playing] == ["t1"] and playing[0].volume == 1.0
    assert not am.active


def test_crash_log_detects_an_unclean_exit(tmp_data_dir):
    import faulthandler

    from cta.gui import app as A

    f, crashed = A._crash_log()
    assert f is not None and not crashed
    faulthandler.disable()
    f.close()                                   # "killed": no clean-exit mark
    f, crashed = A._crash_log()
    assert crashed
    f.write(A.CRASH_MARK + "\n")
    faulthandler.disable()
    f.close()
    f, crashed = A._crash_log()
    assert not crashed
    faulthandler.disable()
    f.close()


def test_seeding_the_playing_song_never_queues_it_again():
    from cta.gui.remote import RemoteBackend

    class Lib:
        n = 5
        titles = [f"t{i}" for i in range(5)]

    tracks = _tracks(3)
    eng = Engine()
    am = AutoMix(eng, lambda i: tracks[i], AutoMixSettings(mix_bars=16), synchronous_loads=True)
    am.set_queue([0, 1])
    am.start()
    _run(eng, 0.5)
    assert _master_title(eng) == "t0"
    main = SimpleNamespace(lib=Lib(), settings=SimpleNamespace(value=lambda k, d=None: d),
                           mix_settings=lambda: None, setlist=[], _refresh_setlist=lambda: None)
    from cta.mixer import SetlistOptions

    main._setlist_options = lambda n: SetlistOptions(length=n)
    dj = SimpleNamespace(engine=eng, automix=am, main=main, played_indices=lambda: {0},
                         _show_queue=lambda: None)
    be = RemoteBackend.__new__(RemoteBackend)
    be.dj, be.eng, be.am = dj, eng, am
    import cta.mixer as M

    orig = M.build_setlist
    M.build_setlist = lambda lib, i, opts, exclude=None, **kw: [i, 2, 1]
    try:
        res = be._cmd_seed(0)
    finally:
        M.build_setlist = orig
    _run(eng, 1.0)
    assert "playing now" in res["message"]
    assert _master_title(eng) == "t0"
    nxt = [d.info.title for d in eng.decks.values() if d.loaded and not d.playing]
    assert nxt == ["t2"] and list(am.queue) == [1]


def test_live_pcm_stream_stays_live():
    from cta.dj.stream import StreamServer

    port = _free_port()
    eng = Engine()
    y = synth_track(126.0, 20)
    a = np.stack([y, y])
    eng.load("A", a, fit_grid(y, SR, 126.0), TrackInfo(title="a", bpm=126.0, loudness=-10))
    eng.set_playing("A", True, quantize=False)
    import threading
    import time as _t

    srv = StreamServer(eng, port=port, pin="1")
    srv.start()
    got = bytearray()
    done = threading.Event()
    hdr = {}

    def listen():
        req = urllib.request.Request(f"http://127.0.0.1:{port}/live.pcm?k=1")
        with urllib.request.urlopen(req, timeout=10) as r:
            hdr["sr"] = r.headers.get("X-Sample-Rate")
            while len(got) < 4 * 8192:
                got.extend(r.read(4096))
        done.set()

    threading.Thread(target=listen, daemon=True).start()
    try:
        t0 = _t.time()
        while not done.is_set() and _t.time() - t0 < 10:
            eng.render(512)
            _t.sleep(0.001)
        assert done.is_set() and hdr["sr"] == str(SR)
        pcm = np.frombuffer(bytes(got[: len(got) // 4 * 4]), "<i2").reshape(-1, 2)
        assert np.abs(pcm).max() > 1000                 # real audio, stereo 16-bit
        assert srv.listeners >= 0
    finally:
        srv.stop()


def test_now_playing_follows_the_track_you_hear():
    from cta.gui.remote import RemoteBackend

    tracks = _tracks(2)
    eng = Engine()
    am = AutoMix(eng, lambda i: tracks[i], AutoMixSettings(mix_bars=16), synchronous_loads=True)
    for n, i in (("A", 0), ("B", 1)):
        t = tracks[i]
        eng.load(n, t.audio, t.grid, t.info, bands=t.bands)
        eng.set_playing(n, True, quantize=False)
    eng.master_deck = "A"
    be = RemoteBackend.__new__(RemoteBackend)
    be.dj = SimpleNamespace(main=SimpleNamespace(lib=None))
    be.eng, be.am = eng, am
    eng.decks["A"].volume, eng.decks["B"].volume = 1.0, 0.4
    st = be.state()
    assert [d["deck"] for d in st["decks"] if d["master"]] == ["A"]
    eng.decks["A"].volume, eng.decks["B"].volume = 0.5, 1.0        # mid-blend, B now louder
    st = be.state()
    assert [d["deck"] for d in st["decks"] if d["master"]] == ["B"]
    json.dumps(st)


def test_radio_stream_is_continuous_alongside_live():
    """Real-time engine, one Radio (MP3) and one Live (PCM) listener at once:
    the MP3 must contain all the audio, with nothing dropped."""
    import io
    import threading
    import time as _t

    import av

    from cta.dj.stream import StreamServer

    port = _free_port()
    eng = Engine()
    y = synth_track(126.0, 30)
    a = np.stack([y, y])
    eng.load("A", a, fit_grid(y, SR, 126.0), TrackInfo(title="a", bpm=126.0, loudness=-10))
    eng.set_playing("A", True, quantize=False)
    srv = StreamServer(eng, port=port, pin="1")
    srv.start()
    mp3, pcm = bytearray(), bytearray()
    stop = threading.Event()

    def reader(path, buf):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}?k=1", timeout=10) as r:
            while not stop.is_set():
                chunk = r.read(2048)
                if not chunk:
                    break
                buf.extend(chunk)

    ts = [threading.Thread(target=reader, args=(p_, b), daemon=True)
          for p_, b in (("/live.mp3", mp3), ("/live.pcm", pcm))]
    for t in ts:
        t.start()
    eng.start_virtual(block=512)                    # paced like a sound card
    try:
        _t.sleep(0.5)
        n0, t0 = len(mp3), _t.time()
        _t.sleep(6.0)
        wall = _t.time() - t0
        stop.set()
    finally:
        eng.stop()
        srv.stop()
        for t in ts:
            t.join(5)
    assert not any(t.is_alive() for t in ts)       # the server closed the streams when it stopped
    assert srv.mp3_dropped == 0
    frames = 0
    with av.open(io.BytesIO(bytes(mp3[n0 - n0 % 1:])), format="mp3") as c:
        for fr in c.decode(audio=0):
            frames += fr.samples
    got = frames / SR
    # the MP3 holds (nearly) every second that went by: no gaps
    assert got > wall * 0.9, (got, wall)
    assert len(pcm) > 4 * SR * 5                    # live kept flowing too
