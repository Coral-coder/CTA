"""Decode -> analyse -> database -> library -> mixer -> export, end to end."""
import os

import numpy as np
import pytest
from conftest import synth_track, write_audio

from cta import audio, model
from cta.db import Database
from cta.export import export
from cta.library import Library
from cta.mixer import MixSettings, SetlistOptions, build_setlist, suggest_next
from cta.scanner import Scanner, ScanSettings


def test_decode_formats(tmp_path):
    y = synth_track(128, 12)
    for codec, ext in (("flac", "flac"), ("pcm_s16le", "wav")):
        p = str(tmp_path / f"t.{ext}")
        write_audio(p, y, codec=codec if codec == "flac" else "pcm_s16le")
        d = audio.decode(p)
        assert abs(d.duration - 12) < 0.2
        assert abs(d.y.size / 22050 - 12) < 0.2
        assert abs(d.y16.size / 16000 - 12) < 0.2


def test_decode_garbage(tmp_path):
    p = tmp_path / "broken.mp3"
    p.write_bytes(b"\x00not audio at all" * 100)
    with pytest.raises(audio.DecodeError):
        audio.decode(str(p))


def _make_library(folder, specs):
    os.makedirs(folder, exist_ok=True)
    for name, bpm, chord in specs:
        write_audio(os.path.join(folder, name), synth_track(bpm, 25, chord=chord))


SPECS = [
    ("a_128_Am.flac", 128.0, (57, 60, 64)),
    ("b_126_Em.flac", 126.0, (52, 55, 59)),
    ("c_130_C.flac", 130.0, (60, 64, 67)),
    ("d_174_Dm.flac", 174.0, (62, 65, 69)),
    ("e_90_Am.flac", 90.0, (57, 60, 64)),
]


def test_scan_and_mix(tmp_data_dir):
    music = str(tmp_data_dir / "music")
    _make_library(music, SPECS)
    (tmp_data_dir / "music" / "broken.mp3").write_bytes(b"junk" * 1000)
    (tmp_data_dir / "music" / "._a_128_Am.flac").write_bytes(b"appledouble")

    db = Database(str(tmp_data_dir / "lib.sqlite3"))
    p = Scanner(db, ScanSettings(workers=2, use_ai=False)).run([music])
    assert p.phase == "done"
    assert p.done == 6 and p.failed == 1

    lib = Library(db)
    assert lib.n == 5 and len(lib.errors) == 1
    bpms = {os.path.basename(lib.paths[i]): lib.bpm[i] for i in range(lib.n)}
    for name, bpm, _ in SPECS:
        assert abs(bpms[name] - bpm) < 0.1

    # unchanged files are skipped on rescan
    p2 = Scanner(db, ScanSettings(workers=2, use_ai=False)).run([music])
    assert p2.skipped == 6 and p2.total == 0      # includes the known-broken file

    a = lib.paths.index(os.path.join(music, "a_128_Am.flac"))
    sug = suggest_next(lib, a, MixSettings(max_pitch_pct=6))
    names = [os.path.basename(lib.paths[s.idx]) for s in sug]
    assert "a_128_Am.flac" not in names
    assert "d_174_Dm.flac" not in names       # outside the pitch range
    assert names[0] == "b_126_Em.flac"        # closest tempo + adjacent key

    seq = build_setlist(lib, a, SetlistOptions(length=3))
    assert seq[0] == a and len(set(seq)) == len(seq) == 3

    # removed files are hidden from the library
    os.remove(os.path.join(music, "e_90_Am.flac"))
    Scanner(db, ScanSettings(workers=1, use_ai=False)).run([music])
    assert Library(db).n == 4

    for ext in ("m3u8", "xml", "csv"):
        out = str(tmp_data_dir / f"set.{ext}")
        export(lib, seq, out)
        txt = open(out, encoding="utf-8-sig").read()
        assert "a_128_Am" in txt


@pytest.mark.skipif(model.find_model() is None and not os.environ.get("CTA_TEST_MODEL"),
                    reason="AI model not available offline")
def test_ai_model(tmp_data_dir):
    path = model.ensure_model()
    assert path
    m = model.StyleModel(path, "cpu")
    patches = np.random.default_rng(0).random((5, 128, 96), dtype=np.float32) * 3
    out = m.track_vectors([patches[:2], patches[2:]])
    assert len(out) == 2
    assert out[0][0].shape == (400,) and out[0][1].shape == (1280,)
    assert len(model.labels()) == 400
