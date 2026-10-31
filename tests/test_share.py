"""The library bridge: the home CTA shares its library, the CTA you take
along mirrors the catalogue and downloads what it plays."""
import os
import ssl

import numpy as np
import pytest
from conftest import synth_track, write_audio

from cta.db import Database
from cta.library import Library
from cta.scanner import Scanner, ScanSettings
from cta.share import (
    Downloader,
    ShareClient,
    ShareServer,
    forget_library,
    make_code,
    mirror_catalog,
    mirror_dir,
    parse_code,
)


def _home(tmp_path):
    music = tmp_path / "home-music"
    music.mkdir()
    for n, bpm in enumerate((124.0, 128.0)):
        write_audio(str(music / f"home{n}.flac"), synth_track(bpm, 12, chord=(57, 60, 64)))
    db = Database(str(tmp_path / "home.sqlite3"))
    Scanner(db, ScanSettings(workers=1, use_ai=False)).run([str(music)])
    p0 = next(r["path"] for r in db.all_tracks() if r["path"].endswith("home0.flac"))
    db.set_markers(p0, 4.0, 8.0, 10.0)
    db.set_cues(p0, [1.5, None, 3.0])
    db.set_grid(p0, 0.25, 124.0, version=2, phrase_origin=3)      # a grid refined (or edited) at home
    return db


def test_pairing_code_round_trip():
    code = make_code("home.example.org", 8008, "k" * 43, "ab" * 32)
    d = parse_code(code)
    assert d == {"host": "home.example.org", "port": 8008, "key": "k" * 43, "fingerprint": "ab" * 32}
    assert parse_code("CTA-SHARE|[::1]:9|key|")["host"] == "::1"
    with pytest.raises(ValueError):
        parse_code("nope")


@pytest.mark.parametrize("tls", [True, False])
def test_home_library_is_mirrored_and_downloaded_on_demand(tmp_path, tls):
    home_db = _home(tmp_path)
    home_dir = tmp_path / "home-data"
    home_dir.mkdir()
    srv = ShareServer(home_db, str(home_dir), port=0, name="Studio PC", tls=tls)
    srv.start()
    try:
        assert bool(srv.tls) == tls
        code = srv.code("127.0.0.1")
        info = parse_code(code)
        client = ShareClient(info["host"], info["port"], info["key"], info["fingerprint"])
        meta = client.info()
        assert meta["name"] == "Studio PC" and meta["count"] == 2 and meta["tls"] == tls
        # the wrong key is refused, a wrong certificate fingerprint never connects
        with pytest.raises(ConnectionError):
            ShareClient(info["host"], info["port"], "wrong", info["fingerprint"]).info()
        if tls:
            with pytest.raises(ssl.SSLError):
                ShareClient(info["host"], info["port"], info["key"], "00" * 32).info()
        # the away machine: mirror the catalogue into its own library
        away_db = Database(str(tmp_path / "away.sqlite3"))
        away_dir = str(tmp_path / "away-data")
        stats = mirror_catalog(away_db, away_dir, meta["library"], client.catalog())
        assert stats == {"added": 2, "updated": 0, "removed": 0}
        lib = Library(away_db)
        assert lib.n == 2 and not lib.local.any() and all(r for r in lib.remote)
        assert sorted(np.round(lib.bpm, 0)) == [124.0, 128.0]           # the analysis came along
        i0 = next(i for i in range(lib.n) if lib.paths[i].endswith("home0.flac"))
        assert away_db.get_markers(lib.paths[i0]) == (4.0, 8.0) and away_db.get_out_end(lib.paths[i0]) == 10.0
        assert away_db.get_cues(lib.paths[i0])[0] == 1.5
        assert away_db.get_grid(lib.paths[i0]) == (0.25, 124.0, 2, 3)
        assert lib.paths[i0].startswith(mirror_dir(away_dir, meta["library"]))
        # a second sync is an update, not a duplicate
        assert mirror_catalog(away_db, away_dir, meta["library"], client.catalog())["updated"] == 2
        # a deck asks for the file: the downloader fetches it (and resumes a half file)
        dl = Downloader(away_db, lambda lib_id: client if lib_id == meta["library"] else None, away_dir)
        try:
            path = lib.paths[i0]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", "wb") as f:                       # a download that was cut off
                f.write(b"\0" * 1000)
            assert dl.ensure(path) == path and os.path.isfile(path) and not os.path.exists(path + ".part")
            home_path = next(r["path"] for r in home_db.all_tracks() if r["path"].endswith("home0.flac"))
            assert os.path.getsize(path) == os.path.getsize(home_path)
            with open(path, "rb") as a, open(home_path, "rb") as b:
                assert a.read()[1000:] == b.read()[1000:]              # bytes after the resume point match
            assert Library(away_db).local[i0]
            assert srv.served_bytes > 0
        finally:
            dl.close()
        # forgetting the home library removes its rows (and, if asked, its files)
        assert forget_library(away_db, away_dir, meta["library"], delete_files=True) == 2
        assert Library(away_db).n == 0 and not os.path.exists(path)
    finally:
        srv.stop()
