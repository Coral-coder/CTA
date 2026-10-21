import numpy as np

from cta.db import Database, vec_to_blob
from cta.library import Library
from cta.mixer import MixSettings, SetlistOptions, build_setlist, energy_target, suggest_next


def _fake_db(path, n=400, seed=0):
    rng = np.random.default_rng(seed)
    db = Database(path)
    rows = []
    for i in range(n):
        emb = rng.random(1280).astype(np.float32)
        sv = rng.dirichlet(np.ones(400) * 0.05).astype(np.float32)
        rows.append(dict(path=f"/m/{i}.mp3", mtime=0, size=0, title=f"T{i}", artist=f"A{i % 50}", album="",
                         tag_genre="", tag_bpm=None, tag_key=-1, duration=300, bpm=float(rng.uniform(118, 132)),
                         bpm_conf=1, key=int(rng.integers(0, 24)), key_conf=1, loudness=float(rng.uniform(-14, -6)),
                         centroid=2000, onset_rate=3, genre="House", styles=None, style_vec=vec_to_blob(sv),
                         embedding=vec_to_blob(emb), status="ok", error=None, analyzed_at=0, version=1))
    db.upsert_many(rows)
    return db


def test_suggestions_respect_limits(tmp_path):
    lib = Library(_fake_db(str(tmp_path / "x.db")))
    s = MixSettings(max_pitch_pct=3, strict_key=True)
    for a in range(0, 50, 7):
        for sug in suggest_next(lib, a, s, n=30):
            assert abs(sug.pitch_pct) <= 3 + 1e-9
            assert sug.key >= 0.5
            assert sug.idx != a


def test_setlist_unique_and_ramp(tmp_path):
    lib = Library(_fake_db(str(tmp_path / "x.db")))
    seed = int(np.argmin(np.abs(lib.bpm - 120)))
    seq = build_setlist(lib, seed, SetlistOptions(length=25, end_bpm=130, mix=MixSettings(max_pitch_pct=4)))
    assert len(seq) == 25 and len(set(seq)) == 25 and seq[0] == seed
    assert lib.bpm[seq[-1]] > lib.bpm[seq[0]] + 4
    for a, b in zip(seq, seq[1:]):
        assert abs(lib.bpm[a] / lib.bpm[b] - 1) * 100 <= 4 + 1e-6


def test_setlist_variety_changes_result(tmp_path):
    lib = Library(_fake_db(str(tmp_path / "x.db")))
    a = build_setlist(lib, 0, SetlistOptions(length=15, variety=1.0, seed=1))
    b = build_setlist(lib, 0, SetlistOptions(length=15, variety=1.0, seed=2))
    assert a != b


def test_energy_curves():
    assert energy_target("steady", 5, 0.5) == 5
    assert energy_target("build", 5, 1.0) == 9
    assert energy_target("cooldown", 5, 1.0) == 1
    assert energy_target("peak", 5, 0.65) == 9
