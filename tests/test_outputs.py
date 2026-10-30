"""Output plumbing: second sound card for the headphones, alignment delays,
Bluetooth latency calibration, the audio settings dialog."""
import os

import numpy as np
import pytest

from cta.dj.outputs import Delay, Ring, estimate_latency, is_bluetooth

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_ring_keeps_two_clocks_together():
    r = Ring(target=1024, capacity=8192)
    x = np.arange(4000, dtype=np.float32)[:, None].repeat(2, 1)
    r.write(x[:1000])
    out = r.read(600)
    assert np.array_equal(out[:, 0], x[:600, 0])
    out = r.read(600)                                   # only 400 left: rest silence
    assert np.array_equal(out[:400, 0], x[600:1000, 0]) and not out[400:].any() and r.underruns == 1
    r.write(x)                                          # far too much queued: skip ahead to the target
    assert r.fill == 1024 and r.skips == 1
    assert r.read(1)[0, 0] == x[4000 - 1024, 0]


def test_delay_line_shifts_by_exact_frames_across_blocks():
    d = Delay(44100, max_s=0.5)
    d.frames = 700
    sig = np.zeros((2, 3000), np.float32)
    sig[:, 10] = 1.0
    out = np.concatenate([d.process(sig[:, i:i + 512]) for i in range(0, 3000, 512)], axis=1)
    assert int(np.argmax(out[0])) == 710 and out[0, 710] == 1.0


def test_tap_calibration_finds_the_delay():
    rng = np.random.default_rng(1)
    clicks = list(np.arange(0, 20, 0.6))
    taps = [c + 0.23 + rng.normal(0, 0.015) for c in clicks[2:24]]
    est = estimate_latency(taps, clicks)
    assert est is not None and abs(est - 0.25) < 0.02          # 230 ms + typical anticipation
    assert estimate_latency(taps[:5], clicks) is None           # too few
    wild = [c + rng.uniform(0, 0.55) for c in clicks[2:24]]
    assert estimate_latency(wild, clicks) is None               # all over the place


def test_bluetooth_names():
    assert is_bluetooth("AirPods Pro (Core Audio, 2 ch)") and is_bluetooth("JBL Flip 6")
    assert is_bluetooth("Headphones (WH-1000XM4 Hands-Free AG Audio)")
    assert not is_bluetooth("DDJ-T1 (ASIO, 4 ch)") and not is_bluetooth("MacBook Pro Speakers")


def test_engine_lines_up_master_and_headphones():
    from cta.dj.engine import SR, Engine

    eng = Engine()
    eng.phones_chans = (2, 3)
    eng.master_delay_s, eng.phones_delay_s = 0.21, 0.01           # Bluetooth master, wired phones
    eng.update_alignment()
    assert eng._pdelay.frames == round(0.20 * SR) and eng._mdelay.frames == 0
    assert abs(eng.display_latency - 0.21) < 1e-9
    m = np.zeros((2, 2048), np.float32)
    p = np.zeros((2, 2048), np.float32)
    m[:, 0] = p[:, 0] = 1.0
    out_m, out_p = eng._route(m, p, 2048)
    assert out_m[0, 0] == 1.0 and not out_p[0, :500].any()          # phones held back
    eng.phones_chans = None
    eng.update_alignment()
    assert eng._pdelay.frames == 0 and abs(eng.display_latency - 0.21) < 1e-9


def test_audio_dialog_scans_live_and_suggests_bluetooth(tmp_path, monkeypatch):
    """Opening Audio and Rescan never stop the music: devices are found by a
    helper process. New Bluetooth speakers can join the master, in sync."""
    pytest.importorskip("PySide6.QtWidgets")
    import time
    from types import SimpleNamespace

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    import cta.dj.outputs as O
    import cta.gui.dj_window as DW
    from cta.dj.engine import Engine

    app = QApplication.instance() or QApplication([])
    known = [(0, "MacBook Pro Speakers (Core Audio, 2 ch)", 2), (3, "DDJ-T1 (Core Audio, 4 ch)", 4)]
    monkeypatch.setattr(DW, "list_output_devices", lambda: list(known))
    scanned = [{"label": lb, "name": lb.rsplit(" (", 1)[0], "channels": ch, "bluetooth": O.is_bluetooth(lb)}
               for _i, lb, ch in known]
    monkeypatch.setattr(O, "scan_devices", lambda timeout=15.0: list(scanned))
    monkeypatch.setattr(O, "rescan_devices", lambda: (_ for _ in ()).throw(AssertionError("audio stopped")))
    eng = Engine()
    stops = []
    monkeypatch.setattr(eng, "stop", lambda: stops.append(1))
    st = QSettings(str(tmp_path / "a.ini"), QSettings.IniFormat)
    dlg = DW.AudioSettings(st, None, dj=SimpleNamespace(engine=eng))

    def settle():
        t0 = time.time()
        while time.time() - t0 < 1.0:
            app.processEvents()
            time.sleep(0.01)
    settle()
    assert not stops                                              # the music never stopped
    assert [dlg.dev.itemText(i) for i in range(dlg.dev.count())][1:] == [lb for _i, lb, _c in known]
    dlg.dev.setCurrentIndex(2)                                    # DDJ-T1: 4 channels -> two pairs
    assert [dlg.master.itemText(i) for i in range(dlg.master.count())] == ["1-2", "3-4"]
    # speakers connected while CTA runs: Rescan finds them (helper process, music untouched)
    for lb in ("AirPods Pro (Core Audio, 2 ch)", "JBL Flip 6 (Core Audio, 2 ch)"):
        scanned.append({"label": lb, "name": lb.rsplit(" (", 1)[0], "channels": 2, "bluetooth": True})
    dlg.rescan(force=True)
    settle()
    assert not stops
    ix = dlg.pdev.findText("AirPods Pro (Core Audio, 2 ch)")
    assert ix > 0
    dlg.pdev.setCurrentIndex(ix)
    assert not dlg.p_auto.isChecked() and dlg.pdelay.value() == 200 and "Bluetooth" in dlg.p_info.text()
    # the JBL joins the master as an extra output
    dlg.extra_add.setCurrentIndex(dlg.extra_add.findData("JBL Flip 6 (Core Audio, 2 ch)"))
    dlg._extra_add()
    assert [e["label"] for e in dlg.extras] == ["JBL Flip 6 (Core Audio, 2 ch)"]
    dlg.e_auto.setChecked(False)
    dlg.edelay.setValue(240)
    dlg.phones.setCurrentText("1-2")
    dlg.accept()
    import json

    assert st.value("dj_phones_device_name") == "AirPods Pro (Core Audio, 2 ch)"
    assert int(st.value("dj_phones_delay_ms")) == 200 and st.value("dj_master_delay_ms") == ""
    assert json.loads(st.value("dj_extra_outputs")) == [{"label": "JBL Flip 6 (Core Audio, 2 ch)", "delay_ms": 240}]
    assert dlg.restart_needed and dlg.reinit_needed            # headphones moved to a device found by the scan


def test_tap_dialog_measures_and_restores(monkeypatch):
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    import cta.gui.dj_window as DW
    from cta.dj.engine import Engine

    QApplication.instance() or QApplication([])
    eng = Engine()
    eng._mdelay.frames = 123
    dlg = DW.TapCalibration(eng, "master")
    assert eng.click_target == "master" and eng._mdelay.frames == 0
    clicks = [100.0 + 0.6 * k for k in range(40)]
    eng.click_times[:] = clicks
    t = iter([c + 0.18 for c in clicks[1:30]])
    monkeypatch.setattr(dlg._t, "monotonic", lambda: next(t))
    for _ in range(20):
        dlg.tap()
    assert dlg.result_ms is not None and abs(dlg.result_ms - 200) < 5 and dlg.ok.isEnabled()
    dlg.reject()
    assert eng.click_target is None and eng._mdelay.frames == 123


def test_extra_output_helper_receives_the_mix(tmp_path):
    """A second master output (e.g. another Bluetooth speaker) runs in its own
    helper process, fed live by the engine; outputs line up to the slowest."""
    import json
    import sys
    import time

    from cta.dj.engine import SR, Engine
    from cta.dj.outputs import ExtraOutput

    stats = str(tmp_path / "stats.json")
    eo = ExtraOutput("JBL Flip 6 (Core Audio, 2 ch)", cmd=[sys.executable, "-m", "cta"], null=True, stats=stats)
    eng = Engine()
    eng.master_delay_s = 0.02
    eng.add_output(eo)
    assert abs(eng.display_latency - 0.2) < 1e-9                  # Bluetooth guess is the slowest
    assert eng._mdelay.frames == round(0.18 * SR) and eo.align.frames == 0
    blk = np.zeros((2, 512), np.float32)
    t0 = time.monotonic()
    for k in range(200):                                          # ~2.3 s of audio, in real time
        eng._route(blk.copy(), blk.copy(), 512)
        time.sleep(max(0.0, t0 + (k + 1) * 512 / SR - time.monotonic()))
    t0 = time.time()
    while not eo.q.empty() and time.time() - t0 < 10:
        time.sleep(0.05)
    eng.remove_output(eo.label)
    assert not eng.extra_outputs and eng.display_latency == 0.02
    got = json.load(open(stats))["frames"]
    assert got == 200 * 512, got


def test_devices_command_writes_json(tmp_path):
    import json
    import subprocess
    import sys

    out = tmp_path / "d.json"
    subprocess.run([sys.executable, "-m", "cta", "devices", "--out", str(out)], check=True, timeout=60)
    assert isinstance(json.loads(out.read_text()), list)


def test_speakers_rejoin_and_new_bluetooth_is_offered(tmp_path, monkeypatch):
    pytest.importorskip("PySide6.QtWidgets")
    import json
    from types import SimpleNamespace

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QLabel, QPushButton

    import cta.dj.outputs as O
    import cta.gui.dj_window as DW
    from cta.dj.engine import Engine

    QApplication.instance() or QApplication([])

    class FakeOut:
        def __init__(self, label, delay_s=None, block=512, **kw):
            self.label, self.delay_s, self.alive = label, delay_s, True
            self.align = O.Delay(44100)

        effective_delay = O.ExtraOutput.effective_delay

        def feed(self, b):
            pass

        def close(self):
            self.alive = False
    monkeypatch.setattr(O, "ExtraOutput", FakeOut)
    st = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    st.setValue("dj_extra_outputs", json.dumps([{"label": "JBL Flip 6 (Core Audio, 2 ch)", "delay_ms": 240}]))
    eng = Engine()
    dj = SimpleNamespace(engine=eng, settings=st, main=SimpleNamespace(status_label=QLabel()),
                         offer_btn=QPushButton())
    for name in ("wanted_extras", "apply_extras", "_on_devices", "_accept_offer"):
        setattr(dj, name, getattr(DW.DJPanel, name).__get__(dj))
    rec = lambda lb, bt=True: {"label": lb, "name": O.base_name(lb), "channels": 2, "bluetooth": bt}  # noqa: E731
    speakers = rec("MacBook Pro Speakers (Core Audio, 2 ch)", False)
    dj._on_devices([speakers])                                      # JBL not around: nothing started
    assert not eng.extra_outputs
    dj._on_devices([speakers, rec("JBL Flip 6 (Core Audio, 2 ch)")])  # it connects: rejoins by itself
    assert [o.label for o in eng.extra_outputs] == ["JBL Flip 6 (Core Audio, 2 ch)"]
    assert abs(eng.display_latency - 0.24) < 1e-9 and "in sync" in dj.main.status_label.text()
    eng.extra_outputs[0].alive = False                              # switched off
    dj._on_devices([speakers])
    assert not eng.extra_outputs and "disconnected" in dj.main.status_label.text()
    dj._on_devices([speakers, rec("Party Speaker BT (Core Audio, 2 ch)")])   # a new Bluetooth device
    assert not dj.offer_btn.isHidden() and "Party Speaker BT" in dj.offer_btn.text()
    dj._accept_offer()
    assert "Party Speaker BT (Core Audio, 2 ch)" in [o.label for o in eng.extra_outputs]
    assert len(dj.wanted_extras()) == 2
