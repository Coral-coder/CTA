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


def test_audio_dialog_lists_every_device_and_suggests_bluetooth(tmp_path, monkeypatch):
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    import cta.dj.outputs as O
    import cta.gui.dj_window as DW
    from cta.dj.engine import Engine

    QApplication.instance() or QApplication([])
    devices = [(0, "MacBook Pro Speakers (Core Audio, 2 ch)", 2), (3, "DDJ-T1 (Core Audio, 4 ch)", 4)]
    monkeypatch.setattr(DW, "list_output_devices", lambda: list(devices))
    rescans = []
    monkeypatch.setattr(O, "rescan_devices", lambda: rescans.append(1))
    from types import SimpleNamespace

    eng = Engine()
    dj = SimpleNamespace(engine=eng)
    st = QSettings(str(tmp_path / "a.ini"), QSettings.IniFormat)
    dlg = DW.AudioSettings(st, None, dj=dj)
    assert rescans == [1] and dlg.restart_needed                 # idle: devices re-read on open
    assert [dlg.dev.itemText(i) for i in range(dlg.dev.count())][1:] == [d[1] for d in devices]
    dlg.dev.setCurrentIndex(2)                                    # DDJ-T1: 4 channels -> two pairs
    assert [dlg.master.itemText(i) for i in range(dlg.master.count())] == ["1-2", "3-4"]
    # a Bluetooth headset connected while CTA runs: Rescan finds it
    devices.append((5, "AirPods Pro (Core Audio, 2 ch)", 2))
    dlg.rescan(force=True)
    ix = dlg.pdev.findText("AirPods Pro (Core Audio, 2 ch)")
    assert ix > 0
    dlg.pdev.setCurrentIndex(ix)
    assert not dlg.p_auto.isChecked() and dlg.pdelay.value() == 200 and "Bluetooth" in dlg.p_info.text()
    dlg.phones.setCurrentText("1-2")
    dlg.save()
    assert st.value("dj_phones_device_name") == "AirPods Pro (Core Audio, 2 ch)"
    assert int(st.value("dj_phones_delay_ms")) == 200 and st.value("dj_master_delay_ms") == ""
    dlg.close()


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
