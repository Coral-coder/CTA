"""Command line interface. Running without a command starts the GUI.

    cta scan <folder> [<folder>...]     add folders and analyse them
    cta list [--search text]            show the analysed library
    cta next <track>                    best next tracks to mix
    cta setlist <track> -n 20 -o set.m3u8
    cta selftest                        verify decoding, DSP and the AI model
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from . import __version__


def _setup_logging(verbose: bool = False):
    from . import paths

    handlers = [logging.StreamHandler(sys.stderr)] if sys.stderr else []
    try:
        handlers.append(logging.FileHandler(paths.log_path(), encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)


def _open_db(args):
    from . import paths
    from .db import Database

    return Database(args.db or paths.db_path())


def _find_track(lib, query: str) -> int:
    if os.path.exists(query):
        p = os.path.normpath(os.path.abspath(query))
        for i, tp in enumerate(lib.paths):
            if os.path.normpath(tp) == p:
                return i
    hits = lib.search(query)
    if len(hits) == 0:
        raise SystemExit(f"no track matches '{query}'")
    return int(hits[0])


def _fmt(lib, i) -> str:
    t = lib.track(i)
    genre = (t.genre or t.tag_genre)[:18]
    return f"{t.bpm:7.2f}  {t.key_label:9s}  E{t.energy:4.1f}  {genre:18s}  {t.artist} - {t.title}"


def cmd_scan(args):
    from .scanner import Scanner, ScanSettings

    db = _open_db(args)
    folders = [os.path.normpath(os.path.abspath(f)) for f in args.folders] or db.folders()
    for f in folders:
        if not os.path.isdir(f):
            raise SystemExit(f"not a folder: {f}")
        db.add_folder(f)
    st = ScanSettings(workers=args.workers, use_ai=not args.no_ai, force=args.force,
                      ai_device="cpu" if args.cpu else "auto", retry_failed=args.retry_failed,
                      min_bpm=args.min_bpm, max_bpm=args.max_bpm)
    last = [0.0]

    def prog(p):
        now = time.time()
        if now - last[0] > 1.0 or p.phase in ("done", "stopped", "error"):
            last[0] = now
            extra = f" on {p.device}" if p.device else ""
            if p.phase == "analyzing":
                print(f"\r[{p.done}/{p.total}] {p.rate:5.1f} files/s  ETA {int(p.eta)}s{extra}   ",
                      end="", flush=True)
            elif p.message:
                print(f"\n{p.message}", flush=True)

    sc = Scanner(db, st, on_progress=prog)
    try:
        res = sc.run(folders)
    except KeyboardInterrupt:
        sc.stop()
        res = sc.progress
    print()
    return 0 if res.phase in ("done", "stopped") else 1


def cmd_list(args):
    from .library import Library

    lib = Library(_open_db(args))
    idx = lib.search(args.search or "")
    for i in idx[: args.limit]:
        print(_fmt(lib, int(i)))
    print(f"{len(idx)} tracks ({lib.n} analysed, {len(lib.errors)} failed)")
    return 0


def _mix_settings(args):
    from .mixer import MixSettings

    return MixSettings(max_pitch_pct=args.pitch, strict_key=args.strict_key)


def cmd_next(args):
    from .library import Library
    from .mixer import suggest_next

    lib = Library(_open_db(args))
    a = _find_track(lib, args.track)
    print("NOW:  " + _fmt(lib, a))
    t0 = time.perf_counter()
    sug = suggest_next(lib, a, _mix_settings(args), n=args.limit)
    dt = (time.perf_counter() - t0) * 1000
    for s in sug:
        print(f"{s.score:5.2f}  {s.pitch_pct:+5.1f}%  " + _fmt(lib, s.idx))
    print(f"({dt:.1f} ms)")
    return 0


def cmd_setlist(args):
    from .export import export
    from .library import Library
    from .mixer import SetlistOptions, build_setlist

    lib = Library(_open_db(args))
    a = _find_track(lib, args.track)
    opt = SetlistOptions(length=args.length, end_bpm=args.end_bpm, energy_curve=args.curve,
                         variety=args.variety, mix=_mix_settings(args))
    t0 = time.perf_counter()
    seq = build_setlist(lib, a, opt)
    dt = (time.perf_counter() - t0) * 1000
    for n, i in enumerate(seq, 1):
        print(f"{n:3d}. " + _fmt(lib, i))
    print(f"({dt:.0f} ms)")
    if args.output:
        export(lib, seq, args.output)
        print(f"saved {args.output}")
    return 0


def cmd_selftest(args):
    """End-to-end check used by CI on the frozen executable."""
    import tempfile

    import av
    import numpy as np

    from . import dsp, model
    from .db import Database
    from .library import Library
    from .mixer import MixSettings, SetlistOptions, build_setlist, suggest_next
    from .scanner import Scanner, ScanSettings

    ok = True
    tmp = tempfile.mkdtemp(prefix="cta-selftest-")
    print(f"PyAV {av.__version__}, numpy {np.__version__}")
    try:
        import onnxruntime as ort
        print(f"onnxruntime {ort.__version__}, providers: {ort.get_available_providers()}")
    except Exception as e:  # noqa: BLE001
        print(f"onnxruntime import FAILED: {e}")
        ok = False

    sr = 44100
    for n, (bpm, root) in enumerate([(128.0, 57), (124.0, 60), (174.0, 62)]):
        t = np.arange(int(sr * 40)) / sr
        y = np.zeros_like(t)
        beat = 60.0 / bpm
        for b in range(int(40 / beat)):
            s = int(b * beat * sr)
            tt = np.arange(int(0.15 * sr)) / sr
            seg = np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 20)
            y[s:s + seg.size] += seg[: max(0, min(seg.size, y.size - s))]
        for m in (root, root + 3, root + 7):
            y += 0.08 * np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * t)
        y = (y / np.abs(y).max() * 0.8 * 32767).astype(np.int16)
        path = os.path.join(tmp, f"test{n}.flac")
        with av.open(path, "w") as c:
            st = c.add_stream("flac", rate=sr, layout="mono")
            fr = av.AudioFrame.from_ndarray(y[None, :], format="s16", layout="mono")
            fr.sample_rate = sr
            for p in st.encode(fr):
                c.mux(p)
            for p in st.encode(None):
                c.mux(p)

    db = Database(os.path.join(tmp, "test.sqlite3"))
    prog = Scanner(db, ScanSettings(workers=2, use_ai=not args.no_ai)).run([tmp])
    print(f"scan: {prog.phase} done={prog.done} failed={prog.failed} device={prog.device or 'n/a'}")
    lib = Library(db)
    if lib.n != 3 or prog.failed:
        print("FAIL: expected 3 analysed tracks")
        for r in lib.errors:
            print("  ", r["path"], r["error"])
        ok = False
    for i in range(lib.n):
        print("   " + _fmt(lib, i))
    want = {"test0.flac": 128.0, "test1.flac": 124.0, "test2.flac": 174.0}
    for i in range(lib.n):
        exp = want.get(os.path.basename(lib.paths[i]))
        if exp and abs(lib.bpm[i] - exp) > 0.5:
            print(f"FAIL: BPM {lib.bpm[i]} != {exp}")
            ok = False
    if not args.no_ai:
        if model.find_model() is None:
            print("FAIL: AI model not available")
            ok = False
        elif not lib.has_ai.all():
            print("FAIL: AI vectors missing")
            ok = False
    if lib.n:
        suggest_next(lib, 0, _mix_settings(args))
        seq = build_setlist(lib, 0, SetlistOptions(length=3, mix=MixSettings(max_pitch_pct=40)))
        if len(seq) != 3:
            print("FAIL: setlist length")
            ok = False
    # DJ engine: key-locked time-stretch, EQ filters, audio + MIDI libraries
    try:
        from .dj.beatgrid import BeatGrid
        from .dj.engine import SR as DSR
        from .dj.engine import Engine, TrackInfo

        eng = Engine()
        tone = (0.4 * np.sin(2 * np.pi * 440 * np.arange(DSR * 6) / DSR)).astype(np.float32)
        eng.load("A", np.stack([tone, tone]), BeatGrid(120.0, 0.0, 6.0), TrackInfo(loudness=-10))
        eng.set("A", keylock=True, tempo=1.0, tempo_range=0.08, eq=(0.5, 1.0, 1.0))
        eng.set_playing("A", True)
        out = np.concatenate([eng.render(512)[0] for _ in range(int(3 * DSR / 512))], axis=1)[0, DSR:]
        f = np.fft.rfftfreq(out.size, 1 / DSR)[np.argmax(np.abs(np.fft.rfft(out * np.hanning(out.size))))]
        print(f"dj engine: key lock at +8% -> {f:.1f} Hz (expect 440)")
        if abs(f - 440) > 3:
            print("FAIL: key lock")
            ok = False
        import rtmidi  # noqa: F401
        import sounddevice  # noqa: F401
        print("audio/MIDI libraries: OK")
        ctx = av.CodecContext.create("libmp3lame", "w")
        ctx.sample_rate, ctx.layout, ctx.format, ctx.bit_rate = DSR, "stereo", "fltp", 192000
        ctx.open()
        fr = av.AudioFrame.from_ndarray(np.zeros((2, ctx.frame_size or 1152), np.float32), format="fltp",
                                        layout="stereo")
        fr.sample_rate = DSR
        list(ctx.encode(fr))
        print("live stream encoder (MP3): OK")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: DJ engine: {e}")
        ok = False
    try:
        import socket
        import urllib.request

        import qrcode
        from PySide6.QtNetwork import QLocalServer  # noqa: F401  (single-instance check)

        from .dj.engine import Engine
        from .dj.stream import StreamServer

        q = qrcode.QRCode()
        q.add_data("http://192.168.1.2:8000/?pin=123456")
        q.make(fit=True)
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        srv = StreamServer(Engine(), port=port, pin="1234")
        srv.start()
        try:
            page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
            icon = urllib.request.urlopen(f"http://127.0.0.1:{port}/apple-touch-icon.png", timeout=5).read()
        finally:
            srv.stop()
        if b"CTA live remote" not in page or icon[:4] != b"\x89PNG":
            raise RuntimeError("phone page or home-screen icon missing")
        print(f"phone remote / mix server: OK (QR {len(q.get_matrix())} modules)")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: phone remote: {e}")
        ok = False
    _ = dsp  # imported to verify packaging
    print("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def cmd_devices(args):
    import json

    from .dj.outputs import device_records

    try:
        recs = device_records()
    except Exception as e:  # noqa: BLE001 - no audio system at all
        logging.getLogger(__name__).warning("device scan failed: %s", e)
        recs = []
    text = json.dumps(recs, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        print(text)
    return 0


def cmd_output_helper(args):
    from .dj.outputs import run_output_helper

    a, b = (int(x) for x in args.chans.split(","))
    return run_output_helper(args.device, (a, b), args.block, null=args.null, stats=args.stats)


def cmd_update(args):
    from . import updater

    if args.from_zip:
        # stage a local build (manual install / CI test of the update path)
        new_root = updater.prepare_from_zip(args.from_zip, args.expect_version)
        updater.launch_swap(new_root, relaunch=not args.no_relaunch)
        print(f"update staged from {args.from_zip}; it is applied as soon as this process exits")
        return 0
    print(f"CTA {__version__} - checking {updater.REPO} for updates…")
    rel = updater.check_latest()
    if not updater.is_newer(rel.version) and not args.force:
        print(f"You are up to date (latest release {rel.version}).")
        return 0
    print(f"New version available: {rel.version}  ({rel.url})")
    if args.check:
        return 0
    ok, why = updater.can_self_update()
    if not ok:
        print(f"Cannot update automatically: {why}")
        return 1
    if not rel.installable:
        print("This release has no build for your platform.")
        return 1

    def prog(done, total):
        if total:
            print(f"\rdownloading {done * 100 // total}%", end="", flush=True)

    new_root = updater.prepare_update(rel, prog)
    print("\nverified; installing - CTA will restart" if not args.no_relaunch else "\nverified; installing")
    updater.launch_swap(new_root, relaunch=not args.no_relaunch)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="cta", description="CTA - DJ library analyser and setlist builder")
    p.add_argument("--version", action="version", version=f"CTA {__version__}")
    p.add_argument("--db", help="database file (default: per-user data folder)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--background", action="store_true",
                   help="start the desktop app as a mix server in the system tray (no window)")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("scan", help="add folders and analyse new/changed files")
    s.add_argument("folders", nargs="*")
    s.add_argument("--workers", type=int, default=0)
    s.add_argument("--no-ai", action="store_true")
    s.add_argument("--cpu", action="store_true", help="run the AI model on the CPU")
    s.add_argument("--force", action="store_true", help="re-analyse every file")
    s.add_argument("--retry-failed", action="store_true")
    s.add_argument("--min-bpm", type=float, default=70.0)
    s.add_argument("--max-bpm", type=float, default=180.0)
    s.set_defaults(fn=cmd_scan)

    s = sub.add_parser("list", help="list analysed tracks")
    s.add_argument("--search")
    s.add_argument("--limit", type=int, default=100)
    s.set_defaults(fn=cmd_list)

    def mix_args(sp):
        sp.add_argument("--pitch", type=float, default=6.0, help="max tempo change in %%")
        sp.add_argument("--strict-key", action="store_true")

    s = sub.add_parser("next", help="suggest the next track to mix")
    s.add_argument("track", help="file path or search text")
    s.add_argument("--limit", type=int, default=20)
    mix_args(s)
    s.set_defaults(fn=cmd_next)

    s = sub.add_parser("setlist", help="build a setlist from a seed track")
    s.add_argument("track")
    s.add_argument("-n", "--length", type=int, default=20)
    s.add_argument("--end-bpm", type=float)
    s.add_argument("--curve", default="steady", choices=["steady", "build", "peak", "wave", "cooldown"])
    s.add_argument("--variety", type=float, default=0.0)
    s.add_argument("-o", "--output", help=".m3u8, .xml (rekordbox) or .csv")
    mix_args(s)
    s.set_defaults(fn=cmd_setlist)

    s = sub.add_parser("selftest", help="end-to-end self test")
    s.add_argument("--no-ai", action="store_true")
    mix_args(s)
    s.set_defaults(fn=cmd_selftest)

    s = sub.add_parser("update", help="check for and install a new version")
    s.add_argument("--check", action="store_true", help="only check, do not install")
    s.add_argument("--force", action="store_true", help="reinstall even if up to date")
    s.add_argument("--no-relaunch", action="store_true", help="do not restart CTA afterwards")
    s.add_argument("--from-zip", help=argparse.SUPPRESS)
    s.add_argument("--expect-version", help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_update)

    s = sub.add_parser("devices", help="list the audio output devices connected right now (JSON)")
    s.add_argument("--out", help="write the list to this file (default: print it)")
    s.set_defaults(fn=cmd_devices)

    s = sub.add_parser("output-helper", help=argparse.SUPPRESS)       # extra output device, fed by CTA
    s.add_argument("--device", required=True)
    s.add_argument("--chans", default="0,1")
    s.add_argument("--block", type=int, default=512)
    s.add_argument("--null", action="store_true")
    s.add_argument("--stats")
    s.set_defaults(fn=cmd_output_helper)

    s = sub.add_parser("gui", help="start the desktop app (default)")
    s.set_defaults(fn=None)
    return p


def main(argv=None) -> int:
    import multiprocessing

    multiprocessing.freeze_support()
    argv = sys.argv[1:] if argv is None else argv
    # macOS passes -psn_* when launched from Finder
    argv = [a for a in argv if not a.startswith("-psn_")]
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    if not args.cmd or args.fn is None:
        from .gui.app import run_gui

        return run_gui(db_path=args.db, background=args.background)
    return args.fn(args)
