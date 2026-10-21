"""Library scanning: finds audio files, analyses them in a process pool and
runs the AI model in batches on the GPU (or CPU) from the coordinating thread.

Workers do decoding + DSP (CPU bound, embarrassingly parallel); the model is
loaded once, here, so GPU memory is used by a single session regardless of
how many workers run.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field

from . import model as model_mod
from .analyzer import ANALYSIS_VERSION, analyze_file, style_tempo_fix, summarize_styles, worker_init
from .audio import AUDIO_EXTENSIONS
from .db import Database, vec_to_blob

log = logging.getLogger(__name__)


@dataclass
class ScanSettings:
    workers: int = 0                 # 0 = auto (all cores but one)
    min_bpm: float = 70.0
    max_bpm: float = 180.0
    use_ai: bool = True
    ai_device: str = "auto"          # auto | cpu
    force: bool = False              # re-analyse everything
    retry_failed: bool = False


@dataclass
class ScanProgress:
    phase: str = "idle"
    total: int = 0
    done: int = 0
    failed: int = 0
    skipped: int = 0
    current: str = ""
    started: float = field(default_factory=time.time)
    device: str = ""
    message: str = ""

    @property
    def rate(self) -> float:
        el = time.time() - self.started
        return self.done / el if el > 0 else 0.0

    @property
    def eta(self) -> float:
        r = self.rate
        return (self.total - self.done) / r if r > 0 else 0.0


def default_workers() -> int:
    n = os.cpu_count() or 2
    # ProcessPoolExecutor on Windows is limited to 61 workers
    return max(1, min(n - 1, 61))


def find_audio_files(folders: list[str], stop: threading.Event | None = None) -> list[str]:
    out = []
    for root in folders:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            if stop is not None and stop.is_set():
                return out
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                if fn.startswith("._") or fn.startswith("."):
                    continue   # macOS AppleDouble files and hidden files
                if os.path.splitext(fn)[1].lower() in AUDIO_EXTENSIONS:
                    out.append(os.path.normpath(os.path.join(dirpath, fn)))
    return out


class Scanner:
    def __init__(self, db: Database, settings: ScanSettings | None = None,
                 on_progress=None, on_batch_saved=None):
        self.db = db
        self.settings = settings or ScanSettings()
        self.progress = ScanProgress()
        self._on_progress = on_progress or (lambda p: None)
        self._on_batch_saved = on_batch_saved or (lambda n: None)
        self._stop = threading.Event()
        self._model: model_mod.StyleModel | None = None
        self._labels: list[str] | None = None

    def stop(self):
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _emit(self, **kw):
        for k, v in kw.items():
            setattr(self.progress, k, v)
        try:
            self._on_progress(self.progress)
        except Exception:  # noqa: BLE001
            log.exception("progress callback failed")

    # ------------------------------------------------------------------
    def _load_model(self):
        if not self.settings.use_ai:
            return
        self._emit(phase="model", message="Preparing AI model…")

        def dl(done, total):
            self._emit(message=f"Downloading AI model {done * 100 // max(total, 1)}%")

        path = model_mod.ensure_model(dl)
        if not path:
            self._emit(message="AI model unavailable - genres from tags only")
            return
        try:
            self._model = model_mod.StyleModel(path, self.settings.ai_device)
            self._labels = model_mod.labels()
            self._emit(device=self._model.device_name)
        except Exception as e:  # noqa: BLE001
            log.warning("model load failed: %s", e)
            self._emit(message=f"AI model failed to load ({e}) - continuing without it")
            self._model = None

    def _plan(self, folders: list[str]) -> list[str]:
        self._emit(phase="listing", message="Finding audio files…")
        files = find_audio_files(folders, self._stop)
        known = self.db.known_files()
        found = set(files)
        prefixes = tuple(os.path.normpath(f) + os.sep for f in folders)
        missing = [p for p in known if p.startswith(prefixes) and p not in found]
        if missing:
            self.db.set_missing(missing, True)
        todo = []
        skipped = 0
        have_ai = self._model is not None
        for p in files:
            k = known.get(p)
            if k is None or self.settings.force:
                todo.append(p)
                continue
            mtime, size, version, status, has_ai = k
            try:
                st = os.stat(p)
            except OSError:
                continue
            changed = (mtime is None or abs(st.st_mtime - mtime) > 1e-3 or st.st_size != size)
            if (changed or version != ANALYSIS_VERSION
                    or (status != "ok" and self.settings.retry_failed)
                    or (status == "ok" and have_ai and not has_ai)):
                todo.append(p)
            else:
                skipped += 1
        # files previously flagged missing that are back
        back = [p for p in files if p in known]
        if back:
            self.db.set_missing(back, False)
        self._emit(skipped=skipped)
        return todo

    def _rows_from_results(self, results: list[dict]) -> list[dict]:
        groups, owners = [], []
        for i, r in enumerate(results):
            p = r.pop("patches", None)
            if r.get("ok") and p is not None and self._model is not None:
                groups.append(p)
                owners.append(i)
        vecs = {}
        if groups:
            try:
                out = self._model.track_vectors(groups)
                for i, v in zip(owners, out):
                    vecs[i] = v
            except Exception as e:  # noqa: BLE001
                log.exception("inference failed: %s", e)

        rows = []
        for i, r in enumerate(results):
            row = {k: r.get(k) for k in (
                "path", "mtime", "size", "title", "artist", "album", "tag_genre", "tag_bpm",
                "tag_key", "duration", "bpm", "bpm_conf", "key", "key_conf", "loudness",
                "centroid", "onset_rate", "analyzed_at", "version")}
            if r.get("ok"):
                row["status"] = "ok"
                row["error"] = None
                if i in vecs:
                    probs, emb = vecs[i]
                    styles, parent = summarize_styles(probs, self._labels)
                    fixed = style_tempo_fix(row.get("bpm") or 0.0, styles)
                    if fixed != row.get("bpm"):
                        log.info("tempo octave from style %s: %s -> %s", styles[0][0], row["bpm"], fixed)
                        row["bpm"] = fixed
                    row["genre"] = styles[0][0].split("---")[-1] if styles else None
                    row["styles"] = json.dumps({"parent": parent, "top": styles})
                    row["style_vec"] = vec_to_blob(probs)
                    row["embedding"] = vec_to_blob(emb)
            else:
                row["status"] = "error"
                row["error"] = r.get("error", "unknown error")
                log.info("failed %s: %s", r.get("path"), row["error"])
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    def run(self, folders: list[str]) -> ScanProgress:
        self.progress = ScanProgress()
        try:
            self._load_model()
            if self._stop.is_set():
                return self._finish("stopped")
            todo = self._plan(folders)
            self._emit(phase="analyzing", total=len(todo), done=0, started=time.time(),
                       message=f"Analysing {len(todo)} files" + (f" ({self.progress.skipped} unchanged)"
                                                               if self.progress.skipped else ""))
            if todo:
                self._analyze(todo)
            return self._finish("stopped" if self._stop.is_set() else "done")
        except Exception as e:  # noqa: BLE001
            log.exception("scan failed")
            self._emit(phase="error", message=f"Scan failed: {e}")
            return self.progress

    def _finish(self, phase):
        p = self.progress
        msg = f"Scan {phase}: {p.done} analysed, {p.failed} failed, {p.skipped} unchanged"
        self._emit(phase=phase, message=msg, current="")
        return p

    def _new_pool(self, workers):
        ctx = mp.get_context("spawn")
        try:
            return cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=worker_init,
                                          max_tasks_per_child=500)
        except TypeError:  # Python < 3.11
            return cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=worker_init)

    def _analyze(self, todo: list[str]):
        s = self.settings
        workers = s.workers or default_workers()
        want_model = self._model is not None
        queue = list(reversed(todo))
        retries: dict[str, int] = {}
        inflight: dict[cf.Future, str] = {}
        pending: list[dict] = []
        last_flush = time.time()
        pool = self._new_pool(workers)
        window = workers * 3

        def flush():
            nonlocal pending, last_flush
            if not pending:
                return
            rows = self._rows_from_results(pending)
            self.db.upsert_many(rows)
            self._on_batch_saved(len(rows))
            pending = []
            last_flush = time.time()

        try:
            while (queue or inflight) and not self._stop.is_set():
                while queue and len(inflight) < window:
                    p = queue.pop()
                    try:
                        fut = pool.submit(analyze_file, p, s.min_bpm, s.max_bpm, want_model)
                    except BrokenProcessPool:
                        queue.append(p)
                        break
                    inflight[fut] = p
                done, _ = cf.wait(list(inflight), timeout=0.25, return_when=cf.FIRST_COMPLETED)
                broken = False
                for fut in done:
                    p = inflight.pop(fut)
                    try:
                        r = fut.result()
                    except BrokenProcessPool:
                        broken = True
                        n = retries.get(p, 0) + 1
                        retries[p] = n
                        if n >= 3:
                            r = {"path": p, "ok": False, "error": "decoder crashed repeatedly",
                                 "version": ANALYSIS_VERSION, "analyzed_at": time.time()}
                            try:
                                st = os.stat(p)
                                r["mtime"], r["size"] = st.st_mtime, st.st_size
                            except OSError:
                                pass
                        else:
                            queue.append(p)
                            continue
                    except Exception as e:  # noqa: BLE001
                        r = {"path": p, "ok": False, "error": f"{type(e).__name__}: {e}",
                             "version": ANALYSIS_VERSION, "analyzed_at": time.time()}
                    pending.append(r)
                    if r.get("ok"):
                        self.progress.done += 1
                    else:
                        self.progress.done += 1
                        self.progress.failed += 1
                    self._emit(current=os.path.basename(p))
                if broken:
                    log.warning("worker pool crashed; restarting it")
                    queue.extend(inflight.values())
                    inflight.clear()
                    pool.shutdown(wait=False, cancel_futures=True)
                    pool = self._new_pool(workers)
                # batch the GPU work: flush when enough tracks or time passed
                if len(pending) >= 16 or (pending and time.time() - last_flush > 1.0):
                    flush()
            flush()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
