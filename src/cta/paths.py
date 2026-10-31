"""Per-user data locations (database, models, logs)."""

from __future__ import annotations

import logging
import os
import shutil

from platformdirs import user_data_dir

from . import APP_NAME, LEGACY_NAME

log = logging.getLogger(__name__)
_migrated = False


def data_dir() -> str:
    d = os.environ.get("TEMPEST_DATA_DIR") or os.environ.get("CTA_DATA_DIR") or user_data_dir(APP_NAME, appauthor=False)
    _migrate_legacy(d)
    os.makedirs(d, exist_ok=True)
    return d


def _migrate_legacy(new: str) -> None:
    """The first start after the rename: the old CTA data folder (library,
    analysis, settings file, learned mixes) becomes the Tempest one - moved
    when possible, copied otherwise. Nothing is lost either way."""
    global _migrated
    if _migrated or os.environ.get("TEMPEST_DATA_DIR") or os.environ.get("CTA_DATA_DIR"):
        _migrated = True
        return
    _migrated = True
    old = user_data_dir(LEGACY_NAME, appauthor=False)
    if (os.path.isdir(new) and os.listdir(new)) or not os.path.isdir(old) \
            or os.path.normcase(old) == os.path.normcase(new):
        return
    try:
        if os.path.isdir(new):
            os.rmdir(new)
        os.rename(old, new)
        log.info("moved the %s data folder to %s", LEGACY_NAME, new)
    except OSError:
        try:
            shutil.copytree(old, new, dirs_exist_ok=True)
            log.info("copied the %s data folder to %s", LEGACY_NAME, new)
        except OSError:
            log.exception("could not migrate the %s data folder", LEGACY_NAME)


def models_dir() -> str:
    d = os.path.join(data_dir(), "models")
    os.makedirs(d, exist_ok=True)
    return d


def db_path() -> str:
    return os.path.join(data_dir(), "library.sqlite3")


def log_path() -> str:
    return os.path.join(data_dir(), "tempest.log")
