"""Per-user data locations (database, models, logs)."""

from __future__ import annotations

import os

from platformdirs import user_data_dir

APP_NAME = "CTA"


def data_dir() -> str:
    d = os.environ.get("CTA_DATA_DIR") or user_data_dir(APP_NAME, appauthor=False)
    os.makedirs(d, exist_ok=True)
    return d


def models_dir() -> str:
    d = os.path.join(data_dir(), "models")
    os.makedirs(d, exist_ok=True)
    return d


def db_path() -> str:
    return os.path.join(data_dir(), "library.sqlite3")


def log_path() -> str:
    return os.path.join(data_dir(), "cta.log")
