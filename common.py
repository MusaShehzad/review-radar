"""Shared paths and file helpers used by every step of the pipeline.

A *workspace* is a folder with its own config/ and data/. The default workspace is this project
folder (the ImagineArt analysis). To analyze other apps without mixing their data in, point the
RADAR_WORKSPACE environment variable at another folder; run.py does this for you.
"""
import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("RADAR_WORKSPACE") or ROOT).resolve()
CONFIG_DIR = WORKSPACE / "config"
DEFAULT_CONFIG_DIR = ROOT / "config"  # used for any config file a workspace doesn't have
DATA_DIR = WORKSPACE / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"


def config_path(name: str) -> Path:
    """config/<name>.yaml from the workspace, or the project's default if the workspace has none."""
    own = CONFIG_DIR / f"{name}.yaml"
    return own if own.exists() else DEFAULT_CONFIG_DIR / f"{name}.yaml"


def load_config(name: str) -> dict:
    """Read a config file into a dict (see config_path for which file is used)."""
    return yaml.safe_load(config_path(name).read_text(encoding="utf-8"))


def select_apps(only: str | None = None) -> list[dict]:
    """Return the apps from apps.yaml, with the shared defaults filled in.

    Pass `only` (an app key such as "imagineart") to get just that one app.
    """
    cfg = load_config("apps")
    apps = [{**cfg["defaults"], **app} for app in cfg["apps"]]  # an app's own values win
    if only:
        apps = [a for a in apps if a["key"] == only]
        if not apps:
            raise SystemExit(f"No app with key '{only}' in {config_path('apps')}")
    return apps


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a CSV with a UTF-8 byte-order mark, so Excel shows emoji and non-Latin text correctly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_reviews(path: Path) -> pd.DataFrame:
    """Load a reviews CSV written by this pipeline.

    app_version is read as text on purpose: read as a number, version "7.10" would become 7.1.
    """
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"review_id": "string", "app_version": "string", "text": "string"},
        parse_dates=["date"],
    )
