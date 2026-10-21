"""local_paths — machine-specific locations, read from a local config file.

All scripts take their input/output locations from here instead of hard-coded absolute paths.
The values live in ``config/local_paths.json`` at the repository root, which is NOT version-controlled
(copy ``config/local_paths.example.json`` and fill in your own folders). Set the environment variable
``TCVNS_PATHS_CONFIG`` to use a config file elsewhere.

Keys
----
PROJECT_ROOT        analysis project folder (event tables, Final analysis/, outputs, figures)
DATA_ROOT           root of the preprocessed recordings delivered by the recording pipeline
DATA_ROOT_ALT       optional second copy of the recording data (searched after DATA_ROOT)
SOSPINDLE_ROOT_ALT  optional older event-detection output folder
AUDIT_ROOT          validation/audit output folder
SCRATCH_DIR         folder for temporary text dumps

Access a key as an attribute (``local_paths.PROJECT_ROOT``); a missing key raises an error that names
the key and the config file. ``optional(key, *parts)`` returns ``None`` when the key is not set.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = Path(__file__).resolve().parent
KEYS = ("PROJECT_ROOT", "DATA_ROOT", "DATA_ROOT_ALT", "SOSPINDLE_ROOT_ALT", "AUDIT_ROOT", "SCRATCH_DIR")


def config_file() -> Path:
    env = os.environ.get("TCVNS_PATHS_CONFIG")
    return Path(env) if env else REPO_ROOT / "config" / "local_paths.json"


@lru_cache(maxsize=1)
def _load() -> dict:
    f = config_file()
    if not f.exists():
        return {}
    with open(f, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: v for k, v in raw.items() if k in KEYS and v}


def optional(key: str, *parts: str) -> Path | None:
    """Return the configured folder (joined with `parts`), or None if the key is not set."""
    if key not in KEYS:
        raise KeyError(f"unknown path key: {key}")
    v = _load().get(key)
    return Path(v).joinpath(*parts) if v else None


def get(key: str, *parts: str) -> Path:
    """Return the configured folder (joined with `parts`); raise a clear error if it is not set."""
    p = optional(key, *parts)
    if p is None:
        raise KeyError(f"path '{key}' is not set — add it to {config_file()} "
                       f"(template: config/local_paths.example.json)")
    return p


def __getattr__(name: str) -> Path:
    if name in KEYS:
        return get(name)
    raise AttributeError(name)
