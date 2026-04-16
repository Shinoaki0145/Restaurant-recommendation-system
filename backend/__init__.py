"""Backend package boundaries and shared runtime helpers.

- ``api`` exposes HTTP endpoints and orchestrates inference requests.
- ``restaurant_ranker`` owns artifact training/loading and ranking logic.
- ``restaurant_repository`` fetches restaurant metadata from the database.
- ``search`` retrieves candidate restaurant IDs from Pinecone.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENV_FILE_ENV_VARS = ("BACKEND_ENV_FILE", "ENV_FILE", "DOTENV_PATH")
DEFAULT_ENV_PATHS = (
    PROJECT_ROOT / ".env",
    BACKEND_DIR / ".env",
)

_ENV_LOADED = False
_LOADED_ENV_PATH: Path | None = None


def _resolve_candidate_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def load_backend_environment() -> Path | None:
    global _ENV_LOADED, _LOADED_ENV_PATH
    if _ENV_LOADED:
        return _LOADED_ENV_PATH

    candidates: list[Path] = []
    seen: set[Path] = set()

    for env_var_name in ENV_FILE_ENV_VARS:
        configured_path = (os.getenv(env_var_name) or "").strip()
        if not configured_path:
            continue
        candidate = _resolve_candidate_path(configured_path)
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    for candidate in DEFAULT_ENV_PATHS:
        resolved_candidate = candidate.resolve()
        if resolved_candidate not in seen:
            candidates.append(resolved_candidate)
            seen.add(resolved_candidate)

    for candidate in candidates:
        if candidate.is_file():
            load_dotenv(candidate)
            _LOADED_ENV_PATH = candidate
            break

    _ENV_LOADED = True
    return _LOADED_ENV_PATH


def get_loaded_env_path() -> Path | None:
    return load_backend_environment()


def resolve_config_path(configured_path: str | None, default_path: Path, fallback_base: Path | None = None) -> Path:
    raw_value = (configured_path or "").strip()
    if not raw_value:
        return default_path

    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path.resolve()

    env_path = get_loaded_env_path()
    fallback_dir = fallback_base or BACKEND_DIR
    fallback_candidate = (fallback_dir / path).resolve()
    if env_path is None:
        return fallback_candidate

    env_candidate = (env_path.parent / path).resolve()
    if env_candidate.exists() or not fallback_candidate.exists():
        return env_candidate
    return fallback_candidate
