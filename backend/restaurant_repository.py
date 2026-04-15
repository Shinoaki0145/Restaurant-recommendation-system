"""Repository layer for loading restaurant metadata needed by the API."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

from .restaurant_ranker import normalize_restaurant_id, slugify_column_name


def _is_placeholder(value: str) -> bool:
    text = (value or "").strip()
    return not text or text.startswith("<") or "YOUR_" in text or text == "None"


def _safe_identifier(identifier: str) -> str:
    parts = [part.strip() for part in identifier.split(".") if part.strip()]
    if not parts:
        raise ValueError("Identifier is empty")
    for part in parts:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            raise ValueError(f"Unsafe SQL identifier: {identifier}")
    return ".".join(parts)


def _quote_identifier(identifier: str) -> str:
    safe_identifier = _safe_identifier(identifier)
    return ".".join(f'"{part}"' for part in safe_identifier.split("."))


def _build_database_url() -> str:
    database_url = (os.getenv("DATABASE_URL") or "").strip()
    if database_url and not _is_placeholder(database_url):
        return database_url

    host = (os.getenv("DB_HOST") or "").strip()
    port = (os.getenv("DB_PORT") or "5432").strip()
    name = (os.getenv("DB_NAME") or "").strip()
    user = (os.getenv("DB_USER") or "").strip()
    password = (os.getenv("DB_PASSWORD") or "").strip()
    if any(_is_placeholder(value) for value in [host, name, user, password]):
        return ""
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def _normalize_sqlalchemy_url(database_url: str) -> str:
    if not database_url or "://" not in database_url:
        return database_url

    scheme, rest = database_url.split("://", 1)
    if "+" in scheme or scheme != "postgresql":
        return database_url

    try:
        import psycopg  # noqa: F401

        return f"postgresql+psycopg://{rest}"
    except ImportError:
        return database_url


class RestaurantRepository:
    def __init__(
        self,
        db_enabled: bool | None = None,
        database_url: str | None = None,
        table_name: str | None = None,
        id_column: str | None = None,
    ) -> None:
        self.db_enabled = (
            str(os.getenv("RESTAURANT_DB_ENABLED", "false")).strip().lower() in {"1", "true", "yes"}
            if db_enabled is None
            else bool(db_enabled)
        )
        self.database_url = database_url or _build_database_url()
        self.sqlalchemy_url = _normalize_sqlalchemy_url(self.database_url)
        self.table_name = table_name or os.getenv("RESTAURANT_DB_TABLE") or "restaurants"
        self.id_column = id_column or os.getenv("RESTAURANT_DB_ID_COLUMN") or "restaurant_id"

    def _fetch_from_postgres(self, restaurant_ids: list[str]) -> pd.DataFrame:
        if not self.db_enabled:
            return pd.DataFrame()
        if _is_placeholder(self.database_url):
            raise RuntimeError("DATABASE_URL/DB_* is not configured yet")

        table_name = _quote_identifier(self.table_name)
        id_column = _quote_identifier(self.id_column)
        params = {f"id_{index}": restaurant_id for index, restaurant_id in enumerate(restaurant_ids)}
        placeholders = ", ".join(f":id_{index}" for index in range(len(restaurant_ids)))
        query = text(f"SELECT * FROM {table_name} WHERE CAST({id_column} AS TEXT) IN ({placeholders})")

        engine = create_engine(self.sqlalchemy_url)
        with engine.connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def fetch_by_ids(self, restaurant_ids: list[str | int]) -> tuple[pd.DataFrame, dict[str, Any]]:
        normalized_ids: list[str] = []
        seen: set[str] = set()
        for restaurant_id in restaurant_ids:
            normalized = normalize_restaurant_id(restaurant_id)
            if normalized and normalized not in seen:
                seen.add(normalized)
                normalized_ids.append(normalized)

        if not normalized_ids:
            return pd.DataFrame(), {"source": "none", "count": 0}

        db_error = None
        database_df = pd.DataFrame()
        if self.db_enabled:
            try:
                database_df = self._fetch_from_postgres(normalized_ids)
            except Exception as exc:
                db_error = str(exc)
        else:
            db_error = "Database access is disabled"

        resolved_ids: set[str] = set()
        if not database_df.empty:
            possible_id_col = None
            for column in database_df.columns:
                if slugify_column_name(column) in {"restaurantid", "restaurant_id"}:
                    possible_id_col = column
                    break
            if possible_id_col is not None:
                resolved_ids = set(database_df[possible_id_col].map(normalize_restaurant_id))

        missing_ids = [restaurant_id for restaurant_id in normalized_ids if restaurant_id not in resolved_ids]
        source = "database"
        if db_error and database_df.empty:
            source = "database_error"
        return database_df, {
            "source": source,
            "count": int(len(database_df)),
            "requested_count": len(normalized_ids),
            "missing_candidate_count": len(missing_ids),
            "db_error": db_error,
        }

    def health(self) -> dict[str, Any]:
        db_ready = self.db_enabled and not _is_placeholder(self.database_url)
        return {
            "db_enabled": self.db_enabled,
            "db_ready": db_ready,
            "db_table": self.table_name,
            "db_id_column": self.id_column,
        }
