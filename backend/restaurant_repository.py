from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
LEGACY_FRONTEND_DIR = PROJECT_ROOT / "restaurant_web"
if not __package__ and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(LEGACY_FRONTEND_DIR / ".env")

if __package__:
    from .restaurant_ranker import DEFAULT_RESTAURANTS_PATH, normalize_restaurant_id, slugify_column_name
else:
    from backend.restaurant_ranker import DEFAULT_RESTAURANTS_PATH, normalize_restaurant_id, slugify_column_name


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


class RestaurantRepository:
    def __init__(
        self,
        csv_path: Path | str = DEFAULT_RESTAURANTS_PATH,
        db_enabled: bool | None = None,
        allow_csv_fallback: bool | None = None,
        database_url: str | None = None,
        table_name: str | None = None,
        id_column: str | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.db_enabled = (
            str(os.getenv("RESTAURANT_DB_ENABLED", "false")).strip().lower() in {"1", "true", "yes"}
            if db_enabled is None
            else bool(db_enabled)
        )
        self.allow_csv_fallback = (
            str(os.getenv("RESTAURANT_ALLOW_CSV_FALLBACK", "false")).strip().lower() in {"1", "true", "yes"}
            if allow_csv_fallback is None
            else bool(allow_csv_fallback)
        )
        self.database_url = database_url or _build_database_url()
        self.table_name = table_name or os.getenv("RESTAURANT_DB_TABLE") or "restaurants"
        self.id_column = id_column or os.getenv("RESTAURANT_DB_ID_COLUMN") or "restaurant_id"

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_csv(csv_path: str) -> pd.DataFrame:
        return pd.read_csv(csv_path, encoding="utf-8")

    def _fetch_from_csv(self, restaurant_ids: list[str]) -> pd.DataFrame:
        csv_df = self._load_csv(str(self.csv_path)).copy()
        id_col = None
        for candidate in ("RestaurantID", "restaurant_id", "restaurantid"):
            if candidate in csv_df.columns:
                id_col = candidate
                break
        if id_col is None:
            raise KeyError("CSV metadata file is missing RestaurantID/restaurant_id")
        csv_df["_restaurant_id_norm"] = csv_df[id_col].map(normalize_restaurant_id)
        csv_df = csv_df[csv_df["_restaurant_id_norm"].isin(restaurant_ids)].copy()
        csv_df = csv_df.drop(columns=["_restaurant_id_norm"])
        return csv_df

    def _get_postgres_driver(self):
        try:
            import psycopg

            return "psycopg", psycopg
        except ImportError:
            try:
                import psycopg2

                return "psycopg2", psycopg2
            except ImportError as exc:
                raise RuntimeError("No PostgreSQL driver found. Install 'psycopg' or 'psycopg2-binary'.") from exc

    def _fetch_from_postgres(self, restaurant_ids: list[str]) -> pd.DataFrame:
        if not self.db_enabled:
            return pd.DataFrame()
        if _is_placeholder(self.database_url):
            raise RuntimeError("DATABASE_URL/DB_* is not configured yet")

        table_name = _quote_identifier(self.table_name)
        id_column = _quote_identifier(self.id_column)
        placeholders = ",".join(["%s"] * len(restaurant_ids))
        query = f"SELECT * FROM {table_name} WHERE CAST({id_column} AS TEXT) IN ({placeholders})"

        driver_name, driver = self._get_postgres_driver()
        if driver_name == "psycopg":
            with driver.connect(self.database_url) as conn:
                return pd.read_sql_query(query, conn, params=restaurant_ids)

        conn = driver.connect(self.database_url)
        try:
            return pd.read_sql_query(query, conn, params=restaurant_ids)
        finally:
            conn.close()

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
        if not self.allow_csv_fallback:
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

        csv_df = self._fetch_from_csv(missing_ids) if missing_ids else pd.DataFrame()

        if database_df.empty and csv_df.empty:
            return pd.DataFrame(), {"source": "empty", "count": 0, "db_error": db_error}
        if database_df.empty:
            return csv_df, {"source": "csv", "count": int(len(csv_df)), "db_error": db_error}
        if csv_df.empty:
            return database_df, {"source": "database", "count": int(len(database_df))}

        combined = pd.concat([database_df, csv_df], ignore_index=True, sort=False)
        return combined, {"source": "database+csv", "count": int(len(combined)), "db_error": db_error}

    def health(self) -> dict[str, Any]:
        db_ready = self.db_enabled and not _is_placeholder(self.database_url)
        return {
            "csv_path": str(self.csv_path),
            "db_enabled": self.db_enabled,
            "allow_csv_fallback": self.allow_csv_fallback,
            "db_ready": db_ready,
            "db_table": self.table_name,
            "db_id_column": self.id_column,
        }
