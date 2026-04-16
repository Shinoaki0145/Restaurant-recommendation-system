"""Inference-only FastAPI entrypoint for the restaurant ranking service.

Responsibilities:
- validate API requests
- retrieve candidate restaurant IDs from the request or Pinecone
- fetch candidate restaurant metadata from the repository
- load a pre-trained ranker artifact and return ranking results

This module never trains a model. If the artifact is missing, the caller is
asked to train it first via ``python -m backend.restaurant_ranker``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from backend import BACKEND_DIR, get_loaded_env_path, load_backend_environment, resolve_config_path
    from backend.restaurant_ranker import (
        DEFAULT_ARTIFACT_PATH,
        RestaurantRankerService,
        normalize_restaurant_id,
    )
    from backend.restaurant_repository import RestaurantRepository
    from backend.search import search_candidates
else:
    from . import BACKEND_DIR, get_loaded_env_path, load_backend_environment, resolve_config_path
    from .restaurant_ranker import (
        DEFAULT_ARTIFACT_PATH,
        RestaurantRankerService,
        normalize_restaurant_id,
    )
    from .restaurant_repository import RestaurantRepository
    from .search import search_candidates

load_backend_environment()

TRAIN_COMMAND = "python -m backend.restaurant_ranker"


def resolve_artifact_path() -> Path:
    configured_path = (os.getenv("RANKER_ARTIFACT_PATH") or "").strip()
    return resolve_config_path(
        configured_path=configured_path,
        default_path=DEFAULT_ARTIFACT_PATH,
        fallback_base=BACKEND_DIR,
    )


ARTIFACT_PATH = resolve_artifact_path()

_service: RestaurantRankerService | None = None
_repository: RestaurantRepository | None = None


class ServiceNotReadyError(RuntimeError):
    """Raised when the API cannot serve inference requests yet."""


def build_missing_artifact_message(path: Path) -> str:
    return (
        f"Missing ranker artifact at '{path}'. "
        f"Train it first with `{TRAIN_COMMAND}` and restart the API."
    )


def normalize_candidate_ids(values: list[str | int]) -> list[str]:
    normalized_ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized_value = normalize_restaurant_id(value)
        if normalized_value and normalized_value not in seen:
            seen.add(normalized_value)
            normalized_ids.append(normalized_value)
    return normalized_ids


def attach_retrieval_scores(results: list[dict[str, Any]], retrieval_matches: list[dict[str, Any]]) -> None:
    retrieval_score_by_id = {
        normalize_restaurant_id(match["id"]): float(match["score"])
        for match in retrieval_matches
        if normalize_restaurant_id(match["id"])
    }
    for item in results:
        restaurant_id = normalize_restaurant_id(item["restaurant_id"])
        if restaurant_id in retrieval_score_by_id:
            item["retrieval_score"] = retrieval_score_by_id[restaurant_id]


class RankRequest(BaseModel):
    query: str = Field(..., description="Natural-language restaurant query")
    top_k: int = Field(default=5, ge=1, le=20)
    pinecone_top_k: int = Field(default=30, ge=1, le=100)
    candidate_restaurant_ids: list[str | int] | None = Field(
        default=None,
        description="Optional candidate IDs. If omitted, the API will try Pinecone first.",
    )
    use_pinecone: bool = Field(default=True)


def get_service() -> RestaurantRankerService:
    global _service
    if _service is None:
        if not ARTIFACT_PATH.exists():
            raise ServiceNotReadyError(build_missing_artifact_message(ARTIFACT_PATH))
        try:
            _service = RestaurantRankerService.load(ARTIFACT_PATH)
        except FileNotFoundError as exc:
            raise ServiceNotReadyError(build_missing_artifact_message(ARTIFACT_PATH)) from exc
        except Exception as exc:
            raise ServiceNotReadyError(f"Failed to load ranker artifact '{ARTIFACT_PATH}': {exc}") from exc
    return _service


app = FastAPI(title="Restaurant Ranking API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_repository() -> RestaurantRepository:
    global _repository
    if _repository is None:
        _repository = RestaurantRepository()
    return _repository


@app.get("/health")
def health() -> dict[str, Any]:
    repository = get_repository()
    env_path = get_loaded_env_path()
    ranker_status: dict[str, Any] = {
        "artifact_path": str(ARTIFACT_PATH),
        "artifact_exists": ARTIFACT_PATH.exists(),
        "ready": False,
    }
    try:
        ranker_status.update(get_service().health())
        ranker_status["ready"] = True
    except ServiceNotReadyError as exc:
        ranker_status["error"] = str(exc)
    return {
        "status": "ok" if ranker_status["ready"] else "degraded",
        "env_file": str(env_path) if env_path is not None else None,
        "ranker": ranker_status,
        "repository": repository.health(),
    }


@app.post("/rank")
def rank_restaurants(payload: RankRequest) -> dict[str, Any]:
    try:
        service = get_service()
    except ServiceNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    repository = get_repository()

    retrieval_matches: list[dict[str, Any]] = []
    candidate_ids: list[str] = []

    if payload.candidate_restaurant_ids:
        candidate_ids = normalize_candidate_ids(payload.candidate_restaurant_ids)
        retrieval_source = "request"
    elif payload.use_pinecone:
        try:
            retrieval_matches = search_candidates(
                query=payload.query,
                top_k=payload.pinecone_top_k,
                include_metadata=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Pinecone retrieval failed: {exc}") from exc

        candidate_ids = normalize_candidate_ids([match["id"] for match in retrieval_matches])
        retrieval_source = "pinecone"
    else:
        raise HTTPException(
            status_code=400,
            detail="Full local-catalog inference is disabled. Provide candidate_restaurant_ids or use_pinecone=true.",
        )

    if not candidate_ids:
        raise HTTPException(status_code=404, detail="No candidate restaurant IDs were found")

    candidate_rows, repository_info = repository.fetch_by_ids(candidate_ids)
    if repository_info.get("db_error") and candidate_rows.empty:
        raise HTTPException(
            status_code=503,
            detail=f"Database fetch failed: {repository_info['db_error']}",
        )
    if candidate_rows.empty:
        raise HTTPException(status_code=404, detail="No restaurant metadata was found in the database for the retrieved candidate IDs")

    try:
        results = service.rank(
            query=payload.query,
            candidate_restaurant_ids=candidate_ids,
            candidate_restaurants=candidate_rows,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {exc}") from exc

    attach_retrieval_scores(results, retrieval_matches)

    response = {
        "query": payload.query,
        "top_k": min(payload.top_k, len(results)),
        "candidate_count": len(candidate_ids),
        "retrieval_source": retrieval_source,
        "repository": repository_info,
        "results": results,
    }
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=False)
