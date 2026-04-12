from __future__ import annotations

from contextlib import asynccontextmanager
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if not __package__ and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(BACKEND_DIR / ".env")

if __package__:
    from .restaurant_ranker import (
        DEFAULT_ARTIFACT_PATH,
        DEFAULT_RESTAURANTS_PATH,
        RestaurantRankerService,
        normalize_restaurant_id,
    )
    from .restaurant_repository import RestaurantRepository
    from .search import search_candidates
else:
    from backend.restaurant_ranker import (
        DEFAULT_ARTIFACT_PATH,
        DEFAULT_RESTAURANTS_PATH,
        RestaurantRankerService,
        normalize_restaurant_id,
    )
    from backend.restaurant_repository import RestaurantRepository
    from backend.search import search_candidates

_service: Optional[RestaurantRankerService] = None
_repository: Optional[RestaurantRepository] = None


def resolve_config_path(env_name: str, default_path: Path, preferred_base: Path) -> Path:
    raw_value = (os.getenv(env_name) or "").strip()
    if not raw_value:
        return default_path

    configured_path = Path(raw_value)
    if configured_path.is_absolute():
        return configured_path

    candidates: list[Path] = []
    for base_dir in (preferred_base, PROJECT_ROOT, BACKEND_DIR, Path.cwd()):
        candidate = base_dir / configured_path
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return preferred_base / configured_path


RESTAURANTS_PATH = resolve_config_path(
    env_name="RANKER_RESTAURANTS_PATH",
    default_path=DEFAULT_RESTAURANTS_PATH,
    preferred_base=PROJECT_ROOT,
)
ARTIFACT_PATH = resolve_config_path(
    env_name="RANKER_ARTIFACT_PATH",
    default_path=DEFAULT_ARTIFACT_PATH,
    preferred_base=BACKEND_DIR,
)


class RankRequest(BaseModel):
    query: str = Field(..., description="Natural-language restaurant query")
    top_k: int = Field(default=5, ge=1, le=20)
    pinecone_top_k: int = Field(default=30, ge=1, le=100)
    candidate_restaurant_ids: Optional[list[str | int]] = Field(
        default=None,
        description="Optional candidate IDs. If omitted, the API will try Pinecone first.",
    )
    use_pinecone: bool = Field(default=True)


def get_service() -> RestaurantRankerService:
    global _service
    if _service is None:
        _service = RestaurantRankerService.load_existing(artifact_path=ARTIFACT_PATH)
    return _service


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_service()
    yield


app = FastAPI(title="Restaurant Ranking API", version="1.0.0", lifespan=lifespan)


def get_repository() -> RestaurantRepository:
    global _repository
    if _repository is None:
        _repository = RestaurantRepository(
            csv_path=RESTAURANTS_PATH,
            db_enabled=True,
            allow_csv_fallback=False,
        )
    return _repository


@app.get("/health")
def health() -> dict[str, Any]:
    service = get_service()
    repository = get_repository()
    return {
        "status": "ok",
        "ranker": service.health(),
        "repository": repository.health(),
    }


@app.post("/rank")
def rank_restaurants(payload: RankRequest) -> dict[str, Any]:
    service = get_service()
    repository = get_repository()

    retrieval_matches: list[dict[str, Any]] = []
    candidate_ids: list[str] = []

    if payload.candidate_restaurant_ids:
        candidate_ids = [normalize_restaurant_id(value) for value in payload.candidate_restaurant_ids if normalize_restaurant_id(value)]
        retrieval_source = "request"
    elif payload.use_pinecone:
        try:
            retrieval_matches = search_candidates(
                query=payload.query,
                top_k=payload.pinecone_top_k,
                include_metadata=True,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Pinecone retrieval failed: {exc}") from exc

        candidate_ids = [normalize_restaurant_id(match["id"]) for match in retrieval_matches if normalize_restaurant_id(match["id"])]
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
            detail=f"Supabase fetch failed: {repository_info['db_error']}",
        )
    if candidate_rows.empty:
        raise HTTPException(status_code=404, detail="No restaurant metadata was found in Supabase for the retrieved candidate IDs")

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

    retrieval_score_by_id = {
        normalize_restaurant_id(match["id"]): float(match["score"])
        for match in retrieval_matches
        if normalize_restaurant_id(match["id"])
    }
    for item in results:
        restaurant_id = normalize_restaurant_id(item["restaurant_id"])
        if restaurant_id in retrieval_score_by_id:
            item["retrieval_score"] = retrieval_score_by_id[restaurant_id]

    response = {
        "query": payload.query,
        "top_k": min(payload.top_k, len(results)),
        "candidate_count": len(candidate_ids),
        "retrieval_source": retrieval_source,
        "repository": repository_info,
        "results": results,
    }
    return jsonable_encoder(response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
