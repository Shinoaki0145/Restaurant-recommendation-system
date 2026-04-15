"""Candidate retrieval helpers backed by Pinecone embeddings search."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(BACKEND_DIR / ".env")


def _load_embed_text():
    try:
        from embed_model import embed_text

        return embed_text
    except ImportError:
        from dataset.embed_model import embed_text

        return embed_text


def get_pinecone_api_key() -> str:
    return (os.getenv("PINECONE_API_KEY") or os.getenv("API_KEY") or "").strip()


def get_pinecone_index_name() -> str:
    return (os.getenv("PINECONE_INDEX_NAME") or "restaurant").strip()


def get_pinecone_index(index_name: str | None = None):
    api_key = get_pinecone_api_key()
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY/API_KEY in .env")

    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise RuntimeError("Package 'pinecone' is not installed") from exc

    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name or get_pinecone_index_name())


def convert_embedding(embedding: Any) -> list[float]:
    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().numpy()
    elif hasattr(embedding, "cpu") and hasattr(embedding, "numpy"):
        embedding = embedding.cpu().numpy()
    elif hasattr(embedding, "numpy"):
        embedding = embedding.numpy()

    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
        embedding = embedding[0]
    if not isinstance(embedding, list):
        raise ValueError("Unsupported embedding type")
    return [float(value) for value in embedding]


def embed_query(query: str) -> list[float]:
    if not str(query).strip():
        raise ValueError("query must not be empty")
    embed_text = _load_embed_text()
    query_embedding = embed_text([query])
    return convert_embedding(query_embedding)


def search_candidates(
    query: str,
    top_k: int = 30,
    index=None,
    include_metadata: bool = False,
) -> list[dict[str, Any]]:
    if index is None:
        index = get_pinecone_index()

    query_embedding = embed_query(query)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=include_metadata,
    )
    matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, "matches", [])

    normalized: list[dict[str, Any]] = []
    for match in matches:
        if isinstance(match, dict):
            normalized.append(
                {
                    "id": str(match.get("id", "")),
                    "score": float(match.get("score", 0.0)),
                    "metadata": match.get("metadata") or {},
                }
            )
        else:
            normalized.append(
                {
                    "id": str(getattr(match, "id", "")),
                    "score": float(getattr(match, "score", 0.0)),
                    "metadata": getattr(match, "metadata", None) or {},
                }
            )
    return normalized
