"""Backend package boundaries.

- ``api`` exposes HTTP endpoints and orchestrates inference requests.
- ``restaurant_ranker`` owns artifact training/loading and ranking logic.
- ``restaurant_repository`` fetches restaurant metadata from the database.
- ``search`` retrieves candidate restaurant IDs from Pinecone.
"""
