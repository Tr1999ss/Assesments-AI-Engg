"""
Vector database access, isolated in one place.

This is the file you'd rewrite if you ever swapped ChromaDB for Qdrant or pgvector —
everything else in the app just calls get_collection() / get_or_create_collection()
and doesn't care what's underneath.
"""

import chromadb

from ragapp.config import CHROMA_PATH, COLLECTION_NAME


def get_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_or_create_collection(fresh: bool = False):
    """
    Get the collection, optionally wiping it first.

    fresh=True is used by `ingest` — every re-ingest rebuilds from scratch so old
    chunk sizes never mix with new ones in the same collection.
    """
    client = get_client()
    if fresh:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        return client.create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine distance, not raw L2 — keeps distances in a 0-2 range
        )
    return client.get_collection(COLLECTION_NAME)