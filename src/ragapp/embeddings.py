"""Turning text into vectors — the one thing both ingestion and querying need."""

import ollama

from ragapp.config import EMBED_MODEL


def embed(text: str) -> list[float]:
    """Turn a chunk of text (or a question) into a vector using a local Ollama model."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]