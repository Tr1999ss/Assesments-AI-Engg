"""Load -> Chunk. Pure logic, no printing, no CLI — that's the commands/ layer's job."""

import glob
import os

from ragapp.config import DOCS_DIR


def load_documents() -> list[tuple[str, str]]:
    """Read every .md file in DOCS_DIR and return (filename, text) pairs."""
    docs = []
    for path in glob.glob(os.path.join(DOCS_DIR, "*.md")):
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks of `chunk_size` characters.

    Overlap matters: without it, a sentence straddling a chunk boundary gets cut in
    half and neither chunk contains the full answer. Overlap gives each chunk a bit
    of its neighbor's context.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks