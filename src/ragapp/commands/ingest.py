"""The `ingest` command. CLI concerns only — actual logic lives in ragapp.ingestion / store."""

import typer

from ragapp.config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from ragapp.embeddings import embed
from ragapp.ingestion import chunk_text, load_documents
from ragapp.store import get_or_create_collection


def ingest(
    chunk_size: int = typer.Option(DEFAULT_CHUNK_SIZE, help="Characters per chunk."),
    overlap: int = typer.Option(DEFAULT_OVERLAP, help="Overlapping characters between chunks."),
):
    """Load documents, chunk them, embed them, and store them in the vector DB."""
    print("Loading documents ...")
    documents = load_documents()
    print(f"Found {len(documents)} document(s): {[d[0] for d in documents]}")

    collection = get_or_create_collection(fresh=True)

    total_chunks = 0
    for filename, text in documents:
        chunks = chunk_text(text, chunk_size, overlap)
        print(f"  {filename}: {len(chunks)} chunks", flush=True)

        for i, chunk in enumerate(chunks):
            print(f"    embedding chunk {i+1}/{len(chunks)} ...", flush=True)
            vector = embed(chunk)
            collection.add(
                ids=[f"{filename}-{i}"],
                embeddings=[vector],
                documents=[chunk],
                metadatas=[{"source": filename, "chunk_size": chunk_size}],
            )
            print("    added ok.", flush=True)
            total_chunks += 1

    print(f"\nDone. Stored {total_chunks} chunks (chunk_size={chunk_size}, overlap={overlap})")