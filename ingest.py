"""
ingest.py — Step 1-4 of RAG: Load -> Chunk -> Embed -> Store

Run this whenever you add/change documents in the docs/ folder.

Usage:
    python ingest.py                # default chunk size (500 chars, 50 overlap)
    python ingest.py --chunk-size 300 --overlap 50
    python ingest.py --chunk-size 800 --overlap 100

Try different chunk sizes and compare retrieval quality in query.py —
that comparison is literally one of the things your mentor checks for.
"""

import argparse
import glob
import os

import chromadb
import ollama

DOCS_DIR = "docs"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "dev_docs"
EMBED_MODEL = "nomic-embed-text"  # pull with: ollama pull nomic-embed-text


def load_documents():
    """Read every .md file in docs/ and return (filename, text) pairs."""
    docs = []
    for path in glob.glob(os.path.join(DOCS_DIR, "*.md")):
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


def chunk_text(text, chunk_size, overlap):
    """
    Split text into overlapping chunks of `chunk_size` characters.

    Overlap matters: without it, a sentence that straddles a chunk
    boundary gets cut in half and neither chunk contains the full
    answer. Overlap gives each chunk a bit of its neighbor's context.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed(text):
    """Turn a chunk of text into a vector (list of numbers) using a local Ollama model."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading documents from {DOCS_DIR}/ ...")
    documents = load_documents()
    print(f"Found {len(documents)} document(s): {[d[0] for d in documents]}")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Fresh start each time you re-ingest, so old chunk sizes don't mix with new ones
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    for filename, text in documents:
        chunks = chunk_text(text, args.chunk_size, args.overlap)
        print(f"  {filename}: {len(chunks)} chunks", flush=True)

        for i, chunk in enumerate(chunks):
            print(f"    embedding chunk {i+1}/{len(chunks)} ...", flush=True)
            vector = embed(chunk)
            print(f"    embedded ok, dim={len(vector)}. adding to chroma ...", flush=True)
            collection.add(
                ids=[f"{filename}-{i}"],
                embeddings=[vector],
                documents=[chunk],
                metadatas=[{"source": filename, "chunk_size": args.chunk_size}],
            )
            print(f"    added ok.", flush=True)
            total_chunks += 1

    print(f"\nDone. Stored {total_chunks} chunks in {CHROMA_PATH}/ "
          f"(chunk_size={args.chunk_size}, overlap={args.overlap})")


if __name__ == "__main__":
    main()