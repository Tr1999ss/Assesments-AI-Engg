import argparse
import json
import time
from pathlib import Path

import chromadb
import ollama
from rank_bm25 import BM25Okapi

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "dev_docs"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma4:latest"  # matches what "ollama list" shows on this machine
TOP_K = 3
SIMILARITY_FLOOR = 0.35
TRACES_FILE = "traces.jsonl"


def embed(text):
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


# ---------- retrieval method 1: embeddings only (what you had before) ----------

def retrieve_embedding(question, collection, top_k=TOP_K):
    """Find the top_k most relevant chunks by meaning (embedding similarity)."""
    query_vector = embed(question)
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    distances = results["distances"][0]  # lower = more similar (cosine distance)
    return list(zip(chunks, sources, distances))


# ---------- retrieval method 2: BM25 only (keyword search) ----------

def retrieve_bm25(question, collection, top_k=TOP_K):
    """
    Find the top_k most relevant chunks by exact keyword overlap (BM25).
    Unlike embeddings, this is great at catching exact terms like error codes
    or function names that might not carry much "meaning" on their own.
    """
    all_data = collection.get(include=["documents", "metadatas"])
    all_chunks = all_data["documents"]
    all_sources = [m["source"] for m in all_data["metadatas"]]

    tokenized_corpus = [doc.lower().split() for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # higher BM25 score = better match, so sort descending
    ranked = sorted(zip(all_chunks, all_sources, scores), key=lambda x: -x[2])
    top = ranked[:top_k]
    # convert score to a "distance-like" number so downstream code stays consistent
    # (higher score = lower "distance"; this is just for display, not a real distance metric)
    return [(chunk, source, 1.0 / (1.0 + score)) for chunk, source, score in top]


# ---------- retrieval method 3: hybrid (Reciprocal Rank Fusion of the two above) ----------

def retrieve_hybrid(question, collection, top_k=TOP_K, k_const=60):
    """
    Combine embedding search and BM25 search using Reciprocal Rank Fusion (RRF).

    RRF is simple and doesn't need score normalization: for each chunk, add
    1 / (k_const + rank) from EACH method it appears in, then re-sort by that
    combined score. A chunk that ranks well in both methods rises to the top;
    a chunk that only one method liked still gets some credit.
    """
    # get a generous top-10 from each method to fuse from, not just top-3
    emb_results = retrieve_embedding(question, collection, top_k=10)
    bm25_results = retrieve_bm25(question, collection, top_k=10)

    scores = {}  # chunk_text -> {"score": float, "source": str}
    for rank, (chunk, source, _) in enumerate(emb_results):
        scores.setdefault(chunk, {"score": 0.0, "source": source})
        scores[chunk]["score"] += 1.0 / (k_const + rank)
    for rank, (chunk, source, _) in enumerate(bm25_results):
        scores.setdefault(chunk, {"score": 0.0, "source": source})
        scores[chunk]["score"] += 1.0 / (k_const + rank)

    ranked = sorted(scores.items(), key=lambda x: -x[1]["score"])
    top = ranked[:top_k]
    # again, turn the fused score into a "distance-like" number just for consistent printing
    return [(chunk, data["source"], 1.0 / (1.0 + data["score"] * 100)) for chunk, data in top]


RETRIEVERS = {
    "embedding": retrieve_embedding,
    "bm25": retrieve_bm25,
    "hybrid": retrieve_hybrid,
}


def generate_answer(question, retrieved):
    """Ask the local LLM to answer ONLY from the retrieved chunks, and cite the source."""
    best_distance = min(d for _, _, d in retrieved)
    if best_distance > SIMILARITY_FLOOR:
        return "I don't know — I couldn't find anything about that in the documents.", []

    context_blocks = []
    used_sources = []
    for chunk, source, _ in retrieved:
        context_blocks.append(f"[Source: {source}]\n{chunk}")
        used_sources.append(source)

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a documentation assistant. Answer the question using ONLY the context below.
If the answer is not contained in the context, say "I don't know" — do not make anything up.
Always mention which source file(s) your answer came from.

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"], list(dict.fromkeys(used_sources))


def log_trace(question, method, retrieved, answer, sources):
    """
    Append one complete record of this request to traces.jsonl.

    This is the "trace" concept from Week 5: enough detail to fully understand
    (and replay) what happened for this question, without re-running anything.
    JSONL (one JSON object per line) is used so you can append cheaply and
    read it back one trace at a time.
    """
    trace = {
        "timestamp": time.time(),
        "question": question,
        "method": method,
        "retrieved": [
            {"source": source, "distance": round(distance, 4), "chunk_preview": chunk[:200]}
            for chunk, source, distance in retrieved
        ],
        "answer": answer,
        "sources": sources,
    }
    with open(TRACES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Your question, in quotes.")
    parser.add_argument(
        "--method", choices=list(RETRIEVERS.keys()), default="embedding",
        help="Retrieval method to use. Default: embedding (same as before)."
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Skip writing this query to traces.jsonl."
    )
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    retrieve_fn = RETRIEVERS[args.method]
    retrieved = retrieve_fn(args.question, collection)

    print(f"--- Retrieved chunks (method={args.method}) ---")
    for chunk, source, distance in retrieved:
        print(f"[{source}] distance={distance:.3f}\n{chunk[:120]}...\n")

    answer, sources = generate_answer(args.question, retrieved)

    print("--- Answer ---")
    print(answer)
    if sources:
        print(f"\nSource(s): {', '.join(sources)}")

    if not args.no_log:
        log_trace(args.question, args.method, retrieved, answer, sources)
        print(f"\n(logged to {TRACES_FILE})")


if __name__ == "__main__":
    main()
