"""Retrieve -> Generate (grounded, with citation). Pure logic, no printing."""

import ollama

from ragapp.config import CHAT_MODEL, SIMILARITY_FLOOR, TOP_K
from ragapp.embeddings import embed


def retrieve(question: str, collection, top_k: int = TOP_K) -> list[tuple[str, str, float]]:
    """Find the top_k most relevant chunks for the question. Returns (chunk, source, distance)."""
    query_vector = embed(question)
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)

    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    distances = results["distances"][0]  # lower = more similar (cosine distance)
    return list(zip(chunks, sources, distances))


def generate_answer(question: str, retrieved: list[tuple[str, str, float]]) -> tuple[str, list[str]]:
    """
    Ask the local LLM to answer ONLY from the retrieved chunks, and cite the source.

    Returns (answer_text, list_of_source_filenames). If nothing retrieved is close
    enough, the LLM is never called — this is what makes "I don't know" reliable
    instead of hoping the model follows instructions.
    """
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

    response = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"], list(dict.fromkeys(used_sources))