import sys

import chromadb
import ollama

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "dev_docs"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma4:latest"  # matches what "ollama list" shows on this machine
TOP_K = 3
# If the closest chunk isn't at least this similar, we treat it as "no relevant doc found"
SIMILARITY_FLOOR = 0.35


def embed(text):
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def retrieve(question, collection, top_k=TOP_K):
    """Find the top_k most relevant chunks for the question."""
    query_vector = embed(question)
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)

    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    distances = results["distances"][0]  # lower = more similar (Chroma default: cosine distance)
    return list(zip(chunks, sources, distances))


def generate_answer(question, retrieved):
    """Ask the local LLM to answer ONLY from the retrieved chunks, and cite the source."""
    # Guardrail: if nothing retrieved is actually close enough, don't even ask the model —
    # this is what lets the app say "I don't know" instead of the model inventing an answer.
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
    return response["message"]["content"], list(dict.fromkeys(used_sources))  # dedupe, keep order


def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    retrieved = retrieve(question, collection)

    print("--- Retrieved chunks (for debugging / transparency) ---")
    for chunk, source, distance in retrieved:
        print(f"[{source}] distance={distance:.3f}\n{chunk[:120]}...\n")

    answer, sources = generate_answer(question, retrieved)

    print("--- Answer ---")
    print(answer)
    if sources:
        print(f"\nSource(s): {', '.join(sources)}")


if __name__ == "__main__":
    main()
