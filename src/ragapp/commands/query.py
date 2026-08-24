"""The `query` command. CLI concerns only — actual logic lives in ragapp.retrieval / store."""

from ragapp.retrieval import generate_answer, retrieve
from ragapp.store import get_or_create_collection


def query(question: str):
    """Ask a question. Answers ONLY from the ingested documents, with a source citation."""
    collection = get_or_create_collection(fresh=False)
    retrieved = retrieve(question, collection)

    print("--- Retrieved chunks (for debugging / transparency) ---")
    for chunk, source, distance in retrieved:
        print(f"[{source}] distance={distance:.3f}\n{chunk[:120]}...\n")

    answer, sources = generate_answer(question, retrieved)

    print("--- Answer ---")
    print(answer)
    if sources:
        print(f"\nSource(s): {', '.join(sources)}")