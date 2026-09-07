"""
eval.py — measure retrieval quality with a real number: hit-rate@3.

hit-rate@3 = (number of test questions where the correct document appeared
              anywhere in the top-3 retrieved chunks) / (total test questions)

This is how you PROVE a change helped, instead of guessing. Run this before
your Week 4 change and after it — the two numbers are your before/after.

Usage:
    python eval.py --method embedding
    python eval.py --method bm25
    python eval.py --method hybrid
"""

import argparse

import chromadb

from query import RETRIEVERS

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "dev_docs"

# ---------- your labeled test set ----------
# Each entry: a question, and which source file contains the correct answer.
# Add more of these as you find real failing questions from actual usage —
# that's exactly what Week 5's trace reading will surface.
TEST_SET = [
    {"question": "How do I rotate my API key?", "expected_source": "authentication.md"},
    {"question": "What happens if I exceed the rate limit?", "expected_source": "authentication.md"},
    {"question": "What scopes can an API key have?", "expected_source": "authentication.md"},
    {"question": "What Python version is required?", "expected_source": "installation.md"},
    {"question": "How do I install the SDK with async support?", "expected_source": "installation.md"},
    {"question": "How do I point the client at a different region?", "expected_source": "installation.md"},
    {"question": "How many times can a failed job be retried?", "expected_source": "jobs-api.md"},
    {"question": "What statuses can a job have?", "expected_source": "jobs-api.md"},
    {"question": "How do I verify a webhook payload is legitimate?", "expected_source": "jobs-api.md"},
    {"question": "Can I delete a job that's still running?", "expected_source": "jobs-api.md"},
]


def run_eval(method: str):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    retrieve_fn = RETRIEVERS[method]

    hits = 0
    print(f"Running eval with method='{method}' on {len(TEST_SET)} questions...\n")

    for case in TEST_SET:
        question = case["question"]
        expected = case["expected_source"]

        retrieved = retrieve_fn(question, collection, top_k=3)
        retrieved_sources = [source for _, source, _ in retrieved]

        hit = expected in retrieved_sources
        hits += hit

        status = "HIT " if hit else "MISS"
        print(f"[{status}] {question}")
        print(f"        expected: {expected} | got top-3: {retrieved_sources}")

    hit_rate = hits / len(TEST_SET)
    print(f"\n=== hit-rate@3 ({method}): {hit_rate:.2%} ({hits}/{len(TEST_SET)}) ===")
    return hit_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(RETRIEVERS.keys()), default="embedding")
    args = parser.parse_args()
    run_eval(args.method)