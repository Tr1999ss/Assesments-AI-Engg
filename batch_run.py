"""
batch_run.py — run a batch of questions through query.py's pipeline and log them all
as traces, so you have a realistic ~20-trace batch to read for Week 5 (and material
to find real failures in for Week 4), instead of typing questions one at a time.

Mix of easy questions, tricky ones, and genuinely out-of-scope ones — real usage
looks like this, not just the questions you already know work.

Usage:
    python batch_run.py                  # uses embedding retrieval (default)
    python batch_run.py --method hybrid  # compare against your Week 4 improvement
"""

import argparse

import chromadb

from query import RETRIEVERS, generate_answer, log_trace

QUESTIONS = [
    # straightforward — should work fine
    "How do I rotate my API key?",
    "What Python version is required?",
    "How do I install the SDK?",
    "What job statuses exist?",
    "How do I delete a job?",
    # trickier phrasing of the same info — tests retrieval robustness
    "my key might be compromised, what do I do",
    "steps for zero downtime when swapping keys",
    "does this work with python 3.8",
    "async support?",
    "what happens when a job fails",
    "job retry limit",
    "can I cancel a running job",
    # exact terms / codes — good candidates for BM25 to help with
    "what does 429 mean",
    "what is X-Acme-Signature",
    "JobStillRunningError",
    # vague / conversational — likely to stress the pipeline
    "hi",
    "help",
    "what can you do",
    # genuinely out of scope — should say "I don't know"
    "what's the weather today",
    "who won the world cup",
    "what's your favorite color",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(RETRIEVERS.keys()), default="embedding")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("dev_docs")
    retrieve_fn = RETRIEVERS[args.method]

    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {question}")
        retrieved = retrieve_fn(question, collection)
        answer, sources = generate_answer(question, retrieved)
        log_trace(question, args.method, retrieved, answer, sources)

    print(f"\nDone. Logged {len(QUESTIONS)} traces to traces.jsonl (method={args.method}).")
    print("Next: python export_traces.py --sample 20")


if __name__ == "__main__":
    main()