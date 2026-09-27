"""
STEP 1 — Trajectory Eval Set
==============================
7 questions against the real indexed docs.  For each question we specify in
advance which retrieval method is "correct" given the nature of the query, then
run it through run_agent_rag() and inspect the first retrieve call the model
actually chose.  We compute tool-choice accuracy = right / total.

Rationale for expected methods
-------------------------------
Q1 – "401 Unauthorized" / "403 Forbidden": exact error-code terms → bm25
Q2 – what is scope on a key: conceptual/semantic → embedding
Q3 – retry(job_id) function name: exact function name → bm25
Q4 – how do webhooks work: conceptual → embedding
Q5 – pip install acme-sdk[async]: exact token → bm25
Q6 – difference between auth and authz: conceptual comparison → embedding
Q7 – rate-limit retry header: hybrid (keyword + conceptual) → hybrid

Results appended to traces.jsonl with mode="trajectory_eval".
"""

import json
import time
import sys
import chromadb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_agent import run_agent_rag, log_trace, CHROMA_PATH, COLLECTION_NAME

TRAJECTORY_QUESTIONS = [
    {
        "id": "T1",
        "question": "What HTTP status code does the API return when an API key is missing or revoked?",
        "expected_method": "bm25",
        "rationale": "Exact token '401' and 'revoked' — BM25 keyword match wins",
    },
    {
        "id": "T2",
        "question": "What does it mean that scope is a property of the key, not the request?",
        "expected_method": "embedding",
        "rationale": "Conceptual/semantic explanation — embedding search fits better",
    },
    {
        "id": "T3",
        "question": "How do I call retry on a failed job using the client object?",
        "expected_method": "bm25",
        "rationale": "Exact function name 'client.jobs.retry' — BM25 keyword match",
    },
    {
        "id": "T4",
        "question": "How does webhook event delivery work for job status changes?",
        "expected_method": "embedding",
        "rationale": "Conceptual how-to — embedding semantic match fits",
    },
    {
        "id": "T5",
        "question": "What is the pip install command for async support in acme-sdk?",
        "expected_method": "bm25",
        "rationale": "Exact token 'acme-sdk[async]' — BM25 keyword match",
    },
    {
        "id": "T6",
        "question": "What is the conceptual difference between authentication and authorization in this SDK?",
        "expected_method": "embedding",
        "rationale": "Abstract conceptual comparison — embedding search",
    },
    {
        "id": "T7",
        "question": "What header should a client read when it receives a rate limit response and how long should it wait?",
        "expected_method": "hybrid",
        "rationale": "Mix of exact term 'Retry-After' header + conceptual usage — hybrid",
    },
]

RESULT_FILE = "trajectory_eval_results.jsonl"


def first_retrieve_method(steps):
    """Extract the method used in the first retrieve tool call."""
    for s in steps:
        if s.get("tool") == "retrieve":
            return s["args"].get("method")
    return None


def run_trajectory_eval():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    results = []
    print(f"\n{'='*60}")
    print(f"STEP 1 — Trajectory Eval  ({len(TRAJECTORY_QUESTIONS)} questions)")
    print(f"{'='*60}\n")

    for q in TRAJECTORY_QUESTIONS:
        print(f"[{q['id']}] {q['question'][:70]}...")
        print(f"       Expected method: {q['expected_method']}")

        start = time.time()
        answer, sources, steps, tokens = run_agent_rag(q["question"], collection)
        elapsed = time.time() - start

        actual_method = first_retrieve_method(steps)
        correct = actual_method == q["expected_method"]
        all_methods = [s["args"].get("method") for s in steps if s.get("tool") == "retrieve"]

        print(f"       Actual method  : {actual_method}  ({'CORRECT' if correct else 'WRONG'})")
        print(f"       All methods    : {all_methods}")
        print(f"       Answer preview : {answer[:80]}...")
        print(f"       Time: {elapsed:.1f}s\n")

        row = {
            "id": q["id"],
            "question": q["question"],
            "expected_method": q["expected_method"],
            "actual_first_method": actual_method,
            "all_methods": all_methods,
            "correct": correct,
            "answer_preview": answer[:120],
            "sources": sources,
            "steps": steps,
            "tokens": tokens,
            "seconds": round(elapsed, 3),
        }
        results.append(row)

        log_trace(q["question"], "agent", None, steps, answer, sources, elapsed, tokens=tokens)

        with open(RESULT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        time.sleep(1)

    correct_count = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct_count / total

    print(f"\n{'='*60}")
    print(f"TOOL-CHOICE ACCURACY: {correct_count}/{total} = {accuracy:.1%}")
    print(f"{'='*60}")
    print(f"\nPer-question breakdown:")
    for r in results:
        status = "OK" if r["correct"] else "MISS"
        print(f"  {status} [{r['id']}] expected={r['expected_method']:9s}  "
              f"actual={str(r['actual_first_method']):9s}  "
              f"q={r['question'][:50]}...")

    summary = {
        "step": 1,
        "timestamp": time.time(),
        "correct": correct_count,
        "total": total,
        "accuracy": round(accuracy, 4),
        "results": results,
    }
    with open("step1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n-> Full results saved to step1_summary.json and {RESULT_FILE}")
    return summary


if __name__ == "__main__":
    run_trajectory_eval()
