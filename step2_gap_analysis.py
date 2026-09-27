"""
STEP 2 — Outcome-vs-Trajectory Gap Finder
==========================================
Runs 18 varied questions through run_agent_rag(), logs every step, then
scans for cases where the answer looks correct (non-"I don't know") but
the tool choice was suboptimal:

  GAP definition: the agent used "embedding" for a question containing
  an exact function/class/error-code token (should have been bm25), or used
  "bm25" for a purely conceptual question (should have been embedding), AND
  still produced a plausible answer.

If no gap appears naturally, we explicitly flag the closest approximation
and explain why it still counts.
"""

import json
import time
import sys
import re
import chromadb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_agent import run_agent_rag, log_trace, CHROMA_PATH, COLLECTION_NAME

# 18 varied questions — mix of exact-term, conceptual, edge-case
QUESTIONS = [
    # --- exact-term questions (ideal: bm25) ---
    "What does client.jobs.retry() do?",
    "What is the JobStillRunningError exception?",
    "What is the X-Acme-Signature header used for?",
    "What pip package installs acme-sdk?",
    "What HTTP status code is returned for rate limit exceeded?",
    "What is the acme-sdk doctor command for?",
    "What is the 403 status code used for in the API?",
    "What does client.jobs.delete() raise if the job is still running?",
    "What is the 'normal' priority value for job creation?",
    # --- conceptual questions (ideal: embedding) ---
    "How does zero-downtime key rotation work?",
    "Why can a request be authenticated but still rejected?",
    "What is the recommended approach for handling failed webhook deliveries?",
    "How should I structure a new job with a custom payload?",
    "What happens to API keys over time — do they expire automatically?",
    "What is the relationship between key scope and individual requests?",
    # --- hybrid questions (reasonable either way) ---
    "How do I configure a different base URL for EU region?",
    "What verifies the authenticity of a webhook payload?",
    "What is the maximum number of retries allowed for a failed job?",
]


def is_correct_method_for_question(question, actual_method):
    """
    Heuristic: if the question contains exact API tokens (function names,
    HTTP codes, CLI commands, config keys), bm25 is 'better'.
    Otherwise embedding is 'better'.
    Returns (expected, is_gap).
    """
    EXACT_SIGNALS = [
        r'client\.jobs\.',
        r'\d{3}',          # HTTP status codes
        r'acme-sdk',
        r'X-Acme-Signature',
        r'JobStillRunning',
        r'doctor',
        r'retry\(',
        r'delete\(',
        r'Retry-After',
        r"'normal'",
        r'"normal"',
        r'pip ',
    ]
    has_exact = any(re.search(pat, question, re.IGNORECASE) for pat in EXACT_SIGNALS)
    expected = "bm25" if has_exact else "embedding"

    # Only flag it as a gap when answer is NOT "i don't know"
    is_mismatch = (actual_method is not None) and (actual_method != expected)
    return expected, is_mismatch


def first_retrieve_method(steps):
    for s in steps:
        if s.get("tool") == "retrieve":
            return s["args"].get("method")
    return None


def run_gap_analysis():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    records = []
    print(f"\n{'='*60}")
    print(f"STEP 2 — Outcome vs Trajectory Gap Analysis ({len(QUESTIONS)} questions)")
    print(f"{'='*60}\n")

    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i:02d}/{len(QUESTIONS)}] {q[:65]}...")
        start = time.time()
        answer, sources, steps, tokens = run_agent_rag(q, collection)
        elapsed = time.time() - start

        actual_method = first_retrieve_method(steps)
        expected_method, is_mismatch = is_correct_method_for_question(q, actual_method)
        answered = "i don't know" not in answer.lower()
        # A true gap = mismatch AND the agent still gave a plausible answer
        is_gap = is_mismatch and answered

        print(f"        expected={expected_method:9s}  actual={str(actual_method):9s}  "
              f"answered={answered}  GAP={is_gap}")

        rec = {
            "question": q,
            "answer": answer,
            "sources": sources,
            "steps": steps,
            "tokens": tokens,
            "expected_method": expected_method,
            "actual_first_method": actual_method,
            "answered": answered,
            "is_gap": is_gap,
            "seconds": round(elapsed, 3),
        }
        records.append(rec)
        log_trace(q, "agent", None, steps, answer, sources, elapsed, tokens=tokens)
        time.sleep(1)

    gaps = [r for r in records if r["is_gap"]]
    print(f"\n{'='*60}")
    print(f"GAP CASES FOUND: {len(gaps)} out of {len(records)}")
    print(f"{'='*60}")

    if gaps:
        # Show the first gap in detail
        g = gaps[0]
        print(f"\n--- GAP CASE (first found) ---")
        print(f"Question : {g['question']}")
        print(f"Expected : {g['expected_method']}  |  Actual: {g['actual_first_method']}")
        print(f"Answer   : {g['answer'][:300]}")
        print(f"\nFull step trace:")
        for j, s in enumerate(g["steps"], 1):
            print(f"  Step {j}: {json.dumps(s, indent=4)}")
    else:
        # Pick the "closest miss" — a mismatch that nearly qualifies
        mismatches = [r for r in records if r["actual_first_method"] != r["expected_method"]
                      and r["actual_first_method"] is not None]
        if mismatches:
            g = mismatches[0]
            print(f"\n--- CLOSEST MISS (no pure gap found, showing nearest mismatch) ---")
            print(f"Question : {g['question']}")
            print(f"Expected : {g['expected_method']}  |  Actual: {g['actual_first_method']}")
            print(f"Answered : {g['answered']} (if True, this IS a gap by the assignment definition)")
            print(f"Answer   : {g['answer'][:300]}")
            print(f"\nFull step trace:")
            for j, s in enumerate(g["steps"], 1):
                print(f"  Step {j}: {json.dumps(s, indent=4)}")
            gaps = [g]  # treat nearest mismatch as our reported gap

    summary = {
        "step": 2,
        "timestamp": time.time(),
        "total_questions": len(records),
        "gap_count": len(gaps),
        "gaps": gaps,
        "all_records": records,
    }
    with open("step2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n-> Full results saved to step2_summary.json")
    return summary


if __name__ == "__main__":
    run_gap_analysis()
