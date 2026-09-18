"""
eval_golden.py — run the golden set through the full pipeline (retrieve + generate)
and check not just "was the right doc found" but "did the model actually get the
nuance right, or fall into a known trap."

This is a different, stricter check than eval.py's hit-rate@3. A question can pass
hit-rate@3 (correct doc retrieved) and still fail here (model still got the answer
wrong from that correct doc) - that gap IS the generation-failure category from
Week 4.

Usage:
    python eval_golden.py --method embedding
    python eval_golden.py --method hybrid
"""

import argparse

import chromadb

from golden_set import GOLDEN_SET
from query import RETRIEVERS, generate_answer, log_trace
import re


def check_answer(answer: str, expected_keywords: list, trap_keywords: list):
    """
    Whole-word matching, not substring matching — otherwise short trap words
    like "no" false-match inside unrelated words like "know" (as in "I don't
    know"), producing a fake trap detection.
    """
    answer_lower = answer.lower()

    def contains_whole_word(text, phrase):
        # multi-word phrases (e.g. "rate limit") still use simple substring
        # matching; single words get a word-boundary regex to avoid the
        # "no" inside "know" problem
        if " " in phrase:
            return phrase.lower() in text
        return re.search(rf"\b{re.escape(phrase.lower())}\b", text) is not None

    found_expected = [kw for kw in expected_keywords if contains_whole_word(answer_lower, kw)]
    found_traps = [kw for kw in trap_keywords if contains_whole_word(answer_lower, kw)]

    missing_expected = [kw for kw in expected_keywords if kw not in found_expected]

    if found_traps:
        return "TRAP_HIT", found_expected, found_traps
    if not missing_expected:
        return "PASS", found_expected, found_traps
    return "FAIL", found_expected, found_traps


def run_golden_eval(method: str):
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("dev_docs")
    retrieve_fn = RETRIEVERS[method]

    results = {"PASS": 0, "FAIL": 0, "TRAP_HIT": 0}

    print(f"Running golden set eval with method='{method}' on {len(GOLDEN_SET)} questions...\n")

    for case in GOLDEN_SET:
        question = case["question"]
        retrieved = retrieve_fn(question, collection, top_k=3)
        retrieved_sources = [source for _, source, _ in retrieved]
        retrieved_distances = [round(d, 3) for _, _, d in retrieved]
        retrieval_hit = case["expected_source"] in retrieved_sources
        best_distance = min(retrieved_distances)

        answer, sources = generate_answer(question, retrieved,"openrouter")
        status, found_expected, found_traps = check_answer(
            answer, case["expected_keywords"], case["trap_keywords"]
        )
        results[status] += 1

        log_trace(question, method, retrieved, answer, sources)

        print(f"[{status}] {question}")
        print(f"    retrieval hit: {retrieval_hit} | best_distance={best_distance} "
              f"(expected {case['expected_source']}, got {retrieved_sources})")
        print(f"    answer: {answer[:150]}...")
        if status == "TRAP_HIT":
            print(f"    !! trap keyword(s) found: {found_traps} -- {case['note']}")
        elif status == "FAIL":
            print(f"    missing expected keyword(s): {[k for k in case['expected_keywords'] if k not in found_expected]}")
        print()

    total = len(GOLDEN_SET)
    print("=== Golden set results ===")
    print(f"PASS:     {results['PASS']}/{total}")
    print(f"FAIL:     {results['FAIL']}/{total}  (missed the expected answer)")
    print(f"TRAP_HIT: {results['TRAP_HIT']}/{total}  (fell into a known confusion)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(RETRIEVERS.keys()), default="embedding")
    args = parser.parse_args()
    run_golden_eval(args.method)