"""
checks.py -- rule-based, deterministic assertion checks for the eval harness.

Each check function takes the eval case (from eval_set.jsonl) and the live result
from the app, and returns a dict:
    {
        "check": <check_name>,
        "passed": True | False,
        "reason": <short human-readable explanation>
    }

WHY rule-based first:
  - They are free (no LLM calls), fast, and fully reproducible.
  - They catch the most common failure modes without any judge variance.
  - The LLM judge (Phase 3) only handles what rules cannot: tone, helpfulness, etc.

Usage (mostly called from run_evals.py, but can be tested standalone):
    python checks.py
"""

import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_not_empty(case: dict, result: dict) -> dict:
    """
    The answer must not be blank.
    This catches total failures (empty string, None, whitespace-only).
    """
    answer = (result.get("answer") or "").strip()
    passed = len(answer) > 0
    return {
        "check": "not_empty",
        "passed": passed,
        "reason": "Answer is non-empty" if passed else "Answer is empty or missing",
    }


def check_not_refuse(case: dict, result: dict) -> dict:
    """
    The answer must NOT be a refusal ('I don't know').

    Used for 'Cautious Generation' regression cases where the answer IS in the
    docs -- a refusal here means the model is being too conservative.
    We check for common refusal patterns, not just the exact phrase.
    """
    answer = (result.get("answer") or "").lower()
    # Patterns that indicate the model gave up instead of answering
    refusal_patterns = [
        r"i don.?t know",
        r"i couldn.?t find",
        r"i cannot find",
        r"no information",
        r"not mentioned in",
        r"not covered in",
        r"not provided in",
    ]
    refused = any(re.search(p, answer) for p in refusal_patterns)
    passed = not refused
    return {
        "check": "not_refuse",
        "passed": passed,
        "reason": "Answer does not refuse" if passed else "Answer is a refusal (I don't know / couldn't find)",
    }


def check_contains_fact(case: dict, result: dict) -> dict:
    """
    At least ONE of the expected_facts must appear in the answer (case-insensitive).

    WHY 'any' not 'all': multiple expected_facts are synonyms / alternative
    phrasings -- we only need one to confirm the right fact was stated.
    For example: ["do not expire", "not expire", "recommended"] all capture
    the same doc fact about 90-day rotation being a recommendation, not auto-expiry.
    """
    expected_facts = case.get("expected_facts", [])
    if not expected_facts:
        # No facts to check -- skip this assertion (treat as passed)
        return {
            "check": "contains_fact",
            "passed": True,
            "reason": "No expected_facts defined -- skipped",
        }

    answer = (result.get("answer") or "").lower()
    matched = [f for f in expected_facts if f.lower() in answer]
    passed = len(matched) > 0
    return {
        "check": "contains_fact",
        "passed": passed,
        "reason": (
            f"Found expected fact(s): {matched}"
            if passed
            else f"None of expected_facts found in answer: {expected_facts}"
        ),
    }


def check_cites_source(case: dict, result: dict) -> dict:
    """
    The expected_source doc must appear in the list of sources the app cited.

    'sources' in the result is the list returned by generate_answer() in query.py.
    We do a case-insensitive substring match so 'authentication.md' matches
    even if the model lowercased it.
    """
    expected_source = (case.get("expected_source") or "").strip()
    if not expected_source:
        return {
            "check": "cites_source",
            "passed": True,
            "reason": "No expected_source defined -- skipped",
        }

    cited_sources = [s.lower() for s in (result.get("sources") or [])]
    passed = expected_source.lower() in cited_sources
    return {
        "check": "cites_source",
        "passed": passed,
        "reason": (
            f"Cited expected source '{expected_source}'"
            if passed
            else f"Expected source '{expected_source}' not in cited: {result.get('sources', [])}"
        ),
    }


def check_cited_source_was_retrieved(case: dict, result: dict) -> dict:
    """
    Every source cited in the answer must have actually been retrieved.

    WHY: If the model cites a source that wasn't in the retrieved context,
    it hallucinated a citation. This is a faithfulness failure.

    'retrieved' in the result is a list of {source, distance, chunk_preview}
    as logged by log_trace() in query.py.
    """
    cited_sources = set(s.lower() for s in (result.get("sources") or []))
    retrieved_sources = set(
        r["source"].lower()
        for r in (result.get("retrieved") or [])
    )

    # Sources that were cited but NOT retrieved
    hallucinated = cited_sources - retrieved_sources

    passed = len(hallucinated) == 0
    return {
        "check": "cited_source_was_retrieved",
        "passed": passed,
        "reason": (
            "All cited sources were retrieved"
            if passed
            else f"Hallucinated citations (cited but not retrieved): {hallucinated}"
        ),
    }


# ---------------------------------------------------------------------------
# Dispatcher: run the right checks for each expected_behavior
# ---------------------------------------------------------------------------

# Map each expected_behavior value to the list of checks to run.
# Always run not_empty and cited_source_was_retrieved as guards.
BEHAVIOR_CHECKS = {
    "must_not_refuse_and_contain_fact": [
        check_not_empty,
        check_not_refuse,
        check_contains_fact,
        check_cited_source_was_retrieved,
    ],
    "must_cite_source_and_contain_fact": [
        check_not_empty,
        check_cites_source,
        check_contains_fact,
        check_cited_source_was_retrieved,
    ],
    "must_cite_source": [
        check_not_empty,
        check_cites_source,
        check_cited_source_was_retrieved,
    ],
    "must_refuse": [
        # Future: for truly out-of-scope questions the model should refuse
        # Add a check_does_refuse() here if needed
        check_not_empty,
    ],
}


def run_checks(case: dict, result: dict) -> list[dict]:
    """
    Run all checks appropriate for this case's expected_behavior.
    Returns a list of check result dicts.
    """
    behavior = case.get("expected_behavior", "must_cite_source")
    check_fns = BEHAVIOR_CHECKS.get(behavior, [check_not_empty])
    return [fn(case, result) for fn in check_fns]


def case_passed(check_results: list[dict]) -> bool:
    """A case passes only if EVERY check passes."""
    return all(r["passed"] for r in check_results)


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load a few cases from eval_set.jsonl and run checks against synthetic results
    print("=== checks.py self-test ===\n")

    # Simulate a GOOD result for tc_003 (write-key 403 question)
    good_result = {
        "answer": "A read-only key cannot perform write operations. The API returns 403 Forbidden. [Source: authentication.md]",
        "sources": ["authentication.md"],
        "retrieved": [{"source": "authentication.md", "distance": 0.23, "chunk_preview": "..."}],
    }

    # Simulate a BAD result (refusal) for tc_003
    bad_result = {
        "answer": "I don't know -- I couldn't find anything about that in the documents.",
        "sources": ["authentication.md"],
        "retrieved": [{"source": "authentication.md", "distance": 0.23, "chunk_preview": "..."}],
    }

    test_case = {
        "id": "tc_003",
        "question": "I have a valid read-only key but I tried to make a write request. What error do I get?",
        "expected_behavior": "must_not_refuse_and_contain_fact",
        "expected_facts": ["403"],
        "expected_source": "authentication.md",
    }

    print(f"Case: {test_case['id']} -- {test_case['question'][:60]}")
    print("\n-- Good result (should all pass) --")
    results = run_checks(test_case, good_result)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['check']}: {r['reason']}")

    print("\n-- Bad result (not_refuse and contains_fact should fail) --")
    results = run_checks(test_case, bad_result)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['check']}: {r['reason']}")

    print("\n=== Self-test done ===")
