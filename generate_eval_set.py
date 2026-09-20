"""
generate_eval_set.py -- build a first-draft eval_set.jsonl from traces_review.csv.

Rules:
- Every failure_type != "success" becomes a REGRESSION test.
- Every failure_type == "success" becomes a BASELINE test.
- expected_facts are extracted from the real source docs -- nothing is invented.
- Rows where we could not confidently extract a fact are tagged "needs_review": true.

Usage:
    python generate_eval_set.py
    # -> writes eval_set.jsonl; review needs_review=true rows by hand before running evals
"""

import csv
import json

INPUT_CSV  = "traces_review.csv"
OUTPUT_JSONL = "eval_set.jsonl"

# ---------------------------------------------------------------------------
# Ground-truth facts extracted directly from docs/
# Each key is a lowercase substring of the question that identifies the case.
# ---------------------------------------------------------------------------
KNOWN_FACTS = {
    # authentication.md: valid key with insufficient scope -> 403
    "read-only key but i tried to make a write": {
        "expected_facts": ["403"],
        "expected_source": "authentication.md",
        "expected_behavior": "must_not_refuse_and_contain_fact",
    },
    # authentication.md: 429 = rate limit, NOT invalid key
    "429 errors. does that mean my api key is invalid": {
        "expected_facts": ["rate limit", "429"],
        "expected_source": "authentication.md",
        "expected_behavior": "must_not_refuse_and_contain_fact",
    },
    # authentication.md: 429 = rate limit, key was NOT revoked
    "started getting 429s. was my key revoked": {
        "expected_facts": ["rate limit", "429"],
        "expected_source": "authentication.md",
        "expected_behavior": "must_not_refuse_and_contain_fact",
    },
    # authentication.md: rotating does NOT copy scope; new key may have different scope
    "rotate my api key, does the new key automatically get the same scope": {
        "expected_facts": ["scope"],
        "expected_source": "authentication.md",
        "expected_behavior": "must_not_refuse_and_contain_fact",
    },
    # authentication.md: "API keys do not expire automatically"
    "do api keys expire automatically after 90 days": {
        "expected_facts": ["do not expire", "not expire", "recommended"],
        "expected_source": "authentication.md",
        "expected_behavior": "must_not_refuse_and_contain_fact",
    },
    # authentication.md: "Once revoked, a key cannot be used again"
    "can i use a revoked api key again": {
        "expected_facts": ["cannot", "revoked"],
        "expected_source": "authentication.md",
        "expected_behavior": "must_not_refuse_and_contain_fact",
    },
    # authentication.md: API key, access key, bearer token are equivalent names
    "access key' and my code expects an 'api key'": {
        "expected_facts": ["equivalent", "same"],
        "expected_source": "authentication.md",
        "expected_behavior": "must_not_refuse_and_contain_fact",
    },
    # installation.md: requires Python 3.9+, so 3.8 is not supported
    "does acmesdk work with python 3.8": {
        "expected_facts": ["3.9"],
        "expected_source": "installation.md",
        "expected_behavior": "must_not_refuse_and_contain_fact",
    },
}

SUCCESS_FACTS = {
    "how do i rotate my api key": {
        "expected_facts": ["90 days", "revoke"],
        "expected_source": "authentication.md",
    },
    "how many times can a failed job be retried": {
        "expected_facts": ["5"],
        "expected_source": "jobs-api.md",
    },
    "if my api key is invalid, what status code": {
        "expected_facts": ["401"],
        "expected_source": "authentication.md",
    },
    "during a key rotation, can i use both the old and new key": {
        "expected_facts": ["two active", "zero-downtime"],
        "expected_source": "authentication.md",
    },
    "what does a 429 response mean": {
        "expected_facts": ["rate limit", "Retry-After"],
        "expected_source": "authentication.md",
    },
    "how can i check if my environment is set up correctly": {
        "expected_facts": ["acme-sdk doctor"],
        "expected_source": "installation.md",
    },
    "what's the default api endpoint": {
        "expected_facts": ["api.acme.dev"],
        "expected_source": "installation.md",
    },
    "are failed jobs retried automatically": {
        "expected_facts": ["not retried automatically", "client.jobs.retry"],
        "expected_source": "jobs-api.md",
    },
    "what happens if i try to delete a job that's still running": {
        "expected_facts": ["JobStillRunningError"],
        "expected_source": "jobs-api.md",
    },
    "what priority levels can i set when creating a job": {
        "expected_facts": ["low", "normal", "high"],
        "expected_source": "jobs-api.md",
    },
    "how do i know when a job finishes without constantly polling": {
        "expected_facts": ["webhook"],
        "expected_source": "jobs-api.md",
    },
    "what header should i check to verify a webhook payload": {
        "expected_facts": ["X-Acme-Signature"],
        "expected_source": "jobs-api.md",
    },
}


def find_facts(question, lookup):
    """Match a question to a facts entry by case-insensitive substring."""
    q = question.lower()
    for key, val in lookup.items():
        if key.lower() in q:
            return val
    return None


def build_case(row_index, row, is_failure):
    question = row["question"].strip()
    problem_type = (row.get("problem_group") or row.get("failure_type") or "unknown").strip()

    case = {
        "id": f"tc_{row_index:03d}",
        "question": question,
        "problem_type": problem_type,
        "origin": f"traces_review.csv:row{row_index}",
    }

    if is_failure:
        entry = find_facts(question, KNOWN_FACTS)
        if entry:
            case.update(entry)
            case["needs_review"] = False
        else:
            case["expected_behavior"] = "must_not_refuse_and_contain_fact"
            case["expected_facts"]    = []   # fill in by hand
            case["expected_source"]   = (row.get("sources_cited") or "").split(";")[0].strip()
            case["needs_review"]      = True
            case["_flag"]             = "Could not find grounded expected_facts in docs; fill in by hand"
    else:
        entry = find_facts(question, SUCCESS_FACTS)
        if entry:
            case["expected_behavior"] = "must_cite_source_and_contain_fact"
            case["expected_facts"]    = entry["expected_facts"]
            case["expected_source"]   = entry["expected_source"]
            case["needs_review"]      = False
        else:
            case["expected_behavior"] = "must_cite_source"
            case["expected_facts"]    = []
            case["expected_source"]   = (row.get("sources_cited") or "").split(";")[0].strip()
            case["needs_review"]      = True
            case["_flag"]             = "Success row with no known facts; fill in expected_facts by hand"

    return case


def main():
    cases = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            is_failure = (row.get("failure_type") or "").strip() != "success"
            cases.append(build_case(i, row, is_failure))

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    n_reg  = sum(1 for c in cases if "must_not_refuse" in c.get("expected_behavior",""))
    n_base = len(cases) - n_reg
    n_flag = sum(1 for c in cases if c.get("needs_review"))

    print(f"Written {len(cases)} cases to {OUTPUT_JSONL}")
    print(f"  {n_reg} regression tests (failures to fix)")
    print(f"  {n_base} baseline tests (must not regress)")
    if n_flag:
        print(f"\n  WARNING: {n_flag} case(s) tagged needs_review=true -- open eval_set.jsonl and fill in expected_facts.")
    else:
        print("\n  All cases grounded in docs -- no manual review needed.")


if __name__ == "__main__":
    main()
