"""
run_evals.py -- run the app on every eval case and score the results.

Usage:
    python run_evals.py                             # uses PROVIDER from .env
    python run_evals.py --provider openrouter       # force openrouter
    python run_evals.py --method hybrid             # retrieval method
    python run_evals.py --run-name before_fix       # label this run
    python run_evals.py --no-judge                  # skip LLM judge (rule checks only)

Results are saved to results/<run_name>.json.
Print a per-problem-type pass-rate table plus an overall score.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import chromadb

# Reuse the exact same retrieval and generation pipeline from query.py
# (import functions directly -- no duplication)
from query import (
    CHROMA_PATH,
    COLLECTION_NAME,
    RETRIEVERS,
    GENERATORS,
    OPENROUTER_MODEL,
    OLLAMA_CHAT_MODEL,
    generate_answer,
    PROVIDER as DEFAULT_PROVIDER,
)
from checks import run_checks, case_passed
from judge import run_judge

EVAL_SET   = "eval_set.jsonl"
RESULTS_DIR = Path("results")


def load_eval_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_one(case: dict, collection, method: str, provider: str, use_judge: bool) -> dict:
    """
    Run the app on one eval case and return a result dict with all scores.
    """
    question = case["question"]
    retrieve_fn = RETRIEVERS[method]

    # --- Retrieve ---
    retrieved = retrieve_fn(question, collection)

    # --- Generate ---
    answer, sources = generate_answer(question, retrieved, provider=provider)

    # Build a result dict that matches what checks.py and judge.py expect
    result = {
        "answer":    answer,
        "sources":   sources,
        "retrieved": [
            {"source": src, "distance": round(dist, 4), "chunk_preview": chunk[:200]}
            for chunk, src, dist in retrieved
        ],
    }

    # --- Rule checks (free, deterministic) ---
    check_results = run_checks(case, result)
    rules_passed  = case_passed(check_results)

    # --- LLM judge (only if enabled) ---
    judge_result = None
    if use_judge:
        judge_result = run_judge(case, result, provider=provider)

    return {
        "case_id":       case["id"],
        "question":      question,
        "problem_type":  case.get("problem_type", "unknown"),
        "expected_behavior": case.get("expected_behavior", ""),
        "answer":        answer,
        "sources":       sources,
        "retrieved":     result["retrieved"],
        "check_results": check_results,
        "rules_passed":  rules_passed,
        "judge":         judge_result,
        # overall pass = rules AND judge both pass (if judge was run)
        "passed":        rules_passed and (judge_result is None or judge_result["verdict"] == "PASS"),
    }


def print_table(results: list[dict], use_judge: bool):
    """Print a per-problem-type pass-rate table, then overall."""
    # Group by problem_type
    by_type: dict[str, list] = {}
    for r in results:
        pt = r["problem_type"]
        by_type.setdefault(pt, []).append(r)

    col1 = max(len(pt) for pt in by_type) + 2
    col1 = max(col1, 20)
    header = f"{'problem_type':<{col1}} {'total':>6} {'pass':>6} {'pass%':>7}"
    if use_judge:
        header += f"  {'judge_pass':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    all_results = []
    for pt, rows in sorted(by_type.items()):
        n        = len(rows)
        n_pass   = sum(1 for r in rows if r["passed"])
        pct      = n_pass / n * 100
        line     = f"{pt:<{col1}} {n:>6} {n_pass:>6} {pct:>6.1f}%"
        if use_judge:
            jpass = sum(1 for r in rows if r["judge"] and r["judge"]["verdict"] == "PASS")
            line += f"  {jpass:>4}/{n:<4}"
        print(line)
        all_results.extend(rows)

    print("-" * len(header))
    total  = len(all_results)
    passed = sum(1 for r in all_results if r["passed"])
    overall_pct = passed / total * 100 if total else 0
    print(f"{'OVERALL':<{col1}} {total:>6} {passed:>6} {overall_pct:>6.1f}%")
    print("=" * len(header))
    print()

    # Show any failures with their first failing check reason
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"Failures ({len(failures)}):")
        for r in failures:
            first_fail = next((c for c in r["check_results"] if not c["passed"]), None)
            reason = first_fail["reason"] if first_fail else (
                r["judge"]["reason"] if r["judge"] else "unknown"
            )
            print(f"  [{r['case_id']}] {r['question'][:60]}")
            print(f"         -> {reason}")
        print()


def save_results(results: list[dict], run_name: str, meta: dict) -> str:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{run_name}.json"
    payload = {
        "run_name":   run_name,
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "meta":       meta,
        "results":    results,
        "summary": {
            "total":  len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "pass_pct": round(sum(1 for r in results if r["passed"]) / len(results) * 100, 1) if results else 0,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description="Run the eval suite and score the app.")
    parser.add_argument("--eval-set",  default=EVAL_SET)
    parser.add_argument("--provider",  default=None, help="openrouter or ollama")
    parser.add_argument("--method",    default="hybrid", choices=list(RETRIEVERS.keys()))
    parser.add_argument("--run-name",  default=None, help="Label for this run (default: timestamp)")
    parser.add_argument("--no-judge",  action="store_true", help="Skip LLM judge (rule checks only)")
    args = parser.parse_args()

    provider  = args.provider or DEFAULT_PROVIDER
    use_judge = not args.no_judge
    run_name  = args.run_name or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    model     = OPENROUTER_MODEL if provider == "openrouter" else OLLAMA_CHAT_MODEL

    print(f"run_evals.py")
    print(f"  eval set : {args.eval_set}")
    print(f"  provider : {provider} ({model})")
    print(f"  method   : {args.method}")
    print(f"  judge    : {'yes' if use_judge else 'no (--no-judge)'}")
    print(f"  run name : {run_name}")
    print()

    cases = load_eval_set(args.eval_set)
    print(f"Loaded {len(cases)} cases from {args.eval_set}")

    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    results = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i:02d}/{len(cases)}] {case['id']} -- {case['question'][:55]}", end="", flush=True)
        t0 = time.time()
        r  = run_one(case, collection, args.method, provider, use_judge)
        elapsed = time.time() - t0
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  -> {status}  ({elapsed:.1f}s)")
        results.append(r)

    print()
    print_table(results, use_judge)

    meta = {"provider": provider, "model": model, "method": args.method, "judge": use_judge}
    out  = save_results(results, run_name, meta)
    print(f"Results saved to: {out}")


if __name__ == "__main__":
    main()
