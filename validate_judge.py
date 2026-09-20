"""
validate_judge.py -- measure how well the LLM judge agrees with YOUR hand labels.

WHY this step is critical:
  Before trusting the judge numbers in your final report, you must confirm the
  judge agrees with a human (you) on real examples. If agreement is low, you
  tune judge_prompt.txt and re-run -- NOT the code, just the prompt.

Two-stage workflow:
  Stage 1 (export):
      python validate_judge.py --export
      -> writes judge_validation.csv with an empty human_label column.
      Fill in human_label (pass / fail) for every row BY HAND, then save.

  Stage 2 (score):
      python validate_judge.py --score
      -> runs the judge on the same rows, compares to your labels, prints:
           * agreement %
           * Cohen kappa (accounts for chance agreement; >0.6 is good)
           * confusion matrix
           * list of disagreements so you can see where the judge is wrong

Tuning loop (repeat until agreement is good enough):
  1. Edit judge_prompt.txt
  2. python validate_judge.py --score   (re-runs judge, re-computes agreement)
  3. Check if agreement improved

Suggested threshold: kappa >= 0.6 (moderate-to-good agreement).
A kappa of 0.8+ is strong. Below 0.4 means the judge is not reliable.
"""

import argparse
import csv
import json
import os
import random

from dotenv import load_dotenv

load_dotenv()

TRACES_FILE      = "traces.jsonl"
VALIDATION_CSV   = "judge_validation.csv"
PROVIDER         = os.getenv("PROVIDER", "ollama")
SAMPLE_SIZE      = 25   # how many traces to export for hand labelling
RANDOM_SEED      = 42   # fixed seed so the same traces are always selected


# ---------------------------------------------------------------------------
# Stage 1: export a sample for hand labelling
# ---------------------------------------------------------------------------

def export_for_labelling(traces_file: str, out_csv: str, n: int, seed: int):
    """
    Randomly sample n traces, write them to a CSV with an empty human_label column.

    WHY random sample (not first-N):
      First-N would bias toward your earliest (often simpler) questions.
      A random sample gives a more representative view of real app behaviour.
    """
    with open(traces_file, "r", encoding="utf-8") as f:
        traces = [json.loads(line) for line in f if line.strip()]

    random.seed(seed)
    sample = random.sample(traces, min(n, len(traces)))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "row_id",
            "question",
            "answer",
            "retrieved_sources",
            "retrieved_preview",
            "human_label",   # YOU fill this in: pass or fail
            "_trace_json",   # full trace stored here so --score can replay it
        ])
        writer.writeheader()
        for i, trace in enumerate(sample):
            retrieved_sources  = "; ".join(r["source"] for r in trace.get("retrieved", []))
            retrieved_preview  = " | ".join(r["chunk_preview"][:80] for r in trace.get("retrieved", []))
            writer.writerow({
                "row_id":            i,
                "question":          trace.get("question", ""),
                "answer":            trace.get("answer", ""),
                "retrieved_sources": retrieved_sources,
                "retrieved_preview": retrieved_preview,
                "human_label":       "",          # <- fill this in by hand
                "_trace_json":       json.dumps(trace),
            })

    print(f"Exported {len(sample)} rows to {out_csv}")
    print()
    print("Next steps:")
    print("  1. Open judge_validation.csv in Excel or VS Code.")
    print("  2. For each row, read the question, retrieved context, and answer.")
    print("  3. Write 'pass' or 'fail' in the human_label column.")
    print("  4. Save the file, then run:  python validate_judge.py --score")


# ---------------------------------------------------------------------------
# Stage 2: run the judge and compute agreement metrics
# ---------------------------------------------------------------------------

def cohen_kappa(human_labels: list, judge_labels: list) -> float:
    """
    Cohen's kappa measures agreement corrected for chance.

    kappa = (observed agreement - chance agreement) / (1 - chance agreement)

    Interpretation:
      < 0.20  slight      -- the judge is barely better than random
      0.20-0.40  fair
      0.40-0.60  moderate
      0.60-0.80  good       -- aim for this as minimum before trusting the judge
      0.80-1.00  very good / near-perfect
    """
    assert len(human_labels) == len(judge_labels)
    n = len(human_labels)
    if n == 0:
        return 0.0

    # Count agreements
    observed_agree = sum(h == j for h, j in zip(human_labels, judge_labels)) / n

    # Marginal probabilities
    h_pass = human_labels.count("pass") / n
    j_pass = judge_labels.count("pass") / n
    h_fail = human_labels.count("fail") / n
    j_fail = judge_labels.count("fail") / n

    # Chance agreement = P(both say pass) + P(both say fail)
    chance_agree = (h_pass * j_pass) + (h_fail * j_fail)

    if chance_agree == 1.0:
        return 1.0   # perfect agreement, no denominator issue

    kappa = (observed_agree - chance_agree) / (1.0 - chance_agree)
    return round(kappa, 4)


def score(validation_csv: str, provider: str):
    """Load the hand-labelled CSV, run the judge, print agreement metrics."""
    # Import here so the module can be imported without requiring judge installed
    from judge import run_judge

    rows = []
    with open(validation_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Filter to only rows the human has labelled
    labelled = [r for r in rows if r["human_label"].strip().lower() in ("pass", "fail")]
    unlabelled = len(rows) - len(labelled)

    if unlabelled > 0:
        print(f"WARNING: {unlabelled} row(s) have no human_label -- they will be skipped.")
    if not labelled:
        print("No labelled rows found. Fill in the human_label column and re-run.")
        return

    print(f"Scoring {len(labelled)} labelled rows using judge (provider={provider}) ...")
    print()

    human_labels = []
    judge_labels = []
    disagreements = []
    confusion = {"pass/pass": 0, "pass/fail": 0, "fail/pass": 0, "fail/fail": 0}

    for r in labelled:
        trace   = json.loads(r["_trace_json"])
        # build a minimal case dict (judge only needs question + retrieved)
        case    = {"question": trace.get("question", "")}
        result  = {
            "answer":    trace.get("answer", ""),
            "sources":   trace.get("sources", []),
            "retrieved": trace.get("retrieved", []),
        }

        verdict_obj = run_judge(case, result, provider=provider)
        judge_verdict = verdict_obj["verdict"].lower()   # "pass" or "fail"
        human_verdict = r["human_label"].strip().lower()

        human_labels.append(human_verdict)
        judge_labels.append(judge_verdict)

        key = f"{human_verdict}/{judge_verdict}"
        confusion[key] = confusion.get(key, 0) + 1

        if human_verdict != judge_verdict:
            disagreements.append({
                "row_id":        r["row_id"],
                "question":      r["question"][:80],
                "answer":        r["answer"][:100],
                "human_label":   human_verdict,
                "judge_verdict": judge_verdict,
                "judge_reason":  verdict_obj.get("reason", ""),
            })

    # --- Metrics ---
    n = len(labelled)
    agree = sum(h == j for h, j in zip(human_labels, judge_labels))
    agreement_pct = round(agree / n * 100, 1)
    kappa = cohen_kappa(human_labels, judge_labels)

    print("=" * 55)
    print(f"  Rows scored        : {n}")
    print(f"  Agreement          : {agree}/{n}  ({agreement_pct}%)")
    print(f"  Cohen's kappa      : {kappa}")

    if kappa >= 0.80:
        print("  Quality            : VERY GOOD (kappa >= 0.80)")
    elif kappa >= 0.60:
        print("  Quality            : GOOD (kappa >= 0.60) -- safe to use judge")
    elif kappa >= 0.40:
        print("  Quality            : MODERATE -- consider tuning judge_prompt.txt")
    else:
        print("  Quality            : LOW -- tune judge_prompt.txt before trusting scores")

    print()
    print("Confusion matrix (human label / judge verdict):")
    print(f"  human=pass, judge=pass  (both agree PASS) : {confusion.get('pass/pass', 0)}")
    print(f"  human=fail, judge=fail  (both agree FAIL) : {confusion.get('fail/fail', 0)}")
    print(f"  human=pass, judge=FAIL  (judge too strict) : {confusion.get('pass/fail', 0)}")
    print(f"  human=fail, judge=PASS  (judge too lenient): {confusion.get('fail/pass', 0)}")

    if disagreements:
        print()
        print(f"Disagreements ({len(disagreements)} total) -- review these to tune the prompt:")
        print("-" * 55)
        for d in disagreements:
            print(f"  row {d['row_id']}: human={d['human_label']}, judge={d['judge_verdict']}")
            print(f"    Q: {d['question']}")
            print(f"    A: {d['answer']}")
            print(f"    Judge reason: {d['judge_reason']}")
            print()
    else:
        print()
        print("No disagreements -- perfect agreement!")

    print("=" * 55)
    print()
    print("To tune the judge: edit judge_prompt.txt, then re-run:  python validate_judge.py --score")
    print("Suggested minimum kappa before trusting judge scores: 0.60")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the LLM judge against human labels.")
    parser.add_argument("--export", action="store_true",
                        help="Stage 1: export a sample of traces to judge_validation.csv for hand labelling.")
    parser.add_argument("--score",  action="store_true",
                        help="Stage 2: run the judge on the labelled CSV and report agreement metrics.")
    parser.add_argument("--provider", default=None,
                        help="LLM provider: openrouter or ollama. Defaults to PROVIDER env var.")
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE,
                        help=f"Number of rows to export for labelling. Default: {SAMPLE_SIZE}")
    args = parser.parse_args()

    provider = args.provider or PROVIDER

    if args.export:
        export_for_labelling(TRACES_FILE, VALIDATION_CSV, args.n, RANDOM_SEED)
    elif args.score:
        score(VALIDATION_CSV, provider)
    else:
        parser.print_help()
