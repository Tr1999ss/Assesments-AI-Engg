"""
export_traces.py — turn traces.jsonl into a spreadsheet you can actually read and annotate.

This one CSV serves BOTH weeks' manual review step:
  - Week 4: fill in `failure_type` per row (retrieval_failure / generation_failure / success)
  - Week 5: fill in `honest_note` per row (one sentence on what went wrong), then later
            group similar notes into `problem_group` once you've read them all

Usage:
    python export_traces.py                  # exports every trace in traces.jsonl
    python export_traces.py --sample 20       # exports a random sample of 20 (Week 5 wants this)
"""

import argparse
import csv
import json
import random


def load_traces(path="traces.jsonl"):
    traces = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def export(traces, out_path="traces_review.csv"):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trace_id", "question", "method", "retrieved_sources", "retrieved_preview",
            "answer", "sources_cited",
            "failure_type",   # you fill this in: retrieval_failure / generation_failure / success
            "honest_note",    # you fill this in: one sentence, written BEFORE grouping (Week 5)
            "problem_group",  # you fill this in LATER, after reading all notes (Week 5)
        ])
        for i, trace in enumerate(traces):
            retrieved_sources = "; ".join(r["source"] for r in trace["retrieved"])
            retrieved_preview = " | ".join(r["chunk_preview"][:80] for r in trace["retrieved"])
            writer.writerow([
                i,
                trace["question"],
                trace.get("method", "embedding"),
                retrieved_sources,
                retrieved_preview,
                trace["answer"],
                "; ".join(trace["sources"]),
                "",  # failure_type - blank, fill by hand
                "",  # honest_note - blank, fill by hand
                "",  # problem_group - blank, fill by hand
            ])
    print(f"Exported {len(traces)} traces to {out_path}")
    print("Open it in Excel/Google Sheets and fill in the blank columns by hand.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                         help="Export a random sample of N traces instead of all of them.")
    parser.add_argument("--input", default="traces.jsonl")
    parser.add_argument("--output", default="traces_review.csv")
    args = parser.parse_args()

    traces = load_traces(args.input)
    print(f"Loaded {len(traces)} traces from {args.input}")

    if args.sample and args.sample < len(traces):
        # random.sample, not traces[:20] — a RANDOM sample avoids only picking the
        # questions you remember being good, which Week 5 explicitly checks for
        traces = random.sample(traces, args.sample)
        print(f"Randomly sampled {args.sample} of them")

    export(traces, args.output)