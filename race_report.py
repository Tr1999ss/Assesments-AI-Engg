"""
Race report — reads traces.jsonl (written by rag_agent.py) and summarizes
fixed workflow vs agent on speed, step count, and how often each gave up.

Usage:
    python race_report.py
    python race_report.py --file traces.jsonl
"""

import argparse
import json
from pathlib import Path


def load_traces(path):
    traces = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return traces


def summarize(traces, mode):
    rows = [t for t in traces if t.get("mode") == mode]
    if not rows:
        return None

    seconds = [t["seconds"] for t in rows]
    dont_knows = sum(1 for t in rows if "i don't know" in t["answer"].lower())
    if mode == "agent":
        steps = [t.get("step_count", 0) for t in rows]
    else:
        steps = [1 for _ in rows]  # fixed workflow is always exactly 1 retrieval

    return {
        "runs": len(rows),
        "avg_seconds": round(sum(seconds) / len(seconds), 3),
        "avg_steps": round(sum(steps) / len(steps), 2),
        "dont_know_rate": f"{dont_knows}/{len(rows)}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="traces.jsonl")
    args = parser.parse_args()

    if not Path(args.file).exists():
        raise SystemExit(f"{args.file} not found — run rag_agent.py a few times first.")

    traces = load_traces(args.file)
    fixed = summarize(traces, "fixed")
    agent = summarize(traces, "agent")

    print(f"Loaded {len(traces)} traces from {args.file}\n")
    print(f"{'':15}{'runs':>6}{'avg sec':>10}{'avg steps':>11}{'gave up':>10}")
    for label, stats in [("fixed", fixed), ("agent", agent)]:
        if stats is None:
            print(f"{label:15}{'(no runs logged yet)':>47}")
        else:
            print(f"{label:15}{stats['runs']:>6}{stats['avg_seconds']:>10}"
                  f"{stats['avg_steps']:>11}{stats['dont_know_rate']:>10}")

    if fixed and agent:
        print("\n--- Verdict inputs ---")
        print(f"Agent is {round(agent['avg_seconds'] / fixed['avg_seconds'], 2)}x the fixed workflow's time")
        print(f"Agent avg steps: {agent['avg_steps']} vs fixed's fixed 1 step")
        print(f"Fixed 'I don't know' rate: {fixed['dont_know_rate']}  |  "
              f"Agent 'I don't know' rate: {agent['dont_know_rate']}")
        print("-> if the agent's don't-know rate is meaningfully lower, that's your")
        print("   case for shipping it despite the extra time/token cost. If the")
        print("   rates are about the same, ship the fixed workflow — it's cheaper.")


if __name__ == "__main__":
    main()