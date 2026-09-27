"""
STEP 5 — Token Cost Analysis
==============================
Reads traces.jsonl and computes:
  - Mean tokens-per-task  (fixed vs agent)
  - p99 tokens-per-task   (fixed vs agent)

Fixed-workflow traces don't call the OpenRouter API directly for retrieval
(they use Ollama embed + one OpenRouter generate_answer call).  The fixed
workflow currently doesn't persist token counts (no response.usage capture
in generate_answer), so we note that honestly.

For agent traces, `tokens` is now populated by every run after the Step 5
patch.  Older traces that pre-date the patch will have tokens=0 and are
excluded from statistics (reported separately).
"""

import json
import math
from pathlib import Path

TRACES_FILE = "traces.jsonl"


def load_traces(path):
    traces = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return traces


def percentile(data, p):
    """Compute the p-th percentile of a sorted list."""
    if not data:
        return 0
    sorted_data = sorted(data)
    index = (p / 100) * (len(sorted_data) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_data) - 1)
    frac = index - lower
    return sorted_data[lower] + frac * (sorted_data[upper] - sorted_data[lower])


def analyze_cost(traces, mode):
    rows = [t for t in traces if t.get("mode") == mode]
    if not rows:
        return None

    with_tokens = [t for t in rows if t.get("tokens", 0) > 0]
    without_tokens = [t for t in rows if t.get("tokens", 0) == 0]

    token_counts = [t["tokens"] for t in with_tokens]

    result = {
        "mode": mode,
        "total_runs": len(rows),
        "runs_with_token_data": len(with_tokens),
        "runs_missing_token_data": len(without_tokens),
    }

    if token_counts:
        result["mean_tokens"] = round(sum(token_counts) / len(token_counts), 1)
        result["min_tokens"] = min(token_counts)
        result["max_tokens"] = max(token_counts)
        result["p99_tokens"] = round(percentile(token_counts, 99), 1)
        result["all_token_counts"] = sorted(token_counts)
    else:
        result["mean_tokens"] = None
        result["p99_tokens"] = None
        result["note"] = "No token data available — all runs pre-date the Step 5 patch."

    return result


def main():
    if not Path(TRACES_FILE).exists():
        raise SystemExit(f"{TRACES_FILE} not found.")

    traces = load_traces(TRACES_FILE)
    print(f"\nLoaded {len(traces)} traces from {TRACES_FILE}")

    fixed_stats = analyze_cost(traces, "fixed")
    agent_stats = analyze_cost(traces, "agent")

    print(f"\n{'='*60}")
    print(f"STEP 5 — Token Cost Analysis")
    print(f"{'='*60}")

    for stats in [fixed_stats, agent_stats]:
        if stats is None:
            continue
        mode = stats["mode"].upper()
        print(f"\n  [{mode}]")
        print(f"    Total runs          : {stats['total_runs']}")
        print(f"    Runs with token data: {stats['runs_with_token_data']}")
        print(f"    Runs missing data   : {stats['runs_missing_token_data']}")
        if stats.get("mean_tokens") is not None:
            print(f"    Mean tokens/task    : {stats['mean_tokens']}")
            print(f"    Min  tokens/task    : {stats['min_tokens']}")
            print(f"    Max  tokens/task    : {stats['max_tokens']}")
            print(f"    p99  tokens/task    : {stats['p99_tokens']}")
        else:
            print(f"    NOTE: {stats.get('note', 'no data')}")

    if fixed_stats and agent_stats:
        if (fixed_stats.get("mean_tokens") is not None and
                agent_stats.get("mean_tokens") is not None):
            ratio = agent_stats["mean_tokens"] / fixed_stats["mean_tokens"]
            print(f"\n  Agent uses {ratio:.1f}x the tokens of the fixed workflow (mean)")
        elif agent_stats.get("mean_tokens") is not None:
            print(f"\n  Fixed workflow token data not available (Ollama generate_answer "
                  f"doesn't call OpenRouter directly; token counting would require wrapping "
                  f"the Ollama API).")
            print(f"  Agent mean: {agent_stats['mean_tokens']} tokens/task")
            print(f"  Agent p99 : {agent_stats['p99_tokens']} tokens/task")

    # Save summary
    summary = {
        "fixed": fixed_stats,
        "agent": agent_stats,
    }
    with open("step5_cost_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n-> Full cost summary saved to step5_cost_summary.json")
    return summary


if __name__ == "__main__":
    main()
