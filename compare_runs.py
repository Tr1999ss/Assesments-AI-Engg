"""
compare_runs.py -- compare two eval result files (before vs after an improvement).

Usage:
    python compare_runs.py results/before_fix.json results/after_fix.json

Prints a table:
    problem_type | before_pass% | after_pass% | delta | status

Regressions (negative delta) are printed with a warning marker so they are
impossible to miss. That is the single most important thing this script does.
"""

import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pass_rates_by_type(run: dict) -> dict[str, tuple[int, int]]:
    """Return {problem_type: (passed, total)} for every problem type in the run."""
    by_type: dict[str, list] = {}
    for r in run["results"]:
        pt = r.get("problem_type", "unknown")
        by_type.setdefault(pt, []).append(r)
    return {
        pt: (sum(1 for r in rows if r["passed"]), len(rows))
        for pt, rows in by_type.items()
    }


def fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return "  n/a "
    return f"{n}/{total} ({n/total*100:.0f}%)"


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_runs.py <before.json> <after.json>")
        sys.exit(1)

    before_path = sys.argv[1]
    after_path  = sys.argv[2]

    before = load(before_path)
    after  = load(after_path)

    before_rates = pass_rates_by_type(before)
    after_rates  = pass_rates_by_type(after)

    # Union of all problem types across both runs
    all_types = sorted(set(before_rates) | set(after_rates))

    col_type  = max(len(t) for t in all_types + ["problem_type"]) + 2
    col_val   = 14

    print()
    print(f"Before : {Path(before_path).stem}  ({before['meta'].get('model','?')}, {before['meta'].get('method','?')})")
    print(f"After  : {Path(after_path).stem}   ({after['meta'].get('model','?')}, {after['meta'].get('method','?')})")
    print()

    header = (
        f"{'problem_type':<{col_type}} "
        f"{'before':>{col_val}} "
        f"{'after':>{col_val}} "
        f"{'delta':>8}  status"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    overall_before_pass  = 0
    overall_before_total = 0
    overall_after_pass   = 0
    overall_after_total  = 0

    for pt in all_types:
        b_pass, b_total = before_rates.get(pt, (0, 0))
        a_pass, a_total = after_rates.get(pt, (0, 0))

        b_pct = b_pass / b_total * 100 if b_total else None
        a_pct = a_pass / a_total * 100 if a_total else None

        if b_pct is not None and a_pct is not None:
            delta = a_pct - b_pct
            if delta > 5:
                status = "IMPROVED ✓"
            elif delta < -5:
                status = "REGRESSION ⚠  <-- CHECK THIS"
            else:
                status = "unchanged"
            delta_str = f"{delta:+.0f}pp"
        else:
            delta_str = "   n/a"
            status    = "(only in one run)"

        b_str = fmt_pct(b_pass, b_total)
        a_str = fmt_pct(a_pass, a_total)

        print(f"{pt:<{col_type}} {b_str:>{col_val}} {a_str:>{col_val}} {delta_str:>8}  {status}")

        overall_before_pass  += b_pass
        overall_before_total += b_total
        overall_after_pass   += a_pass
        overall_after_total  += a_total

    print("-" * len(header))

    # Overall row
    b_ov   = overall_before_pass / overall_before_total * 100 if overall_before_total else 0
    a_ov   = overall_after_pass  / overall_after_total  * 100 if overall_after_total  else 0
    d_ov   = a_ov - b_ov
    d_str  = f"{d_ov:+.0f}pp"
    status = "IMPROVED ✓" if d_ov > 2 else ("REGRESSION ⚠  <-- CHECK THIS" if d_ov < -2 else "unchanged")
    b_str  = fmt_pct(overall_before_pass, overall_before_total)
    a_str  = fmt_pct(overall_after_pass,  overall_after_total)
    print(f"{'OVERALL':<{col_type}} {b_str:>{col_val}} {a_str:>{col_val}} {d_str:>8}  {status}")
    print("=" * len(header))
    print()

    # Highlight any regressions clearly
    regressions = []
    for pt in all_types:
        b_pass, b_total = before_rates.get(pt, (0, 0))
        a_pass, a_total = after_rates.get(pt, (0, 0))
        if b_total and a_total:
            b_pct = b_pass / b_total * 100
            a_pct = a_pass / a_total * 100
            if a_pct - b_pct < -5:
                regressions.append((pt, b_pct, a_pct))

    if regressions:
        print("⚠  REGRESSIONS DETECTED -- investigate before shipping:")
        for pt, b, a in regressions:
            print(f"   {pt}: {b:.0f}% -> {a:.0f}%  ({a-b:+.0f}pp)")
        print()
    else:
        print("No regressions detected.")
        print()


if __name__ == "__main__":
    main()
