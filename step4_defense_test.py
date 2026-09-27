"""
STEP 4 — Post-Defense Injection Test
=====================================
Re-runs the exact same injection question 8 times with the hardened
AGENT_SYSTEM_PROMPT and output_validation() now in place.
Reports new hijack rate and compares to pre-defense.
"""

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Import the injection test runner from step3 (reuse logic)
from step3_injection_test import run_injection_test

if __name__ == "__main__":
    print("\nRunning post-defense injection test...")
    summary = run_injection_test(
        label="POST-DEFENSE",
        output_file="step4_injection_results.jsonl",
    )
    with open("step4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Compare pre vs post
    pre_file = Path("step3_summary.json")
    if pre_file.exists():
        with open(pre_file) as f:
            pre = json.load(f)
        print(f"\n{'='*60}")
        print(f"DEFENSE COMPARISON")
        print(f"{'='*60}")
        print(f"  Pre-defense  hijack rate : {pre['hijack_count']}/{pre['total_runs']} = {pre['hijack_rate']:.1%}")
        print(f"  Post-defense hijack rate : {summary['hijack_count']}/{summary['total_runs']} = {summary['hijack_rate']:.1%}")
        reduction = pre['hijack_rate'] - summary['hijack_rate']
        print(f"  Reduction                : {reduction:.1%}")

    print(f"\n-> Results saved to step4_summary.json and step4_injection_results.jsonl")
