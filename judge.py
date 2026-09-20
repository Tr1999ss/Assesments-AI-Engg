"""
judge.py -- LLM-as-judge for answer quality.

WHY an LLM judge at all?
  Rule checks (checks.py) are great at verifiable facts: did the model cite a source,
  did it mention "403", did it refuse. But they cannot assess:
    - Faithfulness: did the model add things NOT in the retrieved context?
    - Helpfulness: is the answer actually useful, or just technically non-empty?
  The judge evaluates exactly those two things.

WHY G-Eval style?
  G-Eval (Liu et al. 2023) asks the model to reason step-by-step BEFORE giving a
  verdict. This reduces positional bias and makes the verdict more calibrated than
  asking for a score directly.

Judge design choices:
  - BINARY verdict (PASS/FAIL), not a 1-5 score. Binary is more reproducible and
    easier to validate with Cohen's kappa (Phase 4).
  - temperature=0 for maximum reproducibility.
  - Prompt lives in judge_prompt.txt so you can tune it without touching Python.
  - JSON output is parsed safely -- a malformed response is treated as FAIL with
    an explanatory reason, not a crash.

Usage:
    from judge import run_judge
    result = run_judge(case, app_result, provider="openrouter")
    # result: {"verdict": "PASS"/"FAIL", "reason": "...", "reasoning": "..."}

    # Or run standalone self-test:
    python judge.py
"""

import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

PROMPT_FILE = "judge_prompt.txt"

# Use the same provider config as query.py so no new credentials are needed.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OLLAMA_CHAT_MODEL  = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:latest")


def _load_prompt_template() -> str:
    """Load the judge prompt from disk so it can be edited without touching code."""
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()


def _build_prompt(question: str, context: str, answer: str) -> str:
    """Fill in the three placeholders in the prompt template."""
    template = _load_prompt_template()
    return (
        template
        .replace("{question}", question)
        .replace("{context}", context)
        .replace("{answer}", answer)
    )


def _build_context(result: dict) -> str:
    """
    Reconstruct the retrieved context string from the result dict.
    Each retrieved chunk is shown with its source label, exactly as the app
    would have seen it -- so the judge evaluates faithfulness against the
    ACTUAL context the model had, not some idealised version.
    """
    blocks = []
    for chunk in result.get("retrieved", []):
        source = chunk.get("source", "unknown")
        preview = chunk.get("chunk_preview", "")
        blocks.append(f"[Source: {source}]\n{preview}")
    return "\n\n---\n\n".join(blocks) if blocks else "(no context retrieved)"


def _call_openrouter(prompt: str) -> str:
    """Call OpenRouter with temperature=0 for reproducibility."""
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,          # fixed -- reduces verdict variance
        max_tokens=512,
        response_format={"type": "json_object"},  # ask for JSON directly
    )
    return response.choices[0].message.content


def _call_ollama(prompt: str) -> str:
    """Call local Ollama model (no temperature param in current ollama lib)."""
    import ollama
    response = ollama.chat(
        model=OLLAMA_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def _parse_verdict(raw: str) -> dict:
    """
    Safely parse the LLM response as JSON.

    WHY careful parsing: the model may wrap JSON in markdown fences, or emit
    extra text. We try json.loads first, then strip fences and retry, then
    fall back to a FAIL verdict so a bad parse never silently counts as a pass.
    """
    # 1. Direct parse
    try:
        obj = json.loads(raw)
        return _validate_verdict_obj(obj)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences and retry
    stripped = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    stripped = re.sub(r"```\s*$", "", stripped.strip(), flags=re.MULTILINE)
    try:
        obj = json.loads(stripped)
        return _validate_verdict_obj(obj)
    except json.JSONDecodeError:
        pass

    # 3. Regex fallback: look for "verdict": "PASS" or "FAIL" anywhere in text
    m = re.search(r'"verdict"\s*:\s*"(PASS|FAIL)"', raw, re.IGNORECASE)
    if m:
        return {
            "verdict": m.group(1).upper(),
            "reason": "Extracted from malformed JSON response",
            "reasoning": raw[:300],
            "parse_warning": True,
        }

    # 4. Give up -- treat as FAIL so a bad parse never inflates pass rate
    return {
        "verdict": "FAIL",
        "reason": "Judge response could not be parsed as JSON",
        "reasoning": raw[:300],
        "parse_warning": True,
    }


def _validate_verdict_obj(obj: dict) -> dict:
    """Ensure required fields exist and verdict is PASS or FAIL."""
    verdict = str(obj.get("verdict", "")).upper()
    if verdict not in ("PASS", "FAIL"):
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "reason": obj.get("reason", ""),
        "reasoning": obj.get("reasoning", ""),
        "parse_warning": False,
    }


def run_judge(case: dict, result: dict, provider: str = None) -> dict:
    """
    Run the LLM judge on one (case, result) pair.

    Returns:
        {
            "verdict":   "PASS" or "FAIL",
            "reason":    one-line explanation,
            "reasoning": step-by-step thinking from the judge,
            "parse_warning": True if the JSON had to be rescued
        }
    """
    provider = provider or os.getenv("PROVIDER", "ollama")

    question = case.get("question", "")
    answer   = result.get("answer", "")
    context  = _build_context(result)

    prompt = _build_prompt(question, context, answer)

    if provider == "openrouter":
        raw = _call_openrouter(prompt)
    else:
        raw = _call_ollama(prompt)

    return _parse_verdict(raw)


# ---------------------------------------------------------------------------
# Standalone self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    provider = "openrouter" if "--provider" not in sys.argv else sys.argv[sys.argv.index("--provider") + 1]
    # default to openrouter if key is set, else ollama
    if not OPENROUTER_API_KEY:
        provider = "ollama"

    print(f"=== judge.py self-test (provider={provider}) ===\n")

    test_case = {
        "id": "tc_003",
        "question": "I have a valid read-only key but I tried to make a write request. What error do I get?",
        "expected_facts": ["403"],
        "expected_source": "authentication.md",
    }

    # Good answer -- faithful, correct, helpful
    good_result = {
        "answer": (
            "A read-only key cannot perform write operations. "
            "The API returns 403 Forbidden in this case, which indicates an authorization "
            "failure -- your key authenticated successfully but lacks the required scope. "
            "[Source: authentication.md]"
        ),
        "sources": ["authentication.md"],
        "retrieved": [
            {
                "source": "authentication.md",
                "distance": 0.23,
                "chunk_preview": (
                    "API key valid but insufficient scope | 403 | Authorization failure. "
                    "A valid read key can authenticate successfully but cannot perform a write operation."
                ),
            }
        ],
    }

    # Bad answer -- unfaithful (adds info not in context) and a refusal variant
    bad_result = {
        "answer": "I don't know -- I couldn't find anything about that in the documents.",
        "sources": ["authentication.md"],
        "retrieved": [
            {
                "source": "authentication.md",
                "distance": 0.23,
                "chunk_preview": (
                    "API key valid but insufficient scope | 403 | Authorization failure."
                ),
            }
        ],
    }

    for label, result in [("GOOD answer (expect PASS)", good_result), ("BAD answer (expect FAIL)", bad_result)]:
        print(f"--- {label} ---")
        verdict = run_judge(test_case, result, provider=provider)
        print(f"  Verdict   : {verdict['verdict']}")
        print(f"  Reason    : {verdict['reason']}")
        print(f"  Reasoning : {verdict['reasoning'][:200]}")
        if verdict.get("parse_warning"):
            print("  WARNING: JSON had to be rescued from malformed output")
        print()

    print("=== Self-test done ===")
