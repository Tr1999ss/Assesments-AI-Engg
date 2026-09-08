"""
golden_set.py — a hand-curated, trusted set of question -> correct-answer pairs.

Different from eval.py's TEST_SET: that one only checks RETRIEVAL (did the right
document show up in top-k). This checks GENERATION too — did the model actually
get the answer right, including subtle distinctions the docs specifically warn
are "easy to confuse."

Each entry:
    question            - what you'd actually ask
    expected_source     - which doc file should be cited
    expected_keywords    - substrings that MUST appear in a correct answer
                            (case-insensitive). If none are found, the answer
                            is probably wrong even if it sounds confident.
    trap_keywords        - substrings that would indicate a classic WRONG answer
                            (e.g. confusing 401 with 429). If these appear, flag
                            it for manual review even if expected_keywords also matched.
    note                 - why this question is actually hard / what it's testing
"""

GOLDEN_SET = [
    # ---------- straightforward (sanity checks - should never fail) ----------
    {
        "question": "How do I rotate my API key?",
        "expected_source": "authentication.md",
        "expected_keywords": ["generate", "revoke"],
        "trap_keywords": [],
        "note": "Baseline - if this fails, something more basic broke.",
    },
    {
        "question": "How many times can a failed job be retried?",
        "expected_source": "jobs-api.md",
        "expected_keywords": ["5"],
        "trap_keywords": [],
        "note": "Baseline - simple fact lookup.",
    },

    # ---------- the "intentionally easy to confuse" status codes ----------
    {
        "question": "If my API key is invalid, what status code do I get?",
        "expected_source": "authentication.md",
        "expected_keywords": ["401"],
        "trap_keywords": ["403", "429"],
        "note": "Tests whether the model picks the right code out of three "
                "similar-looking ones the doc explicitly warns about.",
    },
    {
        "question": "I have a valid read-only key but I tried to make a write request. What error do I get?",
        "expected_source": "authentication.md",
        "expected_keywords": ["403"],
        "trap_keywords": ["401", "invalid"],
        "note": "Classic trap: a model might say '401 unauthorized' since that's "
                "the more commonly known code, when the doc specifically says "
                "insufficient scope is 403, and the key itself is still valid.",
    },
    {
        "question": "I'm getting 429 errors. Does that mean my API key is invalid?",
        "expected_source": "authentication.md",
        "expected_keywords": ["rate", "limit"],
        "trap_keywords": ["invalid", "401"],
        "note": "The doc explicitly says this misconception should NOT be made. "
                "A wrong answer here reveals the model isn't actually reading "
                "carefully, just pattern-matching '429 = error = something's wrong'.",
    },
    {
        "question": "My key was working fine, then I started getting 429s. Was my key revoked?",
        "expected_source": "authentication.md",
        "expected_keywords": ["rate", "retry-after"],
        "trap_keywords": ["revoked", "401"],
        "note": "Same trap as above, phrased more misleadingly (mentions "
                "'revoked' in the question itself, inviting the model to agree).",
    },

    # ---------- scope / rotation lifecycle nuance ----------
    {
        "question": "If I rotate my API key, does the new key automatically get the same scope as the old one?",
        "expected_source": "authentication.md",
        "expected_keywords": ["replacement", "configured"],
        "trap_keywords": ["automatically", "same scope"],
        "note": "Doc says rotation creates a REPLACEMENT key which 'may be "
                "configured' with required scope - it does not say scope "
                "carries over automatically. Watch for the model asserting "
                "automatic inheritance that isn't actually stated.",
    },
    {
        "question": "Do API keys expire automatically after 90 days?",
        "expected_source": "authentication.md",
        "expected_keywords": ["do not expire", "recommended"],
        "trap_keywords": ["automatically expire", "expires after 90"],
        "note": "Doc explicitly warns not to interpret the 90-day recommendation "
                "as automatic expiration. A model conflating 'recommended' with "
                "'enforced' is a real, subtle failure mode.",
    },
    {
        "question": "During a key rotation, can I use both the old and new key at the same time?",
        "expected_source": "authentication.md",
        "expected_keywords": ["yes", "both", "active"],
        "trap_keywords": ["no", "cannot"],
        "note": "Tests basic recall of zero-downtime rotation, stated directly "
                "in the doc.",
    },
    {
        "question": "Can I use a revoked API key again later if I need to?",
        "expected_source": "authentication.md",
        "expected_keywords": ["cannot be used again", "revoked"],
        "trap_keywords": ["can be reactivated", "yes"],
        "note": "Doc states this plainly - a wrong answer here is a clear "
                "generation failure, not a retrieval one.",
    },

    # ---------- terminology equivalence ----------
    {
        "question": "My config file has an 'access key' and my code expects an 'API key' - are these different credentials I need to manage separately?",
        "expected_source": "authentication.md",
        "expected_keywords": ["equivalent", "same"],
        "trap_keywords": ["different credentials", "separately"],
        "note": "Doc explicitly says these terms (API key / access key / bearer "
                "token) are equivalent names, not three different secrets. Tests "
                "whether the model tracks defined terminology instead of taking "
                "the question's framing ('separately') at face value.",
    },

    # ---------- exact-term / code retrieval (from Week 4's hybrid search work) ----------
    {
        "question": "What does a 429 response mean?",
        "expected_source": "authentication.md",
        "expected_keywords": ["too many requests", "rate"],
        "trap_keywords": [],
        "note": "Exact-code lookup - good test case for whether hybrid/BM25 "
                "retrieval actually helps vs plain embeddings (this was a "
                "known miss in the Week 4 trace review).",
    },
]