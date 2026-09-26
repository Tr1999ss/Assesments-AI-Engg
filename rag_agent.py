import argparse
import json
import os
import time
from pathlib import Path

import chromadb
import ollama
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "dev_docs"
EMBED_MODEL = "nomic-embed-text"

PROVIDER = os.getenv("PROVIDER", "ollama")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:latest")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

TOP_K = 3
SIMILARITY_FLOOR = 0.35
TRACES_FILE = "traces.jsonl"

# --- agent-specific stop conditions ---
AGENT_MAX_STEPS = 4          # never more than 4 tool calls before forcing an answer
AGENT_TOKEN_BUDGET = 3000    # bail if we've burned this many tokens


def embed(text):
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


# ---------- retrieval methods (unchanged — these ARE the agent's tools too) ----------

def retrieve_embedding(question, collection, top_k=TOP_K):
    query_vector = embed(question)
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    distances = results["distances"][0]
    return list(zip(chunks, sources, distances))


def retrieve_bm25(question, collection, top_k=TOP_K):
    all_data = collection.get(include=["documents", "metadatas"])
    all_chunks = all_data["documents"]
    all_sources = [m["source"] for m in all_data["metadatas"]]
    tokenized_corpus = [doc.lower().split() for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(all_chunks, all_sources, scores), key=lambda x: -x[2])
    top = ranked[:top_k]
    return [(chunk, source, 1.0 / (1.0 + score)) for chunk, source, score in top]


def retrieve_hybrid(question, collection, top_k=TOP_K, k_const=60):
    emb_results = retrieve_embedding(question, collection, top_k=10)
    bm25_results = retrieve_bm25(question, collection, top_k=10)
    scores = {}
    for rank, (chunk, source, _) in enumerate(emb_results):
        scores.setdefault(chunk, {"score": 0.0, "source": source})
        scores[chunk]["score"] += 1.0 / (k_const + rank)
    for rank, (chunk, source, _) in enumerate(bm25_results):
        scores.setdefault(chunk, {"score": 0.0, "source": source})
        scores[chunk]["score"] += 1.0 / (k_const + rank)
    ranked = sorted(scores.items(), key=lambda x: -x[1]["score"])
    top = ranked[:top_k]
    return [(chunk, data["source"], 1.0 / (1.0 + data["score"] * 100)) for chunk, data in top]


RETRIEVERS = {
    "embedding": retrieve_embedding,
    "bm25": retrieve_bm25,
    "hybrid": retrieve_hybrid,
}


def _build_prompt(question, context):
    return f"""You are a documentation assistant. Answer the question using ONLY the context below.
If the answer is not contained in the context, say "I don't know" — do not make anything up.
Always mention which source file(s) your answer came from.

Context:
{context}

Question: {question}

Answer:"""


def _generate_ollama(prompt):
    response = ollama.chat(model=OLLAMA_CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def _generate_openrouter(prompt):
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return response.choices[0].message.content


GENERATORS = {"ollama": _generate_ollama, "openrouter": _generate_openrouter}


def generate_answer(question, retrieved, provider=None):
    """FIXED WORKFLOW: exactly one retrieval (already done by the caller),
    exactly one generation call. This is the comparison baseline."""
    provider = provider or PROVIDER
    best_distance = min(d for _, _, d in retrieved)
    if best_distance > SIMILARITY_FLOOR:
        return "I don't know — I couldn't find anything about that in the documents.", []

    context_blocks = [f"[Source: {source}]\n{chunk}" for chunk, source, _ in retrieved]
    used_sources = [source for _, source, _ in retrieved]
    context = "\n\n---\n\n".join(context_blocks)
    prompt = _build_prompt(question, context)
    content = GENERATORS[provider](prompt)
    return content, list(dict.fromkeys(used_sources))


# ---------------------------------------------------------------
# AGENT WORKFLOW — same retrieval tools, but the model decides:
#   - which method to use
#   - whether to reformulate the query and try again
#   - when it has enough context to answer (or to give up)
# Currently wired for OpenRouter (OpenAI-style tool calling). Local Ollama
# models only support tool calling if the model itself does (e.g. llama3.1,
# qwen2.5 — NOT gemma). Swap _generate_ollama-style call in if yours supports it.
# ---------------------------------------------------------------

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Search the dev docs. Use 'bm25' for exact terms like error codes, "
                "function/class names, or config keys. Use 'embedding' for conceptual "
                "or how-to questions. Use 'hybrid' if unsure, or after a first search "
                "wasn't good enough and you want another angle."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["embedding", "bm25", "hybrid"]},
                    "query": {
                        "type": "string",
                        "description": "Search query — can be the original question or a reformulated/narrower version.",
                    },
                },
                "required": ["method", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Call once you have enough context to answer, or to say you don't know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["answer"],
            },
        },
    },
]

AGENT_SYSTEM_PROMPT = (
    "You are a documentation assistant with a `retrieve` tool. Call retrieve as "
    "many times as needed (different method and/or a reformulated query) until "
    "you have enough context, then call final_answer with the answer and the "
    "source files you used. If, after a couple of tries, nothing relevant turns "
    "up, call final_answer with 'I don't know'. Never answer from outside the "
    "retrieved context."
)


def run_agent_rag(question, collection):
    """AGENT WORKFLOW: model-driven retrieval loop. Returns (answer, sources, steps)."""
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps = []
    tokens_used = 0

    for _ in range(AGENT_MAX_STEPS):
        if tokens_used > AGENT_TOKEN_BUDGET:
            steps.append({"stop_reason": "token budget exceeded"})
            return "I don't know — ran out of budget before finding an answer.", [], steps

        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages,
            tools=AGENT_TOOLS,
            max_tokens=500,
        )
        tokens_used += response.usage.total_tokens
        msg = response.choices[0].message
        messages.append(msg.model_dump())

        if not msg.tool_calls:
            steps.append({"stop_reason": "model answered without calling final_answer"})
            return msg.content, [], steps

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)

            if call.function.name == "final_answer":
                steps.append({"tool": "final_answer", "args": args})
                return args["answer"], args.get("sources", []), steps

            if call.function.name == "retrieve":
                retrieve_fn = RETRIEVERS[args["method"]]
                retrieved = retrieve_fn(args["query"], collection)
                best_distance = min(d for _, _, d in retrieved)
                observation = {
                    "best_distance": round(best_distance, 3),
                    "chunks": [
                        {"source": src, "distance": round(dist, 3), "text": chunk[:300]}
                        for chunk, src, dist in retrieved
                    ],
                }
                steps.append({"tool": "retrieve", "args": args, "observation": observation})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(observation),
                })

    steps.append({"stop_reason": "max steps reached"})
    return "I don't know — hit the step limit before resolving.", [], steps


# ---------------------------------------------------------------
# Tracing — extended to record which workflow ran and, for the agent,
# every step it took. This is what you'll pull numbers from for the race.
# ---------------------------------------------------------------

def log_trace(question, mode, method, retrieved_or_steps, answer, sources, seconds):
    trace = {
        "timestamp": time.time(),
        "question": question,
        "mode": mode,               # "fixed" or "agent"
        "method": method,           # fixed: the --method used. agent: None
        "seconds": round(seconds, 4),
        "answer": answer,
        "sources": sources,
    }
    if mode == "fixed":
        trace["retrieved"] = [
            {"source": source, "distance": round(distance, 4), "chunk_preview": chunk[:200]}
            for chunk, source, distance in retrieved_or_steps
        ]
    else:
        trace["steps"] = retrieved_or_steps
        trace["step_count"] = sum(1 for s in retrieved_or_steps if "tool" in s)

    with open(TRACES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Your question, in quotes.")
    parser.add_argument("--method", choices=list(RETRIEVERS.keys()), default="embedding",
                         help="Fixed-workflow retrieval method. Ignored if --agent is set.")
    parser.add_argument("--provider", choices=list(GENERATORS.keys()), default=None)
    parser.add_argument("--agent", action="store_true",
                         help="Use the model-driven agent loop instead of the fixed one-shot pipeline (requires PROVIDER=openrouter).")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    provider = args.provider or PROVIDER
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    start = time.time()

    if args.agent:
        if provider != "openrouter":
            raise SystemExit("--agent currently requires --provider openrouter (tool calling).")
        answer, sources, steps = run_agent_rag(args.question, collection)
        elapsed = time.time() - start

        print(f"--- Agent steps ({sum(1 for s in steps if 'tool' in s)}) ---")
        for i, s in enumerate(steps, 1):
            if "tool" in s:
                print(f"  {i}. {s['tool']}({s['args']})")
            else:
                print(f"  {i}. STOP: {s['stop_reason']}")

        print("\n--- Answer ---")
        print(answer)
        if sources:
            print(f"\nSource(s): {', '.join(sources)}")
        print(f"\n(took {elapsed:.2f}s)")

        if not args.no_log:
            log_trace(args.question, "agent", None, steps, answer, sources, elapsed)

    else:
        retrieve_fn = RETRIEVERS[args.method]
        retrieved = retrieve_fn(args.question, collection)

        print(f"--- Retrieved chunks (method={args.method}) ---")
        for chunk, source, distance in retrieved:
            print(f"[{source}] distance={distance:.3f}\n{chunk[:120]}...\n")

        answer, sources = generate_answer(args.question, retrieved, provider=provider)
        elapsed = time.time() - start

        print("--- Answer ---")
        print(answer)
        if sources:
            print(f"\nSource(s): {', '.join(sources)}")
        print(f"\n(took {elapsed:.2f}s)")

        if not args.no_log:
            log_trace(args.question, "fixed", args.method, retrieved, answer, sources, elapsed)

    if not args.no_log:
        print(f"(logged to {TRACES_FILE})")


if __name__ == "__main__":
    main()