# Ask My Docs — local RAG app (Developer Docs track, Week 3)

A minimal "ask my documents" app that runs 100% locally using Ollama — no API keys, no cost,
no data leaving your machine.

Given a question, it finds the most relevant chunks from your documents, answers **only** using
those chunks, names which file the answer came from, and says "I don't know" if nothing relevant
is found — instead of making something up.

## How this maps to the course concepts

| Course topic | Where it happens in this code |
|---|---|
| Loading documents | `load_documents()` in `ingest.py` |
| Chunking strategies, chunk size & overlap | `chunk_text()` in `ingest.py` — try different `--chunk-size` / `--overlap` |
| Embeddings & dense retrieval | `embed()` in both files, using Ollama's `nomic-embed-text` model |
| Vector database | ChromaDB, stored locally in `chroma_db/` |
| Similarity search & top-K | `retrieve()` in `query.py`, using cosine distance |
| Grounded generation & citations | `generate_answer()` in `query.py` — the prompt forces the model to only use retrieved context and name its source |
| Saying "I don't know" | `SIMILARITY_FLOOR` check in `generate_answer()` — if nothing retrieved is close enough, the LLM is never even called |

## Requirements

- **Python 3.12** (important — see note below)
- [Ollama](https://ollama.com/download) installed and running

> **Why 3.12 and not 3.13?** Some of chromadb's native dependencies (e.g. `onnxruntime`) don't
> yet have stable support for Python 3.13, and can crash silently with no error message
> (Windows access violation). Python 3.12 has full, stable wheel support across the board.
> If you only have 3.13 installed, grab 3.12 from
> [python.org/downloads/release/python-3120](https://www.python.org/downloads/release/python-3120/)
> — it installs alongside 3.13 without removing it.

## Setup

1. **Install Ollama** (if you haven't): https://ollama.com/download — make sure it's actually
   running (check your system tray, or run `ollama serve` in its own terminal window).

2. **Pull the embedding model:**
   ```
   ollama pull nomic-embed-text
   ```
   This is a small model whose only job is turning text into a vector (list of numbers) for
   similarity search — different from a chat model. Your chat model can be anything you already
   have pulled (`ollama list` to check) — set it in `query.py` via `CHAT_MODEL`.

3. **Create a venv using Python 3.12 specifically and activate it:**
   ```
   py -3.12 -m venv venv
   venv\Scripts\activate
   python --version    # should print Python 3.12.x
   ```

4. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Run it

**Step 1 — Ingest the docs** (load → chunk → embed → store):
```
python ingest.py
```
This processes everything in `docs/`. It ships with 3 sample SDK reference pages
(`authentication.md`, `installation.md`, `jobs-api.md`) so you can test the pipeline immediately.
Prints progress per chunk so any failure is easy to spot.

**Step 2 — Ask questions:**
```
python query.py "How do I rotate my API key?"
python query.py "How many times can I retry a failed job?"
python query.py "What's the capital of France?"     # should say "I don't know"
```

Each answer prints the retrieved chunks with their **distance** first (lower = more similar —
cosine distance ranges roughly 0 = identical meaning to 1+ = unrelated), then the final answer
and which source file it came from.

## Try different chunk sizes (mentor checkpoint item)

Re-ingesting rebuilds the whole database from scratch with new settings:

```
python ingest.py --chunk-size 300 --overlap 50
python query.py "How do I rotate my API key?"

python ingest.py --chunk-size 800 --overlap 100
python query.py "How do I rotate my API key?"
```

What to look for: smaller chunks tend to point more precisely at one specific idea, often giving
a lower (better) distance for narrow questions. Bigger chunks carry more surrounding context in
the answer, but mix multiple ideas into one vector, which can blur the match slightly. Write down
what you actually observed 

## Swap in your real SDK reference pages

When you get the actual assignment documents:
1. Drop your real `.md` files into `docs/` (delete the sample files if you want)
2. Re-run `python ingest.py`

## Troubleshooting

- **`ingest.py` dies silently with no error, right after printing chunk counts** — this is the
  Python 3.13 / onnxruntime crash described above. Switch to a Python 3.12 venv.
- **Everything returns "I don't know" even for good questions** — check the collection is using
  cosine distance (`metadata={"hnsw:space": "cosine"}` in `create_collection`, already set in
  `ingest.py`). If you changed that, re-run `ingest.py` to rebuild the DB with the fix.
- **`pip install` fails trying to compile numpy from source** — same root cause as above; the
  Python version doesn't have a pre-built wheel available. Use Python 3.12.
- **A script call to Ollama just hangs** — the Ollama background service probably isn't running.
  Test directly:
  ```
  Invoke-RestMethod -Uri http://localhost:11434/api/embeddings -Method Post -Body '{"model":"nomic-embed-text","prompt":"hello"}' -ContentType "application/json"
  ```
  If that hangs too, start Ollama (open the app, or run `ollama serve`).

## Mentor checklist coverage

- [x] Can answer correctly from the documents (grounded prompt, not general knowledge)
- [x] Shows which document it came from (printed after every answer)
- [x] Says "I don't know" for out-of-scope questions (similarity floor + cosine distance)
- [ ] Tried more than one chunk size and noted the difference — **do this yourself, see above,
      and write down what you observed**
