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
| Grounded generation & citations | `generate_answer()` — the prompt forces the model to only use retrieved context and name its source |
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

## Project structure

```
src/ragapp/
├── cli.py             # main entry point — wires commands onto the Typer app
├── config.py          # every setting in one place
├── embeddings.py       # embed() — shared by ingest & query
├── store.py            # all ChromaDB access, isolated (swap vector DBs here later)
├── ingestion.py         # load_documents(), chunk_text() — pure logic
├── retrieval.py          # retrieve(), generate_answer() — pure logic
└── commands/
    ├── ingest.py          # CLI wrapper around ingestion.py
    └── query.py           # CLI wrapper around retrieval.py
```

Logic (`ingestion.py`, `retrieval.py`, `store.py`, `embeddings.py`) is separated from the CLI
layer (`commands/`) on purpose — the logic functions don't know or care that they're being called
from a terminal, which makes them easy to test or reuse later (e.g. behind a small web API).

To add a new command later: write a function in `commands/your_command.py`, then register it in
`cli.py` with one line (`app.command()(your_command)`).

## Setup

1. **Install Ollama** and make sure it's running (check system tray, or `ollama serve`).

2. **Pull the embedding model:**
   ```
   ollama pull nomic-embed-text
   ```

3. **Create a venv on Python 3.12** and activate it:
   ```
   py -3.12 -m venv venv
   venv\Scripts\activate
   python --version    # should print Python 3.12.x
   ```

4. **Install the package in editable mode** (this reads `pyproject.toml` and installs
   `chromadb`, `ollama`, and `typer` automatically, plus gives you a `ragapp` command):
   ```
   pip install -e .
   ```

## Run it

```
ragapp ingest
ragapp query "How do I rotate my API key?"
ragapp --help              # Typer auto-generates this
ragapp ingest --help       # ...and per-command help too
```

(If `ragapp` isn't recognized, use `python -m ragapp ingest` instead — same thing.)

Each answer prints the retrieved chunks with their **distance** (lower = more similar — cosine
distance ranges roughly 0 = identical meaning to 1+ = unrelated), then the final answer and which
source file it came from.

## Try different chunk sizes (mentor checkpoint item)

Re-ingesting rebuilds the whole database from scratch with new settings:

```
ragapp ingest --chunk-size 300 --overlap 50
ragapp query "How do I rotate my API key?"

ragapp ingest --chunk-size 800 --overlap 100
ragapp query "How do I rotate my API key?"
```

What to look for: smaller chunks tend to point more precisely at one specific idea, often giving
a lower (better) distance for narrow questions. Bigger chunks carry more surrounding context in
the answer, but mix multiple ideas into one vector, which can blur the match slightly. Note what
you observe — that's exactly what your mentor will ask about.

## Swap in your real SDK reference pages

When you get the actual assignment documents:
1. Drop your real `.md` files into `docs/` (delete the sample files if you want)
2. Re-run `ragapp ingest`

## Troubleshooting

- **`ragapp ingest` dies silently with no error, right after printing chunk counts** — this is
  the Python 3.13 / onnxruntime crash described above. Switch to a Python 3.12 venv.
- **Everything returns "I don't know" even for good questions** — check the collection is using
  cosine distance (`metadata={"hnsw:space": "cosine"}` in `store.py`, already set there). If you
  changed that, re-run `ragapp ingest` to rebuild the DB with the fix.
- **`pip install -e .` fails trying to compile numpy from source** — same root cause as above;
  the Python version doesn't have a pre-built wheel available. Use Python 3.12.
- **A script call to Ollama just hangs** — the Ollama background service probably isn't running.
  Test directly: `Invoke-RestMethod -Uri http://localhost:11434/api/embeddings -Method Post -Body '{"model":"nomic-embed-text","prompt":"hello"}' -ContentType "application/json"` — if that
  hangs too, start Ollama (open the app, or run `ollama serve`).

## Mentor checklist coverage

- [x] Can answer correctly from the documents (grounded prompt, not general knowledge)
- [x] Shows which document it came from (printed after every answer)
- [x] Says "I don't know" for out-of-scope questions (similarity floor + cosine distance)
- [ ] Tried more than one chunk size and noted the difference — **do this yourself, see above,
      and write down what you observed**
