"""All tunable settings in one place, so nothing is hardcoded deep inside logic files."""

DOCS_DIR = "docs"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "dev_docs"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma4:latest"  # change to match whatever `ollama list` shows on your machine

TOP_K = 3
SIMILARITY_FLOOR = 0.35  # cosine distance above this = "not relevant enough", triggers "I don't know"

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50