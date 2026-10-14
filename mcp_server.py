"""
mcp_server.py — exposes your docs search as an MCP tool.

Any MCP host (ollmcp, Claude Desktop, a teammate's agent) can connect, discover the
`search_docs` tool, and call it. This server never runs an AI model to *answer* —
it only returns relevant passages. The host's LLM does the answering. That split
("the server offers tools; the AI runs on the host side") is the core MCP idea.

Run it:
    python mcp_server.py                 # stdio — a host launches it for you
    python mcp_server.py --http          # HTTP on 127.0.0.1:8000/mcp (this machine only)
    python mcp_server.py --http --lan    # HTTP on 0.0.0.0:8000/mcp (other machines can reach it)

IMPORTANT for stdio: never print() to stdout in this file or anything it imports.
stdout carries the protocol messages; a stray print corrupts them. Use stderr to log.
"""

import os
import sys
from pathlib import Path

# The app stores its index at the relative path "chroma_db". A host may launch this
# script from any folder, so pin the working directory to this file's folder first.
os.chdir(Path(__file__).resolve().parent)

import chromadb  # noqa: E402
from fastmcp import FastMCP  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402

from query import CHROMA_PATH, COLLECTION_NAME, RETRIEVERS  # noqa: E402

mcp = FastMCP("docs-search")


@mcp.tool()
def search_docs(question: str, top_k: int = 3, method: str = "hybrid") -> list[dict]:
    """Search the AcmeSDK developer documentation and return the most relevant passages.

    Use this for questions about authentication (API keys, scopes, status codes like
    401/403/429), installation, or the Jobs API. Each result has the passage text, the
    source file it came from, and a distance score (lower = more similar). Answer the
    user's question yourself using only these passages, and cite the source file.
    If nothing relevant comes back, say you don't know.

    Args:
        question: What to look up, in plain language or an exact term (e.g. "429").
        top_k: How many passages to return (1-10).
        method: "hybrid" (meaning + keywords, default), "embedding", or "bm25".
    """
    if method not in RETRIEVERS:
        raise ToolError(f"Unknown method '{method}'. Choose one of: {', '.join(RETRIEVERS)}.")
    top_k = max(1, min(top_k, 10))

    try:
        collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION_NAME)
    except Exception:
        # Recoverable error: tell the model/user exactly how to fix it.
        raise ToolError(
            "The document index doesn't exist yet. Run `python ingest.py` in the "
            "project folder, then try again."
        )

    retrieved = RETRIEVERS[method](question, collection, top_k=top_k)
    return [
        {"source": source, "distance": round(distance, 3), "text": chunk}
        for chunk, source, distance in retrieved
    ]


if __name__ == "__main__":
    if "--http" in sys.argv:
        host = "0.0.0.0" if "--lan" in sys.argv else "127.0.0.1"
        mcp.run(transport="http", host=host, port=8000)
    else:
        mcp.run()  # stdio