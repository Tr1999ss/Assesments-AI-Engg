"""
Offline tests for rag_agent.py — verifies the LOGIC (memory, similarity,
retrieval dispatch) without needing Ollama, Chroma, or an OpenRouter key.
Run this first, before touching your real services.

    python test_agent_logic.py
"""

import os
import json
import tempfile

import rag_agent


def test_cosine_similarity():
    identical = rag_agent._cosine_similarity([1, 2, 3], [1, 2, 3])
    opposite = rag_agent._cosine_similarity([1, 0], [-1, 0])
    orthogonal = rag_agent._cosine_similarity([1, 0], [0, 1])
    assert abs(identical - 1.0) < 1e-9, f"expected 1.0, got {identical}"
    assert abs(opposite - (-1.0)) < 1e-9, f"expected -1.0, got {opposite}"
    assert abs(orthogonal - 0.0) < 1e-9, f"expected 0.0, got {orthogonal}"
    print("PASS: _cosine_similarity")


def test_memory_round_trip(tmp_path):
    """Save a Q/A pair, then confirm a near-identical question recalls it,
    and a totally different question does not."""
    rag_agent.MEMORY_FILE = str(tmp_path / "agent_memory.json")

    # fake embed(): same text -> same vector, deterministic and fast (no ollama)
    def fake_embed(text):
        seed = sum(ord(c) for c in text)
        return [((seed * (i + 1)) % 97) / 97 for i in range(8)]

    rag_agent.embed = fake_embed

    rag_agent.save_memory_entry(
        "What does CONN_TIMEOUT mean?", "It means the proxy dropped the connection.", ["errors.md"]
    )

    hit = rag_agent.recall_memory("What does CONN_TIMEOUT mean?")
    assert hit is not None, "expected a memory hit for the exact same question"
    assert hit["answer"] == "It means the proxy dropped the connection."
    print(f"PASS: recall_memory finds exact match (similarity={hit['similarity']})")

    miss = rag_agent.recall_memory("How do I install the CLI on Windows?")
    assert miss is None, f"expected no match for an unrelated question, got {miss}"
    print("PASS: recall_memory correctly ignores an unrelated question")


def test_retrieve_dispatch_with_fake_collection():
    """Fakes a chromadb collection so retrieve_embedding/bm25/hybrid can run
    without a real vector DB — just checking the wiring, not real relevance."""

    class FakeCollection:
        def query(self, query_embeddings, n_results):
            return {
                "documents": [["Chunk about timeouts", "Chunk about installs"][:n_results]],
                "metadatas": [[{"source": "errors.md"}, {"source": "setup.md"}][:n_results]],
                "distances": [[0.1, 0.5][:n_results]],
            }

        def get(self, include):
            return {
                "documents": ["Chunk about timeouts", "Chunk about installs"],
                "metadatas": [{"source": "errors.md"}, {"source": "setup.md"}],
            }

    rag_agent.embed = lambda text: [0.1, 0.2, 0.3]  # fixed fake vector
    fake_collection = FakeCollection()

    for name, fn in rag_agent.RETRIEVERS.items():
        result = fn("timeout error", fake_collection, top_k=2)
        assert len(result) > 0, f"{name} returned nothing"
        assert all(len(r) == 3 for r in result), f"{name} rows should be (chunk, source, distance)"
        print(f"PASS: retrieve_{name} runs and returns well-formed rows -> {result[0][1]}")


if __name__ == "__main__":
    test_cosine_similarity()
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        test_memory_round_trip(Path(tmp))
    test_retrieve_dispatch_with_fake_collection()
    print("\nAll offline tests passed.")