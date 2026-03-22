"""
End-to-end retrieval pipeline for legal documents using ColBERT.

Usage:
    # Index a contract and run queries
    python retrieve.py interactive example_contract.pdf

    # Index only
    python retrieve.py index example_contract.pdf

    # Query a previously indexed contract
    python retrieve.py query "What are the events of default?"
"""

import pickle
import sys
from pathlib import Path

import torch
from pylate import models

from chunk import Chunk, chunk_pdf


INDEX_DIR = "legal-index"
CHUNKS_PATH = "legal-index/chunks.pkl"
EMBEDDINGS_PATH = "legal-index/embeddings.pkl"
DEFAULT_MODEL = "kmad00/legal-colbert-v1"


def load_model(model_name: str = DEFAULT_MODEL) -> models.ColBERT:
    """Load the ColBERT model. Uses local model/ dir if available."""
    local_model = Path("model")
    if local_model.exists() and (local_model / "model.safetensors").exists():
        print(f"Loading model from local: {local_model}")
        return models.ColBERT(model_name_or_path=str(local_model))
    print(f"Loading model from HuggingFace: {model_name}")
    return models.ColBERT(model_name_or_path=model_name)


def colbert_score(query_emb, doc_emb) -> float:
    """Compute ColBERT MaxSim score between a query and document embedding.

    For each query token, find max similarity across all doc tokens, then sum.
    """
    # query_emb: (num_query_tokens, dim)
    # doc_emb: (num_doc_tokens, dim)
    q = torch.tensor(query_emb) if not isinstance(query_emb, torch.Tensor) else query_emb
    d = torch.tensor(doc_emb) if not isinstance(doc_emb, torch.Tensor) else doc_emb
    # Similarity matrix: (num_query_tokens, num_doc_tokens)
    sim = torch.matmul(q, d.T)
    # MaxSim: for each query token, take max over doc tokens
    max_sim = sim.max(dim=1).values
    return max_sim.sum().item()


def index_document(pdf_path: str, model_name: str = DEFAULT_MODEL, max_tokens: int = 384):
    """Chunk a PDF and encode all chunks."""
    print(f"Chunking {pdf_path}...")
    chunks = chunk_pdf(pdf_path, max_tokens=max_tokens)
    print(f"  {len(chunks)} chunks")

    # Filter out tiny chunks (< 5 tokens)
    chunks = [c for c in chunks if len(c.text.split()) >= 5]
    print(f"  {len(chunks)} after filtering tiny chunks")

    model = load_model(model_name)

    # Encode chunks with section-path context prepended
    doc_texts = [c.full_text for c in chunks]

    print("Encoding chunks...")
    doc_embeddings = model.encode(
        doc_texts,
        batch_size=32,
        is_query=False,
        show_progress_bar=True,
    )

    # Save chunks and embeddings
    Path(INDEX_DIR).mkdir(exist_ok=True)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(doc_embeddings, f)

    print(f"Indexed: {len(chunks)} chunks saved to {INDEX_DIR}/")
    return model, chunks, doc_embeddings


def load_index():
    """Load previously saved chunks and embeddings."""
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    with open(EMBEDDINGS_PATH, "rb") as f:
        doc_embeddings = pickle.load(f)
    return chunks, doc_embeddings


def query(
    query_text: str,
    model: models.ColBERT,
    chunks: list[Chunk],
    doc_embeddings: list,
    k: int = 5,
) -> list[tuple[Chunk, float]]:
    """Query the indexed chunks and return top-k results with scores."""
    # Encode query
    query_embedding = model.encode(
        [query_text],
        batch_size=1,
        is_query=True,
        show_progress_bar=False,
    )

    # Score every chunk using ColBERT MaxSim
    scores = []
    for i, doc_emb in enumerate(doc_embeddings):
        score = colbert_score(query_embedding[0], doc_emb)
        scores.append((i, score))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    # Return top-k
    results = []
    for idx, score in scores[:k]:
        results.append((chunks[idx], score))

    return results


def print_results(query_text: str, results: list[tuple[Chunk, float]]):
    """Pretty-print query results."""
    print(f"\n{'='*80}")
    print(f"Query: {query_text}")
    print(f"{'='*80}\n")

    for i, (chunk, score) in enumerate(results):
        print(f"  [{i+1}] Score: {score:.1f}  |  {chunk.chunk_type}  |  Page {chunk.page}")
        print(f"      Path: {chunk.header}")
        # Show first 300 chars of chunk text
        preview = chunk.text[:300].replace("\n", "\n      ")
        print(f"      {preview}")
        if len(chunk.text) > 300:
            print(f"      ...")
        print()


def interactive_mode(pdf_path: str):
    """Index a document and enter interactive query loop."""
    model, chunks, doc_embeddings = index_document(pdf_path)

    print("\n" + "="*80)
    print("Ready. Type a query and press Enter. Type 'quit' to exit.")
    print("="*80 + "\n")

    while True:
        try:
            query_text = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query_text or query_text.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        results = query(query_text, model=model, chunks=chunks, doc_embeddings=doc_embeddings, k=5)
        print_results(query_text, results)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python retrieve.py index <pdf_path>")
        print("  python retrieve.py query <query_text>")
        print("  python retrieve.py interactive <pdf_path>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "index":
        pdf_path = sys.argv[2] if len(sys.argv) > 2 else "example_contract.pdf"
        index_document(pdf_path)

    elif command == "query":
        query_text = " ".join(sys.argv[2:])
        model = load_model()
        chunks, doc_embeddings = load_index()
        results = query(query_text, model=model, chunks=chunks, doc_embeddings=doc_embeddings)
        print_results(query_text, results)

    elif command == "interactive":
        pdf_path = sys.argv[2] if len(sys.argv) > 2 else "example_contract.pdf"
        interactive_mode(pdf_path)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
