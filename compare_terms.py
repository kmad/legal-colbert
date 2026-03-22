"""
Pull sample contracts from CUAD and compare similar terms across them.

Extracts full contract text from CUAD QA contexts, chunks each contract,
then queries for common legal terms and shows side-by-side comparison.
"""

from collections import defaultdict
from datasets import load_dataset
from pylate import models

from chunk import Chunk, chunk_document, estimate_tokens
from retrieve import colbert_score


# Common legal terms to compare across contracts
QUERY_TERMS = [
    "limitation of liability",
    "indemnification",
    "governing law and jurisdiction",
    "termination for cause",
    "confidentiality and non-disclosure",
    "intellectual property ownership",
    "assignment of rights",
    "warranty disclaimer",
]

# Pick diverse contract types
TARGET_CONTRACTS = [
    "PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT",
    "HERTZGLOBALHOLDINGS,INC_07_07_2016-EX-10.4-INTELLECTUAL PROPERTY AGREEMENT",
    "LohaCompanyltd_20191209_F-1_EX-10.16_11917878_EX-10.16_Supply Agreement",
    "CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic Alliance Agreement",
]


def load_cuad_contracts(titles: list[str] | None = None, max_contracts: int = 4):
    """Load full contract texts from CUAD QA dataset.

    Each contract's text is reconstructed by concatenating unique contexts.
    """
    print("Loading CUAD QA dataset...")
    ds = load_dataset("theatticusproject/cuad-qa", split="test", trust_remote_code=True)

    # Group contexts by contract title
    contract_texts = defaultdict(list)
    seen = defaultdict(set)
    for ex in ds:
        title = ex["title"]
        ctx = ex["context"]
        ctx_key = ctx[:200]
        if ctx_key not in seen[title]:
            seen[title].add(ctx_key)
            contract_texts[title].append(ctx)

    if titles:
        selected = {t: contract_texts[t] for t in titles if t in contract_texts}
    else:
        # Pick contracts with most content
        by_size = sorted(contract_texts.items(), key=lambda x: sum(len(c) for c in x[1]), reverse=True)
        selected = dict(by_size[:max_contracts])

    contracts = {}
    for title, contexts in selected.items():
        full_text = "\n\n".join(contexts)
        short_name = title.split("-")[-1].strip() if "-" in title else title[:50]
        contracts[short_name] = full_text
        print(f"  {short_name}: {len(contexts)} sections, {len(full_text)} chars")

    return contracts


def chunk_contracts(contracts: dict[str, str], max_tokens: int = 384) -> dict[str, list[Chunk]]:
    """Chunk each contract."""
    all_chunks = {}
    for name, text in contracts.items():
        chunks = chunk_document(text, max_tokens=max_tokens)
        # Filter tiny chunks
        chunks = [c for c in chunks if len(c.text.split()) >= 10]
        all_chunks[name] = chunks
        print(f"  {name}: {len(chunks)} chunks")
    return all_chunks


def encode_all(
    model: models.ColBERT,
    contract_chunks: dict[str, list[Chunk]],
) -> dict[str, list]:
    """Encode all chunks for all contracts."""
    all_embeddings = {}
    for name, chunks in contract_chunks.items():
        texts = [c.full_text for c in chunks]
        if not texts:
            all_embeddings[name] = []
            continue
        embeddings = model.encode(texts, batch_size=32, is_query=False, show_progress_bar=False)
        all_embeddings[name] = embeddings
        print(f"  Encoded {name}: {len(embeddings)} chunks")
    return all_embeddings


def find_best_match(
    query_emb,
    chunks: list[Chunk],
    doc_embeddings: list,
) -> tuple[Chunk | None, float]:
    """Find the best matching chunk for a query."""
    if not chunks or not doc_embeddings:
        return None, 0.0
    best_chunk = None
    best_score = -float("inf")
    for i, doc_emb in enumerate(doc_embeddings):
        score = colbert_score(query_emb, doc_emb)
        if score > best_score:
            best_score = score
            best_chunk = chunks[i]
    return best_chunk, best_score


def compare_terms(
    model: models.ColBERT,
    contract_chunks: dict[str, list[Chunk]],
    contract_embeddings: dict[str, list],
    queries: list[str],
):
    """Query each contract for each term and display comparison."""
    contract_names = list(contract_chunks.keys())

    for query_text in queries:
        print(f"\n{'='*100}")
        print(f"  TERM: {query_text}")
        print(f"{'='*100}")

        query_emb = model.encode([query_text], batch_size=1, is_query=True, show_progress_bar=False)

        for name in contract_names:
            chunk, score = find_best_match(
                query_emb[0],
                contract_chunks[name],
                contract_embeddings[name],
            )
            if chunk is None:
                print(f"\n  [{name}] — No match")
                continue

            print(f"\n  [{name}]  Score: {score:.1f}  |  Page {chunk.page}")
            if chunk.header:
                print(f"  Path: {chunk.header}")
            # Show first 400 chars
            preview = chunk.text[:400].replace("\n", "\n    ")
            print(f"    {preview}")
            if len(chunk.text) > 400:
                print(f"    ...")


if __name__ == "__main__":
    import sys

    # Allow custom contract count
    max_contracts = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    # Load contracts
    contracts = load_cuad_contracts(titles=TARGET_CONTRACTS, max_contracts=max_contracts)

    # Chunk
    print("\nChunking contracts...")
    contract_chunks = chunk_contracts(contracts)

    # Load model and encode
    print("\nLoading model...")
    from pathlib import Path
    local_model = Path("model")
    if local_model.exists() and (local_model / "model.safetensors").exists():
        model = models.ColBERT(model_name_or_path=str(local_model))
    else:
        model = models.ColBERT(model_name_or_path="kmad00/legal-colbert-v1")

    print("\nEncoding chunks...")
    contract_embeddings = encode_all(model, contract_chunks)

    # Compare
    print("\n" + "#"*100)
    print("  CROSS-CONTRACT TERM COMPARISON")
    print("#"*100)
    compare_terms(model, contract_chunks, contract_embeddings, QUERY_TERMS)
