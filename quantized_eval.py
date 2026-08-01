"""Battery for a quantized ONNX export of a legal-colbert checkpoint.

Two stages:
  1. Encode parity vs the fp32 PyLate reference (embedding diff, score diff,
     ranking agreement on fixture pairs).
  2. Full retrieval gates (MLEB, clause_mleb, blind_v2adj, depth_v3) scored
     entirely through the quantized encoder, using the exact protocols of
     evaluate.py so numbers are comparable with stored fp32 results.

The encode contract ([Q]/[D] prefixes, no query expansion, document skiplist,
truncation-with-trailing-SEP, per-token L2 norm) is read from the quantized
bundle's own config_sentence_transformers.json + model_meta.json.

Usage:
    python quantized_eval.py --onnx-dir ~/dev/contract_browser/public/models/legal-colbert \
        --reference-model output/legal-colbert-p10f-stack-p8b/final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2n(e: np.ndarray) -> np.ndarray:
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)


def maxsim(q: np.ndarray, d: np.ndarray) -> float:
    return float((q @ d.T).max(axis=1).sum())


def ndcg(rels, k=10):
    dcg = lambda r: sum(x / np.log2(i + 2) for i, x in enumerate(r[:k]))
    ideal = dcg(sorted(rels, reverse=True))
    return dcg(rels) / ideal if ideal > 0 else 0.0


class QuantEncoder:
    def __init__(self, onnx_dir: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_files = sorted(onnx_dir.glob("*.onnx"))
        assert onnx_files, f"no .onnx in {onnx_dir}"
        self.session = ort.InferenceSession(str(onnx_files[0]), providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(onnx_dir / "tokenizer.json"))
        # PyLate checkpoints persist `padding: Fixed 299` (pad token [MASK]) in
        # tokenizer.json; Tokenizer.from_file honors it, so a short string
        # encodes to 299 mostly-[MASK] ids. Combined with a fabricated all-ones
        # attention mask this silently destroys retrieval (the 2026-08-01
        # audit's original near-random result). Strip persisted state and
        # sanity-check.
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        probe = self.tokenizer.encode("short probe", add_special_tokens=True)
        assert len(probe.ids) < 20, (
            f"tokenizer still pads short input to {len(probe.ids)} ids — "
            "persisted padding state survived; do not trust masks built as ones"
        )
        cfg = json.loads((onnx_dir / "config_sentence_transformers.json").read_text())
        self.q_prefix = cfg.get("query_prefix", "[Q] ")
        self.d_prefix = cfg.get("document_prefix", "[D] ")
        self.q_len = cfg.get("query_length", 48)
        self.d_len = cfg.get("document_length", 300)
        assert not cfg.get("do_query_expansion", False), "expansion unsupported here"
        skip = set(cfg.get("skiplist_words") or [])
        self.skip_ids = {t for t in (self.tokenizer.token_to_id(c) for c in sorted(skip)) if t is not None}

    def encode(self, text: str, is_query: bool) -> np.ndarray:
        prefix = self.q_prefix if is_query else self.d_prefix
        ids = list(self.tokenizer.encode(prefix + text, add_special_tokens=True).ids)
        limit = self.q_len if is_query else self.d_len
        if len(ids) > limit:
            ids = ids[: limit - 1] + [ids[-1]]
        arr = np.asarray([ids], dtype=np.int64)
        out = self.session.run(None, {"input_ids": arr, "attention_mask": np.ones_like(arr)})[0][0]
        if not is_query:
            keep = [i for i, t in enumerate(ids) if t not in self.skip_ids]
            out = out[keep]
        return l2n(out)


def gate_scores(enc, queries: dict, corpus: dict, qrels: dict, protocol: str, categories=None):
    cids = list(corpus)
    c_embs = [enc.encode(corpus[c], is_query=False) for c in cids]
    out_per_cat = {}
    ndcgs, aps = [], []
    for qid, qtext in queries.items():
        qe = enc.encode(qtext, is_query=True)
        order = sorted(range(len(cids)), key=lambda i: -maxsim(qe, c_embs[i]))
        ranked = [cids[i] for i in order]
        row = qrels.get(qid, {})
        if protocol == "graded_mean":
            rels = [row.get(d, 0) for d in ranked]
            relevant = {d for d, s in row.items() if s >= 1}
        else:
            relevant = {d for d, s in row.items() if s >= 1}
            rels = [1 if d in relevant else 0 for d in ranked]
        n = ndcg(rels)
        ap = 0.0
        if relevant:
            hits = tot = 0
            for rank, d in enumerate(ranked, 1):
                if d in relevant:
                    hits += 1
                    tot += hits / rank
            ap = tot / len(relevant)
        if protocol == "macro_binary":
            out_per_cat[(categories or {}).get(qid, qid)] = {
                "ndcg@10": round(n, 4), "map": round(ap, 4), "n_relevant": len(relevant),
            }
        else:
            ndcgs.append(n)
            aps.append(ap)
    if protocol == "macro_binary":
        cats = sorted(out_per_cat)
        big = [c for c in cats if out_per_cat[c]["n_relevant"] >= 5]
        return {
            "macro_ndcg@10": round(float(np.mean([out_per_cat[c]["ndcg@10"] for c in cats])), 4),
            "macro_ndcg@10_n>=5": round(float(np.mean([out_per_cat[c]["ndcg@10"] for c in big])), 4) if big else None,
        }
    return {"ndcg@10": round(float(np.mean(ndcgs)), 4), "map": round(float(np.mean(aps)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--reference-model", required=True)
    ap.add_argument("--output-json", default="output/quantized_eval.json")
    args = ap.parse_args()
    onnx_dir = Path(args.onnx_dir).expanduser()

    enc = QuantEncoder(onnx_dir)
    results: dict = {"onnx_dir": str(onnx_dir)}
    meta_p = onnx_dir / "model_meta.json"
    if meta_p.exists():
        results["quantized_meta"] = json.loads(meta_p.read_text())

    # ---- Stage 1: encode parity vs fp32 pylate ----
    from pylate import models
    import torch

    ref = models.ColBERT(model_name_or_path=args.reference_model)
    fixtures_docs = [
        "The Company may terminate this Agreement for convenience upon thirty (30) days prior written notice to Supplier.",
        "Neither party may assign this Agreement without the prior written consent of the other party, not to be unreasonably withheld.",
        "All notices under this Agreement shall be in writing and delivered to the addresses set forth on Schedule A.",
        "Each party shall indemnify and hold harmless the other from third-party claims arising out of its gross negligence.",
        "This Agreement shall be governed by the laws of the State of Delaware without regard to conflicts of law principles.",
    ]
    fixtures_queries = [
        'Find provisions related to "Termination for Convenience".',
        'Find provisions related to "No Assignment".',
        'Find provisions related to "Notices".',
        'Find provisions related to "Indemnification".',
        'Find provisions related to "Governing Law".',
    ]
    ref_d = ref.encode(fixtures_docs, is_query=False, show_progress_bar=False)
    ref_q = ref.encode(fixtures_queries, is_query=True, show_progress_bar=False)
    emb_diffs, score_pairs, rank_hits = [], [], 0
    for qi, qt in enumerate(fixtures_queries):
        qe_q = enc.encode(qt, is_query=True)
        qe_r = np.asarray(torch.as_tensor(ref_q[qi]))
        if qe_q.shape == qe_r.shape:
            emb_diffs.append(float(np.abs(qe_q - qe_r).max()))
        q_scores, r_scores = [], []
        for di, dt in enumerate(fixtures_docs):
            de_q = enc.encode(dt, is_query=False)
            de_r = np.asarray(torch.as_tensor(ref_d[di]))
            if de_q.shape == de_r.shape:
                emb_diffs.append(float(np.abs(de_q - de_r).max()))
            sq = maxsim(qe_q, de_q)
            sr = maxsim(qe_r, de_r)
            q_scores.append(sq)
            r_scores.append(sr)
            score_pairs.append((sq, sr))
        if int(np.argmax(q_scores)) == int(np.argmax(r_scores)) == qi:
            rank_hits += 1
    rel = [abs(a - b) / max(abs(b), 1e-9) for a, b in score_pairs]
    results["parity"] = {
        "max_emb_abs_diff": round(max(emb_diffs), 5) if emb_diffs else None,
        "max_score_rel_diff": round(max(rel), 5),
        "mean_score_rel_diff": round(float(np.mean(rel)), 5),
        "fixture_top1_agreement": f"{rank_hits}/5",
    }
    print("PARITY:", json.dumps(results["parity"]))

    # ---- Stage 2: gates ----
    from datasets import load_dataset

    DID = "isaacus/contractual-clause-retrieval"
    REV = "48ed7bcb1f50896a0f71461a04b2df0ca84329d9"
    corpus = {r["_id"]: r["text"] for r in load_dataset(DID, "corpus", revision=REV, split="corpus")}
    queries = {r["_id"]: r["text"] for r in load_dataset(DID, "queries", revision=REV, split="queries")}
    qrels: dict = {}
    for r in load_dataset(DID, "default", revision=REV, split="test"):
        qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = r["score"]
    results["mleb"] = gate_scores(enc, queries, corpus, qrels, "graded_mean")
    print("MLEB:", json.dumps(results["mleb"]))

    def load_split(d, prefix):
        base = Path(d)
        j = lambda k: json.loads((base / f"{prefix}_{k}.json").read_text())
        cats = j("categories") if (base / f"{prefix}_categories.json").exists() else None
        return j("queries"), j("corpus"), j("qrels"), cats

    for gate, d, prefix in [
        ("blind_v2adj", "eval_blind_edgar_v2", "blind_v2adj"),
        ("depth_v3", "eval_depth_v3", "depth_v3"),
        ("clause_mleb", "eval_clause_mleb", "clause_test"),
    ]:
        q, c, r, cats = load_split(d, prefix)
        results[gate] = gate_scores(enc, q, c, r, "macro_binary", cats)
        print(f"{gate}:", json.dumps(results[gate]))

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(results, indent=2))
    print("wrote", args.output_json)


if __name__ == "__main__":
    main()
