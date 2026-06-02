#!/usr/bin/env bash
# P1 on the pod: cross-encoder teacher scoring, then two continuation runs from
# V1 (model/) on descriptive queries + clean spans + hard negatives:
#   p1-distill     = KD from BGE-reranker-v2-m3
#   p1-contrastive = same candidates, plain contrastive (isolates distillation)
# Select on the trusted eval_clause_mleb clause_test macro_ndcg@10; MLEB is the gate.
set -euo pipefail
cd /home/ubuntu/legal-colbert
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
uv venv --python 3.10 .venv 2>/dev/null || true
uv pip install --python .venv/bin/python pylate sentence-transformers datasets rank_bm25 accelerate 2>&1 | tail -2
PY=.venv/bin/python
$PY -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
mkdir -p output

echo "=== [$(date -u)] teacher scoring (BGE-reranker-v2-m3) ==="
$PY score_teacher.py --data-dir data_p1_distill --teacher BAAI/bge-reranker-v2-m3 --batch-size 128 2>&1 | tail -3

eval_one () {
  local out="$1"; local lbl="$2"
  echo "=== [$(date -u)] EVAL $lbl ==="
  $PY eval_clause.py --model-path "$out/final" --data-dir eval_clause_mleb --split clause_test --label "$lbl" --output-json "$out.clause_mleb.json" 2>&1 | grep -E "macro_ndcg@10|macro_map|n_big" | head
  $PY benchmark.py --model "$out/final" --label "$lbl" --output-json "$out.mleb.json" 2>&1 | grep -E "NDCG@10|MAP|Recall@10"
}

echo "=== [$(date -u)] TRAIN p1-distill (KD, 2 epochs, from V1) ==="
$PY train_distill.py --data-dir data_p1_distill --model-name model \
  --output-dir output/legal-colbert-p1-distill --run-name p1-distill \
  --num-epochs 2 --batch-size 16 --learning-rate 2e-6 --warmup-ratio 0.05 --n-ways 16 --bf16 \
  2>&1 | tee output/legal-colbert-p1-distill.train.log | tail -3
eval_one output/legal-colbert-p1-distill p1-distill

echo "=== [$(date -u)] TRAIN p1-contrastive (control, 2 epochs, from V1) ==="
$PY train_v2.py --data-dir data_p1_contrastive --model-name model \
  --output-dir output/legal-colbert-p1-contrastive --run-name p1-contrastive \
  --num-epochs 2 --batch-size 32 --gradient-accumulation-steps 1 \
  --learning-rate 2e-6 --temperature 0.05 --warmup-ratio 0.05 \
  --eval-steps 1000 --save-steps 1000 --bf16 \
  2>&1 | tee output/legal-colbert-p1-contrastive.train.log | tail -3
eval_one output/legal-colbert-p1-contrastive p1-contrastive

echo "=== [$(date -u)] P1 DONE ==="
