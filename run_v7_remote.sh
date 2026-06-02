#!/usr/bin/env bash
# Driver for the V7 balanced-CUAD continuation run on a Prime pod.
# Trains 1 epoch from local V1 model/ using the same recipe as V4, then evals
# CUAD dev/test (+ACORD diagnostic) and the MLEB clause-retrieval gate.
set -euo pipefail
cd /home/ubuntu/legal-colbert

OUT=output/legal-colbert-v7-balanced
mkdir -p output

echo "=== [$(date -u)] installing deps via uv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python "pylate" "sentence-transformers" "datasets" "rank_bm25" "accelerate" 2>&1 | tail -3
PY=.venv/bin/python
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=== [$(date -u)] training V7 (balanced CUAD, 1 epoch continuation from V1) ==="
$PY train_v2.py \
  --data-dir data_v7_clause_cuad_balanced \
  --model-name model \
  --output-dir "$OUT" \
  --run-name legal-colbert-v7-balanced \
  --num-epochs 1 \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-6 \
  --temperature 0.05 \
  --warmup-ratio 0.05 \
  --eval-steps 500 \
  --save-steps 500 \
  --bf16 2>&1 | tee "$OUT.train.log"

echo "=== [$(date -u)] eval on data_v7 splits (cuad_dev/test, acord) ==="
$PY eval_v2.py \
  --model-path "$OUT/final" \
  --data-dir data_v7_clause_cuad_balanced \
  --split all \
  --output-json "$OUT.eval_metrics.json" 2>&1 | tee "$OUT.eval.log"

echo "=== [$(date -u)] MLEB clause-retrieval gate ==="
$PY benchmark.py \
  --model "$OUT/final" \
  --label legal-colbert-v7-balanced \
  --output-json "$OUT.mleb.json" 2>&1 | tee "$OUT.mleb.log"

echo "=== [$(date -u)] V7 DONE ==="
