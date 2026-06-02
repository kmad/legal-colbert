#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/legal-colbert
export PATH="$HOME/.local/bin:$PATH"
PY=.venv/bin/python
out=output/legal-colbert-v10-gentlecap
echo "=== [$(date -u)] TRAIN v10 ==="
$PY train_v2.py --data-dir data_v10_clause_gentlecap --model-name model \
  --output-dir "$out" --run-name legal-colbert-v10-gentlecap \
  --num-epochs 1 --batch-size 16 --gradient-accumulation-steps 2 \
  --learning-rate 1e-6 --temperature 0.05 --warmup-ratio 0.05 \
  --eval-steps 500 --save-steps 500 --bf16 2>&1 | tee "$out.train.log" | tail -2
echo "=== [$(date -u)] EVAL v10 ==="
$PY eval_v2.py --model-path "$out/final" --data-dir data_v10_clause_gentlecap \
  --split all --output-json "$out.eval_metrics.json" 2>&1 | tail -1
echo "=== [$(date -u)] MLEB v10 ==="
$PY benchmark.py --model "$out/final" --label legal-colbert-v10-gentlecap \
  --output-json "$out.mleb.json" 2>&1 | grep -E "NDCG@10|MAP|Recall@10"
echo "=== [$(date -u)] V10 DONE ==="
