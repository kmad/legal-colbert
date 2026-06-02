#!/usr/bin/env bash
# Resume P1: train_scored already exists; just run the two trainings with
# memory-safe settings (KD OOM'd at batch 16x16). Effective batch kept ~16.
set -euo pipefail
cd /home/ubuntu/legal-colbert
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
mkdir -p output

eval_one () {
  local out="$1"; local lbl="$2"
  echo "=== [$(date -u)] EVAL $lbl ==="
  $PY eval_clause.py --model-path "$out/final" --data-dir eval_clause_mleb --split clause_test --label "$lbl" --output-json "$out.clause_mleb.json" 2>&1 | grep -E "macro_ndcg@10|macro_map|n_big" | head
  $PY benchmark.py --model "$out/final" --label "$lbl" --output-json "$out.mleb.json" 2>&1 | grep -E "NDCG@10|MAP|Recall@10"
}

echo "=== [$(date -u)] TRAIN p1-distill (KD, 2 epochs, batch 4 x ga4) ==="
$PY train_distill.py --data-dir data_p1_distill --model-name model \
  --output-dir output/legal-colbert-p1-distill --run-name p1-distill \
  --num-epochs 2 --batch-size 4 --gradient-accumulation-steps 4 \
  --learning-rate 2e-6 --warmup-ratio 0.05 --n-ways 16 --bf16 \
  2>&1 | tee output/legal-colbert-p1-distill.train.log | tail -3
eval_one output/legal-colbert-p1-distill p1-distill

echo "=== [$(date -u)] TRAIN p1-contrastive (control, 2 epochs) ==="
$PY train_v2.py --data-dir data_p1_contrastive --model-name model \
  --output-dir output/legal-colbert-p1-contrastive --run-name p1-contrastive \
  --num-epochs 2 --batch-size 16 --gradient-accumulation-steps 2 \
  --learning-rate 2e-6 --temperature 0.05 --warmup-ratio 0.05 \
  --eval-steps 2000 --save-steps 2000 --bf16 \
  2>&1 | tee output/legal-colbert-p1-contrastive.train.log | tail -3
eval_one output/legal-colbert-p1-contrastive p1-contrastive
echo "=== [$(date -u)] P1 DONE ==="
