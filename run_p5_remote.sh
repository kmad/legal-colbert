#!/usr/bin/env bash
# P5: tiny V9 continuation on document-side LEDGAR breadth only.
# Select primarily on MLEB; use ACORD as a drafting-query regression check.
# LEDGAR/ContractNLI are diagnostics because they do not align with MLEB in
# existing model comparisons.
set -euo pipefail
cd /home/ubuntu/legal-colbert
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /home/ubuntu/.config/uv output
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
uv venv --python 3.10 .venv 2>/dev/null || true
uv pip install --python .venv/bin/python pylate sentence-transformers datasets rank_bm25 accelerate 2>&1 | tail -2
PY=.venv/bin/python
$PY -c "import torch; print('cuda', torch.cuda.is_available())"

NAME=legal-colbert-p5-v9-ledgar-400
OUT=output/$NAME

echo "=== [$(date -u)] TRAIN $NAME ==="
$PY train_v2.py \
  --data-dir data_p3_ledgar_docside \
  --model-name output/legal-colbert-v9-spans-capped/final \
  --output-dir "$OUT" \
  --run-name "$NAME" \
  --num-epochs 1 \
  --max-steps 400 \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --learning-rate 2e-7 \
  --temperature 0.05 \
  --warmup-ratio 0.02 \
  --eval-steps 200 \
  --save-steps 200 \
  --logging-steps 25 \
  --bf16 2>&1 | tee "$OUT.train.log"

echo "=== [$(date -u)] EVAL $NAME ==="
$PY benchmark.py --model "$OUT/final" --label "$NAME" \
  --output-json "$OUT.mleb.json" 2>&1 | tee "$OUT.mleb.log"

$PY eval_v2.py --model-path "$OUT/final" --data-dir data_p2a_acord_cuad \
  --split acord_test --output-json "$OUT.acord_test.json" 2>&1 | tee "$OUT.acord_test.log"

$PY eval_clause.py --model-path "$OUT/final" --data-dir eval_clause_mleb \
  --split clause_test --label "$NAME" --output-json "$OUT.clause_mleb.json" 2>&1 | tee "$OUT.clause_mleb.log"

$PY eval_v2.py --model-path "$OUT/final" --data-dir data_p3_ledgar_docside \
  --split ledgar_validation --output-json "$OUT.ledgar_validation.json" 2>&1 | tee "$OUT.ledgar_validation.log"

$PY eval_clause.py --model-path "$OUT/final" --data-dir eval_contractnli_blind \
  --split contractnli_test --label "$NAME" --output-json "$OUT.contractnli.json" 2>&1 | tee "$OUT.contractnli.log"

echo "=== [$(date -u)] P5 DONE ==="
