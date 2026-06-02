#!/usr/bin/env bash
# P6: aggressive bounded continuation sweep from preserved P5 weights.
# Selection: beat P5 on MLEB, with ACORD used as a regression check.
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

BASE=output/legal-colbert-p5-v9-ledgar-400/final

eval_one () {
  local out="$1"; local name="$2"
  echo "=== [$(date -u)] EVAL $name ==="
  $PY benchmark.py --model "$out/final" --label "$name" \
    --output-json "$out.mleb.json" 2>&1 | tee "$out.mleb.log"
  $PY eval_v2.py --model-path "$out/final" --data-dir data_p2a_acord_cuad \
    --split acord_test --output-json "$out.acord_test.json" 2>&1 | tee "$out.acord_test.log"
  $PY eval_clause.py --model-path "$out/final" --data-dir eval_clause_mleb \
    --split clause_test --label "$name" --output-json "$out.clause_mleb.json" 2>&1 | tee "$out.clause_mleb.log"
  $PY eval_v2.py --model-path "$out/final" --data-dir data_p3_ledgar_docside \
    --split ledgar_validation --output-json "$out.ledgar_validation.json" 2>&1 | tee "$out.ledgar_validation.log"
  $PY eval_clause.py --model-path "$out/final" --data-dir eval_contractnli_blind \
    --split contractnli_test --label "$name" --output-json "$out.contractnli.json" 2>&1 | tee "$out.contractnli.log"
}

train_one () {
  local name="$1"; local data="$2"; local steps="$3"; local lr="$4"; local seed="$5"
  local out="output/$name"
  echo "=== [$(date -u)] TRAIN $name data=$data steps=$steps lr=$lr seed=$seed ==="
  $PY train_v2.py \
    --data-dir "$data" \
    --model-name "$BASE" \
    --output-dir "$out" \
    --run-name "$name" \
    --num-epochs 1 \
    --max-steps "$steps" \
    --batch-size 16 \
    --gradient-accumulation-steps 2 \
    --learning-rate "$lr" \
    --temperature 0.05 \
    --warmup-ratio 0.02 \
    --eval-steps 200 \
    --save-steps 200 \
    --logging-steps 25 \
    --seed "$seed" \
    --bf16 2>&1 | tee "$out.train.log"
  eval_one "$out" "$name"
}

train_one legal-colbert-p6a-p5-anchor-200 data_p6_ledgar_acord_anchor 200 1e-7 43
train_one legal-colbert-p6b-p5-anchor-300 data_p6_ledgar_acord_anchor 300 7.5e-8 44
train_one legal-colbert-p6c-p5-ledgar-200 data_p3_ledgar_docside 200 1e-7 45

echo "=== [$(date -u)] P6 DONE ==="
