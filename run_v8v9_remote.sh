#!/usr/bin/env bash
# Train + eval the V8 (better spans) and V9 (better spans + cap) candidates.
# Same 1-epoch continuation recipe as V4/V7. Deps/venv already set up by V7 run.
set -euo pipefail
cd /home/ubuntu/legal-colbert
export PATH="$HOME/.local/bin:$PATH"
PY=.venv/bin/python
mkdir -p output

run_one () {
  local data_dir="$1"; local name="$2"
  local out="output/$name"
  echo "=== [$(date -u)] TRAIN $name (data=$data_dir) ==="
  $PY train_v2.py \
    --data-dir "$data_dir" --model-name model \
    --output-dir "$out" --run-name "$name" \
    --num-epochs 1 --batch-size 16 --gradient-accumulation-steps 2 \
    --learning-rate 1e-6 --temperature 0.05 --warmup-ratio 0.05 \
    --eval-steps 500 --save-steps 500 --bf16 2>&1 | tee "$out.train.log" | tail -3
  echo "=== [$(date -u)] EVAL $name (cuad/acord) ==="
  $PY eval_v2.py --model-path "$out/final" --data-dir "$data_dir" \
    --split all --output-json "$out.eval_metrics.json" 2>&1 | tail -1
  echo "=== [$(date -u)] MLEB $name ==="
  $PY benchmark.py --model "$out/final" --label "$name" \
    --output-json "$out.mleb.json" 2>&1 | grep -E "NDCG@10|MAP|Recall@10"
}

run_one data_v8_clause_better_spans legal-colbert-v8-better-spans
run_one data_v9_clause_spans_capped legal-colbert-v9-spans-capped
echo "=== [$(date -u)] V8/V9 DONE ==="
