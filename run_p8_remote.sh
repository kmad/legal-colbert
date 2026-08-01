#!/usr/bin/env bash
# P8: extractor-line sweep from p7b on wide-family data with model-mined hard
# negatives. Arms isolate the hard-negative effect (p8c = BM25 control).
# Promotion happens LOCALLY against the adjudicated blind v2 gate; pod gates
# here are proxies (depth_v2 + MLEB floor + blind v1 continuity).
set -euo pipefail
cd /home/ubuntu/legal-colbert
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sudo chown -R ubuntu:ubuntu /home/ubuntu/.config 2>/dev/null || true
mkdir -p /home/ubuntu/.config/uv output
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
uv venv --python 3.10 .venv 2>/dev/null || true
uv pip install --python .venv/bin/python pylate sentence-transformers datasets rank_bm25 accelerate 2>&1 | tail -2
PY=.venv/bin/python
$PY -c "import torch; print('cuda', torch.cuda.is_available())"

BASE=output/legal-colbert-p7b-depth-200/final

# Build both datasets on-pod (model-mined + BM25 control)
$PY build_extractor_data.py --provisions extractor_provisions.json --cuad-records-json cuad_anchor_records.json \
  --hard-negative-model "$BASE" --output-dir data_p8_extractor_mined 2>&1 | tail -8
$PY build_extractor_data.py --provisions extractor_provisions.json --cuad-records-json cuad_anchor_records.json \
  --output-dir data_p8_extractor_bm25 2>&1 | tail -8

eval_one () {
  local out="$1"; local name="$2"
  echo "=== [$(date -u)] EVAL $name ==="
  $PY benchmark.py --model "$out/final" --label "$name" \
    --output-json "$out.mleb.json" 2>&1 | tee "$out.mleb.log" | tail -4
  $PY eval_clause.py --model-path "$out/final" --data-dir eval_depth_v2 \
    --split depth_v2 --label "$name" --output-json "$out.depth_v2.json" 2>&1 | tail -6
  $PY eval_clause.py --model-path "$out/final" --data-dir eval_blind_edgar_feed \
    --split blind_edgar --label "$name" --output-json "$out.blind_edgar.json" 2>&1 | tail -4
  $PY eval_clause.py --model-path "$out/final" --data-dir eval_blind_edgar_v2 \
    --split blind_v2adj --label "$name" --output-json "$out.blind_v2adj.json" 2>&1 | tail -4
  $PY eval_clause.py --model-path "$out/final" --data-dir eval_clause_mleb \
    --split clause_test --label "$name" --output-json "$out.clause_mleb.json" 2>&1 | tail -4
  $PY eval_v2.py --model-path "$out/final" --data-dir data_p2a_acord_cuad \
    --split acord_test --output-json "$out.acord_test.json" 2>&1 | tail -3
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
    --save-steps 10000 \
    --logging-steps 25 \
    --seed "$seed" \
    --bf16 2>&1 | tee "$out.train.log" | tail -3
  eval_one "$out" "$name"
}

# Baselines on the new depth_v2 diagnostic
$PY eval_clause.py --model-path "$BASE" --data-dir eval_depth_v2 \
  --split depth_v2 --label p7b-baseline \
  --output-json output/p7b-baseline.depth_v2.json 2>&1 | tail -4
$PY eval_clause.py --model-path output/legal-colbert-p6b-p5-anchor-300/final \
  --data-dir eval_depth_v2 --split depth_v2 --label p6b-baseline \
  --output-json output/p6b-baseline.depth_v2.json 2>&1 | tail -4
$PY eval_clause.py --model-path "$BASE" --data-dir eval_blind_edgar_v2 \
  --split blind_v2adj --label p7b-baseline \
  --output-json output/p7b-baseline.blind_v2adj.json 2>&1 | tail -4
$PY eval_clause.py --model-path output/legal-colbert-p6b-p5-anchor-300/final \
  --data-dir eval_blind_edgar_v2 --split blind_v2adj --label p6b-baseline \
  --output-json output/p6b-baseline.blind_v2adj.json 2>&1 | tail -4

train_one legal-colbert-p8a-mined-300 data_p8_extractor_mined 300 7.5e-8 49
train_one legal-colbert-p8b-mined-200 data_p8_extractor_mined 200 1e-7 50
train_one legal-colbert-p8c-bm25-300 data_p8_extractor_bm25 300 7.5e-8 51

echo "=== [$(date -u)] P8 DONE ==="
