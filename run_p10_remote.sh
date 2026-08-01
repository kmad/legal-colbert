#!/usr/bin/env bash
# P10: aggressive round — uncovered MLEB types + FAR-52 register data.
# Questions this sweep answers:
#   - seed stability (a vs b identical recipes)
#   - batch-size lever (d: bs64 in-batch negatives)
#   - restart-from-p7b vs stacking-on-p8c/p8b (a/c vs e/f)
#   - mined vs bm25 negatives on the enriched data (c vs a)
set -euo pipefail
cd "$HOME/legal-colbert"
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HOME/.config/uv" output
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
uv venv --python 3.10 .venv 2>/dev/null || true
uv pip install --python .venv/bin/python pylate sentence-transformers datasets rank_bm25 accelerate 2>&1 | tail -2
PY=.venv/bin/python
$PY -c "import torch; print('cuda', torch.cuda.is_available())"

P7B=output/legal-colbert-p7b-depth-200/final
P8B=output/legal-colbert-p8b-mined-200/final
P8C=output/legal-colbert-p8c-bm25-300/final

# Datasets (combined provisions incl. FAR register data, shipped as p10_provisions.json)
[ -d data_p10_bm25/train ] || $PY build_extractor_data.py --provisions p10_provisions.json --cuad-records-json cuad_anchor_records.json \
  --output-dir data_p10_bm25 2>&1 | tail -6
[ -d data_p10_mined/train ] || $PY build_extractor_data.py --provisions p10_provisions.json --cuad-records-json cuad_anchor_records.json \
  --hard-negative-model "$P7B" --output-dir data_p10_mined 2>&1 | tail -6

eval_one () {
  local out="$1"; local name="$2"
  echo "=== [$(date -u)] EVAL $name ==="
  $PY benchmark.py --model "$out/final" --label "$name" \
    --output-json "$out.mleb.json" 2>&1 | tail -3
  $PY eval_clause.py --model-path "$out/final" --data-dir eval_blind_edgar_v2 \
    --split blind_v2adj --label "$name" --output-json "$out.blind_v2adj.json" 2>&1 | tail -4
  $PY eval_clause.py --model-path "$out/final" --data-dir eval_depth_v3 \
    --split depth_v3 --label "$name" --output-json "$out.depth_v3.json" 2>&1 | tail -4
  $PY eval_clause.py --model-path "$out/final" --data-dir eval_clause_mleb \
    --split clause_test --label "$name" --output-json "$out.clause_mleb.json" 2>&1 | tail -4
  $PY eval_v2.py --model-path "$out/final" --data-dir data_p2a_acord_cuad \
    --split acord_test --output-json "$out.acord_test.json" 2>&1 | tail -3
}

train_one () {
  local name="$1"; local base="$2"; local data="$3"; local steps="$4"; local lr="$5"; local seed="$6"; local bs="$7"; local ga="$8"
  local out="output/$name"
  echo "=== [$(date -u)] TRAIN $name base=$base data=$data steps=$steps lr=$lr seed=$seed bs=$bs ga=$ga ==="
  $PY train_v2.py \
    --data-dir "$data" \
    --model-name "$base" \
    --output-dir "$out" \
    --run-name "$name" \
    --num-epochs 1 \
    --max-steps "$steps" \
    --batch-size "$bs" \
    --gradient-accumulation-steps "$ga" \
    --learning-rate "$lr" \
    --temperature 0.05 \
    --warmup-ratio 0.02 \
    --eval-steps 1000 \
    --save-steps 100000 \
    --logging-steps 25 \
    --seed "$seed" \
    --bf16 2>&1 | tee "$out.train.log" | tail -3
  eval_one "$out" "$name"
}

# Baselines on depth_v3 (new diagnostic incl. uncovered types)
for pair in "p7b-baseline:$P7B" "p8b-baseline:$P8B" "p8c-baseline:$P8C" "p6b-baseline:output/legal-colbert-p6b-p5-anchor-300/final"; do
  nm="${pair%%:*}"; mp="${pair#*:}"
  $PY eval_clause.py --model-path "$mp" --data-dir eval_depth_v3 \
    --split depth_v3 --label "$nm" --output-json "output/$nm.depth_v3.json" 2>&1 | tail -3
done

train_one legal-colbert-p10a-bm25-300-s52 "$P7B" data_p10_bm25 300 7.5e-8 52 16 2
train_one legal-colbert-p10b-bm25-300-s53 "$P7B" data_p10_bm25 300 7.5e-8 53 16 2
train_one legal-colbert-p10c-mined-200 "$P7B" data_p10_mined 200 1e-7 54 16 2
train_one legal-colbert-p10d-bm25-bs64 "$P7B" data_p10_bm25 150 1e-7 55 64 1
train_one legal-colbert-p10e-stack-p8c "$P8C" data_p10_bm25 150 5e-8 56 16 2
train_one legal-colbert-p10f-stack-p8b "$P8B" data_p10_mined 150 5e-8 57 16 2

echo "=== [$(date -u)] P10 DONE ==="
