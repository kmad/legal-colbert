#!/usr/bin/env bash
# P2 on the pod. All continuations from V1 (model/), light touch (1 epoch).
# Clean ablation:
#   p2a       = clean CUAD + ACORD (grade>=3)             -> isolates ACORD
#   p2b       = p2a + query-paraphrase augmentation        -> isolates paraphrases
#   p2b-t02   = p2b at contrastive temperature 0.02        -> isolates temp (PyLate rec)
# Leakage-free selection signals: acord_test (validated, Spearman 1.0 with MLEB)
# + MLEB gate. eval_clause_mleb reported too (definitions disjoint from training).
set -euo pipefail
cd /home/ubuntu/legal-colbert
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /home/ubuntu/.config/uv
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
uv venv --python 3.10 .venv 2>/dev/null || true
uv pip install --python .venv/bin/python pylate sentence-transformers datasets rank_bm25 accelerate 2>&1 | tail -2
PY=.venv/bin/python
$PY -c "import torch;print('cuda',torch.cuda.is_available())"
mkdir -p output

eval_one () {
  local out="$1"; local lbl="$2"; local data="$3"
  echo "=== [$(date -u)] EVAL $lbl ==="
  $PY eval_v2.py --model-path "$out/final" --data-dir "$data" --split acord_test \
    --output-json "$out.acord_test.json" 2>&1 | grep -E "ndcg@10|map" | head -3
  $PY benchmark.py --model "$out/final" --label "$lbl" --output-json "$out.mleb.json" 2>&1 | grep -E "NDCG@10|MAP|Recall@10"
  $PY eval_clause.py --model-path "$out/final" --data-dir eval_clause_mleb --split clause_test \
    --label "$lbl" --output-json "$out.clause_mleb.json" 2>&1 | grep -E "macro_ndcg@10|n_big" | head -3
}

train_one () {
  local data="$1"; local name="$2"; local temp="$3"
  local out="output/$name"
  echo "=== [$(date -u)] TRAIN $name (data=$data temp=$temp) ==="
  $PY train_v2.py --data-dir "$data" --model-name model \
    --output-dir "$out" --run-name "$name" \
    --num-epochs 1 --batch-size 16 --gradient-accumulation-steps 2 \
    --learning-rate 1e-6 --temperature "$temp" --warmup-ratio 0.05 \
    --eval-steps 2000 --save-steps 2000 --bf16 2>&1 | tee "$out.train.log" | tail -2
  eval_one "$out" "$name" "$data"
}

train_one data_p2a_acord_cuad                legal-colbert-p2a-acord            0.05
train_one data_p2b_acord_cuad_paraphrased    legal-colbert-p2b-paraphrased      0.05
train_one data_p2b_acord_cuad_paraphrased    legal-colbert-p2b-paraphrased-t02  0.02
echo "=== [$(date -u)] P2 DONE ==="
