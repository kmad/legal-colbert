#!/usr/bin/env bash
# Create an L40S pod, copy P6 inputs, and launch the bounded P5 continuation sweep.
set -euo pipefail
KEY=/Users/kmad/.ssh/prime_legal_colbert_v2
SSHOPT="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
cd "$(dirname "$0")"

echo "Creating L40S pod..."
ID=$(prime availability list --plain --gpu-type L40S_48GB --gpu-count 1 2>&1 | grep -oE "^[0-9a-f]{6}" | head -1)
OUT=$(prime pods create --id "$ID" --gpu-type L40S_48GB --gpu-count 1 \
  --image ubuntu_22_cuda_12 --name legal-colbert-p6 --disk-size 90 --yes --plain 2>&1)
echo "$OUT"
POD=$(echo "$OUT" | grep -oE "pod [0-9a-f]{32}" | awk '{print $2}')
[ -z "$POD" ] && { echo "Pod create failed. Aborting."; exit 1; }
echo "POD=$POD"

echo "Waiting for SSH..."
IP=""
for _ in $(seq 1 50); do
  S=$(prime pods status "$POD" --plain 2>&1)
  L=$(echo "$S" | grep -E "^SSH" | sed 's/^SSH *//' | grep -oE "ubuntu@[0-9.]+" || true)
  if [ -n "$L" ]; then IP=$(echo "$L" | sed 's/ubuntu@//'); echo "SSH ready: $IP"; break; fi
  sleep 20
done
[ -z "$IP" ] && { echo "No SSH after wait. Check 'prime pods status $POD'."; exit 1; }

echo "Fixing perms + rsync..."
ssh $SSHOPT ubuntu@"$IP" 'sudo chown -R ubuntu:ubuntu ~/.config ~/.local 2>/dev/null; mkdir -p ~/legal-colbert/output ~/.config/uv' || true
rsync -az -e "ssh $SSHOPT" \
  train_v2.py eval_v2.py eval_clause.py benchmark.py run_p6_remote.sh \
  data_p3_ledgar_docside data_p6_ledgar_acord_anchor data_p2a_acord_cuad \
  eval_clause_mleb eval_contractnli_blind \
  ubuntu@"$IP":/home/ubuntu/legal-colbert/
rsync -az -e "ssh $SSHOPT" \
  output/legal-colbert-p5-v9-ledgar-400 \
  ubuntu@"$IP":/home/ubuntu/legal-colbert/output/

echo "Launching P6 in tmux..."
ssh $SSHOPT ubuntu@"$IP" 'cd ~/legal-colbert && (which tmux || sudo apt-get install -y tmux >/dev/null 2>&1); chmod +x run_p6_remote.sh && tmux new-session -d -s p6 "bash run_p6_remote.sh > run_p6.log 2>&1" && sleep 2 && tmux ls'
echo ""
echo "Launched. POD=$POD IP=$IP"
echo "Watch:  ssh $SSHOPT ubuntu@$IP 'tail -f ~/legal-colbert/run_p6.log'"
echo "Copy back targeted final dirs and metric files; then terminate: prime pods terminate $POD --yes"
