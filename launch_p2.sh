#!/usr/bin/env bash
# One-shot P2 launcher: create an L40S pod, wait for it, rsync data+scripts, run.
# Prereq: Prime billing must be active (https://app.primeintellect.ai/dashboard/billing).
# Usage:  bash launch_p2.sh
set -euo pipefail
KEY=/Users/kmad/.ssh/prime_legal_colbert_v2
SSHOPT="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
cd "$(dirname "$0")"

echo "Creating L40S pod..."
ID=$(prime availability list --plain --gpu-type L40S_48GB --gpu-count 1 2>&1 | grep -oE "^[0-9a-f]{6}" | head -1)
OUT=$(prime pods create --id "$ID" --gpu-type L40S_48GB --gpu-count 1 \
  --image ubuntu_22_cuda_12 --name legal-colbert-p2 --disk-size 80 --yes --plain 2>&1)
echo "$OUT"
POD=$(echo "$OUT" | grep -oE "pod [0-9a-f]{32}" | awk '{print $2}')
[ -z "$POD" ] && { echo "Pod create failed (billing?). Aborting."; exit 1; }
echo "POD=$POD"

echo "Waiting for SSH..."
IP=""
for i in $(seq 1 50); do
  S=$(prime pods status "$POD" --plain 2>&1)
  L=$(echo "$S" | grep -E "^SSH" | sed 's/^SSH *//' | grep -oE "ubuntu@[0-9.]+" || true)
  if [ -n "$L" ]; then IP=$(echo "$L" | sed 's/ubuntu@//'); echo "SSH ready: $IP"; break; fi
  sleep 20
done
[ -z "$IP" ] && { echo "No SSH after wait. Check 'prime pods status $POD'."; exit 1; }

echo "Fixing perms + rsync..."
ssh $SSHOPT ubuntu@"$IP" 'sudo chown -R ubuntu:ubuntu ~/.config ~/.local 2>/dev/null; mkdir -p ~/legal-colbert ~/.config/uv' || true
rsync -az -e "ssh $SSHOPT" \
  model data_p2a_acord_cuad data_p2b_acord_cuad_paraphrased eval_clause_mleb \
  train_v2.py eval_v2.py eval_clause.py benchmark.py run_p2_remote.sh \
  ubuntu@"$IP":/home/ubuntu/legal-colbert/

echo "Launching P2 in tmux..."
ssh $SSHOPT ubuntu@"$IP" 'cd ~/legal-colbert && (which tmux || sudo apt-get install -y tmux >/dev/null 2>&1); chmod +x run_p2_remote.sh && tmux new-session -d -s p2 "bash run_p2_remote.sh > run_p2.log 2>&1" && sleep 2 && tmux ls'
echo ""
echo "Launched. POD=$POD IP=$IP"
echo "Watch:  ssh $SSHOPT ubuntu@$IP 'tail -f ~/legal-colbert/run_p2.log'"
echo "When done, copy back metrics and: prime pods terminate $POD --yes"
