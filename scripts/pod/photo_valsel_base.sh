#!/usr/bin/env bash
# Photo GCN base, val protocol (branch exp/val-select): epoch by val AUROC, threshold on val. Same detector as Reddit _u.
cd /home/data/baed
export PATH=/home/data/.local/bin:$PATH DGLBACKEND=pytorch PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
L=outputs/logs/photo/valsel; R=outputs/results/photo/valsel; mkdir -p $L $R
echo "[$(date +%T)] start $(git rev-parse --short HEAD) $(git branch --show-current)"
.venv/bin/baed-detect --dataset photo --node-feats 745 --gae-weight photo_gin_relu_256_whole.pt --epochs 50 --seed 15 --batched \
  --data-tag "" --agg-features orig --readout deviation_cat_nb --model GCN --select val_auroc --lr 1e-3 \
  --out $R/GCN_noselfloop_s15.json > $L/GCN_noselfloop_s15.log 2>&1 < /dev/null
echo "[$(date +%T)] photo base done rc=$?"; grep -a "threshold chosen" $L/GCN_noselfloop_s15.log; echo "all done"
