#!/usr/bin/env bash
# Extra seeds of the val-protocol GCN base (the paper averages seeds 15, 42, 63, 87, 94), all in parallel.
# Usage: seeds_base.sh DATASET NUM_FEATURES [DATA_TAG]    e.g. seeds_base.sh photo 745
DS=$1; NF=$2; TAG=${3:-}
cd /home/data/baed
export PATH=/home/data/.local/bin:$PATH DGLBACKEND=pytorch PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
L=outputs/logs/$DS$TAG/valsel; R=outputs/results/$DS$TAG/valsel; mkdir -p $L $R
echo "[$(date +%T)] $DS$TAG seeds start $(git rev-parse --short HEAD)"
for S in 42 63 87 94; do
  ( .venv/bin/baed-detect --dataset $DS --node-feats $NF --gae-weight reddit_gin_relu_256_whole.pt --epochs 50 --seed $S --batched \
      --data-tag "$TAG" --agg-features orig --readout deviation_cat_nb --model GCN --select val_auroc --lr 1e-3 \
      --out $R/GCN_noselfloop_s$S.json > $L/GCN_noselfloop_s$S.log 2>&1 < /dev/null
    echo "[$(date +%T)] $DS$TAG seed $S done rc=$?" ) &
done
wait
echo "[$(date +%T)] all done"
