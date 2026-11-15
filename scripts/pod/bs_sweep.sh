#!/usr/bin/env bash
# DGraph batch-size sweep (--gpu-store, 5 epochs each, in parallel): time per epoch and validation AUROC per epoch,
# to compare with batch 128 (dgraph/valsel/GCN_noselfloop_s15.log). Usage: bs_sweep.sh [DATA_TAG] [EPOCHS] [SIZES...]
TAG=${1:-}; EP=${2:-5}; shift 2 2>/dev/null; SIZES=(${@:-512 1024 2048 4096})
cd /home/data/baed
export PATH=/home/data/.local/bin:$PATH DGLBACKEND=pytorch PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
L=outputs/logs/dgraph$TAG/bs_sweep; R=outputs/results/dgraph$TAG/bs_sweep; mkdir -p $L $R
echo "[$(date +%T)] start $(git rev-parse --short HEAD) sizes ${SIZES[*]} epochs $EP"
for B in "${SIZES[@]}"; do
  ( s=$(date +%s)
    .venv/bin/baed-detect --dataset dgraph --node-feats 17 --gae-weight reddit_gin_relu_256_whole.pt --epochs $EP --seed 15 --batched \
      --data-tag "$TAG" --agg-features orig --readout deviation_cat_nb --model GCN --select val_auroc --lr 1e-3 --gpu-store \
      --batch-size $B --weight-suffix _bs$B --out $R/bs${B}_e${EP}.json > $L/bs${B}_e${EP}.log 2>&1 < /dev/null
    echo "[$(date +%T)] batch $B done rc=$? in $(( $(date +%s) - s )) s" ) &
done
wait
echo "[$(date +%T)] all done"
