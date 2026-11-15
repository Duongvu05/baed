#!/usr/bin/env bash
# Check --gpu-store against the default loader: Photo full run (compare with photo/valsel s15), and the first
# epochs of T-Finance and DGraph (compare the per-epoch train loss and the time per epoch with the running logs).
cd /home/data/baed
export PATH=/home/data/.local/bin:$PATH DGLBACKEND=pytorch PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
L=outputs/logs/gs_check; R=outputs/results/gs_check; mkdir -p $L $R
echo "[$(date +%T)] start $(git rev-parse --short HEAD) $(git branch --show-current)"
C=(--gae-weight reddit_gin_relu_256_whole.pt --seed 15 --batched --data-tag "" --agg-features orig --readout deviation_cat_nb
   --model GCN --select val_auroc --lr 1e-3 --gpu-store)
run() { local name=$1; shift; s=$(date +%s); .venv/bin/baed-detect "${C[@]}" "$@" --out $R/$name.json > $L/$name.log 2>&1 < /dev/null
        echo "[$(date +%T)] $name done rc=$? in $(( $(date +%s) - s )) s"; }
run photo_s15 --dataset photo --node-feats 745 --epochs 50 &
run tfinance_e3 --dataset tfinance --node-feats 10 --epochs 3 &
run dgraph_e3 --dataset dgraph --node-feats 17 --epochs 3 &
wait
echo "[$(date +%T)] all done"
