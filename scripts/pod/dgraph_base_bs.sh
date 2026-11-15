#!/usr/bin/env bash
# DGraph-Fin GCN base under the val protocol with a larger batch (--gpu-store), id order and first-edge-day order (_t),
# in parallel. Ego-graphs as in dgraph_valsel_base.sh / dgraph_t_valsel_base.sh (lean caches must exist).
# Usage: [BS=1024] [SEED=15] dgraph_base_bs.sh
BS=${BS:-1024}; S=${SEED:-15}
cd /home/data/baed
export PATH=/home/data/.local/bin:$PATH DGLBACKEND=pytorch PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "[$(date +%T)] start $(git rev-parse --short HEAD) $(git branch --show-current) batch $BS seed $S"
for T in "" _t; do
  ( L=outputs/logs/dgraph$T/valsel; R=outputs/results/dgraph$T/valsel; mkdir -p $L $R; s=$(date +%s)
    .venv/bin/baed-detect --dataset dgraph --node-feats 17 --gae-weight reddit_gin_relu_256_whole.pt --epochs 50 --seed $S --batched \
      --data-tag "$T" --agg-features orig --readout deviation_cat_nb --model GCN --select val_auroc --lr 1e-3 --gpu-store \
      --batch-size $BS --weight-suffix _bs$BS --out $R/GCN_noselfloop_bs${BS}_s$S.json > $L/GCN_noselfloop_bs${BS}_s$S.log 2>&1 < /dev/null
    echo "[$(date +%T)] dgraph$T done rc=$? in $(( $(date +%s) - s )) s"; grep -a "threshold chosen" $L/GCN_noselfloop_bs${BS}_s$S.log ) &
done
wait
echo "[$(date +%T)] all done"
