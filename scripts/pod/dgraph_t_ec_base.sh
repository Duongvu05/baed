#!/usr/bin/env bash
# DGraph-Fin in time order with no future edges (tag _t_ec, build_subgraphs --edge-cutoff): same train / val / test
# membership as _t, but each segment's ego-graphs use only the edges up to the time of the segment's last center.
# GCN base under the val protocol, --gpu-store, batch 1024 (as dgraph_base_bs.sh). Usage: [BS=1024] [SEED=15] dgraph_t_ec_base.sh
BS=${BS:-1024}; S=${SEED:-15}
set -o pipefail
cd /home/data/baed
export PATH=/home/data/.local/bin:$PATH DGLBACKEND=pytorch PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
L=outputs/logs/dgraph_t_ec/valsel; R=outputs/results/dgraph_t_ec/valsel; mkdir -p $L $R
echo "[$(date +%T)] start $(git rev-parse --short HEAD) $(git branch --show-current) batch $BS seed $S"
prlimit --as=214748364800 .venv/bin/baed-build-subgraphs --dataset dgraph --label-type label_0_1 --lean-only --node-time --edge-cutoff \
  > $L/build.log 2>&1 || { echo "build FAILED"; exit 1; }
echo "[$(date +%T)] build done"; grep -a "segment\|saved" $L/build.log | sed "s/^.*| //"
.venv/bin/baed-detect --dataset dgraph --node-feats 17 --gae-weight reddit_gin_relu_256_whole.pt --epochs 50 --seed $S --batched \
  --data-tag _t_ec --agg-features orig --readout deviation_cat_nb --model GCN --select val_auroc --lr 1e-3 --gpu-store \
  --batch-size $BS --weight-suffix _bs$BS --out $R/GCN_noselfloop_bs${BS}_s$S.json > $L/GCN_noselfloop_bs${BS}_s$S.log 2>&1 < /dev/null
echo "[$(date +%T)] dgraph_t_ec done rc=$?"; grep -a "样本数\|threshold chosen" $L/GCN_noselfloop_bs${BS}_s$S.log; echo "all done"
