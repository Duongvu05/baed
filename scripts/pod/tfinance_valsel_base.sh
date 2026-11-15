#!/usr/bin/env bash
# T-Finance: 1-hop ego-graphs of every node, ego-graphs > 1200 nodes dropped (released code), lean tensors only,
# then GCN base under the val protocol (epoch by val AUROC, threshold on val), same detector as Reddit _u / Photo.
# --gae-weight is loaded by baed-detect but not used by the base (only +BAED uses it); no T-Finance encoder exists yet.
set -o pipefail
cd /home/data/baed
export PATH=/home/data/.local/bin:$PATH DGLBACKEND=pytorch PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
L=outputs/logs/tfinance/valsel; R=outputs/results/tfinance/valsel; mkdir -p $L $R
echo "[$(date +%T)] start $(git rev-parse --short HEAD) $(git branch --show-current)"
.venv/bin/baed-build-subgraphs --dataset tfinance --label-type label_0_1 --lean-only > $L/build.log 2>&1 || { echo "build FAILED"; exit 1; }
echo "[$(date +%T)] build done"; tail -3 $L/build.log
.venv/bin/baed-detect --dataset tfinance --node-feats 10 --gae-weight reddit_gin_relu_256_whole.pt --epochs 50 --seed 15 --batched \
  --data-tag "" --agg-features orig --readout deviation_cat_nb --model GCN --select val_auroc --lr 1e-3 \
  --out $R/GCN_noselfloop_s15.json > $L/GCN_noselfloop_s15.log 2>&1 < /dev/null
echo "[$(date +%T)] tfinance base done rc=$?"; grep -a "threshold chosen" $L/GCN_noselfloop_s15.log; echo "all done"
