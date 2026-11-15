#!/usr/bin/env bash
# T-Finance in time order (tag _t): same ego-graphs as tfinance_valsel_base.sh (every node, 1 hop, > 1200 nodes dropped)
# sorted by registration day (-feature column 0, scripts/make_node_time.py tfinance), so the sequential 70/15/15 split is
# a temporal split. GCN base under the val protocol, seeds 15 42 63 87 94 in parallel, --gpu-store.
set -o pipefail
cd /home/data/baed
export PATH=/home/data/.local/bin:$PATH DGLBACKEND=pytorch PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
L=outputs/logs/tfinance_t/valsel; R=outputs/results/tfinance_t/valsel; mkdir -p $L $R
echo "[$(date +%T)] start $(git rev-parse --short HEAD) $(git branch --show-current)"
.venv/bin/python scripts/make_node_time.py tfinance > $L/node_time.log 2>&1 || { echo "node time FAILED"; exit 1; }
cat $L/node_time.log
.venv/bin/baed-build-subgraphs --dataset tfinance --label-type label_0_1 --lean-only --node-time > $L/build.log 2>&1 || { echo "build FAILED"; exit 1; }
echo "[$(date +%T)] build done"; grep -a "saved" $L/build.log
for S in 15 42 63 87 94; do
  ( .venv/bin/baed-detect --dataset tfinance --node-feats 10 --gae-weight reddit_gin_relu_256_whole.pt --epochs 50 --seed $S --batched \
      --data-tag _t --agg-features orig --readout deviation_cat_nb --model GCN --select val_auroc --lr 1e-3 --gpu-store \
      --out $R/GCN_noselfloop_s$S.json > $L/GCN_noselfloop_s$S.log 2>&1 < /dev/null
    echo "[$(date +%T)] seed $S done rc=$?" ) &
done
wait
echo "[$(date +%T)] all done"
