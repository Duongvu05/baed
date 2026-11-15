#!/usr/bin/env bash
# DGraph-Fin in time order (tag _t): same ego-graphs as dgraph_valsel_base.sh (centers labelled 0/1, 2 hops along
# out-edges, background nodes as neighbours), but sorted by the day of each center's first edge
# (scripts/make_node_time.py dgraph), so the sequential 70/15/15 split is a temporal split, like Elliptic.
set -o pipefail
cd /home/data/baed
export PATH=/home/data/.local/bin:$PATH DGLBACKEND=pytorch PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
L=outputs/logs/dgraph_t/valsel; R=outputs/results/dgraph_t/valsel; mkdir -p $L $R
echo "[$(date +%T)] start $(git rev-parse --short HEAD) $(git branch --show-current)"
.venv/bin/python scripts/make_node_time.py dgraph > $L/node_time.log 2>&1 || { echo "node time FAILED"; exit 1; }
cat $L/node_time.log
.venv/bin/baed-build-subgraphs --dataset dgraph --label-type label_0_1 --lean-only --node-time > $L/build.log 2>&1 || { echo "build FAILED"; exit 1; }
echo "[$(date +%T)] build done"; grep -a "saved" $L/build.log
.venv/bin/baed-detect --dataset dgraph --node-feats 17 --gae-weight reddit_gin_relu_256_whole.pt --epochs 50 --seed 15 --batched \
  --data-tag _t --agg-features orig --readout deviation_cat_nb --model GCN --select val_auroc --lr 1e-3 \
  --out $R/GCN_noselfloop_s15.json > $L/GCN_noselfloop_s15.log 2>&1 < /dev/null
echo "[$(date +%T)] dgraph _t base done rc=$?"; grep -a "threshold chosen" $L/GCN_noselfloop_s15.log; echo "all done"
