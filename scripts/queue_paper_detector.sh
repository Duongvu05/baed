#!/usr/bin/env bash
# Base detectors with the paper's detection model (Sec. 4.1, Eq. 4-8), config A, 166 released features.
#   GCN_GAE      GCN pre-trained with GAE on the whole graph (weights/gae/elliptic_gae_relu_256_whole.pt),
#                fine-tuned, or frozen (--freeze-gnn)
#   GCN          from scratch, with self-loops (A + I)
#   BWGNN, GraphSAGE, BernNet   from scratch (no GAE checkpoint for them yet)
# All use the deviation readout: layers concatenated (Eq. 5), center minus the mean of its neighbours
# (Eq. 6), 2-layer MLP (Eq. 8). Node 0 of every ego-graph is its center (rebuild the pkls with the
# current baed.graph_ops first). All jobs run in parallel.
# Results: outputs/results/elliptic/paper_detector/<name>_s<seed>.json
set -uo pipefail
cd "$(dirname "$0")/.."
export DGLBACKEND=pytorch OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SEED="${SEED:-15}"; RO="${RO:-deviation_cat_nb}"
R=outputs/results/elliptic/paper_detector; L=outputs/logs/elliptic/paper_detector; mkdir -p "$R" "$L"
C=(--dataset elliptic --node-feats 166 --gae-weight elliptic_gin_relu_256_whole.pt --epochs 50 --seed "$SEED"
   --batched --data-tag _full --agg-features orig --readout "$RO")
run() { # name args...
  local name=$1; shift
  local out="$R/${name}_s${SEED}.json"
  [ -s "$out" ] && return
  .venv/bin/baed-detect "${C[@]}" "$@" --out "$out" > "$L/${name}_s${SEED}.log" 2>&1 < /dev/null &
}
run GCN_GAE_ft      --model GCN_GAE --lr 0.001
run GCN_GAE_frozen  --model GCN_GAE --lr 0.001 --freeze-gnn
run GCN_selfloop    --model GCN --lr 0.001 --gcn-self-loop
run BWGNN           --model BWGNN --lr 0.0001
run GraphSAGE       --model GraphSAGE --lr 0.001
run BernNet         --model BernNet --lr 0.001
wait
echo "[$(date -u +%H:%M:%S)] done"
