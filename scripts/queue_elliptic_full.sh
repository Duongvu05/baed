#!/usr/bin/env bash
# Sequential Elliptic queue, main setting = full labelled set (46,564 ego-graphs, 4,545
# anomalies as in the paper's Table 4) + single-node ego-graphs kept. seed 15, batched path.
set -u
cd "$(dirname "$0")/.."
export PATH=/home/data/.local/bin:$PATH
RUN=${RUN:-2026-11-09_09-19-22}
G="elliptic 166 elliptic_gae_relu_256.pt"
R=outputs/results/elliptic
log(){ echo "[$(date +%T)] $*"; }
grid(){ log "start $*"; bash scripts/run_detect_grid.sh "$@"; log "done $*"; }
wait_free(){ while ps -eo args | grep "bin/baed-detect" | grep -v grep > /dev/null; do sleep 60; done; }

log "waiting for the running detect job"; wait_free

FULL="DATATAG=_full ISO=1 BATCHED=1"
for m in GCN BWGNN GraphSAGE BernNet; do for lr in 0.001 0.0001; do
  env $FULL LR=$lr MODELS=$m bash -c "$(declare -f log grid); grid $G base"
done; done
uv run python - <<'PY'
import json, glob, re
best = {}
for f in glob.glob("outputs/results/elliptic/*_base_full_b_iso_lr*_s15.json"):
    m, lr = re.match(r".*/(\w+?)_base_full_b_iso_lr([0-9.]+)_s15.json", f).groups()
    v = json.load(open(f))["steps"][0].get("val_auroc", -1)
    if v > best.get(m, (None, -2))[1]:
        best[m] = (lr, v)
with open("outputs/results/elliptic/lr_choice_full.txt", "w") as fh:
    for m, (lr, v) in best.items():
        fh.write(f"{m} {lr} {v:.4f}\n")
print(best)
PY
lr_full(){ awk -v m="$1" '$1==m{print $2}' $R/lr_choice_full.txt; }
lr_orig(){ awk -v m="$1" '$1==m{print $2}' $R/lr_choice.txt; }

for m in GCN BWGNN GraphSAGE BernNet; do
  env $FULL LR=$(lr_full $m) MODELS=$m bash -c "$(declare -f log grid); grid $G baed $RUN 300"
done
env $FULL LR=$(lr_full GCN) MODELS=GCN bash -c "$(declare -f log grid); grid $G dyn_base"
env $FULL LR=$(lr_full GCN) MODELS=GCN bash -c "$(declare -f log grid); grid $G dyn_baed $RUN 300"

# remaining runs in the authors'-code setting (GCN +BAED already done)
for m in BWGNN GraphSAGE BernNet; do
  BATCHED=1 LR=$(lr_orig $m) MODELS=$m bash -c "$(declare -f log grid); grid $G baed $RUN 300"
done
BATCHED=1 LR=$(lr_orig GCN) MODELS=GCN bash -c "$(declare -f log grid); grid $G dyn_base"
BATCHED=1 LR=$(lr_orig GCN) MODELS=GCN bash -c "$(declare -f log grid); grid $G dyn_baed $RUN 300"
log QUEUE_DONE
