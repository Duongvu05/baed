#!/usr/bin/env bash
# Sequential Elliptic queue (batched + Rust fast path, seed 15). One job at a time.
set -u
cd "$(dirname "$0")/.."
export PATH=/home/data/.local/bin:$PATH
RUN=${RUN:-2026-11-09_09-19-22}
G="elliptic 166 elliptic_gae_relu_256.pt"
R=outputs/results/elliptic
log(){ echo "[$(date +%T)] $*"; }
grid(){ log "start $*"; bash scripts/run_detect_grid.sh "$@"; log "done $*"; }

# 1) baselines, lr in {1e-3, 1e-4}; choose lr per model on validation AUROC
for m in GCN BWGNN GraphSAGE BernNet; do for lr in 0.001 0.0001; do
  BATCHED=1 LR=$lr MODELS=$m grid $G base
done; done
uv run python - <<'PY'
import json, glob, re
best = {}
for f in glob.glob("outputs/results/elliptic/*_base_b_lr*_s15.json"):
    m, lr = re.match(r".*/(\w+)_base_b_lr([0-9.]+)_s15.json", f).groups()
    v = json.load(open(f))["steps"][0].get("val_auroc", -1)
    if v > best.get(m, (None, -2))[1]:
        best[m] = (lr, v)
with open("outputs/results/elliptic/lr_choice.txt", "w") as fh:
    for m, (lr, v) in best.items():
        fh.write(f"{m} {lr} {v:.4f}\n")
print(best)
PY
lr_of(){ awk -v m="$1" '$1==m{print $2}' $R/lr_choice.txt; }

# 2) +BAED with the chosen lr
for m in GCN BWGNN GraphSAGE BernNet; do
  BATCHED=1 LR=$(lr_of $m) MODELS=$m grid $G baed "$RUN" 300
done

# 3) Table 2: incremental setting, GCN
BATCHED=1 LR=$(lr_of GCN) MODELS=GCN grid $G dyn_base
BATCHED=1 LR=$(lr_of GCN) MODELS=GCN grid $G dyn_baed "$RUN" 300

# 4) check variant: stratified random split + keep single-node ego-graphs
for m in GCN BWGNN GraphSAGE BernNet; do
  BATCHED=1 RANDSPLIT=1 ISO=1 LR=$(lr_of $m) MODELS=$m grid $G base
done
for m in GCN BWGNN GraphSAGE BernNet; do
  BATCHED=1 RANDSPLIT=1 ISO=1 LR=$(lr_of $m) MODELS=$m grid $G baed "$RUN" 300
done
log QUEUE_DONE
