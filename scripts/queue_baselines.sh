#!/usr/bin/env bash
# Sequential baseline queue (one GPU job at a time) for Elliptic.
#   1. lr grid (seed 15) on the bidirectional ego-graphs (_full_bi)
#   2. 5 seeds at the val-selected lr, for _full_bi and _full (out-edges, as released)
# Results: outputs/results/elliptic/base/<model>_<tag>_lr<lr>_s<seed>.json
set -uo pipefail
cd "$(dirname "$0")/.."
export DGLBACKEND=pytorch OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
MODELS="${MODELS:-GCN BWGNN GraphSAGE BernNet}"
LRS="${LRS:-0.001 0.0001 0.00001 0.000001}"
SEEDS="${SEEDS:-15 42 63 87 94}"
TAGS="${TAGS:-_full_bi _full}"   # _full_bi = ego-graphs over both edge directions
R=outputs/results/elliptic/base; L=outputs/logs/elliptic/base; mkdir -p $R $L

# wait for any running detector job (keeps the GPU to one job)
while pgrep -f "bin/baed-detect" >/dev/null; do sleep 30; done

run() { # model tag lr seed
  local out=$R/${1}${2}_lr${3}_s${4}.json
  [ -s "$out" ] && return
  echo "[$(date -u +%H:%M:%S)] $1 tag=$2 lr=$3 seed=$4"
  uv run baed-detect --dataset elliptic --model "$1" --node-feats 166 --gae-weight elliptic_gae_relu_256.pt \
    --epochs 50 --seed "$4" --batched --lr "$3" --keep-isolated --data-tag "$2" --out "$out" \
    > $L/${1}${2}_lr${3}_s${4}.log 2>&1 < /dev/null || echo "  FAILED (see $L/${1}${2}_lr${3}_s${4}.log)"
}

best_lr() { # model tag -> lr with highest val AUROC among seed-15 runs
  python3 - "$R" "$1" "$2" <<'PY'
import json, glob, sys, re
R, m, tag = sys.argv[1:]
best = max((json.load(open(f))["steps"][-1].get("val_auroc", 0), re.search(r"_lr([0-9.]+)_s15", f).group(1))
           for f in glob.glob(f"{R}/{m}{tag}_lr*_s15.json"))
print(best[1])
PY
}

for tag in $TAGS; do [ "$tag" = _full ] && continue; for m in $MODELS; do for lr in $LRS; do run $m $tag $lr 15; done; done; done
# reuse the seed-15 lr grid already run for _full (out-edges)
for f in outputs/results/elliptic/*_base_full_b_iso_lr*_s15.json; do
  b=$(basename "$f"); m=${b%%_*}; lr=$(echo "$b" | sed -E 's/.*_lr([0-9.]+)_s15.json/\1/')
  [ -s "$R/${m}_full_lr${lr}_s15.json" ] || cp "$f" "$R/${m}_full_lr${lr}_s15.json"
done
for tag in $TAGS; do for m in $MODELS; do
  lr=$(best_lr $m $tag); echo "[$(date -u +%H:%M:%S)] $m $tag selected lr=$lr" | tee -a $R/lr_choice.txt
  for s in $SEEDS; do run $m $tag $lr $s; done
done; done
echo "[$(date -u +%H:%M:%S)] done"
