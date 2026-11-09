#!/usr/bin/env bash
# Launch the downstream grid for one dataset in parallel on one GPU.
#   bash scripts/run_detect_grid.sh <dataset> <node_feats> <gae_weight> <mode> [run_name] [checkpoint]
#   mode: base | baed | dyn_base | dyn_baed
# Results: outputs/results/<dataset>/<model>_<mode>_s<seed>.json
set -euo pipefail
cd "$(dirname "$0")/.."
DS=$1; NF=$2; GAE=$3; MODE=$4; RUN=${5:-}; CK=${6:-300}
SEEDS="${SEEDS:-15}"   # paper averages 15 42 63 87 94, we run one seed
MODELS="${MODELS:-GCN BWGNN GraphSAGE BernNet}"
EPOCHS="${EPOCHS:-50}"
mkdir -p outputs/results/$DS outputs/logs/$DS
for m in $MODELS; do for s in $SEEDS; do
  extra=()
  case $MODE in
    baed|dyn_baed) extra+=(--use-baed --with-condition --run-name "$RUN" --checkpoint "$CK");;
  esac
  case $MODE in dyn_*) extra+=(--dynamic-segments 4);; esac
  [ -n "${BATCHED:-}" ] && extra+=(--batched)
  [ -n "${LR:-}" ] && extra+=(--lr "$LR")
  [ -n "${RANDSPLIT:-}" ] && extra+=(--split random)
  [ -n "${ISO:-}" ] && extra+=(--keep-isolated)
  out=outputs/results/$DS/${m}_${MODE}${BATCHED:+_b}${RANDSPLIT:+_rand}${ISO:+_iso}${LR:+_lr$LR}_s${s}.json
  [ -s "$out" ] && continue
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 DGLBACKEND=pytorch nohup uv run baed-detect --dataset "$DS" --model "$m" \
    --node-feats "$NF" --gae-weight "$GAE" --epochs "$EPOCHS" --seed "$s" --out "$out" "${extra[@]}" \
    > outputs/logs/$DS/${m}_${MODE}${BATCHED:+_b}${RANDSPLIT:+_rand}${ISO:+_iso}${LR:+_lr$LR}_s${s}.log 2>&1 < /dev/null &
done; done
wait
