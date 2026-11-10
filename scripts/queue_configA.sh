#!/usr/bin/env bash
# Table 1 and Table 2 in "config A" = the released setting:
# out-edge ego-graphs (_full, all 49 timesteps), single-node ego-graphs dropped, batched updates,
# seed 15, one lr per model (val-selected on the out-edge set). +BAED uses the leakage-free
# diffusion run (train-split anomalies) with the paper augmentation (N(0,1) noise, sigmoid curriculum).
# +GGAD / +AEGIS: baselines adapted to ego-graphs (baed.aug_generators), same batch balancing.
# Results: outputs/results/elliptic/configA/<model>_<mode>_s15.json
set -uo pipefail
cd "$(dirname "$0")/.."
export DGLBACKEND=pytorch OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
RUN="${RUN:-elliptic_fulltrain}"; CK="${CK:-300}"; SEED="${SEED:-15}"
R=outputs/results/elliptic/configA; L=outputs/logs/elliptic/configA; mkdir -p $R $L
declare -A LR=([GCN]=0.001 [BWGNN]=0.0001 [GraphSAGE]=0.001 [BernNet]=0.001)

while pgrep -f "bin/baed-detect" >/dev/null; do sleep 30; done   # one GPU job at a time

run() { # model mode
  local m=$1 mode=$2 out=$R/${1}_${2}_s${SEED}.json extra=()
  [ -s "$out" ] && return
  case $mode in
    *baed) extra+=(--use-baed --with-condition --run-name "$RUN" --checkpoint "$CK" --gen-noise 1.0 --curriculum sigmoid);;
    *ggad) extra+=(--use-baed --aug-gen ggad);;
    *aegis) extra+=(--use-baed --aug-gen aegis);;
  esac
  case $mode in dyn_*) extra+=(--dynamic-segments 4);; esac
  echo "[$(date -u +%H:%M:%S)] $m $mode lr=${LR[$m]}"
  uv run baed-detect --dataset elliptic --model "$m" --node-feats 166 --gae-weight elliptic_gae_relu_256.pt \
    --epochs 50 --seed "$SEED" --batched --lr "${LR[$m]}" --data-tag _full --out "$out" "${extra[@]}" \
    > $L/${m}_${mode}_s${SEED}.log 2>&1 < /dev/null || echo "  FAILED (see $L/${m}_${mode}_s${SEED}.log)"
}

run GCN dyn_base                                              # Table 2, base (fast)
for m in GCN BWGNN GraphSAGE BernNet; do run $m baed; done    # Table 1, +BAED
for g in ggad aegis; do for m in GCN BWGNN GraphSAGE BernNet; do run $m $g; done; done   # Table 1, +GGAD / +AEGIS
run GCN dyn_ggad; run GCN dyn_aegis                           # Table 2, +GGAD / +AEGIS
run GCN dyn_baed                                              # Table 2, +BAED
echo "[$(date -u +%H:%M:%S)] done"
