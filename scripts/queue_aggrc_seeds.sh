#!/usr/bin/env bash
# GCN base and GCN +BAED (GIN-guided diffusion, no feature noise) with recomputed 1-hop aggregates
# (--agg-features recompute), config A, the paper's remaining seeds. All jobs run in parallel.
# Results: outputs/results/elliptic/configA_aggrc/GCN_{base,baed_noise0}_s<seed>.json
set -uo pipefail
cd "$(dirname "$0")/.."
export DGLBACKEND=pytorch OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SEEDS="${SEEDS:-42 63 87 94}"
RUN="${RUN:-elliptic_fulltrain_gin_et_b32}"; ENC="${ENC:-elliptic_gin_relu_256_whole.pt}"
R=outputs/results/elliptic/configA_aggrc; L=outputs/logs/elliptic/configA_aggrc; mkdir -p $R $L
COMMON=(--dataset elliptic --model GCN --node-feats 166 --gae-weight "$ENC" --epochs 50 --batched --lr 0.001
        --data-tag _full --agg-features recompute)

for s in $SEEDS; do
  out=$R/GCN_baed_noise0_s$s.json
  [ -s "$out" ] || nohup .venv/bin/baed-detect "${COMMON[@]}" --seed "$s" --out "$out" \
      --use-baed --with-condition --run-name "$RUN" --checkpoint 300 --gen-noise 0 --curriculum sigmoid \
      > $L/GCN_baed_noise0_s$s.log 2>&1 < /dev/null &
done
for s in $SEEDS; do   # base runs are short: one after another, in parallel with the BAED jobs
  out=$R/GCN_base_s$s.json
  [ -s "$out" ] || .venv/bin/baed-detect "${COMMON[@]}" --seed "$s" --out "$out" > $L/GCN_base_s$s.log 2>&1 < /dev/null
done
wait
echo "[$(date -u +%H:%M:%S)] done"
