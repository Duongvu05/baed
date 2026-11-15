#!/usr/bin/env bash
# T-Finance WITHOUT the released 1200-node cap (--no-cap, tag _nocap): all 39,357 ego-graphs, 1 hop, in id order
# (_nocap) and in registration-day order (_t_nocap). GCN base under the val protocol, --gpu-store, one seed.
# Usage: [SEED=15] tfinance_nocap_base.sh
S=${SEED:-15}
cd /home/data/baed
export PATH=/home/data/.local/bin:$PATH DGLBACKEND=pytorch PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "[$(date +%T)] start $(git rev-parse --short HEAD) $(git branch --show-current) seed $S"
for T in _nocap _t_nocap; do
  ( L=outputs/logs/tfinance$T/valsel; R=outputs/results/tfinance$T/valsel; mkdir -p $L $R
    F=$([ $T = _t_nocap ] && echo --node-time)
    .venv/bin/baed-build-subgraphs --dataset tfinance --label-type label_0_1 --lean-only --no-cap $F > $L/build.log 2>&1 \
      || { echo "[$(date +%T)] build $T FAILED"; exit 1; }
    echo "[$(date +%T)] build $T done: $(grep -o "saved.*" $L/build.log)"
    .venv/bin/baed-detect --dataset tfinance --node-feats 10 --gae-weight reddit_gin_relu_256_whole.pt --epochs 50 --seed $S --batched \
      --data-tag $T --agg-features orig --readout deviation_cat_nb --model GCN --select val_auroc --lr 1e-3 --gpu-store \
      --out $R/GCN_noselfloop_s$S.json > $L/GCN_noselfloop_s$S.log 2>&1 < /dev/null
    echo "[$(date +%T)] $T seed $S done rc=$?" ) &
done
wait
echo "[$(date +%T)] all done"
