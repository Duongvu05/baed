#!/usr/bin/env bash
# Reddit users-only (_u): epoch chosen by validation AUROC and threshold chosen on validation (branch exp/val-select).
# GCN base and GCN +BAED (center-flag diffusion reddit_u_train_gin_et_b32_lr1e4_cflag) at lr 1e-3 and 1e-6.
cd /home/data/baed
export PATH=/home/data/.local/bin:$PATH DGLBACKEND=pytorch PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
L=outputs/logs/reddit_u/valsel; R=outputs/results/reddit_u/valsel; mkdir -p $L $R
log() { echo "[$(date +%T)] $*"; }
log "start $(git rev-parse --short HEAD) $(git branch --show-current)"
C=(--dataset reddit --node-feats 64 --gae-weight reddit_gin_relu_256_whole.pt --epochs 50 --seed 15 --batched --data-tag _u
   --agg-features orig --readout deviation_cat_nb --model GCN --select val_auroc)
B=(--use-baed --with-condition --run-name reddit_u_train_gin_et_b32_lr1e4_cflag --checkpoint 300 --gen-noise 0 --curriculum sigmoid
   --gen-chunk 1 --gen-center-fill --gen-ego-hops 1 --fast-sampling --compile-denoiser --weight-suffix _cflag)
run() { local name=$1; shift; .venv/bin/baed-detect "${C[@]}" "$@" --out $R/${name}_s15.json > $L/${name}_s15.log 2>&1 < /dev/null; log "$name done rc=$?"; }
run GCN_noselfloop --lr 1e-3
run GCN_noselfloop_baed_cflag_detlr1e-3 --lr 1e-3 "${B[@]}" &
run GCN_noselfloop_baed_cflag_detlr1e-6 --lr 1e-6 "${B[@]}" &
wait
log "all done"
