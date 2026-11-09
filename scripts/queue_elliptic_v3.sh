#!/usr/bin/env bash
# Elliptic, paper-aligned setting (sequential, one job at a time, seed 15):
#  - full labelled set (46,564 ego-graphs / 4,545 anomalies), single-node ego-graphs kept
#  - diffusion trained only on TRAIN-split anomalies (no test leakage), 300 epochs
#  - downstream lr tuned on validation over {1e-3, 1e-4, 1e-5, 1e-6}
#  - +BAED with Gaussian feature noise and sigmoid curriculum weights (paper Sec. 4.4 / App. D)
set -u
cd "$(dirname "$0")/.."
export PATH=/home/data/.local/bin:$PATH DGLBACKEND=pytorch
RUN=elliptic_fulltrain
G="elliptic 166 elliptic_gae_relu_256.pt"
R=outputs/results/elliptic
log(){ echo "[$(date +%T)] $*"; }

log "train-split anomalies"
uv run python scripts/make_train_anomalies.py --src-tag _full --out-tag _fulltrain

if [ ! -s outputs/runs/elliptic/multinomial_diffusion/multistep/$RUN/check/checkpoint_299.pt ]; then
  log "diffusion training (300 epochs) on train-split anomalies"
  uv run baed-train --name $RUN --data_tag _fulltrain \
    --epochs 300 --num_generation 64 --diffusion_dim 64 --diffusion_steps 128 \
    --device cuda:0 --dataset elliptic --batch_size 8 --clip_value 1 --lr 1e-4 --optimizer adam \
    --final_prob_edge 1 0 --sample_time_method importance --check_every 10 --eval_every 10 \
    --noise_schedule linear --dp_rate 0.1 --loss_type vb_ce_xt \
    --arch TGNN_embedding_guided --parametrization xt --empty_graph_sampler empirical \
    --degree --num_heads 8 8 8 8 1 --log_wandb False > /home/data/baed_data/diffusion_fulltrain.log 2>&1
  log "diffusion done (rc=$?)"
fi

FULL="DATATAG=_full ISO=1 BATCHED=1"
for m in GCN BWGNN GraphSAGE BernNet; do for lr in 0.001 0.0001 0.00001 0.000001; do
  log "base $m lr=$lr"; env $FULL LR=$lr MODELS=$m bash scripts/run_detect_grid.sh $G base
done; done
uv run python - <<'PY'
import json, glob, re
best = {}
for f in glob.glob("outputs/results/elliptic/*_base_full_b_iso_lr*_s15.json"):
    m, lr = re.match(r".*/(\w+?)_base_full_b_iso_lr([0-9.e-]+)_s15.json", f).groups()
    v = json.load(open(f))["steps"][0].get("val_auroc", -1)
    if v > best.get(m, (None, -2))[1]:
        best[m] = (lr, v)
with open("outputs/results/elliptic/lr_choice_full.txt", "w") as fh:
    for m, (lr, v) in best.items():
        fh.write(f"{m} {lr} {v:.4f}\n")
print(best)
PY
lr_full(){ awk -v m="$1" '$1==m{print $2}' $R/lr_choice_full.txt; }

for m in GCN BWGNN GraphSAGE BernNet; do
  log "baed $m"; env $FULL PAUG=1 LR=$(lr_full $m) MODELS=$m bash scripts/run_detect_grid.sh $G baed $RUN 300
done
log "dyn base GCN"; env $FULL LR=$(lr_full GCN) MODELS=GCN bash scripts/run_detect_grid.sh $G dyn_base
log "dyn baed GCN"; env $FULL PAUG=1 LR=$(lr_full GCN) MODELS=GCN bash scripts/run_detect_grid.sh $G dyn_baed $RUN 300
log QUEUE_DONE
