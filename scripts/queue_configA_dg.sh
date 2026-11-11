#!/usr/bin/env bash
# Config A, +BAED with the degree-guided + anomaly-guided diffusion (arch TGNN_degree_embedding_guided),
# guidance embeddings from a GIN encoder pre-trained by link prediction (paper Sec. 4.3).
#   1. train the diffusion on train-split anomalies (300 epochs, paper App. D), run elliptic_fulltrain_dg
#   2. +BAED for GCN, BWGNN, GraphSAGE, BernNet (seed 15, one lr per model, as in queue_configA.sh)
set -u
cd "$(dirname "$0")/.."
export PATH=/home/data/.local/bin:$PATH DGLBACKEND=pytorch
ENC="${ENC:-elliptic_gin_relu_256.pt}"  # guidance encoder: GIN pre-trained by link prediction
RUN="${RUN:-elliptic_fulltrain_dg_gin}"; CK="${CK:-300}"; SEED="${SEED:-15}"
R=outputs/results/elliptic/configA_dg; L=outputs/logs/elliptic/configA_dg
mkdir -p "$R" "$L"
declare -A LR=([GCN]=0.001 [BWGNN]=0.0001 [GraphSAGE]=0.001 [BernNet]=0.001)
log(){ echo "[$(date +%T)] $*"; }
while pgrep -f "bin/baed-detect" >/dev/null; do sleep 30; done

if [ ! -s "weights/gae/$ENC" ]; then
  log "pre-train GIN guidance encoder (link prediction)"
  [ -s data/gae/elliptic_gae.pt ] || uv run baed-gae-dataset --dataset elliptic
  uv run baed-gae-train --dataset elliptic --gae-type gin --activation relu --out-channels 256 --num-features 166 \
    > "$L/gin_pretrain.log" 2>&1 || { log "FAILED gin"; exit 1; }
  grep -E "Best|saved" "$L/gin_pretrain.log"
fi

if [ ! -s outputs/runs/elliptic/multinomial_diffusion/multistep/$RUN/check/checkpoint_$((CK - 1)).pt ]; then
  log "diffusion training $RUN"
  uv run baed-train --name "$RUN" --data_tag _fulltrain \
    --epochs "$CK" --num_generation 64 --diffusion_dim 64 --diffusion_steps 128 \
    --device cuda:0 --dataset elliptic --batch_size 8 --clip_value 1 --lr 1e-4 --optimizer adam \
    --final_prob_edge 1 0 --sample_time_method importance --check_every 10 --eval_every 10 \
    --noise_schedule linear --dp_rate 0.1 --loss_type vb_ce_xt_prescribred_st \
    --arch TGNN_degree_embedding_guided --parametrization xt_prescribed_st --empty_graph_sampler empirical \
    --degree --num_heads 8 8 8 8 1 --log_wandb False --guidance_encoder "$ENC" > "$L/diffusion_$RUN.log" 2>&1 || { log "FAILED diffusion"; exit 1; }
  log "diffusion done"
fi

for m in GCN BWGNN GraphSAGE BernNet; do
  out="$R/${m}_baed_s${SEED}.json"
  [ -s "$out" ] && continue
  log "$m baed lr=${LR[$m]}"
  uv run baed-detect --dataset elliptic --model "$m" --node-feats 166 --gae-weight "$ENC" \
    --epochs 50 --seed "$SEED" --batched --lr "${LR[$m]}" --data-tag _full --out "$out" \
    --use-baed --with-condition --run-name "$RUN" --checkpoint "$CK" --gen-noise 1.0 --curriculum sigmoid \
    > "$L/${m}_baed_s${SEED}.log" 2>&1 || log "FAILED $m"
done
log done
