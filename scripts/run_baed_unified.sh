#!/usr/bin/env bash
# Unified +BAED setting on Elliptic (see README, "Unified +BAED setting").
#
#   split      config A: ego-graphs in time order, sequential 70/15/15, single-node ego-graphs dropped
#   features   timestep + 93 local features, 1-hop aggregates recomputed on each ego-graph's own edges
#              (--agg-features recompute), for real and generated ego-graphs alike
#   generator  GIN-guided ego-graph diffusion (TGNN_gin_guided, run elliptic_fulltrain_gin_et_b32)
#   features of synthetic nodes   copied from the template ego-graph, NO Gaussian noise (--gen-noise 0)
#   curriculum paper Eq. 17-18 (--curriculum sigmoid)
#
# Usage: MODEL=GCN SEEDS="15" bash scripts/run_baed_unified.sh
#   runs base and +BAED for each seed. Steps already done (encoder, diffusion, result files) are skipped.
#   GEN_CHUNK=1 reproduces the per-batch sampling loop exactly; 16 (default) is ~5-8x faster, the guidance
#   of 16 consecutive batches then uses the detector at the start of the chunk.
set -uo pipefail
cd "$(dirname "$0")/.."
export DGLBACKEND=pytorch OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
MODEL="${MODEL:-GCN}"; SEEDS="${SEEDS:-15}"; GEN_CHUNK="${GEN_CHUNK:-16}"
ENC="${ENC:-elliptic_gin_relu_256_whole.pt}"; RUN="${RUN:-elliptic_fulltrain_gin_et_b32}"; CK="${CK:-300}"
declare -A LR=([GCN]=0.001 [BWGNN]=0.0001 [GraphSAGE]=0.001 [BernNet]=0.001)
R=outputs/results/elliptic/unified; L=outputs/logs/elliptic/unified; mkdir -p "$R" "$L"
log(){ echo "[$(date +%T)] $*"; }

if [ ! -s "weights/gae/$ENC" ]; then
  log "pre-train GIN guidance encoder (link prediction on the whole graph)"
  [ -s data/gae/elliptic_whole_gae.pt ] || uv run baed-gae-dataset --dataset elliptic --whole
  uv run baed-gae-train --dataset elliptic --data-tag _whole --epochs 1000 --gae-type gin --activation relu \
    --out-channels 256 --num-features 166 > "$L/gin_pretrain.log" 2>&1 || { log "FAILED encoder"; exit 1; }
fi

if [ ! -s "outputs/runs/elliptic/multinomial_diffusion/multistep/$RUN/check/checkpoint_$((CK - 1)).pt" ]; then
  log "train diffusion $RUN (train-split anomalies)"
  uv run baed-train --name "$RUN" --data_tag _fulltrain \
    --epochs "$CK" --num_generation 64 --diffusion_dim 64 --diffusion_steps 128 \
    --device cuda:0 --dataset elliptic --batch_size 32 --clip_value 1 --lr 2e-4 --optimizer adam \
    --final_prob_edge 1 0 --sample_time_method uniform --check_every 10 --eval_every 10 \
    --noise_schedule linear --dp_rate 0.1 --loss_type vb_kl \
    --arch TGNN_gin_guided --parametrization x0 --empty_graph_sampler empirical \
    --num_heads 8 8 8 8 1 --log_wandb False --guidance_encoder "$ENC" > "$L/diffusion_$RUN.log" 2>&1 \
    || { log "FAILED diffusion"; exit 1; }
fi

COMMON=(--dataset elliptic --model "$MODEL" --node-feats 166 --gae-weight "$ENC" --epochs 50 --batched
        --lr "${LR[$MODEL]}" --data-tag _full --split sequential --agg-features recompute)
for s in $SEEDS; do
  out="$R/${MODEL}_base_s$s.json"
  if [ ! -s "$out" ]; then
    log "$MODEL base seed $s"
    uv run baed-detect "${COMMON[@]}" --seed "$s" --out "$out" > "$L/${MODEL}_base_s$s.log" 2>&1 || log "FAILED base $s"
  fi
  out="$R/${MODEL}_baed_s$s.json"
  if [ ! -s "$out" ]; then
    log "$MODEL +BAED seed $s (gen-chunk $GEN_CHUNK)"
    uv run baed-detect "${COMMON[@]}" --seed "$s" --out "$out" \
      --use-baed --with-condition --run-name "$RUN" --checkpoint "$CK" \
      --gen-noise 0 --curriculum sigmoid --gen-chunk "$GEN_CHUNK" \
      > "$L/${MODEL}_baed_s$s.log" 2>&1 || log "FAILED +BAED $s"
  fi
done
log done
