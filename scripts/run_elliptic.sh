#!/usr/bin/env bash
# Full BAED pipeline on Elliptic (hyper-parameters from the authors README).
# Note: the README passes --parametrization xt_prescribed_st --loss_type vb_ce_xt_prescribred_st,
# which get_model() rejects for --arch TGNN_embedding_guided (those belong to the degree-guided
# EDGE branch). We use the consistent pair xt / vb_ce_xt.
#   bash scripts/run_elliptic.sh [epochs]
set -euo pipefail
cd "$(dirname "$0")/.."
EPOCHS="${1:-300}"   # paper, Appendix D (the upstream README used 2000)
DEVICE="${DEVICE:-cuda:0}"

# 01 ego-graph datasets (Rust kernel)
[ -s data/graphs/elliptic_init_1_dataset.pkl ] || uv run baed-build-subgraphs --dataset elliptic

# 02 GAE for condition embeddings (elliptic: 166 features -> 256)
[ -s data/gae/elliptic_gae.pt ] || uv run baed-gae-dataset --dataset elliptic
[ -s weights/gae/elliptic_gae_relu_256.pt ] || \
  uv run baed-gae-train --dataset elliptic --gae-type gae --activation relu --out-channels 256 --num-features 166

# 03 BAED diffusion model
uv run baed-train \
  --epochs "$EPOCHS" --num_generation 64 --diffusion_dim 64 --diffusion_steps 128 \
  --device "$DEVICE" --dataset elliptic --batch_size 8 --clip_value 1 --lr 1e-4 --optimizer adam \
  --final_prob_edge 1 0 --sample_time_method importance --check_every 10 --eval_every 10 \
  --noise_schedule linear --dp_rate 0.1 --loss_type vb_ce_xt \
  --arch TGNN_embedding_guided --parametrization xt --empty_graph_sampler empirical \
  --degree --num_heads 8 8 8 8 1 --log_wandb False
