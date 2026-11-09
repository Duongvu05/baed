"""Benchmark BAED sampling: eager vs torch.compile of the denoiser.

    uv run python scripts/bench_sampling.py --run-name <run> [--dataset elliptic] [--checkpoint 300]

1. equivalence: denoiser output on the same noisy batch / timestep, eager vs compiled
2. speed: wall time of model.sample(n, embedding) for both
"""

import argparse
import copy
import time

import torch

from baed.pipelines.generate import GenerateDataArgs, GraphDataGenerator


def timed(fn, reps):
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--checkpoint", type=int, default=300)
    ap.add_argument("--n", type=int, default=110)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mode", default="default", help="torch.compile mode")
    ap.add_argument("--dynamic", default="true", choices=["true", "false", "none"])
    a = ap.parse_args()

    gen = GraphDataGenerator(GenerateDataArgs(run_name=a.run_name, dataset=a.dataset, seed=0, checkpoint=a.checkpoint))
    model = gen.model
    emb = torch.randn(model._denoise_fn.adapter.output_size if hasattr(model._denoise_fn.adapter, "output_size") else 256,
                      device=model.device)

    # --- equivalence on one denoising step -----------------------------------------
    torch.manual_seed(0)
    g = model._prepare_data_for_sampling(model.initial_graph_sampler.sample(a.n).to(model.device))
    t = model.num_timesteps // 2
    nn_, ne_ = int(g.nodes_per_graph.sum()), int(g.edges_per_graph.sum())
    tn = torch.full((nn_,), t, device=model.device, dtype=torch.long)
    te = torch.full((ne_,), t, device=model.device, dtype=torch.long)
    eager_fn = model._denoise_fn
    with torch.no_grad():
        ref = eager_fn(copy.copy(g), tn, te, emb)
    dyn = {"true": True, "false": False, "none": None}[a.dynamic]
    compiled = torch.compile(eager_fn, dynamic=dyn, mode=a.mode)
    t0 = time.time()
    with torch.no_grad():
        out = compiled(copy.copy(g), tn, te, emb)
    print(f"compile (first call): {time.time() - t0:.1f}s")
    for r, o in zip(ref, out):
        if r is None:
            continue
        print(f"max |eager - compiled| = {(r - o).abs().max().item():.2e}  (scale {r.abs().max().item():.2e})")

    # --- speed ------------------------------------------------------------------------
    torch.manual_seed(1)
    model._denoise_fn = eager_fn
    te_eager = timed(lambda: model.sample(a.n, emb), a.reps)
    model._denoise_fn = compiled
    model.sample(a.n, emb)  # warm up / recompiles for new shapes
    te_comp = timed(lambda: model.sample(a.n, emb), a.reps)
    print(f"sample({a.n}): eager {te_eager:.3f}s | compiled {te_comp:.3f}s | x{te_eager / te_comp:.2f}")


if __name__ == "__main__":
    main()
