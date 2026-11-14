"""Equivalence checks of the fast BAED sampling path for a trained diffusion run, then generation statistics.

  python scripts/test_fast_sampling.py <run_name> <dataset> <ego hops>

1. native segment ops == torch_scatter, 2-3. static (masked) denoiser forward and its torch.compile version ==
the original forward, 4. CUDA-graph sampling loop == eager loop with the Gumbel noise off, 5-6. statistics and
speed of center-fill generation for eager / CUDA graph / CUDA graph + compile. Prints ALL OK when every check passes.
"""
import random, sys, time
import numpy as np
import torch
import torch_scatter
from baed.diffusion import diffusion_base as DB
from baed.pipelines.generate import GenerateDataArgs, GraphDataGenerator

RUN = sys.argv[1]; DS = sys.argv[2]; HOPS = int(sys.argv[3])
ok = True
def check(name, cond):
    global ok
    ok &= bool(cond); print(f"  [{'OK' if cond else 'FAIL'}] {name}")

g = GraphDataGenerator(GenerateDataArgs(run_name=RUN, dataset=DS, checkpoint=300,
                                        device="cuda:0"))
M, den = g.model, g.model._denoise_fn
pool = [ft for v in M.initial_graph_sampler.empirical_node_feat_dist.values() for ft in v]
emb = torch.stack([ft["embedding"].max(0).values for ft in pool]).mean(0).to("cuda:0")


def gen_stats(fast=False, n_calls=16, bs=120, seed=2):
    M.fast_sampling = fast
    torch.manual_seed(seed); random.seed(seed)
    ne, deg0, ts = [], [], []
    for _ in range(n_calls):
        tm = g.draw_templates(bs)
        torch.cuda.synchronize(); t0 = time.time()
        res = g.generate_center_egos(tm, int(pool[0]["feature"].shape[1]) if pool[0]["feature"].dim() == 2 else 166, emb, HOPS)
        torch.cuda.synchronize(); ts.append(time.time() - t0)
        for r in res:
            if r is not None:
                ne.append(r[1].shape[1] // 2); deg0.append(int((r[1][0] == 0).sum()))
    return np.array(ne), np.array(deg0), ts


from baed.layers.layers import _seg_sum, _seg_max, _seg_mean
print("1. native segment ops == torch_scatter (with empty segments)")
idx = torch.tensor([0, 0, 2, 2, 2, 5], device="cuda"); x = torch.randn(6, 3, device="cuda")
for name, f, red in [("sum", _seg_sum, "sum"), ("max", _seg_max, "max"), ("mean", _seg_mean, "mean")]:
    a = f(x, idx, 7); b = torch_scatter.scatter(x, idx, dim=0, dim_size=7, reduce=red)
    check(f"{name}: max diff {(a - b).abs().max().item():.1e}", torch.allclose(a, b, atol=1e-6))
xm = torch.full((4, 2), float("-inf"), device="cuda"); im = torch.tensor([0, 0, 1, 1], device="cuda")
check("max of all -inf stays -inf (then repaired to 0 in the softmax)", torch.isinf(_seg_max(xm, im, 3)[:2]).all() and (_seg_max(xm, im, 3)[2] == 0).all())

random.seed(0); torch.manual_seed(0)
print("2-3. static forward (native ops, 1-row time embedding) and its compiled version vs the original forward")
batches = []
for trial in range(5):
    tm = g.draw_templates(110)
    bg = M.initial_graph_sampler._generate_empty_data([k for k, _ in tm], [ft for _, ft in tm]).to("cuda:0")
    bg = M._prepare_data_for_sampling(bg)
    cls = (torch.rand(bg.full_edge_index.shape[1], device="cuda") < 0.2 + 0.15 * trial).long()
    bg.log_full_edge_attr_t = DB.index_to_log_onehot(cls, 2)
    t = int(torch.randint(0, M.num_timesteps, (1,)))
    batches.append((bg, torch.full((bg.num_nodes,), t, device="cuda", dtype=torch.long),
                    torch.full((bg.full_edge_index.shape[1],), t, device="cuda", dtype=torch.long)))
d_static = d_comp = 0.0
den.enable_compile()
t0 = time.time()
with torch.no_grad():
    for bg, tn, te in batches:
        den.static = False; _, a = den(bg, tn, te, emb)
        den.static = True; _, b = den._forward_static(bg, tn, te, emb); _, c = den(bg, tn, te, emb)
        d_static = max(d_static, (a - b).abs().max().item()); d_comp = max(d_comp, (b - c).abs().max().item())
den.static = False
print(f"     (first compiled calls incl. compilation: {time.time() - t0:.1f} s for {len(batches)} shapes)")
check(f"static vs original: max |logit diff| {d_static:.1e}", d_static < 1e-3)
check(f"compiled vs static: max |logit diff| {d_comp:.1e}", d_comp < 1e-3)
den._static_impl = None

print("4. deterministic chain (Gumbel noise off): eager loop == CUDA-graph loop")
orig = M.log_sample_categorical
M.log_sample_categorical = lambda logits, k: DB.index_to_log_onehot(logits.argmax(1), k)
tm = g.draw_templates(150); res = []
for fast in (False, True):
    M.fast_sampling = fast
    init = M.initial_graph_sampler._generate_empty_data([k for k, _ in tm], [ft for _, ft in tm])
    res.append(M.sample(len(tm), emb, init_graph=init).edge_index.cpu())
M.log_sample_categorical = orig
check(f"identical edge lists ({res[0].shape[1]} vs {res[1].shape[1]} edges)", res[0].shape == res[1].shape and torch.equal(res[0], res[1]))

print(RUN, "5-6. generation statistics and speed")
for tag, fast, comp in [("eager", False, False), ("cuda-graph", True, False), ("cuda-graph+compile", True, True)]:
    den._static_impl = None
    M.__dict__.pop("_cg_cache", None)
    if comp:
        den.enable_compile()
    ne, d0, ts = gen_stats(fast)
    print(f"{tag:20s} edges/graph {ne.mean():.3f} | center degree {d0.mean():.3f} | share 1..5 edges "
          f"{[round(np.mean(ne == k), 4) for k in range(1, 6)]} | per call: first {1000*ts[0]:.0f} ms, median(after 2) "
          f"{1000*np.median(ts[2:]):.0f} ms")
print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
