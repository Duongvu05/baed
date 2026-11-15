# Run scripts used on the A100 pod

Copies of the chain scripts run from `/home/data/baed_data/` on the pod (repo at `/home/data/baed`, branch
`exp/val-select`). All use the val protocol: epoch chosen by validation AUROC (`--select val_auroc`) and the
decision threshold chosen on validation (keys `thr_*` in the result JSON). Detector for every dataset: GCN, no
self-loop, readout `deviation_cat_nb`, lr 1e-3, 50 epochs, seed 15, sequential 70/15/15 split in file order,
single-node ego-graphs dropped.

| Script | Dataset | Ego-graphs |
|---|---|---|
| `reddit_u_valsel.sh` | Reddit | users only as centers (`_u`), 1 hop; base and +BAED (center-flag diffusion) at lr 1e-3 and 1e-6 |
| `photo_valsel_base.sh` | Photo | every node, 1 hop (released code); base only |
| `tfinance_valsel_base.sh` | T-Finance | every node, 1 hop, ego-graphs > 1200 nodes dropped (released code); base only |
| `dgraph_t_valsel_base.sh` | DGraph-Fin | as `dgraph_valsel_base.sh`, sorted by each center's first edge day (temporal split, tag `_t`); base only |
| `tfinance_t_base.sh` | T-Finance | as `tfinance_valsel_base.sh`, sorted by registration day (-feature 0, temporal split, tag `_t`); base, 5 seeds, `--gpu-store` |
| `tfinance_nocap_base.sh` | T-Finance | without the 1200-node cap (`--no-cap`), id order and registration-day order; base, one seed |
| `seeds_base.sh` | any | extra seeds 42, 63, 87, 94 of the base, run in parallel |
| `dgraph_valsel_base.sh` | DGraph-Fin | nodes labelled 0/1 as centers, 2 hops along out-edges, background nodes 2/3 kept as neighbours (released code); base only |

The base runs pass a `--gae-weight` file only because `baed-detect` always loads one. The base does not use it.
