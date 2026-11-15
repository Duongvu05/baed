"""Evaluate a saved detector checkpoint (outputs/detect_weights/*.pth) on the val and test splits.

Lets a running baed-detect job be checked without stopping it: the training loop saves the lowest-train-loss
weights after every improving epoch, and this script copies that file and scores it with the same split,
filtering and metrics as baed-detect (--batched).

Example:
  python scripts/eval_checkpoint.py --dataset reddit --data-tag _t --node-feats 64 --readout deviation_cat_nb \
      --weights outputs/detect_weights/reddit_GCN_BAED_GCN_BAED_s15_b_paug_cfill_devcatnb_t.pth
"""
import argparse
import json
import shutil
import tempfile

import torch
from torch.utils.data import DataLoader

from baed import paths
from baed.datasets.lean import keep, lean_collate, load_lean
from baed.pipelines import detect as D


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-tag", default="")
    p.add_argument("--node-feats", type=int, required=True)
    p.add_argument("--model", default="GCN")
    p.add_argument("--readout", default="deviation_cat_nb")
    p.add_argument("--gcn-self-loop", action="store_true")
    p.add_argument("--keep-isolated", action="store_true")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weights", required=True, help="checkpoint path (relative to the repo root or absolute)")
    p.add_argument("--gpu-store", action="store_true", help="score from GPU stores (as baed-detect --gpu-store)")
    p.add_argument("--out", default=None, help="write the metrics as JSON here")
    a = p.parse_args()

    pkl = str(paths.graph_file(f"{a.dataset}{a.data_tag}_init_0_1_dataset.pkl"))
    graphs = load_lean(pkl, lambda: D.load_dataset(pkl), paths.root() / "data" / "cache")
    _, eval_nx, test_nx = D.split_dataset(graphs)  # sequential 70/15/15, as baed-detect --split sequential
    sets = [[it for it in x if keep(it, a.keep_isolated)] for x in (eval_nx, test_nx)]
    if a.gpu_store:
        from baed.datasets.gpu_store import GpuStore

        val_dl, test_dl = (DataLoader(range(len(x)), batch_size=a.batch_size) for x in sets)
        for d, x in zip((val_dl, test_dl), sets):
            d.store = GpuStore(x, D.device, keep_isolated=a.keep_isolated)
    else:
        val_dl, test_dl = (DataLoader(x, batch_size=a.batch_size, shuffle=False, collate_fn=lean_collate) for x in sets)

    detector = D.build_detector(a.model, a.node_feats, a.readout, a.gcn_self_loop).to(D.device)
    src = a.weights if a.weights.startswith("/") else str(paths.root() / a.weights)
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:  # snapshot: the training job may overwrite the file
        shutil.copyfile(src, tmp.name)
        print("val:")
        v = D.test_batched(detector, val_dl, tmp.name)
        print("test:")
        m = D.test_batched(detector, test_dl, tmp.name)
        # threshold chosen on the validation set (macro F1), applied to the test set, as baed-detect does
        with torch.no_grad():
            yv, pv = D._probs(detector, val_dl)
        thr = D.val_threshold(yv, pv)
        print(f"test at the val threshold {thr:.4f}:")
        t = D.test_batched(detector, test_dl, tmp.name, thr=thr)
    m.update({f"val_{k}": v[k] for k in ("auroc", "auprc", "f1_macro", "f1_anomaly") if k in v})
    m.update(val_threshold=thr, thr_f1_macro=t.get("f1_macro"), thr_f1_anomaly=t.get("f1_anomaly"))
    print(f"threshold chosen on val {thr:.4f}: test macro F1 {m['thr_f1_macro']:.4f}, F1 class 1 {m['thr_f1_anomaly']:.4f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "metrics": m}, f, indent=2)


if __name__ == "__main__":
    main()
