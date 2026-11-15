"""Lean ego-graphs held on the GPU, batched by index gathers.

``detect._batch`` builds every mini-batch in Python: per ego-graph it checks the edge list
(``e.max()``, ``torch.unique``), shifts it by the node offset and concatenates, then copies the
batch to the device. With small ego-graphs (DGraph: 4 nodes on average, 1.2 M graphs) this loop,
run again for the validation set after every epoch, is what bounds the run time.

``GpuStore`` does that work once: it applies the same skip rules to every ego-graph, keeps the
surviving features and local edge lists concatenated on the device, and builds a batch from a
tensor of item indices with ``repeat_interleave`` gathers. The batched graph is the same as the one
``_batch`` builds for the same items (same node order, same edge order, same batch sizes), so the
detector sees identical inputs. Training batches come from a DataLoader over ``range(len(items))``
with the same batch size and shuffle, which draws the same permutation as a DataLoader over the
items themselves.
"""

from __future__ import annotations

import dgl
import torch


class GpuStore:
    def __init__(self, items, device, keep_isolated: bool = False):
        feats, edges, n, ne, y = [], [], [], [], []
        slot = torch.full((len(items),), -1, dtype=torch.long)
        for i, (f, e, label) in enumerate(items):
            k = f.shape[0]
            if k == 0:
                continue
            if e.numel() == 0:
                if not keep_isolated:
                    continue
                e = torch.arange(k).repeat(2, 1)  # self-loops, as in detect._batch / _to_dgl
            elif int(e.max()) + 1 != k or torch.unique(e).numel() != k:
                continue  # skipped by detect._batch as well (never happens for ego-graphs)
            slot[i] = len(n)
            feats.append(f)
            edges.append(e)
            n.append(k)
            ne.append(e.size(1))
            y.append(int(label))
        self.device = device
        self.slot = slot  # item index -> row of the store, -1 for skipped items (stays on the CPU)
        if not n:
            self.size = 0
            return
        self.size = len(n)
        max_n = max(n)
        edtype = torch.int16 if max_n <= torch.iinfo(torch.int16).max else torch.int32
        self.x = torch.cat(feats).to(device)
        self.e = torch.cat(edges, 1).to(edtype).to(device)  # local node ids, int16 when the graphs allow it
        self.n = torch.tensor(n, device=device)
        self.ne = torch.tensor(ne, device=device)
        self.node_ptr = torch.cumsum(self.n, 0) - self.n
        self.edge_ptr = torch.cumsum(self.ne, 0) - self.ne
        self.y = torch.tensor(y, device=device)

    @staticmethod
    def _ranges(starts, lengths, total):
        """Concatenation of arange(s, s + l) for every (s, l)."""
        first = torch.cumsum(lengths, 0) - lengths
        return torch.repeat_interleave(starts - first, lengths, output_size=total) + torch.arange(total, device=starts.device)

    def batch(self, idx, features_fn=None):
        """Same return value as detect._batch for the items ``idx`` (in this order): (graph, labels, positions)."""
        idx = torch.as_tensor(idx, dtype=torch.long)
        rows = self.slot[idx]
        keep = rows >= 0
        pos = torch.nonzero(keep).flatten().tolist()
        if not pos:
            return None, None, pos
        s = rows[keep].to(self.device)
        n, ne = self.n[s], self.ne[s]
        tot_n, tot_e = int(n.sum()), int(ne.sum())
        x = self.x[self._ranges(self.node_ptr[s], n, tot_n)]
        offset = torch.repeat_interleave(torch.cumsum(n, 0) - n, ne, output_size=tot_e)
        e = self.e[:, self._ranges(self.edge_ptr[s], ne, tot_e)].long() + offset
        g = dgl.graph((e[0], e[1]), num_nodes=tot_n)
        g.set_batch_num_nodes(n)
        g.set_batch_num_edges(ne)
        g.ndata['feature'] = features_fn(x, e, tot_n) if features_fn is not None else x
        return g, self.y[s], pos

    def nbytes(self) -> int:
        if not self.size:
            return 0
        return sum(t.numel() * t.element_size() for t in (self.x, self.e, self.n, self.ne, self.y))
