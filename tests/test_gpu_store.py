"""GpuStore.batch must build the same batched graph as detect._batch for the same items."""
import torch

from baed.datasets.gpu_store import GpuStore
from baed.pipelines import detect as D


def _items(seed=0, count=300):
    g = torch.Generator().manual_seed(seed)
    items = []
    for i in range(count):
        n = int(torch.randint(1, 30, (1,), generator=g))
        f = torch.randn(n, 7, generator=g, dtype=torch.float64)
        if n == 1 or i % 37 == 0:
            e = torch.zeros((2, 0), dtype=torch.long)  # no edges (isolated center)
        else:
            m = int(torch.randint(n - 1, 3 * n, (1,), generator=g))
            e = torch.randint(0, n, (2, m), generator=g)
            e[0, : n - 1] = 0
            e[1, : n - 1] = torch.arange(1, n)  # star part: every node appears
            if i % 53 == 0:
                e[1, 0] = n + 5  # broken edge list, skipped by both
        items.append((f, e, int(torch.randint(0, 2, (1,), generator=g))))
    return items


def _same(a, b):
    ga, ya, pa = a
    gb, yb, pb = b
    assert pa == pb
    if ga is None:
        assert gb is None
        return
    assert torch.equal(ya.cpu(), yb.cpu())
    assert ga.num_nodes() == gb.num_nodes() and ga.num_edges() == gb.num_edges()
    for x, y in zip(ga.edges(), gb.edges()):
        assert torch.equal(x.cpu(), y.cpu())
    assert torch.equal(ga.batch_num_nodes().cpu(), gb.batch_num_nodes().cpu())
    assert torch.equal(ga.batch_num_edges().cpu(), gb.batch_num_edges().cpu())
    fa, fb = ga.ndata["feature"], gb.ndata["feature"]
    assert fa.dtype == fb.dtype and torch.equal(fa.cpu(), fb.cpu())


def _check(keep_isolated, dev):
    D.device = dev
    D.KEEP_ISOLATED = keep_isolated
    items = _items()
    store = GpuStore(items, dev, keep_isolated=keep_isolated)
    perm = torch.randperm(len(items), generator=torch.Generator().manual_seed(1))
    for idx in perm.split(64):
        sub = [items[i] for i in idx.tolist()]
        ref = D._batch([it[0] for it in sub], [it[1] for it in sub], torch.tensor([it[2] for it in sub]))
        _same(store.batch(idx), ref)


def test_same_batches_cpu():
    for k in (False, True):
        _check(k, torch.device("cpu"))


def test_same_batches_cuda():
    if not torch.cuda.is_available():
        return
    for k in (False, True):
        _check(k, torch.device("cuda:0"))


def test_index_loader_same_permutation():
    items = list(range(1000))
    torch.manual_seed(15)
    a = [b.tolist() for b in torch.utils.data.DataLoader(items, batch_size=128, shuffle=True, collate_fn=lambda x: torch.tensor(x))]
    torch.manual_seed(15)
    b = [x.tolist() for x in torch.utils.data.DataLoader(range(len(items)), batch_size=128, shuffle=True)]
    assert a == b
