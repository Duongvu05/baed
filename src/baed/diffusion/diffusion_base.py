import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
from torch_scatter import scatter
import torch_geometric as pyg
"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

# logx ,logy ->log(x+y)
def log_add_exp(a, b):
    # one kernel instead of 8 (max, 2 sub, 2 exp, add, log, add); same value up to float rounding
    return torch.logaddexp(a, b)


def log_sub_exp(a,b):
    assert torch.any(a > b), f'Error: {a > b}'
    return a + torch.log1p(-torch.exp(b-a))


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    # assert x.max().item() < num_classes, \
    #     f'Error: {x.max().item()} >= {num_classes}'
    # if x.numel() == 0:
    #     x=torch.tensor([0],device="cuda:0")
    # scatter_ instead of F.one_hot: same result for valid indices, but F.one_hot checks the index range with a
    # device -> host sync, which also breaks CUDA-graph capture of the sampling loop
    x_onehot = torch.zeros(*x.shape, num_classes, dtype=torch.long, device=x.device).scatter_(-1, x.unsqueeze(-1), 1)

    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_x


def create_node_selections(log_x_t, log_x_tminus1, batched_graph):
    d_t = scatter(log_x_t.argmax(1), batched_graph.row, dim=1, dim_size=batched_graph.max_num_nodes) +  scatter(log_x_t.argmax(1), batched_graph.col, dim=1, dim_size=batched_graph.max_num_nodes) 
    d_tminus1 = scatter(log_x_tminus1.argmax(1), batched_graph.row, dim=1, dim_size=batched_graph.max_num_nodes) +  scatter(log_x_tminus1.argmax(1), batched_graph.col, dim=1, dim_size=batched_graph.max_num_nodes)
    return (d_tminus1 > d_t).long()


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return 1 - torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64).numpy()


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def Tt1_beta_schedule(timesteps):
    return 1/torch.linspace(1+1e-8, timesteps+1e-8, timesteps, dtype = torch.float64).flip(0).numpy() 


class DiffusionBase(torch.nn.Module):
    def __init__(self, num_node_classes, num_edge_classes, initial_graph_sampler, denoise_fn, timesteps=1000,
                sample_time_method='importance', device='cuda'):
        super(DiffusionBase, self).__init__()

        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        
        self._denoise_fn = denoise_fn
        self.initial_graph_sampler = initial_graph_sampler
        self.num_timesteps = timesteps
        self.sample_time_method = sample_time_method
        self.device = device
        

    def _q_pred(self, batched_graph, t_node, t_edge):
        raise NotImplementedError()

    def _p_pred(self, batched_graph, t_node, t_edge):
        raise NotImplementedError()

    def _prepare_data_for_sampling(self, batched_graph):
        raise NotImplementedError()

    def _eval_loss(self, batched_graph):
        raise NotImplementedError()

    def _train_loss(self, batched_graph):
        raise NotImplementedError()

    def _sample_time(self, b, device, method):
        raise NotImplementedError()
    def _calc_num_entries(self, batched_graph):
        raise NotImplementedError()

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    # q(xt|x0)
    def q_sample(self, batched_graph, t_node, t_edge):

        log_prob_node, log_prob_edge = self._q_pred(batched_graph, t_node, t_edge)

        # sample nodes
        log_out_node = self.log_sample_categorical(log_prob_node, self.num_node_classes)

        log_out_edge = self.log_sample_categorical(log_prob_edge, self.num_edge_classes)

        return log_out_node, log_out_edge 
    
    # p
    @torch.no_grad()
    def p_sample(self, batched_graph, t_node, t_edge,embedding):
        # p_sample is always one step prediction!
        if not getattr(self._denoise_fn, "predicts_nodes", True):
            # The denoiser returns the clean node one-hot and never reads the noisy node state, so the node chain does
            # not influence the edges: keep the node state as is and run only the edge chain.
            log_model_prob_edge = self._p_pred_edges(batched_graph, t_node, t_edge, embedding)
            return batched_graph.log_node_attr_t, self.log_sample_categorical(log_model_prob_edge, self.num_edge_classes)
        log_model_prob_node, log_model_prob_edge = self._p_pred(batched_graph, t_node, t_edge,embedding)
        
        log_out_node = self.log_sample_categorical(log_model_prob_node, self.num_node_classes)

        log_out_edge = self.log_sample_categorical(log_model_prob_edge, self.num_edge_classes)
        return log_out_node, log_out_edge

    def log_sample_categorical(self, logits, num_classes):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        # print("gumbel_noise:",gumbel_noise)
        # print("logits:",logits)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, num_classes)
        return log_sample


    def log_prob(self, batched_graph):
        # 通过VLB来计算loss
        if self.training:
            return self._train_loss(batched_graph)
        else:
            return self._eval_loss(batched_graph)


    def sample(self, num_samples,embedding, init_graph=None):
        # init_graph: an empty batch built from given templates (sizes + node features); None = draw at random
        batched_graph = self.initial_graph_sampler.sample(num_samples) if init_graph is None else init_graph
        batched_graph.to(self.device)

        num_nodes = batched_graph.nodes_per_graph.sum()
        num_edges = batched_graph.edges_per_graph.sum()
        batched_graph = self._prepare_data_for_sampling(batched_graph)

        if getattr(self, "fast_sampling", False) and str(self.device).startswith("cuda") and embedding is not None:
            self._sample_loop_cuda_graph(batched_graph, embedding, int(num_nodes), int(num_edges))
        else:
            self._sample_loop(batched_graph, embedding, num_nodes, num_edges)
        return self._finish_sampling(batched_graph)

    def _sample_loop(self, batched_graph, embedding, num_nodes, num_edges):
        # print()
        for t in reversed(range(0, self.num_timesteps)):
            # print(f'Sample timestep {t:4d}', end='\r')
            t_node = torch.full((num_nodes,), t, device=self.device, dtype=torch.long)
            t_edge = torch.full((num_edges,), t, device=self.device, dtype=torch.long)

            log_node_attr_tmin1, log_full_edge_attr_tmin1 = self.p_sample(batched_graph, t_node, t_edge,embedding)
            batched_graph.log_full_edge_attr_t = log_full_edge_attr_tmin1
            batched_graph.log_node_attr_t = log_node_attr_tmin1

    @staticmethod
    def _bucket(n, step):
        return ((n + step - 1) // step) * step

    @torch.no_grad()
    def _sample_loop_cuda_graph(self, batched_graph, embedding, num_nodes, num_edges):
        """The same reverse chain as _sample_loop, replayed from a captured CUDA graph.

        The denoiser runs in its static, masked mode (TGNN_gin_guided._forward_static), so one step only depends on
        the sizes (nodes, candidate edges, graphs). Sizes are rounded up to buckets and the extra rows form one
        padding graph (its nodes and candidate edges never touch a real graph: every operation is per node, per edge
        inside a graph, or per graph). One graph is captured per bucket and reused across calls; t is written to a
        static buffer before every replay. Removes the per-step launch overhead (sampling is launch-bound).
        """
        B = int(batched_graph.num_graphs)
        Nc = self._bucket(num_nodes + 2, 64)          # >= 2 padding nodes for the padding edges
        Ec = self._bucket(num_edges + 1, 128)
        Bc = self._bucket(B + 1, 32)                  # +1: the padding graph
        emb_per_graph = embedding is not None and embedding.dim() == 2
        key = (Nc, Ec, Bc, embedding is None, emb_per_graph)
        cache = self.__dict__.setdefault("_cg_cache", {})
        if key not in cache:
            cache[key] = self._capture_bucket(Nc, Ec, Bc, batched_graph, embedding)
        c = cache[key]
        st = c["static"]
        # load the real batch into the static buffers, padding = one extra graph (index B)
        st.feature.zero_(); st.feature[:num_nodes].copy_(batched_graph.feature)
        st.batch.fill_(B); st.batch[:num_nodes].copy_(batched_graph.batch)
        st.full_edge_index[0].fill_(num_nodes); st.full_edge_index[1].fill_(num_nodes + 1)
        st.full_edge_index[:, :num_edges].copy_(batched_graph.full_edge_index)
        st.log_full_edge_attr_t.copy_(c["pad_e"]); st.log_full_edge_attr_t[:num_edges].copy_(batched_graph.log_full_edge_attr_t)
        st.log_node_attr_t.copy_(c["pad_n"]); st.log_node_attr_t[:num_nodes].copy_(batched_graph.log_node_attr_t)
        st.log_node_attr.copy_(c["pad_n"]); st.log_node_attr[:num_nodes].copy_(batched_graph.log_node_attr)
        if embedding is not None:
            e = embedding.detach().to(self.device, torch.float)
            if emb_per_graph:
                c["emb"].zero_(); c["emb"][:B].copy_(e)
            else:
                c["emb"].copy_(e)
        for t in reversed(range(0, self.num_timesteps)):
            c["t_node"].fill_(t); c["t_edge"].fill_(t)
            c["graph"].replay()
        batched_graph.log_full_edge_attr_t = st.log_full_edge_attr_t[:num_edges].clone()
        batched_graph.log_node_attr_t = st.log_node_attr_t[:num_nodes].clone()

    def _capture_bucket(self, Nc, Ec, Bc, like, embedding):
        from types import SimpleNamespace
        dev = self.device
        F_ = like.feature.shape[1]
        st = SimpleNamespace(
            num_nodes=Nc, num_graphs=Bc,
            feature=torch.zeros(Nc, F_, device=dev, dtype=like.feature.dtype),
            batch=torch.zeros(Nc, device=dev, dtype=torch.long),
            full_edge_index=torch.zeros(2, Ec, device=dev, dtype=torch.long),
            log_full_edge_attr_t=torch.zeros(Ec, like.log_full_edge_attr_t.shape[1], device=dev),
            log_node_attr_t=torch.zeros(Nc, like.log_node_attr_t.shape[1], device=dev),
            log_node_attr=torch.zeros(Nc, like.log_node_attr.shape[1], device=dev))
        pad_e = like.log_full_edge_attr_t[:1].expand(Ec, -1).clone()   # any valid one-hot state
        pad_n = like.log_node_attr_t[:1].expand(Nc, -1).clone()
        st.log_full_edge_attr_t.copy_(pad_e); st.log_node_attr_t.copy_(pad_n); st.log_node_attr.copy_(pad_n)
        st.full_edge_index[1].fill_(1)
        if embedding is None:
            emb = None
        elif embedding.dim() == 2:
            emb = torch.zeros(Bc, embedding.shape[1], device=dev)
        else:
            emb = embedding.detach().to(dev, torch.float).clone()
        t_node = torch.full((Nc,), self.num_timesteps - 1, device=dev, dtype=torch.long)
        t_edge = torch.full((Ec,), self.num_timesteps - 1, device=dev, dtype=torch.long)
        den = self._denoise_fn

        def step():
            log_n, log_e = self.p_sample(st, t_node, t_edge, emb)
            st.log_node_attr_t.copy_(log_n)
            st.log_full_edge_attr_t.copy_(log_e)

        den.static = True
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):  # warm-up (lazy init of cuBLAS / cuDNN handles)
                for _ in range(2):
                    step()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
        finally:
            den.static = False
        return dict(static=st, graph=graph, t_node=t_node, t_edge=t_edge, emb=emb, pad_e=pad_e, pad_n=pad_n)

    def _finish_sampling(self, batched_graph):
        # print()
        edge_attr = batched_graph.log_full_edge_attr_t.argmax(-1)
        is_edge_indices = edge_attr.nonzero(as_tuple=True)[0]

        edge_index = batched_graph.full_edge_index[:, is_edge_indices]
        batched_graph.edge_index = edge_index 

        edge_attr = edge_attr[is_edge_indices]
        batched_graph.edge_attr = edge_attr

        
        batched_graph.node_attr = batched_graph.log_node_attr_t.argmax(-1)

        # preparation for splitting batched graph
        # see https://github.com/pyg-team/pytorch_geometric/blob/259cfa7fb220d9cb504ab9de52bcd9dc5267befe/torch_geometric/data/separate.py#L12
        edge_slice = batched_graph.batch[batched_graph.edge_index[0]]
        edge_slice = scatter(torch.ones_like(edge_slice), edge_slice, dim_size=batched_graph.num_graphs )
        edge_slice = torch.nn.functional.pad(edge_slice, (1,0), 'constant', 0)
        edge_slice = torch.cumsum(edge_slice, 0)
        batched_graph._slice_dict['edge_index'] = edge_slice
        batched_graph._inc_dict['edge_index'] = batched_graph._inc_dict['full_edge_index']

        return batched_graph
