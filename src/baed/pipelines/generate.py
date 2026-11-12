import numpy as np
import torch
import pickle
import argparse
import time
import os
import math
import networkx as nx
import torch_geometric as pyg

# Data
from baed import paths
from baed.datasets.data import get_data
# Model
from baed.model import get_model
from baed.gae.encoder import GCNEncoder
import random



class GenerateDataArgs:
    def __init__(self, run_name='2024-11-24_17-24-26', dataset='dgraph', num_samples=1, seed=0, checkpoint=1500, device='cuda:0'):
        self.device = device
        self.run_name = run_name
        self.dataset = dataset
        self.num_samples = num_samples
        self.seed = seed
        self.checkpoint = checkpoint

class GraphDataGenerator:
    def __init__(self, args):
        torch.manual_seed(args.seed)
        self.args = args
        self.log_dir = str(paths.run_dir(args.dataset, args.run_name))
        self.path_args = f'{self.log_dir}/args.pickle'
        self.path_check = f'{self.log_dir}/check/checkpoint_{args.checkpoint - 1}.pt'
        
        # Load args from file
        with open(self.path_args, 'rb') as f:
            self.args = pickle.load(f)
        
        self.args.device = getattr(args, 'device', 'cuda:0')
        self.train_loader, self.eval_loader, self.test_loader, self.num_node_feat, self.num_node_classes, self.num_edge_classes, self.max_degree, self.augmented_feature_dict, self.initial_graph_sampler, self.eval_evaluator, self.test_evaluator, self.monitoring_statistics = get_data(self.args)
        
        # Initialize model
        self.model = get_model(self.args, initial_graph_sampler=self.initial_graph_sampler)
        self.checkpoint = torch.load(self.path_check, map_location=self.args.device)
        self.model.load_state_dict(self.checkpoint['model'])
        
        if torch.cuda.is_available():
            self.model = self.model.to(self.args.device)
        self.model.eval()
    
    def generate_data(self, num_samples,embedding=None):
        # sample
        sampled_pygraph = self.model.sample(num_samples, embedding)
        pyg_datas = sampled_pygraph.to_data_list()
        # print("已生成pyg_datas:",len(pyg_datas))
        generated_nxgraphs = []
        for pyg_data in pyg_datas:
            feature_keys = [key for key in pyg_data.keys() if key.startswith('feature')]
            feature_keys.append("label")
            g_gen = pyg.utils.to_networkx(pyg_data, to_undirected=True)
            
            # 将属性添加到 NetworkX 图的节点属性中
            for i, node in enumerate(g_gen.nodes()):
                for key in feature_keys:
                    g_gen.nodes[node][key] = pyg_data[key][i]
            largest_cc = max(nx.connected_components(g_gen), key=len)
            g_gen = g_gen.subgraph(largest_cc)
            generated_nxgraphs.append(g_gen)
        
        return generated_nxgraphs

    def generate_arrays(self, num_samples, feature_num, embedding=None):
        """Same samples as ``generate_data`` but returned as tensors (see ``postprocess_batch``)."""
        return postprocess_batch(self.model.sample(num_samples, embedding), feature_num)

    def draw_templates(self, n, min_nodes=2):
        """n template ego-graphs (node features of a real train fraud ego-graph) with >= min_nodes nodes, drawn
        uniformly from the empirical pool (so sizes follow the empirical distribution restricted to >= min_nodes)."""
        if not hasattr(self, "_pool"):
            self._pool = {}
        if min_nodes not in self._pool:
            d = self.initial_graph_sampler.empirical_node_feat_dist
            self._pool[min_nodes] = [(k, ft) for k, v in d.items() if k >= min_nodes for ft in v]
        return random.choices(self._pool[min_nodes], k=n)

    def generate_center_egos(self, templates, feature_num, embedding, hops):
        """Sample edges for the given templates. Node 0 of every template is its center A.

        Returns one entry per template: (features, edge_index) of the hops-ego-graph around A in the generated
        graph, A renumbered to 0, edges in both directions; or None when A has no neighbour other than itself
        (edges between other nodes, or a self-loop on A, do not count).
        """
        init = self.initial_graph_sampler._generate_empty_data([k for k, _ in templates], [ft for _, ft in templates])
        emb = embedding
        if emb is not None and emb.dim() == 1:
            emb = emb.reshape(1, -1).expand(len(templates), -1)
        batch = self.model.sample(len(templates), emb, init_graph=init)
        ptr = batch.ptr.cpu().numpy()
        ei = batch.edge_index.cpu().numpy() if batch.edge_index.numel() else np.zeros((2, 0), np.int64)
        x = torch.stack([batch[f"feature{i}"].reshape(batch.num_nodes) for i in range(feature_num)], dim=1).cpu()
        gid = np.searchsorted(ptr, ei[0], side="right") - 1
        out = []
        for g in range(len(templates)):
            s, t = ptr[g], ptr[g + 1]
            m = gid == g
            u, v = ei[0, m] - s, ei[1, m] - s
            keep = u != v
            u, v = u[keep], v[keep]
            adj = [set() for _ in range(t - s)]
            for a, b in zip(u.tolist(), v.tolist()):
                adj[a].add(b); adj[b].add(a)
            if not adj[0]:
                out.append(None)
                continue
            seen, frontier = {0}, {0}
            for _ in range(hops):
                frontier = {w for q in frontier for w in adj[q]} - seen
                if not frontier:
                    break
                seen |= frontier
            nodes = [0] + sorted(seen - {0})
            pos = {q: i for i, q in enumerate(nodes)}
            e = [(pos[a], pos[b]) for a in nodes for b in adj[a] if b in pos]
            out.append((x[torch.tensor(nodes) + int(s)], torch.tensor(e, dtype=torch.long).t().contiguous()))
        return out
    

def postprocess_batch(batch, feature_num):
    """Fast equivalent of ``generate_data`` + ``utils.processGenerateSubgraph``.

    Takes the sampled PyG batch, keeps the largest connected component of every graph
    (Rust kernel ``baed_graph.largest_components``) and returns ``(feats, edge_indices)``:
    lists of ``[k, feature_num]`` tensors and ``[2, E]`` long tensors, without the
    pyg -> networkx -> pyg round trip (which deep-copies every node attribute).
    Same graphs as the original path up to a permutation of nodes inside a graph.
    """
    from baed_graph import largest_components

    n = batch.num_nodes
    if hasattr(batch, "ptr") and batch.ptr is not None:
        ptr = batch.ptr.cpu().numpy().astype("int64")
    else:
        counts = torch.bincount(batch.batch, minlength=batch.num_graphs)
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).cpu().numpy().astype("int64")
    ei = batch.edge_index.cpu().numpy().astype("int64") if batch.edge_index.numel() else np.zeros((2, 0), "int64")
    nodes, node_ptr, edges, edge_ptr = largest_components(ptr, np.ascontiguousarray(ei[0]), np.ascontiguousarray(ei[1]))
    x = torch.stack([batch[f"feature{i}"].reshape(n) for i in range(feature_num)], dim=1).cpu()
    nodes_t = torch.from_numpy(nodes)
    edges_t = torch.from_numpy(edges).reshape(-1, 2)
    feats, eidx = [], []
    for g in range(len(node_ptr) - 1):
        feats.append(x[nodes_t[node_ptr[g]:node_ptr[g + 1]]])
        eidx.append(edges_t[edge_ptr[g]:edge_ptr[g + 1]].t().contiguous())
    return feats, eidx


def main(argv=None):
    """Step 04 - synthesize ego-graphs with a trained BAED checkpoint."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--run-name", required=True, help="run folder under outputs/runs/<dataset>/multinomial_diffusion/multistep/")
    parser.add_argument("--dataset", default="elliptic")
    parser.add_argument("--checkpoint", type=int, default=50, help="loads check/checkpoint_{N-1}.pt")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num-items", type=int, default=12800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--keep", type=int, default=15509, help="truncate the generated list to this many graphs (original script)")
    parser.add_argument("--device", default="cuda:0")
    cli = parser.parse_args(argv)

    dataset = cli.dataset
    args = GenerateDataArgs(run_name=cli.run_name, dataset=dataset, num_samples=64, seed=cli.seed,
                            checkpoint=cli.checkpoint, device=cli.device)
    generator = GraphDataGenerator(args)

    item_num = cli.num_items
    batch_size = cli.batch_size
    iter_num = item_num // batch_size
    generate_dataset = []
    file_name = paths.ensure(paths.generated_dir() / f"{dataset}_generate_{item_num}.pkl")

    start_time = time.time()
    for i in range(iter_num):
        generate_data = generator.generate_data(batch_size)
        generate_dataset = generate_dataset + generate_data
    end_time = time.time()
    generate_dataset = generate_dataset[0:cli.keep]
    print("len", len(generate_dataset), f"({end_time - start_time:.1f}s)")

    with open(file_name, 'wb') as f:
        pickle.dump(generate_dataset, f)
    print("saved:", file_name)

    # demo: dump the last generated batch as a node/edge list
    combined_graph = nx.Graph()
    for graph in generate_data:
        combined_graph = nx.compose(combined_graph, graph)
    nodes = [{'id': str(node), 'label': int(combined_graph.nodes[node]["label"].item())} for node in combined_graph.nodes]
    edges = [{'source': str(edge[0]), 'target': str(edge[1])} for edge in combined_graph.edges]
    print("result:", {'nodes': nodes, 'edges': edges})


if __name__ == "__main__":
    main()
