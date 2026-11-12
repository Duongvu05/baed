# 训练子图异常检测模型的代码
from baed import paths
import numpy as np
import math
import torch
import os
import networkx as nx
import numpy as np
import dgl
from dgl.data.utils import load_graphs, save_graphs
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch_geometric.data import data
import torch_geometric as pyg
import random
from functools import partial
from torch_geometric.datasets import QM9
from sklearn.model_selection import train_test_split
import pandas as pd
import torch_geometric.data
import pickle

from torch_geometric.utils import to_dense_adj
from torch_geometric import transforms as T
from torch.nn import functional as F

import torch.optim as optim

# 引入子图异常检测模型
from baed.models.gnn import BWGNN, BernNet, GCN, GIN, GraphSAGE
from tqdm import tqdm
# 引入数据预处理
from baed.datasets.graphDataset import GraphDataset
from baed.utils import preprocess, preprocessDataset, collate_fn, processGenerateSubgraph
# 引入gae
from baed.gae.encoder import GCNEncoder
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GAE
# 引入BAED
from baed.pipelines.generate import GraphDataGenerator, GenerateDataArgs


import torch.multiprocessing
from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, classification_report, f1_score
import random
import copy


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sample_losses = []



def load_dataset(dataset_path):
    """
    Load the dataset from the given path.

    Parameters:
    dataset_path (str): The path to the dataset file.

    Returns:
    list: The loaded dataset.
    """
    with open(dataset_path, 'rb') as file:
        nx_graphs = pickle.load(file)
    return nx_graphs


def split_dataset(nx_graphs, train_ratio=0.7, eval_ratio=0.15, test_ratio=0.03):
    """
    Split the dataset into training, evaluation, and test sets.

    Parameters:
    nx_graphs (list): The dataset to be split.
    train_ratio (float): The ratio of the training set.
    eval_ratio (float): The ratio of the evaluation set.
    test_ratio (float): The ratio of the test set.

    Returns:
    tuple: A tuple containing the training, evaluation, and test sets.
    """
    total_length = len(nx_graphs)
    train_size = int(train_ratio * total_length)
    eval_size = int(eval_ratio * total_length)
    test_size = eval_size  # Assuming test_ratio is equal to eval_ratio

    train_nx_graphs = nx_graphs[:train_size]
    eval_nx_graphs = nx_graphs[train_size:train_size + eval_size]
    test_nx_graphs = nx_graphs[train_size + eval_size:train_size + eval_size + test_size]

    return train_nx_graphs, eval_nx_graphs, test_nx_graphs


def create_dataloaders(train_set, eval_set, test_set, batch_size, num_workers, pin_memory, collate_fn):
    """
    Create DataLoader objects for training, evaluation, and test sets.

    Parameters:
    train_set (Dataset): The training dataset.
    eval_set (Dataset): The evaluation dataset.
    test_set (Dataset): The test dataset.
    batch_size (int): The batch size for the DataLoader.
    collate_fn (function): The collate function to be used by the DataLoader.

    Returns:
    tuple: A tuple containing the DataLoader objects for training, evaluation, and test sets.
    """
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                              shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                             shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                             shuffle=False, collate_fn=collate_fn)

    return train_loader, eval_loader, test_loader


def construct_graph(feat, edge_index):
    # print("feat:",feat.shape)
    # print("edge_index:",edge_index)
    """
    Construct a graph using networkx from node features and edge indices.

    Parameters:
    feat (torch.Tensor): Node features.
    edge_index (torch.Tensor): Edge indices.

    Returns:
    nx.Graph: The constructed graph.
    """
    # 获取所有唯一节点索引
    unique_nodes = torch.unique(edge_index)
    # 提取相应的节点特征
    unique_feat = feat[unique_nodes]
    # 创建 DGL 图
    graph = dgl.graph((edge_index[0], edge_index[1]))
    # 添加节点特征和标签
    graph.ndata['feature'] = unique_feat.to(device)
    return graph


def test_subgraph_anomaly_detector(detector, test_loader, device, model_path):
    """
    Test a subgraph anomaly detection model on the test set and return evaluation metrics.
    """
    detector.load_state_dict(torch.load(model_path, map_location=device))
    detector.to(device)
    detector.eval()

    all_labels = []
    all_probs = []      # 正类 (anomaly) 概率
    all_predicted = []

    with torch.no_grad():
        for data in test_loader:
            if not data:
                continue
            feats, labels = data.subFeature, data.sub_label
            edges_per_subgraph = data.sub_edge_index

            for i in range(len(feats)):
                feat = feats[i].to(device)
                edge_index = edges_per_subgraph[i].to(device)
                label = labels[i].item()

                if feat.shape[0] == 0:
                    continue

                graph = construct_graph(feat, edge_index)
                if graph.ndata['feature'].shape[0] == 0:
                    continue

                output = detector(graph)
                prob = F.softmax(output, dim=0).cpu().numpy()
                pred = int(np.argmax(prob))

                all_labels.append(label)
                all_probs.append(prob[1])  # P(anomaly)
                all_predicted.append(pred)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_predicted = np.array(all_predicted)

    accuracy = np.mean(all_predicted == all_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    metrics = {'accuracy': accuracy}

    # Only compute AUC metrics if both classes present
    if len(np.unique(all_labels)) > 1:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
        report = classification_report(all_labels, all_predicted, digits=5, target_names=["正常", "异常"])
        f1_macro = f1_score(all_labels, all_predicted, average='macro')

        metrics.update({
            'auroc': float(auroc),
            'auprc': float(auprc),
            'f1_macro': float(f1_macro),
            'f1_anomaly': float(f1_score(all_labels, all_predicted, pos_label=1)),
            'n_test': int(len(all_labels)),
            'n_test_anomaly': int(all_labels.sum()),
            'classification_report': report
        })

        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print("Classification Report:\n", report)
    else:
        print("Warning: Only one class in test set. Skipping AUC metrics.")

    return metrics

# 训练模型
def train_subgraph_anomaly_detector(detector, generator, gae, train_loader, eval_loader, test_loader, num_epochs,
                                    optimizer, device, dataset, use_baed=True,withcondition=True, run_tag="run"):
    """ Train a subgraph anomaly detection model.
    Parameters:
    detector (torch.nn.Module): The anomaly detection model.
    generator (object): The generator for generating negative samples.
    gae (torch.nn.Module): The graph autoencoder model.
    train_loader (DataLoader): The DataLoader for the training set.
    num_epochs (int): The number of epochs to train the model.
    optimizer (torch.optim.Optimizer): The optimizer for training.
    device (torch.device): The device to run the training on.
    use_baed (bool): Whether to use BAED for generating negative samples.

    Returns:
    None
    """
    best_avg_error = float('inf')
    if use_baed:
        generatorName = "BAED"
    else:
        generatorName = "None"
    model_path = str(paths.ensure(paths.root() / "outputs" / "detect_weights" / f"{dataset}_{detector.modelName}_{generatorName}_{run_tag}.pth"))
    # 训练集训练+验证集验证

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        epoch_loss = 0.0
        num_samples = 0
        detector.train()
        batch_num = 0

        for data in tqdm(train_loader, leave=False):
            # print("data:", data)
            batch_sample_loss = []
            if data:
                feats, labels = data.subFeature, data.sub_label
                # print("labels:",labels)
                edges_per_subgraph = data.sub_edge_index
                num_normal = (labels == 0).sum().item()
                num_anomalies = (labels == 1).sum().item()
                print("difference:", num_normal - num_anomalies)
                try:
                    if use_baed and num_anomalies < num_normal:
                        if withcondition:
                            difference = num_normal - num_anomalies
                            indices = torch.where(labels == 1)
                            indices_list = indices[0].tolist() if len(indices) else []
                            abnormal_subgraph_feats = [feats[index] for index in indices_list]
                            abnormal_edge = [edges_per_subgraph[index] for index in indices_list]
                            abnormal_subgraph_loss = [sample_losses[batch_num][index] if epoch != 0 else 1 for index in
                                                            indices_list]

                            sub_node_embedding_list = []
                            if abnormal_subgraph_feats:
                                for i in range(len(abnormal_subgraph_feats)):
                                    item_feat = abnormal_subgraph_feats[i].to(device)
                                    item_edge_index = abnormal_edge[i].to(device)
                                    item_loss = abnormal_subgraph_loss[i]
                                    total_sub_node_embedding = gae.encode(item_feat.to(dtype=torch.float), item_edge_index)
                                    max_pooled = torch.max(total_sub_node_embedding, dim=0).values
                                    sub_node_embedding_list.append(max_pooled)
                                weighted_node_embedding = torch.zeros_like(sub_node_embedding_list[0])
                                for i in range(len(sub_node_embedding_list)):
                                    weight = abnormal_subgraph_loss[i] / sum(abnormal_subgraph_loss)
                                    weighted_node_embedding += weight * sub_node_embedding_list[i]
                            else:
                                weighted_node_embedding = None
                        else:
                            weighted_node_embedding = None
                                    # print("weighted_node_embedding:",weighted_node_embedding)
                        print("需要合成{}个负样本数据".format(difference))
                        if difference >= 64:
                            batch_generate_size = 64
                            total = difference
                            num_batches = (total + batch_generate_size - 1) // batch_generate_size
                            for batch_idx in range(num_batches):
                                start_idx = batch_idx * batch_generate_size
                                end_idx = min((batch_idx + 1) * batch_generate_size, total)
                                batch_difference = end_idx - start_idx
                                generated_nxgraphs = generator.generate_data(batch_difference,
                                                                                    embedding=weighted_node_embedding)
                                generate_feat, generate_edge_index = processGenerateSubgraph(generated_nxgraphs,
                                                                                                    feats[0].size(1))
                                feats.extend(generate_feat)
                                edges_per_subgraph.extend(generate_edge_index)
                        elif difference > 0 and difference < 64:
                            generated_nxgraphs = generator.generate_data(difference, embedding=weighted_node_embedding)
                            generate_feat, generate_edge_index = processGenerateSubgraph(generated_nxgraphs,
                                                                                                feats[0].size(1))
                            feats.extend(generate_feat)
                            edges_per_subgraph.extend(generate_edge_index)
                        new_labels = torch.ones(len(generate_feat), dtype=torch.long)
                        labels = torch.cat((labels, new_labels))
                except Exception as e:
                    print("error:", e)
                    continue
                optimizer.zero_grad()
                for i in range(len(feats)):
                    try:
                        feat = feats[i].to(device)
                        edge_index = edges_per_subgraph[i].to(device)
                        # print("feat:",feat.shape)
                        # print("feat:",feat)
                        # if feat.size(0)==0:
                        #     batch_sample_loss.append(0)
                        #     num_samples += 1
                        #     continue
                        if feat.shape[0] == 0:
                            continue  # 跳过该样本
                        graph = construct_graph(feat, edge_index)
                        # print("graph:", graph.ndata['feature'].shape)
                        if graph.ndata['feature'].shape[0] == 0:
                            continue
                        output = detector(graph)
                        # print(f"Output shape: {output.shape}")
                        if output.shape[0] == 0:
                            continue
                        if i >= len(labels):
                            continue
                        item_label = torch.tensor([labels[i].item()], device=device)
                        # 计算交叉熵损失
                        loss = F.cross_entropy(output.unsqueeze(0), item_label)
                        batch_sample_loss.append(loss.item())
                        epoch_loss += loss.item()
                        num_samples += 1
                        loss.backward()
                        optimizer.step()
                    except Exception as e:
                        print("error:",e)
                        batch_sample_loss.append(0)
                        num_samples += 1
                        continue
                sample_losses[batch_num] = batch_sample_loss
                batch_sample_loss = []
                batch_num += 1
        avg_error = epoch_loss / num_samples
        if avg_error < best_avg_error:
            best_epoch = epoch
            best_avg_error = avg_error
            torch.save(detector.state_dict(), model_path)
        print(f"Epoch {epoch + 1}: Average Error = {avg_error}")

        # # 验证集验证
        # eval_loss = 0.0
        # num_eval_samples = 0
        # # 开启评估模式
        # detector.eval()
        # for data in tqdm(eval_loader,leave=False):
        #     if data:
        #         feats, labels = data.subFeature, data.sub_label
        #         edges_per_subgraph = data.sub_edge_index
        #         # Iterate over each sample
        #         for i in range(len(feats)):
        #             feat = feats[i].to(device)
        #             edge_index = edges_per_subgraph[i].to(device)
        #             embedding = gae.encode(feat, edge_index)
        #             output = detector(embedding)
        #             # Calculate the loss for the sample
        #             loss = F.cross_entropy(output, labels[i].unsqueeze(0))
        #             eval_loss += loss.item()
        #             num_eval_samples += 1

        # # # Calculate the average loss for the eval loop
        # avg_eval_loss = eval_loss / num_eval_samples
        # print(f"Epoch {epoch+1}: Validation Loss = {avg_eval_loss}")
        # 测试集测试：
    return test_subgraph_anomaly_detector(detector, test_loader, device, model_path=model_path)


# ---------------------------------------------------------------------------
# Batched variant (``--batched``). Same graphs, same skip rule, same per-graph
# forward (dgl.batch + per-graph max readout). Differences vs. the original loop:
# one optimizer step per mini-batch (mean loss) instead of one per sample, and
# BAED samples for a batch are generated in a single call.
# ---------------------------------------------------------------------------

KEEP_ISOLATED = False  # set by --keep-isolated
AUG = {"gen_noise": 0.0, "curriculum": "linear", "alpha": 1.0, "beta": None, "progress": 1.0}  # set by the CLI


def _to_dgl(feat, edge_index):
    """Same rule as ``construct_graph``: graphs without edges are skipped (return None).

    With ``--keep-isolated`` an edgeless ego-graph is kept as its nodes plus self-loops
    (GraphConv rejects zero in-degree nodes).
    """
    if feat.shape[0] == 0:
        return None
    if edge_index.numel() == 0:
        if not KEEP_ISOLATED:
            return None
        g = dgl.add_self_loop(dgl.graph(([], []), num_nodes=feat.shape[0], device=feat.device))
        g.ndata['feature'] = feat
        return g
    n = int(edge_index.max()) + 1
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=n)
    g.ndata['feature'] = feat[torch.unique(edge_index)]
    return g


def _batch(feats, edges, labels):
    """Build one batched DGL graph directly (edge lists concatenated with node offsets).

    Equivalent to ``dgl.batch([_to_dgl(f, e) ...])`` with the same skip rule, but with a
    single host->device copy instead of one small graph per ego-graph.
    """
    fs, es, ys, pos, nn_, ne_ = [], [], [], [], [], []
    off = 0
    for i in range(min(len(feats), len(labels))):
        f, e = feats[i], edges[i]
        n = f.shape[0]
        if n == 0:
            continue
        if e.numel() == 0:
            if not KEEP_ISOLATED:
                continue
            e = torch.arange(n).repeat(2, 1)  # self-loops, as in _to_dgl
        elif int(e.max()) + 1 != n or torch.unique(e).numel() != n:
            continue  # construct_graph would fail on such a graph (never happens for ego-graphs)
        fs.append(f)
        es.append(e + off)
        nn_.append(n)
        ne_.append(e.size(1))
        ys.append(int(labels[i]))
        pos.append(i)
        off += n
    if not fs:
        return None, None, pos
    e = torch.cat(es, 1).to(device)
    g = dgl.graph((e[0], e[1]), num_nodes=off)
    g.set_batch_num_nodes(torch.tensor(nn_, device=device))
    g.set_batch_num_edges(torch.tensor(ne_, device=device))
    x = torch.cat(fs).to(device)
    g.ndata['feature'] = recompute_aggregates(x.float(), e, off) if AGG_MODE == "recompute" else x
    return g, torch.tensor(ys, device=device), pos


AGG_MODE = "orig"  # set by --agg-features
N_LOCAL = 94       # Elliptic: column 0 = timestep, columns 1-93 = local features, 94-165 = aggregated
AGG_DIM = N_LOCAL + 4 * (N_LOCAL - 1) + 1


def recompute_aggregates(x, edge_index, num_nodes):
    """Replace the precomputed 1-hop aggregates of Elliptic (columns 94-165, anonymised and computed on
    the full graph) by statistics over the ego-graph's own edges, so that real (A) and generated
    (A_hat) ego-graphs get their aggregates from the same function and the same edge set.

    Per node: [timestep + 93 local features, max | min | mean | std of the 93 local features over the
    1-hop neighbours (undirected, self-loops removed), log(1 + degree)]. No neighbour -> zeros.
    """
    import torch_scatter
    loc = x[:, 1:N_LOCAL]
    e = torch.cat([edge_index, edge_index.flip(0)], 1)
    e = e[:, e[0] != e[1]]
    e = torch.unique(e, dim=1)
    src, dst = e
    msg = loc[src]
    kw = dict(dim=0, dim_size=num_nodes)
    mx = torch_scatter.scatter(msg, dst, reduce="max", **kw)
    mn = torch_scatter.scatter(msg, dst, reduce="min", **kw)
    mean = torch_scatter.scatter(msg, dst, reduce="mean", **kw)
    sq = torch_scatter.scatter(msg * msg, dst, reduce="mean", **kw)
    std = (sq - mean * mean).clamp(min=0).sqrt()
    deg = torch_scatter.scatter(torch.ones_like(dst, dtype=x.dtype), dst, reduce="sum", **kw)
    return torch.cat([x[:, :N_LOCAL], mx, mn, mean, std, torch.log1p(deg)[:, None]], 1)


def _guidance(gae, feats, edges, labels, losses):
    """Loss-weighted max-pooled GAE embedding of the anomalous ego-graphs (as in the original loop)."""
    idx = torch.where(labels == 1)[0].tolist()
    if not idx:
        return None
    embs, w = [], []
    for j in idx:
        z = gae.encode(feats[j].to(device, dtype=torch.float), edges[j].to(device))
        embs.append(torch.max(z, dim=0).values)
        w.append(losses.get(j, 1.0) if losses else 1.0)
    w = torch.tensor(w, device=device, dtype=torch.float)
    if AUG["curriculum"] == "sigmoid" and losses:
        # paper Eq. 17: w_i = sigmoid(alpha (l_i - beta)) * t/T  (alpha, beta not given in the paper;
        # default beta = mean anomaly loss of the batch). t/T cancels after the normalisation of Eq. 18.
        beta = AUG["beta"] if AUG["beta"] is not None else float(w.mean())
        w = torch.sigmoid(AUG["alpha"] * (w - beta)) * AUG["progress"]
    w = w / w.sum() if w.sum() > 0 else torch.full_like(w, 1.0 / len(w))
    return (torch.stack(embs) * w[:, None]).sum(0)


@torch.no_grad()
def _current_losses(detector, feats, edges, labels):
    """{position in batch: cross-entropy loss} of the current detector on the real ego-graphs."""
    bg, y, pos = _batch(feats, edges, labels)
    if bg is None:
        return {}
    was_training = detector.training
    detector.eval()
    loss = F.cross_entropy(detector(bg).reshape(-1, 2), y, reduction='none')
    detector.train(was_training)
    return dict(zip(pos, loss.cpu().tolist()))


@torch.no_grad()
def test_batched(detector, test_loader, model_path):
    detector.load_state_dict(torch.load(model_path, map_location=device))
    detector.eval()
    ys, probs = [], []
    for data in test_loader:
        if not data:
            continue
        bg, y, _ = _batch(data.subFeature, data.sub_edge_index, data.sub_label)
        if bg is None:
            continue
        out = detector(bg).reshape(-1, 2)
        probs.append(F.softmax(out, dim=1)[:, 1].cpu())
        ys.append(y.cpu())
    y = torch.cat(ys).numpy()
    prob = torch.cat(probs).numpy()
    pred = (prob > 0.5).astype(int)  # == argmax of a 2-class softmax
    m = {'accuracy': float((pred == y).mean()), 'n_test': int(len(y)), 'n_test_anomaly': int(y.sum())}
    if len(np.unique(y)) > 1:
        m.update(auroc=float(roc_auc_score(y, prob)), auprc=float(average_precision_score(y, prob)),
                 f1_macro=float(f1_score(y, pred, average='macro')), f1_anomaly=float(f1_score(y, pred, pos_label=1)),
                 classification_report=classification_report(y, pred, digits=5, target_names=["正常", "异常"]))
    print({k: v for k, v in m.items() if k != 'classification_report'})
    return m


def train_batched(detector, generator, gae, train_loader, test_loader, num_epochs, optimizer, dataset,
                  use_baed, withcondition, run_tag, eval_loader=None):
    model_path = str(paths.ensure(paths.root() / "outputs" / "detect_weights" /
                                  f"{dataset}_{detector.modelName}_{'BAED' if use_baed else 'None'}_{run_tag}.pth"))
    best = float('inf')
    for epoch in range(num_epochs):
        AUG["progress"] = (epoch + 1) / num_epochs
        detector.train()
        tot, cnt = 0.0, 0
        for b, data in enumerate(tqdm(train_loader, leave=False)):
            if not data:
                continue
            feats, edges, labels = list(data.subFeature), list(data.sub_edge_index), data.sub_label
            n_norm, n_anom = int((labels == 0).sum()), int((labels == 1).sum())
            if use_baed and n_anom < n_norm:
                # paper Eq. 17: l_i is the detection loss of the i-th anomaly of THIS batch. The released
                # code looked losses up by (batch number, position) from the previous epoch, but the train
                # loader is shuffled, so those losses belonged to other samples. Here l_i is the loss of the
                # current detector (before this step's update) on the batch's own real ego-graphs.
                # Epoch 0 keeps the average weights (Algorithm 3, line 10).
                emb = _guidance(gae, feats, edges, labels, _current_losses(detector, feats, edges, labels)
                                if epoch else None) if withcondition else None
                diff = n_norm - n_anom
                gfeat, gedge = generator.generate_arrays(diff, feats[0].size(1), embedding=emb)
                if AUG["gen_noise"] > 0:  # paper: synthetic features = original values + N(0, 1)
                    gfeat = [f.float() + AUG["gen_noise"] * torch.randn(f.shape) for f in gfeat]
                feats += gfeat
                edges += gedge
                labels = torch.cat((labels, torch.ones(len(gfeat), dtype=torch.long)))
            bg, y, pos = _batch(feats, edges, labels)
            if bg is None:
                continue
            out = detector(bg).reshape(-1, 2)
            loss_each = F.cross_entropy(out, y, reduction='none')
            loss = loss_each.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += float(loss_each.sum())
            cnt += len(pos)
        avg = tot / max(cnt, 1)
        if avg < best:
            best = avg
            torch.save(detector.state_dict(), model_path)
        print(f"Epoch {epoch + 1}: Average Error = {avg}")
    m = test_batched(detector, test_loader, model_path)
    if eval_loader is not None:
        v = test_batched(detector, eval_loader, model_path)
        m.update({f"val_{k}": v[k] for k in ("auroc", "auprc", "f1_macro") if k in v})
    return m


def build_detector(name, nodes_feats):
    if name == "BWGNN":
        return BWGNN(in_feats=nodes_feats, activation='Sigmoid')
    if name == "BernNet":
        return BernNet(in_feats=nodes_feats, activation='Sigmoid')
    if name == "GCN":
        return GCN(in_feats=nodes_feats)
    if name == "GIN":
        return GIN(in_feats=nodes_feats)
    if name == "GraphSAGE":
        return GraphSAGE(in_feats=nodes_feats)
    raise ValueError(name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chronological_segments(items, k):
    """Split an ordered list into k consecutive, (almost) equal chunks."""
    n = len(items)
    bounds = [round(j * n / k) for j in range(k + 1)]
    return [items[bounds[j]:bounds[j + 1]] for j in range(k)]


def main(argv=None):
    """Step 05 entry point (console script). Returns None so the exit code is 0."""
    run(argv)


def run(argv=None):
    """Step 05 - downstream graph anomaly detection (GCN / BernNet / BWGNN / GraphSAGE / GIN)."""
    import argparse
    import json
    import time

    global device, sample_losses
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--dataset", default="reddit")
    parser.add_argument("--model", default="GraphSAGE", choices=["BWGNN", "BernNet", "GCN", "GIN", "GraphSAGE"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--node-feats", type=int, default=64)
    parser.add_argument("--gae-weight", default=None, help="file name under weights/gae/ (default <dataset>_gae_relu_128.pt)")
    parser.add_argument("--use-baed", action="store_true", help="augment training with BAED-generated anomalous ego-graphs")
    parser.add_argument("--with-condition", action="store_true", help="anomaly-guided generation (guidance embedding)")
    parser.add_argument("--run-name", default="2024-12-06_17-50-36", help="BAED run used when --use-baed")
    parser.add_argument("--checkpoint", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dynamic-segments", type=int, default=0,
                        help="K>0: incremental setting (Table 2). Split the time-ordered ego-graphs into K+1 consecutive "
                             "segments, at step T train on segment T (continuing the model) and test on segment T+1")
    parser.add_argument("--split", default="sequential", choices=["sequential", "random"],
                        help="sequential = original code (no shuffle); random = stratified 70/15/15 with --seed")
    parser.add_argument("--keep-isolated", action="store_true",
                        help="(--batched only) keep ego-graphs without edges instead of skipping them")
    parser.add_argument("--batched", action="store_true",
                        help="batch the ego-graphs of a mini-batch (dgl.batch): one optimizer step per batch, much faster")
    parser.add_argument("--gen-noise", type=float, default=0.0,
                        help="(--batched) add gen_noise * N(0,1) to synthetic node features (paper: 1.0)")
    parser.add_argument("--curriculum", default="linear", choices=["linear", "sigmoid"],
                        help="(--batched) guidance weights: linear = released code (loss / sum), sigmoid = paper Eq. 17")
    parser.add_argument("--cur-alpha", type=float, default=1.0)
    parser.add_argument("--cur-beta", type=float, default=None, help="default: mean anomaly loss of the batch")
    parser.add_argument("--aug-gen", default="baed", choices=["baed", "ggad", "aegis"],
                        help="(--batched, with --use-baed) generator of synthetic anomalies: BAED diffusion, or the "
                             "GGAD / AEGIS baselines adapted to ego-graphs (see baed.aug_generators)")
    parser.add_argument("--agg-features", default="orig", choices=["orig", "recompute"],
                        help="(--batched, Elliptic) orig = the 166 released features; recompute = keep columns 0-93 "
                             "(timestep + local) and replace the precomputed aggregates 94-165 by max/min/mean/std of "
                             "the local features over the ego-graph's own 1-hop neighbours + log(1+degree), for real "
                             f"and generated ego-graphs alike (node features become {AGG_DIM}-dim)")
    parser.add_argument("--data-tag", default="", help="ego-graph file suffix, e.g. _full -> <dataset>_full_init_0_1_dataset.pkl")
    parser.add_argument("--out", default=None, help="write metrics as JSON here")
    cli = parser.parse_args(argv)
    global KEEP_ISOLATED, AGG_MODE
    KEEP_ISOLATED = cli.keep_isolated
    AGG_MODE = cli.agg_features
    if AGG_MODE == "recompute":
        if not cli.batched:
            parser.error("--agg-features recompute requires --batched")
        cli.node_feats = AGG_DIM  # detector input; the guidance encoder and generator keep the raw 166 features
    AUG.update(gen_noise=cli.gen_noise, curriculum=cli.curriculum, alpha=cli.cur_alpha, beta=cli.cur_beta)
    if cli.keep_isolated and not cli.batched:
        parser.error("--keep-isolated requires --batched")
    if cli.aug_gen != "baed" and not (cli.batched and cli.use_baed):
        parser.error("--aug-gen ggad/aegis requires --batched and --use-baed")
    set_seed(cli.seed)
    dataset = cli.dataset
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gae = torch.load(str(paths.gae_weight(cli.gae_weight or f'{dataset}_gae_relu_128.pt')), map_location=device)
    dataset_path = str(paths.graph_file(f"{dataset}{cli.data_tag}_init_0_1_dataset.pkl"))
    if cli.batched:
        from baed.datasets.lean import keep, lean_collate, load_lean

        nx_graphs = load_lean(dataset_path, lambda: load_dataset(dataset_path), paths.root() / "data" / "cache")
        label_of = lambda it: it[2]  # noqa: E731
    else:
        nx_graphs = load_dataset(dataset_path)
        label_of = lambda g: g.graph["sublabel"]  # noqa: E731
    print("数据集样本总数为:", len(nx_graphs))

    generator = None
    if cli.use_baed and cli.aug_gen == "baed":
        gen_args = GenerateDataArgs(run_name=cli.run_name, dataset=dataset, num_samples=64, seed=cli.seed, checkpoint=cli.checkpoint)
        generator = GraphDataGenerator(gen_args)
        set_seed(cli.seed)  # the generator setup consumes RNG state

    detector = build_detector(cli.model, cli.node_feats).to(device)
    optimizer = optim.Adam(detector.parameters(), lr=cli.lr)
    tag = f"{cli.model}_{(cli.aug_gen.upper() if cli.use_baed else 'None')}_s{cli.seed}" + ("_b" if cli.batched else "") + ("_rand" if cli.split == "random" else "") + ("_iso" if cli.keep_isolated else "") + ("_paug" if cli.gen_noise or cli.curriculum != "linear" else "") + (f"_dyn{cli.dynamic_segments}" if cli.dynamic_segments else "") + ("_aggrc" if AGG_MODE == "recompute" else "")
    batch_size, num_workers, pin_memory = cli.batch_size, 0, True
    print("epoch:{};lr:{}".format(cli.epochs, cli.lr))

    def loaders(train_nx, eval_nx, test_nx):
        nonlocal generator
        if cli.batched:
            sets = [[it for it in x if keep(it, cli.keep_isolated)] for x in (train_nx, eval_nx, test_nx)]
            if cli.use_baed and cli.aug_gen != "baed":  # GGAD / AEGIS: fit on the current training set only
                from baed.aug_generators import make_generator
                generator = make_generator(cli.aug_gen, sets[0], seed=cli.seed, device=str(device))
                set_seed(cli.seed)
            print("训练集/验证集/测试集样本数:", [len(x) for x in sets], "train anomalies:", sum(it[2] for it in sets[0]))
            dl = [DataLoader(x, batch_size=batch_size, shuffle=(k == 0), collate_fn=lean_collate) for k, x in enumerate(sets)]
            return sets, dl
        sets = [GraphDataset(preprocessDataset(x)) for x in (train_nx, eval_nx, test_nx)]
        n0 = sum(int(d.sub_label == 0) for d in sets[0].pyg_datas) if hasattr(sets[0], "pyg_datas") else None
        print("训练集/验证集/测试集样本数:", [len(x) for x in sets], "train normal:", n0)
        return sets, create_dataloaders(*sets, batch_size, num_workers, pin_memory, collate_fn)

    def train_fn(train_loader, test_loader, run_tag, eval_loader=None):
        if cli.batched:
            return train_batched(detector, generator, gae, train_loader, test_loader, cli.epochs, optimizer, dataset,
                                 cli.use_baed, cli.with_condition, run_tag, eval_loader=eval_loader)
        return train_subgraph_anomaly_detector(detector, generator, gae, train_loader, None, test_loader, cli.epochs,
                                               optimizer, device, dataset, use_baed=cli.use_baed,
                                               withcondition=cli.with_condition, run_tag=run_tag)

    t0 = time.time()
    results = {"args": vars(cli), "steps": []}
    if cli.dynamic_segments <= 0:
        if cli.split == "random":
            y = [label_of(g) for g in nx_graphs]
            idx = np.arange(len(nx_graphs))
            tr, rest = train_test_split(idx, train_size=0.7, stratify=y, random_state=cli.seed)
            va, te = train_test_split(rest, test_size=0.5, stratify=[y[i] for i in rest], random_state=cli.seed)
            train_nx, eval_nx, test_nx = ([nx_graphs[i] for i in part] for part in (tr, va, te))
        else:
            train_nx, eval_nx, test_nx = split_dataset(nx_graphs)
        (train_set, _, _), (train_loader, eval_loader, test_loader) = loaders(train_nx, eval_nx, test_nx)
        sample_losses = [[] for _ in range(len(train_set) // batch_size + 1)]
        m = train_fn(train_loader, test_loader, tag, eval_loader=eval_loader)
        results["steps"].append(m)
    else:
        segs = chronological_segments(nx_graphs, cli.dynamic_segments + 1)
        for t in range(1, cli.dynamic_segments + 1):
            (train_set, _, _), (train_loader, eval_loader, test_loader) = loaders(segs[t - 1], segs[t][:1], segs[t])
            sample_losses = [[] for _ in range(len(train_set) // batch_size + 1)]
            m = train_fn(train_loader, test_loader, f"{tag}_T{t}")
            m["T"] = t
            results["steps"].append(m)
            print(f"[dynamic] T={t}: auroc={m.get('auroc')} auprc={m.get('auprc')} f1={m.get('f1_macro')}")
    results["elapsed_s"] = time.time() - t0
    if cli.out:
        out = paths.ensure(paths.root() / cli.out) if not os.path.isabs(cli.out) else cli.out
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print("saved:", out)
    return results


if __name__ == "__main__":
    main()
