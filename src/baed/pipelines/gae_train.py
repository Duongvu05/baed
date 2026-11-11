from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch_geometric.data import data
import torch_geometric as pyg
import random
from functools import partial
import pandas as pd
#读取数据集
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GAE
import networkx as nx
import pickle as pkl
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
import pickle
import numpy as np
from torch_geometric.nn import GAE
import pandas as pd
import os
from dgl.data.utils import load_graphs, save_graphs
from torch_geometric.utils import subgraph
from baed import paths
import copy

from baed.gae.encoder import GCNEncoder, GINEncoder, VariationalGCNEncoder
from torch_geometric.nn import VGAE



def load_data(filepath):
    data = torch.load(filepath)
    print("data:", data)
    return data

def configure_model(data, out_channels=64, num_features=17,activation='relu',gae_type="GAE"):
    if gae_type=="gae":
        model = GAE(GCNEncoder(num_features, out_channels, activation=activation))
    elif gae_type=="gin":  # paper Sec. 4.3 guidance encoder, same link-prediction pre-training
        model = GAE(GINEncoder(num_features, out_channels, activation=activation))
    elif gae_type=="VGAE":
        model = VGAE(VariationalGCNEncoder(num_features, out_channels, activation=activation))  
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    model = model.to(device)
    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, device, x, train_pos_edge_index, optimizer

def train(model, optimizer, x, train_pos_edge_index):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)

def test(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)



def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Step 02b - pre-train the GAE that produces embedding condition vectors.")
    parser.add_argument("--dataset", default="dgraph")
    parser.add_argument("--out-channels", type=int, default=64)
    parser.add_argument("--num-features", type=int, default=17)
    parser.add_argument("--gae-type", default="VGAE", choices=["gae", "gin", "VGAE"])
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args(argv)
    """
    elliptic:
    out_channels = 256
    num_features = 166
    """
    dataset = args.dataset
    out_channels = args.out_channels
    num_features = args.num_features
    gae_type = args.gae_type
    activation = args.activation
    epochs = args.epochs
    data = load_data(paths.gae_data_dir() / f"{dataset}_gae.pt")
    model, device, x, train_pos_edge_index, optimizer = configure_model(data, out_channels, num_features,activation,gae_type)

    best_auc = 0.0
    best_model = None
    for epoch in range(1, epochs + 1):
        loss = train(model, optimizer, x, train_pos_edge_index)
        auc, ap = test(model, x, train_pos_edge_index, data.test_pos_edge_index, data.test_neg_edge_index)
        val_auc, _ = test(model, x, train_pos_edge_index, data.val_pos_edge_index, data.val_neg_edge_index)
        print('Epoch: {:03d}, val AUC: {:.4f}, test AUC: {:.4f}, AP: {:.4f}'.format(epoch, val_auc, auc, ap))
        # select on validation edges; deepcopy (the released code kept a reference, i.e. the last epoch)
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = copy.deepcopy(model)

    print("Best val AUC:", best_auc)
    out = paths.ensure(paths.gae_weight(f'{dataset}_{gae_type}_{activation}_{out_channels}.pt'))
    torch.save(best_model, out)
    print("saved:", out)
    print("x:", x.size())
    print("train_pos_edge_index:", train_pos_edge_index.size())
    best_model.eval()
    Z = best_model.encode(x, train_pos_edge_index)
    print("Z:", Z.size())


if __name__ == "__main__":
    main()
