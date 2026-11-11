import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

#gae的encoder
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', dropout_rate=0.5):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=False)
        
        # Dropout层
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # 设置激活函数
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError("Invalid activation function. Use 'relu', 'sigmoid', or 'tanh'.")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)  # 在第一层后添加dropout
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)
        # 设置激活函数
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError("Invalid activation function. Use 'relu', 'sigmoid', or 'tanh'.")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)  # 使用自定义激活函数
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)





class GINEncoder(torch.nn.Module):
    """GIN encoder for the guidance embedding (paper Sec. 4.3: h_g = GIN(G_i^K)).

    Same interface and output size as ``GCNEncoder`` (two message-passing layers, the last one
    linear), so it is pre-trained by the same GAE link-prediction objective and used through
    ``GAE.encode`` everywhere the GCN encoder was used.
    """

    def __init__(self, in_channels, out_channels, activation='relu', dropout_rate=0.5):
        super().__init__()
        from torch_geometric.nn import GINConv

        hidden = 2 * out_channels
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden)), train_eps=True)
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, out_channels)), train_eps=True)
        self.dropout = torch.nn.Dropout(dropout_rate)
        acts = {'relu': F.relu, 'sigmoid': torch.sigmoid, 'tanh': torch.tanh}
        if activation not in acts:
            raise ValueError("Invalid activation function. Use 'relu', 'sigmoid', or 'tanh'.")
        self.activation = acts[activation]

    def forward(self, x, edge_index):
        x = self.dropout(self.activation(self.conv1(x, edge_index)))
        return self.conv2(x, edge_index)
