# gnn_polymer_predictor.py
import torch
import torch.nn as nn

# -------------------------------
# Manual GCN Layer (sem PyG)
# -------------------------------
class ManualGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # edge_index = [[row...], [col...]]
        row, col = edge_index

        # grau dos nós
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        # normalização tipo GCN
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # agregação manual
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col] * norm.unsqueeze(1))

        # transformação linear
        return self.linear(agg)


# -------------------------------
# Simple GNN (duas camadas GCN)
# -------------------------------
class SimpleGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.gcn1 = ManualGCNLayer(in_channels, hidden_channels)
        self.relu = nn.ReLU()

        self.gcn2 = ManualGCNLayer(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(p=0.2)

        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.relu(x)

        x = self.gcn2(x, edge_index)
        x = self.relu(x)

        x = self.dropout(x)

        return self.linear(x)

