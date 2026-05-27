import torch.nn as nn
from torch_geometric.nn import MessagePassing

class JSSP_Conv_Jalur(MessagePassing):
    """Implementasi Agregasi Lokal Terpisah (Poin a)"""
    def __init__(self, in_dim, out_dim):
        # Sigma sesuai rumus agregasi di paper
        super(JSSP_Conv_Jalur, self).__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), # Sesuai ukuran di paper
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x, edge_index):
        # 1. Agregasi tetangga (Message Passing)
        aggr_out = self.propagate(edge_index, x=x)
        # 2. Aplikasi fungsi F (MLP) setelah agregasi
        return self.mlp(aggr_out)