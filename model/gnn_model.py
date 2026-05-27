import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

from .msg_passing import JSSP_Conv_Jalur

class JSSPGNN(nn.Module):
    def __init__(self, num_node_features=8):
        super(JSSPGNN, self).__init__()

        # (a) Agregasi Lokal Terpisah (Fp, Fs, Fd)
        self.Fp = JSSP_Conv_Jalur(8, 8)
        self.Fs = JSSP_Conv_Jalur(8, 8)
        self.Fd = JSSP_Conv_Jalur(8, 8)

        # (d) Fusi Akhir (Fn) - Menerima 48 dimensi (8*6 kombinasi fitur)
        self.Fn = nn.Sequential(
            nn.Linear(48, 256), # Tetap 256 sesuai paper
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 8) # Kembali ke dimensi embedding 8
        )

        # Layer untuk Policy (Actor) dan Value (Critic)
        self.actor_layer = nn.Linear(8, 1)
        self.critic_layer = nn.Linear(8, 1)

    def forward(self, data):
        # Mengambil data dari objek Data PyG (Hasil dari JSSPEnv)
        x = data.x

        edge_index_p = data.edge_index_p
        edge_index_s = data.edge_index_s
        edge_index_d = data.edge_index_d

        # Penanda untuk operasi yang belum selesai
        not_finished_mask = (x[:, 2] == 0).float().unsqueeze(-1)

       # 1. PASTIKAN BATCH ADALAH 1D (Penting!)
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch.view(-1)
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h_initial = x * not_finished_mask # h(0)v - Fitur statis/awal
        h_current = x * not_finished_mask # h(k-1)v

        # PROSES ITERASI (3 KALI SESUAI RUMUS PAPER)
        for k in range(3):
            # (a) Agregasi Lokal Jalur Terpisah
            out_p = F.relu(self.Fp(h_current, edge_index_p)) * not_finished_mask
            out_s = F.relu(self.Fs(h_current, edge_index_s)) * not_finished_mask
            out_d = F.relu(self.Fd(h_current, edge_index_d)) * not_finished_mask

            # (b) Global Pooling (Status Global Pabrik)
            # Menggunakan batch agar informasi global tidak bercampur antar graf
            sum_v = global_add_pool(h_current, batch)

            # Broadcast kembali informasi global ke setiap node dalam batch yang sama
            out_global = sum_v[batch] * not_finished_mask

            # Debugging untuk memastikan semua 2D
            # print(f"p: {out_p.shape}, s: {out_s.shape}, d: {out_d.shape}, global: {out_global.shape}, cur: {h_current.shape}, ini: {h_initial.shape}")

            # (c) & (d) Fusi Konkat 48 Dimensi
            # Urutan: p, s, d, global, current, initial
            fusi_48 = torch.cat([
                out_p, out_s, out_d, out_global, h_current, h_initial
            ], dim=-1)

            # Update h_current untuk iterasi k berikutnya
            h_current = self.Fn(fusi_48) * not_finished_mask

        # Output Actor: Logits untuk tiap node (untuk pemilihan aksi)
        logits = self.actor_layer(h_current).squeeze(-1)

        # Output Critic: Value tunggal untuk satu state graf
        # Mengambil rata-rata/pool dari semua embedding node untuk representasi graf
        graph_repr = global_add_pool(h_current, batch)
        value = self.critic_layer(graph_repr)

        return logits, value