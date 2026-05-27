import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

class DisjunctiveGraph:
    def __init__(self, instance):
        # 1. Ambil data dari instance
        self.num_jobs = instance.num_jobs
        self.num_machines = instance.num_machines
        self.total_ops = instance.num_operations

        # Mapping untuk mempermudah pelacakan (Node ID -> Job ID, Step)
        self.node_to_job = {}

        self.duration_matrix = np.array(instance.duration_matrix)
        self.machine_matrix = np.array(instance.machines_matrix)

        # 2. Konstanta Normalisasi
        self.max_duration = instance.max_duration if instance.max_duration > 0 else 1.0

        # 3. Fitur Node (Tensor 8 dimensi)
        self.node_features = torch.zeros((self.total_ops, 8), dtype=torch.float)

        # 4. State Pelacakan Waktu
        self.arrival_times = np.full(self.total_ops, -1.0)
        self.start_times = np.full(self.total_ops, -1.0)
        self.finish_times = np.full(self.total_ops, -1.0)

        # 5. Struktur Graf
        self.nx_graph = nx.DiGraph()
        self.ops_in_job = {j: [] for j in range(self.num_jobs)}
        self.ops_in_machine = {m: [] for m in range(self.num_machines)}

        self._build_graph()

    def _build_graph(self):
        global_id = 0
        for j in range(self.num_jobs):
            total_job_time = np.sum(self.duration_matrix[j])

            for step in range(self.num_machines):
                duration = self.duration_matrix[j, step]
                machine_id = self.machine_matrix[j, step]

                # Simpan mapping
                self.node_to_job[global_id] = (j, step)
                self.ops_in_job[j].append(global_id)
                self.ops_in_machine[machine_id].append(global_id)

                # Inisialisasi 8 Fitur
                self.node_features[global_id, 0] = 1.0  # Unscheduled
                self.node_features[global_id, 3] = float(duration) / self.max_duration
                self.node_features[global_id, 4] = (duration / total_job_time) if total_job_time > 0 else 0
                self.node_features[global_id, 5] = ((self.num_machines - 1) - step) / self.num_machines

                self.nx_graph.add_node(global_id)

                # Conjunctive Edge (Hanya searah di NetworkX)
                if step > 0:
                    self.nx_graph.add_edge(global_id - 1, global_id, type='conjunctive')

                global_id += 1

        # Disjunctive Edges (Hanya satu arah di NetworkX untuk efisiensi)
        for m, ops in self.ops_in_machine.items():
            for i in range(len(ops)):
                for k in range(i + 1, len(ops)):
                    self.nx_graph.add_edge(ops[i], ops[k], type='disjunctive')

        # Waktu kedatangan awal untuk operasi pertama setiap job
        for j in range(self.num_jobs):
            first_op = self.ops_in_job[j][0]
            self.arrival_times[first_op] = 0.0

    def update_graph_state(self, current_time):
        for node_id in range(self.total_ops):
            is_unscheduled = (self.node_features[node_id, 0] == 1.0)
            is_processing = (self.node_features[node_id, 1] == 1.0)

            # (e) Waktu Tunggu
            arrival = self.arrival_times[node_id]
            if is_unscheduled and arrival != -1 and current_time >= arrival:
                wait_duration = current_time - arrival
                self.node_features[node_id, 6] = float(wait_duration) / self.max_duration

            # (f) Sisa Waktu
            if is_processing:
                remaining = max(0, self.finish_times[node_id] - current_time)
                self.node_features[node_id, 7] = float(remaining) / self.max_duration
            else:
                self.node_features[node_id, 7] = 0.0

    def mark_operation_start(self, node_id, current_time):
        # Gunakan durasi dari matriks aslinya
        j, step = self.node_to_job[node_id]
        duration = self.duration_matrix[j, step]

        self.node_features[node_id, 0] = 0.0
        self.node_features[node_id, 1] = 1.0 # Processing
        self.node_features[node_id, 2] = 0.0

        self.start_times[node_id] = current_time
        self.finish_times[node_id] = current_time + duration
        self.node_features[node_id, 7] = float(duration) / self.max_duration

    def mark_operation_finish(self, node_id, current_time):
        self.node_features[node_id, 0] = 0.0
        self.node_features[node_id, 1] = 0.0
        self.node_features[node_id, 2] = 1.0 # Completed
        self.node_features[node_id, 7] = 0.0

        j, step = self.node_to_job[node_id]
        if step < self.num_machines - 1:
            next_node = self.ops_in_job[j][step + 1]
            self.arrival_times[next_node] = current_time

    def get_graph_data(self):
        x = self.node_features.clone()
        edge_p, edge_s, edge_d = [], [], []

        for u, v, data in self.nx_graph.edges(data=True):
            if data['type'] == 'conjunctive':
                edge_p.append([u, v])
                edge_s.append([v, u])
            elif data['type'] == 'disjunctive':
                edge_d.append([u, v])
                edge_d.append([v, u])

        # Fungsi pembantu untuk membuat tensor aman (mengatasi list kosong)
        def to_tensor(edge_list):
            if len(edge_list) > 0:
                return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                return torch.empty((2, 0), dtype=torch.long)

        return Data(
            x=x,
            edge_index_p=to_tensor(edge_p),
            edge_index_s=to_tensor(edge_s),
            edge_index_d=to_tensor(edge_d)
        )

