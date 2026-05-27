import gymnasium as gym
from gymnasium import spaces
import torch
from .disjungtif_graph import DisjunctiveGraph

class JSSPEnv(gym.Env):
    def __init__(self, generator):
        super(JSSPEnv, self).__init__()
        self.generator = generator
        self.current_time = 0.0
        self.graph = None
        # Action space disesuaikan dengan total operasi maksimal (misal 1000)
        self.action_space = spaces.Discrete(1000)

    def _get_observations(self):
        # Sekarang mengembalikan objek Data utuh sesuai revisi kita sebelumnya
        return self.graph.get_graph_data()

    def get_legal_actions(self):
        legal_actions = []

        # 1. Identifikasi mesin yang sedang sibuk
        busy_machines = set()
        for node_id in range(self.graph.total_ops):
            if self.graph.node_features[node_id, 1] == 1.0: # is_processing
                # Kita cari mesinnya dari duration_matrix/machine_matrix
                job_id, step = self.graph.node_to_job[node_id]
                m_id = self.graph.machine_matrix[job_id, step]
                busy_machines.add(m_id)

        # 2. Cari node yang siap dikerjakan
        for node_id in range(self.graph.total_ops):
            is_unscheduled = (self.graph.node_features[node_id, 0] == 1.0)

            # Cek mesin untuk node ini
            j_id, s_id = self.graph.node_to_job[node_id]
            m_id = self.graph.machine_matrix[j_id, s_id]
            machine_is_free = (m_id not in busy_machines)

            # Cet Precedence (apakah operasi sebelumnya dalam job yang sama sudah selesai)
            predecessor_done = True
            if s_id > 0:
                prev_node = self.graph.ops_in_job[j_id][s_id - 1]
                if self.graph.node_features[prev_node, 2] != 1.0: # index 2: completed
                    predecessor_done = False

            if is_unscheduled and machine_is_free and predecessor_done:
                legal_actions.append(node_id)

        return legal_actions # Indentasi diperbaiki (di luar loop for)

    def _advance_time(self):
        """Mekanisme lompat waktu (Event-based)"""
        while True:
            legal_actions = self.get_legal_actions()
            done = self._check_if_done()

            # Berhenti jika ada yang bisa dikerjakan atau semua tamat
            if len(legal_actions) > 0 or done:
                break

            # Jika macet (tidak ada legal action tapi belum done), lompat ke finish time terdekat
            active_finish_times = [
                self.graph.finish_times[i]
                for i in range(self.graph.total_ops)
                if self.graph.node_features[i, 1] == 1.0
            ]

            if active_finish_times:
                self.current_time = min(active_finish_times)
                # Tandai semua yang selesai pada waktu tersebut
                for i in range(self.graph.total_ops):
                    if self.graph.node_features[i, 1] == 1.0 and self.graph.finish_times[i] <= self.current_time:
                        # Tambahkan self.current_time sebagai argumen sesuai method di DisjunctiveGraph
                        self.graph.mark_operation_finish(i, self.current_time)

                self.graph.update_graph_state(self.current_time)
            else:
                break

    def step(self, action):
        # 1. Jalankan aksi
        self.graph.mark_operation_start(action, self.current_time)

        # 2. Majukan waktu jika perlu (sampai ada aksi legal baru tersedia)
        self._advance_time()

        # 3. Update kondisi fitur
        self.graph.update_graph_state(self.current_time)

        # 4. REWARD SESUAI PAPER: Negatif dari jumlah "Waiting Jobs"
        waiting_jobs = 0
        for j in range(self.graph.num_jobs):
            # Ambil ID operasi terakhir dari job ke-j
            last_op_id = self.graph.ops_in_job[j][-1]

            # Jika operasi terakhir belum selesai, berarti job ini masih "menunggu/berjalan"
            if self.graph.node_features[last_op_id, 2] != 1.0:
                waiting_jobs += 1

        reward = -float(waiting_jobs)

        done = self._check_if_done()
        obs = self._get_observations()

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None, instance=None):
        super().reset(seed=seed)

        # 1. Jika instance diberikan (untuk Validasi atau Batch Training), gunakan itu.
        # Jika tidak, ambil dari generator (untuk Training biasa).
        if instance is not None:
            selected_instance = instance
        else:
            selected_instance = next(self.generator)

        # 2. Buat Graf Disjungtif baru dari instance terpilih
        self.graph = DisjunctiveGraph(selected_instance)

        # 3. Reset waktu simulasi
        self.current_time = 0.0

        # 4. Ambil observasi awal
        obs = self._get_observations()

        return obs, {}

    def _check_if_done(self):
        # Jika semua node fitur index 2 (Completed) bernilai 1.0, maka selesai
        # Kita gunakan torch.all karena self.graph.node_features adalah tensor
        return torch.all(self.graph.node_features[:, 2] == 1.0).item()

    def generate_instance(self, jumlah):
        return [next(self.generator) for _ in range(jumlah)]