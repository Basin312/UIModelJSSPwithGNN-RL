from ortools.sat.python import cp_model
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import copy
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.loader import DataLoader
  

class PPO:
    def __init__(self, model,device, lr=2.5e-4, gamma=1.0, K_epochs=4, eps_clip=0.2):
        self.device = device        # untuk penggunaan GPU
        self.gamma = gamma          # Discount factor (sesuai tabel: 1.0)
        self.eps_clip = eps_clip    # Clipping parameter (0.2)
        self.K_epochs = K_epochs    # Epochs (4)
        self.gae_lambda = 0.95      # Lambda GAE
        self.v_coeff = 0.5          # Value coefficient (alpha)
        self.ent_coeff = 0.01       # Entropy coefficient (beta)

        self.policy = model
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Policy Old untuk menghitung rasio r_t(theta)
        self.policy_old = copy.deepcopy(model).to(device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, legal_actions, deterministic=False):
        with torch.no_grad():
            logits, value = self.policy_old(state)

            # 1. Masking Ilegal Actions
            mask = torch.full(logits.shape, -1e9).to(logits.device)
            mask[legal_actions] = 0
            masked_logits = logits + mask

            # 2. Distribusi Probabilitas (PERBAIKAN STABILITAS)
            dist = Categorical(logits=masked_logits)

            # --- PERBAIKAN UTAMA: MODUL EVALUASI VS TRAINING ---
            if deterministic:
                # Saat TESTING: Ambil aksi dengan logit/probabilitas tertinggi (Argmax)
                action = torch.argmax(masked_logits)
            else:
                # Saat TRAINING: Lakukan sampling stokastik untuk eksplorasi
                action = dist.sample()

            action_logprob = dist.log_prob(action)

        return action.item(), action_logprob.item(), value.item(), mask

    def update(self, buffer):
        self.policy.train()
        # --- 1. PERHITUNGAN GAE & TARGET RETURNS ---
        rewards_target = []
        advantages = []
        gae = 0

        # Tambahkan 0 untuk state terminal masa depan
        values_list = buffer.values + [0]

        for i in reversed(range(len(buffer.rewards))):
            # mask=0 jika terminal, mask=1 jika tidak
            mask = 1.0 - buffer.is_terminals[i].float()

            # delta_t = R_t + gamma * V(s_t+1) - V(s_t)
            delta = buffer.rewards[i] + (self.gamma * values_list[i+1] * mask) - values_list[i]

            # A_hat_t = delta_t + (gamma * lambda * A_hat_t+1)
            gae = delta + (self.gamma * self.gae_lambda * gae * mask)

            advantages.insert(0, gae)
            # V_target = A_hat_t + V(s_t)
            rewards_target.insert(0, gae + values_list[i])

        # Konversi ke tensor
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        rewards_target = torch.tensor(rewards_target, dtype=torch.float32).to(self.device)
        old_actions = torch.tensor(buffer.actions).to(self.device)
        old_logprobs = torch.tensor(buffer.log_probs).to(self.device)
        old_masks = torch.cat(buffer.legal_masks, dim=0).to(self.device)

        # Normalisasi Advantage (Penting untuk stabilitas)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Menggunakan DataLoader untuk batching objek Data PyG
        loader = DataLoader(buffer.states, batch_size=len(buffer.states))
        batch_state = next(iter(loader))
        batch_state = batch_state.to(self.device)

        # --- 2. OPTIMASI K-EPOCHS ---
      # --- 2. OPTIMASI K-EPOCHS ---
        for _ in range(self.K_epochs):
            # Evaluasi state dengan policy saat ini
            logits, values = self.policy(batch_state)
            values = values.squeeze(-1) 

            masked_logits = logits + old_masks 
            
            # Jika logits kamu ada dimensi tambahan seperti [Batch_Size, Jumlah_Aksi, 1], sesuaikan view-nya:
            # masked_logits = logits.squeeze(-1) + old_masks
            
            dist = Categorical(logits=masked_logits)

            logprobs = dist.log_prob(old_actions) # old_actions harus berbentuk [Batch_Size]
            dist_entropy = dist.entropy()         # dist_entropy akan berbentuk [Batch_Size]

            # Rasio r_t(theta) = pi_new / pi_old
            ratios = torch.exp(logprobs - old_logprobs)

            # Clipped Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss_actor = -torch.min(surr1, surr2).mean()

            # Value Loss (MSE) - Persamaan (10)
            loss_critic = self.v_coeff * self.MseLoss(values, rewards_target)

            # Entropy Loss (Mendorong Eksplorasi)
            loss_entropy = -self.ent_coeff * dist_entropy.mean()

            # Total Loss
            total_loss = loss_actor + loss_critic + loss_entropy

            # Gradient Descent
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # Update policy_old dengan bobot terbaru
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss_actor.item(), loss_critic.item(), loss_entropy.item()
    def save(self, checkpoint_path):
        """Menyimpan checkpoint model dan state optimizer."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_path)
        print(f"--> Checkpoint berhasil disimpan di: {checkpoint_path}")

    def load(self, checkpoint_path):
        """Memuat checkpoint model dan state optimizer dari file."""
        # Map ke device yang sesuai (bisa CPU atau GPU)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Masukkan bobot ke policy utama dan policy lama
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.policy_old.load_state_dict(checkpoint['model_state_dict'])

        # Masukkan status optimizer (agar momentum dan learning rate tetap sinkron)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"--> Checkpoint berhasil dimuat dari: {checkpoint_path}")