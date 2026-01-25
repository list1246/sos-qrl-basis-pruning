import torch
import torch.optim as optim
import numpy as np
import random

from src.model import TwoStreamQNet
from src.buffer import PrioritizedReplayBuffer


class DoubleDQNAgentPER:
    def __init__(self, coeff_dim, mask_dim, device='cuda', lr=1e-4, gamma=0.99, base_dim=256, embed_dim=8, capacity=50000, 
                total_episode=None, batch_size=256):
        self.device = device
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998

        self.policy_net = TwoStreamQNet(coeff_dim, mask_dim, base_dim=base_dim, embed_dim=embed_dim, dropout=0.0).to(device)
        self.target_net = TwoStreamQNet(coeff_dim, mask_dim, base_dim=base_dim, embed_dim=embed_dim, dropout=0.0).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)

        if total_episode is not None:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_episode, eta_min=1e-6)
            print(f"[Agent] LR Scheduler enabled: CosineAnnealingLR (T_max={total_episode})")
        else:
            self.scheduler = None
            print("[Agent] LR Scheduler disabled (Fixed LR)")

        self.memory = PrioritizedReplayBuffer(capacity=capacity, alpha=0.6)
        self.batch_size = batch_size
        self.beta = 0.4
        self.beta_increment = 1e-5
        self.mask_dim = mask_dim

    def select_action(self, coeffs, mask):
        valid_indices = torch.nonzero(mask).squeeze(1).tolist()
        stop_action_idx = self.mask_dim
        available_actions = valid_indices + [stop_action_idx]

        if random.random() < self.epsilon:
            return random.choice(available_actions)

        with torch.no_grad():
            c_in = coeffs.unsqueeze(0)
            m_in = mask.unsqueeze(0)
            q_values = self.policy_net(c_in, m_in).squeeze(0)

            full_mask_penalty = torch.full_like(q_values, -float('inf'))
            if valid_indices:
                full_mask_penalty[valid_indices] = 0
            full_mask_penalty[stop_action_idx] = 0

            masked_q_values = q_values + full_mask_penalty
            return masked_q_values.argmax().item()

    # 存储、学习、更新代码保持不变...
    def store_transition(self, state, action, reward, next_state, done):
        max_p = np.max(self.memory.tree.tree[-self.memory.tree.capacity:])
        if max_p == 0: max_p = 1.0
        trans = (state[0].cpu().numpy(), state[1].cpu().numpy(), action, reward,
                 next_state[0].cpu().numpy(), next_state[1].cpu().numpy(), done)
        self.memory.add(max_p, trans)

    def learn(self):
        if self.memory.tree.data_pointer < self.batch_size and self.memory.tree.data[-1] == 0:
            return
        transitions, idxs, probs = self.memory.sample(self.batch_size)
        weights = np.power(np.array(probs) * self.memory.tree.capacity, -self.beta)
        weights /= weights.max()
        weights_t = torch.tensor(weights, device=self.device, dtype=torch.float32).unsqueeze(1)

        batch_c = torch.tensor(np.array([t[0] for t in transitions]), device=self.device)
        batch_m = torch.tensor(np.array([t[1] for t in transitions]), device=self.device)
        batch_a = torch.tensor(np.array([t[2] for t in transitions]), device=self.device).unsqueeze(1)
        batch_r = torch.tensor(np.array([t[3] for t in transitions]), device=self.device).unsqueeze(1)
        batch_nc = torch.tensor(np.array([t[4] for t in transitions]), device=self.device)
        batch_nm = torch.tensor(np.array([t[5] for t in transitions]), device=self.device)
        batch_done = torch.tensor(np.array([t[6] for t in transitions]), device=self.device).float().unsqueeze(1)

        q_current = self.policy_net(batch_c, batch_m).gather(1, batch_a)

        with torch.no_grad():
            next_q_online = self.policy_net(batch_nc, batch_nm)
            illegal_prune_mask = (batch_nm == 0)
            stop_col = torch.zeros((self.batch_size, 1), device=self.device, dtype=torch.bool)
            illegal_full_mask = torch.cat([illegal_prune_mask, stop_col], dim=1)
            next_q_online[illegal_full_mask] = -float('inf')

            best_actions = next_q_online.argmax(dim=1, keepdim=True)
            q_target_next = self.target_net(batch_nc, batch_nm).gather(1, best_actions)
            q_target = batch_r + (1 - batch_done) * self.gamma * q_target_next

        td_errors = q_target - q_current
        loss = (td_errors.pow(2) * weights_t).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        new_p = np.abs(td_errors.detach().cpu().numpy()).flatten() + 1e-5
        for i in range(self.batch_size):
            self.memory.update(idxs[i], float(new_p[i]))
        self.beta = min(1.0, self.beta + self.beta_increment)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
