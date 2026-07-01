import torch
import numpy as np
import random
import json


class SOSPruningEnvGT:
    def __init__(self, generator, dataset_path, device='cuda', train_data_size=10000):
        self.device = device
        self.generator = generator

        self.coeff_dim = generator.coeff_dim
        self.mask_dim = generator.mask_dim
        self.max_steps_limit = self.mask_dim
        self.target_size = train_data_size

        self.dataset = self._load_dataset(dataset_path)

        self.current_coeffs = None
        self.current_mask = None
        self.gt_mask_tensor = None
        self.current_gt_size = 0
        self.min_feasible_size = 0
        self.steps = 0

        # Record the initial Gap and Budget
        self.initial_gap = 0
        self.process_budget = 5.0
        self.stop_budget = 5.0

    def _load_dataset(self, path):
        print(f"[Env] Loading dataset from {path}...")
        with open(path, 'r') as f:
            data = json.load(f)

        valid_data = []
        for d in data['data']:
            if d['label'] == 1 and 'minimal_mask' in d:
                valid_data.append(d)

        target_size = self.target_size
        if len(valid_data) > target_size:
            valid_data = random.sample(valid_data, target_size)

        print(f"[Env] Loaded {len(valid_data)} samples with Ground Truth.")
        return valid_data

    # === Change:Support resetting with a specified sample_idx (for iterating through tests) ===
    def reset(self, sample_idx=None):
        if sample_idx is not None and 0 <= sample_idx < len(self.dataset):
            sample = self.dataset[sample_idx]
        else:
            sample = random.choice(self.dataset)

        # Initialize state
        raw_coeffs = np.array(sample['coeffs'], dtype=np.float32)
        coeffs_t = torch.from_numpy(raw_coeffs)
        self.current_coeffs = torch.sign(coeffs_t) * torch.log1p(torch.abs(coeffs_t))
        self.current_mask = torch.ones(self.mask_dim, dtype=torch.long)

        # Initialize GT
        self.gt_mask_tensor = torch.tensor(sample['minimal_mask'], dtype=torch.long, device=self.device)
        self.current_gt_size = sum(sample['minimal_mask'])

        self.steps = 0
        self.min_feasible_size = self.mask_dim

        # Compute Initial Gap
        self.initial_gap = self.mask_dim - self.current_gt_size
        if self.initial_gap <= 0: self.initial_gap = 1.0

        return self._get_state()

    def _get_state(self):
        return self.current_coeffs.to(self.device), self.current_mask.to(self.device)

    def _calculate_potential(self, gap):
        # Potential calculation: Budget * (progress)^2
        progress_ratio = 1.0 - (gap / self.initial_gap)
        progress_ratio = max(0.0, min(1.0, progress_ratio))
        return self.process_budget * (progress_ratio ** 2)

    def _check_feasibility_by_gt(self, current_mask_tensor):
        current_mask_tensor = current_mask_tensor.to(self.device)
        is_covered = torch.all((current_mask_tensor >= self.gt_mask_tensor))
        return is_covered.item()

    # === Change:Add the is_eval parameter ===
    def step(self, action_idx, is_eval=False):
        self.steps += 1
        time_limit_reached = (self.steps >= self.max_steps_limit)

        # --- A: STOP action ---
        if action_idx == self.mask_dim:
            is_feasible = self._check_feasibility_by_gt(self.current_mask)
            active_count = self.current_mask.sum().item()
            current_gap = active_count - self.current_gt_size

            done = True

            if not is_feasible:
                reward = -5.0
            else:
                if active_count < self.min_feasible_size:
                    self.min_feasible_size = active_count

                # Normalize the terminal reward
                if self.initial_gap > 0:
                    gap_clamped = max(0, current_gap)
                    completion_ratio = (self.initial_gap - gap_clamped) / self.initial_gap
                else:
                    completion_ratio = 1.0

                reward = self.stop_budget * (completion_ratio ** 2)

            return self._get_state(), reward, done, {}

        # --- B: Pruning action ---
        is_critical = (self.gt_mask_tensor[action_idx] == 1)

        if is_critical:
            # === Pruned a critical term ===

            if is_eval:
                # [Evaluation Mode] Strict mode: stop immediately after pruning a critical term; do not count it as completed
                reward = 0.0  # During testing, reward is not important; the important value is where the Gap stops
                done = True
                # Gap remains at the pre-pruning state (because the term was not removed), representing the true capability boundary of the Agent
            else:
                # [Training Mode] Soft penalty: give a negative score and continue exploration
                reward = -1
                done = False
                if time_limit_reached:
                    done = True
        else:
            # === Correctly pruned a redundant term ===
            prev_active_count = self.current_mask.sum().item()
            prev_gap = prev_active_count - self.current_gt_size

            self.current_mask[action_idx] = 0

            curr_active_count = self.current_mask.sum().item()
            curr_gap = curr_active_count - self.current_gt_size

            if curr_active_count < self.min_feasible_size:
                self.min_feasible_size = curr_active_count

            # Potential-difference reward
            pot_old = self._calculate_potential(prev_gap)
            pot_new = self._calculate_potential(curr_gap)
            reward = pot_new - pot_old

            done = False
            if curr_active_count == 0:
                done = True
            elif time_limit_reached:
                done = True

        return self._get_state(), reward, done, {}
