from src.env import SOSPruningEnvGT

class SOSPruningEnvBonus(SOSPruningEnvGT):
    def __init__(self, generator, dataset_path, device='cuda', train_data_size=10000):
        super().__init__(generator, dataset_path, device, train_data_size)

        # self.stop_budget = 10.0

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
                reward = -0.8
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