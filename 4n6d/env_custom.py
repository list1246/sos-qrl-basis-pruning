from src.env import SOSPruningEnvGT

class SOSPruningEnvBonus(SOSPruningEnvGT):
    def __init__(self, generator, dataset_path, device='cuda', train_data_size=10000):
        super().__init__(generator, dataset_path, device, train_data_size)

        # self.stop_budget = 10.0

    def step(self, action_idx, is_eval=False):
        self.steps += 1
        time_limit_reached = (self.steps >= self.max_steps_limit)

        # --- A: STOP 动作 ---
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

                # 归一化结果奖励
                if self.initial_gap > 0:
                    gap_clamped = max(0, current_gap)
                    completion_ratio = (self.initial_gap - gap_clamped) / self.initial_gap
                else:
                    completion_ratio = 1.0

                reward = self.stop_budget * (completion_ratio ** 2)

            return self._get_state(), reward, done, {}

        # --- B: 剪枝动作 ---
        is_critical = (self.gt_mask_tensor[action_idx] == 1)

        if is_critical:
            # === 剪错关键项 ===

            if is_eval:
                # [Evaluation Mode] 严格模式：剪错立即停止，不算完成
                reward = 0.0  # 测试时 reward 不重要，重要的是 Gap 停在了哪里
                done = True
                # Gap 保持在剪之前的状态（因为没剪下去），这代表了 Agent 的真实能力边界
            else:
                # [Training Mode] 软惩罚：给负分，继续探索
                reward = -0.8
                done = False
                if time_limit_reached:
                    done = True
        else:
            # === 剪对冗余项 ===
            prev_active_count = self.current_mask.sum().item()
            prev_gap = prev_active_count - self.current_gt_size

            self.current_mask[action_idx] = 0

            curr_active_count = self.current_mask.sum().item()
            curr_gap = curr_active_count - self.current_gt_size

            if curr_active_count < self.min_feasible_size:
                self.min_feasible_size = curr_active_count

            # 势能差奖励
            pot_old = self._calculate_potential(prev_gap)
            pot_new = self._calculate_potential(curr_gap)
            reward = pot_new - pot_old

            done = False
            if curr_active_count == 0:
                done = True
            elif time_limit_reached:
                done = True

        return self._get_state(), reward, done, {}