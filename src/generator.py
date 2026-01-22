import json
import os

import numpy as np


# # 必须在 import numpy 或 torch 之前设置
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SOSDataGenerator:
    def __init__(self, num_vars=4, degree=4):
        """
        初始化 SOS 数据生成器
        :param num_vars: 变量数量 (n)
        :param degree: 多项式最高次数 (2d), 必须是偶数
        """
        if degree % 2 != 0:
            raise ValueError("Degree (2d) must be even.")

        self.n = num_vars
        self.degree = degree
        self.half_degree = degree // 2

        # 1. 构建基向量 Z(x) 的单项式 (次数 <= d)
        self.basis_monomials = self._get_monomials(self.n, self.half_degree)
        self.mask_dim = len(self.basis_monomials)

        # 2. 构建多项式 P(x) 的单项式 (次数 <= 2d)
        self.poly_monomials = self._get_monomials(self.n, self.degree)
        self.coeff_dim = len(self.poly_monomials)

        # 3. 构建映射表: Q_to_P_map
        # 将 Q 矩阵的 (i, j) 映射到 P 系数向量的索引 k
        self.Q_to_P_map = {}
        self.poly_monomials_to_idx = {m: i for i, m in enumerate(self.poly_monomials)}
        self._build_map()

        print(f"Initialized SOS Generator: Vars={self.n}, Degree={self.degree}")
        print(f"  - Basis Dim (Mask): {self.mask_dim}")
        print(f"  - Poly Dim (Coeffs): {self.coeff_dim}")

    def _get_monomials(self, n, d):
        """生成所有变量 n，次数 <= d 的单项式指数元组"""
        monomials = []
        for k in range(d + 1):
            for p in self._partitions(k, n):
                monomials.append(p)
        return sorted(monomials)

    def _partitions(self, total, n):
        """生成和为 total，长度为 n 的非负整数元组"""
        if n == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for p in self._partitions(total - i, n - 1):
                yield (i,) + p

    def _build_map(self):
        """构建 Q矩阵 到 P系数 的索引映射"""
        for i in range(self.mask_dim):
            for j in range(self.mask_dim):
                exp1 = self.basis_monomials[i]
                exp2 = self.basis_monomials[j]
                product_exp = tuple(e1 + e2 for e1, e2 in zip(exp1, exp2))

                if product_exp in self.poly_monomials_to_idx:
                    k = self.poly_monomials_to_idx[product_exp]
                    self.Q_to_P_map[(i, j)] = k
                else:
                    raise ValueError(f"Product monomial {product_exp} not found in poly basis.")

    def _generate_general_psd_matrix(self, dim, rank=None):
        """生成通用的半定正矩阵"""
        if rank is None:
            rank = dim
        H = np.random.randn(dim, rank)
        U, _ = np.linalg.qr(H)
        cond_log = np.random.uniform(0, 4)
        evals = 10 ** np.random.uniform(0, cond_log, size=rank)
        S = np.diag(evals)
        Q = U @ S @ U.T
        global_scale = 10 ** np.random.uniform(-1, 2)
        Q = Q * global_scale
        return Q

    def _Q_to_coeffs(self, Q):
        """将 Q 矩阵累加到系数向量"""
        coeffs = np.zeros(self.coeff_dim)
        for (i, j), k in self.Q_to_P_map.items():
            coeffs[k] += Q[i, j]
        return coeffs

    def _monomial_to_string(self, exponents):
        parts = []
        for i, exp in enumerate(exponents):
            if exp == 0: continue
            var_name = f"x{i}"
            if exp == 1:
                parts.append(var_name)
            else:
                parts.append(f"{var_name}^{exp}")
        return "*".join(parts) if parts else "1"

    def _poly_to_string(self, coeffs):
        parts = []
        indices = np.where(np.abs(coeffs) > 1e-9)[0]
        indices = sorted(indices, key=lambda k: abs(coeffs[k]), reverse=True)
        for k in indices:
            val = coeffs[k]
            mono_str = self._monomial_to_string(self.poly_monomials[k])
            parts.append(f"{val:.4f}*{mono_str}")
        if not parts: return "0"
        return " + ".join(parts)

    # def generate_dataset(self, total_samples=200000, ratio=(1, 1, 1, 3)):
    #     """生成包含 minimal_mask (理论最小基) 的数据集"""

    #     # 计算各类型数量
    #     unit = total_samples // sum(ratio)
    #     count_robust = unit * ratio[0]
    #     count_redundant = unit * ratio[1]
    #     if ratio[3] == 0:
    #         count_minimal = total_samples - (count_robust + count_redundant)
    #     else:
    #         count_minimal = unit * ratio[2]
    #     count_camo_neg = total_samples - (count_robust + count_redundant + count_minimal)

    #     dataset = []
    #     print(f"Generating {total_samples} samples with 1:1:1:3 Ratio...")
    #     print(f"  [Pos] Robust:    {count_robust}")
    #     print(f"  [Pos] Redundant: {count_redundant}")
    #     print(f"  [Pos] Minimal:   {count_minimal}")
    #     print(f"  [Neg] Camouflaged: {count_camo_neg}")
    #     print("-" * 50)

    #     # ==============================================================================
    #     # Type 1: 基础正样本 (Robust Positive)
    #     # ==============================================================================
    #     for _ in range(count_robust):
    #         # 这里的基是满秩生成的，所以 MaskDim 全是核心
    #         Q = self._generate_general_psd_matrix(self.mask_dim, rank=self.mask_dim)
    #         coeffs = self._Q_to_coeffs(Q)

    #         mask = np.ones(self.mask_dim, dtype=int)
    #         minimal_mask = np.ones(self.mask_dim, dtype=int)  # 全是核心

    #         dataset.append({
    #             "type": "robust_positive",
    #             "label": 1,
    #             "mask": mask.tolist(),
    #             "minimal_mask": minimal_mask.tolist(),  # <--- 新增
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     # ==============================================================================
    #     # Type 2: 冗余正样本 (Redundant Positive)
    #     # ==============================================================================
    #     for _ in range(count_redundant):
    #         # 1. 核心基
    #         num_core = np.random.randint(self.n + 1, self.mask_dim // 2 + 2)
    #         core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

    #         # 2. 生成 P(x)
    #         Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
    #         Q_full = np.zeros((self.mask_dim, self.mask_dim))
    #         Q_full[np.ix_(core_indices, core_indices)] = Q_small
    #         coeffs = self._Q_to_coeffs(Q_full)

    #         # 3. 构造 Minimal Mask (这就是 Agent 的终极目标)
    #         minimal_mask = np.zeros(self.mask_dim, dtype=int)
    #         minimal_mask[core_indices] = 1

    #         # 4. 构造实际 Mask (核心 + 随机冗余)
    #         mask = np.zeros(self.mask_dim, dtype=int)
    #         mask[core_indices] = 1  # 先把核心加进去

    #         remaining_indices = [i for i in range(self.mask_dim) if i not in core_indices]
    #         if remaining_indices:
    #             max_redundant = len(remaining_indices)
    #             num_redundant = np.random.randint(0, max_redundant + 1)
    #             if num_redundant > 0:
    #                 redundant_indices = np.random.choice(remaining_indices, num_redundant, replace=False)
    #                 mask[redundant_indices] = 1

    #         dataset.append({
    #             "type": "redundant_positive",
    #             "label": 1,
    #             "mask": mask.tolist(),
    #             "minimal_mask": minimal_mask.tolist(),  # <--- 新增 (仅包含核心)
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     # ==============================================================================
    #     # Type 3: 极简正样本 (Minimal Positive)
    #     # ==============================================================================
    #     for _ in range(count_minimal):
    #         num_core = np.random.randint(self.n + 1, self.mask_dim // 2 + 1)
    #         core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

    #         Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
    #         Q_full = np.zeros((self.mask_dim, self.mask_dim))
    #         Q_full[np.ix_(core_indices, core_indices)] = Q_small
    #         coeffs = self._Q_to_coeffs(Q_full)

    #         # Mask 本身就是 Minimal
    #         mask = np.zeros(self.mask_dim, dtype=int)
    #         mask[core_indices] = 1
    #         minimal_mask = mask.copy()

    #         dataset.append({
    #             "type": "minimal_positive",
    #             "label": 1,
    #             "mask": mask.tolist(),
    #             "minimal_mask": minimal_mask.tolist(),  # <--- 新增
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     # ==============================================================================
    #     # Type 4: 伪装负样本 (Camouflaged Negative)
    #     # ==============================================================================
    #     for _ in range(count_camo_neg):
    #         num_core = np.random.randint(self.n + 2, self.mask_dim // 2 + 3)
    #         core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

    #         Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
    #         Q_full = np.zeros((self.mask_dim, self.mask_dim))
    #         Q_full[np.ix_(core_indices, core_indices)] = Q_small
    #         coeffs = self._Q_to_coeffs(Q_full)

    #         # 虽然这个样本是负的(缺核心)，但我们依然记录生成它所用的“潜在核心”
    #         # 这有助于分析 Agent 是否能识别出“缺失的核心”
    #         minimal_mask = np.zeros(self.mask_dim, dtype=int)
    #         minimal_mask[core_indices] = 1

    #         mask = np.zeros(self.mask_dim, dtype=int)
    #         # 核心破坏
    #         num_keep_core = np.random.randint(1, num_core)
    #         keep_indices = np.random.choice(core_indices, num_keep_core, replace=False)
    #         mask[keep_indices] = 1

    #         # 冗余伪装
    #         remaining_indices = [i for i in range(self.mask_dim) if i not in core_indices]
    #         if remaining_indices:
    #             max_redundant = len(remaining_indices)
    #             num_redundant = np.random.randint(0, max_redundant + 1)
    #             if num_redundant > 0:
    #                 redundant_indices = np.random.choice(remaining_indices, num_redundant, replace=False)
    #                 mask[redundant_indices] = 1

    #         dataset.append({
    #             "type": "camouflaged_negative",
    #             "label": 0,
    #             "mask": mask.tolist(),
    #             "minimal_mask": minimal_mask.tolist(),  # <--- 新增 (潜在的完整核心)
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     np.random.shuffle(dataset)
    #     return dataset

    # def generate_dataset(self, total_samples=200000, ratio=(1, 1, 1, 3)):
    #     """生成包含 minimal_mask (理论最小基) 的数据集"""

    #     # 计算各类型数量
    #     unit = total_samples // sum(ratio)
    #     count_robust = unit * ratio[0]
    #     count_redundant = unit * ratio[1]
    #     if ratio[3] == 0:
    #         count_minimal = total_samples - (count_robust + count_redundant)
    #     else:
    #         count_minimal = unit * ratio[2]
    #     count_camo_neg = total_samples - (count_robust + count_redundant + count_minimal)

    #     dataset = []
    #     print(f"Generating {total_samples} samples with 1:1:1:3 Ratio...")
    #     print(f"  [Pos] Robust:    {count_robust}")
    #     print(f"  [Pos] Redundant: {count_redundant}")
    #     print(f"  [Pos] Minimal:   {count_minimal}")
    #     print(f"  [Neg] Camouflaged: {count_camo_neg}")
    #     print("-" * 50)

    #     # ==============================================================================
    #     # Type 1: 基础正样本 (Robust Positive)
    #     # ==============================================================================
    #     for _ in range(count_robust):
    #         # 这里的基是满秩生成的，所以 MaskDim 全是核心
    #         Q = self._generate_general_psd_matrix(self.mask_dim, rank=self.mask_dim)
    #         coeffs = self._Q_to_coeffs(Q)

    #         mask = np.ones(self.mask_dim, dtype=int)
    #         minimal_mask = np.ones(self.mask_dim, dtype=int)  # 全是核心

    #         dataset.append({
    #             "type": "robust_positive",
    #             "label": 1,
    #             "mask": mask.tolist(),
    #             "minimal_mask": minimal_mask.tolist(),  # <--- 新增
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     # ==============================================================================
    #     # Type 2: 冗余正样本 (Redundant Positive)
    #     # ==============================================================================
    #     for _ in range(count_redundant):
    #         # 1. 核心基
    #         # === [修改] 增加边界检查，兼容 5n2d 等低维情况 ===
    #         orig_low = self.n + 1
    #         orig_high = self.mask_dim // 2 + 2

    #         if orig_low < orig_high:
    #             # [原有逻辑] 保持 4n4d 等正常情况下的随机序列完全一致
    #             num_core = np.random.randint(orig_low, orig_high)
    #         else:
    #             # [备用逻辑] 当 low >= high 时使用安全范围
    #             safe_high = self.mask_dim + 1
    #             safe_low = min(self.n, self.mask_dim)
    #             if safe_low >= safe_high: safe_low = max(1, safe_high - 1)
    #             num_core = np.random.randint(safe_low, safe_high)

    #         # 防止采样数超过总数
    #         num_core = min(num_core, self.mask_dim)
    #         core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

    #         # 2. 生成 P(x)
    #         Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
    #         Q_full = np.zeros((self.mask_dim, self.mask_dim))
    #         Q_full[np.ix_(core_indices, core_indices)] = Q_small
    #         coeffs = self._Q_to_coeffs(Q_full)

    #         # 3. 构造 Minimal Mask (这就是 Agent 的终极目标)
    #         minimal_mask = np.zeros(self.mask_dim, dtype=int)
    #         minimal_mask[core_indices] = 1

    #         # 4. 构造实际 Mask (核心 + 随机冗余)
    #         mask = np.zeros(self.mask_dim, dtype=int)
    #         mask[core_indices] = 1  # 先把核心加进去

    #         remaining_indices = [i for i in range(self.mask_dim) if i not in core_indices]
    #         if remaining_indices:
    #             max_redundant = len(remaining_indices)
    #             num_redundant = np.random.randint(0, max_redundant + 1)
    #             if num_redundant > 0:
    #                 redundant_indices = np.random.choice(remaining_indices, num_redundant, replace=False)
    #                 mask[redundant_indices] = 1

    #         dataset.append({
    #             "type": "redundant_positive",
    #             "label": 1,
    #             "mask": mask.tolist(),
    #             "minimal_mask": minimal_mask.tolist(),  # <--- 新增 (仅包含核心)
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     # ==============================================================================
    #     # Type 3: 极简正样本 (Minimal Positive)
    #     # ==============================================================================
    #     for _ in range(count_minimal):
    #         # === [修改] 增加边界检查 ===
    #         orig_low = self.n + 1
    #         orig_high = self.mask_dim // 2 + 1

    #         if orig_low < orig_high:
    #             # [原有逻辑]
    #             num_core = np.random.randint(orig_low, orig_high)
    #         else:
    #             # [备用逻辑]
    #             safe_high = self.mask_dim + 1
    #             safe_low = min(self.n, self.mask_dim)
    #             if safe_low >= safe_high: safe_low = max(1, safe_high - 1)
    #             num_core = np.random.randint(safe_low, safe_high)

    #         num_core = min(num_core, self.mask_dim)
    #         core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

    #         Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
    #         Q_full = np.zeros((self.mask_dim, self.mask_dim))
    #         Q_full[np.ix_(core_indices, core_indices)] = Q_small
    #         coeffs = self._Q_to_coeffs(Q_full)

    #         # Mask 本身就是 Minimal
    #         mask = np.zeros(self.mask_dim, dtype=int)
    #         mask[core_indices] = 1
    #         minimal_mask = mask.copy()

    #         dataset.append({
    #             "type": "minimal_positive",
    #             "label": 1,
    #             "mask": mask.tolist(),
    #             "minimal_mask": minimal_mask.tolist(),  # <--- 新增
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     # ==============================================================================
    #     # Type 4: 伪装负样本 (Camouflaged Negative)
    #     # ==============================================================================
    #     for _ in range(count_camo_neg):
    #         # === [修改] 增加边界检查 ===
    #         orig_low = self.n + 2
    #         orig_high = self.mask_dim // 2 + 3

    #         if orig_low < orig_high:
    #             # [原有逻辑]
    #             num_core = np.random.randint(orig_low, orig_high)
    #         else:
    #             # [备用逻辑]
    #             safe_high = self.mask_dim + 1
    #             safe_low = min(self.n + 1, self.mask_dim)
    #             if safe_low >= safe_high: safe_low = max(1, safe_high - 1)
    #             num_core = np.random.randint(safe_low, safe_high)

    #         num_core = min(num_core, self.mask_dim)
    #         core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

    #         Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
    #         Q_full = np.zeros((self.mask_dim, self.mask_dim))
    #         Q_full[np.ix_(core_indices, core_indices)] = Q_small
    #         coeffs = self._Q_to_coeffs(Q_full)

    #         # 虽然这个样本是负的(缺核心)，但我们依然记录生成它所用的“潜在核心”
    #         # 这有助于分析 Agent 是否能识别出“缺失的核心”
    #         minimal_mask = np.zeros(self.mask_dim, dtype=int)
    #         minimal_mask[core_indices] = 1

    #         mask = np.zeros(self.mask_dim, dtype=int)
    #         # 核心破坏
    #         num_keep_core = np.random.randint(1, num_core)
    #         keep_indices = np.random.choice(core_indices, num_keep_core, replace=False)
    #         mask[keep_indices] = 1

    #         # 冗余伪装
    #         remaining_indices = [i for i in range(self.mask_dim) if i not in core_indices]
    #         if remaining_indices:
    #             max_redundant = len(remaining_indices)
    #             num_redundant = np.random.randint(0, max_redundant + 1)
    #             if num_redundant > 0:
    #                 redundant_indices = np.random.choice(remaining_indices, num_redundant, replace=False)
    #                 mask[redundant_indices] = 1

    #         dataset.append({
    #             "type": "camouflaged_negative",
    #             "label": 0,
    #             "mask": mask.tolist(),
    #             "minimal_mask": minimal_mask.tolist(),  # <--- 新增 (潜在的完整核心)
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     np.random.shuffle(dataset)
    #     return dataset

    def generate_dataset(self, total_samples=200000, ratio=(1, 1, 1, 3)):
        """生成包含 minimal_mask (理论最小基) 的数据集"""

        # 计算各类型数量
        unit = total_samples // sum(ratio)
        count_robust = unit * ratio[0]
        count_redundant = unit * ratio[1]
        if ratio[3] == 0:
            count_minimal = total_samples - (count_robust + count_redundant)
        else:
            count_minimal = unit * ratio[2]
        count_camo_neg = total_samples - (count_robust + count_redundant + count_minimal)

        dataset = []
        print(f"Generating {total_samples} samples with 1:1:1:3 Ratio...")
        print(f"  [Pos] Robust:    {count_robust}")
        print(f"  [Pos] Redundant: {count_redundant}")
        print(f"  [Pos] Minimal:   {count_minimal}")
        print(f"  [Neg] Camouflaged: {count_camo_neg}")
        print("-" * 50)

        # ==============================================================================
        # Type 1: 基础正样本 (Robust Positive) - 逻辑保持不变
        # ==============================================================================
        for _ in range(count_robust):
            # 这里的基是满秩生成的，所以 MaskDim 全是核心
            Q = self._generate_general_psd_matrix(self.mask_dim, rank=self.mask_dim)
            coeffs = self._Q_to_coeffs(Q)

            mask = np.ones(self.mask_dim, dtype=int)
            minimal_mask = np.ones(self.mask_dim, dtype=int)  # 全是核心

            dataset.append({
                "type": "robust_positive",
                "label": 1,
                "mask": mask.tolist(),
                "minimal_mask": minimal_mask.tolist(),
                "coeffs": coeffs.tolist(),
                "poly_str": self._poly_to_string(coeffs)
            })

        # ==============================================================================
        # Type 2: 冗余正样本 (Redundant Positive) - 修改
        # ==============================================================================
        for _ in range(count_redundant):
            # 1. 核心基
            # === [修改] 增加边界检查，兼容低维情况并增加样本多样性 ===
            orig_low = self.n + 1
            orig_high = self.mask_dim // 2 + 2

            if orig_low < orig_high:
                # [原有逻辑] 保持 4n4d 等正常情况下的随机序列完全一致
                num_core = np.random.randint(orig_low, orig_high)
            else:
                # [备用逻辑] 当 low >= high 时使用安全范围
                # 关键修改：下界设为 mask_dim的一半，保证至少有一半的冗余空间，制造更大的 Gap
                safe_high = self.mask_dim + 1
                safe_low = max(1, self.mask_dim // 2)
                
                # 双重保险：防止 safe_low 还是太大
                if safe_low >= safe_high: safe_low = max(1, safe_high - 1)
                
                num_core = np.random.randint(safe_low, safe_high)

            # 防止采样数超过总数
            num_core = min(num_core, self.mask_dim)
            core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

            # 2. 生成 P(x)
            Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
            Q_full = np.zeros((self.mask_dim, self.mask_dim))
            Q_full[np.ix_(core_indices, core_indices)] = Q_small
            coeffs = self._Q_to_coeffs(Q_full)

            # 3. 构造 Minimal Mask (这就是 Agent 的终极目标)
            minimal_mask = np.zeros(self.mask_dim, dtype=int)
            minimal_mask[core_indices] = 1

            # 4. 构造实际 Mask (核心 + 随机冗余)
            mask = np.zeros(self.mask_dim, dtype=int)
            mask[core_indices] = 1  # 先把核心加进去

            remaining_indices = [i for i in range(self.mask_dim) if i not in core_indices]
            if remaining_indices:
                max_redundant = len(remaining_indices)
                num_redundant = np.random.randint(0, max_redundant + 1)
                if num_redundant > 0:
                    redundant_indices = np.random.choice(remaining_indices, num_redundant, replace=False)
                    mask[redundant_indices] = 1

            dataset.append({
                "type": "redundant_positive",
                "label": 1,
                "mask": mask.tolist(),
                "minimal_mask": minimal_mask.tolist(),
                "coeffs": coeffs.tolist(),
                "poly_str": self._poly_to_string(coeffs)
            })

        # ==============================================================================
        # Type 3: 极简正样本 (Minimal Positive) - 修改
        # ==============================================================================
        for _ in range(count_minimal):
            # === [修改] 增加边界检查 ===
            orig_low = self.n + 1
            orig_high = self.mask_dim // 2 + 1

            if orig_low < orig_high:
                # [原有逻辑]
                num_core = np.random.randint(orig_low, orig_high)
            else:
                # [备用逻辑] 
                safe_high = self.mask_dim + 1
                safe_low = max(1, self.mask_dim // 2) # 放宽下界
                
                if safe_low >= safe_high: safe_low = max(1, safe_high - 1)
                num_core = np.random.randint(safe_low, safe_high)

            num_core = min(num_core, self.mask_dim)
            core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

            Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
            Q_full = np.zeros((self.mask_dim, self.mask_dim))
            Q_full[np.ix_(core_indices, core_indices)] = Q_small
            coeffs = self._Q_to_coeffs(Q_full)

            # Mask 本身就是 Minimal
            mask = np.zeros(self.mask_dim, dtype=int)
            mask[core_indices] = 1
            minimal_mask = mask.copy()

            dataset.append({
                "type": "minimal_positive",
                "label": 1,
                "mask": mask.tolist(),
                "minimal_mask": minimal_mask.tolist(),
                "coeffs": coeffs.tolist(),
                "poly_str": self._poly_to_string(coeffs)
            })

        # ==============================================================================
        # Type 4: 伪装负样本 (Camouflaged Negative) - 修改
        # ==============================================================================
        for _ in range(count_camo_neg):
            # === [修改] 增加边界检查 ===
            orig_low = self.n + 2
            orig_high = self.mask_dim // 2 + 3

            if orig_low < orig_high:
                # [原有逻辑]
                num_core = np.random.randint(orig_low, orig_high)
            else:
                # [备用逻辑]
                safe_high = self.mask_dim + 1
                # 负样本稍微多留一点核心，增加迷惑性，但也不要全满
                safe_low = max(2, self.mask_dim // 2) 
                
                if safe_low >= safe_high: safe_low = max(1, safe_high - 1)
                num_core = np.random.randint(safe_low, safe_high)

            num_core = min(num_core, self.mask_dim)
            core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

            Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
            Q_full = np.zeros((self.mask_dim, self.mask_dim))
            Q_full[np.ix_(core_indices, core_indices)] = Q_small
            coeffs = self._Q_to_coeffs(Q_full)

            # 虽然这个样本是负的(缺核心)，但我们依然记录生成它所用的“潜在核心”
            minimal_mask = np.zeros(self.mask_dim, dtype=int)
            minimal_mask[core_indices] = 1

            mask = np.zeros(self.mask_dim, dtype=int)
            # 核心破坏
            num_keep_core = np.random.randint(1, num_core)
            keep_indices = np.random.choice(core_indices, num_keep_core, replace=False)
            mask[keep_indices] = 1

            # 冗余伪装
            remaining_indices = [i for i in range(self.mask_dim) if i not in core_indices]
            if remaining_indices:
                max_redundant = len(remaining_indices)
                num_redundant = np.random.randint(0, max_redundant + 1)
                if num_redundant > 0:
                    redundant_indices = np.random.choice(remaining_indices, num_redundant, replace=False)
                    mask[redundant_indices] = 1

            dataset.append({
                "type": "camouflaged_negative",
                "label": 0,
                "mask": mask.tolist(),
                "minimal_mask": minimal_mask.tolist(),
                "coeffs": coeffs.tolist(),
                "poly_str": self._poly_to_string(coeffs)
            })

        np.random.shuffle(dataset)
        return dataset

    def save_to_json(self, dataset, filename="sos_dataset.json"):
        output = {
            "meta": {
                "num_vars": self.n,
                "degree": self.degree,
                "mask_dim": self.mask_dim,
                "coeff_dim": self.coeff_dim,
                "total_samples": len(dataset)
            },
            "data": dataset
        }
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f"Saved dataset to {filename}")
