import json
import os

import numpy as np


# # Must be set before importing numpy or torch
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SOSDataGenerator:
    def __init__(self, num_vars=4, degree=4):
        """
        Initialize the SOS data generator
        :param num_vars: number of variables (n)
        :param degree: maximum polynomial degree (2d), must be even
        """
        if degree % 2 != 0:
            raise ValueError("Degree (2d) must be even.")

        self.n = num_vars
        self.degree = degree
        self.half_degree = degree // 2

        # 1. Build monomials for the basis vector Z(x) (Degree <= d)
        self.basis_monomials = self._get_monomials(self.n, self.half_degree)
        self.mask_dim = len(self.basis_monomials)

        # 2. Build monomials for polynomial P(x) (Degree <= 2d)
        self.poly_monomials = self._get_monomials(self.n, self.degree)
        self.coeff_dim = len(self.poly_monomials)

        # 3. Build the mapping table: Q_to_P_map
        # Map entry (i, j) of Q to index k in the P coefficient vector
        self.Q_to_P_map = {}
        self.poly_monomials_to_idx = {m: i for i, m in enumerate(self.poly_monomials)}
        self._build_map()

        print(f"Initialized SOS Generator: Vars={self.n}, Degree={self.degree}")
        print(f"  - Basis Dim (Mask): {self.mask_dim}")
        print(f"  - Poly Dim (Coeffs): {self.coeff_dim}")

    def _get_monomials(self, n, d):
        """Generate exponent tuples for all monomials in n variables with degree <= d"""
        monomials = []
        for k in range(d + 1):
            for p in self._partitions(k, n):
                monomials.append(p)
        return sorted(monomials)

    def _partitions(self, total, n):
        """Generate nonnegative integer tuples of length n whose sum is total"""
        if n == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for p in self._partitions(total - i, n - 1):
                yield (i,) + p

    def _build_map(self):
        """Build the index mapping from the Q matrix to P coefficients"""
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
        """Generate a generic positive semidefinite matrix"""
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
        """Accumulate the Q matrix into the coefficient vector"""
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
    #     """Generate a dataset containing minimal_mask (the theoretical minimal basis)"""

    #     # Compute the count for each type
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
    #     # Type 1: Robust positive samples (Robust Positive)
    #     # ==============================================================================
    #     for _ in range(count_robust):
    #         # The basis is generated at full rank, so all MaskDim entries are core terms
    #         Q = self._generate_general_psd_matrix(self.mask_dim, rank=self.mask_dim)
    #         coeffs = self._Q_to_coeffs(Q)

    #         mask = np.ones(self.mask_dim, dtype=int)
    #         minimal_mask = np.ones(self.mask_dim, dtype=int)  # all core terms

    #         dataset.append({
    #             "type": "robust_positive",
    #             "label": 1,
    #             "mask": mask.tolist(),
    #             "minimal_mask": minimal_mask.tolist(),  # <--- added
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     # ==============================================================================
    #     # Type 2: Redundant positive samples (Redundant Positive)
    #     # ==============================================================================
    #     for _ in range(count_redundant):
    #         # 1. Core basis
    #         num_core = np.random.randint(self.n + 1, self.mask_dim // 2 + 2)
    #         core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

    #         # 2. Generate P(x)
    #         Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
    #         Q_full = np.zeros((self.mask_dim, self.mask_dim))
    #         Q_full[np.ix_(core_indices, core_indices)] = Q_small
    #         coeffs = self._Q_to_coeffs(Q_full)

    #         # 3. Build the Minimal Mask (the Agent target)
    #         minimal_mask = np.zeros(self.mask_dim, dtype=int)
    #         minimal_mask[core_indices] = 1

    #         # 4. Build the actual Mask (core + random redundancy)
    #         mask = np.zeros(self.mask_dim, dtype=int)
    #         mask[core_indices] = 1  # Add the core terms first

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
    #             "minimal_mask": minimal_mask.tolist(),  # <--- added (core terms only)
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     # ==============================================================================
    #     # Type 3: Minimal positive samples (Minimal Positive)
    #     # ==============================================================================
    #     for _ in range(count_minimal):
    #         num_core = np.random.randint(self.n + 1, self.mask_dim // 2 + 1)
    #         core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

    #         Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
    #         Q_full = np.zeros((self.mask_dim, self.mask_dim))
    #         Q_full[np.ix_(core_indices, core_indices)] = Q_small
    #         coeffs = self._Q_to_coeffs(Q_full)

    #         # The Mask itself is Minimal
    #         mask = np.zeros(self.mask_dim, dtype=int)
    #         mask[core_indices] = 1
    #         minimal_mask = mask.copy()

    #         dataset.append({
    #             "type": "minimal_positive",
    #             "label": 1,
    #             "mask": mask.tolist(),
    #             "minimal_mask": minimal_mask.tolist(),  # <--- added
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     # ==============================================================================
    #     # Type 4: Camouflaged negative samples (Camouflaged Negative)
    #     # ==============================================================================
    #     for _ in range(count_camo_neg):
    #         num_core = np.random.randint(self.n + 2, self.mask_dim // 2 + 3)
    #         core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

    #         Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
    #         Q_full = np.zeros((self.mask_dim, self.mask_dim))
    #         Q_full[np.ix_(core_indices, core_indices)] = Q_small
    #         coeffs = self._Q_to_coeffs(Q_full)

    #         # Although this sample is negative (missing core terms), still record the latent core used to generate it
    #         # This helps analyze whether the Agent can identify the missing core terms
    #         minimal_mask = np.zeros(self.mask_dim, dtype=int)
    #         minimal_mask[core_indices] = 1

    #         mask = np.zeros(self.mask_dim, dtype=int)
    #         # Core corruption
    #         num_keep_core = np.random.randint(1, num_core)
    #         keep_indices = np.random.choice(core_indices, num_keep_core, replace=False)
    #         mask[keep_indices] = 1

    #         # Redundant camouflage
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
    #             "minimal_mask": minimal_mask.tolist(),  # <--- added (latent complete core)
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     np.random.shuffle(dataset)
    #     return dataset

    # def generate_dataset(self, total_samples=200000, ratio=(1, 1, 1, 3)):
    #     """Generate a dataset containing minimal_mask (the theoretical minimal basis)"""

    #     # Compute the count for each type
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
    #     # Type 1: Robust positive samples (Robust Positive)
    #     # ==============================================================================
    #     for _ in range(count_robust):
    #         # The basis is generated at full rank, so all MaskDim entries are core terms
    #         Q = self._generate_general_psd_matrix(self.mask_dim, rank=self.mask_dim)
    #         coeffs = self._Q_to_coeffs(Q)

    #         mask = np.ones(self.mask_dim, dtype=int)
    #         minimal_mask = np.ones(self.mask_dim, dtype=int)  # all core terms

    #         dataset.append({
    #             "type": "robust_positive",
    #             "label": 1,
    #             "mask": mask.tolist(),
    #             "minimal_mask": minimal_mask.tolist(),  # <--- added
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     # ==============================================================================
    #     # Type 2: Redundant positive samples (Redundant Positive)
    #     # ==============================================================================
    #     for _ in range(count_redundant):
    #         # 1. Core basis
    #         # === [Modified] Add boundary checks to support low-dimensional cases such as 5n2d ===
    #         orig_low = self.n + 1
    #         orig_high = self.mask_dim // 2 + 2

    #         if orig_low < orig_high:
    #             # [Original logic] Keep the random sequence exactly the same as in normal cases such as 4n4d
    #             num_core = np.random.randint(orig_low, orig_high)
    #         else:
    #             # [Fallback logic] Use a safe range when low >= high
    #             safe_high = self.mask_dim + 1
    #             safe_low = min(self.n, self.mask_dim)
    #             if safe_low >= safe_high: safe_low = max(1, safe_high - 1)
    #             num_core = np.random.randint(safe_low, safe_high)

    #         # Prevent the sample count from exceeding the total
    #         num_core = min(num_core, self.mask_dim)
    #         core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

    #         # 2. Generate P(x)
    #         Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
    #         Q_full = np.zeros((self.mask_dim, self.mask_dim))
    #         Q_full[np.ix_(core_indices, core_indices)] = Q_small
    #         coeffs = self._Q_to_coeffs(Q_full)

    #         # 3. Build the Minimal Mask (the Agent target)
    #         minimal_mask = np.zeros(self.mask_dim, dtype=int)
    #         minimal_mask[core_indices] = 1

    #         # 4. Build the actual Mask (core + random redundancy)
    #         mask = np.zeros(self.mask_dim, dtype=int)
    #         mask[core_indices] = 1  # Add the core terms first

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
    #             "minimal_mask": minimal_mask.tolist(),  # <--- added (core terms only)
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     # ==============================================================================
    #     # Type 3: Minimal positive samples (Minimal Positive)
    #     # ==============================================================================
    #     for _ in range(count_minimal):
    #         # === [Modified] Add boundary checks ===
    #         orig_low = self.n + 1
    #         orig_high = self.mask_dim // 2 + 1

    #         if orig_low < orig_high:
    #             # [Original logic]
    #             num_core = np.random.randint(orig_low, orig_high)
    #         else:
    #             # [Fallback logic]
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

    #         # The Mask itself is Minimal
    #         mask = np.zeros(self.mask_dim, dtype=int)
    #         mask[core_indices] = 1
    #         minimal_mask = mask.copy()

    #         dataset.append({
    #             "type": "minimal_positive",
    #             "label": 1,
    #             "mask": mask.tolist(),
    #             "minimal_mask": minimal_mask.tolist(),  # <--- added
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     # ==============================================================================
    #     # Type 4: Camouflaged negative samples (Camouflaged Negative)
    #     # ==============================================================================
    #     for _ in range(count_camo_neg):
    #         # === [Modified] Add boundary checks ===
    #         orig_low = self.n + 2
    #         orig_high = self.mask_dim // 2 + 3

    #         if orig_low < orig_high:
    #             # [Original logic]
    #             num_core = np.random.randint(orig_low, orig_high)
    #         else:
    #             # [Fallback logic]
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

    #         # Although this sample is negative (missing core terms), still record the latent core used to generate it
    #         # This helps analyze whether the Agent can identify the missing core terms
    #         minimal_mask = np.zeros(self.mask_dim, dtype=int)
    #         minimal_mask[core_indices] = 1

    #         mask = np.zeros(self.mask_dim, dtype=int)
    #         # Core corruption
    #         num_keep_core = np.random.randint(1, num_core)
    #         keep_indices = np.random.choice(core_indices, num_keep_core, replace=False)
    #         mask[keep_indices] = 1

    #         # Redundant camouflage
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
    #             "minimal_mask": minimal_mask.tolist(),  # <--- added (latent complete core)
    #             "coeffs": coeffs.tolist(),
    #             "poly_str": self._poly_to_string(coeffs)
    #         })

    #     np.random.shuffle(dataset)
    #     return dataset

    def generate_dataset(self, total_samples=200000, ratio=(1, 1, 1, 3)):
        """Generate a dataset containing minimal_mask (the theoretical minimal basis)"""

        # Compute the count for each type
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
        # Type 1: Robust positive samples (Robust Positive) - logic unchanged
        # ==============================================================================
        for _ in range(count_robust):
            # The basis is generated at full rank, so all MaskDim entries are core terms
            Q = self._generate_general_psd_matrix(self.mask_dim, rank=self.mask_dim)
            coeffs = self._Q_to_coeffs(Q)

            mask = np.ones(self.mask_dim, dtype=int)
            minimal_mask = np.ones(self.mask_dim, dtype=int)  # all core terms

            dataset.append({
                "type": "robust_positive",
                "label": 1,
                "mask": mask.tolist(),
                "minimal_mask": minimal_mask.tolist(),
                "coeffs": coeffs.tolist(),
                "poly_str": self._poly_to_string(coeffs)
            })

        # ==============================================================================
        # Type 2: Redundant positive samples (Redundant Positive) - Modified
        # ==============================================================================
        for _ in range(count_redundant):
            # 1. Core basis
            # === [Modified] Add boundary checks to support low-dimensional cases and increase sample diversity ===
            orig_low = self.n + 1
            orig_high = self.mask_dim // 2 + 2

            if orig_low < orig_high:
                # [Original logic] Keep the random sequence exactly the same as in normal cases such as 4n4d
                num_core = np.random.randint(orig_low, orig_high)
            else:
                # [Fallback logic] Use a safe range when low >= high
                # Key change: set the lower bound to half of mask_dim to ensure at least half redundant space and create a larger Gap
                safe_high = self.mask_dim + 1
                safe_low = max(1, self.mask_dim // 2)
                
                # Extra guard: prevent safe_low from still being too large
                if safe_low >= safe_high: safe_low = max(1, safe_high - 1)
                
                num_core = np.random.randint(safe_low, safe_high)

            # Prevent the sample count from exceeding the total
            num_core = min(num_core, self.mask_dim)
            core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

            # 2. Generate P(x)
            Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
            Q_full = np.zeros((self.mask_dim, self.mask_dim))
            Q_full[np.ix_(core_indices, core_indices)] = Q_small
            coeffs = self._Q_to_coeffs(Q_full)

            # 3. Build the Minimal Mask (the Agent target)
            minimal_mask = np.zeros(self.mask_dim, dtype=int)
            minimal_mask[core_indices] = 1

            # 4. Build the actual Mask (core + random redundancy)
            mask = np.zeros(self.mask_dim, dtype=int)
            mask[core_indices] = 1  # Add the core terms first

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
        # Type 3: Minimal positive samples (Minimal Positive) - Modified
        # ==============================================================================
        for _ in range(count_minimal):
            # === [Modified] Add boundary checks ===
            orig_low = self.n + 1
            orig_high = self.mask_dim // 2 + 1

            if orig_low < orig_high:
                # [Original logic]
                num_core = np.random.randint(orig_low, orig_high)
            else:
                # [Fallback logic] 
                safe_high = self.mask_dim + 1
                safe_low = max(1, self.mask_dim // 2) # relax the lower bound
                
                if safe_low >= safe_high: safe_low = max(1, safe_high - 1)
                num_core = np.random.randint(safe_low, safe_high)

            num_core = min(num_core, self.mask_dim)
            core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

            Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
            Q_full = np.zeros((self.mask_dim, self.mask_dim))
            Q_full[np.ix_(core_indices, core_indices)] = Q_small
            coeffs = self._Q_to_coeffs(Q_full)

            # The Mask itself is Minimal
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
        # Type 4: Camouflaged negative samples (Camouflaged Negative) - Modified
        # ==============================================================================
        for _ in range(count_camo_neg):
            # === [Modified] Add boundary checks ===
            orig_low = self.n + 2
            orig_high = self.mask_dim // 2 + 3

            if orig_low < orig_high:
                # [Original logic]
                num_core = np.random.randint(orig_low, orig_high)
            else:
                # [Fallback logic]
                safe_high = self.mask_dim + 1
                # Keep slightly more core terms in negative samples to increase ambiguity, but do not make them full
                safe_low = max(2, self.mask_dim // 2) 
                
                if safe_low >= safe_high: safe_low = max(1, safe_high - 1)
                num_core = np.random.randint(safe_low, safe_high)

            num_core = min(num_core, self.mask_dim)
            core_indices = np.random.choice(self.mask_dim, num_core, replace=False)

            Q_small = self._generate_general_psd_matrix(num_core, rank=num_core)
            Q_full = np.zeros((self.mask_dim, self.mask_dim))
            Q_full[np.ix_(core_indices, core_indices)] = Q_small
            coeffs = self._Q_to_coeffs(Q_full)

            # Although this sample is negative (missing core terms), still record the latent core used to generate it
            minimal_mask = np.zeros(self.mask_dim, dtype=int)
            minimal_mask[core_indices] = 1

            mask = np.zeros(self.mask_dim, dtype=int)
            # Core corruption
            num_keep_core = np.random.randint(1, num_core)
            keep_indices = np.random.choice(core_indices, num_keep_core, replace=False)
            mask[keep_indices] = 1

            # Redundant camouflage
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
        # Ensure directories exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f"Saved dataset to {filename}")
