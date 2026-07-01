import sys

sys.path.append("..")

import json
import time
import numpy as np
from scipy.optimize import linprog

from src import *

def is_in_convex_hull(point, hull_points):
    """
    Use linear programming to determine whether a point lies in the convex hull of a given point set.
    Principle: whether there exist lambda_i >= 0 with sum(lambda_i) = 1 such that sum(lambda_i * hull_points_i) = point
    """
    n_points = hull_points.shape[0]
    # Objective coefficients (only feasibility matters, so set them to 0)
    c = np.zeros(n_points)
    # Equality constraints: A_eq @ lambda = [point_x, point_y, ..., 1]
    # The last row is sum(lambda) = 1
    A_eq = np.vstack([hull_points.T, np.ones(n_points)])
    b_eq = np.concatenate([point, [1]])
    
    # Use the highs method; it is faster and stable
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method='highs')
    return res.success

def evaluate_newton_polytope(test_json_path, num_vars, degree):
    # 1. Initialize the generator to obtain the mapping between monomials and exponents
    gen = SOSDataGenerator(num_vars=num_vars, degree=degree)
    basis_monomials = np.array(gen.basis_monomials) # shape: (mask_dim, n)
    poly_monomials = np.array(gen.poly_monomials)   # shape: (coeff_dim, n)
    
    # 2. Load data
    with open(test_json_path, 'r') as f:
        dataset = json.load(f)['data']
    
    total_samples = len(dataset)
    gaps = []
    durations = []

    print(f"Starting evaluation on {total_samples} samples...")

    for i, sample in enumerate(dataset):
        coeffs = np.array(sample['coeffs'])
        gt_mask = np.array(sample['minimal_mask'])
        gt_size = np.sum(gt_mask)
        
        start_time = time.time()
        
        # 3. Extract the support set S of the current polynomial (exponents of terms with nonzero coefficients)
        support_indices = np.where(np.abs(coeffs) > 1e-9)[0]
        S = poly_monomials[support_indices]
        
        # 4. Filter basis vectors
        # By Reznick theorem: beta belongs to 1/2 NP(P) <=> 2 * beta belongs to NP(P)
        np_basis_count = 0
        for beta in basis_monomials:
            target = 2 * beta
            if is_in_convex_hull(target, S):
                np_basis_count += 1
        
        end_time = time.time()
        
        # 5. Compute metrics
        gap = np_basis_count - gt_size
        gaps.append(gap)
        durations.append(end_time - start_time)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{total_samples}...")

    # 6. Output results
    avg_gap = np.mean(gaps)
    avg_time = np.mean(durations)
    
    print("\n" + "="*30)
    print(f"Newton Polytope Baseline Result:")
    print(f"  - Average Gap: {avg_gap:.4f}")
    print(f"  - Average Time per Sample: {avg_time*1000:.2f} ms")
    print(f"  - Total Evaluated: {total_samples}")
    print("="*30)

if __name__ == "__main__":
    # Based on the configuration parameters in generate.py
    evaluate_newton_polytope("./data/test.json", num_vars=4, degree=6)