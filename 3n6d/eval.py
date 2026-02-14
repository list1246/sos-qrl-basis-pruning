import sys

sys.path.append("..")

import json
import time
import numpy as np
from scipy.optimize import linprog

from src import *

def is_in_convex_hull(point, hull_points):
    """
    使用线性规划判断点是否在给定点集的凸包内。
    原理：是否存在 lambda_i >= 0 且 sum(lambda_i) = 1, 使得 sum(lambda_i * hull_points_i) = point
    """
    n_points = hull_points.shape[0]
    # 目标函数系数 (我们只关心可行性，设为 0)
    c = np.zeros(n_points)
    # 等式约束: A_eq @ lambda = [point_x, point_y, ..., 1]
    # 最后一行是 sum(lambda) = 1
    A_eq = np.vstack([hull_points.T, np.ones(n_points)])
    b_eq = np.concatenate([point, [1]])
    
    # 使用 highs 方法，速度较快且稳定
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method='highs')
    return res.success

def evaluate_newton_polytope(test_json_path, num_vars, degree):
    # 1. 初始化生成器以获取单项式与指数的映射关系
    gen = SOSDataGenerator(num_vars=num_vars, degree=degree)
    basis_monomials = np.array(gen.basis_monomials) # 形状: (mask_dim, n)
    poly_monomials = np.array(gen.poly_monomials)   # 形状: (coeff_dim, n)
    
    # 2. 加载数据
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
        
        # 3. 提取当前多项式的支持集 S (系数非 0 的项的指数)
        support_indices = np.where(np.abs(coeffs) > 1e-9)[0]
        S = poly_monomials[support_indices]
        
        # 4. 判定基向量过滤
        # 根据 Reznick 定理: beta 属于 1/2 NP(P) <=> 2 * beta 属于 NP(P)
        np_basis_count = 0
        for beta in basis_monomials:
            target = 2 * beta
            if is_in_convex_hull(target, S):
                np_basis_count += 1
        
        end_time = time.time()
        
        # 5. 计算指标
        gap = np_basis_count - gt_size
        gaps.append(gap)
        durations.append(end_time - start_time)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{total_samples}...")

    # 6. 输出结果
    avg_gap = np.mean(gaps)
    avg_time = np.mean(durations)
    
    print("\n" + "="*30)
    print(f"Newton Polytope Baseline Result:")
    print(f"  - Average Gap: {avg_gap:.4f}")
    print(f"  - Average Time per Sample: {avg_time*1000:.2f} ms")
    print(f"  - Total Evaluated: {total_samples}")
    print("="*30)

if __name__ == "__main__":
    # 根据你的 generate.py 配置参数
    evaluate_newton_polytope("./data/test.json", num_vars=3, degree=6)