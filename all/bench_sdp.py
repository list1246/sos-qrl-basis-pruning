"""
综合基准测试：对 all/result_real.csv 中24个样本，一站式输出所有基准结果到 all/res.csv

包含以下测试：
1. 各方法剪枝结果对比 — RL / NP / TSSOS / 随机剪枝基线
2. SDP求解时间基准 — 全基 / RL / NP / TSSOS / 随机剪枝 (MOSEK, 中位数)
3. 加速比 — 全基时间 ÷ 各种剪枝后SDP时间
"""

import sys
import os
import csv
import re
import time
import random
import numpy as np
import torch
import sympy
from sympy import symbols, Poly, expand, sympify
from scipy.optimize import linprog
import cvxpy as cp
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.generator import SOSDataGenerator
from src.agent import DoubleDQNAgentPER

MODEL_CONFIG = {
    '2n6d':  {'base_dim': 512,  'embed_dim': 16},
    '2n8d':  {'base_dim': 512,  'embed_dim': 16},
    '2n10d': {'base_dim': 1024, 'embed_dim': 32},
    '3n4d':  {'base_dim': 128,  'embed_dim': 8},
    '3n6d':  {'base_dim': 256,  'embed_dim': 8},
    '3n8d':  {'base_dim': 1024, 'embed_dim': 32},
    '4n4d':  {'base_dim': 256,  'embed_dim': 8},
    '4n6d':  {'base_dim': 1024, 'embed_dim': 32},
    '5n4d':  {'base_dim': 256,  'embed_dim': 8},
    '6n4d':  {'base_dim': 1024, 'embed_dim': 32},
    '7n4d':  {'base_dim': 4096, 'embed_dim': 256},
    '8n2d':  {'base_dim': 128,  'embed_dim': 8},
}

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def detect_poly_info(expr_str):
    expr = sympify(expr_str)
    expr = expand(expr)
    free_syms = expr.free_symbols
    max_var_idx = 0
    for s in free_syms:
        name = str(s)
        m = re.match(r'^x(\d+)$', name)
        if m:
            max_var_idx = max(max_var_idx, int(m.group(1)))
    num_vars = max_var_idx
    if num_vars == 0:
        return None, None
    var_symbols = symbols([f'x{i+1}' for i in range(num_vars)])
    if num_vars == 1:
        var_symbols = (var_symbols,)
    poly = Poly(expr, *var_symbols)
    total_deg = poly.total_degree()
    if total_deg % 2 != 0:
        total_deg += 1
    return num_vars, total_deg


def parse_sympy_expr(expr_str, num_vars):
    var_names = [f'x{i+1}' for i in range(num_vars)]
    var_symbols = symbols(var_names)
    expr = sympify(expr_str)
    expr = expand(expr)
    poly = Poly(expr, *var_symbols)
    terms = {}
    for monom, coeff in poly.as_dict().items():
        terms[monom] = float(coeff)
    return terms


def is_in_convex_hull(point, hull_points):
    n_points = hull_points.shape[0]
    c = np.zeros(n_points)
    A_eq = np.vstack([hull_points.T, np.ones(n_points)])
    b_eq = np.concatenate([point, [1]])
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method='highs')
    return res.success


def newton_polytope_prune_indices(generator, support_exponents):
    """NP剪枝，返回保留的基索引列表"""
    S = np.array(support_exponents, dtype=float)
    basis_monomials = np.array(generator.basis_monomials)
    indices = []
    for idx, beta in enumerate(basis_monomials):
        target = 2.0 * beta
        if is_in_convex_hull(target, S):
            indices.append(idx)
    return indices


def tssos_prune_indices(basis_monomials, support_set):
    """TSSOS项稀疏性剪枝，返回保留的基索引列表"""
    support = set(support_set)
    indices = []
    for i, bi in enumerate(basis_monomials):
        keep = False
        for bj in basis_monomials:
            prod = tuple(a + b for a, b in zip(bi, bj))
            if prod in support:
                keep = True
                break
        if keep:
            indices.append(i)
    return indices


def build_sdp_problem(generator, raw_coeffs, active_indices):
    """构建SDP问题，返回 cvxpy Problem 对象（不求解）"""
    k = len(active_indices)
    if k == 0:
        return None

    coeff_to_pairs = defaultdict(list)
    for local_i, global_i in enumerate(active_indices):
        for local_j, global_j in enumerate(active_indices):
            if (global_i, global_j) in generator.Q_to_P_map:
                coeff_k = generator.Q_to_P_map[(global_i, global_j)]
                coeff_to_pairs[coeff_k].append((local_i, local_j))

    for idx in range(generator.coeff_dim):
        if abs(raw_coeffs[idx]) > 1e-10 and idx not in coeff_to_pairs:
            return None

    Q = cp.Variable((k, k), symmetric=True)
    constraints = [Q >> 0]

    for idx in range(generator.coeff_dim):
        target_val = float(raw_coeffs[idx])
        if idx in coeff_to_pairs:
            expr = sum(Q[i, j] for i, j in coeff_to_pairs[idx])
            constraints.append(expr == target_val)
        elif abs(target_val) > 1e-10:
            return None

    return cp.Problem(cp.Minimize(0), constraints)


def solve_sdp_timed(generator, raw_coeffs, active_indices, n_runs=3):
    """用MOSEK求解SDP，跑n_runs次取中位数，返回 (是否可行, 求解时间秒)"""
    prob = build_sdp_problem(generator, raw_coeffs, active_indices)
    if prob is None:
        return False, -1.0

    times = []
    feasible = False
    for _ in range(n_runs):
        # 每次需要重建problem，因为solve会修改内部状态
        prob = build_sdp_problem(generator, raw_coeffs, active_indices)
        try:
            t0 = time.perf_counter()
            prob.solve(solver=cp.MOSEK, verbose=False)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            if prob.status in ('optimal', 'optimal_inaccurate'):
                feasible = True
        except Exception as e:
            print(f"    MOSEK异常: {e}")
            return False, -1.0

    median_time = sorted(times)[len(times) // 2]
    return feasible, median_time


def verify_sdp_quick(generator, raw_coeffs, active_indices):
    """快速SDP验证（用SCS），用于RL SDP-in-the-loop"""
    k = len(active_indices)
    if k == 0:
        return False

    coeff_to_pairs = defaultdict(list)
    for local_i, global_i in enumerate(active_indices):
        for local_j, global_j in enumerate(active_indices):
            if (global_i, global_j) in generator.Q_to_P_map:
                coeff_k = generator.Q_to_P_map[(global_i, global_j)]
                coeff_to_pairs[coeff_k].append((local_i, local_j))

    for idx in range(generator.coeff_dim):
        if abs(raw_coeffs[idx]) > 1e-10 and idx not in coeff_to_pairs:
            return False

    Q = cp.Variable((k, k), symmetric=True)
    constraints = [Q >> 0]
    for idx in range(generator.coeff_dim):
        target_val = float(raw_coeffs[idx])
        if idx in coeff_to_pairs:
            expr = sum(Q[i, j] for i, j in coeff_to_pairs[idx])
            constraints.append(expr == target_val)
        elif abs(target_val) > 1e-10:
            return False

    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
        if prob.status in ('optimal', 'optimal_inaccurate'):
            return True
    except Exception:
        pass
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        return prob.status in ('optimal', 'optimal_inaccurate')
    except Exception:
        return False


def rl_prune_get_indices(agent, generator, raw_coeffs):
    """SDP-in-the-loop RL推理，返回剪枝后的活跃基索引列表"""
    coeffs_t = torch.from_numpy(raw_coeffs).float().to(DEVICE)
    state_coeffs = torch.sign(coeffs_t) * torch.log1p(torch.abs(coeffs_t))
    state_mask = torch.ones(generator.mask_dim, dtype=torch.long, device=DEVICE)

    while True:
        action = agent.select_action(state_coeffs, state_mask)
        if action == generator.mask_dim:
            break
        state_mask[action] = 0
        active_indices = torch.nonzero(state_mask).squeeze(-1).tolist()
        if isinstance(active_indices, int):
            active_indices = [active_indices]
        if len(active_indices) == 0:
            state_mask[action] = 1
            break
        sdp_ok = verify_sdp_quick(generator, raw_coeffs, active_indices)
        if not sdp_ok:
            state_mask[action] = 1
            break

    return torch.nonzero(state_mask).squeeze(-1).tolist()


# ==================== 随机剪枝基线 ====================

def random_prune_greedy(generator, raw_coeffs, seed=None):
    """
    随机顺序贪婪剪枝基线（与RL推理行为一致）。

    算法：
    1. 以指定 seed 初始化随机数生成器（None 则用系统随机状态）
    2. 生成 [0, mask_dim) 的随机排列
    3. 按排列顺序依次尝试剪除每个基元素：
       - 暂移除，用SCS验证SDP可行性
       - 可行 → 保留移除，继续尝试下一个
       - 不可行 → 恢复该元素，立即 STOP（不再尝试后续元素）
    4. 返回最终保留的基索引列表

    Parameters:
        generator: SOSDataGenerator
        raw_coeffs: 原始系数向量
        seed: 随机种子，None 表示不固定种子

    Returns:
        active_indices: 保留的基索引列表
    """
    rng = random.Random(seed)
    mask_dim = generator.mask_dim
    shuffled_indices = list(range(mask_dim))
    rng.shuffle(shuffled_indices)

    active_mask = np.ones(mask_dim, dtype=bool)

    for idx in shuffled_indices:
        active_mask[idx] = False
        active_indices = np.where(active_mask)[0].tolist()

        if len(active_indices) == 0:
            active_mask[idx] = True
            break

        sdp_ok = verify_sdp_quick(generator, raw_coeffs, active_indices)
        if not sdp_ok:
            # 剪到关键项了 —— 恢复并立即停止（与RL评估模式一致）
            active_mask[idx] = True
            break

    return np.where(active_mask)[0].tolist()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    csv_path = os.path.join(base_dir, "result_real.csv")

    # 读取24个样本
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)

    print(f"读取到 {len(samples)} 个样本")

    generator_cache = {}
    model_cache = {}

    # MOSEK warmup: 首次调用有初始化开销，先跑一个小问题预热
    print("MOSEK预热中...")
    _Q = cp.Variable((3, 3), symmetric=True)
    _prob = cp.Problem(cp.Minimize(0), [_Q >> 0, _Q[0, 0] == 1])
    _prob.solve(solver=cp.MOSEK, verbose=False)
    print("MOSEK预热完成\n")

    results = []
    for i, row in enumerate(samples):
        no = row['序号']
        expr_str = row['多项式'].strip()
        expected_rl = int(row['RL剪枝后'])
        expected_np = int(row['NP剪枝后'])
        expected_tssos = int(row['TSSOS剪枝后'])

        nvars, degree = detect_poly_info(expr_str)
        dir_name = f"{nvars}n{degree}d"
        full_basis = int(row['全基数量'])

        print(f"\n[{i+1}/{len(samples)}] 序号{no}: {dir_name}, 全基={full_basis}")
        print(f"  期望: RL={expected_rl}, NP={expected_np}, TSSOS={expected_tssos}")

        # 获取 generator
        if dir_name not in generator_cache:
            generator_cache[dir_name] = SOSDataGenerator(
                num_vars=nvars, degree=degree
            )
        generator = generator_cache[dir_name]

        # 解析多项式
        terms = parse_sympy_expr(expr_str, nvars)
        raw_coeffs = np.zeros(generator.coeff_dim, dtype=np.float32)
        skip = False
        for mon, coeff in terms.items():
            if mon in generator.poly_monomials_to_idx:
                raw_coeffs[generator.poly_monomials_to_idx[mon]] = coeff
            else:
                print(f"  [跳过] 单项式 {mon} 不在基中")
                skip = True
                break
        if skip:
            continue

        support_exponents = list(terms.keys())

        # === RL 剪枝 ===
        config = MODEL_CONFIG[dir_name]
        model_path = os.path.join(project_dir, dir_name, "model", "train.pth")
        if dir_name not in model_cache:
            agent = DoubleDQNAgentPER(
                generator.coeff_dim, generator.mask_dim,
                device=DEVICE,
                base_dim=config['base_dim'],
                embed_dim=config['embed_dim']
            )
            agent.policy_net.load_state_dict(
                torch.load(model_path, map_location=DEVICE)
            )
            agent.epsilon = 0.0
            agent.policy_net.eval()
            model_cache[dir_name] = agent
        agent = model_cache[dir_name]

        rl_indices = rl_prune_get_indices(agent, generator, raw_coeffs)
        rl_size = len(rl_indices)
        print(f"  RL剪枝: {rl_size} (期望{expected_rl})")

        # === NP 剪枝 ===
        np_indices = newton_polytope_prune_indices(generator, support_exponents)
        np_size = len(np_indices)
        print(f"  NP剪枝: {np_size} (期望{expected_np})")

        # === TSSOS 剪枝 ===
        tssos_indices = tssos_prune_indices(generator.basis_monomials, support_exponents)
        tssos_size = len(tssos_indices)
        print(f"  TSSOS剪枝: {tssos_size} (期望{expected_tssos})")

        # === 随机剪枝基线 ===
        rand_indices = random_prune_greedy(generator, raw_coeffs, None)
        rand_size = len(rand_indices)
        print(f"  随机剪枝: {rand_size}")

        # === 全基 SDP 求解时间 (MOSEK) — 基准对照 ===
        print(f"  正在用MOSEK求解SDP...")

        full_indices = list(range(generator.mask_dim))
        full_ok, full_time = solve_sdp_timed(generator, raw_coeffs, full_indices)
        print(f"    全基 SDP: {'OK' if full_ok else 'FAIL'}, {full_time:.3f}s")

        rl_ok, rl_time = solve_sdp_timed(generator, raw_coeffs, rl_indices)
        print(f"    RL SDP: {'OK' if rl_ok else 'FAIL'}, {rl_time:.3f}s")

        np_ok, np_time = solve_sdp_timed(generator, raw_coeffs, np_indices)
        print(f"    NP SDP: {'OK' if np_ok else 'FAIL'}, {np_time:.3f}s")

        tssos_ok, tssos_time = solve_sdp_timed(generator, raw_coeffs, tssos_indices)
        print(f"    TSSOS SDP: {'OK' if tssos_ok else 'FAIL'}, {tssos_time:.3f}s")

        rand_ok, rand_time = solve_sdp_timed(generator, raw_coeffs, rand_indices)
        print(f"    随机剪枝 SDP: {'OK' if rand_ok else 'FAIL'}, {rand_time:.3f}s")

        # 加速比（全基时间 / 剪枝后时间）
        rl_speedup = full_time / rl_time if (full_ok and rl_ok and rl_time > 0) else 0
        np_speedup = full_time / np_time if (full_ok and np_ok and np_time > 0) else 0
        tssos_speedup = full_time / tssos_time if (full_ok and tssos_ok and tssos_time > 0) else 0
        rand_speedup = full_time / rand_time if (full_ok and rand_ok and rand_time > 0) else 0

        results.append({
            '序号': no,
            '变元数': nvars,
            '次数': degree,
            '全基数量': full_basis,
            'RL剪枝后': rl_size,
            'NP剪枝后': np_size,
            'TSSOS剪枝后': tssos_size,
            '随机剪枝后': rand_size,
            '全基_SDP时间(s)': f"{full_time:.3f}" if full_ok else "N/A",
            'RL_SDP时间(s)': f"{rl_time:.3f}" if rl_ok else "N/A",
            'NP_SDP时间(s)': f"{np_time:.3f}" if np_ok else "N/A",
            'TSSOS_SDP时间(s)': f"{tssos_time:.3f}" if tssos_ok else "N/A",
            '随机剪枝_SDP时间(s)': f"{rand_time:.3f}" if rand_ok else "N/A",
            'RL_加速比': f"{rl_speedup:.2f}x" if rl_speedup > 0 else "N/A",
            'NP_加速比': f"{np_speedup:.2f}x" if np_speedup > 0 else "N/A",
            'TSSOS_加速比': f"{tssos_speedup:.2f}x" if tssos_speedup > 0 else "N/A",
            '随机剪枝_加速比': f"{rand_speedup:.2f}x" if rand_speedup > 0 else "N/A",
        })

    # 写入 res.csv
    out_path = os.path.join(base_dir, "res.csv")
    fieldnames = ['序号', '变元数', '次数', '全基数量',
                  'RL剪枝后', 'NP剪枝后', 'TSSOS剪枝后', '随机剪枝后',
                  '全基_SDP时间(s)', 'RL_SDP时间(s)', 'NP_SDP时间(s)',
                  'TSSOS_SDP时间(s)', '随机剪枝_SDP时间(s)',
                  'RL_加速比', 'NP_加速比', 'TSSOS_加速比', '随机剪枝_加速比']
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # 汇总统计
    print("\n" + "=" * 60)
    print("汇总统计（各方法平均保留基数量）")
    print("=" * 60)
    rl_all = [r['RL剪枝后'] for r in results]
    np_all = [r['NP剪枝后'] for r in results]
    tssos_all = [r['TSSOS剪枝后'] for r in results]
    rand_all = [r['随机剪枝后'] for r in results]
    print(f"  RL平均:      {np.mean(rl_all):.1f}")
    print(f"  NP平均:      {np.mean(np_all):.1f}")
    print(f"  TSSOS平均:   {np.mean(tssos_all):.1f}")
    print(f"  随机剪枝平均: {np.mean(rand_all):.1f}")
    improvement = np.mean([(r - rm) / rm * 100 for r, rm in zip(rl_all, rand_all) if rm > 0])
    print(f"  RL相比随机剪枝平均改进: {improvement:.1f}%")
    rl_better = sum(1 for r, rm in zip(rl_all, rand_all) if r < rm)
    rl_worse = sum(1 for r, rm in zip(rl_all, rand_all) if r > rm)
    rl_tie = sum(1 for r, rm in zip(rl_all, rand_all) if abs(r - rm) < 1e-6)
    print(f"  RL优于随机: {rl_better} 个, RL劣于随机: {rl_worse} 个, 持平: {rl_tie} 个")

    print(f"\n完成！结果已写入 {out_path}")
    print(f"共 {len(results)} 个样本")


if __name__ == "__main__":
    main()
