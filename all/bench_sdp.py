"""
Comprehensive benchmark: run all benchmark results for 24 samples in all/result_real.csv and write them to all/res.csv

Includes the following tests:
1. Pruning result comparison across methods — RL / NP / TSSOS / random-pruning baseline
2. SDP solve-time benchmark — Full basis / RL / NP / TSSOS / Random pruning (MOSEK, median)
3. speedup — full-basis time ÷ SDP time after each pruning method
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
    """NP pruning; return the retained basis index list"""
    S = np.array(support_exponents, dtype=float)
    basis_monomials = np.array(generator.basis_monomials)
    indices = []
    for idx, beta in enumerate(basis_monomials):
        target = 2.0 * beta
        if is_in_convex_hull(target, S):
            indices.append(idx)
    return indices


def tssos_prune_indices(basis_monomials, support_set):
    """TSSOS term-sparsity pruning; return the retained basis index list"""
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
    """Build the SDP problem and return a cvxpy Problem object (without solving)"""
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
    """Solve the SDP with MOSEK, run n_runs times, take the median, and return (feasible, solve time in seconds)"""
    prob = build_sdp_problem(generator, raw_coeffs, active_indices)
    if prob is None:
        return False, -1.0

    times = []
    feasible = False
    for _ in range(n_runs):
        # Rebuild the problem each time because solve mutates internal state
        prob = build_sdp_problem(generator, raw_coeffs, active_indices)
        try:
            t0 = time.perf_counter()
            prob.solve(solver=cp.MOSEK, verbose=False)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            if prob.status in ('optimal', 'optimal_inaccurate'):
                feasible = True
        except Exception as e:
            print(f"    MOSEK exception: {e}")
            return False, -1.0

    median_time = sorted(times)[len(times) // 2]
    return feasible, median_time


def verify_sdp_quick(generator, raw_coeffs, active_indices):
    """Quick SDP verification (with SCS), used for RL SDP-in-the-loop"""
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
    """SDP-in-the-loop RL inference; return the active basis indices after pruning"""
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


# ==================== random-pruning baseline ====================

def random_prune_greedy(generator, raw_coeffs, seed=None):
    """
    Random-order greedy pruning baseline (consistent with RL inference behavior).

    Algorithm:
    1. Initialize the random number generator with the specified seed (None uses the system random state)
    2. Generate a random permutation of [0, mask_dim)
    3. Try pruning each basis element in permutation order:
       - Temporarily remove it and verify SDP feasibility with SCS
       - Feasible -> keep it removed and continue to the next one
       - Infeasible -> restore the element and STOP immediately (do not try later elements)
    4. Return the final retained basis index list

    Parameters:
        generator: SOSDataGenerator
        raw_coeffs: raw coefficient vector
        seed: random seed; None means the seed is not fixed

    Returns:
        active_indices: retained basis index list
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
            # A critical term was pruned: restore it and stop immediately (consistent with RL evaluation mode)
            active_mask[idx] = True
            break

    return np.where(active_mask)[0].tolist()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    csv_path = os.path.join(base_dir, "result_real.csv")

    # Read 24 samples
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)

    print(f"Loaded {len(samples)} samples")

    generator_cache = {}
    model_cache = {}

    # MOSEK warmup: The first call has initialization overhead; run a small warmup problem first
    print("MOSEKwarming up...")
    _Q = cp.Variable((3, 3), symmetric=True)
    _prob = cp.Problem(cp.Minimize(0), [_Q >> 0, _Q[0, 0] == 1])
    _prob.solve(solver=cp.MOSEK, verbose=False)
    print("MOSEKwarmup complete\n")

    results = []
    for i, row in enumerate(samples):
        no = row['SampleID']
        expr_str = row['Polynomial'].strip()
        expected_rl = int(row['RLPrunedSize'])
        expected_np = int(row['NPPrunedSize'])
        expected_tssos = int(row['TSSOSPrunedSize'])

        nvars, degree = detect_poly_info(expr_str)
        dir_name = f"{nvars}n{degree}d"
        full_basis = int(row['FullBasisSize'])

        print(f"\n[{i+1}/{len(samples)}] SampleID{no}: {dir_name}, Full basis={full_basis}")
        print(f"  expected: RL={expected_rl}, NP={expected_np}, TSSOS={expected_tssos}")

        # Get the generator
        if dir_name not in generator_cache:
            generator_cache[dir_name] = SOSDataGenerator(
                num_vars=nvars, degree=degree
            )
        generator = generator_cache[dir_name]

        # Parse the polynomial
        terms = parse_sympy_expr(expr_str, nvars)
        raw_coeffs = np.zeros(generator.coeff_dim, dtype=np.float32)
        skip = False
        for mon, coeff in terms.items():
            if mon in generator.poly_monomials_to_idx:
                raw_coeffs[generator.poly_monomials_to_idx[mon]] = coeff
            else:
                print(f"  [Skip] Monomial {mon} is not in the basis")
                skip = True
                break
        if skip:
            continue

        support_exponents = list(terms.keys())

        # === RL pruning ===
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
        print(f"  RL pruning: {rl_size} (expected {expected_rl})")

        # === NP pruning ===
        np_indices = newton_polytope_prune_indices(generator, support_exponents)
        np_size = len(np_indices)
        print(f"  NP pruning: {np_size} (expected {expected_np})")

        # === TSSOS pruning ===
        tssos_indices = tssos_prune_indices(generator.basis_monomials, support_exponents)
        tssos_size = len(tssos_indices)
        print(f"  TSSOS pruning: {tssos_size} (expected {expected_tssos})")

        # === random-pruning baseline ===
        rand_indices = random_prune_greedy(generator, raw_coeffs, None)
        rand_size = len(rand_indices)
        print(f"  Random pruning: {rand_size}")

        # === Full-basis SDP solve time (MOSEK) — baseline reference ===
        print(f"  Solving SDP with MOSEK...")

        full_indices = list(range(generator.mask_dim))
        full_ok, full_time = solve_sdp_timed(generator, raw_coeffs, full_indices)
        print(f"    Full-basis SDP: {'OK' if full_ok else 'FAIL'}, {full_time:.3f}s")

        rl_ok, rl_time = solve_sdp_timed(generator, raw_coeffs, rl_indices)
        print(f"    RL SDP: {'OK' if rl_ok else 'FAIL'}, {rl_time:.3f}s")

        np_ok, np_time = solve_sdp_timed(generator, raw_coeffs, np_indices)
        print(f"    NP SDP: {'OK' if np_ok else 'FAIL'}, {np_time:.3f}s")

        tssos_ok, tssos_time = solve_sdp_timed(generator, raw_coeffs, tssos_indices)
        print(f"    TSSOS SDP: {'OK' if tssos_ok else 'FAIL'}, {tssos_time:.3f}s")

        rand_ok, rand_time = solve_sdp_timed(generator, raw_coeffs, rand_indices)
        print(f"    Random-pruning SDP: {'OK' if rand_ok else 'FAIL'}, {rand_time:.3f}s")

        # speedup(full-basis time / post-pruning time)
        rl_speedup = full_time / rl_time if (full_ok and rl_ok and rl_time > 0) else 0
        np_speedup = full_time / np_time if (full_ok and np_ok and np_time > 0) else 0
        tssos_speedup = full_time / tssos_time if (full_ok and tssos_ok and tssos_time > 0) else 0
        rand_speedup = full_time / rand_time if (full_ok and rand_ok and rand_time > 0) else 0

        results.append({
            'SampleID': no,
            'NumVars': nvars,
            'Degree': degree,
            'FullBasisSize': full_basis,
            'RLPrunedSize': rl_size,
            'NPPrunedSize': np_size,
            'TSSOSPrunedSize': tssos_size,
            'RandomPrunedSize': rand_size,
            'FullBasis_SDPTime(s)': f"{full_time:.3f}" if full_ok else "N/A",
            'RL_SDPTime(s)': f"{rl_time:.3f}" if rl_ok else "N/A",
            'NP_SDPTime(s)': f"{np_time:.3f}" if np_ok else "N/A",
            'TSSOS_SDPTime(s)': f"{tssos_time:.3f}" if tssos_ok else "N/A",
            'RandomPruning_SDPTime(s)': f"{rand_time:.3f}" if rand_ok else "N/A",
            'RL_Speedup': f"{rl_speedup:.2f}x" if rl_speedup > 0 else "N/A",
            'NP_Speedup': f"{np_speedup:.2f}x" if np_speedup > 0 else "N/A",
            'TSSOS_Speedup': f"{tssos_speedup:.2f}x" if tssos_speedup > 0 else "N/A",
            'RandomPruning_Speedup': f"{rand_speedup:.2f}x" if rand_speedup > 0 else "N/A",
        })

    # Write res.csv
    out_path = os.path.join(base_dir, "res.csv")
    fieldnames = ['SampleID', 'NumVars', 'Degree', 'FullBasisSize',
                  'RLPrunedSize', 'NPPrunedSize', 'TSSOSPrunedSize', 'RandomPrunedSize',
                  'FullBasis_SDPTime(s)', 'RL_SDPTime(s)', 'NP_SDPTime(s)',
                  'TSSOS_SDPTime(s)', 'RandomPruning_SDPTime(s)',
                  'RL_Speedup', 'NP_Speedup', 'TSSOS_Speedup', 'RandomPruning_Speedup']
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary statistics (average retained basis size by method)")
    print("=" * 60)
    rl_all = [r['RLPrunedSize'] for r in results]
    np_all = [r['NPPrunedSize'] for r in results]
    tssos_all = [r['TSSOSPrunedSize'] for r in results]
    rand_all = [r['RandomPrunedSize'] for r in results]
    print(f"  RL average:      {np.mean(rl_all):.1f}")
    print(f"  NP average:      {np.mean(np_all):.1f}")
    print(f"  TSSOS average:   {np.mean(tssos_all):.1f}")
    print(f"  Random pruning average: {np.mean(rand_all):.1f}")
    improvement = np.mean([(r - rm) / rm * 100 for r, rm in zip(rl_all, rand_all) if rm > 0])
    print(f"  RL average improvement over random pruning: {improvement:.1f}%")
    rl_better = sum(1 for r, rm in zip(rl_all, rand_all) if r < rm)
    rl_worse = sum(1 for r, rm in zip(rl_all, rand_all) if r > rm)
    rl_tie = sum(1 for r, rm in zip(rl_all, rand_all) if abs(r - rm) < 1e-6)
    print(f"  RL better than random: {rl_better} samples, RL worse than random: {rl_worse} samples, tie: {rl_tie} samples")

    print(f"\nDone. Results written to {out_path}")
    print(f"Total {len(results)} samples")


if __name__ == "__main__":
    main()
