"""
Solve the reduced SOS Gram matrix Q_RL.

The script reads polynomials from result_real.csv, reproduces the RL-pruned
basis with the trained model, and solves the SDP

    f(x) = z_RL(x).T @ Q_RL @ z_RL(x),  Q_RL >= 0.

Examples:
    python solve_sos_qrl.py --csv result_real.csv --sample-id 3
    python solve_sos_qrl.py --csv result_real.csv --all --outdir qrl_results
    python solve_sos_qrl.py --expr "x1**2 + x2**2" --nvars 2 --degree 2 --full-basis
    python solve_sos_qrl.py --expr "x1**2 + x2**2" --nvars 2 --degree 2 --active-indices 0,2
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cvxpy as cp
import numpy as np
import sympy as sp
import torch

sys.path.append(".")
from src.agent import DoubleDQNAgentPER
from src.generator import SOSDataGenerator


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_CONFIG = {
    "2n6d": {"base_dim": 512, "embed_dim": 16},
    "2n8d": {"base_dim": 512, "embed_dim": 16},
    "2n10d": {"base_dim": 1024, "embed_dim": 32},
    "3n4d": {"base_dim": 128, "embed_dim": 8},
    "3n6d": {"base_dim": 256, "embed_dim": 8},
    "3n8d": {"base_dim": 1024, "embed_dim": 32},
    "4n4d": {"base_dim": 256, "embed_dim": 8},
    "4n6d": {"base_dim": 1024, "embed_dim": 32},
    "5n4d": {"base_dim": 256, "embed_dim": 8},
    "6n4d": {"base_dim": 1024, "embed_dim": 32},
    "7n4d": {"base_dim": 4096, "embed_dim": 256},
    "8n2d": {"base_dim": 128, "embed_dim": 8},
}


@dataclass
class PolynomialCase:
    sample_id: int | None
    expr: str
    nvars: int
    degree: int


@dataclass
class SolveResult:
    case: PolynomialCase
    active_indices: list[int]
    q_matrix: np.ndarray
    status: str
    min_eig: float
    rank: int
    basis_full: list[tuple[int, ...]]
    basis_rl: list[tuple[int, ...]]
    residual_linf: float


def detect_poly_info(expr: str) -> tuple[int, int]:
    parsed = sp.expand(sp.sympify(expr))
    max_idx = 0
    for sym in parsed.free_symbols:
        match = re.fullmatch(r"x(\d+)", str(sym))
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    if max_idx == 0:
        raise ValueError("Cannot detect variables named x1, x2, ...")

    symbols = sp.symbols([f"x{i}" for i in range(1, max_idx + 1)])
    poly = sp.Poly(parsed, *symbols)
    degree = poly.total_degree()
    if degree % 2:
        degree += 1
    return max_idx, degree


def parse_expression_to_coeffs(expr: str, generator: SOSDataGenerator, nvars: int) -> np.ndarray:
    symbols = sp.symbols([f"x{i}" for i in range(1, nvars + 1)])
    poly = sp.Poly(sp.expand(sp.sympify(expr)), *symbols)
    coeffs = np.zeros(generator.coeff_dim, dtype=np.float64)

    for monomial, coeff in poly.as_dict().items():
        idx = generator.poly_monomials_to_idx.get(tuple(monomial))
        if idx is None:
            raise ValueError(f"Monomial {monomial} is outside degree {generator.degree}")
        coeffs[idx] = float(coeff)
    return coeffs


def read_result_csv(path: Path) -> list[PolynomialCase]:
    cases: list[PolynomialCase] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 4:
            raise ValueError(f"{path} does not look like result_real.csv")

        for row in reader:
            if len(row) < 4 or not row[0].strip() or not row[1].strip():
                continue
            sample_id = int(row[0])
            expr = row[1].strip()
            nvars = int(row[2]) if row[2].strip() else detect_poly_info(expr)[0]
            degree = int(row[3]) if row[3].strip() else detect_poly_info(expr)[1]
            if degree % 2:
                degree += 1
            cases.append(PolynomialCase(sample_id, expr, nvars, degree))
    return cases


def coeff_pair_map(generator: SOSDataGenerator, active_indices: list[int]) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for local_i, global_i in enumerate(active_indices):
        for local_j, global_j in enumerate(active_indices):
            coeff_idx = generator.Q_to_P_map[(global_i, global_j)]
            out[coeff_idx].append((local_i, local_j))
    return out


def uncovered_coefficients(
    generator: SOSDataGenerator,
    coeffs: np.ndarray,
    active_indices: list[int],
    tol: float = 1e-10,
) -> list[int]:
    covered = set(coeff_pair_map(generator, active_indices))
    return [i for i, c in enumerate(coeffs) if abs(float(c)) > tol and i not in covered]


def solve_q_gram(
    generator: SOSDataGenerator,
    coeffs: np.ndarray,
    active_indices: list[int],
    solver: str = "CLARABEL",
    verbose: bool = False,
) -> tuple[np.ndarray | None, str]:
    if not active_indices:
        return None, "empty_basis"

    missing = uncovered_coefficients(generator, coeffs, active_indices)
    if missing:
        return None, f"uncovered_coefficients={missing[:10]}"

    pairs_by_coeff = coeff_pair_map(generator, active_indices)
    k = len(active_indices)
    q = cp.Variable((k, k), symmetric=True)
    constraints = [q >> 0]

    for coeff_idx in range(generator.coeff_dim):
        pairs = pairs_by_coeff.get(coeff_idx)
        target = float(coeffs[coeff_idx])
        if pairs:
            constraints.append(sum(q[i, j] for i, j in pairs) == target)
        elif abs(target) > 1e-10:
            return None, f"uncovered_coefficient={coeff_idx}"

    problem = cp.Problem(cp.Minimize(0), constraints)
    installed = set(cp.installed_solvers())
    candidates = [solver, "CLARABEL", "SCS"]
    seen: set[str] = set()

    for candidate in candidates:
        candidate = candidate.upper()
        if candidate in seen or candidate not in installed:
            continue
        seen.add(candidate)
        try:
            options = {"verbose": verbose}
            if candidate == "SCS":
                options.update({"max_iters": 50000, "eps": 1e-7})
            problem.solve(solver=candidate, **options)
        except Exception as exc:
            last_status = f"{candidate}: {exc}"
            continue

        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and q.value is not None:
            q_value = np.asarray(q.value, dtype=np.float64)
            q_value = (q_value + q_value.T) / 2.0
            return q_value, problem.status
        last_status = f"{candidate}: {problem.status}"

    return None, last_status if "last_status" in locals() else "no_solver_available"


def check_reconstruction(generator: SOSDataGenerator, active_indices: list[int], q: np.ndarray) -> np.ndarray:
    coeffs_from_q = np.zeros(generator.coeff_dim, dtype=np.float64)
    for local_i, global_i in enumerate(active_indices):
        for local_j, global_j in enumerate(active_indices):
            coeff_idx = generator.Q_to_P_map[(global_i, global_j)]
            coeffs_from_q[coeff_idx] += q[local_i, local_j]
    return coeffs_from_q


def load_rl_model(config_name: str, generator: SOSDataGenerator) -> DoubleDQNAgentPER:
    config = MODEL_CONFIG.get(config_name)
    if config is None:
        raise ValueError(f"No MODEL_CONFIG entry for {config_name}")

    model_path = Path(config_name) / "model" / "train.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    agent = DoubleDQNAgentPER(
        generator.coeff_dim,
        generator.mask_dim,
        device=DEVICE,
        base_dim=config["base_dim"],
        embed_dim=config["embed_dim"],
    )
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.epsilon = 0.0
    agent.policy_net.eval()
    return agent


def rl_prune_with_sdp(
    agent: DoubleDQNAgentPER,
    generator: SOSDataGenerator,
    coeffs: np.ndarray,
    solver: str,
    verbose: bool = False,
) -> list[int]:
    coeffs_t = torch.from_numpy(coeffs.astype(np.float32)).to(DEVICE)
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

        q, status = solve_q_gram(generator, coeffs, active_indices, solver=solver, verbose=False)
        if q is None:
            state_mask[action] = 1
            if verbose:
                print(f"  rollback prune index {action}: {status}")
            break

    active = torch.nonzero(state_mask).squeeze(-1).tolist()
    if isinstance(active, int):
        active = [active]
    return active


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    if not value.strip():
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def choose_active_indices(
    args: argparse.Namespace,
    case: PolynomialCase,
    generator: SOSDataGenerator,
    coeffs: np.ndarray,
) -> list[int]:
    explicit = parse_int_list(args.active_indices)
    if explicit is not None:
        return explicit

    mask = parse_int_list(args.mask)
    if mask is not None:
        if len(mask) != generator.mask_dim:
            raise ValueError(f"--mask length is {len(mask)}, expected {generator.mask_dim}")
        return [i for i, keep in enumerate(mask) if keep]

    if args.full_basis:
        return list(range(generator.mask_dim))

    config_name = args.model_dir or f"{case.nvars}n{case.degree}d"
    agent = load_rl_model(config_name, generator)
    return rl_prune_with_sdp(agent, generator, coeffs, solver=args.solver, verbose=args.verbose)


def solve_case(args: argparse.Namespace, case: PolynomialCase) -> SolveResult | None:
    generator = SOSDataGenerator(num_vars=case.nvars, degree=case.degree)
    coeffs = parse_expression_to_coeffs(case.expr, generator, case.nvars)
    active_indices = choose_active_indices(args, case, generator, coeffs)

    q, status = solve_q_gram(generator, coeffs, active_indices, solver=args.solver, verbose=args.verbose)
    if q is None:
        print(f"[fail] sample={case.sample_id} n={case.nvars} degree={case.degree}: {status}")
        return None

    eigvals = np.linalg.eigvalsh(q)
    reconstructed = check_reconstruction(generator, active_indices, q)
    residual = float(np.max(np.abs(reconstructed - coeffs)))

    return SolveResult(
        case=case,
        active_indices=active_indices,
        q_matrix=q,
        status=status,
        min_eig=float(eigvals[0]),
        rank=int(np.sum(eigvals > args.rank_tol)),
        basis_full=list(generator.basis_monomials),
        basis_rl=[tuple(generator.basis_monomials[i]) for i in active_indices],
        residual_linf=residual,
    )


def monomial_string(exp: Iterable[int]) -> str:
    parts = []
    for idx, power in enumerate(exp, start=1):
        if power == 0:
            continue
        if power == 1:
            parts.append(f"x{idx}")
        else:
            parts.append(f"x{idx}^{power}")
    return "*".join(parts) if parts else "1"


def write_outputs(result: SolveResult, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sample = result.case.sample_id if result.case.sample_id is not None else "expr"
    stem = f"sample_{sample}_{result.case.nvars}n{result.case.degree}d"

    np.savetxt(outdir / f"{stem}_Q_RL.csv", result.q_matrix, delimiter=",", fmt="%.16g")
    np.savez_compressed(
        outdir / f"{stem}.npz",
        Q_RL=result.q_matrix,
        active_indices=np.array(result.active_indices, dtype=np.int64),
        basis_full=np.array(result.basis_full, dtype=np.int64),
        basis_rl=np.array(result.basis_rl, dtype=np.int64),
        min_eig=np.array(result.min_eig),
        rank=np.array(result.rank),
        residual_linf=np.array(result.residual_linf),
    )

    meta = {
        "sample_id": result.case.sample_id,
        "expr": result.case.expr,
        "nvars": result.case.nvars,
        "degree": result.case.degree,
        "status": result.status,
        "full_basis_size": len(result.basis_full),
        "rl_basis_size": len(result.basis_rl),
        "active_indices": result.active_indices,
        "basis_full_exponents": [list(x) for x in result.basis_full],
        "basis_rl_exponents": [list(x) for x in result.basis_rl],
        "basis_rl_monomials": [monomial_string(x) for x in result.basis_rl],
        "min_eig": result.min_eig,
        "rank": result.rank,
        "residual_linf": result.residual_linf,
        "q_csv": f"{stem}_Q_RL.csv",
        "npz": f"{stem}.npz",
    }
    with (outdir / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def print_summary(result: SolveResult) -> None:
    sample = result.case.sample_id if result.case.sample_id is not None else "expr"
    print(
        f"[ok] sample={sample} config={result.case.nvars}n{result.case.degree}d "
        f"basis={len(result.basis_rl)}/{len(result.basis_full)} "
        f"Q={result.q_matrix.shape[0]}x{result.q_matrix.shape[1]} "
        f"min_eig={result.min_eig:.3e} rank={result.rank} "
        f"residual={result.residual_linf:.3e}"
    )
    print(f"     active_indices={result.active_indices}")
    print(f"     B_RL={{{', '.join(monomial_string(x) for x in result.basis_rl)}}}")


def select_cases(args: argparse.Namespace) -> list[PolynomialCase]:
    if args.expr:
        nvars = args.nvars
        degree = args.degree
        if nvars is None or degree is None:
            detected_nvars, detected_degree = detect_poly_info(args.expr)
            nvars = nvars or detected_nvars
            degree = degree or detected_degree
        return [PolynomialCase(None, args.expr, int(nvars), int(degree))]

    cases = read_result_csv(Path(args.csv))
    if args.sample_id is not None:
        cases = [case for case in cases if case.sample_id == args.sample_id]
        if not cases:
            raise ValueError(f"sample id {args.sample_id} not found in {args.csv}")
    elif not args.all:
        cases = cases[:1]
        print("No --sample-id or --all was given; solving the first CSV row only.")
    return cases


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve reduced SOS Gram matrix Q_RL.")
    parser.add_argument("--csv", default="result_real.csv", help="CSV file with columns: id, polynomial, nvars, degree.")
    parser.add_argument("--sample-id", type=int, help="Only solve one sample id from the CSV.")
    parser.add_argument("--all", action="store_true", help="Solve every row in the CSV.")
    parser.add_argument("--expr", help="Solve a single SymPy expression instead of reading CSV.")
    parser.add_argument("--nvars", type=int, help="Number of variables for --expr.")
    parser.add_argument("--degree", type=int, help="Even polynomial degree for --expr.")
    parser.add_argument("--model-dir", help="Override model directory, e.g. 3n4d.")
    parser.add_argument("--full-basis", action="store_true", help="Use the complete basis instead of RL pruning.")
    parser.add_argument("--active-indices", help="Comma-separated retained basis indices, e.g. 0,2,4.")
    parser.add_argument("--mask", help="Comma-separated 0/1 mask for retained basis entries.")
    parser.add_argument("--solver", default="CLARABEL", help="CVXPY solver name. Default: CLARABEL.")
    parser.add_argument("--outdir", default="qrl_results", help="Output directory.")
    parser.add_argument("--rank-tol", type=float, default=1e-6, help="Eigenvalue threshold used for rank.")
    parser.add_argument("--verbose", action="store_true", help="Pass verbose=True to CVXPY and print rollback details.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cases = select_cases(args)
    outdir = Path(args.outdir)

    successes = 0
    failures = 0
    for case in cases:
        try:
            result = solve_case(args, case)
        except Exception as exc:
            sample = case.sample_id if case.sample_id is not None else "expr"
            print(f"[fail] sample={sample}: {exc}")
            failures += 1
            continue

        if result is None:
            failures += 1
            continue

        write_outputs(result, outdir)
        print_summary(result)
        successes += 1

    print(f"Done. success={successes}, failed={failures}, outdir={outdir}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
