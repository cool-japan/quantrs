"""QAOA for Max-Cut Optimization — Tutorial 03.

The Quantum Approximate Optimization Algorithm (QAOA) is a variational quantum
algorithm for combinatorial optimization problems.  It interleaves problem-specific
(phase) and mixing (driver) unitaries, controlled by parameters beta and gamma.

**Max-Cut problem**:
Find a partition of graph vertices into two sets (S, S̄) that maximises the
number of edges crossing between the sets.

**Graph**: 4-vertex path graph: 0 — 1 — 2 — 3
- Edges: (0,1), (1,2), (2,3)
- Max-cut = 2 (e.g. partition {0,2} | {1,3} cuts edges (0,1) and (2,3))

**QAOA p=1**:
    |psi(beta,gamma)> = U_B(beta) U_C(gamma) |+>^4

where U_C(gamma) = product over edges of exp(-i*gamma*Z_i*Z_j / 2)
      U_B(beta)  = product over qubits of exp(-i*beta*X_j)

Run as:
    python -m quantrs2.tutorials.03_qaoa_maxcut
"""

from __future__ import annotations

import sys
import math
import itertools
from typing import Dict, List, Tuple

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    print("numpy not available — install with: pip install numpy")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Graph definition: 4-vertex path  0 - 1 - 2 - 3
# ---------------------------------------------------------------------------

_N_VERTICES = 4
_EDGES: List[Tuple[int, int]] = [(0, 1), (1, 2), (2, 3)]
# For path 0-1-2-3, max-cut = 3 by partition {0,2}|{1,3}:
#   edge (0,1): 0 in {0,2}, 1 in {1,3} -> CUT
#   edge (1,2): 1 in {1,3}, 2 in {0,2} -> CUT
#   edge (2,3): 2 in {0,2}, 3 in {1,3} -> CUT
_MAX_CUT_OPTIMAL = 3


# ---------------------------------------------------------------------------
# Classical cut value
# ---------------------------------------------------------------------------

def _cut_value(bitstring: int, edges: List[Tuple[int, int]]) -> int:
    """Compute number of edges cut by the binary partition encoded in bitstring."""
    count = 0
    for u, v in edges:
        b_u = (bitstring >> u) & 1
        b_v = (bitstring >> v) & 1
        if b_u != b_v:
            count += 1
    return count


# ---------------------------------------------------------------------------
# QAOA circuit simulation (numpy state-vector)
# ---------------------------------------------------------------------------

def _apply_phase_unitary(state: np.ndarray, gamma: float,
                          edges: List[Tuple[int, int]], n: int) -> np.ndarray:
    """Apply U_C(gamma) = prod_{(i,j) in E} exp(-i gamma Z_i Z_j / 2)."""
    new_state = state.copy()
    for i_basis in range(2 ** n):
        phase_total = 0.0
        for u, v in edges:
            b_u = (i_basis >> u) & 1
            b_v = (i_basis >> v) & 1
            # Z eigenvalue: 0 -> +1, 1 -> -1
            z_u = 1 - 2 * b_u
            z_v = 1 - 2 * b_v
            phase_total += gamma * z_u * z_v / 2.0
        new_state[i_basis] *= complex(math.cos(phase_total), -math.sin(phase_total))
    return new_state


def _apply_mixer_unitary(state: np.ndarray, beta: float, n: int) -> np.ndarray:
    """Apply U_B(beta) = prod_j exp(-i beta X_j)."""
    # For each qubit j, RX(2*beta) rotation
    for qubit in range(n):
        c = math.cos(beta)
        s = math.sin(beta)
        new_state = np.zeros_like(state)
        for i_basis in range(2 ** n):
            # Flip qubit j
            j_flipped = i_basis ^ (1 << qubit)
            new_state[i_basis] += c * state[i_basis] - 1j * s * state[j_flipped]
        state = new_state
    return state


def qaoa_p1_statevector(beta: float, gamma: float,
                         edges: List[Tuple[int, int]], n: int) -> np.ndarray:
    """Run QAOA p=1 circuit and return the final state vector."""
    # Start in equal superposition |+>^n
    state = np.ones(2 ** n, dtype=complex) / math.sqrt(2 ** n)

    # Apply phase unitary U_C(gamma)
    state = _apply_phase_unitary(state, gamma, edges, n)

    # Apply mixer unitary U_B(beta)
    state = _apply_mixer_unitary(state, beta, n)

    return state


def qaoa_cost(beta: float, gamma: float,
               edges: List[Tuple[int, int]], n: int) -> float:
    """Compute <psi|C|psi> where C = sum_{(i,j)} (1 - Z_i Z_j) / 2."""
    state = qaoa_p1_statevector(beta, gamma, edges, n)
    probs = np.abs(state) ** 2
    expected_cut = 0.0
    for i_basis in range(2 ** n):
        expected_cut += probs[i_basis] * _cut_value(i_basis, edges)
    return expected_cut


# ---------------------------------------------------------------------------
# Grid search over beta in [0, pi], gamma in [0, 2pi]
# ---------------------------------------------------------------------------

def grid_search(n_beta: int = 20, n_gamma: int = 20) -> Tuple[float, float, float]:
    """Search for optimal QAOA parameters."""
    best_cost = -1.0
    best_beta = 0.0
    best_gamma = 0.0
    for i in range(n_beta):
        beta = math.pi * i / n_beta
        for j in range(n_gamma):
            gamma = 2.0 * math.pi * j / n_gamma
            cost = qaoa_cost(beta, gamma, _EDGES, _N_VERTICES)
            if cost > best_cost:
                best_cost = cost
                best_beta = beta
                best_gamma = gamma
    return best_beta, best_gamma, best_cost


def scipy_refine(beta0: float, gamma0: float) -> Tuple[float, float, float]:
    """Optionally refine with scipy."""
    try:
        from scipy.optimize import minimize

        def neg_cost(params: List[float]) -> float:
            return -qaoa_cost(params[0], params[1], _EDGES, _N_VERTICES)

        res = minimize(neg_cost, [beta0, gamma0], method="BFGS",
                       options={"maxiter": 200})
        return float(res.x[0]), float(res.x[1]), -float(res.fun)
    except ImportError:
        return beta0, gamma0, qaoa_cost(beta0, gamma0, _EDGES, _N_VERTICES)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Tutorial 03: QAOA for Max-Cut Optimization")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Problem description
    # ------------------------------------------------------------------
    print("Problem: Max-Cut on 4-vertex path graph")
    print("  Vertices: 0, 1, 2, 3")
    print("  Edges: (0,1), (1,2), (2,3)")
    print("  Optimal cut = 3 (partition {0,2} vs {1,3}: cuts all 3 edges)")
    print()

    # ------------------------------------------------------------------
    # Enumerate all cuts (classical baseline)
    # ------------------------------------------------------------------
    print("All partitions and their cut values:")
    print(f"  {'Partition':>10}  {'Cut':>5}")
    print(f"  {'-'*10}  {'-'*5}")
    max_cut_found = 0
    best_partitions: List[str] = []
    for b in range(2 ** _N_VERTICES):
        cv = _cut_value(b, _EDGES)
        s = format(b, f"0{_N_VERTICES}b")
        print(f"  |{s}>    {cv:>5}")
        if cv > max_cut_found:
            max_cut_found = cv
            best_partitions = [s]
        elif cv == max_cut_found:
            best_partitions.append(s)
    print()
    print(f"  Classical optimal: cut={max_cut_found}, partitions={best_partitions}")
    print()

    # ------------------------------------------------------------------
    # QAOA grid search
    # ------------------------------------------------------------------
    n_beta, n_gamma = 20, 20
    total_evals = n_beta * n_gamma
    print(f"QAOA p=1: grid search ({n_beta}×{n_gamma} = {total_evals} evaluations)")
    best_beta, best_gamma, best_cost = grid_search(n_beta, n_gamma)
    approx_ratio = best_cost / _MAX_CUT_OPTIMAL
    print(f"  Best beta  = {best_beta:.4f} rad")
    print(f"  Best gamma = {best_gamma:.4f} rad")
    print(f"  Expected cut value = {best_cost:.4f}")
    print(f"  Approximation ratio = {approx_ratio:.4f}")
    print()

    # ------------------------------------------------------------------
    # Scipy refinement
    # ------------------------------------------------------------------
    print("Refinement (scipy.optimize.minimize if available)...")
    opt_beta, opt_gamma, opt_cost = scipy_refine(best_beta, best_gamma)
    opt_ratio = opt_cost / _MAX_CUT_OPTIMAL
    print(f"  Refined beta  = {opt_beta:.6f} rad")
    print(f"  Refined gamma = {opt_gamma:.6f} rad")
    print(f"  Expected cut value = {opt_cost:.6f}")
    print(f"  Approximation ratio = {opt_ratio:.6f}")
    print()

    # ------------------------------------------------------------------
    # Sample from the optimal QAOA state
    # ------------------------------------------------------------------
    opt_state = qaoa_p1_statevector(opt_beta, opt_gamma, _EDGES, _N_VERTICES)
    probs = np.abs(opt_state) ** 2

    print("Measurement probabilities (top 4 bitstrings):")
    prob_sorted = sorted(
        [(format(i, f"0{_N_VERTICES}b"), probs[i], _cut_value(i, _EDGES))
         for i in range(2 ** _N_VERTICES)],
        key=lambda x: -x[1]
    )
    print(f"  {'Bitstring':>10}  {'Probability':>12}  {'Cut value':>10}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*10}")
    for bitstr, prob, cv in prob_sorted[:4]:
        print(f"  |{bitstr}>    {prob:>12.6f}  {cv:>10}")
    print()

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    norm_ok = abs(float(np.sum(probs)) - 1.0) < 1e-10
    ratio_ok = opt_ratio >= 0.5  # QAOA p=1 should give at least 0.5 approx ratio
    print("Verification:")
    print(f"  State normalised: {'PASS' if norm_ok else 'FAIL'}")
    print(f"  Approximation ratio >= 0.5: {'PASS' if ratio_ok else 'FAIL'}")
    print()

    print("Key Concepts:")
    print("  - QAOA alternates between problem Hamiltonian (phase) and mixer unitaries")
    print("  - Phase unitary U_C(gamma): encodes the cut-value cost function")
    print("  - Mixer unitary U_B(beta): explores the solution space via X rotations")
    print("  - More layers (p) -> higher approximation ratio but deeper circuit")
    print("  - QAOA p=1 on 3-regular graphs achieves >= 0.6924 approximation ratio")
    print()
    print(f"QAOA complete. Best expected cut: {opt_cost:.4f} / {_MAX_CUT_OPTIMAL}")


if __name__ == "__main__":
    main()
