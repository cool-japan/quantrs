"""QUBO Sampling with Multiple Samplers — Tutorial 05.

QUBO (Quadratic Unconstrained Binary Optimization) is the standard formulation
for combinatorial optimization problems solved by quantum annealers and
classical heuristics.

**Number-partitioning problem**:
Given weights {1, 2, 3, 4, 5} (sum S=15), partition them into two equal-sum
subsets.  Optimal split: {1,4,5} / {2,3} (sums: 10 vs 5 — actually NOT equal)
Or: {2,3,5} / {1,4} (sums: 10 vs 5 — also not equal)
Wait: S/2 = 7.5 is not integer, so the problem has no perfect solution.
Better example: {1,2,3,4} S=10, target=5: {1,4} / {2,3}.

**QUBO formulation** for partitioning {w_i}:
    Q_ii = w_i * (w_i - S/2)
    Q_ij = w_i * w_j   (i < j)
where S = sum(weights), minimising gives best partition.

Run as:
    python -m quantrs2.tutorials.05_qubo_sampling

Note: If native tytan samplers are not available, this tutorial implements
a pure-Python simulated annealing and reports gracefully.
Marked @pytest.mark.slow.
"""

from __future__ import annotations

import sys
import math
import random
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    print("numpy not available — install with: pip install numpy")
    sys.exit(0)


# ---------------------------------------------------------------------------
# QUBO problem: partition {1, 2, 3, 4} into subsets each summing to 5
# ---------------------------------------------------------------------------

_WEIGHTS = [1, 2, 3, 4]
_S = sum(_WEIGHTS)          # 10
_TARGET = _S // 2            # 5


def build_qubo(weights: List[int], target: float) -> np.ndarray:
    """
    Build QUBO matrix Q for the number-partitioning problem.

    xi in {0, 1}: xi=0 means weight i in set A, xi=1 in set B.
    Objective: minimize (sum_i w_i * x_i - target)^2

    Expanding: Q_ii = w_i*(w_i - 2*target), Q_ij = 2*w_i*w_j (i<j)
    """
    n = len(weights)
    Q = np.zeros((n, n), dtype=float)
    for i in range(n):
        Q[i, i] = weights[i] * (weights[i] - 2.0 * target)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = 2.0 * weights[i] * weights[j]
    return Q


def qubo_energy(Q: np.ndarray, assignment: List[int]) -> float:
    """Compute x^T Q x for binary assignment x."""
    x = np.array(assignment, dtype=float)
    return float(x @ Q @ x)


def partition_quality(weights: List[int], assignment: List[int]) -> Tuple[int, int, int]:
    """Return (sum_A, sum_B, |sum_A - sum_B|) for a given assignment."""
    sum_a = sum(w for w, a in zip(weights, assignment) if a == 0)
    sum_b = sum(w for w, a in zip(weights, assignment) if a == 1)
    return sum_a, sum_b, abs(sum_a - sum_b)


# ---------------------------------------------------------------------------
# Pure-Python Simulated Annealing sampler
# ---------------------------------------------------------------------------

def simulated_annealing(
    Q: np.ndarray,
    n_steps: int = 1000,
    t_start: float = 5.0,
    t_end: float = 0.01,
    seed: int = 42,
) -> Tuple[List[int], float]:
    """Simple SA for QUBO minimisation."""
    rng = random.Random(seed)
    n = Q.shape[0]
    current = [rng.randint(0, 1) for _ in range(n)]
    current_e = qubo_energy(Q, current)
    best = list(current)
    best_e = current_e

    for step in range(n_steps):
        t = t_start * (t_end / t_start) ** (step / n_steps)
        flip_idx = rng.randint(0, n - 1)
        candidate = list(current)
        candidate[flip_idx] ^= 1
        candidate_e = qubo_energy(Q, candidate)
        delta = candidate_e - current_e
        if delta < 0 or rng.random() < math.exp(-delta / t):
            current = candidate
            current_e = candidate_e
            if current_e < best_e:
                best = list(current)
                best_e = current_e

    return best, best_e


# ---------------------------------------------------------------------------
# Tabu search sampler
# ---------------------------------------------------------------------------

def tabu_search(
    Q: np.ndarray,
    n_iter: int = 200,
    tabu_tenure: int = 5,
    seed: int = 42,
) -> Tuple[List[int], float]:
    """Simple tabu search for QUBO minimisation."""
    rng = random.Random(seed)
    n = Q.shape[0]
    current = [rng.randint(0, 1) for _ in range(n)]
    current_e = qubo_energy(Q, current)
    best = list(current)
    best_e = current_e
    tabu: List[int] = []

    for _ in range(n_iter):
        best_neighbor: Optional[List[int]] = None
        best_neighbor_e = float("inf")
        best_flip = -1

        for flip_idx in range(n):
            if flip_idx in tabu:
                continue
            neighbor = list(current)
            neighbor[flip_idx] ^= 1
            e = qubo_energy(Q, neighbor)
            if e < best_neighbor_e:
                best_neighbor_e = e
                best_neighbor = neighbor
                best_flip = flip_idx

        if best_neighbor is None:
            break
        current = best_neighbor
        current_e = best_neighbor_e
        tabu.append(best_flip)
        if len(tabu) > tabu_tenure:
            tabu.pop(0)

        if current_e < best_e:
            best = list(current)
            best_e = current_e

    return best, best_e


# ---------------------------------------------------------------------------
# Try native tytan samplers (gracefully skip if not available)
# ---------------------------------------------------------------------------

def _try_native_tytan(Q: np.ndarray) -> Optional[Dict[str, object]]:
    """Attempt to use native tytan module.  Returns None if unavailable."""
    try:
        from quantrs2 import quantrs2 as _q
        if not hasattr(_q, "tytan"):
            return None
        # Native tytan exposes visualization/analysis only, not samplers
        # Return None to signal we use our own samplers
        return None
    except (ImportError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Tutorial 05: QUBO Sampling — Number Partitioning")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Problem description
    # ------------------------------------------------------------------
    print(f"Problem: Partition weights {_WEIGHTS} into two equal-sum subsets")
    print(f"  Sum S = {_S}, Target per subset = {_TARGET}")
    print()

    # ------------------------------------------------------------------
    # Build QUBO
    # ------------------------------------------------------------------
    Q = build_qubo(_WEIGHTS, _TARGET)
    n = len(_WEIGHTS)
    print("QUBO matrix Q (diagonal = linear, off-diag = quadratic):")
    for row in Q:
        print("  " + "  ".join(f"{v:6.1f}" for v in row))
    print()

    # ------------------------------------------------------------------
    # Enumerate all solutions (brute force, 2^4 = 16)
    # ------------------------------------------------------------------
    print("All assignments and QUBO energies:")
    print(f"  {'Assignment':>12}  {'Energy':>8}  {'Sum_A':>6}  {'Sum_B':>6}  {'|diff|':>7}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*7}")
    best_brute_e = float("inf")
    best_brute_assign: List[int] = []
    for b in range(2 ** n):
        assign = [(b >> i) & 1 for i in range(n)]
        e = qubo_energy(Q, assign)
        sa, sb, diff = partition_quality(_WEIGHTS, assign)
        s = "".join(str(a) for a in assign)
        print(f"  [{s}]         {e:>8.1f}  {sa:>6}  {sb:>6}  {diff:>7}")
        if e < best_brute_e:
            best_brute_e = e
            best_brute_assign = list(assign)
    print()
    sa_bf, sb_bf, diff_bf = partition_quality(_WEIGHTS, best_brute_assign)
    print(f"  Brute-force optimal: assign={''.join(str(a) for a in best_brute_assign)}, "
          f"E={best_brute_e:.1f}, A={sa_bf}, B={sb_bf}, |diff|={diff_bf}")
    print()

    # ------------------------------------------------------------------
    # Check for native tytan samplers
    # ------------------------------------------------------------------
    native_result = _try_native_tytan(Q)
    if native_result is not None:
        print("[Native] Used quantrs2 tytan samplers")
    else:
        print("[Fallback] quantrs2 tytan samplers not available — using pure-Python SA and Tabu")
    print()

    # ------------------------------------------------------------------
    # Simulated Annealing
    # ------------------------------------------------------------------
    print("Sampler 1: Simulated Annealing (1000 steps, seed=42)")
    sa_assign, sa_e = simulated_annealing(Q, n_steps=1000, seed=42)
    sa_A, sa_B, sa_diff = partition_quality(_WEIGHTS, sa_assign)
    print(f"  Best assignment: {''.join(str(a) for a in sa_assign)}")
    print(f"  QUBO energy: {sa_e:.1f}")
    print(f"  Set A (xi=0): sum={sa_A}, Set B (xi=1): sum={sa_B}, |diff|={sa_diff}")
    print()

    # ------------------------------------------------------------------
    # Tabu Search
    # ------------------------------------------------------------------
    print("Sampler 2: Tabu Search (200 iterations, tenure=5, seed=42)")
    tabu_assign, tabu_e = tabu_search(Q, n_iter=200, tabu_tenure=5, seed=42)
    tabu_A, tabu_B, tabu_diff = partition_quality(_WEIGHTS, tabu_assign)
    print(f"  Best assignment: {''.join(str(a) for a in tabu_assign)}")
    print(f"  QUBO energy: {tabu_e:.1f}")
    print(f"  Set A (xi=0): sum={tabu_A}, Set B (xi=1): sum={tabu_B}, |diff|={tabu_diff}")
    print()

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    sa_optimal = abs(sa_e - best_brute_e) < 1.0
    tabu_optimal = abs(tabu_e - best_brute_e) < 1.0

    print("Verification:")
    print(f"  SA found optimal solution: {'PASS' if sa_optimal else 'FAIL'}")
    print(f"  Tabu found optimal solution: {'PASS' if tabu_optimal else 'FAIL'}")
    print()

    print("Key Concepts:")
    print("  - QUBO: Q_ii = linear bias, Q_ij = quadratic coupling")
    print("  - Objective x^T Q x is minimised by quantum annealers / heuristics")
    print("  - SA: random walk with temperature schedule (Metropolis criterion)")
    print("  - Tabu search: avoid recently visited solutions (tabu list)")
    print("  - Both converge to near-optimal for small problems")
    print()
    print("QUBO sampling tutorial complete.")


if __name__ == "__main__":
    main()
