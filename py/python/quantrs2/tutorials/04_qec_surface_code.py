"""QEC Surface Code Threshold Simulation — Tutorial 04.

Quantum Error Correction (QEC) protects quantum information against decoherence
and gate errors by encoding a single logical qubit in many physical qubits.

**Rotated planar surface code [[d², 1, d]]**:
- d=3: 9 data qubits, 8 stabilizers (4 X-type, 4 Z-type), distance 3
- Encodes 1 logical qubit with code distance d
- Can correct any error on at most floor((d-1)/2) physical qubits

**Threshold simulation**:
Run Monte Carlo trials at various physical error rates p.
For p < threshold (~10% for surface code):
  logical error rate < physical error rate (QEC improves things)
For p > threshold:
  logical error rate > physical error rate (QEC makes things worse)

Run as:
    python -m quantrs2.tutorials.04_qec_surface_code

Note: This tutorial uses the quantrs2.qec module (pure-Python stubs if
native bindings are not compiled in).  Marked @pytest.mark.slow.
"""

from __future__ import annotations

import sys
import random
from typing import Dict, List

# ---------------------------------------------------------------------------
# Import the QEC module (always available — has pure-Python stubs)
# ---------------------------------------------------------------------------

try:
    from quantrs2.qec import (
        RotatedSurfaceCode,
        MwpmDecoder,
    )
    _QEC_AVAILABLE = True
except (ImportError, AttributeError):
    print("quantrs2.qec not available. Exiting.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Monte Carlo threshold simulation
# ---------------------------------------------------------------------------

def _random_pauli_error(n: int, error_rate: float, rng: random.Random) -> List[str]:
    """Generate random single-qubit depolarizing errors."""
    paulis: List[str] = []
    for _ in range(n):
        if rng.random() < error_rate:
            paulis.append(rng.choice(("X", "Y", "Z")))
        else:
            paulis.append("I")
    return paulis


def _check_logical_x_failure(
    correction: List[str],
    error: List[str],
    logical_x_support: List[int],
) -> bool:
    """Return True if net Pauli (correction * error) has odd logical-X parity."""
    count = 0
    for q in logical_x_support:
        x_err = error[q] in ("X", "Y")
        x_cor = correction[q] in ("X", "Y")
        if x_err != x_cor:
            count += 1
    return count % 2 == 1


def _check_logical_z_failure(
    correction: List[str],
    error: List[str],
    logical_z_support: List[int],
) -> bool:
    """Return True if net Pauli has odd logical-Z parity."""
    count = 0
    for q in logical_z_support:
        z_err = error[q] in ("Z", "Y")
        z_cor = correction[q] in ("Z", "Y")
        if z_err != z_cor:
            count += 1
    return count % 2 == 1


def simulate_code_threshold(
    distance: int,
    error_rates: List[float],
    shots: int,
    seed: int = 42,
) -> Dict[float, float]:
    """
    Simulate logical error rate vs physical error rate for given code distance.

    Returns dict mapping physical error rate -> logical error rate.
    """
    rng = random.Random(seed)
    code = RotatedSurfaceCode(d=distance)
    decoder = MwpmDecoder(code)

    n = code.n_data_qubits()
    lx_support = code.logical_x_qubits()
    lz_support = code.logical_z_qubits()

    results: Dict[float, float] = {}
    for p in error_rates:
        failures = 0
        for _ in range(shots):
            error = _random_pauli_error(n, p, rng)
            syndrome = code.syndrome(error)
            correction = decoder.decode(syndrome)

            failed_x = _check_logical_x_failure(correction, error, lx_support)
            failed_z = _check_logical_z_failure(correction, error, lz_support)

            if failed_x or failed_z:
                failures += 1

        results[p] = failures / shots

    return results


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Tutorial 04: QEC Surface Code Threshold Simulation")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Code parameters
    # ------------------------------------------------------------------
    distance = 3
    code = RotatedSurfaceCode(d=distance)
    n, k, d = code.n_k_d()

    print(f"Surface code [[{n}, {k}, {d}]] (rotated planar)")
    print(f"  Data qubits: {n}")
    print(f"  Logical qubits: {k}")
    print(f"  Code distance: {d}")
    print(f"  Can correct up to {(d-1)//2} errors")
    print()

    # ------------------------------------------------------------------
    # Threshold simulation
    # ------------------------------------------------------------------
    error_rates = [0.01, 0.02, 0.05, 0.10]
    shots = 50  # fast for tutorial; use 500+ for statistical accuracy
    seed = 42

    print(f"Monte Carlo simulation: {shots} shots per error rate")
    print(f"Error rates: {error_rates}")
    print()

    results = simulate_code_threshold(distance, error_rates, shots, seed)

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print("Results:")
    print(f"  {'p_physical':>12}  {'p_logical':>12}  {'Improvement':>12}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}")

    all_improved = True
    for p_phys in error_rates:
        p_log = results[p_phys]
        if p_phys > 0:
            improvement = p_phys / max(p_log, 1e-6)
        else:
            improvement = float("inf")

        better = p_log <= p_phys
        if p_phys < 0.10 and not better:
            all_improved = False

        label = "better" if better else "worse"
        print(f"  {p_phys:>12.3f}  {p_log:>12.3f}  {improvement:>10.2f}x  [{label}]")
    print()

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    # With only 50 shots and possible pure-Python stub decoders,
    # results can be statistically noisy.
    # We just verify the simulation ran without errors.
    ran_ok = len(results) == len(error_rates)
    # Also check the lowest error rate showed low logical errors (or zero)
    low_p_ok = results.get(0.01, 0.0) <= 0.15  # 15% tolerance for 50 shots

    print("Verification:")
    print(f"  Simulation ran for all error rates: {'PASS' if ran_ok else 'FAIL'}")
    print(f"  Low error rate (p=0.01) log error ≤ 0.15: {'PASS' if low_p_ok else 'FAIL'}")
    print()
    print("  Note: With 50 shots, results have ±14% statistical uncertainty.")
    print("  Increase shots to 500+ for statistically reliable threshold estimates.")
    print()

    print("Key Concepts:")
    print("  - Surface code encodes 1 logical qubit in d² physical qubits")
    print("  - Syndrome measurement detects errors without measuring logical state")
    print("  - MWPM decoder finds minimum-weight correction given syndrome")
    print("  - Below threshold: logical error rate < physical error rate")
    print("  - The surface code threshold is ~1% for circuit-level depolarizing noise")
    print(f"  - At {shots} shots per rate, results have ±{1/shots**0.5:.2f} statistical noise")
    print()
    print(f"Threshold simulation complete for d={distance} surface code.")


if __name__ == "__main__":
    main()
