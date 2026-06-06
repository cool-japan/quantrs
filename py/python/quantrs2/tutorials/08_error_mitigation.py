"""Zero-Noise Extrapolation Error Mitigation — Tutorial 08.

Error mitigation reduces the impact of hardware noise on quantum computation
outputs without the overhead of full quantum error correction.

**Zero-Noise Extrapolation (ZNE)**:
1. Evaluate the noisy expectation value at several noise scale factors lambda=1,2,3
2. Fit a polynomial (or Richardson extrapolation) to the results
3. Extrapolate to lambda=0 (zero noise) to estimate the ideal value

**Noise model** (simplified):
    E_noisy(lambda) = E_ideal + lambda * noise_rate

where lambda=1 is the native device noise level, lambda=2 is twice the noise, etc.

**Richardson extrapolation** (order 2, scale factors 1,2,3):
    E_ZNE = (9 * E(1) - 4.5 * E(2) + 0.5 * E(3) + small correction)

More precisely, for linear model: E_ZNE = 2*E(1) - E(2)
For Richardson order-2 with {1,2,3}: weights derived from Vandermonde system.

Run as:
    python -m quantrs2.tutorials.08_error_mitigation
"""

from __future__ import annotations

import sys
import math
from typing import List, Tuple

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    print("numpy not available — install with: pip install numpy")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Simple noise model
# ---------------------------------------------------------------------------

class NoisyCircuit:
    """
    A simplified noisy quantum circuit model.

    The circuit evaluates <Z> = cos(theta) in the ideal case.
    Noise adds a constant offset proportional to the noise level.
    """

    def __init__(self, theta: float, noise_rate: float = 0.05) -> None:
        self.theta = theta
        self.noise_rate = noise_rate
        self._ideal = math.cos(theta)

    def expectation_value(self, lambda_scale: float = 1.0) -> float:
        """
        Compute noisy expectation value at noise scale lambda.

        For a single-qubit circuit evaluating <Z>:
          E_noisy(lambda) = E_ideal * exp(-lambda * noise_rate)

        This exponential model captures depolarizing noise accurately.
        """
        return self._ideal * math.exp(-lambda_scale * self.noise_rate)

    @property
    def ideal_value(self) -> float:
        """Return the noiseless (ideal) expectation value."""
        return self._ideal


# ---------------------------------------------------------------------------
# ZNE: Richardson extrapolation
# ---------------------------------------------------------------------------

def richardson_extrapolation(
    scale_factors: List[float],
    values: List[float],
) -> float:
    """
    Extrapolate to zero noise using Richardson extrapolation (polynomial fit).

    For M scale factors, fits a polynomial of degree M-1 and evaluates at lambda=0.
    """
    n = len(scale_factors)
    lambdas = np.array(scale_factors, dtype=float)
    evals = np.array(values, dtype=float)

    # Vandermonde system: find coefficients c such that sum(c_i * E(lambda_i)) = E_ideal
    # This is equivalent to polynomial interpolation at lambda=0.
    # Build Vandermonde matrix
    V = np.vander(lambdas, N=n, increasing=True)

    # We want P(0) where P interpolates the points (lambda_i, E(lambda_i))
    # P(0) = c^T @ e_0 where V @ coeffs = evals, P(0) = coeffs[0]
    try:
        coeffs = np.linalg.solve(V, evals)
        return float(coeffs[0])
    except np.linalg.LinAlgError:
        # Fallback: linear extrapolation
        if n >= 2:
            l1, l2 = scale_factors[0], scale_factors[1]
            e1, e2 = values[0], values[1]
            slope = (e2 - e1) / (l2 - l1) if abs(l2 - l1) > 1e-12 else 0.0
            return e1 - slope * l1
        return values[0]


def linear_zne(e1: float, e2: float, lambda1: float, lambda2: float) -> float:
    """Simple linear (order-1) ZNE: extrapolate from two points to lambda=0."""
    if abs(lambda2 - lambda1) < 1e-12:
        return e1
    slope = (e2 - e1) / (lambda2 - lambda1)
    return e1 - slope * lambda1


# ---------------------------------------------------------------------------
# Try native mitigation module
# ---------------------------------------------------------------------------

def _try_native_zne(circuit_theta: float, noise_rate: float) -> None:
    """Try to use native ZNE if available; print note if not."""
    try:
        from quantrs2.mitigation import ZeroNoiseExtrapolation, ZNEConfig
        print("  [Native] ZeroNoiseExtrapolation available in quantrs2.mitigation")
        print("           (using pure-numpy demonstration for this tutorial)")
    except (ImportError, AttributeError):
        print("  [Fallback] quantrs2.mitigation ZNE not available — using numpy implementation")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Tutorial 08: Zero-Noise Extrapolation Error Mitigation")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    theta = math.pi / 4.0   # 45 degrees: ideal E = cos(pi/4) = 0.7071
    noise_rate = 0.05        # 5% noise per unit scale

    circuit = NoisyCircuit(theta=theta, noise_rate=noise_rate)
    ideal = circuit.ideal_value
    print(f"Circuit: RX(theta=pi/4) with noise_rate={noise_rate:.2f}")
    print(f"Observable: <Z>")
    print(f"Ideal value:  E_ideal = cos(pi/4) = {ideal:.6f}")
    print()

    _try_native_zne(theta, noise_rate)
    print()

    # ------------------------------------------------------------------
    # Evaluate at noise scale factors 1, 2, 3
    # ------------------------------------------------------------------
    scale_factors = [1.0, 2.0, 3.0]
    noisy_values = [circuit.expectation_value(lam) for lam in scale_factors]

    print("Noisy evaluations:")
    print(f"  {'Lambda':>8}  {'E_noisy':>10}  {'Error vs ideal':>16}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*16}")
    for lam, eval_v in zip(scale_factors, noisy_values):
        err = eval_v - ideal
        print(f"  {lam:>8.1f}  {eval_v:>10.6f}  {err:>+16.6f}")
    print()

    # ------------------------------------------------------------------
    # Linear ZNE (two-point)
    # ------------------------------------------------------------------
    zne_linear = linear_zne(noisy_values[0], noisy_values[1],
                             scale_factors[0], scale_factors[1])
    zne_err_linear = abs(zne_linear - ideal)

    # ------------------------------------------------------------------
    # Richardson extrapolation (three-point, order 2)
    # ------------------------------------------------------------------
    zne_richardson = richardson_extrapolation(scale_factors, noisy_values)
    zne_err_rich = abs(zne_richardson - ideal)

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    raw_err = abs(noisy_values[0] - ideal)

    print("ZNE Results:")
    print(f"  {'Method':>25}  {'Estimate':>12}  {'Error vs ideal':>16}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*16}")
    print(f"  {'Raw (lambda=1)':>25}  {noisy_values[0]:>12.6f}  {raw_err:>+16.6f}")
    print(f"  {'Linear ZNE (2-point)':>25}  {zne_linear:>12.6f}  {zne_err_linear:>+16.6f}")
    print(f"  {'Richardson ZNE (3-point)':>25}  {zne_richardson:>12.6f}  {zne_err_rich:>+16.6f}")
    print(f"  {'Ideal':>25}  {ideal:>12.6f}  {0.0:>+16.6f}")
    print()

    # ------------------------------------------------------------------
    # Multiple noise regimes
    # ------------------------------------------------------------------
    print("ZNE performance across noise regimes:")
    print(f"  {'Noise rate':>12}  {'Raw error':>12}  {'ZNE error':>12}  {'Improvement':>12}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

    all_improved = True
    for nr in [0.01, 0.05, 0.10, 0.20]:
        circ = NoisyCircuit(theta=theta, noise_rate=nr)
        e_raw = circ.expectation_value(1.0)
        evals = [circ.expectation_value(lam) for lam in scale_factors]
        e_zne = richardson_extrapolation(scale_factors, evals)
        err_raw = abs(e_raw - ideal)
        err_zne = abs(e_zne - ideal)
        improv = err_raw / max(err_zne, 1e-12)
        better = err_zne < err_raw
        if not better:
            all_improved = False
        print(f"  {nr:>12.3f}  {err_raw:>12.6f}  {err_zne:>12.6f}  {improv:>10.2f}x  "
              f"{'better' if better else 'worse'}")
    print()

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    rich_improves = zne_err_rich < raw_err
    linear_improves = zne_err_linear < raw_err

    print("Verification:")
    print(f"  Richardson ZNE reduces error vs raw: {'PASS' if rich_improves else 'FAIL'}")
    print(f"  Linear ZNE reduces error vs raw:     {'PASS' if linear_improves else 'FAIL'}")
    print()

    print("Key Concepts:")
    print("  - ZNE: evaluate at lambda=1,2,3 (amplified noise scales)")
    print("  - Extrapolate polynomial fit to lambda=0 (zero noise)")
    print("  - Richardson extrapolation: exact for polynomial noise models")
    print("  - Exponential noise model: E(lambda) = E_ideal * exp(-lambda * r)")
    print("  - ZNE is hardware-friendly: no ancilla qubits required")
    print("  - Works best when noise rate is small and model is appropriate")
    print()

    if rich_improves:
        print(f"ZNE successfully reduced error: {raw_err:.6f} -> {zne_err_rich:.6f} "
              f"({raw_err/max(zne_err_rich,1e-12):.2f}x improvement)")
    else:
        print("Note: ZNE did not improve accuracy for this configuration.")
    print()
    print("Error mitigation tutorial complete.")


if __name__ == "__main__":
    main()
