"""Parameterized Circuits and Gradients — Tutorial 07.

Parameterized quantum circuits are the backbone of variational quantum
algorithms.  Computing gradients of quantum circuit outputs with respect
to gate parameters is essential for training these circuits.

**Circuit**: RX(theta) ⊗ RY(phi) on 2 qubits (no entanglement in this example)
**Observable**: Z ⊗ I (Pauli Z on qubit 0, identity on qubit 1)

**Cost function**:
    C(theta, phi) = <psi(theta,phi)| Z⊗I |psi(theta,phi)>
                  = <0|RX(theta)† Z RX(theta)|0>    (qubit 1 factors out)
                  = cos(theta)

**Analytical gradient**:
    dC/dtheta = -sin(theta)   (exact)
    dC/dphi   = 0             (phi only acts on qubit 1; Z⊗I doesn't see it)

**Parameter-shift gradient** (quantum hardware compatible):
    dC/dtheta = (C(theta + pi/2) - C(theta - pi/2)) / 2

The parameter-shift rule is exact for gates of the form exp(-i theta/2 P)
where P is a Pauli operator.

Run as:
    python -m quantrs2.tutorials.07_parameterized_circuits
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
# Gate matrices
# ---------------------------------------------------------------------------

def _rx(theta: float) -> np.ndarray:
    """2x2 RX(theta) matrix: exp(-i theta/2 X)."""
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(phi: float) -> np.ndarray:
    """2x2 RY(phi) matrix: exp(-i phi/2 Y)."""
    c = math.cos(phi / 2.0)
    s = math.sin(phi / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


# Observable: Z ⊗ I (4x4)
_I2 = np.eye(2, dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_ZI = np.kron(_Z, _I2)


# ---------------------------------------------------------------------------
# Circuit and cost function
# ---------------------------------------------------------------------------

def build_state(theta: float, phi: float) -> np.ndarray:
    """Build state |psi(theta,phi)> = (RX(theta)|0>) ⊗ (RY(phi)|0>)."""
    q0 = np.array([1.0, 0.0], dtype=complex)
    q1 = np.array([1.0, 0.0], dtype=complex)
    psi0 = _rx(theta) @ q0
    psi1 = _ry(phi) @ q1
    return np.kron(psi0, psi1)


def cost(theta: float, phi: float) -> float:
    """C(theta, phi) = <psi(theta,phi)| Z⊗I |psi(theta,phi)>."""
    psi = build_state(theta, phi)
    return float(np.real(psi.conj() @ _ZI @ psi))


def analytical_gradient(theta: float, phi: float) -> Tuple[float, float]:
    """Return exact dC/dtheta and dC/dphi."""
    return -math.sin(theta), 0.0


def parameter_shift_gradient(theta: float, phi: float,
                              shift: float = math.pi / 2.0) -> Tuple[float, float]:
    """Compute gradient via the parameter-shift rule."""
    dtheta = (cost(theta + shift, phi) - cost(theta - shift, phi)) / 2.0
    dphi = (cost(theta, phi + shift) - cost(theta, phi - shift)) / 2.0
    return dtheta, dphi


def finite_diff_gradient(theta: float, phi: float,
                          eps: float = 1e-6) -> Tuple[float, float]:
    """Numerical gradient via finite differences."""
    dtheta = (cost(theta + eps, phi) - cost(theta - eps, phi)) / (2.0 * eps)
    dphi = (cost(theta, phi + eps) - cost(theta, phi - eps)) / (2.0 * eps)
    return dtheta, dphi


# ---------------------------------------------------------------------------
# Gradient descent optimisation
# ---------------------------------------------------------------------------

def gradient_descent(
    theta0: float,
    phi0: float,
    n_steps: int = 50,
    lr: float = 0.1,
) -> Tuple[List[Tuple[float, float, float]], float, float]:
    """Minimise C(theta, phi) via gradient descent using parameter-shift gradients."""
    theta = theta0
    phi = phi0
    history: List[Tuple[float, float, float]] = []  # (step, theta, cost)

    for step in range(n_steps):
        c_val = cost(theta, phi)
        history.append((float(step), theta, c_val))
        grad_theta, grad_phi = parameter_shift_gradient(theta, phi)
        theta -= lr * grad_theta
        phi -= lr * grad_phi

    return history, theta, phi


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Tutorial 07: Parameterized Circuits and Gradients")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Circuit description
    # ------------------------------------------------------------------
    print("Circuit: RX(theta) ⊗ RY(phi) on 2 qubits")
    print("  q0: --[RX(theta)]--")
    print("  q1: --[RY(phi)]----")
    print()
    print("Observable: Z ⊗ I")
    print("Cost: C(theta,phi) = <psi| Z⊗I |psi> = cos(theta)")
    print()

    # ------------------------------------------------------------------
    # Gradient verification at a test point
    # ------------------------------------------------------------------
    theta_test = 1.2
    phi_test = 0.7
    print(f"Gradient verification at theta={theta_test:.2f}, phi={phi_test:.2f}:")
    print()

    c_test = cost(theta_test, phi_test)
    dtheta_analytical, dphi_analytical = analytical_gradient(theta_test, phi_test)
    dtheta_shift, dphi_shift = parameter_shift_gradient(theta_test, phi_test)
    dtheta_fd, dphi_fd = finite_diff_gradient(theta_test, phi_test)

    print(f"  Cost C(theta,phi) = {c_test:.8f}  (expected: {math.cos(theta_test):.8f})")
    print()
    print(f"  {'Method':>20}  {'dC/dtheta':>14}  {'dC/dphi':>12}")
    print(f"  {'-'*20}  {'-'*14}  {'-'*12}")
    print(f"  {'Analytical':>20}  {dtheta_analytical:>14.8f}  {dphi_analytical:>12.8f}")
    print(f"  {'Parameter-shift':>20}  {dtheta_shift:>14.8f}  {dphi_shift:>12.8f}")
    print(f"  {'Finite-diff':>20}  {dtheta_fd:>14.8f}  {dphi_fd:>12.8f}")
    print()

    # ------------------------------------------------------------------
    # Gradient match verification
    # ------------------------------------------------------------------
    theta_match = abs(dtheta_shift - dtheta_analytical) < 1e-6
    phi_match = abs(dphi_shift - dphi_analytical) < 1e-6
    fd_match = abs(dtheta_shift - dtheta_fd) < 1e-5

    print("Gradient Match Verification:")
    print(f"  Parameter-shift == Analytical (dtheta): {'PASS' if theta_match else 'FAIL'}")
    print(f"  Parameter-shift == Analytical (dphi):   {'PASS' if phi_match else 'FAIL'}")
    print(f"  Parameter-shift == Finite-diff (dtheta):{'PASS' if fd_match else 'FAIL'}")
    print()

    # ------------------------------------------------------------------
    # Gradient descent optimisation
    # ------------------------------------------------------------------
    theta0 = 1.8   # Starting point: cost = cos(1.8) ≈ -0.227
    phi0 = 0.5
    n_steps = 50
    lr = 0.1

    print(f"Gradient descent: {n_steps} steps, lr={lr}")
    print(f"  Start: theta={theta0:.4f}, C={cost(theta0, phi0):.6f}")

    history, theta_final, phi_final = gradient_descent(theta0, phi0, n_steps, lr)
    c_final = cost(theta_final, phi_final)

    print(f"  End:   theta={theta_final:.6f}, C={c_final:.6f}")
    print()

    # Print every 10th step
    print("  Optimisation trajectory (every 10 steps):")
    print(f"    {'Step':>5}  {'theta':>10}  {'Cost C':>10}")
    print(f"    {'-'*5}  {'-'*10}  {'-'*10}")
    for step, th, cv in history[::10]:
        print(f"    {int(step):>5}  {th:>10.6f}  {cv:>10.6f}")
    print()

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    # Minimum of cos(theta) is -1 at theta=pi
    # We should be close to theta=pi after 50 steps from 1.8
    c_optimal = -1.0
    converged = abs(c_final - c_optimal) < 0.05

    print("Verification:")
    print(f"  Cost C({theta_test:.2f}) = cos({theta_test:.2f}) correct: {'PASS' if abs(c_test - math.cos(theta_test)) < 1e-10 else 'FAIL'}")
    print(f"  Gradient computation (param-shift vs analytical): {'PASS' if theta_match and phi_match else 'FAIL'}")
    print(f"  Gradient descent converged (C near -1.0): {'PASS' if converged else 'PARTIAL'}")
    print()

    print("Key Concepts:")
    print("  - Parameterized gates: RX(theta) = exp(-i theta/2 X)")
    print("  - Cost function: expectation value <psi(params)| O |psi(params)>")
    print("  - Parameter-shift rule: exact gradient for Pauli rotation gates")
    print("  - dC/dtheta = (C(theta+pi/2) - C(theta-pi/2)) / 2")
    print("  - Compatible with real quantum hardware (no access to state vector)")
    print("  - Gradient descent: theta <- theta - lr * dC/dtheta")
    print()
    print(f"Final: theta={theta_final:.6f} (target pi={math.pi:.6f}), C={c_final:.6f}")


if __name__ == "__main__":
    main()
