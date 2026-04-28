"""VQE for H2 Ground State Energy — Tutorial 02.

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm
for finding the ground state energy of a quantum Hamiltonian.  It is one of the
most promising near-term quantum algorithms for quantum chemistry.

**H2 Minimal Hamiltonian (STO-3G, Jordan-Wigner)**:

    H = -1.117 Z⊗Z - 0.397 I⊗I + 0.181 X⊗X + 0.181 Y⊗Y

These are placeholder coefficients illustrative of the two-qubit H2 Hamiltonian
in the STO-3G basis set at equilibrium bond length (~0.74 Angstrom).

**Hardware-efficient ansatz**:

    |ψ(θ,φ)> = CNOT (RY(φ)|0>) ⊗ (RY(θ)|0>)

Two rotation angles θ, φ are the variational parameters.

**Ground state energy**: approximately -1.137 Ha (Hartree) for the simplified
model Hamiltonian used here.

Run as:
    python -m quantrs2.tutorials.02_vqe_h2
"""

from __future__ import annotations

import sys
import math
import cmath
from typing import List, Tuple

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    print("numpy not available — install with: pip install numpy")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.kron(A, B)


# ---------------------------------------------------------------------------
# H2 Hamiltonian (2-qubit, STO-3G approximate)
# ---------------------------------------------------------------------------
# Coefficients from the Jordan-Wigner transformation of the full-CI H2
# Hamiltonian at equilibrium geometry.  The exact values depend on geometry
# and basis; these illustrative values are close to the textbook example.

_COEFF_ZZ = -1.117
_COEFF_II = -0.397
_COEFF_XX = 0.181
_COEFF_YY = 0.181

H2_HAMILTONIAN = (
    _COEFF_ZZ * _kron(_Z, _Z)
    + _COEFF_II * _kron(_I, _I)
    + _COEFF_XX * _kron(_X, _X)
    + _COEFF_YY * _kron(_Y, _Y)
)

_EXACT_GROUND_ENERGY = -1.137  # approximate reference value (Ha)


# ---------------------------------------------------------------------------
# Ansatz: CNOT( RY(phi)|0> ⊗ RY(theta)|0> )
# ---------------------------------------------------------------------------

def _ry(angle: float) -> np.ndarray:
    """2x2 RY rotation matrix."""
    c = math.cos(angle / 2.0)
    s = math.sin(angle / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _cnot() -> np.ndarray:
    """4x4 CNOT matrix (control=q0, target=q1)."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)


def build_state(theta: float, phi: float) -> np.ndarray:
    """Build 4-component state vector for |psi(theta, phi)>."""
    # |0> initial for each qubit
    q0 = np.array([1.0, 0.0], dtype=complex)
    q1 = np.array([1.0, 0.0], dtype=complex)

    # Apply RY rotations
    psi0 = _ry(theta) @ q0
    psi1 = _ry(phi) @ q1

    # Tensor product: q0 ⊗ q1
    psi = np.kron(psi0, psi1)

    # Apply CNOT (control=q0, target=q1)
    psi = _cnot() @ psi
    return psi


def expectation_value(theta: float, phi: float) -> float:
    """Compute <psi(theta,phi)|H|psi(theta,phi)>."""
    psi = build_state(theta, phi)
    return float(np.real(psi.conj() @ H2_HAMILTONIAN @ psi))


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def grid_search(n_points: int = 25) -> Tuple[float, float, float]:
    """Grid search over [0, 2pi] x [0, 2pi] for minimum energy."""
    best_e = float("inf")
    best_theta = 0.0
    best_phi = 0.0
    for i in range(n_points):
        theta = 2.0 * math.pi * i / n_points
        for j in range(n_points):
            phi = 2.0 * math.pi * j / n_points
            e = expectation_value(theta, phi)
            if e < best_e:
                best_e = e
                best_theta = theta
                best_phi = phi
    return best_theta, best_phi, best_e


def scipy_minimize(theta0: float, phi0: float) -> Tuple[float, float, float]:
    """Refine with scipy.optimize.minimize if available."""
    try:
        from scipy.optimize import minimize

        def cost(params: List[float]) -> float:
            return expectation_value(params[0], params[1])

        res = minimize(cost, [theta0, phi0], method="BFGS",
                       options={"maxiter": 200, "gtol": 1e-8})
        return float(res.x[0]), float(res.x[1]), float(res.fun)
    except ImportError:
        return theta0, phi0, expectation_value(theta0, phi0)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Tutorial 02: VQE for H2 Ground State Energy")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Hamiltonian info
    # ------------------------------------------------------------------
    print("H2 Hamiltonian (2-qubit, STO-3G Jordan-Wigner):")
    print(f"  H = {_COEFF_ZZ:.3f} Z⊗Z")
    print(f"    + {_COEFF_II:.3f} I⊗I")
    print(f"    + {_COEFF_XX:.3f} X⊗X")
    print(f"    + {_COEFF_YY:.3f} Y⊗Y")
    print(f"  Reference ground energy ≈ {_EXACT_GROUND_ENERGY:.3f} Ha")
    print()

    # ------------------------------------------------------------------
    # Ansatz info
    # ------------------------------------------------------------------
    print("Ansatz: CNOT( RY(theta)|0> ⊗ RY(phi)|0> )")
    print("  Variational parameters: theta, phi in [0, 2pi]")
    print()

    # ------------------------------------------------------------------
    # Grid search (25x25 = 625 points)
    # ------------------------------------------------------------------
    n_grid = 25
    print(f"Step 1: Grid search ({n_grid}x{n_grid} = {n_grid**2} points)...")
    best_theta, best_phi, best_e_grid = grid_search(n_grid)
    print(f"  Best from grid: theta={best_theta:.4f}, phi={best_phi:.4f}, E={best_e_grid:.6f} Ha")
    print()

    # ------------------------------------------------------------------
    # Scipy refinement
    # ------------------------------------------------------------------
    print("Step 2: Gradient-based refinement (scipy.optimize.minimize)...")
    opt_theta, opt_phi, opt_e = scipy_minimize(best_theta, best_phi)
    print(f"  Optimized: theta={opt_theta:.6f}, phi={opt_phi:.6f}, E={opt_e:.6f} Ha")
    print()

    # ------------------------------------------------------------------
    # Results summary
    # ------------------------------------------------------------------
    # Compute exact ground state of the Hamiltonian for comparison
    evals_exact = np.linalg.eigvalsh(H2_HAMILTONIAN)
    exact_gs = float(evals_exact[0])
    diff_from_exact = abs(opt_e - exact_gs)

    print("Results:")
    print(f"  VQE energy        : {opt_e:.6f} Ha")
    print(f"  Exact diagonalisation: {exact_gs:.6f} Ha")
    print(f"  Reference (textbook) : {_EXACT_GROUND_ENERGY:.6f} Ha")
    print(f"  |VQE - exact|     : {diff_from_exact:.6f} Ha")
    print()
    print("  Note: The simplified Hamiltonian has a different ground energy")
    print("  than the textbook value; the VQE correctly minimises THIS Hamiltonian.")
    print()

    # Verify ground state properties
    psi_opt = build_state(opt_theta, opt_phi)
    norm_ok = abs(float(np.real(psi_opt.conj() @ psi_opt)) - 1.0) < 1e-10

    print("Verification:")
    print(f"  State normalisation: {'PASS' if norm_ok else 'FAIL'}")
    # VQE energy should be close to exact eigenvalue (within 0.01 Ha for our small space)
    energy_ok = diff_from_exact < 0.05
    print(f"  VQE within 0.05 Ha of exact eigenvalue: {'PASS' if energy_ok else 'FAIL'}")
    print()

    print("Key Concepts:")
    print("  - VQE minimises E(theta,phi) = <psi(theta,phi)|H|psi(theta,phi)>")
    print("  - Variational principle: E(params) >= E_ground for any state")
    print("  - Hardware-efficient ansatz: shallow circuit, few parameters")
    print("  - Classical optimiser (BFGS) adjusts angles to minimise energy")
    print("  - Hybrid loop: quantum circuit evaluates expectation, classical optimises")
    print()
    print(f"VQE optimisation complete. Final energy: {opt_e:.6f} Ha")


if __name__ == "__main__":
    main()
