"""Bell State Preparation and Measurement — Tutorial 01.

A Bell state (also called an EPR pair) is a maximally entangled quantum state
of two qubits.  The canonical |Phi+> Bell state is:

    |Phi+> = (|00> + |11>) / sqrt(2)

Its key property: measuring either qubit in the computational basis instantly
determines the result of measuring the other.  This is quantum entanglement.

**Circuit construction**:

    qubit 0: --[H]--[*]--   (Hadamard, then CNOT control)
    qubit 1: -------[X]--   (CNOT target)

H on qubit 0 creates a superposition (|0> + |1>) / sqrt(2).
CNOT then entangles: control=|0> -> target unchanged; control=|1> -> target flipped.
Result: (|00> + |11>) / sqrt(2).

**Expected measurement statistics**:
    P(00) = 0.5,  P(11) = 0.5,  P(01) = P(10) = 0.0

Run as:
    python -m quantrs2.tutorials.01_bell_state
"""

from __future__ import annotations

import sys
import math
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Probe the QuantRS2 native bindings.
# We import the inner 'quantrs2' sub-module (the .abi3.so) directly.
# The outer quantrs2 package patches PySimulationResult in __init__.py
# which makes amplitudes a property returning []; the native module keeps
# it as a callable method returning the full list.
# ---------------------------------------------------------------------------

_CIRCUIT_AVAILABLE = False
_PyCircuit = None

try:
    # Access the compiled extension registered under quantrs2.quantrs2
    import importlib
    _native = importlib.import_module("quantrs2.quantrs2")
    _PyCircuit = _native.PyCircuit
    _CIRCUIT_AVAILABLE = True
except (ImportError, AttributeError):
    pass


# ---------------------------------------------------------------------------
# Pure-numpy fallback: manual Bell state construction
# ---------------------------------------------------------------------------

def _build_bell_numpy() -> Tuple[List[complex], int]:
    """Construct Bell state |Phi+> using matrix arithmetic (no native bindings)."""
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    # Hadamard matrix (2x2)
    H = [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]]

    # Tensor H otimes I for 2-qubit space (4x4) acting on qubit 0
    HI: List[List[float]] = [[0.0] * 4 for _ in range(4)]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                HI[2 * i + k][2 * j + k] = H[i][j]

    # CNOT matrix (qubit-0 is control, qubit-1 is target)
    CNOT = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]

    # Initial state |00>
    state: List[complex] = [complex(1.0), complex(0.0), complex(0.0), complex(0.0)]

    # Apply H otimes I
    next_state = [complex(0.0)] * 4
    for i in range(4):
        for j in range(4):
            next_state[i] += HI[i][j] * state[j]
    state = next_state

    # Apply CNOT
    next_state = [complex(0.0)] * 4
    for i in range(4):
        for j in range(4):
            next_state[i] += CNOT[i][j] * state[j]
    state = next_state

    return state, 2


def _compute_probabilities(amplitudes: List[complex], n_qubits: int) -> Dict[str, float]:
    """Convert amplitude list to probability dict keyed by basis-state strings."""
    probs: Dict[str, float] = {}
    for idx, amp in enumerate(amplitudes):
        prob = abs(amp) ** 2
        if prob > 1e-12:
            label = format(idx, f"0{n_qubits}b")
            probs[label] = prob
    return probs


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Tutorial 01: Bell State Preparation and Measurement")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # 1) Build the Bell state
    # ------------------------------------------------------------------
    if _CIRCUIT_AVAILABLE:
        print("[Method] Using QuantRS2 native circuit (PyCircuit)")
        circuit = _PyCircuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        result = circuit.run()
        # state_probabilities() returns dict {basis: prob}
        probs: Dict[str, float] = result.state_probabilities()
        n_qubits: int = len(next(iter(probs), "00")) if probs else 2
        # Reconstruct amplitudes from probabilities (real-valued for Bell state)
        import math as _math
        prob_list = result.probabilities()
        amplitudes: List[complex] = [
            complex(_math.sqrt(max(p, 0.0))) for p in prob_list
        ]
    else:
        print("[Method] Using pure-numpy fallback (no native bindings found)")
        amplitudes, n_qubits = _build_bell_numpy()
        probs = _compute_probabilities(amplitudes, n_qubits)

    # ------------------------------------------------------------------
    # 2) Print the circuit description
    # ------------------------------------------------------------------
    print()
    print("Circuit (2 qubits):")
    print("  q0: --|H|--[*]--")
    print("  q1: ---------[X]--")
    print()
    print("  Gate sequence: H(q0), CNOT(ctrl=q0, tgt=q1)")
    print()

    # ------------------------------------------------------------------
    # 3) Print state-vector amplitudes
    # ------------------------------------------------------------------
    print("State-vector amplitudes:")
    print(f"  {'Basis':>4}  {'Re':>10}  {'Im':>10}  {'|amp|^2':>10}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}")
    for idx, amp in enumerate(amplitudes):
        basis = format(idx, f"0{n_qubits}b")
        prob = abs(amp) ** 2
        print(f"  |{basis}>  {amp.real:>10.6f}  {amp.imag:>10.6f}  {prob:>10.6f}")
    print()

    # ------------------------------------------------------------------
    # 4) Print measurement probabilities and verify
    # ------------------------------------------------------------------
    print("Measurement probabilities:")
    print(f"  {'State':>5}  {'Probability':>12}  {'Expected':>10}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*10}")
    expected = {"00": 0.5, "11": 0.5, "01": 0.0, "10": 0.0}
    for state_label in ["00", "01", "10", "11"]:
        p_actual = probs.get(state_label, 0.0)
        p_expect = expected[state_label]
        flag = "OK" if abs(p_actual - p_expect) < 0.01 else "MISMATCH"
        print(f"  |{state_label}>   {p_actual:>12.6f}  {p_expect:>10.3f}  [{flag}]")
    print()

    # ------------------------------------------------------------------
    # 5) Verify key properties
    # ------------------------------------------------------------------
    p00 = probs.get("00", 0.0)
    p11 = probs.get("11", 0.0)
    p01 = probs.get("01", 0.0)
    p10 = probs.get("10", 0.0)

    total = p00 + p11 + p01 + p10
    entanglement_ok = (
        abs(p00 - 0.5) < 0.01
        and abs(p11 - 0.5) < 0.01
        and p01 < 0.01
        and p10 < 0.01
    )
    norm_ok = abs(total - 1.0) < 1e-6

    print("Verification:")
    print(f"  Normalisation (sum of probs = 1): {'PASS' if norm_ok else 'FAIL'}")
    print(f"  Bell-state property (P00=P11=0.5, P01=P10=0): {'PASS' if entanglement_ok else 'FAIL'}")
    print()

    # ------------------------------------------------------------------
    # 6) Conceptual summary
    # ------------------------------------------------------------------
    print("Key Concepts:")
    print("  - Hadamard gate: puts q0 into superposition |0> -> (|0>+|1>)/sqrt(2)")
    print("  - CNOT gate: entangles q0 and q1 (flips q1 when q0=|1>)")
    print("  - After both gates: state is (|00>+|11>)/sqrt(2) — a Bell state")
    print("  - Measuring q0=|0> instantly forces q1=|0>, and vice versa")
    print("  - This is quantum entanglement: correlations stronger than classical")
    print()

    if entanglement_ok and norm_ok:
        print("Bell state correctly prepared and verified.")
    else:
        print("Warning: Bell state verification showed unexpected values.")
        sys.exit(1)


if __name__ == "__main__":
    main()
