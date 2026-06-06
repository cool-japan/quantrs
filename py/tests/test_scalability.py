"""Python-level scalability smoke tests for QuantRS2 v0.2.0.

These tests verify that the quantrs2 Python bindings are importable and
that basic circuit construction + simulation works at small-to-medium
qubit counts.  All tests skip gracefully when the native extension is
not installed (e.g. on a fresh CI runner before `maturin develop`).
"""

import pytest

try:
    import quantrs2
    HAS_QUANTRS2 = True
except (ImportError, OSError):
    # OSError covers the case where the .so exists but cannot be loaded
    # (missing symbols, wrong platform, etc.)
    HAS_QUANTRS2 = False

pytestmark = pytest.mark.skipif(
    not HAS_QUANTRS2,
    reason="quantrs2 native extension not installed — run `maturin develop` first"
)


# ---------------------------------------------------------------------------
# Test 1: Package import and module attribute check
# ---------------------------------------------------------------------------
def test_imports_available():
    """Verify that quantrs2 is importable and exposes the expected symbols."""
    import quantrs2 as qr  # noqa: PLC0415

    assert qr is not None, "quantrs2 module should not be None"
    # PyCircuit is the minimal required symbol for Python-level circuit tests
    assert hasattr(qr, "PyCircuit"), (
        "quantrs2.PyCircuit not found — bindings may be out of date"
    )


# ---------------------------------------------------------------------------
# Test 2: 2-qubit Bell state via Python API
# ---------------------------------------------------------------------------
def test_bell_state_2q():
    """Create a Bell state with the Python API and verify it runs without error."""
    import quantrs2 as qr  # noqa: PLC0415

    circuit = qr.PyCircuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)

    result = circuit.run()
    # run() should return a truthy result object (not None)
    assert result is not None, "circuit.run() returned None for a Bell circuit"


# ---------------------------------------------------------------------------
# Test 3: 5-qubit GHZ state construction
# ---------------------------------------------------------------------------
def test_ghz_state_5q():
    """Construct a 5-qubit GHZ state via the Python API."""
    import quantrs2 as qr  # noqa: PLC0415

    n = 5
    circuit = qr.PyCircuit(n)
    circuit.h(0)
    for k in range(1, n):
        circuit.cnot(0, k)

    result = circuit.run()
    assert result is not None, "circuit.run() returned None for a 5-qubit GHZ circuit"


# ---------------------------------------------------------------------------
# Test 4: Repeated single-qubit circuit (idempotency)
# ---------------------------------------------------------------------------
def test_repeated_h_gate_idempotency():
    """H applied twice is identity; verify amplitudes return to |0>."""
    import quantrs2 as qr  # noqa: PLC0415

    circuit = qr.PyCircuit(1)
    circuit.h(0)
    circuit.h(0)  # H^2 = I

    result = circuit.run()
    assert result is not None, "circuit.run() returned None"

    # If the binding exposes probabilities, check |0> has probability ≈ 1.0
    if hasattr(result, "probabilities"):
        probs = result.probabilities()
        if probs:
            assert abs(probs[0] - 1.0) < 1e-6, (
                f"P(|0>) after H^2 should be 1.0, got {probs[0]}"
            )


# ---------------------------------------------------------------------------
# Test 5: Circuit building does not raise for up to 10 qubits
# ---------------------------------------------------------------------------
def test_circuit_sizes_1_to_10():
    """Constructing circuits for n=1..10 qubits should not raise."""
    import quantrs2 as qr  # noqa: PLC0415

    for n in range(1, 11):
        circuit = qr.PyCircuit(n)
        circuit.h(0)
        # Just verify construction + one gate; do not run (may be slow at n=10)
        assert circuit is not None, f"PyCircuit({n}) returned None"
