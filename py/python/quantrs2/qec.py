"""
Quantum Error Correction Module

High-level Python helpers for quantum error correction using surface codes.
Wraps the native Rust bindings exposed via ``quantrs2.qec``.

Supported:
- Rotated planar surface code [[d², 1, d]]
- MWPM (minimum-weight perfect matching) decoder
- Union-Find decoder (near-linear time)
- PauliFrame classical Clifford tracking
- Threshold simulation utility
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

try:
    # The compiled extension is exposed as the ``quantrs2`` sub-attribute
    # within the ``quantrs2`` Python package (i.e. ``quantrs2.quantrs2``).
    from . import quantrs2 as _native_mod  # type: ignore[attr-defined]
    _qec = _native_mod.qec  # type: ignore[attr-defined]
    _BINDINGS_AVAILABLE = True
except (ImportError, AttributeError):
    _qec = None  # type: ignore[assignment]
    _BINDINGS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Re-export native classes when bindings are present
# ---------------------------------------------------------------------------

if _BINDINGS_AVAILABLE:
    RotatedSurfaceCode = _qec.RotatedSurfaceCode
    MwpmDecoder = _qec.MwpmDecoder
    UnionFindDecoder = _qec.UnionFindDecoder
    PauliFrame = _qec.PauliFrame

else:
    # Pure-Python stubs so the module can be imported even without a compiled
    # extension (e.g., during documentation builds or CI without a wheel).

    class RotatedSurfaceCode:  # type: ignore[no-redef]
        """Stub: rotated planar surface code [[d², 1, d]]."""

        def __init__(self, d: int) -> None:
            if d < 2:
                raise ValueError("Surface code distance must be >= 2")
            self.distance = d

        def n_k_d(self) -> Tuple[int, int, int]:
            d = self.distance
            return d * d, 1, d

        def n_data_qubits(self) -> int:
            return self.distance ** 2

        def x_stabilizers(self) -> List[List[int]]:
            return []

        def z_stabilizers(self) -> List[List[int]]:
            return []

        def logical_x_qubits(self) -> List[int]:
            return []

        def logical_z_qubits(self) -> List[int]:
            return []

        def syndrome(self, error_paulis: List[str]) -> List[bool]:
            n = self.distance ** 2
            if len(error_paulis) != n:
                raise ValueError(
                    f"error_paulis length {len(error_paulis)} != n_data {n}"
                )
            return []

    class MwpmDecoder:  # type: ignore[no-redef]
        """Stub: MWPM surface-code decoder."""

        def __init__(self, code: RotatedSurfaceCode) -> None:
            self._code = code

        def decode(self, syndrome: List[bool]) -> List[str]:
            return ["I"] * self._code.n_data_qubits()

        def decode_weight(self, syndrome: List[bool]) -> int:
            return 0

    class UnionFindDecoder:  # type: ignore[no-redef]
        """Stub: Union-Find surface-code decoder."""

        def __init__(self, code: RotatedSurfaceCode) -> None:
            self._code = code

        def decode(self, syndrome: List[bool]) -> List[str]:
            return ["I"] * self._code.n_data_qubits()

        def decode_weight(self, syndrome: List[bool]) -> int:
            return 0

    class PauliFrame:  # type: ignore[no-redef]
        """Stub: classical Pauli frame for Clifford tracking."""

        def __init__(self, n: int) -> None:
            self.x_frame: List[bool] = [False] * n
            self.z_frame: List[bool] = [False] * n
            self._n = n

        def apply_pauli_string(self, paulis: List[str]) -> None:
            for i, p in enumerate(paulis):
                if p == "X":
                    self.x_frame[i] = not self.x_frame[i]
                elif p == "Z":
                    self.z_frame[i] = not self.z_frame[i]
                elif p == "Y":
                    self.x_frame[i] = not self.x_frame[i]
                    self.z_frame[i] = not self.z_frame[i]

        def commute_through_h(self, q: int) -> None:
            self.x_frame[q], self.z_frame[q] = self.z_frame[q], self.x_frame[q]

        def commute_through_s(self, q: int) -> None:
            if self.x_frame[q]:
                self.z_frame[q] = not self.z_frame[q]

        def commute_through_cnot(self, ctrl: int, tgt: int) -> None:
            if self.x_frame[ctrl]:
                self.x_frame[tgt] = not self.x_frame[tgt]
            if self.z_frame[tgt]:
                self.z_frame[ctrl] = not self.z_frame[ctrl]

        def is_identity(self) -> bool:
            return not any(self.x_frame) and not any(self.z_frame)

        def measure_logical_x(self, support: List[int]) -> bool:
            return sum(1 for q in support if self.x_frame[q]) % 2 == 1

        def measure_logical_z(self, support: List[int]) -> bool:
            return sum(1 for q in support if self.z_frame[q]) % 2 == 1


# ---------------------------------------------------------------------------
# Pauli helpers
# ---------------------------------------------------------------------------

_PAULI_LABELS: Tuple[str, ...] = ("I", "X", "Y", "Z")


def _random_pauli_error(
    n: int,
    error_rate: float,
    rng_state: Optional[object] = None,
) -> List[str]:
    """Return a random single-qubit depolarizing error on *n* qubits."""
    import random as _random

    paulis: List[str] = []
    for _ in range(n):
        if _random.random() < error_rate:
            paulis.append(_random.choice(("X", "Y", "Z")))
        else:
            paulis.append("I")
    return paulis


def _error_has_logical_x(
    correction: List[str],
    error: List[str],
    logical_x_support: List[int],
) -> bool:
    """Return True if the net residual has odd logical-X parity."""
    # Combine error + correction on the logical support
    count = 0
    for q in logical_x_support:
        e = error[q]
        c = correction[q]
        # Combined Pauli = c · e; count X-component parity
        x_e = e in ("X", "Y")
        x_c = c in ("X", "Y")
        if x_e != x_c:
            count += 1
    return count % 2 == 1


def _error_has_logical_z(
    correction: List[str],
    error: List[str],
    logical_z_support: List[int],
) -> bool:
    """Return True if the net residual has odd logical-Z parity."""
    count = 0
    for q in logical_z_support:
        e = error[q]
        c = correction[q]
        z_e = e in ("Z", "Y")
        z_c = c in ("Z", "Y")
        if z_e != z_c:
            count += 1
    return count % 2 == 1


# ---------------------------------------------------------------------------
# Threshold simulation
# ---------------------------------------------------------------------------

def simulate_threshold(
    distance_range: Sequence[int] = (3, 5, 7),
    error_rates: Optional[Sequence[float]] = None,
    shots: int = 100,
    decoder: str = "mwpm",
) -> Dict[int, Dict[float, float]]:
    """Rough threshold simulation for the rotated surface code.

    For each code distance *d* and each physical error rate *p*, this function
    runs *shots* Monte-Carlo trials and returns the logical error rate.

    Parameters
    ----------
    distance_range:
        Code distances to evaluate (must each be >= 2).
    error_rates:
        Physical error rates to sweep over.  Defaults to a 10-point log-space
        range from 0.5 % to 20 %.
    shots:
        Number of Monte-Carlo trials per (distance, error_rate) pair.
    decoder:
        Which decoder to use: ``"mwpm"`` or ``"union_find"``.

    Returns
    -------
    dict
        Nested dict ``{d: {p: logical_error_rate}}``.

    Example
    -------
    >>> results = simulate_threshold(distance_range=(3, 5), shots=50)
    >>> # results[3][0.05] is the logical error rate for d=3 at 5% physical error
    """
    import math
    import random

    if error_rates is None:
        # 10 points log-spaced between 0.5 % and 20 %
        error_rates = [
            0.005 * (0.20 / 0.005) ** (i / 9.0) for i in range(10)
        ]

    results: Dict[int, Dict[float, float]] = {}

    for d in distance_range:
        if d < 2:
            raise ValueError(f"distance must be >= 2, got {d}")

        code = RotatedSurfaceCode(d=d)
        n = code.n_data_qubits()
        lx_support = code.logical_x_qubits()
        lz_support = code.logical_z_qubits()

        if decoder == "mwpm":
            dec = MwpmDecoder(code)
        elif decoder == "union_find":
            dec = UnionFindDecoder(code)
        else:
            raise ValueError(f"Unknown decoder '{decoder}'; choose 'mwpm' or 'union_find'")

        results[d] = {}

        for p in error_rates:
            failures = 0

            for _ in range(shots):
                error = _random_pauli_error(n, p)
                syndrome = code.syndrome(error)
                correction = dec.decode(syndrome)

                # A logical failure occurs when the net Pauli (correction ∘ error)
                # anti-commutes with the logical operator.
                failed_x = _error_has_logical_x(correction, error, lx_support)
                failed_z = _error_has_logical_z(correction, error, lz_support)

                if failed_x or failed_z:
                    failures += 1

            results[d][p] = failures / shots

    return results
