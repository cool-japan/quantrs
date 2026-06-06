"""3D Quantum State Visualization — Tutorial 06.

Quantum states can be visualised in multiple ways:
- **Bloch sphere**: single-qubit states as points on a sphere
- **Q-sphere**: multi-qubit states projected onto a sphere
- **Amplitude plot**: bar chart of state amplitudes

QuantRS2 provides the ``QuantumStateVisualizer`` class (in the ``_core`` module)
that generates Plotly-based HTML visualisations.

**States demonstrated**:
1. |+> = (|0> + |1>) / sqrt(2) — single-qubit superposition
2. |Phi+> = (|00> + |11>) / sqrt(2) — Bell state (2 qubits)
3. |GHZ-3> = (|000> + |111>) / sqrt(2) — GHZ state (3 qubits)

Run as:
    python -m quantrs2.tutorials.06_3d_state_visualization
"""

from __future__ import annotations

import sys
import os
import math
import tempfile
from typing import List, Optional, Tuple

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    print("numpy not available — install with: pip install numpy")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Import the QuantumStateVisualizer from the _core native module
# ---------------------------------------------------------------------------

_VIZ_AVAILABLE = False
_QuantumStateVisualizer = None

try:
    from quantrs2._core import QuantumStateVisualizer as _QSV
    _QuantumStateVisualizer = _QSV
    _VIZ_AVAILABLE = True
except (ImportError, AttributeError):
    pass

if not _VIZ_AVAILABLE:
    # Try alternate import path
    try:
        import importlib
        _core = importlib.import_module("quantrs2._core")
        if hasattr(_core, "QuantumStateVisualizer"):
            _QuantumStateVisualizer = _core.QuantumStateVisualizer
            _VIZ_AVAILABLE = True
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# State vectors
# ---------------------------------------------------------------------------

def _plus_state() -> np.ndarray:
    """Single-qubit |+> = (|0> + |1>) / sqrt(2)."""
    return np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=complex)


def _bell_state() -> np.ndarray:
    """2-qubit Bell state |Phi+> = (|00> + |11>) / sqrt(2)."""
    inv = 1.0 / math.sqrt(2.0)
    return np.array([inv, 0.0, 0.0, inv], dtype=complex)


def _ghz3_state() -> np.ndarray:
    """3-qubit GHZ state = (|000> + |111>) / sqrt(2)."""
    inv = 1.0 / math.sqrt(2.0)
    state = np.zeros(8, dtype=complex)
    state[0] = inv   # |000>
    state[7] = inv   # |111>
    return state


def _fallback_html(label: str, state: np.ndarray) -> str:
    """Generate minimal HTML amplitude table when native visualizer unavailable."""
    rows = "".join(
        f"<tr><td>|{format(i, f'0{max(1, len(bin(len(state)-1))-2)}b')}&gt;</td>"
        f"<td>{amp.real:.4f} + {amp.imag:.4f}i</td>"
        f"<td>{abs(amp)**2:.4f}</td></tr>"
        for i, amp in enumerate(state)
    )
    return (
        f"<html><body><h2>{label}</h2><p>plotly not available — fallback table</p>"
        f"<table border='1'><tr><th>Basis</th><th>Amplitude</th><th>Prob</th></tr>"
        f"{rows}</table></body></html>"
    )


def _generate_viz_html(state: np.ndarray, kind: str) -> str:
    """Generate HTML for a given state and visualization kind."""
    if not _VIZ_AVAILABLE:
        return _fallback_html(kind, state)

    try:
        viz = _QuantumStateVisualizer(state)
        if kind == "bloch":
            if len(state) != 2:
                # bloch_sphere_html only works for single qubits
                return viz.amplitude_plot_html()
            return viz.bloch_sphere_html()
        elif kind == "amplitude":
            return viz.amplitude_plot_html()
        else:
            return viz.amplitude_plot_html()
    except Exception as exc:
        return _fallback_html(f"{kind} (error: {exc})", state)


# ---------------------------------------------------------------------------
# Save HTML files
# ---------------------------------------------------------------------------

def _save_html(content: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Tutorial 06: 3D Quantum State Visualization")
    print("=" * 60)
    print()

    if not _VIZ_AVAILABLE:
        print("Note: QuantumStateVisualizer not available in native bindings.")
        print("      Using fallback HTML tables instead of Plotly plots.")
        print("      Install a wheel built with the 'state_viz_3d' feature for")
        print("      full 3D visualization.")
        print()
    else:
        print(f"[Visualizer] Using QuantumStateVisualizer from quantrs2._core")
        print()

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    out_dir = os.path.join(tempfile.gettempdir(), "quantrs2_viz_demo")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print()

    # ------------------------------------------------------------------
    # States to visualize
    # ------------------------------------------------------------------
    states = [
        ("|+> (single qubit)", _plus_state(), "bloch"),
        ("|Phi+> Bell state (2 qubits)", _bell_state(), "amplitude"),
        ("|GHZ-3> (3 qubits)", _ghz3_state(), "amplitude"),
    ]

    files_saved: List[str] = []

    for label, state_vec, viz_kind in states:
        print(f"State: {label}")
        print(f"  Dimension: {len(state_vec)}")
        norm = float(np.sum(np.abs(state_vec) ** 2))
        print(f"  Norm: {norm:.6f}")
        print(f"  Non-zero amplitudes:")
        for idx, amp in enumerate(state_vec):
            if abs(amp) > 1e-10:
                n_q = max(1, len(bin(len(state_vec) - 1)) - 2)
                basis = format(idx, f"0{n_q}b")
                print(f"    |{basis}>: {amp.real:.4f} + {amp.imag:.4f}i  "
                      f"(prob={abs(amp)**2:.4f})")
        print()

        html = _generate_viz_html(state_vec, viz_kind)
        fname = f"{label.split()[0].lstrip('|').rstrip('>').replace('+', 'plus')}_viz.html"
        fpath = os.path.join(out_dir, fname)
        _save_html(html, fpath)
        files_saved.append(fpath)
        has_content = len(html) > 100
        print(f"  -> Saved: {fpath} ({len(html)} bytes, has_content={has_content})")
        print()

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    print("Verification:")
    all_ok = True
    for fpath in files_saved:
        exists = os.path.isfile(fpath)
        if exists:
            with open(fpath, "r", encoding="utf-8") as fh:
                content = fh.read()
            # Accept any HTML-like content: <html, <div, <table, or <body
            has_markup = any(tag in content.lower() for tag in ["<html", "<div", "<table", "<body"])
            ok = exists and has_markup and len(content) > 100
        else:
            ok = False
        print(f"  {os.path.basename(fpath)}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_ok = False
    print()

    print("Key Concepts:")
    print("  - Bloch sphere: |psi> = cos(theta/2)|0> + e^{iphi}sin(theta/2)|1>")
    print("  - Any single-qubit state is a point on the Bloch sphere")
    print("  - |+> sits on the equator (theta=pi/2, phi=0)")
    print("  - Amplitude plot: bar chart of |<basis|psi>|^2 for each basis state")
    print("  - Bell state has equal weight on |00> and |11>: maximally entangled")
    print("  - GHZ state shows macroscopic superposition across 3 qubits")
    print()

    if all_ok:
        print(f"Visualization files saved to {out_dir}")
    else:
        print("Warning: some visualization files had issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
