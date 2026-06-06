"""High-level 3D quantum state visualization helpers."""

try:
    from quantrs2._quantrs2 import QuantumState3DVisualizer  # noqa: F401
except ImportError:
    pass


def display_state_3d(state_amplitudes, n_qubits, kind="bloch_array"):
    """
    Display a quantum state using 3D visualization.

    Parameters
    ----------
    state_amplitudes : list[tuple[float, float]]
        List of ``(re, im)`` tuples for each computational basis state.
        Length must equal ``2**n_qubits``.
    n_qubits : int
        Number of qubits.
    kind : str
        Visualization type. One of:

        - ``"bloch_array"`` — per-qubit Bloch sphere grid
        - ``"qsphere"`` — Qiskit-style Q-sphere
        - ``"wigner"`` — discrete Wigner function (n=1, 2 only)
        - ``"husimi"`` — Husimi Q-distribution
        - ``"density_bars"`` — density matrix 3D bar plots

    Returns
    -------
    str
        Self-contained HTML string for the visualization.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the supported types.
    """
    viz = QuantumState3DVisualizer(state_amplitudes, n_qubits)
    dispatch = {
        "bloch_array": viz.bloch_array_html,
        "qsphere": viz.qsphere_html,
        "wigner": viz.wigner_html,
        "husimi": viz.husimi_html,
        "density_bars": viz.density_bars_html,
    }
    if kind not in dispatch:
        raise ValueError(
            f"Unknown visualization kind: {kind!r}. "
            f"Choose from {list(dispatch.keys())}"
        )
    return dispatch[kind]()
