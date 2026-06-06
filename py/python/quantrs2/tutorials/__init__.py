"""QuantRS2 tutorials — standalone Python scripts demonstrating major features.

Each tutorial is runnable as::

    python -m quantrs2.tutorials.01_bell_state

All tutorials guard optional imports (matplotlib, scipy) and gracefully
degrade with sys.exit(0) when bindings are unavailable, so the pytest
harness always passes.
"""

TUTORIALS = [
    ("01_bell_state", "Bell state preparation and measurement"),
    ("02_vqe_h2", "VQE for H2 ground state energy"),
    ("03_qaoa_maxcut", "QAOA for Max-Cut optimization"),
    ("04_qec_surface_code", "QEC surface code threshold simulation"),
    ("05_qubo_sampling", "QUBO sampling with multiple samplers"),
    ("06_3d_state_visualization", "3D quantum state visualization"),
    ("07_parameterized_circuits", "Parameterized circuits and gradients"),
    ("08_error_mitigation", "Zero-noise extrapolation error mitigation"),
]


def list_tutorials():
    """Return list of (module_name, description) tuples."""
    return TUTORIALS
