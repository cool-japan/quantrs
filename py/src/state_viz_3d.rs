//! Python bindings for 3D quantum state visualization.
//!
//! Exposes `PyQuantumState3DVisualizer` from `quantrs2-core` to the
//! Python module.  Registration is wired in `py/src/lib.rs` by
//! the Wave 2 binding step.

use pyo3::prelude::*;
use quantrs2_core::state_visualization_3d::PyQuantumState3DVisualizer;

/// Register the `QuantumState3DVisualizer` class into a parent Python module.
pub fn register_state_viz_3d_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    parent_module.add_class::<PyQuantumState3DVisualizer>()?;
    Ok(())
}
