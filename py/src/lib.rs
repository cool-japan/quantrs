//! Python bindings for the quantrs framework.
//!
//! This crate provides Python bindings using PyO3,
//! allowing quantrs to be used from Python.

use pyo3::prelude::*;

/// Python module for quantrs
#[pymodule]
fn quantrs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}