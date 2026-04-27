//! Python bindings for quantum error correction functionality.
//!
//! Exposes the rotated surface code, MWPM decoder, Union-Find decoder,
//! and PauliFrame tracking to Python.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use quantrs2_core::error_correction::{
    MwpmSurfaceDecoder, PauliFrame, PauliString, RotatedSurfaceCode, SyndromeDecoder,
    UnionFindDecoder,
};
use quantrs2_core::Pauli;

/// Python wrapper for the rotated planar surface code.
///
/// Represents a [[d², 1, d]] rotated planar surface code.
///
/// # Example
/// ```python
/// import quantrs2
/// code = quantrs2.qec.RotatedSurfaceCode(d=3)
/// print(code.n_k_d())  # (9, 1, 3)
/// ```
#[pyclass(name = "RotatedSurfaceCode")]
pub struct PyRotatedSurfaceCode {
    inner: RotatedSurfaceCode,
}

#[pymethods]
impl PyRotatedSurfaceCode {
    /// Create a new rotated surface code of distance `d`.
    #[new]
    pub fn new(d: usize) -> PyResult<Self> {
        if d < 2 {
            return Err(PyValueError::new_err("Surface code distance must be >= 2"));
        }
        Ok(Self {
            inner: RotatedSurfaceCode::new(d),
        })
    }

    /// Return (n, k, d): physical qubits, logical qubits, distance.
    pub fn n_k_d(&self) -> (usize, usize, usize) {
        self.inner.n_k_d()
    }

    /// Number of physical data qubits (= d²).
    pub fn n_data_qubits(&self) -> usize {
        self.inner.n_data_qubits()
    }

    /// Code distance.
    pub fn distance(&self) -> usize {
        self.inner.distance
    }

    /// Return list of X-stabilizer supports (each is a list of qubit indices).
    pub fn x_stabilizers(&self) -> Vec<Vec<usize>> {
        self.inner.x_stabilizers()
    }

    /// Return list of Z-stabilizer supports (each is a list of qubit indices).
    pub fn z_stabilizers(&self) -> Vec<Vec<usize>> {
        self.inner.z_stabilizers()
    }

    /// Logical X qubit support (left column).
    pub fn logical_x_qubits(&self) -> Vec<usize> {
        self.inner.logical_x_qubits()
    }

    /// Logical Z qubit support (top row).
    pub fn logical_z_qubits(&self) -> Vec<usize> {
        self.inner.logical_z_qubits()
    }

    /// Compute the syndrome for a given Pauli error.
    ///
    /// `error_paulis`: list of Pauli labels as strings ('I', 'X', 'Y', 'Z').
    ///
    /// Returns a boolean list [x_syndrome..., z_syndrome...].
    pub fn syndrome(&self, error_paulis: Vec<String>) -> PyResult<Vec<bool>> {
        let paulis = parse_paulis(&error_paulis)?;
        let ps = PauliString::new(paulis);
        self.inner
            .syndrome(&ps)
            .map_err(|e| PyValueError::new_err(format!("Syndrome computation failed: {e}")))
    }
}

/// Python wrapper for the MWPM surface code decoder.
///
/// Decodes syndromes produced by `RotatedSurfaceCode.syndrome()` using
/// minimum-weight perfect matching via bitmask DP.
#[pyclass(name = "MwpmDecoder")]
pub struct PyMwpmDecoder {
    inner: MwpmSurfaceDecoder,
}

#[pymethods]
impl PyMwpmDecoder {
    /// Create a new MWPM decoder for the given surface code.
    #[new]
    pub fn new(code: &PyRotatedSurfaceCode) -> Self {
        Self {
            inner: MwpmSurfaceDecoder::new(code.inner.clone()),
        }
    }

    /// Decode a syndrome and return the correction as a list of Pauli labels.
    ///
    /// `syndrome`: boolean list from `RotatedSurfaceCode.syndrome()`.
    ///
    /// Returns a list of Pauli strings ('I', 'X', 'Y', 'Z') of length n.
    pub fn decode(&self, syndrome: Vec<bool>) -> PyResult<Vec<String>> {
        let correction = self
            .inner
            .decode(&syndrome)
            .map_err(|e| PyValueError::new_err(format!("MWPM decode failed: {e}")))?;
        Ok(paulis_to_strings(&correction))
    }

    /// Decode and return the correction weight.
    pub fn decode_weight(&self, syndrome: Vec<bool>) -> PyResult<usize> {
        let correction = self
            .inner
            .decode(&syndrome)
            .map_err(|e| PyValueError::new_err(format!("MWPM decode failed: {e}")))?;
        Ok(correction.weight())
    }
}

/// Python wrapper for the Union-Find surface code decoder.
///
/// Near-linear time decoder based on Delfosse-Nickerson Union-Find algorithm.
#[pyclass(name = "UnionFindDecoder")]
pub struct PyUnionFindDecoder {
    inner: UnionFindDecoder,
}

#[pymethods]
impl PyUnionFindDecoder {
    /// Create a new Union-Find decoder for the given surface code.
    #[new]
    pub fn new(code: &PyRotatedSurfaceCode) -> Self {
        Self {
            inner: UnionFindDecoder::new(code.inner.clone()),
        }
    }

    /// Decode a syndrome and return the correction as a list of Pauli labels.
    ///
    /// `syndrome`: boolean list from `RotatedSurfaceCode.syndrome()`.
    ///
    /// Returns a list of Pauli strings ('I', 'X', 'Y', 'Z') of length n.
    pub fn decode(&self, syndrome: Vec<bool>) -> PyResult<Vec<String>> {
        let correction = self
            .inner
            .decode(&syndrome)
            .map_err(|e| PyValueError::new_err(format!("UF decode failed: {e}")))?;
        Ok(paulis_to_strings(&correction))
    }

    /// Decode and return the correction weight.
    pub fn decode_weight(&self, syndrome: Vec<bool>) -> PyResult<usize> {
        let correction = self
            .inner
            .decode(&syndrome)
            .map_err(|e| PyValueError::new_err(format!("UF decode failed: {e}")))?;
        Ok(correction.weight())
    }
}

/// Python wrapper for PauliFrame classical correction tracking.
///
/// Tracks accumulated Pauli corrections through Clifford gates without
/// explicitly storing quantum states.
///
/// # Example
/// ```python
/// frame = quantrs2.qec.PauliFrame(n=3)
/// frame.apply_x(0)
/// frame.apply_h(0)
/// # Now qubit 0 has Z error tracked
/// print(frame.measure_logical_z([0, 1, 2]))
/// ```
#[pyclass(name = "PauliFrame")]
pub struct PyPauliFrame {
    inner: PauliFrame,
}

#[pymethods]
impl PyPauliFrame {
    /// Create a new PauliFrame for `n` qubits (all identity).
    #[new]
    pub fn new(n: usize) -> Self {
        Self {
            inner: PauliFrame::new(n),
        }
    }

    /// Apply a Pauli correction string (list of 'I', 'X', 'Y', 'Z').
    pub fn apply_pauli_string(&mut self, paulis: Vec<String>) -> PyResult<()> {
        let ps = parse_paulis(&paulis)?;
        let pauli_str = PauliString::new(ps);
        self.inner.apply_pauli_string(&pauli_str);
        Ok(())
    }

    /// Propagate frame through a Hadamard gate on qubit `q`.
    pub fn commute_through_h(&mut self, q: usize) {
        self.inner.commute_through_h(q);
    }

    /// Propagate frame through an S gate on qubit `q`.
    pub fn commute_through_s(&mut self, q: usize) {
        self.inner.commute_through_s(q);
    }

    /// Propagate frame through a CNOT gate (control `ctrl`, target `tgt`).
    pub fn commute_through_cnot(&mut self, ctrl: usize, tgt: usize) {
        self.inner.commute_through_cnot(ctrl, tgt);
    }

    /// Check if the frame is all-identity (no tracked errors).
    pub fn is_identity(&self) -> bool {
        self.inner.is_identity()
    }

    /// Measure parity of X-frame over logical Z support → logical X error indicator.
    ///
    /// Returns `True` if there is a logical X error.
    pub fn measure_logical_x(&self, support: Vec<usize>) -> bool {
        self.inner.measure_logical_x(&support)
    }

    /// Measure parity of Z-frame over logical X support → logical Z error indicator.
    ///
    /// Returns `True` if there is a logical Z error.
    pub fn measure_logical_z(&self, support: Vec<usize>) -> bool {
        self.inner.measure_logical_z(&support)
    }

    /// Return the current X-frame as a list of booleans.
    pub fn x_frame(&self) -> Vec<bool> {
        self.inner.x_frame.clone()
    }

    /// Return the current Z-frame as a list of booleans.
    pub fn z_frame(&self) -> Vec<bool> {
        self.inner.z_frame.clone()
    }
}

// --- Helper functions ---

/// Parse a list of Pauli label strings into `Vec<Pauli>`.
fn parse_paulis(labels: &[String]) -> PyResult<Vec<Pauli>> {
    labels
        .iter()
        .map(|s| match s.as_str() {
            "I" => Ok(Pauli::I),
            "X" => Ok(Pauli::X),
            "Y" => Ok(Pauli::Y),
            "Z" => Ok(Pauli::Z),
            other => Err(PyValueError::new_err(format!(
                "Unknown Pauli label '{other}'; expected 'I', 'X', 'Y', or 'Z'"
            ))),
        })
        .collect()
}

/// Convert a `PauliString` into a list of label strings.
fn paulis_to_strings(ps: &PauliString) -> Vec<String> {
    ps.paulis
        .iter()
        .map(|p| match p {
            Pauli::I => "I".to_string(),
            Pauli::X => "X".to_string(),
            Pauli::Y => "Y".to_string(),
            Pauli::Z => "Z".to_string(),
        })
        .collect()
}

/// Register the `qec` submodule into the parent Python module.
pub fn register_qec_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent_module.py();
    let qec_module = PyModule::new(py, "qec")?;

    qec_module.add_class::<PyRotatedSurfaceCode>()?;
    qec_module.add_class::<PyMwpmDecoder>()?;
    qec_module.add_class::<PyUnionFindDecoder>()?;
    qec_module.add_class::<PyPauliFrame>()?;

    parent_module.add_submodule(&qec_module)?;
    Ok(())
}
