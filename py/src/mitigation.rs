//! Quantum error mitigation techniques for Python bindings.
//!
//! This module provides access to various error mitigation methods including:
//! - Zero-Noise Extrapolation (ZNE)
//! - Probabilistic Error Cancellation (PEC)
//! - Virtual Distillation
//! - Symmetry Verification
//!
//! This module requires the `device` feature to be enabled.

#![cfg(feature = "device")]

use crate::measurement::PyMeasurementResult;
use crate::{CircuitOp, PyCircuit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use quantrs2_device::zero_noise_extrapolation::{
    CircuitFolder, ExtrapolationFitter, ExtrapolationMethod, NoiseScalingMethod, Observable,
    ZNEConfig, ZNEResult,
};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use std::collections::HashMap;

/// Zero-Noise Extrapolation configuration
#[pyclass(name = "ZNEConfig")]
#[derive(Clone)]
pub struct PyZNEConfig {
    inner: ZNEConfig,
}

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PyZNEConfig {
    #[new]
    #[pyo3(signature = (scale_factors=None, scaling_method=None, extrapolation_method=None, bootstrap_samples=None, confidence_level=None))]
    fn new(
        scale_factors: Option<Vec<f64>>,
        scaling_method: Option<&str>,
        extrapolation_method: Option<&str>,
        bootstrap_samples: Option<usize>,
        confidence_level: Option<f64>,
    ) -> PyResult<Self> {
        let mut config = ZNEConfig::default();

        if let Some(factors) = scale_factors {
            config.scale_factors = factors;
        }

        if let Some(method) = scaling_method {
            config.scaling_method = match method {
                "global" => NoiseScalingMethod::GlobalFolding,
                "local" => NoiseScalingMethod::LocalFolding,
                "pulse" => NoiseScalingMethod::PulseStretching,
                "digital" => NoiseScalingMethod::DigitalRepetition,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown scaling method: {method}"
                    )))
                }
            };
        }

        if let Some(method) = extrapolation_method {
            config.extrapolation_method = match method {
                "linear" => ExtrapolationMethod::Linear,
                "polynomial2" => ExtrapolationMethod::Polynomial(2),
                "polynomial3" => ExtrapolationMethod::Polynomial(3),
                "exponential" => ExtrapolationMethod::Exponential,
                "richardson" => ExtrapolationMethod::Richardson,
                "adaptive" => ExtrapolationMethod::Adaptive,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown extrapolation method: {method}"
                    )))
                }
            };
        }

        if let Some(samples) = bootstrap_samples {
            config.bootstrap_samples = Some(samples);
        }

        if let Some(level) = confidence_level {
            config.confidence_level = level;
        }

        Ok(Self { inner: config })
    }

    #[getter]
    fn scale_factors(&self) -> Vec<f64> {
        self.inner.scale_factors.clone()
    }

    #[setter]
    fn set_scale_factors(&mut self, factors: Vec<f64>) {
        self.inner.scale_factors = factors;
    }

    #[getter]
    fn scaling_method(&self) -> String {
        match self.inner.scaling_method {
            NoiseScalingMethod::GlobalFolding => "global".to_string(),
            NoiseScalingMethod::LocalFolding => "local".to_string(),
            NoiseScalingMethod::PulseStretching => "pulse".to_string(),
            NoiseScalingMethod::DigitalRepetition => "digital".to_string(),
        }
    }

    #[getter]
    fn extrapolation_method(&self) -> String {
        match self.inner.extrapolation_method {
            ExtrapolationMethod::Linear => "linear".to_string(),
            ExtrapolationMethod::Polynomial(n) => format!("polynomial{n}"),
            ExtrapolationMethod::Exponential => "exponential".to_string(),
            ExtrapolationMethod::Richardson => "richardson".to_string(),
            ExtrapolationMethod::Adaptive => "adaptive".to_string(),
        }
    }

    #[getter]
    fn bootstrap_samples(&self) -> Option<usize> {
        self.inner.bootstrap_samples
    }

    #[getter]
    fn confidence_level(&self) -> f64 {
        self.inner.confidence_level
    }

    fn __repr__(&self) -> String {
        format!(
            "ZNEConfig(scale_factors={:?}, scaling_method='{}', extrapolation_method='{}', bootstrap_samples={:?}, confidence_level={})",
            self.inner.scale_factors,
            self.scaling_method(),
            self.extrapolation_method(),
            self.inner.bootstrap_samples,
            self.inner.confidence_level
        )
    }
}

/// Result from Zero-Noise Extrapolation
#[pyclass(name = "ZNEResult")]
pub struct PyZNEResult {
    inner: ZNEResult,
}

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PyZNEResult {
    #[getter]
    fn mitigated_value(&self) -> f64 {
        self.inner.mitigated_value
    }

    #[getter]
    fn error_estimate(&self) -> Option<f64> {
        self.inner.error_estimate
    }

    #[getter]
    fn raw_data(&self, py: Python) -> PyResult<Py<PyAny>> {
        let list = PyList::empty(py);
        for (scale, value) in &self.inner.raw_data {
            let tuple = (scale, value);
            list.append(tuple)?;
        }
        Ok(list.into())
    }

    #[getter]
    fn fit_params(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        let arr = Array1::from_vec(self.inner.fit_params.clone());
        arr.into_pyarray(py).into()
    }

    #[getter]
    fn r_squared(&self) -> f64 {
        self.inner.r_squared
    }

    #[getter]
    fn extrapolation_fn(&self) -> String {
        self.inner.extrapolation_fn.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ZNEResult(mitigated_value={}, error_estimate={:?}, r_squared={}, function='{}')",
            self.inner.mitigated_value,
            self.inner.error_estimate,
            self.inner.r_squared,
            self.inner.extrapolation_fn
        )
    }
}

/// Observable for expectation value calculation
#[pyclass(name = "Observable")]
#[derive(Clone)]
pub struct PyObservable {
    inner: Observable,
}

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PyObservable {
    #[new]
    #[pyo3(signature = (pauli_string, coefficient=1.0))]
    fn new(pauli_string: Vec<(usize, String)>, coefficient: f64) -> PyResult<Self> {
        // Validate Pauli strings
        for (_, pauli) in &pauli_string {
            if !["I", "X", "Y", "Z"].contains(&pauli.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "Invalid Pauli operator: {pauli}"
                )));
            }
        }

        Ok(Self {
            inner: Observable {
                pauli_string,
                coefficient,
            },
        })
    }

    #[staticmethod]
    fn z(qubit: usize) -> Self {
        Self {
            inner: Observable::z(qubit),
        }
    }

    #[staticmethod]
    fn zz(qubit1: usize, qubit2: usize) -> Self {
        Self {
            inner: Observable::zz(qubit1, qubit2),
        }
    }

    fn expectation_value(&self, result: &PyMeasurementResult) -> f64 {
        // Convert PyMeasurementResult to CircuitResult
        let circuit_result = quantrs2_device::CircuitResult {
            counts: result.counts.clone(),
            shots: result.shots,
            metadata: HashMap::new(),
        };

        self.inner.expectation_value(&circuit_result)
    }

    #[getter]
    fn pauli_string(&self) -> Vec<(usize, String)> {
        self.inner.pauli_string.clone()
    }

    #[getter]
    fn coefficient(&self) -> f64 {
        self.inner.coefficient
    }

    fn __repr__(&self) -> String {
        let pauli_str: Vec<String> = self
            .inner
            .pauli_string
            .iter()
            .map(|(q, p)| format!("{p}_{q}"))
            .collect();
        format!(
            "Observable({} * {})",
            self.inner.coefficient,
            pauli_str.join(" ")
        )
    }
}

/// Zero-Noise Extrapolation executor
#[pyclass(name = "ZeroNoiseExtrapolation")]
pub struct PyZeroNoiseExtrapolation {
    config: PyZNEConfig,
}

#[pymethods]
impl PyZeroNoiseExtrapolation {
    #[allow(clippy::missing_const_for_fn)]
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyZNEConfig>) -> Self {
        Self {
            config: config.unwrap_or_else(|| PyZNEConfig {
                inner: ZNEConfig::default(),
            }),
        }
    }

    /// Apply circuit folding for noise scaling
    ///
    /// Circuit folding amplifies noise by inserting G G† pairs after each gate G.
    /// For a scale factor λ = 2k + 1, each gate G becomes G (G† G)^k.
    ///
    /// Args:
    ///     circuit: The circuit to fold
    ///     scale_factor: The noise amplification factor (must be >= 1.0 and odd integer)
    ///
    /// Returns:
    ///     A new circuit with folded gates
    #[allow(clippy::unused_self)]
    fn fold_circuit(&self, circuit: &PyCircuit, scale_factor: f64) -> PyResult<PyCircuit> {
        fold_circuit_global(circuit, scale_factor)
    }

    /// Perform ZNE given measurement results at different scale factors
    #[allow(clippy::needless_pass_by_value)]
    fn extrapolate(&self, py: Python, data: Vec<(f64, f64)>) -> PyResult<Py<PyZNEResult>> {
        let scale_factors: Vec<f64> = data.iter().map(|(s, _)| *s).collect();
        let values: Vec<f64> = data.iter().map(|(_, v)| *v).collect();

        let result = ExtrapolationFitter::fit_and_extrapolate(
            &scale_factors,
            &values,
            self.config.inner.extrapolation_method,
        )
        .map_err(|e| PyValueError::new_err(format!("Extrapolation failed: {e:?}")))?;

        // Add bootstrap error estimate if requested
        let mut final_result = result;
        if let Some(n_samples) = self.config.inner.bootstrap_samples {
            if let Ok(error) = ExtrapolationFitter::bootstrap_estimate(
                &scale_factors,
                &values,
                self.config.inner.extrapolation_method,
                n_samples,
            ) {
                final_result.error_estimate = Some(error);
            }
        }

        Py::new(
            py,
            PyZNEResult {
                inner: final_result,
            },
        )
    }

    /// Convenience method to run ZNE on an observable
    #[pyo3(signature = (observable, measurements))]
    #[allow(clippy::needless_pass_by_value)]
    fn mitigate_observable(
        &self,
        py: Python,
        observable: &PyObservable,
        measurements: Vec<(f64, PyRef<PyMeasurementResult>)>,
    ) -> PyResult<Py<PyZNEResult>> {
        // Calculate expectation values for each scale factor
        let data: Vec<(f64, f64)> = measurements
            .iter()
            .map(|(scale, result)| (*scale, observable.expectation_value(result)))
            .collect();

        self.extrapolate(py, data)
    }
}

/// Helper function to perform global circuit folding
///
/// For each gate G in the circuit, applies G (G† G)^k where k = (scale_factor - 1) / 2.
fn fold_circuit_global(circuit: &PyCircuit, scale_factor: f64) -> PyResult<PyCircuit> {
    if scale_factor < 1.0 {
        return Err(PyValueError::new_err("Scale factor must be >= 1.0"));
    }

    // Check that scale_factor is close to an odd integer
    let rounded = scale_factor.round();
    if (scale_factor - rounded).abs() > 1e-6 {
        return Err(PyValueError::new_err(
            "Scale factor must be an integer for global folding",
        ));
    }

    let scale_int = rounded as usize;
    if scale_int % 2 == 0 {
        return Err(PyValueError::new_err(
            "Scale factor must be an odd integer (1, 3, 5, ...) for global folding",
        ));
    }

    // Number of fold repetitions: (λ - 1) / 2
    let num_folds = (scale_int - 1) / 2;

    // Create a new circuit with the same number of qubits
    let mut folded = PyCircuit::new(circuit.n_qubits)?;

    // Get the operations from the original circuit
    let ops = circuit.get_operations();

    // For each gate, apply G (G† G)^k
    for &op in ops {
        // Apply the original gate
        folded.apply_op(op)?;

        // Apply (G† G) pairs
        for _ in 0..num_folds {
            folded.apply_op(op.inverse())?;
            folded.apply_op(op)?;
        }
    }

    Ok(folded)
}

/// Helper function to perform local circuit folding with gate weights
///
/// Applies folding selectively based on gate weights.
/// Gates with higher weights are folded more.
fn fold_circuit_local(
    circuit: &PyCircuit,
    scale_factor: f64,
    gate_weights: Option<&[f64]>,
) -> PyResult<PyCircuit> {
    if scale_factor < 1.0 {
        return Err(PyValueError::new_err("Scale factor must be >= 1.0"));
    }

    let ops = circuit.get_operations();
    let num_gates = ops.len();

    // Default weights: all equal
    let default_weights: Vec<f64> = vec![1.0; num_gates];
    let weights = gate_weights.unwrap_or(&default_weights);

    if weights.len() != num_gates {
        return Err(PyValueError::new_err(format!(
            "Number of weights ({}) must match number of gates ({})",
            weights.len(),
            num_gates
        )));
    }

    // Calculate total weight
    let total_weight: f64 = weights.iter().sum();
    if total_weight < 1e-10 {
        return Err(PyValueError::new_err("Total weight must be positive"));
    }

    // Normalize weights
    let normalized_weights: Vec<f64> = weights.iter().map(|w| w / total_weight).collect();

    // Calculate number of folds for each gate to achieve target scale factor
    // Total scale = 1 + 2 * Σ(w_i * k_i) where k_i is the number of folds for gate i
    // We want: scale_factor = 1 + 2 * Σ(w_i * k_i)
    // So: Σ(w_i * k_i) = (scale_factor - 1) / 2
    let target_extra = (scale_factor - 1.0) / 2.0;

    // Create a new circuit with the same number of qubits
    let mut folded = PyCircuit::new(circuit.n_qubits)?;

    // Apply gates with local folding
    for (i, &op) in ops.iter().enumerate() {
        // Apply the original gate
        folded.apply_op(op)?;

        // Calculate number of folds for this gate
        // Proportional to weight * target_extra
        let fold_amount = (normalized_weights[i] * target_extra * (num_gates as f64)).round();
        let num_folds = fold_amount.max(0.0) as usize;

        // Apply (G† G) pairs
        for _ in 0..num_folds {
            folded.apply_op(op.inverse())?;
            folded.apply_op(op)?;
        }
    }

    Ok(folded)
}

/// Circuit folding utilities
#[pyclass(name = "CircuitFolding")]
pub struct PyCircuitFolding;

#[pymethods]
impl PyCircuitFolding {
    #[new]
    const fn new() -> Self {
        Self
    }

    /// Apply global circuit folding
    ///
    /// Each gate G in the circuit becomes G (G† G)^k where k = (scale_factor - 1) / 2.
    ///
    /// Args:
    ///     circuit: The circuit to fold
    ///     scale_factor: The noise amplification factor (must be an odd integer >= 1)
    ///
    /// Returns:
    ///     A new circuit with folded gates
    #[staticmethod]
    fn fold_global(circuit: &PyCircuit, scale_factor: f64) -> PyResult<PyCircuit> {
        fold_circuit_global(circuit, scale_factor)
    }

    /// Apply local circuit folding with optional gate weights
    ///
    /// Folds gates selectively based on their weights. Gates with higher weights
    /// receive more folding, allowing for targeted noise amplification.
    ///
    /// Args:
    ///     circuit: The circuit to fold
    ///     scale_factor: The target noise amplification factor
    ///     gate_weights: Optional weights for each gate (default: uniform weights)
    ///
    /// Returns:
    ///     A new circuit with selectively folded gates
    #[staticmethod]
    #[pyo3(signature = (circuit, scale_factor, gate_weights=None))]
    fn fold_local(
        circuit: &PyCircuit,
        scale_factor: f64,
        gate_weights: Option<Vec<f64>>,
    ) -> PyResult<PyCircuit> {
        fold_circuit_local(circuit, scale_factor, gate_weights.as_deref())
    }
}

/// Extrapolation fitting utilities
#[pyclass(name = "ExtrapolationFitting")]
pub struct PyExtrapolationFitting;

#[pymethods]
impl PyExtrapolationFitting {
    #[allow(clippy::missing_const_for_fn)]
    #[new]
    fn new() -> Self {
        Self
    }

    #[allow(clippy::needless_pass_by_value)]
    #[staticmethod]
    fn fit_linear(
        py: Python,
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyZNEResult>> {
        let x_vec = x.as_slice()?;
        let y_vec = y.as_slice()?;

        let result =
            ExtrapolationFitter::fit_and_extrapolate(x_vec, y_vec, ExtrapolationMethod::Linear)
                .map_err(|e| PyValueError::new_err(format!("Linear fit failed: {e:?}")))?;

        Py::new(py, PyZNEResult { inner: result })
    }

    #[allow(clippy::needless_pass_by_value)]
    #[staticmethod]
    fn fit_polynomial(
        py: Python,
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
        order: usize,
    ) -> PyResult<Py<PyZNEResult>> {
        let x_vec = x.as_slice()?;
        let y_vec = y.as_slice()?;

        let result = ExtrapolationFitter::fit_and_extrapolate(
            x_vec,
            y_vec,
            ExtrapolationMethod::Polynomial(order),
        )
        .map_err(|e| PyValueError::new_err(format!("Polynomial fit failed: {e:?}")))?;

        Py::new(py, PyZNEResult { inner: result })
    }

    #[allow(clippy::needless_pass_by_value)]
    #[staticmethod]
    fn fit_exponential(
        py: Python,
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyZNEResult>> {
        let x_vec = x.as_slice()?;
        let y_vec = y.as_slice()?;

        let result = ExtrapolationFitter::fit_and_extrapolate(
            x_vec,
            y_vec,
            ExtrapolationMethod::Exponential,
        )
        .map_err(|e| PyValueError::new_err(format!("Exponential fit failed: {e:?}")))?;

        Py::new(py, PyZNEResult { inner: result })
    }

    #[allow(clippy::needless_pass_by_value)]
    #[staticmethod]
    fn fit_richardson(
        py: Python,
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyZNEResult>> {
        let x_vec = x.as_slice()?;
        let y_vec = y.as_slice()?;

        let result =
            ExtrapolationFitter::fit_and_extrapolate(x_vec, y_vec, ExtrapolationMethod::Richardson)
                .map_err(|e| {
                    PyValueError::new_err(format!("Richardson extrapolation failed: {e:?}"))
                })?;

        Py::new(py, PyZNEResult { inner: result })
    }

    #[allow(clippy::needless_pass_by_value)]
    #[staticmethod]
    fn fit_adaptive(
        py: Python,
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyZNEResult>> {
        let x_vec = x.as_slice()?;
        let y_vec = y.as_slice()?;

        let result =
            ExtrapolationFitter::fit_and_extrapolate(x_vec, y_vec, ExtrapolationMethod::Adaptive)
                .map_err(|e| PyValueError::new_err(format!("Adaptive fit failed: {e:?}")))?;

        Py::new(py, PyZNEResult { inner: result })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers shared by PEC / VD / SV implementations
// ─────────────────────────────────────────────────────────────────────────────

/// Reconstruct a `PyCircuit` that is an exact copy of `src` by replaying its
/// stored operation list.  This is necessary because `PyCircuit` is a
/// `#[pyclass]` and therefore does not implement `Clone`.
fn rebuild_circuit(src: &PyCircuit) -> PyResult<PyCircuit> {
    let mut dst = PyCircuit::new(src.n_qubits)?;
    for &op in src.get_operations() {
        dst.apply_op(op)?;
    }
    Ok(dst)
}

/// Count how many times a given Pauli gate type appears in the circuit
/// operations list.  `gate_name` is one of `"X"`, `"Y"`, `"Z"`.
fn count_pauli_gates(circuit: &PyCircuit, gate_name: &str) -> usize {
    circuit
        .get_operations()
        .iter()
        .filter(|op| {
            matches!(
                (gate_name, op),
                ("X", CircuitOp::PauliX(_))
                    | ("Y", CircuitOp::PauliY(_))
                    | ("Z", CircuitOp::PauliZ(_))
            )
        })
        .count()
}

/// Check whether all rotation angles in the circuit are multiples of π
/// (sufficient condition for time-reversal / Clifford structure).
fn all_rotations_are_clifford(circuit: &PyCircuit) -> bool {
    use std::f64::consts::PI;
    circuit.get_operations().iter().all(|op| {
        let check_angle = |theta: f64| -> bool {
            let frac = (theta / PI).fract().abs();
            !(1e-9..=1.0 - 1e-9).contains(&frac)
        };
        match *op {
            CircuitOp::Rx(_, theta)
            | CircuitOp::Ry(_, theta)
            | CircuitOp::Rz(_, theta)
            | CircuitOp::P(_, theta)
            | CircuitOp::CRX(_, _, theta)
            | CircuitOp::CRY(_, _, theta)
            | CircuitOp::CRZ(_, _, theta)
            | CircuitOp::RXX(_, _, theta)
            | CircuitOp::RYY(_, _, theta)
            | CircuitOp::RZZ(_, _, theta)
            | CircuitOp::RZX(_, _, theta) => check_angle(theta),
            CircuitOp::U(_, theta, phi, lambda) => {
                check_angle(theta) && check_angle(phi) && check_angle(lambda)
            }
            _ => true, // non-rotation gate — does not affect Clifford status
        }
    })
}

/// Shift all qubit indices in `op` by `offset`, producing a new `CircuitOp`.
///
/// Returns `PyValueError` if any resulting index overflows `u32`.
fn shift_op_qubits(op: CircuitOp, offset: usize) -> PyResult<CircuitOp> {
    use quantrs2_core::qubit::QubitId;

    let shift = |q: QubitId| -> PyResult<QubitId> {
        let new_idx = q.0 as usize + offset;
        let id = u32::try_from(new_idx).map_err(|_| {
            PyValueError::new_err(format!(
                "Qubit index {new_idx} exceeds u32 range during shift"
            ))
        })?;
        Ok(QubitId::new(id))
    };

    let shifted = match op {
        CircuitOp::Hadamard(q) => CircuitOp::Hadamard(shift(q)?),
        CircuitOp::PauliX(q) => CircuitOp::PauliX(shift(q)?),
        CircuitOp::PauliY(q) => CircuitOp::PauliY(shift(q)?),
        CircuitOp::PauliZ(q) => CircuitOp::PauliZ(shift(q)?),
        CircuitOp::S(q) => CircuitOp::S(shift(q)?),
        CircuitOp::SDagger(q) => CircuitOp::SDagger(shift(q)?),
        CircuitOp::T(q) => CircuitOp::T(shift(q)?),
        CircuitOp::TDagger(q) => CircuitOp::TDagger(shift(q)?),
        CircuitOp::SX(q) => CircuitOp::SX(shift(q)?),
        CircuitOp::SXDagger(q) => CircuitOp::SXDagger(shift(q)?),
        CircuitOp::Id(q) => CircuitOp::Id(shift(q)?),
        CircuitOp::Rx(q, t) => CircuitOp::Rx(shift(q)?, t),
        CircuitOp::Ry(q, t) => CircuitOp::Ry(shift(q)?, t),
        CircuitOp::Rz(q, t) => CircuitOp::Rz(shift(q)?, t),
        CircuitOp::P(q, t) => CircuitOp::P(shift(q)?, t),
        CircuitOp::U(q, a, b, c) => CircuitOp::U(shift(q)?, a, b, c),
        CircuitOp::Cnot(c, t) => CircuitOp::Cnot(shift(c)?, shift(t)?),
        CircuitOp::Swap(a, b) => CircuitOp::Swap(shift(a)?, shift(b)?),
        CircuitOp::CY(c, t) => CircuitOp::CY(shift(c)?, shift(t)?),
        CircuitOp::CZ(c, t) => CircuitOp::CZ(shift(c)?, shift(t)?),
        CircuitOp::CH(c, t) => CircuitOp::CH(shift(c)?, shift(t)?),
        CircuitOp::CS(c, t) => CircuitOp::CS(shift(c)?, shift(t)?),
        CircuitOp::ISwap(a, b) => CircuitOp::ISwap(shift(a)?, shift(b)?),
        CircuitOp::ECR(c, t) => CircuitOp::ECR(shift(c)?, shift(t)?),
        CircuitOp::DCX(a, b) => CircuitOp::DCX(shift(a)?, shift(b)?),
        CircuitOp::CRX(c, t, h) => CircuitOp::CRX(shift(c)?, shift(t)?, h),
        CircuitOp::CRY(c, t, h) => CircuitOp::CRY(shift(c)?, shift(t)?, h),
        CircuitOp::CRZ(c, t, h) => CircuitOp::CRZ(shift(c)?, shift(t)?, h),
        CircuitOp::RXX(a, b, h) => CircuitOp::RXX(shift(a)?, shift(b)?, h),
        CircuitOp::RYY(a, b, h) => CircuitOp::RYY(shift(a)?, shift(b)?, h),
        CircuitOp::RZZ(a, b, h) => CircuitOp::RZZ(shift(a)?, shift(b)?, h),
        CircuitOp::RZX(c, t, h) => CircuitOp::RZX(shift(c)?, shift(t)?, h),
        CircuitOp::Toffoli(c1, c2, t) => CircuitOp::Toffoli(shift(c1)?, shift(c2)?, shift(t)?),
        CircuitOp::Fredkin(c, t1, t2) => CircuitOp::Fredkin(shift(c)?, shift(t1)?, shift(t2)?),
    };
    Ok(shifted)
}

// ─────────────────────────────────────────────────────────────────────────────
// Probabilistic Error Cancellation
// ─────────────────────────────────────────────────────────────────────────────

/// Probabilistic Error Cancellation (PEC).
///
/// PEC represents the ideal (noise-free) expectation value as a linear
/// combination of noisy expectation values.  For a depolarising channel
/// with single-qubit error rate ε the quasi-probability representation is:
///
///   E^{-1}(ρ) = (1 + 4ε/3) G(ρ) – (ε/3)(X G X + Y G Y + Z G Z)
///
/// The *sampling overhead* is γ = (1 + 2ε/3)^n where n is the number of
/// noisy gates.  This implementation builds one circuit per term of the
/// expansion and attaches the corresponding quasi-probability coefficient.
#[pyclass(name = "ProbabilisticErrorCancellation")]
pub struct PyProbabilisticErrorCancellation;

#[pymethods]
impl PyProbabilisticErrorCancellation {
    #[allow(clippy::missing_const_for_fn)]
    #[new]
    fn new() -> Self {
        Self
    }

    /// Decompose `circuit` into a quasi-probabilistic mixture that cancels
    /// single-qubit depolarising noise of strength `noise_strength`.
    ///
    /// Returns a list of `(coefficient, circuit_variant)` tuples.
    /// The mitigated expectation value is obtained by taking the weighted
    /// average of expectation values measured from each variant.
    ///
    /// Sampling overhead: γ = (1 + 2·noise_strength/3)^n_qubits.
    #[staticmethod]
    #[pyo3(signature = (circuit, noise_strength = 0.01))]
    fn quasi_probability_decomposition(
        circuit: &PyCircuit,
        noise_strength: f64,
    ) -> PyResult<Vec<(f64, PyCircuit)>> {
        if !(0.0..=1.0).contains(&noise_strength) {
            return Err(PyValueError::new_err("noise_strength must be in [0, 1]"));
        }

        // For ε = 0 there is nothing to cancel; return the original circuit.
        if noise_strength < 1e-15 {
            let original = rebuild_circuit(circuit)?;
            return Ok(vec![(1.0, original)]);
        }

        let n = circuit.n_qubits;

        // Positive quasi-probability term coefficient.
        let positive_coeff = 1.0 + 4.0 * noise_strength / 3.0;
        // Negative quasi-probability term coefficient (one per Pauli, per qubit).
        let negative_coeff = -noise_strength / 3.0;

        let mut decomposition: Vec<(f64, PyCircuit)> = Vec::with_capacity(1 + 3 * n);

        // Term 0: the original circuit with boosted coefficient
        let original = rebuild_circuit(circuit)?;
        decomposition.push((positive_coeff, original));

        // Terms 1..3n: error-inverted variants
        // For each qubit, insert X, Y and Z error operators at the end of
        // the circuit (circuit-level quasi-probability approximation).
        for qubit in 0..n {
            let qid = quantrs2_core::qubit::QubitId::new(qubit as u32);

            // X-error variant
            let mut x_circuit = rebuild_circuit(circuit)?;
            x_circuit.apply_op(CircuitOp::PauliX(qid))?;
            decomposition.push((negative_coeff, x_circuit));

            // Y-error variant
            let mut y_circuit = rebuild_circuit(circuit)?;
            y_circuit.apply_op(CircuitOp::PauliY(qid))?;
            decomposition.push((negative_coeff, y_circuit));

            // Z-error variant
            let mut z_circuit = rebuild_circuit(circuit)?;
            z_circuit.apply_op(CircuitOp::PauliZ(qid))?;
            decomposition.push((negative_coeff, z_circuit));
        }

        Ok(decomposition)
    }

    /// Compute the one-norm sampling overhead γ = (1 + 2ε/3)^n for `n` qubits
    /// and depolarising rate `noise_strength`.
    ///
    /// This equals the factor by which the number of samples must increase to
    /// achieve the same statistical accuracy as a noiseless device.
    #[staticmethod]
    fn sampling_overhead(n_qubits: usize, noise_strength: f64) -> f64 {
        (1.0 + 2.0 * noise_strength / 3.0).powi(n_qubits as i32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Virtual Distillation
// ─────────────────────────────────────────────────────────────────────────────

/// Virtual Distillation.
///
/// Given M noisy copies of the state ρ the ideal expectation value of an
/// observable O can be estimated as
///
///   <O>_ideal ≈ Tr[O ρ^M] / Tr[ρ^M]
///
/// For M = 2 this is implemented via a SWAP test between two circuit
/// registers: a Hadamard ancilla controls pairwise CSWAP (Fredkin)
/// operations between the two copies, followed by a second Hadamard.
/// The ancilla outcome encodes Tr[ρ²] and the purified observable.
///
/// Qubit layout of the returned circuit:
///   qubits 0 .. n-1   → first copy
///   qubits n .. 2n-1  → second copy
///   qubit  2n         → ancilla
#[pyclass(name = "VirtualDistillation")]
pub struct PyVirtualDistillation;

#[pymethods]
impl PyVirtualDistillation {
    #[allow(clippy::missing_const_for_fn)]
    #[new]
    fn new() -> Self {
        Self
    }

    /// Build a virtual-distillation circuit from `circuits`.
    ///
    /// * length 1 → passthrough (returns a copy of the single circuit).
    /// * length 2 → M = 2 SWAP-test circuit with `2n + 1` qubits.
    /// * length > 2 → returns `PyValueError` (not yet supported).
    #[staticmethod]
    fn distill(circuits: Vec<PyRef<PyCircuit>>) -> PyResult<PyCircuit> {
        match circuits.len() {
            0 => Err(PyValueError::new_err("circuits list must not be empty")),
            1 => rebuild_circuit(&circuits[0]),
            2 => {
                let n = circuits[0].n_qubits;
                if circuits[1].n_qubits != n {
                    return Err(PyValueError::new_err(
                        "Both circuits must have the same number of qubits",
                    ));
                }

                // Total qubits: two n-qubit registers + 1 ancilla.
                // PyCircuit::new requires >= 2; 2*1+1 = 3 satisfies this.
                let total = 2 * n + 1;
                let ancilla = (total - 1) as u32; // index 2n

                let mut combined = PyCircuit::new(total)?;

                // Apply first-copy gates on qubits 0..n
                for &op in circuits[0].get_operations() {
                    let shifted = shift_op_qubits(op, 0)?;
                    combined.apply_op(shifted)?;
                }

                // Apply second-copy gates on qubits n..2n
                for &op in circuits[1].get_operations() {
                    let shifted = shift_op_qubits(op, n)?;
                    combined.apply_op(shifted)?;
                }

                // SWAP test: H_ancilla then CSWAP_i(ancilla, i, n+i) then H_ancilla
                let ancilla_id = quantrs2_core::qubit::QubitId::new(ancilla);
                combined.apply_op(CircuitOp::Hadamard(ancilla_id))?;

                for i in 0..n {
                    combined.apply_op(CircuitOp::Fredkin(
                        ancilla_id,
                        quantrs2_core::qubit::QubitId::new(i as u32),
                        quantrs2_core::qubit::QubitId::new((n + i) as u32),
                    ))?;
                }

                combined.apply_op(CircuitOp::Hadamard(ancilla_id))?;

                Ok(combined)
            }
            m => Err(PyValueError::new_err(format!(
                "Virtual distillation for M={m} copies is not yet supported (only M in {{1,2}})"
            ))),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Symmetry Verification
// ─────────────────────────────────────────────────────────────────────────────

/// Symmetry Verification.
///
/// Checks whether a circuit preserves a stated physical symmetry by
/// analysing the gate sequence.  The analysis is purely syntactic
/// (gate counting / angle inspection) and therefore conservative.
///
/// Supported symmetry labels:
///
/// | Label | Meaning |
/// |-------|---------|
/// | `"parity"` / `"z2"` / `"Z2"` | Total X-gate count is even |
/// | `"particle_number"` / `"U1"` / `"u1"` | No bare X gates |
/// | `"time_reversal"` | All rotation angles are multiples of π |
#[pyclass(name = "SymmetryVerification")]
pub struct PySymmetryVerification;

#[pymethods]
impl PySymmetryVerification {
    #[allow(clippy::missing_const_for_fn)]
    #[new]
    fn new() -> Self {
        Self
    }

    /// Verify that `circuit` preserves the symmetry named by `symmetry`.
    ///
    /// Returns `true` if the symmetry check passes, `false` if it is
    /// violated, or a `ValueError` if the symmetry label is unknown.
    #[staticmethod]
    fn verify_symmetry(circuit: &PyCircuit, symmetry: &str) -> PyResult<bool> {
        match symmetry {
            // Z2 parity: total single-qubit X count must be even.
            "parity" | "z2" | "Z2" => {
                let x_count = count_pauli_gates(circuit, "X");
                Ok(x_count % 2 == 0)
            }

            // U(1) particle-number symmetry: no bare X gates allowed.
            // Two-qubit gates (CNOT, SWAP, etc.) conserve number and are fine.
            "particle_number" | "U1" | "u1" => {
                let x_count = count_pauli_gates(circuit, "X");
                Ok(x_count == 0)
            }

            // Time-reversal symmetry: circuit equals its own complex conjugate.
            // Sufficient syntactic condition: every rotation angle is a
            // multiple of π (only real-matrix Clifford gates present).
            "time_reversal" => Ok(all_rotations_are_clifford(circuit)),

            unknown => Err(PyValueError::new_err(format!(
                "Unknown symmetry type: '{unknown}'. \
                 Supported: 'parity'/'z2'/'Z2', \
                 'particle_number'/'U1'/'u1', \
                 'time_reversal'"
            ))),
        }
    }

    /// List all supported symmetry labels.
    #[staticmethod]
    fn supported_symmetries() -> Vec<&'static str> {
        vec![
            "parity",
            "z2",
            "Z2",
            "particle_number",
            "U1",
            "u1",
            "time_reversal",
        ]
    }
}

/// Register the mitigation module
pub fn register_mitigation_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(m.py(), "mitigation")?;

    submodule.add_class::<PyZNEConfig>()?;
    submodule.add_class::<PyZNEResult>()?;
    submodule.add_class::<PyObservable>()?;
    submodule.add_class::<PyZeroNoiseExtrapolation>()?;
    submodule.add_class::<PyCircuitFolding>()?;
    submodule.add_class::<PyExtrapolationFitting>()?;
    submodule.add_class::<PyProbabilisticErrorCancellation>()?;
    submodule.add_class::<PyVirtualDistillation>()?;
    submodule.add_class::<PySymmetryVerification>()?;

    m.add_submodule(&submodule)?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
//
// These tests exercise only pure-Rust logic and do not require a Python
// interpreter.  They can be run with:
//   cargo test -p quantrs2-py --features device -- mitigation
//
// Note: tests that call `PyCircuit::new` / `apply_op` still compile because
// those methods are pub(crate) Rust functions that do not touch PyO3 machinery.

#[cfg(test)]
mod tests {
    use super::*;

    // ── helper ────────────────────────────────────────────────────────────────

    /// 2-qubit Bell-state preparation: H(0) CNOT(0,1).
    fn bell_circuit() -> PyCircuit {
        let mut c = PyCircuit::new(2).expect("create circuit");
        c.apply_op(CircuitOp::Hadamard(quantrs2_core::qubit::QubitId::new(0)))
            .expect("H");
        c.apply_op(CircuitOp::Cnot(
            quantrs2_core::qubit::QubitId::new(0),
            quantrs2_core::qubit::QubitId::new(1),
        ))
        .expect("CNOT");
        c
    }

    // ── rebuild_circuit ───────────────────────────────────────────────────────

    #[test]
    fn rebuild_preserves_n_qubits_and_gate_count() {
        let c = bell_circuit();
        let r = rebuild_circuit(&c).expect("rebuild");
        assert_eq!(r.n_qubits, c.n_qubits);
        assert_eq!(r.get_operations().len(), c.get_operations().len());
    }

    #[test]
    fn rebuild_three_qubit_circuit() {
        let mut c = PyCircuit::new(3).expect("create");
        c.apply_op(CircuitOp::Hadamard(quantrs2_core::qubit::QubitId::new(0)))
            .expect("H");
        c.apply_op(CircuitOp::PauliX(quantrs2_core::qubit::QubitId::new(1)))
            .expect("X");
        c.apply_op(CircuitOp::Cnot(
            quantrs2_core::qubit::QubitId::new(0),
            quantrs2_core::qubit::QubitId::new(2),
        ))
        .expect("CNOT");
        let r = rebuild_circuit(&c).expect("rebuild");
        assert_eq!(r.get_operations().len(), 3);
    }

    // ── shift_op_qubits ───────────────────────────────────────────────────────

    #[test]
    fn shift_identity_offset_zero() {
        let op = CircuitOp::PauliX(quantrs2_core::qubit::QubitId::new(1));
        let shifted = shift_op_qubits(op, 0).expect("shift");
        match shifted {
            CircuitOp::PauliX(q) => assert_eq!(q.0, 1),
            _ => panic!("unexpected gate type"),
        }
    }

    #[test]
    fn shift_single_qubit_increases_index() {
        let op = CircuitOp::Hadamard(quantrs2_core::qubit::QubitId::new(0));
        let shifted = shift_op_qubits(op, 3).expect("shift");
        match shifted {
            CircuitOp::Hadamard(q) => assert_eq!(q.0, 3),
            _ => panic!("unexpected gate type"),
        }
    }

    #[test]
    fn shift_two_qubit_both_indices_moved() {
        let op = CircuitOp::Cnot(
            quantrs2_core::qubit::QubitId::new(0),
            quantrs2_core::qubit::QubitId::new(1),
        );
        let shifted = shift_op_qubits(op, 2).expect("shift");
        match shifted {
            CircuitOp::Cnot(c, t) => {
                assert_eq!(c.0, 2);
                assert_eq!(t.0, 3);
            }
            _ => panic!("unexpected gate type"),
        }
    }

    #[test]
    fn shift_three_qubit_fredkin() {
        let op = CircuitOp::Fredkin(
            quantrs2_core::qubit::QubitId::new(0),
            quantrs2_core::qubit::QubitId::new(1),
            quantrs2_core::qubit::QubitId::new(2),
        );
        let shifted = shift_op_qubits(op, 5).expect("shift");
        match shifted {
            CircuitOp::Fredkin(c, t1, t2) => {
                assert_eq!(c.0, 5);
                assert_eq!(t1.0, 6);
                assert_eq!(t2.0, 7);
            }
            _ => panic!("unexpected gate type"),
        }
    }

    // ── PEC ───────────────────────────────────────────────────────────────────

    #[test]
    fn pec_zero_noise_returns_single_term_with_unit_coefficient() {
        let circuit = bell_circuit();
        let decomp =
            PyProbabilisticErrorCancellation::quasi_probability_decomposition(&circuit, 0.0)
                .expect("decompose");
        assert_eq!(decomp.len(), 1);
        assert!((decomp[0].0 - 1.0).abs() < 1e-12, "coefficient should be 1");
    }

    #[test]
    fn pec_nonzero_noise_term_count_is_one_plus_three_n() {
        let circuit = bell_circuit();
        let decomp =
            PyProbabilisticErrorCancellation::quasi_probability_decomposition(&circuit, 0.01)
                .expect("decompose");
        // 2 qubits: 1 original + 3 * 2 = 7 terms
        assert_eq!(decomp.len(), 1 + 3 * circuit.n_qubits);
    }

    #[test]
    fn pec_first_coefficient_is_positive_rest_are_negative() {
        let circuit = bell_circuit();
        let decomp =
            PyProbabilisticErrorCancellation::quasi_probability_decomposition(&circuit, 0.05)
                .expect("decompose");
        assert!(decomp[0].0 > 0.0, "first coeff should be positive");
        for (coeff, _) in &decomp[1..] {
            assert!(*coeff < 0.0, "expected negative coeff, got {coeff}");
        }
    }

    #[test]
    fn pec_positive_coefficient_is_one_plus_four_thirds_epsilon() {
        let eps = 0.03;
        let circuit = bell_circuit();
        let decomp =
            PyProbabilisticErrorCancellation::quasi_probability_decomposition(&circuit, eps)
                .expect("decompose");
        let expected = 1.0 + 4.0 * eps / 3.0;
        assert!(
            (decomp[0].0 - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            decomp[0].0
        );
    }

    #[test]
    fn pec_negative_coefficient_magnitude_is_epsilon_over_three() {
        let eps = 0.06;
        let circuit = bell_circuit();
        let decomp =
            PyProbabilisticErrorCancellation::quasi_probability_decomposition(&circuit, eps)
                .expect("decompose");
        let expected_neg = -eps / 3.0;
        for (coeff, _) in &decomp[1..] {
            assert!(
                (coeff - expected_neg).abs() < 1e-12,
                "expected {expected_neg}, got {coeff}"
            );
        }
    }

    #[test]
    fn pec_rejects_negative_noise_strength() {
        let circuit = bell_circuit();
        assert!(
            PyProbabilisticErrorCancellation::quasi_probability_decomposition(&circuit, -0.1)
                .is_err()
        );
    }

    #[test]
    fn pec_rejects_noise_strength_above_one() {
        let circuit = bell_circuit();
        assert!(
            PyProbabilisticErrorCancellation::quasi_probability_decomposition(&circuit, 1.5)
                .is_err()
        );
    }

    #[test]
    fn pec_sampling_overhead_unity_at_zero_noise() {
        let overhead = PyProbabilisticErrorCancellation::sampling_overhead(4, 0.0);
        assert!((overhead - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pec_sampling_overhead_grows_with_noise_and_qubit_count() {
        let lo = PyProbabilisticErrorCancellation::sampling_overhead(3, 0.01);
        let hi = PyProbabilisticErrorCancellation::sampling_overhead(3, 0.1);
        assert!(hi > lo);

        let few = PyProbabilisticErrorCancellation::sampling_overhead(2, 0.05);
        let many = PyProbabilisticErrorCancellation::sampling_overhead(10, 0.05);
        assert!(many > few);
    }

    // ── Virtual Distillation (non-PyO3 helpers) ───────────────────────────────

    #[test]
    fn vd_passthrough_for_single_circuit_via_rebuild() {
        let circuit = bell_circuit();
        let rebuilt = rebuild_circuit(&circuit).expect("rebuild");
        assert_eq!(rebuilt.n_qubits, circuit.n_qubits);
        assert_eq!(
            rebuilt.get_operations().len(),
            circuit.get_operations().len()
        );
    }

    // ── Symmetry Verification ─────────────────────────────────────────────────

    #[test]
    fn sv_parity_even_x_count_preserved() {
        let mut c = PyCircuit::new(2).expect("create");
        c.apply_op(CircuitOp::PauliX(quantrs2_core::qubit::QubitId::new(0)))
            .expect("X0");
        c.apply_op(CircuitOp::PauliX(quantrs2_core::qubit::QubitId::new(1)))
            .expect("X1");
        assert!(PySymmetryVerification::verify_symmetry(&c, "parity").expect("verify"));
    }

    #[test]
    fn sv_parity_odd_x_count_violated() {
        let mut c = PyCircuit::new(2).expect("create");
        c.apply_op(CircuitOp::PauliX(quantrs2_core::qubit::QubitId::new(0)))
            .expect("X");
        assert!(!PySymmetryVerification::verify_symmetry(&c, "parity").expect("verify"));
    }

    #[test]
    fn sv_particle_number_no_x_gates_preserved() {
        let circuit = bell_circuit(); // H + CNOT: no X
        assert!(
            PySymmetryVerification::verify_symmetry(&circuit, "particle_number").expect("verify")
        );
    }

    #[test]
    fn sv_particle_number_with_x_gate_violated() {
        let mut c = PyCircuit::new(2).expect("create");
        c.apply_op(CircuitOp::PauliX(quantrs2_core::qubit::QubitId::new(0)))
            .expect("X");
        assert!(!PySymmetryVerification::verify_symmetry(&c, "particle_number").expect("verify"));
    }

    #[test]
    fn sv_time_reversal_clifford_circuit_preserved() {
        let circuit = bell_circuit(); // H and CNOT have no continuous rotation
        assert!(
            PySymmetryVerification::verify_symmetry(&circuit, "time_reversal").expect("verify")
        );
    }

    #[test]
    fn sv_time_reversal_non_clifford_angle_violated() {
        let mut c = PyCircuit::new(2).expect("create");
        // Rx(π/4) is not a multiple of π
        c.apply_op(CircuitOp::Rx(
            quantrs2_core::qubit::QubitId::new(0),
            std::f64::consts::FRAC_PI_4,
        ))
        .expect("Rx");
        assert!(!PySymmetryVerification::verify_symmetry(&c, "time_reversal").expect("verify"));
    }

    #[test]
    fn sv_time_reversal_pi_rotation_preserved() {
        let mut c = PyCircuit::new(2).expect("create");
        // Rx(π) is a multiple of π — allowed
        c.apply_op(CircuitOp::Rx(
            quantrs2_core::qubit::QubitId::new(0),
            std::f64::consts::PI,
        ))
        .expect("Rx(pi)");
        assert!(PySymmetryVerification::verify_symmetry(&c, "time_reversal").expect("verify"));
    }

    #[test]
    fn sv_unknown_symmetry_returns_error() {
        let circuit = bell_circuit();
        assert!(PySymmetryVerification::verify_symmetry(&circuit, "su2").is_err());
    }

    #[test]
    fn sv_z2_aliases_agree_with_parity() {
        let circuit = bell_circuit();
        let via_parity =
            PySymmetryVerification::verify_symmetry(&circuit, "parity").expect("parity");
        assert_eq!(
            via_parity,
            PySymmetryVerification::verify_symmetry(&circuit, "z2").expect("z2")
        );
        assert_eq!(
            via_parity,
            PySymmetryVerification::verify_symmetry(&circuit, "Z2").expect("Z2")
        );
    }

    #[test]
    fn sv_u1_aliases_agree_with_particle_number() {
        let circuit = bell_circuit();
        let via_pn =
            PySymmetryVerification::verify_symmetry(&circuit, "particle_number").expect("pn");
        assert_eq!(
            via_pn,
            PySymmetryVerification::verify_symmetry(&circuit, "U1").expect("U1")
        );
        assert_eq!(
            via_pn,
            PySymmetryVerification::verify_symmetry(&circuit, "u1").expect("u1")
        );
    }

    #[test]
    fn sv_supported_symmetries_lists_all_seven_labels() {
        let labels = PySymmetryVerification::supported_symmetries();
        assert_eq!(labels.len(), 7);
    }
}
