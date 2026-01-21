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
    fn raw_data(&self, py: Python) -> PyResult<PyObject> {
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

/// Probabilistic Error Cancellation (placeholder)
#[pyclass(name = "ProbabilisticErrorCancellation")]
pub struct PyProbabilisticErrorCancellation;

#[pymethods]
impl PyProbabilisticErrorCancellation {
    #[allow(clippy::missing_const_for_fn)]
    #[new]
    fn new() -> Self {
        Self
    }

    #[allow(clippy::unused_self)]
    fn quasi_probability_decomposition(
        &self,
        _circuit: &PyCircuit,
    ) -> PyResult<Vec<(f64, PyCircuit)>> {
        // Placeholder implementation
        Err(PyValueError::new_err("PEC not yet implemented"))
    }
}

/// Virtual Distillation (placeholder)
#[pyclass(name = "VirtualDistillation")]
pub struct PyVirtualDistillation;

#[pymethods]
impl PyVirtualDistillation {
    #[allow(clippy::missing_const_for_fn)]
    #[new]
    fn new() -> Self {
        Self
    }

    #[allow(clippy::unused_self)]
    fn distill(&self, _circuits: Vec<PyRef<PyCircuit>>) -> PyResult<PyCircuit> {
        // Placeholder implementation
        Err(PyValueError::new_err(
            "Virtual distillation not yet implemented",
        ))
    }
}

/// Symmetry Verification (placeholder)
#[pyclass(name = "SymmetryVerification")]
pub struct PySymmetryVerification;

#[pymethods]
impl PySymmetryVerification {
    #[allow(clippy::missing_const_for_fn)]
    #[new]
    fn new() -> Self {
        Self
    }

    #[allow(clippy::unused_self)]
    fn verify_symmetry(&self, _circuit: &PyCircuit, _symmetry: &str) -> PyResult<bool> {
        // Placeholder implementation
        Err(PyValueError::new_err(
            "Symmetry verification not yet implemented",
        ))
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

// Note: Rust unit tests for circuit folding require Python runtime (PyO3).
// See py/tests/test_mitigation.py for comprehensive Python-based tests.
