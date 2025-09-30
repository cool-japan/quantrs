//! Simplified parametric quantum circuits module.
//!
//! This module provides basic parametric gates for variational algorithms.

use scirs2_core::Complex64;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Python wrapper for parametric quantum circuits
#[pyclass(name = "ParametricCircuit")]
pub struct PyParametricCircuit {
    pub n_qubits: usize,
    pub parameters: HashMap<String, f64>,
}

#[pymethods]
impl PyParametricCircuit {
    #[new]
    #[pyo3(signature = (n_qubits, gradient_method="parameter_shift"))]
    pub fn new(n_qubits: usize, gradient_method: &str) -> PyResult<Self> {
        // gradient_method is ignored in this simplified version
        Ok(Self {
            n_qubits,
            parameters: HashMap::new(),
        })
    }

    /// Add a parameter
    pub fn add_parameter(&mut self, name: String, value: f64) -> PyResult<()> {
        self.parameters.insert(name, value);
        Ok(())
    }

    /// Get parameter value
    pub fn get_parameter(&self, name: &str) -> PyResult<f64> {
        self.parameters
            .get(name)
            .copied()
            .ok_or_else(|| PyValueError::new_err(format!("Parameter {} not found", name)))
    }

    /// Set parameter value
    pub fn set_parameter(&mut self, name: &str, value: f64) -> PyResult<()> {
        if self.parameters.contains_key(name) {
            self.parameters.insert(name.to_string(), value);
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "Parameter {} not found",
                name
            )))
        }
    }

    /// Get all parameters
    pub fn get_parameters(&self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        for (name, value) in &self.parameters {
            dict.set_item(name, value).unwrap();
        }
        dict.into()
    }

    /// Set parameters from dictionary
    pub fn set_parameters(&mut self, values: &Bound<'_, PyDict>) -> PyResult<()> {
        for (key, value) in values.iter() {
            let param_name: String = key.extract()?;
            let param_value: f64 = value.extract()?;
            self.set_parameter(&param_name, param_value)?;
        }
        Ok(())
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }

    /// Get state vector (simplified - returns dummy state)
    pub fn get_statevector(&self, py: Python, _params: Vec<f64>) -> PyResult<PyObject> {
        let n_states = 1 << self.n_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); n_states];
        state[0] = Complex64::new(1.0, 0.0);

        let py_array = PyArray1::from_vec(py, state);
        Ok(py_array.into())
    }

    // Placeholder methods for gates
    pub fn rx(
        &mut self,
        _qubit: usize,
        param_name: &str,
        initial_value: Option<f64>,
    ) -> PyResult<()> {
        let value = initial_value.unwrap_or(0.0);
        self.add_parameter(param_name.to_string(), value)
    }

    pub fn ry(
        &mut self,
        _qubit: usize,
        param_name: &str,
        initial_value: Option<f64>,
    ) -> PyResult<()> {
        let value = initial_value.unwrap_or(0.0);
        self.add_parameter(param_name.to_string(), value)
    }

    pub fn rz(
        &mut self,
        _qubit: usize,
        param_name: &str,
        initial_value: Option<f64>,
    ) -> PyResult<()> {
        let value = initial_value.unwrap_or(0.0);
        self.add_parameter(param_name.to_string(), value)
    }

    pub fn cnot(&mut self, _control: usize, _target: usize) -> PyResult<()> {
        Ok(())
    }

    pub fn h(&mut self, _qubit: usize) -> PyResult<()> {
        Ok(())
    }

    pub fn x(&mut self, _qubit: usize) -> PyResult<()> {
        Ok(())
    }

    pub fn rxx(&mut self, _qubit1: usize, _qubit2: usize, param_name: &str) -> PyResult<()> {
        self.add_parameter(param_name.to_string(), 0.0)
    }

    pub fn ryy(&mut self, _qubit1: usize, _qubit2: usize, param_name: &str) -> PyResult<()> {
        self.add_parameter(param_name.to_string(), 0.0)
    }
}

/// Optimizer for parametric circuits
#[pyclass(name = "CircuitOptimizer")]
pub struct PyCircuitOptimizer {
    learning_rate: f64,
}

#[pymethods]
impl PyCircuitOptimizer {
    #[new]
    #[pyo3(signature = (learning_rate=0.01, momentum=0.0))]
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        // momentum is ignored in this simplified version
        Self { learning_rate }
    }
}

/// Register the parametric module
pub fn register_parametric_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(m.py(), "parametric")?;

    submodule.add_class::<PyParametricCircuit>()?;
    submodule.add_class::<PyCircuitOptimizer>()?;

    m.add_submodule(&submodule)?;
    Ok(())
}
