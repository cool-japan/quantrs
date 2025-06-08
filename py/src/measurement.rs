//! Measurement statistics and quantum state tomography.
//!
//! This module provides tools for:
//! - Measurement outcome statistics
//! - Quantum state tomography
//! - Process tomography
//! - Measurement error mitigation

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use num_complex::Complex64;
use std::collections::HashMap;
use rand::Rng;
use crate::{PyCircuit, PySimulationResult};

/// Measurement outcomes from repeated circuit executions
#[pyclass(name = "MeasurementResult")]
pub struct PyMeasurementResult {
    pub counts: HashMap<String, usize>,
    pub shots: usize,
    pub n_qubits: usize,
}

#[pymethods]
impl PyMeasurementResult {
    /// Get the raw counts dictionary
    fn get_counts(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (bitstring, count) in &self.counts {
            dict.set_item(bitstring, count)?;
        }
        Ok(dict.into())
    }
    
    /// Get the measurement probabilities
    fn get_probabilities(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let total = self.shots as f64;
        for (bitstring, count) in &self.counts {
            let prob = *count as f64 / total;
            dict.set_item(bitstring, prob)?;
        }
        Ok(dict.into())
    }
    
    /// Get the most probable outcome
    fn most_probable(&self) -> PyResult<String> {
        self.counts.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(bitstring, _)| bitstring.clone())
            .ok_or_else(|| PyValueError::new_err("No measurement outcomes"))
    }
    
    /// Get the marginal probability for a specific qubit
    fn marginal_probability(&self, qubit: usize) -> PyResult<f64> {
        if qubit >= self.n_qubits {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range for {} qubits",
                qubit, self.n_qubits
            )));
        }
        
        let mut prob_one = 0.0;
        let total = self.shots as f64;
        
        for (bitstring, count) in &self.counts {
            let chars: Vec<char> = bitstring.chars().collect();
            if chars[qubit] == '1' {
                prob_one += *count as f64 / total;
            }
        }
        
        Ok(prob_one)
    }
    
    /// Get the correlation between two qubits
    fn correlation(&self, qubit1: usize, qubit2: usize) -> PyResult<f64> {
        if qubit1 >= self.n_qubits || qubit2 >= self.n_qubits {
            return Err(PyValueError::new_err("Qubit indices out of range"));
        }
        
        let mut count_00 = 0;
        let mut count_01 = 0;
        let mut count_10 = 0;
        let mut count_11 = 0;
        
        for (bitstring, count) in &self.counts {
            let chars: Vec<char> = bitstring.chars().collect();
            match (chars[qubit1], chars[qubit2]) {
                ('0', '0') => count_00 += count,
                ('0', '1') => count_01 += count,
                ('1', '0') => count_10 += count,
                ('1', '1') => count_11 += count,
                _ => {}
            }
        }
        
        let total = self.shots as f64;
        let p00 = count_00 as f64 / total;
        let p01 = count_01 as f64 / total;
        let p10 = count_10 as f64 / total;
        let p11 = count_11 as f64 / total;
        
        let p1_first = p10 + p11;
        let p1_second = p01 + p11;
        
        // Calculate correlation: <Z_i Z_j> = p00 + p11 - p01 - p10
        let correlation = p00 + p11 - p01 - p10;
        
        Ok(correlation)
    }
    
    /// Apply error mitigation using matrix inversion
    fn mitigate_errors(&self, py: Python, error_matrix: PyReadonlyArray2<f64>) -> PyResult<Py<PyMeasurementResult>> {
        let error_mat = error_matrix.as_array();
        let n_states = 1 << self.n_qubits;
        
        if error_mat.shape() != [n_states, n_states] {
            return Err(PyValueError::new_err(format!(
                "Error matrix shape {:?} doesn't match expected ({}, {})",
                error_mat.shape(), n_states, n_states
            )));
        }
        
        // Convert counts to probability vector
        let mut prob_vec = Array1::zeros(n_states);
        let total = self.shots as f64;
        
        for (bitstring, count) in &self.counts {
            let index = usize::from_str_radix(bitstring, 2).unwrap();
            prob_vec[index] = *count as f64 / total;
        }
        
        // Apply inverse of error matrix
        let inv_error_mat = invert_matrix(&error_mat)?;
        let mitigated_probs = inv_error_mat.dot(&prob_vec);
        
        // Convert back to counts (ensuring non-negative)
        let mut new_counts = HashMap::new();
        for (i, &prob) in mitigated_probs.iter().enumerate() {
            if prob > 1e-10 {
                let bitstring = format!("{:0width$b}", i, width = self.n_qubits);
                let count = (prob * total).round() as usize;
                if count > 0 {
                    new_counts.insert(bitstring, count);
                }
            }
        }
        
        Py::new(py, PyMeasurementResult {
            counts: new_counts,
            shots: self.shots,
            n_qubits: self.n_qubits,
        })
    }
}

/// Quantum state tomography
#[pyclass(name = "StateTomography")]
pub struct PyStateTomography {
    n_qubits: usize,
}

#[pymethods]
impl PyStateTomography {
    #[new]
    fn new(n_qubits: usize) -> Self {
        Self { n_qubits }
    }
    
    /// Generate measurement circuits for state tomography
    fn measurement_circuits(&self, py: Python) -> PyResult<PyObject> {
        let bases = vec!["X", "Y", "Z"];
        let n_bases = bases.len();
        let n_circuits = n_bases.pow(self.n_qubits as u32);
        
        let circuits = PyList::empty(py);
        
        for i in 0..n_circuits {
            let mut basis_string = String::new();
            let mut circuit = PyCircuit::new(self.n_qubits)?;
            
            // Convert index to measurement basis for each qubit
            let mut idx = i;
            for qubit in 0..self.n_qubits {
                let basis_idx = idx % n_bases;
                idx /= n_bases;
                
                basis_string.push_str(bases[basis_idx]);
                
                // Apply basis transformation
                match bases[basis_idx] {
                    "X" => {
                        // Measure in X basis: apply Hadamard
                        circuit.h(qubit)?;
                    }
                    "Y" => {
                        // Measure in Y basis: apply S† then H
                        circuit.sdg(qubit)?;
                        circuit.h(qubit)?;
                    }
                    "Z" => {
                        // Measure in Z basis: no transformation needed
                    }
                    _ => unreachable!(),
                }
            }
            
            let circuit_info = PyDict::new(py);
            circuit_info.set_item("circuit", Py::new(py, circuit)?)?;
            circuit_info.set_item("basis", basis_string)?;
            circuits.append(circuit_info)?;
        }
        
        Ok(circuits.into())
    }
    
    /// Reconstruct density matrix from measurement results
    fn reconstruct_state<'py>(
        &self,
        py: Python<'py>,
        measurements: &Bound<'py, PyList>,
    ) -> PyResult<Py<PyArray2<Complex64>>> {
        let dim = 1 << self.n_qubits;
        let mut density_matrix = Array2::zeros((dim, dim));
        
        // Collect all measurement data
        let mut measurement_data = Vec::new();
        for item in measurements {
            let dict = item.downcast::<PyDict>()?;
            let basis: String = dict.get_item("basis")?.unwrap().extract()?;
            let result: PyRef<PyMeasurementResult> = dict.get_item("result")?.unwrap().extract()?;
            measurement_data.push((basis, result));
        }
        
        // Use maximum likelihood estimation (simplified version)
        // In practice, this would use iterative optimization
        
        // For now, reconstruct assuming pure state from Z-basis measurements
        for (basis, result) in &measurement_data {
            if basis.chars().all(|c| c == 'Z') {
                // Use Z-basis measurements to estimate diagonal elements
                for (bitstring, count) in &result.counts {
                    let idx = usize::from_str_radix(bitstring, 2).unwrap();
                    let prob = *count as f64 / result.shots as f64;
                    density_matrix[[idx, idx]] = Complex64::new(prob, 0.0);
                }
                break;
            }
        }
        
        // Normalize trace
        let trace: Complex64 = (0..dim).map(|i| density_matrix[[i, i]]).sum();
        if trace.norm() > 0.0 {
            density_matrix /= trace;
        }
        
        Ok(density_matrix.into_pyarray(py).into())
    }
    
    /// Calculate fidelity between reconstructed and target states
    fn fidelity(&self, state1: PyReadonlyArray2<Complex64>, state2: PyReadonlyArray2<Complex64>) -> PyResult<f64> {
        let rho1 = state1.as_array();
        let rho2 = state2.as_array();
        
        if rho1.shape() != rho2.shape() {
            return Err(PyValueError::new_err("States must have the same dimensions"));
        }
        
        // For pure states: F = |<ψ1|ψ2>|²
        // For mixed states: F = Tr(√(√ρ1 ρ2 √ρ1))²
        
        // Simplified calculation for diagonal matrices
        let mut fidelity = 0.0;
        let n = rho1.nrows();
        for i in 0..n {
            fidelity += (rho1[[i, i]].norm() * rho2[[i, i]].norm()).sqrt();
        }
        
        Ok(fidelity * fidelity)
    }
}

/// Process tomography for quantum operations
#[pyclass(name = "ProcessTomography")]
pub struct PyProcessTomography {
    n_qubits: usize,
}

#[pymethods]
impl PyProcessTomography {
    #[new]
    fn new(n_qubits: usize) -> Self {
        Self { n_qubits }
    }
    
    /// Generate input states for process tomography
    fn input_states(&self, py: Python) -> PyResult<PyObject> {
        let states = PyList::empty(py);
        
        // Standard input states: |0>, |1>, |+>, |->, |+i>, |-i> per qubit
        let state_names = vec!["0", "1", "+", "-", "+i", "-i"];
        let n_states = state_names.len();
        let n_configs = n_states.pow(self.n_qubits as u32);
        
        for i in 0..n_configs {
            let mut config = String::new();
            let mut circuit = PyCircuit::new(self.n_qubits)?;
            
            let mut idx = i;
            for qubit in 0..self.n_qubits {
                let state_idx = idx % n_states;
                idx /= n_states;
                
                config.push_str(state_names[state_idx]);
                if qubit < self.n_qubits - 1 {
                    config.push(',');
                }
                
                // Prepare input state
                match state_names[state_idx] {
                    "0" => {} // |0> is default
                    "1" => circuit.x(qubit)?,
                    "+" => circuit.h(qubit)?,
                    "-" => {
                        circuit.x(qubit)?;
                        circuit.h(qubit)?;
                    }
                    "+i" => {
                        circuit.h(qubit)?;
                        circuit.s(qubit)?;
                    }
                    "-i" => {
                        circuit.h(qubit)?;
                        circuit.sdg(qubit)?;
                    }
                    _ => unreachable!(),
                }
            }
            
            let state_info = PyDict::new(py);
            state_info.set_item("circuit", Py::new(py, circuit)?)?;
            state_info.set_item("state", config)?;
            states.append(state_info)?;
        }
        
        Ok(states.into())
    }
    
    /// Reconstruct process matrix (chi matrix) from tomography data
    fn reconstruct_process<'py>(
        &self,
        py: Python<'py>,
        tomography_data: &Bound<'py, PyDict>,
    ) -> PyResult<Py<PyArray2<Complex64>>> {
        let dim = 1 << self.n_qubits;
        let dim_sq = dim * dim;
        
        // Chi matrix for process tomography
        let mut chi_matrix = Array2::zeros((dim_sq, dim_sq));
        
        // Simplified reconstruction (placeholder)
        // In practice, this would use maximum likelihood estimation
        
        // For now, return identity process
        for i in 0..dim_sq {
            chi_matrix[[i, i]] = Complex64::new(1.0 / dim_sq as f64, 0.0);
        }
        
        Ok(chi_matrix.into_pyarray(py).into())
    }
    
    /// Calculate process fidelity
    fn process_fidelity(&self, chi1: PyReadonlyArray2<Complex64>, chi2: PyReadonlyArray2<Complex64>) -> PyResult<f64> {
        let c1 = chi1.as_array();
        let c2 = chi2.as_array();
        
        if c1.shape() != c2.shape() {
            return Err(PyValueError::new_err("Process matrices must have the same dimensions"));
        }
        
        // Process fidelity: F = Tr(χ1 χ2) / d²
        let dim = (c1.nrows() as f64).sqrt() as usize;
        let mut fidelity = Complex64::new(0.0, 0.0);
        
        for i in 0..c1.nrows() {
            for j in 0..c1.ncols() {
                fidelity += c1[[i, j]] * c2[[j, i]];
            }
        }
        
        Ok(fidelity.re / (dim * dim) as f64)
    }
}

/// Measurement sampler for generating shot-based results
#[pyclass(name = "MeasurementSampler")]
pub struct PyMeasurementSampler {}

#[pymethods]
impl PyMeasurementSampler {
    #[new]
    fn new() -> Self {
        Self {}
    }
    
    /// Sample measurements from a state vector
    fn sample_counts(
        &self,
        py: Python,
        result: &PySimulationResult,
        shots: usize,
    ) -> PyResult<Py<PyMeasurementResult>> {
        let mut rng = rand::thread_rng();
        let mut counts = HashMap::new();
        
        // Get probabilities
        let probs: Vec<f64> = result.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .collect();
        
        // Sample measurements
        for _ in 0..shots {
            let r: f64 = rng.gen();
            let mut cumsum = 0.0;
            
            for (idx, &prob) in probs.iter().enumerate() {
                cumsum += prob;
                if r < cumsum {
                    let bitstring = format!("{:0width$b}", idx, width = result.n_qubits);
                    *counts.entry(bitstring).or_insert(0) += 1;
                    break;
                }
            }
        }
        
        Py::new(py, PyMeasurementResult {
            counts,
            shots,
            n_qubits: result.n_qubits,
        })
    }
    
    /// Sample measurements with readout error
    fn sample_with_error(
        &self,
        py: Python,
        result: &PySimulationResult,
        shots: usize,
        error_rate: f64,
    ) -> PyResult<Py<PyMeasurementResult>> {
        let mut rng = rand::thread_rng();
        let mut counts = HashMap::new();
        
        // Get probabilities
        let probs: Vec<f64> = result.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .collect();
        
        // Sample measurements
        for _ in 0..shots {
            let r: f64 = rng.gen();
            let mut cumsum = 0.0;
            
            for (idx, &prob) in probs.iter().enumerate() {
                cumsum += prob;
                if r < cumsum {
                    let mut bitstring = format!("{:0width$b}", idx, width = result.n_qubits);
                    
                    // Apply readout error
                    let mut chars: Vec<char> = bitstring.chars().collect();
                    for i in 0..chars.len() {
                        if rng.gen::<f64>() < error_rate {
                            chars[i] = if chars[i] == '0' { '1' } else { '0' };
                        }
                    }
                    bitstring = chars.into_iter().collect();
                    
                    *counts.entry(bitstring).or_insert(0) += 1;
                    break;
                }
            }
        }
        
        Py::new(py, PyMeasurementResult {
            counts,
            shots,
            n_qubits: result.n_qubits,
        })
    }
}

/// Helper function to invert a matrix (simplified)
fn invert_matrix(matrix: &ArrayView2<f64>) -> PyResult<Array2<f64>> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(PyValueError::new_err("Matrix must be square"));
    }
    
    // For small matrices, use naive Gaussian elimination
    // In practice, use a proper linear algebra library
    let mut aug = Array2::zeros((n, 2 * n));
    
    // Create augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = matrix[[i, j]];
            if i == j {
                aug[[i, n + j]] = 1.0;
            }
        }
    }
    
    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }
        
        // Swap rows
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }
        
        // Scale pivot row
        let pivot = aug[[i, i]];
        if pivot.abs() < 1e-10 {
            return Err(PyValueError::new_err("Matrix is singular"));
        }
        
        for j in 0..(2 * n) {
            aug[[i, j]] /= pivot;
        }
        
        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[[k, i]];
                for j in 0..(2 * n) {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }
    }
    
    // Extract inverse from augmented matrix
    let mut inverse = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inverse[[i, j]] = aug[[i, n + j]];
        }
    }
    
    Ok(inverse)
}

/// Register the measurement module
pub fn register_measurement_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(m.py(), "measurement")?;
    
    submodule.add_class::<PyMeasurementResult>()?;
    submodule.add_class::<PyStateTomography>()?;
    submodule.add_class::<PyProcessTomography>()?;
    submodule.add_class::<PyMeasurementSampler>()?;
    
    m.add_submodule(&submodule)?;
    Ok(())
}