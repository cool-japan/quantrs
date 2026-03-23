//! `PySimulationResult` — wrapper around quantum simulation results for Python.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict, PyList};
use scirs2_core::Complex64;

/// Python wrapper for simulation results
#[pyclass]
pub struct PySimulationResult {
    /// The state vector amplitudes
    pub(crate) amplitudes: Vec<Complex64>,
    /// The number of qubits
    pub(crate) n_qubits: usize,
}

#[pymethods]
impl PySimulationResult {
    /// Get the state vector amplitudes
    fn amplitudes(&self, py: Python) -> PyResult<PyObject> {
        let result = PyList::empty(py);
        for amp in &self.amplitudes {
            let complex = PyComplex::from_doubles(py, amp.re, amp.im);
            result.append(complex)?;
        }
        Ok(result.into())
    }

    /// Get the probabilities for each basis state
    fn probabilities(&self, py: Python) -> PyResult<PyObject> {
        let result = PyList::empty(py);
        for amp in &self.amplitudes {
            let prob = amp.norm_sqr();
            result.append(prob)?;
        }
        Ok(result.into())
    }

    /// Get the number of qubits
    #[allow(clippy::missing_const_for_fn)]
    #[getter]
    fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Get a dictionary mapping basis states to probabilities
    fn state_probabilities(&self, py: Python) -> PyResult<PyObject> {
        let result = PyDict::new(py);
        for (i, amp) in self.amplitudes.iter().enumerate() {
            let basis_state = format!("{:0width$b}", i, width = self.n_qubits);
            let prob = amp.norm_sqr();
            // Only include states with non-zero probability
            if prob > 1e-10 {
                result.set_item(basis_state, prob)?;
            }
        }
        Ok(result.into())
    }

    /// Get the expectation value of a Pauli operator
    ///
    /// Computes ⟨ψ|P|ψ⟩ where P is the tensor product of Pauli operators.
    /// The operator string should have one character per qubit (I, X, Y, or Z).
    fn expectation_value(&self, operator: &str) -> PyResult<f64> {
        if operator.len() != self.n_qubits {
            return Err(PyValueError::new_err(format!(
                "Operator length ({}) must match number of qubits ({})",
                operator.len(),
                self.n_qubits
            )));
        }

        let paulis: Vec<char> = operator.chars().collect();
        for &c in &paulis {
            if c != 'I' && c != 'X' && c != 'Y' && c != 'Z' {
                return Err(PyValueError::new_err(format!(
                    "Invalid Pauli operator: {c}. Only I, X, Y, Z are allowed"
                )));
            }
        }

        let n = self.n_qubits;
        let dim = 1 << n; // 2^n basis states
        let mut expectation = Complex64::new(0.0, 0.0);

        // For each basis state |i⟩, compute ⟨i|P applied to ψ contribution
        for i in 0..dim {
            // Apply Pauli string to |i⟩ to get phase * |j⟩
            let mut j = i;
            let mut phase = Complex64::new(1.0, 0.0);

            for (qubit_idx, &pauli) in paulis.iter().enumerate() {
                // Qubit 0 corresponds to MSB (leftmost in operator string)
                let bit_position = n - 1 - qubit_idx;
                let bit = (i >> bit_position) & 1;

                #[allow(clippy::collapsible_match)]
                match pauli {
                    'X' => {
                        // X|0⟩ = |1⟩, X|1⟩ = |0⟩ (flip the bit)
                        j ^= 1 << bit_position;
                    }
                    'Y' => {
                        // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                        j ^= 1 << bit_position;
                        if bit == 0 {
                            phase *= Complex64::new(0.0, 1.0); // i
                        } else {
                            phase *= Complex64::new(0.0, -1.0); // -i
                        }
                    }
                    'Z' => {
                        // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
                        if bit == 1 {
                            phase *= Complex64::new(-1.0, 0.0);
                        }
                    }
                    // 'I' and any other already-validated characters: no change
                    _ => {}
                }
            }

            // Contribution: ψ*_i * phase * ψ_j
            expectation += self.amplitudes[i].conj() * phase * self.amplitudes[j];
        }

        // For Hermitian operators (like Pauli strings), the expectation value is real
        Ok(expectation.re)
    }
}
