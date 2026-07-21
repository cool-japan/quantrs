//! Measurement statistics and quantum state tomography.
//!
//! This module provides tools for:
//! - Measurement outcome statistics
//! - Quantum state tomography
//! - Process tomography
//! - Measurement error mitigation

// Allow unused_self for PyO3 method bindings and unnecessary_wraps for future error handling
#![allow(clippy::unused_self)]
#![allow(clippy::unnecessary_wraps)]

use crate::{PyCircuit, PySimulationResult};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use scirs2_numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use std::collections::HashMap;

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
    fn get_counts(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (bitstring, count) in &self.counts {
            dict.set_item(bitstring, count)?;
        }
        Ok(dict.into())
    }

    /// Get the measurement probabilities
    fn get_probabilities(&self, py: Python) -> PyResult<Py<PyAny>> {
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
        self.counts
            .iter()
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
    fn mitigate_errors(
        &self,
        py: Python,
        error_matrix: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<Self>> {
        let error_mat = error_matrix.as_array();
        let n_states = 1 << self.n_qubits;

        if error_mat.shape() != [n_states, n_states] {
            return Err(PyValueError::new_err(format!(
                "Error matrix shape {:?} doesn't match expected ({}, {})",
                error_mat.shape(),
                n_states,
                n_states
            )));
        }

        // Convert counts to probability vector
        let mut prob_vec = Array1::zeros(n_states);
        let total = self.shots as f64;

        for (bitstring, count) in &self.counts {
            let index = usize::from_str_radix(bitstring, 2).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to parse binary bitstring '{bitstring}': {e}"
                ))
            })?;
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

        Py::new(
            py,
            Self {
                counts: new_counts,
                shots: self.shots,
                n_qubits: self.n_qubits,
            },
        )
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
    const fn new(n_qubits: usize) -> Self {
        Self { n_qubits }
    }

    /// Generate measurement circuits for state tomography
    fn measurement_circuits(&self, py: Python) -> PyResult<Py<PyAny>> {
        let bases = ["X", "Y", "Z"];
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

    /// Reconstruct density matrix from measurement results via linear
    /// inversion over the full Pauli basis (all `4^n_qubits` Pauli strings),
    /// using whichever supplied basis setting matches each string's
    /// non-identity factors -- see [`reconstruct_density_matrix_from_counts`]
    /// for the algorithm (shared with `scirs2_bindings.rs`'s
    /// `QuantumNumerics.state_tomography`).
    fn reconstruct_state<'py>(
        &self,
        py: Python<'py>,
        measurements: &Bound<'py, PyList>,
    ) -> PyResult<Py<PyArray2<Complex64>>> {
        // Collect all measurement data: (basis string, outcome counts, shots).
        let mut measurement_data: Vec<(String, HashMap<String, usize>, usize)> = Vec::new();
        for item in measurements {
            let dict = item.cast::<PyDict>()?;
            let basis: String = dict
                .get_item("basis")?
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "Missing 'basis' key in measurement data",
                    )
                })?
                .extract()?;
            let result: PyRef<PyMeasurementResult> = dict
                .get_item("result")?
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "Missing 'result' key in measurement data",
                    )
                })?
                .extract()?;
            measurement_data.push((basis, result.counts.clone(), result.shots));
        }

        let density_matrix =
            reconstruct_density_matrix_from_counts(self.n_qubits, &measurement_data)
                .map_err(PyValueError::new_err)?;

        Ok(density_matrix.into_pyarray(py).into())
    }

    /// Calculate fidelity between reconstructed and target states
    fn fidelity(
        &self,
        state1: PyReadonlyArray2<Complex64>,
        state2: PyReadonlyArray2<Complex64>,
    ) -> PyResult<f64> {
        let rho1 = state1.as_array();
        let rho2 = state2.as_array();

        if rho1.shape() != rho2.shape() {
            return Err(PyValueError::new_err(
                "States must have the same dimensions",
            ));
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

/// Real linear-inversion state tomography over the Pauli basis, built from
/// measurement counts. Pure Rust (`Result<_, String>`, no `PyErr`) so this is
/// directly unit-testable without a Python interpreter (this crate builds
/// `pyo3` with the `extension-module` feature, so a standalone test binary
/// cannot resolve the CPython C-API symbols that `PyErr` construction pulls
/// in, even along an error branch a test doesn't take).
///
/// Reconstructs `rho = (1/d) * sum_P <P> * P` over all `4^n_qubits` Pauli
/// strings `P`, where `<P>` is estimated from whichever supplied basis
/// setting (`measurement_data`'s basis strings, one character per qubit from
/// `{'X', 'Y', 'Z'}`, matching `measurement_circuits()`) matches `P`'s
/// non-identity factors. Qubit `q`'s bit occupies position
/// `n_qubits - 1 - q` of each outcome bitstring (matching this crate's
/// existing bitstring-formatting convention, e.g.
/// `PyMeasurementResult::marginal_probability`).
fn reconstruct_density_matrix_from_counts(
    n_qubits: usize,
    measurement_data: &[(String, HashMap<String, usize>, usize)],
) -> Result<Array2<Complex64>, String> {
    if n_qubits == 0 || n_qubits > 6 {
        return Err("Exact linear-inversion state tomography supports 1 to 6 qubits".to_string());
    }
    let dim = 1usize << n_qubits;

    let mut by_basis: HashMap<&str, usize> = HashMap::new();
    for (row, (basis, _, _)) in measurement_data.iter().enumerate() {
        if basis.chars().count() != n_qubits || !basis.chars().all(|c| matches!(c, 'X' | 'Y' | 'Z'))
        {
            return Err(format!(
                "Invalid basis string '{basis}': expected {n_qubits} character(s) from {{'X','Y','Z'}}"
            ));
        }
        by_basis.entry(basis.as_str()).or_insert(row);
    }
    if by_basis.is_empty() {
        return Err("No measurement data provided".to_string());
    }

    let pauli = |kind: u8| -> Array2<Complex64> {
        match kind {
            0 => scirs2_core::ndarray::array![
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            ],
            1 => scirs2_core::ndarray::array![
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            ],
            2 => scirs2_core::ndarray::array![
                [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
                [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
            ],
            _ => scirs2_core::ndarray::array![
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
            ],
        }
    };

    let kron = |a: &Array2<Complex64>, b: &Array2<Complex64>| -> Array2<Complex64> {
        let (ar, ac) = (a.nrows(), a.ncols());
        let (br, bc) = (b.nrows(), b.ncols());
        let mut out = Array2::<Complex64>::zeros((ar * br, ac * bc));
        for i in 0..ar {
            for j in 0..ac {
                for p in 0..br {
                    for q in 0..bc {
                        out[[i * br + p, j * bc + q]] = a[[i, j]] * b[[p, q]];
                    }
                }
            }
        }
        out
    };

    let mut rho = Array2::<Complex64>::zeros((dim, dim));
    let mut n_nontrivial_terms_used = 0usize;
    let n_pauli_strings = 4usize.pow(n_qubits as u32);

    for code in 0..n_pauli_strings {
        // Base-4 digits of `code`: 0=I, 1=X, 2=Y, 3=Z per qubit.
        let mut digits = vec![0u8; n_qubits];
        let mut c = code;
        for d in digits.iter_mut().rev() {
            *d = (c % 4) as u8;
            c /= 4;
        }

        // <I (x) I (x) ... (x) I> = 1 always: no data needed.
        let expectation = if code == 0 {
            1.0
        } else {
            // For identity factors, 'Z' is used as an arbitrary filler basis:
            // marginalizing (summing) over a qubit's outcome reproduces the
            // reduced-state statistics regardless of which basis that qubit
            // was measured in, so any matching setting is valid.
            let required_basis: String = digits
                .iter()
                .map(|&d| match d {
                    1 => 'X',
                    2 => 'Y',
                    _ => 'Z',
                })
                .collect();

            let Some(&row) = by_basis.get(required_basis.as_str()) else {
                continue; // this Pauli term's required setting was not measured
            };

            let (_, counts, shots) = &measurement_data[row];
            if *shots == 0 {
                continue; // no usable data in this setting
            }
            n_nontrivial_terms_used += 1;

            let mut expectation = 0.0_f64;
            for (bitstring, &count) in counts {
                let idx = usize::from_str_radix(bitstring, 2)
                    .map_err(|e| format!("Failed to parse binary bitstring '{bitstring}': {e}"))?;
                let mut sign = 1.0_f64;
                for (qubit, &d) in digits.iter().enumerate() {
                    if d != 0 {
                        let bit = (idx >> (n_qubits - 1 - qubit)) & 1;
                        if bit == 1 {
                            sign = -sign;
                        }
                    }
                }
                expectation = sign.mul_add(count as f64 / *shots as f64, expectation);
            }
            expectation
        };

        let mut term = pauli(digits[0]);
        for &d in &digits[1..] {
            term = kron(&term, &pauli(d));
        }
        let coeff = Complex64::new(expectation / dim as f64, 0.0);
        rho = rho + term.mapv(|v| v * coeff);
    }

    if n_nontrivial_terms_used == 0 {
        return Err(
            "No usable measurement data: none of the required Pauli-basis settings were found"
                .to_string(),
        );
    }

    Ok(rho)
}

/// Decompose a single-qubit computational-basis matrix element `|a><b|`
/// (`a, b` in `{0, 1}`) as a complex-weighted combination of the six
/// canonical single-qubit state labels used by
/// [`PyProcessTomography::input_states`]. Derived from
/// `|+-> = (|0> +- |1>)/sqrt(2)` and `|+-i> = (|0> +- i|1>)/sqrt(2)`:
/// `|0><1| + |1><0| = rho_+ - rho_-` and
/// `i(|0><1| - |1><0|) = rho_{-i} - rho_{+i}`, solved for `|0><1|`/`|1><0|`.
fn single_qubit_element_decomposition(a: u8, b: u8) -> Vec<(&'static str, Complex64)> {
    match (a, b) {
        (0, 0) => vec![("0", Complex64::new(1.0, 0.0))],
        (1, 1) => vec![("1", Complex64::new(1.0, 0.0))],
        (0, 1) => vec![
            ("+", Complex64::new(0.5, 0.0)),
            ("-", Complex64::new(-0.5, 0.0)),
            ("+i", Complex64::new(0.0, 0.5)),
            ("-i", Complex64::new(0.0, -0.5)),
        ],
        (1, 0) => vec![
            ("+", Complex64::new(0.5, 0.0)),
            ("-", Complex64::new(-0.5, 0.0)),
            ("+i", Complex64::new(0.0, -0.5)),
            ("-i", Complex64::new(0.0, 0.5)),
        ],
        _ => unreachable!("single-qubit computational-basis indices are always 0 or 1"),
    }
}

/// Cartesian-product expansion of independent per-qubit decompositions into
/// full, comma-joined multi-qubit labels with combined (multiplied)
/// coefficients, matching [`PyProcessTomography::input_states`]'s label
/// format (`"s_0,s_1,...,s_{n-1}"`).
fn expand_multi_qubit_labels(
    per_qubit_terms: &[Vec<(&'static str, Complex64)>],
) -> Vec<(String, Complex64)> {
    let mut acc: Vec<(String, Complex64)> = vec![(String::new(), Complex64::new(1.0, 0.0))];
    for (qubit, terms) in per_qubit_terms.iter().enumerate() {
        let mut next = Vec::with_capacity(acc.len() * terms.len());
        for (label_prefix, coeff_prefix) in &acc {
            for (label, coeff) in terms {
                let mut new_label = label_prefix.clone();
                if qubit > 0 {
                    new_label.push(',');
                }
                new_label.push_str(label);
                next.push((new_label, coeff_prefix * coeff));
            }
        }
        acc = next;
    }
    acc
}

/// Real linear-inversion process-tomography reconstruction: assembles the
/// (`1/dim`-normalized) Choi-Jamiolkowski matrix of a quantum process `E`
/// from tomography data `{input_label -> E(rho_input_label)}`. Pure Rust
/// (`Result<_, String>`, no `PyErr`) for direct unit-testability -- see the
/// note on [`reconstruct_density_matrix_from_counts`].
///
/// Qubit `q` occupies bit position `n_qubits - 1 - q` of each computational
/// basis multi-index (consistent with this module's other tomography code).
fn reconstruct_choi_matrix(
    n_qubits: usize,
    tomography_data: &HashMap<String, Array2<Complex64>>,
) -> Result<Array2<Complex64>, String> {
    if n_qubits == 0 || n_qubits > 4 {
        return Err("Process tomography reconstruction supports 1 to 4 qubits".to_string());
    }
    if tomography_data.is_empty() {
        return Err("No tomography data provided".to_string());
    }
    let dim = 1usize << n_qubits;
    let dim_sq = dim * dim;

    let mut choi = Array2::<Complex64>::zeros((dim_sq, dim_sq));
    let mut n_blocks_used = 0usize;

    for a_idx in 0..dim {
        for b_idx in 0..dim {
            let per_qubit_terms: Vec<Vec<(&'static str, Complex64)>> = (0..n_qubits)
                .map(|q| {
                    let a_bit = ((a_idx >> (n_qubits - 1 - q)) & 1) as u8;
                    let b_bit = ((b_idx >> (n_qubits - 1 - q)) & 1) as u8;
                    single_qubit_element_decomposition(a_bit, b_bit)
                })
                .collect();

            let mut e_ab = Array2::<Complex64>::zeros((dim, dim));
            let mut any_term_used = false;
            for (label, coeff) in expand_multi_qubit_labels(&per_qubit_terms) {
                if let Some(output_matrix) = tomography_data.get(&label) {
                    if output_matrix.nrows() != dim || output_matrix.ncols() != dim {
                        return Err(format!(
                            "tomography_data['{label}'] must be a {dim}x{dim} matrix for a \
                             {n_qubits}-qubit process, got {}x{}",
                            output_matrix.nrows(),
                            output_matrix.ncols()
                        ));
                    }
                    e_ab = e_ab + output_matrix.mapv(|v| v * coeff);
                    any_term_used = true;
                }
            }
            if any_term_used {
                n_blocks_used += 1;
            }

            for i in 0..dim {
                for j in 0..dim {
                    choi[[a_idx * dim + i, b_idx * dim + j]] = e_ab[[i, j]];
                }
            }
        }
    }

    if n_blocks_used == 0 {
        return Err(
            "No usable tomography data: none of the required input-state labels were found"
                .to_string(),
        );
    }

    // Normalize by `dim` (documented Choi-matrix convention for this API;
    // see `reconstruct_process`'s doc comment).
    let normalization = Complex64::new(1.0 / dim as f64, 0.0);
    Ok(choi.mapv(|v| v * normalization))
}

/// Process tomography for quantum operations
#[pyclass(name = "ProcessTomography")]
pub struct PyProcessTomography {
    n_qubits: usize,
}

#[pymethods]
impl PyProcessTomography {
    #[new]
    const fn new(n_qubits: usize) -> Self {
        Self { n_qubits }
    }

    /// Generate input states for process tomography
    fn input_states(&self, py: Python) -> PyResult<Py<PyAny>> {
        let states = PyList::empty(py);

        // Standard input states: |0>, |1>, |+>, |->, |+i>, |-i> per qubit
        let state_names = ["0", "1", "+", "-", "+i", "-i"];
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

    /// Reconstruct a process (Choi-Jamiolkowski) matrix from tomography data.
    ///
    /// `tomography_data` must map input-state labels produced by
    /// [`Self::input_states`] (comma-joined per-qubit labels from
    /// `{'0','1','+','-','+i','-i'}`, e.g. `"0,1,+"` for 3 qubits) to the
    /// tomographically-reconstructed *output* density matrix
    /// `E(rho_in)` (a `2^n_qubits x 2^n_qubits` complex numpy array) measured
    /// after sending that input state through the process `E`.
    ///
    /// Real linear-inversion reconstruction: since every input basis state
    /// preparation is a tensor product of single-qubit states, and each
    /// single-qubit computational-basis matrix element `|a><b|` (a,b in
    /// {0,1}) is an exact linear combination of the six prepared single-qubit
    /// states (`|0><1| = (rho_+ - rho_-)/2 + i(rho_{+i} - rho_{-i})/2` and its
    /// conjugate-transpose counterpart), `E(|A><B|)` for *any* pair of
    /// computational-basis multi-indices `A, B` can be recovered as a linear
    /// combination of the supplied `tomography_data` entries (no entangled
    /// inputs needed). Assembling `E(|A><B|)` into the standard Choi-matrix
    /// block layout gives the process's Choi-Jamiolkowski representation,
    /// normalized here by `1/dim` (documented deviation from a literal
    /// Pauli-operator chi-matrix expansion -- see the module-level notes in
    /// this file's `#[cfg(test)]` module for a worked identity-channel check).
    fn reconstruct_process<'py>(
        &self,
        py: Python<'py>,
        tomography_data: &Bound<'py, PyDict>,
    ) -> PyResult<Py<PyArray2<Complex64>>> {
        let mut data: HashMap<String, Array2<Complex64>> = HashMap::new();
        for (key, value) in tomography_data.iter() {
            let label: String = key.extract()?;
            let matrix: PyReadonlyArray2<Complex64> = value.extract()?;
            data.insert(label, matrix.as_array().to_owned());
        }

        let choi = reconstruct_choi_matrix(self.n_qubits, &data).map_err(PyValueError::new_err)?;

        Ok(choi.into_pyarray(py).into())
    }

    /// Calculate process fidelity
    fn process_fidelity(
        &self,
        chi1: PyReadonlyArray2<Complex64>,
        chi2: PyReadonlyArray2<Complex64>,
    ) -> PyResult<f64> {
        let c1 = chi1.as_array();
        let c2 = chi2.as_array();

        if c1.shape() != c2.shape() {
            return Err(PyValueError::new_err(
                "Process matrices must have the same dimensions",
            ));
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
    const fn new() -> Self {
        Self {}
    }

    /// Sample measurements from a state vector
    fn sample_counts(
        &self,
        py: Python,
        result: &PySimulationResult,
        shots: usize,
    ) -> PyResult<Py<PyMeasurementResult>> {
        let mut rng = thread_rng();
        let mut counts = HashMap::new();

        // Get probabilities
        let probs: Vec<f64> = result
            .amplitudes
            .iter()
            .map(scirs2_core::Complex::norm_sqr)
            .collect();

        // Sample measurements
        for _ in 0..shots {
            let r: f64 = rng.random();
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

        Py::new(
            py,
            PyMeasurementResult {
                counts,
                shots,
                n_qubits: result.n_qubits,
            },
        )
    }

    /// Sample measurements with readout error
    fn sample_with_error(
        &self,
        py: Python,
        result: &PySimulationResult,
        shots: usize,
        error_rate: f64,
    ) -> PyResult<Py<PyMeasurementResult>> {
        let mut rng = thread_rng();
        let mut counts = HashMap::new();

        // Get probabilities
        let probs: Vec<f64> = result
            .amplitudes
            .iter()
            .map(scirs2_core::Complex::norm_sqr)
            .collect();

        // Sample measurements
        for _ in 0..shots {
            let r: f64 = rng.random();
            let mut cumsum = 0.0;

            for (idx, &prob) in probs.iter().enumerate() {
                cumsum += prob;
                if r < cumsum {
                    let mut bitstring = format!("{:0width$b}", idx, width = result.n_qubits);

                    // Apply readout error
                    let mut chars: Vec<char> = bitstring.chars().collect();
                    for c in &mut chars {
                        if rng.random::<f64>() < error_rate {
                            *c = if *c == '0' { '1' } else { '0' };
                        }
                    }
                    bitstring = chars.into_iter().collect();

                    *counts.entry(bitstring).or_insert(0) += 1;
                    break;
                }
            }
        }

        Py::new(
            py,
            PyMeasurementResult {
                counts,
                shots,
                n_qubits: result.n_qubits,
            },
        )
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

// Pure-Rust regression tests for the free functions above (no `PyErr`
// involved, hence directly unit-testable without a Python interpreter --
// this crate builds `pyo3` with the `extension-module` feature, so a
// standalone test binary cannot resolve the CPython C-API symbols `PyErr`
// construction pulls in, even along a branch a test never takes).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruct_density_matrix_from_counts_recovers_a_pure_zero_state() {
        // A single qubit prepared in |0>: Z-basis measurement is
        // deterministic, X/Y-basis measurements are maximally random.
        let measurement_data = vec![
            (
                "Z".to_string(),
                HashMap::from([("0".to_string(), 100usize)]),
                100usize,
            ),
            (
                "X".to_string(),
                HashMap::from([("0".to_string(), 50usize), ("1".to_string(), 50usize)]),
                100usize,
            ),
            (
                "Y".to_string(),
                HashMap::from([("0".to_string(), 50usize), ("1".to_string(), 50usize)]),
                100usize,
            ),
        ];

        let rho = reconstruct_density_matrix_from_counts(1, &measurement_data)
            .expect("valid tomography input");

        assert!((rho[[0, 0]].re - 1.0).abs() < 1e-9);
        assert!(rho[[0, 0]].im.abs() < 1e-9);
        assert!(rho[[1, 1]].re.abs() < 1e-9);
        assert!(rho[[0, 1]].norm() < 1e-9);
        assert!(rho[[1, 0]].norm() < 1e-9);
    }

    #[test]
    fn reconstruct_density_matrix_from_counts_rejects_an_invalid_basis() {
        let measurement_data = vec![("W".to_string(), HashMap::new(), 0usize)];
        assert!(reconstruct_density_matrix_from_counts(1, &measurement_data).is_err());
    }

    #[test]
    fn reconstruct_density_matrix_from_counts_rejects_empty_data() {
        assert!(reconstruct_density_matrix_from_counts(1, &[]).is_err());
    }

    #[test]
    fn reconstruct_choi_matrix_of_an_identity_channel_matches_the_analytic_result() {
        // For an identity process E(rho) = rho, the (1/d-normalized) Choi
        // matrix has exactly d^2 nonzero entries, each equal to 1/d, at
        // positions (A*d+A, B*d+B) for every pair of basis indices A, B --
        // this is an independent, hand-derived analytic check, not merely a
        // self-consistency check of the implementation's own algebra.
        let half = Complex64::new(0.5, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let half_i = Complex64::new(0.0, 0.5);

        let rho0 = scirs2_core::ndarray::array![[one, zero], [zero, zero]];
        let rho1 = scirs2_core::ndarray::array![[zero, zero], [zero, one]];
        let rho_plus = scirs2_core::ndarray::array![[half, half], [half, half]];
        let rho_minus = scirs2_core::ndarray::array![[half, -half], [-half, half]];
        let rho_plus_i = scirs2_core::ndarray::array![[half, -half_i], [half_i, half]];
        let rho_minus_i = scirs2_core::ndarray::array![[half, half_i], [-half_i, half]];

        // Identity channel: output == input for every prepared state.
        let tomography_data: HashMap<String, Array2<Complex64>> = HashMap::from([
            ("0".to_string(), rho0),
            ("1".to_string(), rho1),
            ("+".to_string(), rho_plus),
            ("-".to_string(), rho_minus),
            ("+i".to_string(), rho_plus_i),
            ("-i".to_string(), rho_minus_i),
        ]);

        let choi = reconstruct_choi_matrix(1, &tomography_data).expect("valid tomography data");

        assert_eq!(choi.shape(), [4, 4]);
        let expected_nonzero: f64 = 0.5; // 1/d for d=2
        for a in 0..2usize {
            for b in 0..2usize {
                let entry = choi[[a * 2 + a, b * 2 + b]];
                assert!(
                    (entry.re - expected_nonzero).abs() < 1e-9 && entry.im.abs() < 1e-9,
                    "choi[{},{}] = {:?}, expected {expected_nonzero}",
                    a * 2 + a,
                    b * 2 + b,
                    entry
                );
            }
        }
        // An off-diagonal-within-block position must be zero for an identity channel.
        assert!(choi[[1, 1]].norm() < 1e-9, "{:?}", choi[[1, 1]]);
        assert!(choi[[0, 1]].norm() < 1e-9, "{:?}", choi[[0, 1]]);
    }

    #[test]
    fn reconstruct_choi_matrix_rejects_empty_data() {
        let empty: HashMap<String, Array2<Complex64>> = HashMap::new();
        assert!(reconstruct_choi_matrix(1, &empty).is_err());
    }
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
