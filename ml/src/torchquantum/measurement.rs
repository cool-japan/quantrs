//! Quantum measurement functions (TorchQuantum-compatible)
//!
//! This module provides measurement operations:
//! - gen_bitstrings: Generate all bitstrings for n qubits
//! - measure: Sample measurements from quantum state
//! - expval_joint_analytical: Compute expectation value analytically
//! - expval_joint_sampling: Compute expectation value by sampling
//! - TQMeasureAll: Module wrapper for measurement

use super::gates::{TQHadamard, TQS};
use super::{CType, TQDevice, TQModule, TQOperator, TQParameter};
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Generate all bitstrings for n qubits
pub fn gen_bitstrings(n_wires: usize) -> Vec<String> {
    (0..(1 << n_wires))
        .map(|k| format!("{:0width$b}", k, width = n_wires))
        .collect()
}

/// Measure the quantum state and return bitstring distribution
pub fn measure(qdev: &TQDevice, n_shots: usize) -> Vec<HashMap<String, usize>> {
    let bitstring_candidates = gen_bitstrings(qdev.n_wires);
    let probs = qdev.get_probs_1d();

    let mut distributions = Vec::with_capacity(qdev.bsz);

    for batch in 0..qdev.bsz {
        let mut counts = HashMap::new();

        // Initialize all bitstrings with 0 counts
        for bs in &bitstring_candidates {
            counts.insert(bs.clone(), 0);
        }

        // Sample from distribution
        for _ in 0..n_shots {
            let r: f64 = fastrand::f64();
            let mut cumsum = 0.0;

            for (i, &prob) in probs.row(batch).iter().enumerate() {
                cumsum += prob;
                if r < cumsum {
                    *counts.entry(bitstring_candidates[i].clone()).or_insert(0) += 1;
                    break;
                }
            }
        }

        distributions.push(counts);
    }

    distributions
}

/// Compute expectation value analytically
pub fn expval_joint_analytical(qdev: &TQDevice, observable: &str) -> Array1<f64> {
    let observable = observable.to_uppercase();
    let n_wires = qdev.n_wires;

    assert_eq!(
        observable.len(),
        n_wires,
        "Observable length must match n_wires"
    );

    let states_1d = qdev.get_states_1d();

    // Build Hamiltonian matrix
    let pauli_x = Array2::from_shape_vec(
        (2, 2),
        vec![
            CType::new(0.0, 0.0),
            CType::new(1.0, 0.0),
            CType::new(1.0, 0.0),
            CType::new(0.0, 0.0),
        ],
    )
    .unwrap_or_else(|_| Array2::eye(2).mapv(|x| CType::new(x, 0.0)));

    let pauli_y = Array2::from_shape_vec(
        (2, 2),
        vec![
            CType::new(0.0, 0.0),
            CType::new(0.0, -1.0),
            CType::new(0.0, 1.0),
            CType::new(0.0, 0.0),
        ],
    )
    .unwrap_or_else(|_| Array2::eye(2).mapv(|x| CType::new(x, 0.0)));

    let pauli_z = Array2::from_shape_vec(
        (2, 2),
        vec![
            CType::new(1.0, 0.0),
            CType::new(0.0, 0.0),
            CType::new(0.0, 0.0),
            CType::new(-1.0, 0.0),
        ],
    )
    .unwrap_or_else(|_| Array2::eye(2).mapv(|x| CType::new(x, 0.0)));

    let identity = Array2::eye(2).mapv(|x| CType::new(x, 0.0));

    // Build tensor product of Pauli matrices
    let mut hamiltonian = match observable.chars().next().unwrap_or('I') {
        'X' => pauli_x.clone(),
        'Y' => pauli_y.clone(),
        'Z' => pauli_z.clone(),
        _ => identity.clone(),
    };

    for c in observable.chars().skip(1) {
        let op = match c {
            'X' => &pauli_x,
            'Y' => &pauli_y,
            'Z' => &pauli_z,
            _ => &identity,
        };
        hamiltonian = kron(&hamiltonian, op);
    }

    // Compute <psi|H|psi> for each batch
    let mut expvals = Array1::zeros(qdev.bsz);

    for batch in 0..qdev.bsz {
        let state = states_1d.row(batch);
        let mut result = CType::new(0.0, 0.0);

        for i in 0..state.len() {
            for j in 0..state.len() {
                result += state[i].conj() * hamiltonian[[i, j]] * state[j];
            }
        }

        expvals[batch] = result.re;
    }

    expvals
}

/// Compute expectation value via sampling
pub fn expval_joint_sampling(qdev: &TQDevice, observable: &str, n_shots: usize) -> Array1<f64> {
    let observable = observable.to_uppercase();
    let n_wires = qdev.n_wires;

    // Create a clone for measurement basis rotation
    let mut qdev_clone = qdev.clone();

    // Apply rotation to measurement basis
    for (wire, c) in observable.chars().enumerate() {
        match c {
            'X' => {
                // H gate to rotate X basis to Z
                let mut h = TQHadamard::new();
                let _ = h.apply(&mut qdev_clone, &[wire]);
            }
            'Y' => {
                // S†H to rotate Y basis to Z
                let mut s = TQS::new();
                s.set_inverse(true);
                let _ = s.apply(&mut qdev_clone, &[wire]);
                let mut h = TQHadamard::new();
                let _ = h.apply(&mut qdev_clone, &[wire]);
            }
            _ => {} // Z and I don't need rotation
        }
    }

    // Measure
    let distributions = measure(&qdev_clone, n_shots);

    // Compute expectation values
    let mut expvals = Array1::zeros(qdev.bsz);

    // Create mask for non-identity positions
    let mask: Vec<bool> = observable.chars().map(|c| c != 'I').collect();

    for (batch, distri) in distributions.iter().enumerate() {
        let mut n_eigen_one = 0;
        let mut n_eigen_minus_one = 0;

        for (bitstring, &count) in distri {
            // Count parity of masked bits
            let parity: usize = bitstring
                .chars()
                .zip(mask.iter())
                .filter_map(|(c, &m)| {
                    if m {
                        c.to_digit(2).map(|d| d as usize)
                    } else {
                        None
                    }
                })
                .sum();

            if parity % 2 == 0 {
                n_eigen_one += count;
            } else {
                n_eigen_minus_one += count;
            }
        }

        expvals[batch] = (n_eigen_one as f64 - n_eigen_minus_one as f64) / n_shots as f64;
    }

    expvals
}

/// Kronecker product of two matrices
fn kron(a: &Array2<CType>, b: &Array2<CType>) -> Array2<CType> {
    let (m, n) = (a.nrows(), a.ncols());
    let (p, q) = (b.nrows(), b.ncols());

    let mut result = Array2::zeros((m * p, n * q));

    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                for l in 0..q {
                    result[[i * p + k, j * q + l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }

    result
}

/// MeasureAll module for measuring all qubits
#[derive(Debug, Clone)]
pub struct TQMeasureAll {
    /// Observable to measure (default: PauliZ)
    pub observable: String,
    static_mode: bool,
}

impl TQMeasureAll {
    pub fn new(observable: impl Into<String>) -> Self {
        Self {
            observable: observable.into(),
            static_mode: false,
        }
    }

    /// Measure with PauliZ on all qubits
    pub fn pauli_z() -> Self {
        Self::new("Z")
    }

    /// Measure with PauliX on all qubits
    pub fn pauli_x() -> Self {
        Self::new("X")
    }

    /// Measure expectation values for all qubits
    pub fn measure(&self, qdev: &TQDevice) -> Array2<f64> {
        let n_wires = qdev.n_wires;
        let mut results = Array2::zeros((qdev.bsz, n_wires));

        for wire in 0..n_wires {
            // Create observable string with observable at this wire, I elsewhere
            let obs: String = (0..n_wires)
                .map(|w| {
                    if w == wire {
                        self.observable.chars().next().unwrap_or('Z')
                    } else {
                        'I'
                    }
                })
                .collect();

            let expval = expval_joint_analytical(qdev, &obs);

            for (batch, &val) in expval.iter().enumerate() {
                results[[batch, wire]] = val;
            }
        }

        results
    }
}

impl TQModule for TQMeasureAll {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
    }

    fn n_wires(&self) -> Option<usize> {
        None
    }

    fn set_n_wires(&mut self, _n_wires: usize) {}

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
    }

    fn static_off(&mut self) {
        self.static_mode = false;
    }

    fn name(&self) -> &str {
        "MeasureAll"
    }
}
