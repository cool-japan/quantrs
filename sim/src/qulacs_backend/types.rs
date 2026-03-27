//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::*;
use scirs2_core::random::thread_rng;
use scirs2_core::{Complex64, Float};

use super::functions::{QubitIndex, StateIndex};

/// Qulacs-inspired quantum state vector
///
/// This structure provides a high-performance state vector implementation
/// following Qulacs' design principles, adapted to use SciRS2's abstractions.
#[derive(Clone)]
pub struct QulacsStateVector {
    /// The quantum state amplitudes
    state: Array1<Complex64>,
    /// Number of qubits
    pub(super) num_qubits: usize,
    /// Dimension of the state vector (2^num_qubits)
    pub(super) dim: StateIndex,
}
impl QulacsStateVector {
    /// Create a new state vector initialized to |0...0⟩
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - Number of qubits
    ///
    /// # Returns
    ///
    /// A new quantum state vector
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits == 0 {
            return Err(SimulatorError::InvalidQubitCount(
                "Number of qubits must be positive".to_string(),
            ));
        }
        if num_qubits > 30 {
            return Err(SimulatorError::InvalidQubitCount(format!(
                "Number of qubits ({}) exceeds maximum (30)",
                num_qubits
            )));
        }
        let dim = 1 << num_qubits;
        let mut state = Array1::<Complex64>::zeros(dim);
        state[0] = Complex64::new(1.0, 0.0);
        Ok(Self {
            state,
            num_qubits,
            dim,
        })
    }
    /// Create a state vector from raw amplitudes
    ///
    /// # Arguments
    ///
    /// * `amplitudes` - The state amplitudes
    ///
    /// # Returns
    ///
    /// A new quantum state vector
    pub fn from_amplitudes(amplitudes: Array1<Complex64>) -> Result<Self> {
        let dim = amplitudes.len();
        if dim == 0 || (dim & (dim - 1)) != 0 {
            return Err(SimulatorError::InvalidState(
                "Dimension must be a power of 2".to_string(),
            ));
        }
        let num_qubits = dim.trailing_zeros() as usize;
        Ok(Self {
            state: amplitudes,
            num_qubits,
            dim,
        })
    }
    /// Get the number of qubits
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
    /// Get the dimension of the state vector
    #[inline]
    pub fn dim(&self) -> StateIndex {
        self.dim
    }
    /// Get a reference to the state amplitudes
    #[inline]
    pub fn amplitudes(&self) -> &Array1<Complex64> {
        &self.state
    }
    /// Get a mutable reference to the state amplitudes
    #[inline]
    pub fn amplitudes_mut(&mut self) -> &mut Array1<Complex64> {
        &mut self.state
    }
    /// Calculate the squared norm of the state vector
    ///
    /// Uses efficient array operations (SciRS2 ndarray is already optimized)
    pub fn norm_squared(&self) -> f64 {
        self.state.iter().map(|amp| amp.norm_sqr()).sum()
    }
    /// Normalize the state vector
    ///
    /// Uses SciRS2 ndarray operations (already optimized)
    pub fn normalize(&mut self) -> Result<()> {
        let norm = self.norm_squared().sqrt();
        if norm < 1e-15 {
            return Err(SimulatorError::InvalidState(
                "Cannot normalize zero state".to_string(),
            ));
        }
        let scale = 1.0 / norm;
        self.state.mapv_inplace(|amp| amp * scale);
        Ok(())
    }
    /// Calculate inner product with another state vector
    ///
    /// ⟨self|other⟩ using SciRS2 ndarray operations
    pub fn inner_product(&self, other: &Self) -> Result<Complex64> {
        if self.dim != other.dim {
            return Err(SimulatorError::InvalidOperation(
                "State vectors must have the same dimension".to_string(),
            ));
        }
        let result: Complex64 = self
            .state
            .iter()
            .zip(other.state.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        Ok(result)
    }
    /// Reset state to |0...0⟩
    pub fn reset(&mut self) {
        self.state.fill(Complex64::new(0.0, 0.0));
        self.state[0] = Complex64::new(1.0, 0.0);
    }
    /// Calculate probability of measuring |1⟩ on a specific qubit
    ///
    /// This does not collapse the state
    ///
    /// # Arguments
    ///
    /// * `target` - Target qubit index
    ///
    /// # Returns
    ///
    /// Probability of measuring 1 (0.0 to 1.0)
    pub fn probability_one(&self, target: QubitIndex) -> Result<f64> {
        if target >= self.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target,
                num_qubits: self.num_qubits,
            });
        }
        let mask = 1usize << target;
        let mut prob_one = 0.0;
        for i in 0..self.dim {
            if (i & mask) != 0 {
                prob_one += self.state[i].norm_sqr();
            }
        }
        Ok(prob_one)
    }
    /// Calculate probability of measuring |0⟩ on a specific qubit
    ///
    /// This does not collapse the state
    ///
    /// # Arguments
    ///
    /// * `target` - Target qubit index
    ///
    /// # Returns
    ///
    /// Probability of measuring 0 (0.0 to 1.0)
    pub fn probability_zero(&self, target: QubitIndex) -> Result<f64> {
        Ok(1.0 - self.probability_one(target)?)
    }
    /// Measure a single qubit in the computational basis
    ///
    /// This performs a projective measurement and collapses the state
    ///
    /// # Arguments
    ///
    /// * `target` - Target qubit index
    ///
    /// # Returns
    ///
    /// Measurement outcome (false = 0, true = 1)
    pub fn measure(&mut self, target: QubitIndex) -> Result<bool> {
        if target >= self.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target,
                num_qubits: self.num_qubits,
            });
        }
        let prob_one = self.probability_one(target)?;
        use scirs2_core::random::prelude::*;
        let mut rng = thread_rng();
        let random_value: f64 = rng.random::<f64>();
        let outcome = random_value < prob_one;
        self.collapse_to(target, outcome)?;
        Ok(outcome)
    }
    /// Collapse the state to a specific measurement outcome
    ///
    /// # Arguments
    ///
    /// * `target` - Target qubit index
    /// * `outcome` - Measurement outcome (false = 0, true = 1)
    pub(super) fn collapse_to(&mut self, target: QubitIndex, outcome: bool) -> Result<()> {
        let mask = 1usize << target;
        let mut norm_sqr = 0.0;
        for i in 0..self.dim {
            let qubit_value = (i & mask) != 0;
            if qubit_value != outcome {
                self.state[i] = Complex64::new(0.0, 0.0);
            } else {
                norm_sqr += self.state[i].norm_sqr();
            }
        }
        if norm_sqr < 1e-15 {
            return Err(SimulatorError::InvalidState(
                "Cannot collapse to zero-probability outcome".to_string(),
            ));
        }
        let norm = norm_sqr.sqrt();
        let scale = 1.0 / norm;
        for i in 0..self.dim {
            if ((i & mask) != 0) == outcome {
                self.state[i] *= scale;
            }
        }
        Ok(())
    }
    /// Sample measurement outcomes without collapsing the state
    ///
    /// # Arguments
    ///
    /// * `shots` - Number of measurement samples
    ///
    /// # Returns
    ///
    /// Vector of measurement outcomes (bit strings)
    pub fn sample(&self, shots: usize) -> Result<Vec<Vec<bool>>> {
        if shots == 0 {
            return Ok(Vec::new());
        }
        let mut rng = thread_rng();
        let mut results = Vec::with_capacity(shots);
        let mut cumulative_probs = Vec::with_capacity(self.dim);
        let mut cumsum = 0.0;
        for i in 0..self.dim {
            cumsum += self.state[i].norm_sqr();
            cumulative_probs.push(cumsum);
        }
        for _ in 0..shots {
            let random_value: f64 = rng.random::<f64>();
            let outcome_index = cumulative_probs
                .binary_search_by(|&prob| {
                    if prob < random_value {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    }
                })
                .unwrap_or_else(|x| x);
            let mut bitstring = Vec::with_capacity(self.num_qubits);
            for q in 0..self.num_qubits {
                bitstring.push((outcome_index & (1 << q)) != 0);
            }
            results.push(bitstring);
        }
        Ok(results)
    }
    /// Get measurement counts (histogram) without collapsing the state
    ///
    /// # Arguments
    ///
    /// * `shots` - Number of measurement samples
    ///
    /// # Returns
    ///
    /// HashMap mapping bit strings to counts
    pub fn get_counts(&self, shots: usize) -> Result<std::collections::HashMap<Vec<bool>, usize>> {
        use std::collections::HashMap;
        let samples = self.sample(shots)?;
        let mut counts = HashMap::new();
        for bitstring in samples {
            *counts.entry(bitstring).or_insert(0) += 1;
        }
        Ok(counts)
    }
    /// Sample measurements of specific qubits
    ///
    /// # Arguments
    ///
    /// * `qubits` - Qubit indices to measure
    /// * `shots` - Number of measurement samples
    ///
    /// # Returns
    ///
    /// Vector of partial measurement outcomes
    pub fn sample_qubits(&self, qubits: &[QubitIndex], shots: usize) -> Result<Vec<Vec<bool>>> {
        for &q in qubits {
            if q >= self.num_qubits {
                return Err(SimulatorError::InvalidQubitIndex {
                    index: q,
                    num_qubits: self.num_qubits,
                });
            }
        }
        let full_samples = self.sample(shots)?;
        let results: Vec<Vec<bool>> = full_samples
            .into_iter()
            .map(|bitstring| qubits.iter().map(|&q| bitstring[q]).collect())
            .collect();
        Ok(results)
    }
}
