//! Non-unitary quantum operations and measurements
//!
//! This module provides support for non-unitary quantum operations including:
//! - Projective measurements
//! - POVM measurements
//! - Quantum channels
//! - Reset operations

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::qubit::QubitId;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;
use rand::prelude::*;
use rand::rng;
use std::fmt::Debug;

/// Trait for quantum operations (both unitary and non-unitary)
pub trait QuantumOperation: Debug + Send + Sync {
    /// Apply the operation to a state vector
    fn apply_to_state(&self, state: &ArrayView1<Complex64>) -> QuantRS2Result<OperationResult>;

    /// Apply the operation to a density matrix
    fn apply_to_density_matrix(
        &self,
        rho: &ArrayView2<Complex64>,
    ) -> QuantRS2Result<Array2<Complex64>>;

    /// Get the qubits this operation acts on
    fn qubits(&self) -> Vec<QubitId>;

    /// Check if the operation is deterministic
    fn is_deterministic(&self) -> bool;
}

/// Result of a quantum operation
#[derive(Debug, Clone)]
pub enum OperationResult {
    /// Deterministic result (new state)
    Deterministic(Array1<Complex64>),
    /// Probabilistic result (outcome, probability, new state)
    Probabilistic {
        outcome: usize,
        probability: f64,
        state: Array1<Complex64>,
    },
    /// Multiple possible outcomes
    MultiOutcome(Vec<MeasurementOutcome>),
}

/// A single measurement outcome
#[derive(Debug, Clone)]
pub struct MeasurementOutcome {
    /// The measurement result
    pub outcome: usize,
    /// The probability of this outcome
    pub probability: f64,
    /// The post-measurement state
    pub state: Array1<Complex64>,
}

/// Projective measurement in the computational basis
#[derive(Debug, Clone)]
pub struct ProjectiveMeasurement {
    /// Qubits to measure
    pub qubits: Vec<QubitId>,
    /// Optional specific outcome to project onto
    pub outcome: Option<usize>,
}

impl ProjectiveMeasurement {
    /// Create a new projective measurement
    pub fn new(qubits: Vec<QubitId>) -> Self {
        Self {
            qubits,
            outcome: None,
        }
    }

    /// Create a measurement that projects onto a specific outcome
    pub fn with_outcome(qubits: Vec<QubitId>, outcome: usize) -> Self {
        Self {
            qubits,
            outcome: Some(outcome),
        }
    }

    /// Calculate measurement probabilities
    pub fn get_probabilities(&self, state: &ArrayView1<Complex64>) -> QuantRS2Result<Vec<f64>> {
        let n_qubits = (state.len() as f64).log2() as usize;
        let n_outcomes = 1 << self.qubits.len();
        let mut probabilities = vec![0.0; n_outcomes];

        // Calculate probability for each outcome
        for (idx, amp) in state.iter().enumerate() {
            let outcome = self.extract_outcome_from_index(idx, n_qubits);
            probabilities[outcome] += amp.norm_sqr();
        }

        Ok(probabilities)
    }

    /// Extract the measurement outcome from a basis state index
    fn extract_outcome_from_index(&self, index: usize, total_qubits: usize) -> usize {
        let mut outcome = 0;
        for (i, &qubit) in self.qubits.iter().enumerate() {
            let bit = (index >> (total_qubits - 1 - qubit.0 as usize)) & 1;
            outcome |= bit << (self.qubits.len() - 1 - i);
        }
        outcome
    }

    /// Project state onto a specific outcome
    fn project_onto_outcome(
        &self,
        state: &ArrayView1<Complex64>,
        outcome: usize,
    ) -> QuantRS2Result<(f64, Array1<Complex64>)> {
        let n_qubits = (state.len() as f64).log2() as usize;
        let mut projected = Array1::zeros(state.len());
        let mut norm_squared = 0.0;

        for (idx, &amp) in state.iter().enumerate() {
            if self.extract_outcome_from_index(idx, n_qubits) == outcome {
                projected[idx] = amp;
                norm_squared += amp.norm_sqr();
            }
        }

        if norm_squared < 1e-10 {
            return Err(QuantRS2Error::InvalidInput(
                "Measurement outcome has zero probability".to_string(),
            ));
        }

        // Normalize the projected state
        let norm = norm_squared.sqrt();
        projected = projected / norm;

        Ok((norm_squared, projected))
    }
}

impl QuantumOperation for ProjectiveMeasurement {
    fn apply_to_state(&self, state: &ArrayView1<Complex64>) -> QuantRS2Result<OperationResult> {
        match self.outcome {
            Some(outcome) => {
                // Project onto specific outcome
                let (prob, new_state) = self.project_onto_outcome(state, outcome)?;
                Ok(OperationResult::Probabilistic {
                    outcome,
                    probability: prob,
                    state: new_state,
                })
            }
            None => {
                // Calculate all outcomes
                let probabilities = self.get_probabilities(state)?;
                let mut outcomes = Vec::new();

                for (outcome, &prob) in probabilities.iter().enumerate() {
                    if prob > 1e-10 {
                        let (_, new_state) = self.project_onto_outcome(state, outcome)?;
                        outcomes.push(MeasurementOutcome {
                            outcome,
                            probability: prob,
                            state: new_state,
                        });
                    }
                }

                Ok(OperationResult::MultiOutcome(outcomes))
            }
        }
    }

    fn apply_to_density_matrix(
        &self,
        rho: &ArrayView2<Complex64>,
    ) -> QuantRS2Result<Array2<Complex64>> {
        let n_qubits = (rho.nrows() as f64).log2() as usize;
        let mut result = Array2::zeros(rho.raw_dim());

        // Apply measurement operators: ρ' = Σ_i P_i ρ P_i
        for outcome in 0..(1 << self.qubits.len()) {
            let projector = self.create_projector(outcome, n_qubits)?;
            let proj_rho = projector.dot(rho).dot(&projector);
            result = result + proj_rho;
        }

        Ok(result)
    }

    fn qubits(&self) -> Vec<QubitId> {
        self.qubits.clone()
    }

    fn is_deterministic(&self) -> bool {
        false
    }
}

impl ProjectiveMeasurement {
    /// Create projector matrix for a specific outcome
    fn create_projector(
        &self,
        outcome: usize,
        total_qubits: usize,
    ) -> QuantRS2Result<Array2<Complex64>> {
        let dim = 1 << total_qubits;
        let mut projector = Array2::zeros((dim, dim));

        for idx in 0..dim {
            if self.extract_outcome_from_index(idx, total_qubits) == outcome {
                projector[[idx, idx]] = Complex64::new(1.0, 0.0);
            }
        }

        Ok(projector)
    }
}

/// POVM (Positive Operator-Valued Measure) measurement
#[derive(Debug, Clone)]
pub struct POVMMeasurement {
    /// The POVM elements (must sum to identity)
    pub elements: Vec<Array2<Complex64>>,
    /// Qubits this POVM acts on
    pub qubits: Vec<QubitId>,
}

impl POVMMeasurement {
    /// Create a new POVM measurement
    pub fn new(elements: Vec<Array2<Complex64>>, qubits: Vec<QubitId>) -> QuantRS2Result<Self> {
        // Verify that elements sum to identity
        let dim = elements[0].nrows();
        let mut sum = Array2::<Complex64>::zeros((dim, dim));

        for element in &elements {
            if element.nrows() != dim || element.ncols() != dim {
                return Err(QuantRS2Error::InvalidInput(
                    "All POVM elements must have the same dimension".to_string(),
                ));
            }
            sum = sum + element;
        }

        // Check if sum is identity (within tolerance)
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff: Complex64 = sum[[i, j]] - expected;
                if diff.norm_sqr().sqrt() > 1e-10 {
                    return Err(QuantRS2Error::InvalidInput(
                        "POVM elements do not sum to identity".to_string(),
                    ));
                }
            }
        }

        Ok(Self { elements, qubits })
    }

    /// Get measurement probabilities
    pub fn get_probabilities(&self, state: &ArrayView1<Complex64>) -> QuantRS2Result<Vec<f64>> {
        let mut probabilities = Vec::new();

        for element in &self.elements {
            // Probability = <ψ|M_i|ψ>
            let temp = element.dot(state);
            let prob = state
                .iter()
                .zip(temp.iter())
                .map(|(&psi, &m_psi)| psi.conj() * m_psi)
                .sum::<Complex64>()
                .re;
            probabilities.push(prob);
        }

        Ok(probabilities)
    }
}

impl QuantumOperation for POVMMeasurement {
    fn apply_to_state(&self, state: &ArrayView1<Complex64>) -> QuantRS2Result<OperationResult> {
        let probabilities = self.get_probabilities(state)?;
        let mut outcomes = Vec::new();

        for (i, (&prob, element)) in probabilities.iter().zip(&self.elements).enumerate() {
            if prob > 1e-10 {
                // Apply measurement operator: |ψ'> = M_i|ψ>/√p_i
                let new_state = element.dot(state) / prob.sqrt();
                outcomes.push(MeasurementOutcome {
                    outcome: i,
                    probability: prob,
                    state: new_state,
                });
            }
        }

        Ok(OperationResult::MultiOutcome(outcomes))
    }

    fn apply_to_density_matrix(
        &self,
        rho: &ArrayView2<Complex64>,
    ) -> QuantRS2Result<Array2<Complex64>> {
        let mut result = Array2::zeros(rho.raw_dim());

        // Apply POVM: ρ' = Σ_i M_i ρ M_i†
        for element in &self.elements {
            let m_dag = element.t().mapv(|x| x.conj());
            let term = element.dot(rho).dot(&m_dag);
            result = result + term;
        }

        Ok(result)
    }

    fn qubits(&self) -> Vec<QubitId> {
        self.qubits.clone()
    }

    fn is_deterministic(&self) -> bool {
        false
    }
}

/// Reset operation that sets qubits to |0⟩
#[derive(Debug, Clone)]
pub struct Reset {
    /// Qubits to reset
    pub qubits: Vec<QubitId>,
}

impl Reset {
    /// Create a new reset operation
    pub fn new(qubits: Vec<QubitId>) -> Self {
        Self { qubits }
    }
}

impl QuantumOperation for Reset {
    fn apply_to_state(&self, state: &ArrayView1<Complex64>) -> QuantRS2Result<OperationResult> {
        let n_qubits = (state.len() as f64).log2() as usize;
        let mut new_state = Array1::zeros(state.len());

        // Project onto |0⟩ for specified qubits and normalize
        let mut norm_squared = 0.0;
        for (idx, &amp) in state.iter().enumerate() {
            let mut should_keep = true;
            for &qubit in &self.qubits {
                let bit = (idx >> (n_qubits - 1 - qubit.0 as usize)) & 1;
                if bit != 0 {
                    should_keep = false;
                    break;
                }
            }
            if should_keep {
                new_state[idx] = amp;
                norm_squared += amp.norm_sqr();
            }
        }

        if norm_squared < 1e-10 {
            // If projection gives zero, just set to |0...0⟩
            new_state[0] = Complex64::new(1.0, 0.0);
        } else {
            new_state = new_state / norm_squared.sqrt();
        }

        Ok(OperationResult::Deterministic(new_state))
    }

    fn apply_to_density_matrix(
        &self,
        rho: &ArrayView2<Complex64>,
    ) -> QuantRS2Result<Array2<Complex64>> {
        // For reset, we trace out the qubits and replace with |0⟩⟨0|
        // This is a simplified implementation
        let n_qubits = (rho.nrows() as f64).log2() as usize;
        let mut result = Array2::zeros(rho.raw_dim());

        // Project onto |0⟩ for reset qubits
        for i in 0..rho.nrows() {
            for j in 0..rho.ncols() {
                let mut should_keep = true;
                for &qubit in &self.qubits {
                    let bit_i = (i >> (n_qubits - 1 - qubit.0 as usize)) & 1;
                    let bit_j = (j >> (n_qubits - 1 - qubit.0 as usize)) & 1;
                    if bit_i != 0 || bit_j != 0 {
                        should_keep = false;
                        break;
                    }
                }
                if should_keep {
                    result[[i, j]] = rho[[i, j]];
                }
            }
        }

        // Renormalize
        let trace = (0..result.nrows()).map(|i| result[[i, i]].re).sum::<f64>();
        if trace > 1e-10 {
            result = result / trace;
        }

        Ok(result)
    }

    fn qubits(&self) -> Vec<QubitId> {
        self.qubits.clone()
    }

    fn is_deterministic(&self) -> bool {
        true
    }
}

/// Sample from measurement outcomes according to their probabilities
pub fn sample_outcome(probabilities: &[f64]) -> QuantRS2Result<usize> {
    let mut rng = rng();
    let r: f64 = rng.random();

    let mut cumsum = 0.0;
    for (i, &prob) in probabilities.iter().enumerate() {
        cumsum += prob;
        if r < cumsum {
            return Ok(i);
        }
    }

    // Should not reach here if probabilities sum to 1
    Err(QuantRS2Error::ComputationError(
        "Probabilities do not sum to 1".to_string(),
    ))
}

/// Apply a quantum operation and sample an outcome
pub fn apply_and_sample<O: QuantumOperation>(
    operation: &O,
    state: &ArrayView1<Complex64>,
) -> QuantRS2Result<(usize, Array1<Complex64>)> {
    match operation.apply_to_state(state)? {
        OperationResult::Deterministic(new_state) => Ok((0, new_state)),
        OperationResult::Probabilistic { outcome, state, .. } => Ok((outcome, state)),
        OperationResult::MultiOutcome(outcomes) => {
            let probabilities: Vec<f64> = outcomes.iter().map(|o| o.probability).collect();
            let sampled_idx = sample_outcome(&probabilities)?;
            let outcome = &outcomes[sampled_idx];
            Ok((outcome.outcome, outcome.state.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projective_measurement() {
        // |+⟩ state
        let state = Array1::from_vec(vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ]);

        let measurement = ProjectiveMeasurement::new(vec![QubitId(0)]);
        let probs = measurement.get_probabilities(&state.view()).unwrap();

        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_reset_operation() {
        // |1⟩ state
        let state = Array1::from_vec(vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]);

        let reset = Reset::new(vec![QubitId(0)]);
        let result = reset.apply_to_state(&state.view()).unwrap();

        match result {
            OperationResult::Deterministic(new_state) => {
                assert!((new_state[0].norm() - 1.0).abs() < 1e-10);
                assert!(new_state[1].norm() < 1e-10);
            }
            _ => panic!("Reset should be deterministic"),
        }
    }
}
