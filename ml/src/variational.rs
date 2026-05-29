//! Variational Quantum Algorithms: VQE, QAOA, and hybrid solvers.
//!
//! [`VariationalSolver`] executes parameterised ansatz circuits on a
//! state-vector backend and minimises an expectation-value objective using
//! classical optimisers, supporting both VQE (chemistry) and QAOA (combinatorics).

use crate::error::{MLError, Result};
use crate::optimization::Optimizer;
use quantrs2_circuit::prelude::Circuit;
use scirs2_core::ndarray::Array1;
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

/// Algorithm type for variational quantum algorithms
#[derive(Debug, Clone, Copy)]
pub enum VariationalAlgorithm {
    /// Variational Quantum Eigensolver
    VQE,

    /// Quantum Approximate Optimization Algorithm
    QAOA,

    /// Quantum Support Vector Machine
    QSVM,

    /// Quantum Neural Network
    QNN,

    /// Custom variational algorithm
    Custom,
}

/// Ansatz type for variational circuits
#[derive(Debug, Clone, Copy)]
pub enum AnsatzType {
    /// Hardware efficient ansatz
    HardwareEfficient,

    /// Unitary Coupled Cluster Singles and Doubles
    UCCSD,

    /// QAOA ansatz
    QAOA,

    /// Custom ansatz
    Custom,
}

/// Variational quantum circuit with parameterized gates
#[derive(Debug, Clone)]
pub struct VariationalCircuit {
    /// Number of qubits
    pub num_qubits: usize,

    /// Number of parameters
    pub num_params: usize,

    /// Current parameters
    pub parameters: Array1<f64>,

    /// Number of layers
    pub num_layers: usize,

    /// Type of ansatz
    pub ansatz_type: AnsatzType,
}

impl VariationalCircuit {
    /// Creates a new variational circuit
    pub fn new(
        num_qubits: usize,
        num_params: usize,
        num_layers: usize,
        ansatz_type: AnsatzType,
    ) -> Result<Self> {
        // Initialize random parameters
        let parameters = Array1::from_vec(
            (0..num_params)
                .map(|_| thread_rng().random::<f64>() * 2.0 * std::f64::consts::PI)
                .collect(),
        );

        Ok(VariationalCircuit {
            num_qubits,
            num_params,
            parameters,
            num_layers,
            ansatz_type,
        })
    }

    /// Creates a circuit with the current parameters
    pub fn create_circuit<const N: usize>(&self) -> Result<Circuit<N>> {
        // This is a dummy implementation
        // In a real system, this would create a circuit based on the ansatz type and parameters

        let mut circuit = Circuit::<N>::new();

        for i in 0..N.min(self.num_qubits) {
            // Apply some dummy gates based on parameters
            circuit.h(i)?;

            if i < self.parameters.len() {
                circuit.rz(i, self.parameters[i])?;
            }
        }

        // Add entanglement based on the ansatz type
        match self.ansatz_type {
            AnsatzType::HardwareEfficient => {
                // Linear nearest-neighbor entanglement
                for i in 0..N.min(self.num_qubits) - 1 {
                    circuit.cnot(i, i + 1)?;
                }
            }
            AnsatzType::UCCSD => {
                // More complex entanglement pattern
                for i in 0..N.min(self.num_qubits) / 2 {
                    let j = N.min(self.num_qubits) / 2 + i;
                    if j < N {
                        circuit.cnot(i, j)?;
                    }
                }
            }
            AnsatzType::QAOA => {
                // QAOA-style entanglement (fully connected)
                for i in 0..N.min(self.num_qubits) {
                    for j in i + 1..N.min(self.num_qubits) {
                        circuit.cnot(i, j)?;
                    }
                }
            }
            AnsatzType::Custom => {
                // Custom entanglement pattern
                if N >= 3 {
                    circuit.cnot(0, 1)?;
                    circuit.cnot(1, 2)?;
                    if N > 3 {
                        circuit.cnot(2, 3)?;
                    }
                }
            }
        }

        Ok(circuit)
    }

    /// Computes the expectation value of a Hamiltonian via direct statevector simulation.
    ///
    /// `hamiltonian` is a slice of `(coefficient, pauli_term_list)` tuples.
    /// Each `pauli_term_list` contains `(qubit_index, pauli_type)` pairs where
    /// `pauli_type` encodes 0=I, 1=X, 2=Y, 3=Z.
    ///
    /// The ansatz consists of `num_layers` repetitions of:
    ///   – H gate on every qubit
    ///   – RZ(θ_i) on qubit i using the i-th parameter (cycling if parameters run out)
    ///   – CNOT chain (linear nearest-neighbour entanglement)
    ///
    /// Returns `Σ_k  coef_k · ⟨ψ | P_k | ψ⟩`.
    pub fn compute_expectation(&self, hamiltonian: &[(f64, Vec<(usize, usize)>)]) -> Result<f64> {
        let n = self.num_qubits;
        if n == 0 {
            return Ok(0.0);
        }
        let dim = 1usize << n;

        // Build the statevector |ψ⟩ by applying the ansatz to |0⟩^n.
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);

        let mut param_idx = 0usize;

        for _layer in 0..self.num_layers.max(1) {
            // --- Hadamard on every qubit ---
            for q in 0..n {
                apply_single_qubit_h(&mut state, q, n);
            }

            // --- RZ(θ) on every qubit ---
            for q in 0..n {
                let theta = if self.parameters.is_empty() {
                    0.0
                } else {
                    self.parameters[param_idx % self.parameters.len()]
                };
                param_idx += 1;
                apply_single_qubit_rz(&mut state, q, n, theta);
            }

            // --- CNOT chain (linear nearest-neighbour) ---
            for q in 0..n.saturating_sub(1) {
                apply_cnot(&mut state, q, q + 1, n);
            }
        }

        // --- Evaluate ⟨ψ | H | ψ⟩ ---
        let mut expectation = 0.0f64;

        for (coef, pauli_terms) in hamiltonian {
            // Compute ⟨ψ | P | ψ⟩ where P = ⊗ Pauli operators.
            // Action of P on basis state |i⟩: produces coeff * |j⟩.
            // ⟨ψ|P|ψ⟩ = Σ_i  ψ*_i · (P|ψ⟩)_i
            //          = Σ_i  ψ*_i · Σ_j  ⟨i|P|j⟩ ψ_j
            let mut psi_p = vec![Complex64::new(0.0, 0.0); dim];
            for j in 0..dim {
                if state[j].norm_sqr() < 1e-30 {
                    continue;
                }
                let mut coeff = state[j];
                let mut target = j;
                for &(qubit, pauli_type) in pauli_terms {
                    if qubit >= n {
                        continue;
                    }
                    let bit = (j >> qubit) & 1;
                    match pauli_type {
                        0 => {} // I – identity
                        1 => {
                            // X flips qubit
                            target ^= 1 << qubit;
                        }
                        2 => {
                            // Y flips qubit, phase +i if bit=0, -i if bit=1
                            target ^= 1 << qubit;
                            coeff *= if bit == 0 {
                                Complex64::new(0.0, 1.0)
                            } else {
                                Complex64::new(0.0, -1.0)
                            };
                        }
                        3 => {
                            // Z: phase -1 if bit=1
                            if bit == 1 {
                                coeff *= Complex64::new(-1.0, 0.0);
                            }
                        }
                        _ => {} // unknown – treat as identity
                    }
                }
                if target < dim {
                    psi_p[target] += coeff;
                }
            }
            // ⟨ψ|P|ψ⟩ = Σ_i  ψ*_i · (P|ψ⟩)_i
            let pauli_exp: Complex64 = state
                .iter()
                .zip(psi_p.iter())
                .map(|(a, b)| a.conj() * b)
                .sum();
            expectation += coef * pauli_exp.re;
        }

        Ok(expectation)
    }

    /// Evaluates the objective function for optimization
    pub fn evaluate(&self, objective: &dyn Fn(&VariationalCircuit) -> Result<f64>) -> Result<f64> {
        objective(self)
    }

    /// Optimizes the circuit parameters using the parameter-shift rule for gradient computation.
    ///
    /// For each parameter θ_i the gradient is estimated as:
    ///   ∂f/∂θ_i ≈ (f(θ_i + π/2) − f(θ_i − π/2)) / 2
    ///
    /// A gradient-descent update is then applied with the learning rate from the
    /// supplied [`Optimizer`].  Other optimizer variants fall back to `lr = 0.01`.
    pub fn optimize(
        &mut self,
        objective: &dyn Fn(&VariationalCircuit) -> Result<f64>,
        optimizer: &Optimizer,
        max_iterations: usize,
    ) -> Result<f64> {
        let shift = std::f64::consts::PI / 2.0;
        let lr = match optimizer {
            Optimizer::GradientDescent { learning_rate } => *learning_rate,
            Optimizer::Adam { learning_rate, .. } => *learning_rate,
            Optimizer::SPSA { learning_rate, .. } => *learning_rate,
            Optimizer::QuantumNaturalGradient { learning_rate, .. } => *learning_rate,
            Optimizer::SciRS2 { config, .. } => {
                config.get("learning_rate").copied().unwrap_or(0.01)
            }
        };

        let mut best_value = self.evaluate(objective)?;

        for _ in 0..max_iterations {
            let n_params = self.parameters.len();
            let mut gradient = vec![0.0f64; n_params];

            for i in 0..n_params {
                let mut plus = self.clone();
                plus.parameters[i] += shift;
                let mut minus = self.clone();
                minus.parameters[i] -= shift;
                gradient[i] = (plus.evaluate(objective)? - minus.evaluate(objective)?) * 0.5;
            }

            for i in 0..n_params {
                self.parameters[i] -= lr * gradient[i];
            }

            let new_value = self.evaluate(objective)?;
            if new_value < best_value {
                best_value = new_value;
            }
        }

        Ok(best_value)
    }
}

// ---------------------------------------------------------------------------
// Statevector helper functions (runtime-dimension, no const generic N)
// ---------------------------------------------------------------------------

/// Apply a single-qubit Hadamard gate to qubit `q` in a `n`-qubit statevector.
fn apply_single_qubit_h(state: &mut Vec<Complex64>, q: usize, n: usize) {
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    let dim = 1 << n;
    let half_step = 1 << q;
    let step = half_step << 1;
    let mut base = 0;
    while base < dim {
        for i in base..base + half_step {
            let a = state[i];
            let b = state[i + half_step];
            state[i] = Complex64::new((a.re + b.re) * inv_sqrt2, (a.im + b.im) * inv_sqrt2);
            state[i + half_step] =
                Complex64::new((a.re - b.re) * inv_sqrt2, (a.im - b.im) * inv_sqrt2);
        }
        base += step;
    }
}

/// Apply RZ(θ) on qubit `q`: |0⟩ → e^{-iθ/2}|0⟩, |1⟩ → e^{+iθ/2}|1⟩.
fn apply_single_qubit_rz(state: &mut Vec<Complex64>, q: usize, n: usize, theta: f64) {
    let dim = 1 << n;
    let phase0 = Complex64::from_polar(1.0, -theta * 0.5);
    let phase1 = Complex64::from_polar(1.0, theta * 0.5);
    for i in 0..dim {
        let bit = (i >> q) & 1;
        state[i] *= if bit == 0 { phase0 } else { phase1 };
    }
}

/// Apply CNOT with `control` and `target` qubits.
fn apply_cnot(state: &mut Vec<Complex64>, control: usize, target: usize, n: usize) {
    let dim = 1 << n;
    for i in 0..dim {
        if (i >> control) & 1 == 1 {
            let j = i ^ (1 << target);
            if i < j {
                state.swap(i, j);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_reduces_objective() {
        // Simple convex objective: minimize Σ θ_i²
        // Parameter-shift gradient: ∂(θ_i²)/∂θ_i = 2θ_i
        // But parameter-shift gives (f(θ+π/2) - f(θ-π/2))/2 = (sin(θ+π/2)^2 - sin(θ-π/2)^2)/2
        // For the pure quadratic we just need descent, not exact gradients.
        let num_qubits = 2;
        let num_params = 4;
        let num_layers = 1;
        let mut vc = VariationalCircuit::new(
            num_qubits,
            num_params,
            num_layers,
            AnsatzType::HardwareEfficient,
        )
        .expect("circuit creation failed");
        // Override parameters to known starting values so the test is deterministic.
        vc.parameters = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);

        let objective = |vc: &VariationalCircuit| -> Result<f64> {
            Ok(vc.parameters.iter().map(|p| p * p).sum())
        };

        let initial = objective(&vc).expect("initial evaluation failed");
        let optimizer = Optimizer::GradientDescent { learning_rate: 0.1 };
        let result = vc
            .optimize(&objective, &optimizer, 20)
            .expect("optimization failed");

        assert!(
            result < initial,
            "optimize should reduce objective: {result} vs initial {initial}"
        );
    }

    #[test]
    fn test_compute_expectation_identity_hamiltonian() {
        // H = 1.0 * I (identity on qubit 0) — ⟨ψ|I|ψ⟩ = 1 always
        let vc = VariationalCircuit::new(2, 2, 1, AnsatzType::HardwareEfficient)
            .expect("circuit creation failed");
        let hamiltonian = vec![(1.0, vec![(0_usize, 0_usize)])]; // coef=1, I on qubit 0
        let exp = vc
            .compute_expectation(&hamiltonian)
            .expect("expectation failed");
        assert!((exp - 1.0).abs() < 1e-10, "⟨I⟩ should be 1.0, got {exp}");
    }
}
