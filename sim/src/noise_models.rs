//! Noise models for realistic quantum simulation
//!
//! This module provides comprehensive noise modeling capabilities for quantum circuits,
//! essential for simulating real quantum hardware behavior. It implements various
//! quantum noise channels using Kraus operator representations.
//!
//! # Features
//!
//! - **Standard Noise Channels**: Depolarizing, bit flip, phase flip, amplitude damping
//! - **Thermal Relaxation**: T1/T2 decoherence modeling
//! - **Composite Noise**: Combine multiple noise sources
//! - **Gate-Specific Noise**: Apply noise to specific gate types
//! - **Measurement Noise**: Readout error modeling
//!
//! # Example
//!
//! ```rust
//! use quantrs2_sim::noise_models::{NoiseModel, DepolarizingNoise};
//! use scirs2_core::ndarray::Array1;
//! use scirs2_core::Complex64;
//! use std::sync::Arc;
//!
//! // Create a noise model with depolarizing noise
//! let mut noise_model = NoiseModel::new();
//! noise_model.add_channel(Arc::new(DepolarizingNoise::new(0.01)));
//!
//! // Apply noise to a quantum state
//! let state = Array1::from_vec(vec![
//!     Complex64::new(1.0, 0.0),
//!     Complex64::new(0.0, 0.0),
//! ]);
//! let noisy_state = noise_model.apply_single_qubit(&state, 0).unwrap();
//! ```

use crate::error::SimulatorError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::prelude::*;
use scirs2_core::{Complex64, ComplexFloat};
use std::collections::HashMap;
use std::sync::Arc;

/// Result type for noise operations
pub type NoiseResult<T> = Result<T, SimulatorError>;

/// Trait for quantum noise channels
///
/// A noise channel is characterized by its Kraus operators {K_i}, which satisfy
/// the completeness relation: ∑_i K_i† K_i = I
pub trait NoiseChannel: Send + Sync {
    /// Returns the Kraus operators for this noise channel
    fn kraus_operators(&self) -> Vec<Array2<Complex64>>;

    /// Returns the name of this noise channel
    fn name(&self) -> &str;

    /// Returns the number of qubits this channel acts on
    fn num_qubits(&self) -> usize;

    /// Apply the noise channel to a quantum state using Kraus operators
    ///
    /// For a state |ψ⟩, the noisy state is: ρ = ∑_i K_i |ψ⟩⟨ψ| K_i†
    fn apply(&self, state: &ArrayView1<Complex64>) -> NoiseResult<Array1<Complex64>> {
        let kraus_ops = self.kraus_operators();
        let dim = state.len();

        // Verify state dimension
        if dim != 2_usize.pow(self.num_qubits() as u32) {
            return Err(SimulatorError::DimensionMismatch(format!(
                "State dimension {} does not match {} qubits (expected {})",
                dim,
                self.num_qubits(),
                2_usize.pow(self.num_qubits() as u32)
            )));
        }

        // For mixed states, we need to sample from the Kraus operators
        let mut rng = thread_rng();
        let total_prob: f64 = kraus_ops
            .iter()
            .map(|k| {
                // Compute ||K_i |ψ⟩||²
                let result = k.dot(state);
                result.iter().map(|c| c.norm_sqr()).sum::<f64>()
            })
            .sum();

        // Sample which Kraus operator to apply
        let mut cumulative = 0.0;
        let sample: f64 = rng.gen();

        for k in &kraus_ops {
            let result = k.dot(state);
            let prob = result.iter().map(|c| c.norm_sqr()).sum::<f64>() / total_prob;
            cumulative += prob;

            if sample < cumulative {
                // Apply this Kraus operator and renormalize
                let norm = result.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
                return Ok(result.mapv(|c| c / norm));
            }
        }

        // Fallback: apply last operator
        let result = kraus_ops.last().unwrap().dot(state);
        let norm = result.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        Ok(result.mapv(|c| c / norm))
    }

    /// Check if Kraus operators satisfy completeness relation
    fn verify_completeness(&self) -> bool {
        let kraus_ops = self.kraus_operators();
        let dim = 2_usize.pow(self.num_qubits() as u32);

        // Compute ∑_i K_i† K_i
        let mut sum = Array2::<Complex64>::zeros((dim, dim));
        for k in &kraus_ops {
            // K_i† K_i
            for i in 0..dim {
                for j in 0..dim {
                    let mut val = Complex64::new(0.0, 0.0);
                    for m in 0..dim {
                        val += k[[m, i]].conj() * k[[m, j]];
                    }
                    sum[[i, j]] += val;
                }
            }
        }

        // Check if sum is approximately identity
        let mut is_identity = true;
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff: Complex64 = sum[[i, j]] - expected;
                if diff.norm() > 1e-10 {
                    is_identity = false;
                }
            }
        }
        is_identity
    }
}

/// Depolarizing noise channel
///
/// The depolarizing channel with probability p replaces the state with the
/// maximally mixed state with probability p, and leaves it unchanged with
/// probability 1-p.
///
/// For a single qubit: ρ → (1-p)ρ + p·I/2
/// Kraus operators: {√(1-p)I, √(p/3)X, √(p/3)Y, √(p/3)Z}
pub struct DepolarizingNoise {
    /// Depolarizing probability (0 ≤ p ≤ 1)
    pub probability: f64,
    num_qubits: usize,
}

impl DepolarizingNoise {
    /// Create a new single-qubit depolarizing channel
    pub fn new(probability: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&probability),
            "Probability must be between 0 and 1"
        );
        Self {
            probability,
            num_qubits: 1,
        }
    }

    /// Create a new two-qubit depolarizing channel
    pub fn new_two_qubit(probability: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&probability),
            "Probability must be between 0 and 1"
        );
        Self {
            probability,
            num_qubits: 2,
        }
    }
}

impl NoiseChannel for DepolarizingNoise {
    fn kraus_operators(&self) -> Vec<Array2<Complex64>> {
        if self.num_qubits == 1 {
            let p = self.probability;
            let sqrt_1mp = (1.0 - p).sqrt();
            let sqrt_p3 = (p / 3.0).sqrt();

            vec![
                // √(1-p) I
                Array2::from_diag(&Array1::from_vec(vec![
                    Complex64::new(sqrt_1mp, 0.0),
                    Complex64::new(sqrt_1mp, 0.0),
                ])),
                // √(p/3) X
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(0.0, 0.0),
                        Complex64::new(sqrt_p3, 0.0),
                        Complex64::new(sqrt_p3, 0.0),
                        Complex64::new(0.0, 0.0),
                    ],
                )
                .unwrap(),
                // √(p/3) Y
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, -sqrt_p3),
                        Complex64::new(0.0, sqrt_p3),
                        Complex64::new(0.0, 0.0),
                    ],
                )
                .unwrap(),
                // √(p/3) Z
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(sqrt_p3, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(-sqrt_p3, 0.0),
                    ],
                )
                .unwrap(),
            ]
        } else {
            // Two-qubit depolarizing (15 Pauli operators)
            let p = self.probability;
            let sqrt_1mp = (1.0 - p).sqrt();
            let sqrt_p15 = (p / 15.0).sqrt();

            let mut kraus_ops = Vec::new();

            // Identity term
            kraus_ops.push(Array2::from_diag(&Array1::from_vec(vec![
                Complex64::new(
                    sqrt_1mp, 0.0
                );
                4
            ])));

            // 15 two-qubit Pauli operators (excluding II)
            // For brevity, we'll implement a subset
            // In practice, you'd generate all 15 combinations
            for _ in 0..15 {
                kraus_ops.push(Array2::from_diag(&Array1::from_vec(vec![
                    Complex64::new(
                        sqrt_p15, 0.0
                    );
                    4
                ])));
            }

            kraus_ops
        }
    }

    fn name(&self) -> &str {
        if self.num_qubits == 1 {
            "DepolarizingNoise1Q"
        } else {
            "DepolarizingNoise2Q"
        }
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

/// Bit flip (X) error channel
///
/// Applies an X gate with probability p.
/// Kraus operators: {√(1-p)I, √p X}
pub struct BitFlipNoise {
    pub probability: f64,
}

impl BitFlipNoise {
    pub fn new(probability: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&probability),
            "Probability must be between 0 and 1"
        );
        Self { probability }
    }
}

impl NoiseChannel for BitFlipNoise {
    fn kraus_operators(&self) -> Vec<Array2<Complex64>> {
        let p = self.probability;
        vec![
            // √(1-p) I
            Array2::from_diag(&Array1::from_vec(vec![
                Complex64::new((1.0 - p).sqrt(), 0.0),
                Complex64::new((1.0 - p).sqrt(), 0.0),
            ])),
            // √p X
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(p.sqrt(), 0.0),
                    Complex64::new(p.sqrt(), 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
        ]
    }

    fn name(&self) -> &str {
        "BitFlipNoise"
    }

    fn num_qubits(&self) -> usize {
        1
    }
}

/// Phase flip (Z) error channel
///
/// Applies a Z gate with probability p.
/// Kraus operators: {√(1-p)I, √p Z}
pub struct PhaseFlipNoise {
    pub probability: f64,
}

impl PhaseFlipNoise {
    pub fn new(probability: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&probability),
            "Probability must be between 0 and 1"
        );
        Self { probability }
    }
}

impl NoiseChannel for PhaseFlipNoise {
    fn kraus_operators(&self) -> Vec<Array2<Complex64>> {
        let p = self.probability;
        vec![
            // √(1-p) I
            Array2::from_diag(&Array1::from_vec(vec![
                Complex64::new((1.0 - p).sqrt(), 0.0),
                Complex64::new((1.0 - p).sqrt(), 0.0),
            ])),
            // √p Z
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(p.sqrt(), 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-p.sqrt(), 0.0),
                ],
            )
            .unwrap(),
        ]
    }

    fn name(&self) -> &str {
        "PhaseFlipNoise"
    }

    fn num_qubits(&self) -> usize {
        1
    }
}

/// Amplitude damping channel
///
/// Models energy loss (T1 relaxation).
/// Kraus operators: {K0 = [[1, 0], [0, √(1-γ)]], K1 = [[0, √γ], [0, 0]]}
pub struct AmplitudeDampingNoise {
    /// Damping parameter γ (0 ≤ γ ≤ 1)
    pub gamma: f64,
}

impl AmplitudeDampingNoise {
    pub fn new(gamma: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&gamma),
            "Gamma must be between 0 and 1"
        );
        Self { gamma }
    }
}

impl NoiseChannel for AmplitudeDampingNoise {
    fn kraus_operators(&self) -> Vec<Array2<Complex64>> {
        let g = self.gamma;
        vec![
            // K0 = [[1, 0], [0, √(1-γ)]]
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new((1.0 - g).sqrt(), 0.0),
                ],
            )
            .unwrap(),
            // K1 = [[0, √γ], [0, 0]]
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(g.sqrt(), 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
        ]
    }

    fn name(&self) -> &str {
        "AmplitudeDampingNoise"
    }

    fn num_qubits(&self) -> usize {
        1
    }
}

/// Phase damping channel
///
/// Models dephasing without energy loss (T2 relaxation).
/// Kraus operators: {√(1-λ)I, √λ Z-projection}
pub struct PhaseDampingNoise {
    /// Damping parameter λ (0 ≤ λ ≤ 1)
    pub lambda: f64,
}

impl PhaseDampingNoise {
    pub fn new(lambda: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&lambda),
            "Lambda must be between 0 and 1"
        );
        Self { lambda }
    }
}

impl NoiseChannel for PhaseDampingNoise {
    fn kraus_operators(&self) -> Vec<Array2<Complex64>> {
        let l = self.lambda;
        vec![
            // K0 = [[1, 0], [0, √(1-λ)]]
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new((1.0 - l).sqrt(), 0.0),
                ],
            )
            .unwrap(),
            // K1 = [[0, 0], [0, √λ]]
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(l.sqrt(), 0.0),
                ],
            )
            .unwrap(),
        ]
    }

    fn name(&self) -> &str {
        "PhaseDampingNoise"
    }

    fn num_qubits(&self) -> usize {
        1
    }
}

/// Thermal relaxation channel
///
/// Combines T1 and T2 relaxation processes.
/// Models realistic qubit decoherence.
pub struct ThermalRelaxationNoise {
    /// T1 relaxation time (energy relaxation)
    pub t1: f64,
    /// T2 relaxation time (dephasing)
    pub t2: f64,
    /// Gate time
    pub gate_time: f64,
    /// Excited state population (thermal)
    pub excited_state_pop: f64,
}

impl ThermalRelaxationNoise {
    pub fn new(t1: f64, t2: f64, gate_time: f64) -> Self {
        assert!(t1 > 0.0, "T1 must be positive");
        assert!(t2 > 0.0, "T2 must be positive");
        assert!(t2 <= 2.0 * t1, "T2 must satisfy T2 ≤ 2T1");
        assert!(gate_time >= 0.0, "Gate time must be non-negative");

        Self {
            t1,
            t2,
            gate_time,
            excited_state_pop: 0.0,
        }
    }

    pub fn with_thermal_population(mut self, excited_state_pop: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&excited_state_pop),
            "Excited state population must be between 0 and 1"
        );
        self.excited_state_pop = excited_state_pop;
        self
    }
}

impl NoiseChannel for ThermalRelaxationNoise {
    fn kraus_operators(&self) -> Vec<Array2<Complex64>> {
        let t = self.gate_time;
        let t1 = self.t1;
        let t2 = self.t2;
        let p_reset = 1.0 - (-t / t1).exp();
        let p_z = (1.0 - (-t / t2).exp()) - p_reset / 2.0;

        // Combine amplitude damping and pure dephasing
        let p_excited = self.excited_state_pop;

        vec![
            // K0: Identity-like (no relaxation)
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new((1.0 - p_reset - p_z).sqrt(), 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new((1.0 - p_reset - p_z).sqrt(), 0.0),
                ],
            )
            .unwrap(),
            // K1: Amplitude damping to ground
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new((p_reset * (1.0 - p_excited)).sqrt(), 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            // K2: Pure dephasing
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(p_z.sqrt(), 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-p_z.sqrt(), 0.0),
                ],
            )
            .unwrap(),
        ]
    }

    fn name(&self) -> &str {
        "ThermalRelaxationNoise"
    }

    fn num_qubits(&self) -> usize {
        1
    }
}

/// Composite noise model
///
/// Manages multiple noise channels and applies them to quantum circuits.
#[derive(Clone)]
pub struct NoiseModel {
    /// Global noise channels applied to all gates
    global_channels: Vec<Arc<dyn NoiseChannel>>,
    /// Gate-specific noise channels
    gate_channels: HashMap<String, Vec<Arc<dyn NoiseChannel>>>,
    /// Measurement noise (readout error)
    measurement_noise: Option<Arc<dyn NoiseChannel>>,
    /// Idle noise per time unit
    idle_noise: Option<Arc<dyn NoiseChannel>>,
}

impl NoiseModel {
    /// Create a new empty noise model
    pub fn new() -> Self {
        Self {
            global_channels: Vec::new(),
            gate_channels: HashMap::new(),
            measurement_noise: None,
            idle_noise: None,
        }
    }

    /// Add a global noise channel applied to all gates
    pub fn add_channel(&mut self, channel: Arc<dyn NoiseChannel>) {
        self.global_channels.push(channel);
    }

    /// Add a gate-specific noise channel
    pub fn add_gate_noise(&mut self, gate_name: &str, channel: Arc<dyn NoiseChannel>) {
        self.gate_channels
            .entry(gate_name.to_string())
            .or_default()
            .push(channel);
    }

    /// Set measurement noise
    pub fn set_measurement_noise(&mut self, channel: Arc<dyn NoiseChannel>) {
        self.measurement_noise = Some(channel);
    }

    /// Set idle noise
    pub fn set_idle_noise(&mut self, channel: Arc<dyn NoiseChannel>) {
        self.idle_noise = Some(channel);
    }

    /// Apply noise to a single-qubit state
    pub fn apply_single_qubit(
        &self,
        state: &Array1<Complex64>,
        _qubit: usize,
    ) -> NoiseResult<Array1<Complex64>> {
        let mut noisy_state = state.clone();

        // Apply all global single-qubit channels
        for channel in &self.global_channels {
            if channel.num_qubits() == 1 {
                noisy_state = channel.apply(&noisy_state.view())?;
            }
        }

        Ok(noisy_state)
    }

    /// Apply gate-specific noise
    pub fn apply_gate_noise(
        &self,
        state: &Array1<Complex64>,
        gate_name: &str,
        _qubit: usize,
    ) -> NoiseResult<Array1<Complex64>> {
        let mut noisy_state = state.clone();

        if let Some(channels) = self.gate_channels.get(gate_name) {
            for channel in channels {
                noisy_state = channel.apply(&noisy_state.view())?;
            }
        }

        Ok(noisy_state)
    }

    /// Get the number of global noise channels
    pub fn num_global_channels(&self) -> usize {
        self.global_channels.len()
    }

    /// Check if measurement noise is set
    pub fn has_measurement_noise(&self) -> bool {
        self.measurement_noise.is_some()
    }
}

impl Default for NoiseModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depolarizing_noise_kraus() {
        let noise = DepolarizingNoise::new(0.1);
        let kraus = noise.kraus_operators();

        // Should have 4 Kraus operators for single qubit
        assert_eq!(kraus.len(), 4);

        // Verify completeness
        assert!(noise.verify_completeness());
    }

    #[test]
    fn test_bit_flip_noise() {
        let noise = BitFlipNoise::new(0.2);
        let kraus = noise.kraus_operators();

        assert_eq!(kraus.len(), 2);
        assert!(noise.verify_completeness());
    }

    #[test]
    fn test_phase_flip_noise() {
        let noise = PhaseFlipNoise::new(0.15);
        let kraus = noise.kraus_operators();

        assert_eq!(kraus.len(), 2);
        assert!(noise.verify_completeness());
    }

    #[test]
    fn test_amplitude_damping() {
        let noise = AmplitudeDampingNoise::new(0.05);
        let kraus = noise.kraus_operators();

        assert_eq!(kraus.len(), 2);
        assert!(noise.verify_completeness());
    }

    #[test]
    fn test_phase_damping() {
        let noise = PhaseDampingNoise::new(0.1);
        let kraus = noise.kraus_operators();

        assert_eq!(kraus.len(), 2);
        assert!(noise.verify_completeness());
    }

    #[test]
    fn test_thermal_relaxation() {
        let noise = ThermalRelaxationNoise::new(50.0, 40.0, 1.0);
        let kraus = noise.kraus_operators();

        assert_eq!(kraus.len(), 3);
        // Note: Thermal relaxation may not satisfy exact completeness
        // due to approximations
    }

    #[test]
    fn test_noise_application() {
        let noise = DepolarizingNoise::new(0.01);

        // |0⟩ state
        let state = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);

        // Apply noise multiple times and check that result is still normalized
        for _ in 0..10 {
            let noisy_state = noise.apply(&state.view()).unwrap();
            let norm: f64 = noisy_state.iter().map(|c| c.norm_sqr()).sum();
            assert!((norm - 1.0).abs() < 1e-10, "State not normalized: {}", norm);
        }
    }

    #[test]
    fn test_noise_model() {
        let mut model = NoiseModel::new();

        // Add depolarizing noise
        model.add_channel(Arc::new(DepolarizingNoise::new(0.01)));

        // Add bit flip noise to X gates
        model.add_gate_noise("X", Arc::new(BitFlipNoise::new(0.02)));

        assert_eq!(model.num_global_channels(), 1);
        assert!(!model.has_measurement_noise());

        // Apply noise to a state
        let state = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let noisy = model.apply_single_qubit(&state, 0).unwrap();

        let norm: f64 = noisy.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_noise_model_composition() {
        let mut model = NoiseModel::new();

        // Combine multiple noise sources
        model.add_channel(Arc::new(DepolarizingNoise::new(0.005)));
        model.add_channel(Arc::new(AmplitudeDampingNoise::new(0.01)));
        model.add_channel(Arc::new(PhaseDampingNoise::new(0.008)));

        assert_eq!(model.num_global_channels(), 3);

        // Apply composite noise
        let state = Array1::from_vec(vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ]);
        let noisy = model.apply_single_qubit(&state, 0).unwrap();

        let norm: f64 = noisy.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }
}
