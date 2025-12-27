//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use crate::quantum_advanced_diffusion::types::*;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView1, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::random::ChaCha20Rng;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::Complex64;
use std::f64::consts::PI;
#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_circuit::builder::Circuit;
    #[test]
    fn test_quantum_advanced_diffusion_creation() {
        let config = QuantumAdvancedDiffusionConfig::default();
        let model = QuantumAdvancedDiffusionModel::new(config);
        assert!(model.is_ok());
    }
    #[test]
    fn test_quantum_noise_schedule() {
        let config = QuantumAdvancedDiffusionConfig::default();
        let (betas, alphas, alphas_cumprod) =
            QuantumAdvancedDiffusionModel::compute_quantum_schedule(&config)
                .expect("Failed to compute quantum schedule");
        assert_eq!(betas.len(), config.num_timesteps);
        assert_eq!(alphas.len(), config.num_timesteps);
        assert_eq!(alphas_cumprod.len(), config.num_timesteps);
        for i in 1..alphas_cumprod.len() {
            assert!(alphas_cumprod[i] <= alphas_cumprod[i - 1]);
        }
    }
    #[test]
    fn test_quantum_forward_diffusion() {
        let config = QuantumAdvancedDiffusionConfig {
            data_dim: 4,
            num_qubits: 8,
            num_timesteps: 100,
            ..Default::default()
        };
        let model =
            QuantumAdvancedDiffusionModel::new(config).expect("Failed to create diffusion model");
        let x0 = Array1::from_vec(vec![0.5, -0.3, 0.8, -0.1]);
        let result = model.quantum_forward_diffusion(&x0, 50);
        assert!(result.is_ok());
        let (xt, quantum_noise, quantum_state) = result.expect("Forward diffusion should succeed");
        assert_eq!(xt.len(), 4);
        assert_eq!(quantum_noise.len(), 4);
        assert!(quantum_state.entanglement_measure >= 0.0);
        assert!(quantum_state.entanglement_measure <= 1.0);
    }
    #[test]
    fn test_quantum_denoising_network() {
        let config = QuantumAdvancedDiffusionConfig {
            data_dim: 8,
            num_qubits: 4,
            ..Default::default()
        };
        let network = QuantumDenoisingNetwork::new(&config);
        assert!(network.is_ok());
        let network = network.expect("Failed to create denoising network");
        assert!(!network.quantum_layers.is_empty());
        assert!(!network.classical_layers.is_empty());
    }
    #[test]
    fn test_quantum_generation() {
        let config = QuantumAdvancedDiffusionConfig {
            data_dim: 2,
            num_qubits: 4,
            num_timesteps: 10,
            ..Default::default()
        };
        let model =
            QuantumAdvancedDiffusionModel::new(config).expect("Failed to create diffusion model");
        let result = model.quantum_generate(3, None, None);
        assert!(result.is_ok());
        let output = result.expect("Quantum generation should succeed");
        assert_eq!(output.samples.shape(), &[3, 2]);
        assert_eq!(output.generation_metrics.len(), 3);
    }
    #[test]
    fn test_multi_scale_noise_schedule() {
        let config = QuantumAdvancedDiffusionConfig {
            noise_schedule: QuantumNoiseSchedule::MultiScale {
                scales: vec![1.0, 0.5, 0.25],
                weights: Array1::from_vec(vec![0.5, 0.3, 0.2]),
                coherence_times: Array1::from_vec(vec![10.0, 5.0, 2.0]),
            },
            ..Default::default()
        };
        let result = QuantumAdvancedDiffusionModel::compute_quantum_schedule(&config);
        assert!(result.is_ok());
        let (betas, _, _) = result.expect("Multi-scale noise schedule should compute");
        assert!(betas.iter().all(|&beta| beta >= 0.0 && beta <= 1.0));
    }
    #[test]
    fn test_quantum_metrics_computation() {
        let quantum_state = QuantumState {
            classical_data: Array1::from_vec(vec![0.1, 0.2, 0.3]),
            quantum_phase: Complex64::new(0.8, 0.6),
            entanglement_measure: 0.7,
            coherence_time: 0.9,
            fidelity: 0.85,
        };
        assert!((quantum_state.quantum_phase.norm() - 1.0).abs() < 1e-10);
        assert!(quantum_state.entanglement_measure >= 0.0);
        assert!(quantum_state.entanglement_measure <= 1.0);
        assert!(quantum_state.fidelity >= 0.0);
        assert!(quantum_state.fidelity <= 1.0);
    }
}
