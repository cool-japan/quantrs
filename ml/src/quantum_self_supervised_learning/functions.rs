//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::types::{
    DecoherenceModel, EvolutionType, NegativeSamplingStrategy, NoiseType, PreparationMethod,
    QuantumActivation, QuantumAugmentationStrategy, QuantumAugmenter, QuantumDecoder,
    QuantumEncoder, QuantumEncoderDecoder, QuantumMaskingStrategy, QuantumProjectionHead,
    QuantumSSLMethod, QuantumSelfSupervisedConfig, QuantumSelfSupervisedLearner,
    QuantumSimilarityMetric, QuantumState, QuantumStateEvolution, QuantumStatePreparation,
    ReconstructionObjective, ReconstructionStrategy, SSLTrainingConfig,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_quantum_ssl_creation() {
        let config = QuantumSelfSupervisedConfig::default();
        let ssl = QuantumSelfSupervisedLearner::new(config);
        assert!(ssl.is_ok());
    }
    #[test]
    fn test_quantum_augmentations() {
        let config = QuantumSelfSupervisedConfig::default();
        let augmenter = QuantumAugmenter {
            augmentation_strategies: vec![QuantumAugmentationStrategy::QuantumNoise {
                noise_type: NoiseType::Gaussian,
                strength: 0.1,
            }],
            augmentation_strength: 0.5,
            quantum_coherence_preservation: 0.9,
        };
        let sample = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let views = augmenter.generate_augmented_views(&sample, 2);
        assert!(views.is_ok());
        assert_eq!(views.expect("views should be ok").len(), 2);
    }
    #[test]
    fn test_ssl_training_config() {
        let config = SSLTrainingConfig::default();
        assert_eq!(config.batch_size, 256);
        assert_eq!(config.epochs, 100);
    }
    #[test]
    fn test_quantum_contrastive_method() {
        let config = QuantumSelfSupervisedConfig {
            ssl_method: QuantumSSLMethod::QuantumContrastive {
                similarity_metric: QuantumSimilarityMetric::QuantumCosine,
                negative_sampling_strategy: NegativeSamplingStrategy::Random,
                quantum_projection_head: QuantumProjectionHead {
                    hidden_dims: vec![128, 64],
                    output_dim: 32,
                    use_batch_norm: true,
                    quantum_layers: Vec::new(),
                    activation: QuantumActivation::QuantumReLU,
                },
            },
            ..Default::default()
        };
        let ssl = QuantumSelfSupervisedLearner::new(config);
        assert!(ssl.is_ok());
    }
    #[test]
    fn test_quantum_masked_method() {
        let config = QuantumSelfSupervisedConfig {
            ssl_method: QuantumSSLMethod::QuantumMasked {
                masking_strategy: QuantumMaskingStrategy::Random {
                    mask_probability: 0.15,
                },
                reconstruction_objective: ReconstructionObjective::MSE,
                quantum_encoder_decoder: QuantumEncoderDecoder {
                    encoder: QuantumEncoder {
                        layers: Vec::new(),
                        quantum_state_evolution: QuantumStateEvolution {
                            evolution_type: EvolutionType::Unitary,
                            time_steps: Array1::linspace(0.0, 1.0, 10),
                            hamiltonian: Array2::<f64>::eye(8).mapv(|x| Complex64::new(x, 0.0)),
                            decoherence_model: DecoherenceModel::default(),
                        },
                        measurement_points: vec![0, 1],
                    },
                    decoder: QuantumDecoder {
                        layers: Vec::new(),
                        quantum_state_preparation: QuantumStatePreparation {
                            preparation_method: PreparationMethod::DirectPreparation,
                            target_state: QuantumState::default(),
                            fidelity_threshold: 0.95,
                        },
                        reconstruction_strategy: ReconstructionStrategy::FullReconstruction,
                    },
                    shared_quantum_state: true,
                    entanglement_coupling: 0.5,
                },
            },
            ..Default::default()
        };
        let ssl = QuantumSelfSupervisedLearner::new(config);
        assert!(ssl.is_ok());
    }
}
