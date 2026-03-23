//! # QuantumSelfSupervisedConfig - Trait Implementations
//!
//! This module contains trait implementations for `QuantumSelfSupervisedConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    ClusteringConfig, ContrastiveConfig, ContrastiveLossFunction, MaskEvolutionType, MaskStrategy,
    MaskedLearningConfig, MomentumConfig, NegativePairStrategy, PositivePairStrategy,
    PrototypeUpdateStrategy, QuantumAssignmentMethod, QuantumClusteringMethod,
    QuantumMaskEvolution, QuantumProjector, QuantumSSLMethod, QuantumSelfSupervisedConfig,
    ReconstructionTarget, TargetNetworkUpdate, TemperatureScheduling,
};

impl Default for QuantumSelfSupervisedConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            representation_dim: 128,
            num_qubits: 8,
            ssl_method: QuantumSSLMethod::QuantumSimCLR {
                batch_size: 256,
                augmentation_strength: 0.5,
                quantum_projector: QuantumProjector {
                    projection_layers: Vec::new(),
                    output_normalization: true,
                    quantum_enhancement: 1.0,
                },
            },
            quantum_enhancement_level: 1.0,
            temperature: 0.1,
            momentum_coefficient: 0.999,
            use_quantum_augmentations: true,
            enable_entanglement_similarity: true,
            contrastive_config: ContrastiveConfig {
                positive_pair_strategy: PositivePairStrategy::Augmentation,
                negative_pair_strategy: NegativePairStrategy::Random,
                loss_function: ContrastiveLossFunction::InfoNCE,
                temperature_scheduling: TemperatureScheduling::Fixed,
            },
            masked_learning_config: MaskedLearningConfig {
                mask_ratio: 0.15,
                mask_strategy: MaskStrategy::Random,
                reconstruction_target: ReconstructionTarget::RawPixels,
                quantum_mask_evolution: QuantumMaskEvolution {
                    evolution_type: MaskEvolutionType::Static,
                    adaptation_rate: 0.01,
                    quantum_coherence_preservation: 0.9,
                },
            },
            momentum_config: MomentumConfig {
                momentum_coefficient: 0.999,
                target_network_update: TargetNetworkUpdate::Soft,
                quantum_momentum_preservation: 0.9,
            },
            clustering_config: ClusteringConfig {
                num_clusters: 256,
                clustering_method: QuantumClusteringMethod::QuantumKMeans,
                prototype_update_strategy: PrototypeUpdateStrategy::MovingAverage,
                quantum_assignment_method: QuantumAssignmentMethod::SoftAssignment,
            },
        }
    }
}
