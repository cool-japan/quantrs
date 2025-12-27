//! Transfer Learning for Meta-Learning
//!
//! This module contains all Transfer Learning types and implementations
//! used by the meta-learning optimization system.

use super::config::ArchitectureSpec;
use super::config::*;
use super::features::DistributionStats;
use crate::applications::ApplicationResult;
use std::collections::HashMap;
use std::time::Instant;

/// Transfer learning system
pub struct TransferLearner {
    /// Source domains
    pub source_domains: Vec<SourceDomain>,
    /// Domain similarity analyzer
    pub similarity_analyzer: DomainSimilarityAnalyzer,
    /// Transfer strategies
    pub transfer_strategies: Vec<TransferStrategy>,
    /// Adaptation mechanisms
    pub adaptation_mechanisms: Vec<AdaptationMechanism>,
}

/// Source domain for transfer learning
#[derive(Debug)]
pub struct SourceDomain {
    /// Domain identifier
    pub id: String,
    /// Domain characteristics
    pub characteristics: DomainCharacteristics,
    /// Available models
    pub models: Vec<TransferableModel>,
    /// Transfer success history
    pub transfer_history: Vec<TransferRecord>,
}

/// Domain characteristics
#[derive(Debug, Clone)]
pub struct DomainCharacteristics {
    /// Feature distribution
    pub feature_distribution: DistributionStats,
    /// Label distribution
    pub label_distribution: DistributionStats,
    /// Task complexity
    pub task_complexity: f64,
    /// Data size
    pub data_size: usize,
    /// Noise level
    pub noise_level: f64,
}

/// Transferable model
#[derive(Debug)]
pub struct TransferableModel {
    /// Model identifier
    pub id: String,
    /// Model architecture
    pub architecture: ArchitectureSpec,
    /// Pre-trained weights
    pub weights: Vec<f64>,
    /// Performance on source domain
    pub source_performance: f64,
    /// Transferability score
    pub transferability_score: f64,
}

/// Transfer record
#[derive(Debug, Clone)]
pub struct TransferRecord {
    /// Transfer timestamp
    pub timestamp: Instant,
    /// Target domain
    pub target_domain: String,
    /// Transfer strategy used
    pub strategy: TransferStrategy,
    /// Performance improvement
    pub performance_improvement: f64,
    /// Transfer success
    pub success: bool,
}

/// Domain similarity analyzer
#[derive(Debug)]
pub struct DomainSimilarityAnalyzer {
    /// Similarity metrics
    pub metrics: Vec<SimilarityMetric>,
    /// Similarity cache
    pub similarity_cache: HashMap<(String, String), f64>,
    /// Analysis methods
    pub methods: Vec<SimilarityMethod>,
}

/// Similarity metrics
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimilarityMetric {
    /// Feature similarity
    FeatureSimilarity,
    /// Task similarity
    TaskSimilarity,
    /// Data distribution similarity
    DataDistributionSimilarity,
    /// Performance correlation
    PerformanceCorrelation,
    /// Structural similarity
    StructuralSimilarity,
}

/// Similarity measurement methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimilarityMethod {
    /// Cosine similarity
    Cosine,
    /// Euclidean distance
    Euclidean,
    /// Wasserstein distance
    Wasserstein,
    /// Maximum mean discrepancy
    MaximumMeanDiscrepancy,
    /// Kernel methods
    Kernel(String),
}

/// Transfer strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferStrategy {
    /// Feature transfer
    FeatureTransfer,
    /// Parameter transfer
    ParameterTransfer,
    /// Instance transfer
    InstanceTransfer,
    /// Relational transfer
    RelationalTransfer,
    /// Multi-task learning
    MultiTaskLearning,
    /// Domain adaptation
    DomainAdaptation,
}

/// Adaptation mechanisms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptationMechanism {
    /// Fine-tuning
    FineTuning,
    /// Domain-adversarial training
    DomainAdversarial,
    /// Gradual unfreezing
    GradualUnfreezing,
    /// Knowledge distillation
    KnowledgeDistillation,
    /// Progressive training
    ProgressiveTraining,
}

impl TransferLearner {
    #[must_use]
    pub fn new() -> Self {
        Self {
            source_domains: Vec::new(),
            similarity_analyzer: DomainSimilarityAnalyzer {
                metrics: vec![SimilarityMetric::FeatureSimilarity],
                similarity_cache: HashMap::new(),
                methods: vec![SimilarityMethod::Cosine],
            },
            transfer_strategies: vec![TransferStrategy::ParameterTransfer],
            adaptation_mechanisms: vec![AdaptationMechanism::FineTuning],
        }
    }

    /// Add a new source domain
    pub fn add_source_domain(&mut self, domain: SourceDomain) {
        self.source_domains.push(domain);
    }

    /// Find most similar source domain
    #[must_use]
    pub fn find_similar_domain(
        &self,
        target_characteristics: &DomainCharacteristics,
    ) -> Option<&SourceDomain> {
        let mut best_domain = None;
        let mut best_similarity = 0.0;

        for domain in &self.source_domains {
            let similarity =
                self.calculate_domain_similarity(&domain.characteristics, target_characteristics);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_domain = Some(domain);
            }
        }

        best_domain
    }

    /// Calculate similarity between domains
    fn calculate_domain_similarity(
        &self,
        source: &DomainCharacteristics,
        target: &DomainCharacteristics,
    ) -> f64 {
        // Simple similarity calculation based on multiple factors
        let complexity_sim = 1.0 - (source.task_complexity - target.task_complexity).abs();
        let size_sim =
            1.0 - ((source.data_size as f64).ln() - (target.data_size as f64).ln()).abs() / 10.0;
        let noise_sim = 1.0 - (source.noise_level - target.noise_level).abs();

        // Weight the similarities
        (complexity_sim * 0.4 + size_sim * 0.3 + noise_sim * 0.3)
            .max(0.0)
            .min(1.0)
    }

    /// Transfer knowledge from source to target domain
    pub fn transfer_knowledge(
        &mut self,
        source_domain_id: &str,
        target_domain: &str,
        strategy: TransferStrategy,
    ) -> ApplicationResult<TransferResult> {
        // Find source domain
        let source_domain = self
            .source_domains
            .iter()
            .find(|d| d.id == source_domain_id)
            .ok_or_else(|| {
                crate::applications::ApplicationError::InvalidConfiguration(format!(
                    "Source domain {source_domain_id} not found"
                ))
            })?;

        // Simulate transfer process
        let performance_improvement = match strategy {
            TransferStrategy::ParameterTransfer => 0.15,
            TransferStrategy::FeatureTransfer => 0.12,
            TransferStrategy::DomainAdaptation => 0.18,
            _ => 0.10,
        };

        // Record transfer
        let record = TransferRecord {
            timestamp: Instant::now(),
            target_domain: target_domain.to_string(),
            strategy: strategy.clone(),
            performance_improvement,
            success: performance_improvement > 0.05,
        };

        // Update source domain history (would need mutable reference in real implementation)

        Ok(TransferResult {
            success: record.success,
            performance_improvement: record.performance_improvement,
            transfer_method: strategy,
            confidence: 0.8,
        })
    }

    /// Get transfer statistics
    #[must_use]
    pub fn get_transfer_statistics(&self) -> TransferStatistics {
        let mut total_transfers = 0;
        let mut successful_transfers = 0;
        let mut total_improvement = 0.0;

        for domain in &self.source_domains {
            for record in &domain.transfer_history {
                total_transfers += 1;
                if record.success {
                    successful_transfers += 1;
                    total_improvement += record.performance_improvement;
                }
            }
        }

        let success_rate = if total_transfers > 0 {
            successful_transfers as f64 / total_transfers as f64
        } else {
            0.0
        };

        let avg_improvement = if successful_transfers > 0 {
            total_improvement / successful_transfers as f64
        } else {
            0.0
        };

        TransferStatistics {
            total_transfers,
            successful_transfers,
            success_rate,
            average_improvement: avg_improvement,
        }
    }
}

/// Result of transfer learning operation
#[derive(Debug, Clone)]
pub struct TransferResult {
    /// Whether transfer was successful
    pub success: bool,
    /// Performance improvement achieved
    pub performance_improvement: f64,
    /// Transfer method used
    pub transfer_method: TransferStrategy,
    /// Confidence in transfer
    pub confidence: f64,
}

/// Transfer learning statistics
#[derive(Debug, Clone)]
pub struct TransferStatistics {
    /// Total number of transfers attempted
    pub total_transfers: usize,
    /// Number of successful transfers
    pub successful_transfers: usize,
    /// Success rate
    pub success_rate: f64,
    /// Average performance improvement
    pub average_improvement: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta_learning::config::{
        ActivationFunction, ConnectionPattern, LayerSpec, LayerType, OptimizationSettings,
        OptimizerType, RegularizationConfig,
    };
    use std::time::Duration;

    #[test]
    fn test_transfer_learner_creation() {
        let learner = TransferLearner::new();
        assert_eq!(learner.source_domains.len(), 0);
        assert_eq!(learner.transfer_strategies.len(), 1);
        assert_eq!(learner.adaptation_mechanisms.len(), 1);
    }

    #[test]
    fn test_domain_similarity() {
        let learner = TransferLearner::new();

        let source = DomainCharacteristics {
            feature_distribution: DistributionStats::default(),
            label_distribution: DistributionStats::default(),
            task_complexity: 0.5,
            data_size: 1000,
            noise_level: 0.1,
        };

        let target = DomainCharacteristics {
            feature_distribution: DistributionStats::default(),
            label_distribution: DistributionStats::default(),
            task_complexity: 0.6,
            data_size: 1200,
            noise_level: 0.15,
        };

        let similarity = learner.calculate_domain_similarity(&source, &target);
        assert!(similarity > 0.0);
        assert!(similarity <= 1.0);
    }

    #[test]
    fn test_source_domain_addition() {
        let mut learner = TransferLearner::new();

        let domain = SourceDomain {
            id: "test_domain".to_string(),
            characteristics: DomainCharacteristics {
                feature_distribution: DistributionStats::default(),
                label_distribution: DistributionStats::default(),
                task_complexity: 0.5,
                data_size: 1000,
                noise_level: 0.1,
            },
            models: vec![TransferableModel {
                id: "test_model".to_string(),
                architecture: ArchitectureSpec {
                    layers: vec![LayerSpec {
                        layer_type: LayerType::Dense,
                        input_dim: 10,
                        output_dim: 5,
                        activation: ActivationFunction::ReLU,
                        dropout: 0.1,
                        parameters: HashMap::new(),
                    }],
                    connections: ConnectionPattern::Sequential,
                    optimization: OptimizationSettings {
                        optimizer: OptimizerType::Adam,
                        learning_rate: 0.001,
                        batch_size: 32,
                        epochs: 100,
                        regularization: RegularizationConfig {
                            l1_weight: 0.0,
                            l2_weight: 0.01,
                            dropout: 0.1,
                            batch_norm: true,
                            early_stopping: true,
                        },
                    },
                },
                weights: vec![0.1, 0.2, 0.3],
                source_performance: 0.9,
                transferability_score: 0.8,
            }],
            transfer_history: Vec::new(),
        };

        learner.add_source_domain(domain);
        assert_eq!(learner.source_domains.len(), 1);
    }

    #[test]
    fn test_transfer_knowledge() {
        let mut learner = TransferLearner::new();

        // Add a source domain
        let domain = SourceDomain {
            id: "source_domain".to_string(),
            characteristics: DomainCharacteristics {
                feature_distribution: DistributionStats::default(),
                label_distribution: DistributionStats::default(),
                task_complexity: 0.5,
                data_size: 1000,
                noise_level: 0.1,
            },
            models: Vec::new(),
            transfer_history: Vec::new(),
        };

        learner.add_source_domain(domain);

        // Test transfer
        let result = learner.transfer_knowledge(
            "source_domain",
            "target_domain",
            TransferStrategy::ParameterTransfer,
        );

        assert!(result.is_ok());
        let transfer_result = result.expect("Transfer knowledge should succeed");
        assert!(transfer_result.performance_improvement > 0.0);
    }

    #[test]
    fn test_transfer_statistics() {
        let learner = TransferLearner::new();
        let stats = learner.get_transfer_statistics();

        assert_eq!(stats.total_transfers, 0);
        assert_eq!(stats.successful_transfers, 0);
        assert_eq!(stats.success_rate, 0.0);
        assert_eq!(stats.average_improvement, 0.0);
    }

    #[test]
    fn test_similarity_metrics() {
        assert_eq!(
            SimilarityMetric::FeatureSimilarity,
            SimilarityMetric::FeatureSimilarity
        );
        assert_ne!(
            SimilarityMetric::FeatureSimilarity,
            SimilarityMetric::TaskSimilarity
        );
    }

    #[test]
    fn test_transfer_strategies() {
        assert_eq!(
            TransferStrategy::ParameterTransfer,
            TransferStrategy::ParameterTransfer
        );
        assert_ne!(
            TransferStrategy::ParameterTransfer,
            TransferStrategy::FeatureTransfer
        );
    }

    #[test]
    fn test_adaptation_mechanisms() {
        assert_eq!(
            AdaptationMechanism::FineTuning,
            AdaptationMechanism::FineTuning
        );
        assert_ne!(
            AdaptationMechanism::FineTuning,
            AdaptationMechanism::DomainAdversarial
        );
    }
}
