//! Transfer Learning Configuration Types

/// Transfer learning configuration
#[derive(Debug, Clone)]
pub struct TransferConfig {
    /// Enable transfer learning
    pub enabled: bool,
    /// Source domain similarity threshold
    pub similarity_threshold: f64,
    /// Transfer learning method
    pub transfer_method: TransferLearningMethod,
    /// Source data weighting
    pub source_weight: f64,
    /// Domain adaptation strategy
    pub adaptation_strategy: DomainAdaptationStrategy,
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            similarity_threshold: 0.7,
            transfer_method: TransferLearningMethod::ModelTransfer,
            source_weight: 0.1,
            adaptation_strategy: DomainAdaptationStrategy::Gradual,
        }
    }
}

/// Transfer learning methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferLearningMethod {
    /// Transfer entire model
    ModelTransfer,
    /// Transfer hyperparameters only
    HyperparameterTransfer,
    /// Transfer kernel parameters
    KernelTransfer,
    /// Transfer initial points
    InitialPointTransfer,
}

/// Domain adaptation strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DomainAdaptationStrategy {
    /// Immediate adaptation
    Immediate,
    /// Gradual adaptation
    Gradual,
    /// Weighted adaptation
    Weighted,
}
