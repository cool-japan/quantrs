//! Ensemble Learning Configuration Types

use serde::{Deserialize, Serialize};

/// Ensemble configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Enable ensemble methods
    pub enable_ensemble: bool,
    /// Ensemble methods
    pub ensemble_methods: Vec<EnsembleMethod>,
    /// Number of models in ensemble
    pub num_models: usize,
    /// Voting strategy
    pub voting_strategy: VotingStrategy,
    /// Diversity measures
    pub diversity_measures: Vec<DiversityMeasure>,
    /// Dynamic ensemble selection
    pub dynamic_selection: bool,
}

/// Ensemble methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnsembleMethod {
    Bagging,
    Boosting,
    Stacking,
    VotingClassifier,
    RandomSubspace,
    DynamicSelection,
}

/// Voting strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VotingStrategy {
    Majority,
    Weighted,
    Stacking,
    BayesianAveraging,
    PerformanceBased,
}

/// Diversity measures
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiversityMeasure {
    PairwiseDisagreement,
    EntropyMeasure,
    CorrelationCoefficient,
    QStatistic,
    KappaDiversity,
}
