//! ML Training Configuration Types

use serde::{Deserialize, Serialize};

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Maximum training iterations
    pub max_iterations: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Early stopping criteria
    pub early_stopping: EarlyStoppingConfig,
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Training data split
    pub train_test_split: f64,
    /// Optimization algorithm for training
    pub optimizer: TrainingOptimizer,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Enable early stopping
    pub enable_early_stopping: bool,
    /// Patience (iterations without improvement)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_improvement: f64,
    /// Restoration of best weights
    pub restore_best_weights: bool,
}

/// Training optimizers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrainingOptimizer {
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    LBFGS,
}

/// Regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    /// L1 regularization strength
    pub l1_lambda: f64,
    /// L2 regularization strength
    pub l2_lambda: f64,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Batch normalization
    pub batch_normalization: bool,
    /// Weight decay
    pub weight_decay: f64,
}
