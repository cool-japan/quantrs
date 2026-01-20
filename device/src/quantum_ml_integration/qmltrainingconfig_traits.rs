//! # QMLTrainingConfig - Trait Implementations
//!
//! This module contains trait implementations for `QMLTrainingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

// Import types from sibling modules
use super::types::*;
// Merged into super::types
// Merged into super::types

impl Default for QMLTrainingConfig {
    fn default() -> Self {
        Self {
            max_epochs: 100,
            learning_rate: 0.01,
            batch_size: 32,
            early_stopping: EarlyStoppingConfig {
                enabled: true,
                patience: 10,
                min_delta: 1e-4,
                monitor_metric: "val_loss".to_string(),
                mode: ImprovementMode::Minimize,
            },
            gradient_method: GradientMethod::ParameterShift,
            loss_function: LossFunction::MeanSquaredError,
            regularization: RegularizationConfig {
                l1_lambda: 0.0,
                l2_lambda: 0.01,
                dropout_rate: 0.1,
                quantum_noise: 0.0,
                parameter_constraints: ParameterConstraints {
                    min_value: Some(-10.0),
                    max_value: Some(10.0),
                    enforce_unitarity: false,
                    enforce_hermiticity: false,
                    custom_constraints: Vec::new(),
                },
            },
            validation_config: ValidationConfig {
                validation_split: 0.2,
                cv_folds: None,
                validation_frequency: 1,
                enable_test_evaluation: true,
            },
        }
    }
}
