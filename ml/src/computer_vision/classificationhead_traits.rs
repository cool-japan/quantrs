//! # ClassificationHead - Trait Implementations
//!
//! This module contains trait implementations for `ClassificationHead`.
//!
//! ## Implemented Traits
//!
//! - `TaskHead`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
use scirs2_core::{Complex32, Complex64};
use std::f64::consts::PI;

use super::types::ClassificationHead;

impl TaskHead for ClassificationHead {
    fn forward(&self, features: &Array4<f64>) -> Result<TaskOutput> {
        let (batch_size, _, _, _) = features.dim();
        let pooled = features
            .mean_axis(Axis(2))
            .ok_or_else(|| {
                MLError::ComputationError("Failed to compute mean over axis 2".to_string())
            })?
            .mean_axis(Axis(2))
            .ok_or_else(|| {
                MLError::ComputationError(
                    "Failed to compute mean over axis 2 (second pass)".to_string(),
                )
            })?;
        let mut logits = Array2::zeros((batch_size, self.num_classes));
        let mut probabilities = Array2::zeros((batch_size, self.num_classes));
        for i in 0..batch_size {
            let feature_vec = pooled.slice(s![i, ..]).to_owned();
            let class_logits = self.classifier.forward(&feature_vec)?;
            let max_logit = class_logits
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_logits = class_logits.mapv(|x| (x - max_logit).exp());
            let sum_exp = exp_logits.sum();
            let probs = exp_logits / sum_exp;
            logits.slice_mut(s![i, ..]).assign(&class_logits);
            probabilities.slice_mut(s![i, ..]).assign(&probs);
        }
        Ok(TaskOutput::Classification {
            logits,
            probabilities,
        })
    }
    fn parameters(&self) -> &Array1<f64> {
        &self.classifier.parameters
    }
    fn update_parameters(&mut self, params: &Array1<f64>) -> Result<()> {
        self.classifier.parameters = params.clone();
        Ok(())
    }
    fn clone_box(&self) -> Box<dyn TaskHead> {
        Box::new(self.clone())
    }
}
