//! # FeatureExtractionHead - Trait Implementations
//!
//! This module contains trait implementations for `FeatureExtractionHead`.
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

use super::types::FeatureExtractionHead;

impl TaskHead for FeatureExtractionHead {
    fn forward(&self, features: &Array4<f64>) -> Result<TaskOutput> {
        let (batch_size, channels, _, _) = features.dim();
        let pooled = features
            .mean_axis(Axis(2))
            .unwrap()
            .mean_axis(Axis(2))
            .unwrap();
        let mut extracted_features = Array2::zeros((batch_size, self.feature_dim));
        for i in 0..batch_size {
            let feature_vec = pooled.slice(s![i, ..]).to_owned();
            for j in 0..self.feature_dim {
                extracted_features[[i, j]] = feature_vec[j % channels];
            }
            if self.normalize {
                let norm = extracted_features
                    .slice(s![i, ..])
                    .mapv(|x| x * x)
                    .sum()
                    .sqrt();
                if norm > 1e-10 {
                    extracted_features
                        .slice_mut(s![i, ..])
                        .mapv_inplace(|x| x / norm);
                }
            }
        }
        Ok(TaskOutput::Features {
            features: extracted_features,
            attention_maps: None,
        })
    }
    fn parameters(&self) -> &Array1<f64> {
        &self.parameters
    }
    fn update_parameters(&mut self, _params: &Array1<f64>) -> Result<()> {
        Ok(())
    }
    fn clone_box(&self) -> Box<dyn TaskHead> {
        Box::new(self.clone())
    }
}
