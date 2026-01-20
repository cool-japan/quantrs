//! # SegmentationHead - Trait Implementations
//!
//! This module contains trait implementations for `SegmentationHead`.
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

use super::types::SegmentationHead;

impl TaskHead for SegmentationHead {
    fn forward(&self, features: &Array4<f64>) -> Result<TaskOutput> {
        let (batch_size, _, height, width) = features.dim();
        let masks = Array4::zeros((batch_size, self.num_classes, height, width));
        let class_scores = Array4::zeros((batch_size, self.num_classes, height, width));
        Ok(TaskOutput::Segmentation {
            masks,
            class_scores,
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
