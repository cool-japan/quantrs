//! # DetectionHead - Trait Implementations
//!
//! This module contains trait implementations for `DetectionHead`.
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

use super::types::DetectionHead;

impl TaskHead for DetectionHead {
    fn forward(&self, features: &Array4<f64>) -> Result<TaskOutput> {
        let (batch_size, _, _, _) = features.dim();
        let boxes = Array3::zeros((batch_size, 100, 4));
        let scores = Array2::zeros((batch_size, 100));
        let classes = Array2::<f64>::zeros((batch_size, 100));
        Ok(TaskOutput::Detection {
            boxes,
            scores,
            classes: classes.mapv(|x| x as usize),
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
