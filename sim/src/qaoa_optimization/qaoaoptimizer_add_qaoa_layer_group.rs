//! # QAOAOptimizer - add_qaoa_layer_group Methods
//!
//! This module contains method implementations for `QAOAOptimizer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::random::prelude::*;

use super::qaoaoptimizer_type::QAOAOptimizer;

impl QAOAOptimizer {
    pub(super) fn add_qaoa_layer(&mut self) -> Result<()> {
        self.gammas.push(0.1);
        self.betas.push(0.1);
        self.best_gammas.push(0.1);
        self.best_betas.push(0.1);
        Ok(())
    }
}
