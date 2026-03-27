//! # QAOAOptimizer - hybrid_parameter_optimization_group Methods
//!
//! This module contains method implementations for `QAOAOptimizer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::random::prelude::*;

use super::qaoaoptimizer_type::QAOAOptimizer;

impl QAOAOptimizer {
    /// Hybrid classical-quantum optimization
    pub(super) fn hybrid_parameter_optimization(&mut self) -> Result<()> {
        if self.stats.total_time.as_secs() % 2 == 0 {
            self.classical_parameter_optimization()?;
        } else {
            self.quantum_parameter_optimization()?;
        }
        Ok(())
    }
}
