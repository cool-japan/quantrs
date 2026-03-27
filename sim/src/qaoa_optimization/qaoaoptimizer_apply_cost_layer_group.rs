//! # QAOAOptimizer - apply_cost_layer_group Methods
//!
//! This module contains method implementations for `QAOAOptimizer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use crate::error::Result;
use scirs2_core::random::prelude::*;

use super::types::QAOAProblemType;

use super::qaoaoptimizer_type::QAOAOptimizer;

impl QAOAOptimizer {
    /// Apply cost layer to circuit
    pub(super) fn apply_cost_layer(
        &self,
        circuit: &mut InterfaceCircuit,
        gamma: f64,
    ) -> Result<()> {
        match self.problem_type {
            QAOAProblemType::MaxCut => {
                self.apply_maxcut_cost_layer(circuit, gamma)?;
            }
            QAOAProblemType::MaxWeightIndependentSet => {
                self.apply_mwis_cost_layer(circuit, gamma)?;
            }
            QAOAProblemType::TSP => {
                self.apply_tsp_cost_layer(circuit, gamma)?;
            }
            QAOAProblemType::PortfolioOptimization => {
                self.apply_portfolio_cost_layer(circuit, gamma)?;
            }
            QAOAProblemType::Boolean3SAT => {
                self.apply_3sat_cost_layer(circuit, gamma)?;
            }
            QAOAProblemType::QUBO => {
                self.apply_qubo_cost_layer(circuit, gamma)?;
            }
            _ => {
                self.apply_generic_cost_layer(circuit, gamma)?;
            }
        }
        Ok(())
    }
}
