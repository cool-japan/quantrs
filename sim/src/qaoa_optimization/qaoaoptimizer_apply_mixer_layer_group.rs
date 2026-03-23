//! # QAOAOptimizer - apply_mixer_layer_group Methods
//!
//! This module contains method implementations for `QAOAOptimizer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use crate::error::Result;
use scirs2_core::random::prelude::*;

use super::types::{QAOAMixerType, QAOAProblemType};

use super::qaoaoptimizer_type::QAOAOptimizer;

impl QAOAOptimizer {
    /// Apply mixer layer to circuit
    pub(super) fn apply_mixer_layer(
        &self,
        circuit: &mut InterfaceCircuit,
        beta: f64,
    ) -> Result<()> {
        match self.config.mixer_type {
            QAOAMixerType::Standard => {
                self.apply_standard_mixer(circuit, beta)?;
            }
            QAOAMixerType::XY => {
                self.apply_xy_mixer(circuit, beta)?;
            }
            QAOAMixerType::Ring => {
                self.apply_ring_mixer(circuit, beta)?;
            }
            QAOAMixerType::Grover => {
                self.apply_grover_mixer(circuit, beta)?;
            }
            QAOAMixerType::Dicke => {
                self.apply_dicke_mixer(circuit, beta)?;
            }
            QAOAMixerType::Custom => {
                self.apply_custom_mixer(circuit, beta)?;
            }
        }
        Ok(())
    }
    /// Apply custom mixer
    pub(super) fn apply_custom_mixer(
        &self,
        circuit: &mut InterfaceCircuit,
        beta: f64,
    ) -> Result<()> {
        match self.problem_type {
            QAOAProblemType::TSP => {
                self.apply_tsp_custom_mixer(circuit, beta)?;
            }
            QAOAProblemType::PortfolioOptimization => {
                self.apply_portfolio_custom_mixer(circuit, beta)?;
            }
            _ => {
                self.apply_standard_mixer(circuit, beta)?;
            }
        }
        Ok(())
    }
}
