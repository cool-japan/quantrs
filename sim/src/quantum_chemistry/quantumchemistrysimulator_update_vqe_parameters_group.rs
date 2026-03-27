//! # QuantumChemistrySimulator - update_vqe_parameters_group Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::{ChemistryOptimizer, MolecularHamiltonian};

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Update VQE parameters using optimizer
    pub(super) fn update_vqe_parameters(
        &mut self,
        circuit: &InterfaceCircuit,
        hamiltonian: &MolecularHamiltonian,
    ) -> Result<()> {
        match self.config.vqe_config.optimizer {
            ChemistryOptimizer::GradientDescent => {
                self.gradient_descent_update(circuit, hamiltonian)?;
            }
            ChemistryOptimizer::Adam => {
                self.adam_update(circuit, hamiltonian)?;
            }
            _ => {
                self.random_perturbation_update()?;
            }
        }
        self.stats.parameter_updates += 1;
        Ok(())
    }
    /// Gradient descent parameter update
    pub(super) fn gradient_descent_update(
        &mut self,
        circuit: &InterfaceCircuit,
        hamiltonian: &MolecularHamiltonian,
    ) -> Result<()> {
        let gradient = self.compute_parameter_gradient(circuit, hamiltonian)?;
        for i in 0..self.vqe_optimizer.parameters.len() {
            self.vqe_optimizer.parameters[i] -= self.vqe_optimizer.learning_rate * gradient[i];
        }
        Ok(())
    }
    /// Adam optimizer update (simplified)
    pub(super) fn adam_update(
        &mut self,
        circuit: &InterfaceCircuit,
        hamiltonian: &MolecularHamiltonian,
    ) -> Result<()> {
        let gradient = self.compute_parameter_gradient(circuit, hamiltonian)?;
        for i in 0..self.vqe_optimizer.parameters.len() {
            self.vqe_optimizer.parameters[i] -= self.vqe_optimizer.learning_rate * gradient[i];
        }
        Ok(())
    }
    /// Random perturbation update for non-gradient optimizers
    pub(super) fn random_perturbation_update(&mut self) -> Result<()> {
        for i in 0..self.vqe_optimizer.parameters.len() {
            let perturbation = (thread_rng().random::<f64>() - 0.5) * 0.1;
            self.vqe_optimizer.parameters[i] += perturbation;
        }
        Ok(())
    }
}
