//! # QAOAOptimizer - evaluate_qaoa_cost_group Methods
//!
//! This module contains method implementations for `QAOAOptimizer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::qaoaoptimizer_type::QAOAOptimizer;

impl QAOAOptimizer {
    /// Evaluate QAOA cost function
    pub(super) fn evaluate_qaoa_cost(&self, gammas: &[f64], betas: &[f64]) -> Result<f64> {
        let circuit = self.generate_qaoa_circuit(gammas, betas)?;
        let state = self.simulate_circuit(&circuit)?;
        let cost = self.calculate_cost_expectation(&state)?;
        Ok(cost)
    }
    /// Calculate cost expectation value
    pub(super) fn calculate_cost_expectation(&self, state: &Array1<Complex64>) -> Result<f64> {
        let mut expectation = 0.0;
        for (idx, amplitude) in state.iter().enumerate() {
            let probability = amplitude.norm_sqr();
            if probability > 1e-10 {
                let bitstring = format!("{:0width$b}", idx, width = self.graph.num_vertices);
                let cost = self.evaluate_classical_cost(&bitstring)?;
                expectation += probability * cost;
            }
        }
        Ok(expectation)
    }
    /// Parameter optimization methods
    /// Classical parameter optimization
    pub(super) fn classical_parameter_optimization(&mut self) -> Result<()> {
        let epsilon = 1e-4;
        let mut gamma_gradients = vec![0.0; self.gammas.len()];
        let mut beta_gradients = vec![0.0; self.betas.len()];
        for i in 0..self.gammas.len() {
            let mut gammas_plus = self.gammas.clone();
            let mut gammas_minus = self.gammas.clone();
            gammas_plus[i] += epsilon;
            gammas_minus[i] -= epsilon;
            let cost_plus = self.evaluate_qaoa_cost(&gammas_plus, &self.betas)?;
            let cost_minus = self.evaluate_qaoa_cost(&gammas_minus, &self.betas)?;
            gamma_gradients[i] = (cost_plus - cost_minus) / (2.0 * epsilon);
        }
        for i in 0..self.betas.len() {
            let mut betas_plus = self.betas.clone();
            let mut betas_minus = self.betas.clone();
            betas_plus[i] += epsilon;
            betas_minus[i] -= epsilon;
            let cost_plus = self.evaluate_qaoa_cost(&self.gammas, &betas_plus)?;
            let cost_minus = self.evaluate_qaoa_cost(&self.gammas, &betas_minus)?;
            beta_gradients[i] = (cost_plus - cost_minus) / (2.0 * epsilon);
        }
        for i in 0..self.gammas.len() {
            self.gammas[i] += self.config.learning_rate * gamma_gradients[i];
        }
        for i in 0..self.betas.len() {
            self.betas[i] += self.config.learning_rate * beta_gradients[i];
        }
        Ok(())
    }
    /// Quantum parameter optimization using parameter shift rule
    pub(super) fn quantum_parameter_optimization(&mut self) -> Result<()> {
        let shift = std::f64::consts::PI / 2.0;
        for i in 0..self.gammas.len() {
            let mut gammas_plus = self.gammas.clone();
            let mut gammas_minus = self.gammas.clone();
            gammas_plus[i] += shift;
            gammas_minus[i] -= shift;
            let cost_plus = self.evaluate_qaoa_cost(&gammas_plus, &self.betas)?;
            let cost_minus = self.evaluate_qaoa_cost(&gammas_minus, &self.betas)?;
            let gradient = (cost_plus - cost_minus) / 2.0;
            self.gammas[i] += self.config.learning_rate * gradient;
        }
        for i in 0..self.betas.len() {
            let mut betas_plus = self.betas.clone();
            let mut betas_minus = self.betas.clone();
            betas_plus[i] += shift;
            betas_minus[i] -= shift;
            let cost_plus = self.evaluate_qaoa_cost(&self.gammas, &betas_plus)?;
            let cost_minus = self.evaluate_qaoa_cost(&self.gammas, &betas_minus)?;
            let gradient = (cost_plus - cost_minus) / 2.0;
            self.betas[i] += self.config.learning_rate * gradient;
        }
        Ok(())
    }
}
