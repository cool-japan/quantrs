//! # QuantumChemistryProblem - Trait Implementations
//!
//! This module contains trait implementations for `QuantumChemistryProblem`.
//!
//! ## Implemented Traits
//!
//! - `OptimizationProblem`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::applications::{
    ApplicationError, ApplicationResult, IndustrySolution, OptimizationProblem,
};
use crate::ising::{IsingModel, QuboModel};
use std::collections::{HashMap, HashSet, VecDeque};

use super::types::{
    ChemistryObjective, QuantumChemistryOptimizer, QuantumChemistryProblem, QuantumChemistryResult,
};

impl OptimizationProblem for QuantumChemistryProblem {
    type Solution = QuantumChemistryResult;
    type ObjectiveValue = f64;
    fn description(&self) -> String {
        format!(
            "Quantum computational chemistry optimization for system: {}",
            self.system.id
        )
    }
    fn size_metrics(&self) -> HashMap<String, usize> {
        let mut metrics = HashMap::new();
        metrics.insert("atoms".to_string(), self.system.atoms.len());
        metrics.insert(
            "basis_functions".to_string(),
            self.system
                .atoms
                .iter()
                .map(|a| a.basis_functions.len())
                .sum(),
        );
        metrics
    }
    fn validate(&self) -> ApplicationResult<()> {
        if self.system.atoms.is_empty() {
            return Err(ApplicationError::InvalidConfiguration(
                "No atoms in molecular system".to_string(),
            ));
        }
        Ok(())
    }
    fn to_qubo(&self) -> ApplicationResult<(QuboModel, HashMap<String, usize>)> {
        let mut optimizer = QuantumChemistryOptimizer::new(self.config.clone())?;
        optimizer.molecular_system_to_qubo(&self.system)
    }
    fn evaluate_solution(
        &self,
        solution: &Self::Solution,
    ) -> ApplicationResult<Self::ObjectiveValue> {
        let mut score = 0.0;
        for objective in &self.objectives {
            let obj_score = match objective {
                ChemistryObjective::MinimizeEnergy => -solution.total_energy,
                ChemistryObjective::MaximizeStability => solution.total_energy.abs(),
                ChemistryObjective::OptimizeGeometry => solution.total_energy.abs() / 10.0,
                ChemistryObjective::MinimizeInteractionEnergy => -solution.electronic_energy,
            };
            score += obj_score;
        }
        Ok(score / self.objectives.len() as f64)
    }
    fn is_feasible(&self, solution: &Self::Solution) -> bool {
        solution.metadata.scf_converged && solution.total_energy.is_finite()
    }
}
