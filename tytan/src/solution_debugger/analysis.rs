//! Analysis results for the solution debugger.

use super::constraint_analyzer::ConstraintViolation;
use super::energy_analyzer::EnergyBreakdown;
use serde::Serialize;

/// Constraint analysis results
#[derive(Debug, Clone, Serialize)]
pub struct ConstraintAnalysis {
    /// Total number of constraints
    pub total_constraints: usize,
    /// Number of satisfied constraints
    pub satisfied: usize,
    /// Number of violated constraints
    pub violated: usize,
    /// Satisfaction rate (0.0 to 1.0)
    pub satisfaction_rate: f64,
    /// Total penalty incurred
    pub penalty_incurred: f64,
    /// Detailed violations
    pub violations: Vec<ConstraintViolation>,
}

/// Energy analysis results
#[derive(Debug, Clone, Serialize)]
pub struct EnergyAnalysis {
    /// Total energy of the solution
    pub total_energy: f64,
    /// Detailed energy breakdown
    pub breakdown: EnergyBreakdown,
    /// Most critical variables (highest energy contribution)
    pub critical_variables: Vec<(String, f64)>,
    /// Most critical interactions
    pub critical_interactions: Vec<((String, String), f64)>,
    /// Estimated improvement potential (0.0 to 1.0)
    pub improvement_potential: f64,
}
