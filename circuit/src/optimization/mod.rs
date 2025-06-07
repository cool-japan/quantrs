//! Advanced circuit optimization module
//!
//! This module provides a comprehensive optimization framework that uses gate properties
//! to optimize quantum circuits through various optimization passes.

pub mod gate_properties;
pub mod passes;
pub mod pass_manager;
pub mod cost_model;
pub mod analysis;

pub use gate_properties::{GateProperties, GateCost, GateError, CommutationTable};
pub use passes::{
    OptimizationPass, GateCancellation, GateCommutation, GateMerging, 
    DecompositionOptimization, CostBasedOptimization, RotationMerging,
    TwoQubitOptimization, TemplateMatching, CircuitRewriting
};
pub use pass_manager::{PassManager, PassConfig, OptimizationLevel};
pub use cost_model::{CostModel, HardwareCostModel, AbstractCostModel};
pub use analysis::{CircuitAnalyzer, CircuitMetrics, OptimizationReport};

use crate::builder::Circuit;
use quantrs2_core::error::QuantRS2Result;
use self::cost_model::CircuitCostExt;

/// Main optimization interface
pub struct CircuitOptimizer2<const N: usize> {
    pass_manager: PassManager,
    cost_model: Box<dyn CostModel>,
    analyzer: CircuitAnalyzer,
}

impl<const N: usize> CircuitOptimizer2<N> {
    /// Create a new optimizer with default settings
    pub fn new() -> Self {
        Self {
            pass_manager: PassManager::default(),
            cost_model: Box::new(AbstractCostModel::default()),
            analyzer: CircuitAnalyzer::new(),
        }
    }

    /// Create an optimizer with a specific optimization level
    pub fn with_level(level: OptimizationLevel) -> Self {
        Self {
            pass_manager: PassManager::with_level(level),
            cost_model: Box::new(AbstractCostModel::default()),
            analyzer: CircuitAnalyzer::new(),
        }
    }

    /// Create an optimizer for specific hardware
    pub fn for_hardware(hardware: &str) -> Self {
        Self {
            pass_manager: PassManager::for_hardware(hardware),
            cost_model: Box::new(HardwareCostModel::for_backend(hardware)),
            analyzer: CircuitAnalyzer::new(),
        }
    }

    /// Optimize a circuit
    pub fn optimize(&mut self, circuit: &Circuit<N>) -> QuantRS2Result<OptimizationReport> {
        // Analyze initial circuit
        let initial_metrics = self.analyzer.analyze(circuit)?;
        
        // Run optimization passes
        let optimized_circuit = self.pass_manager.run(circuit, &*self.cost_model)?;
        
        // Analyze optimized circuit
        let final_metrics = self.analyzer.analyze(&optimized_circuit)?;
        
        // Generate report
        Ok(OptimizationReport {
            initial_metrics,
            final_metrics,
            applied_passes: self.pass_manager.get_applied_passes(),
        })
    }

    /// Add a custom optimization pass
    pub fn add_pass(&mut self, pass: Box<dyn OptimizationPass>) {
        self.pass_manager.add_pass(pass);
    }

    /// Set a custom cost model
    pub fn set_cost_model(&mut self, cost_model: Box<dyn CostModel>) {
        self.cost_model = cost_model;
    }

    /// Configure the optimizer
    pub fn configure(&mut self, config: PassConfig) {
        self.pass_manager.configure(config);
    }
}

impl<const N: usize> Default for CircuitOptimizer2<N> {
    fn default() -> Self {
        Self::new()
    }
}