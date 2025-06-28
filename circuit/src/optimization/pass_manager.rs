//! Pass manager for orchestrating optimization passes
//!
//! This module manages the execution of optimization passes in a configurable way.

use crate::builder::Circuit;
use crate::optimization::cost_model::{CircuitCostExt, CostModel};
use crate::optimization::passes::{OptimizationPassExt, *};
use quantrs2_core::error::QuantRS2Result;
use serde::{Serialize, Deserialize};
use std::collections::HashSet;

/// Optimization level presets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Light optimization - fast passes only
    Light,
    /// Medium optimization - balanced
    Medium,
    /// Heavy optimization - all passes, may be slow
    Heavy,
    /// Custom optimization - user-defined passes
    Custom,
}

/// Configuration for the pass manager
#[derive(Debug, Clone)]
pub struct PassConfig {
    /// Maximum iterations for iterative passes
    pub max_iterations: usize,
    /// Enable aggressive optimizations
    pub aggressive: bool,
    /// Target gate set
    pub target_gates: HashSet<String>,
    /// Hardware backend name
    pub backend: Option<String>,
    /// Optimization level
    pub level: OptimizationLevel,
    /// Specific passes to disable
    pub disabled_passes: HashSet<String>,
}

impl Default for PassConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            aggressive: false,
            target_gates: HashSet::new(),
            backend: None,
            level: OptimizationLevel::Medium,
            disabled_passes: HashSet::new(),
        }
    }
}

/// Pass manager that orchestrates optimization passes
pub struct PassManager {
    passes: Vec<Box<dyn OptimizationPass>>,
    config: PassConfig,
    applied_passes: Vec<String>,
}

impl PassManager {
    /// Create a new pass manager with default configuration
    pub fn new() -> Self {
        Self::with_level(OptimizationLevel::Medium)
    }

    /// Create a pass manager with a specific optimization level
    pub fn with_level(level: OptimizationLevel) -> Self {
        let config = PassConfig {
            level,
            ..Default::default()
        };

        let passes = Self::create_passes_for_level(level, &config);

        Self {
            passes,
            config,
            applied_passes: Vec::new(),
        }
    }

    /// Create a pass manager optimized for specific hardware
    pub fn for_hardware(hardware: &str) -> Self {
        let mut config = PassConfig {
            level: OptimizationLevel::Medium,
            backend: Some(hardware.to_string()),
            ..Default::default()
        };

        // Set hardware-specific target gates
        config.target_gates = match hardware {
            "ibm" => vec!["X", "Y", "Z", "H", "S", "T", "RZ", "CNOT", "CZ"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            "google" => vec!["X", "Y", "Z", "H", "RZ", "CZ", "SQRT_X"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            "aws" => vec!["X", "Y", "Z", "H", "RZ", "RX", "RY", "CNOT", "CZ"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            _ => HashSet::new(),
        };

        let passes = Self::create_passes_for_hardware(hardware, &config);

        Self {
            passes,
            config,
            applied_passes: Vec::new(),
        }
    }

    /// Configure the pass manager
    pub fn configure(&mut self, config: PassConfig) {
        self.config = config;
        self.passes = Self::create_passes_for_level(self.config.level, &self.config);
    }

    /// Add a custom optimization pass
    pub fn add_pass(&mut self, pass: Box<dyn OptimizationPass>) {
        self.passes.push(pass);
    }

    /// Remove a pass by name
    pub fn remove_pass(&mut self, name: &str) {
        self.passes.retain(|p| p.name() != name);
    }

    /// Run all optimization passes on a circuit
    pub fn run<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
        cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Circuit<N>> {
        self.applied_passes.clear();
        let mut current_circuit = circuit.clone();
        let mut iteration = 0;
        let mut improved = true;

        while improved && iteration < self.config.max_iterations {
            improved = false;
            let start_cost = cost_model.circuit_cost(&current_circuit);

            for pass in &self.passes {
                if self.config.disabled_passes.contains(pass.name()) {
                    continue;
                }

                if pass.should_apply() {
                    let optimized = pass.apply(&current_circuit, cost_model)?;
                    let new_cost = cost_model.circuit_cost(&optimized);

                    if new_cost < start_cost {
                        current_circuit = optimized;
                        self.applied_passes.push(pass.name().to_string());
                        improved = true;
                    }
                }
            }

            iteration += 1;
        }

        Ok(current_circuit)
    }

    /// Get the list of applied passes
    pub fn get_applied_passes(&self) -> Vec<String> {
        self.applied_passes.clone()
    }

    /// Create passes for a given optimization level
    fn create_passes_for_level(
        level: OptimizationLevel,
        config: &PassConfig,
    ) -> Vec<Box<dyn OptimizationPass>> {
        match level {
            OptimizationLevel::None => vec![],

            OptimizationLevel::Light => vec![
                Box::new(GateCancellation::new(false)),
                Box::new(RotationMerging::new(1e-10)),
            ],

            OptimizationLevel::Medium => vec![
                Box::new(GateCancellation::new(false)),
                Box::new(GateCommutation::new(5)),
                Box::new(PeepholeOptimization::new(3)),
                Box::new(RotationMerging::new(1e-10)),
                Box::new(GateMerging::new(true, 1e-10)),
                Box::new(TemplateMatching::new()),
            ],

            OptimizationLevel::Heavy => vec![
                Box::new(GateCancellation::new(true)),
                Box::new(GateCommutation::new(10)),
                Box::new(PeepholeOptimization::new(4)),
                Box::new(RotationMerging::new(1e-12)),
                Box::new(GateMerging::new(true, 1e-12)),
                Box::new(DecompositionOptimization::new(
                    config.target_gates.clone(),
                    true,
                )),
                Box::new(TwoQubitOptimization::new(true, true)),
                Box::new(TemplateMatching::new()),
                Box::new(CircuitRewriting::new(100)),
                Box::new(CostBasedOptimization::new(CostTarget::Balanced, 20)),
            ],

            OptimizationLevel::Custom => vec![],
        }
    }

    /// Create passes optimized for specific hardware
    fn create_passes_for_hardware(
        hardware: &str,
        config: &PassConfig,
    ) -> Vec<Box<dyn OptimizationPass>> {
        let mut passes: Vec<Box<dyn OptimizationPass>> = vec![
            Box::new(GateCancellation::new(false)),
            Box::new(GateCommutation::new(5)),
            Box::new(PeepholeOptimization::new(3)),
            Box::new(RotationMerging::new(1e-10)),
        ];

        match hardware {
            "ibm" => {
                passes.push(Box::new(DecompositionOptimization::for_hardware("ibm")));
                passes.push(Box::new(TwoQubitOptimization::new(false, true)));
            }
            "google" => {
                passes.push(Box::new(DecompositionOptimization::for_hardware("google")));
                passes.push(Box::new(TwoQubitOptimization::new(true, false)));
            }
            "aws" => {
                passes.push(Box::new(DecompositionOptimization::for_hardware("aws")));
                passes.push(Box::new(CostBasedOptimization::new(
                    CostTarget::TotalError,
                    10,
                )));
            }
            _ => {
                passes.push(Box::new(DecompositionOptimization::new(
                    config.target_gates.clone(),
                    true,
                )));
            }
        }

        passes.push(Box::new(TemplateMatching::new()));
        passes
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}
