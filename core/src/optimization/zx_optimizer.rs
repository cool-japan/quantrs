//! ZX-calculus based optimization pass
//!
//! This module integrates ZX-calculus optimization into the
//! general optimization framework.

use crate::{
    error::QuantRS2Result, gate::GateOp, optimization::OptimizationPass, zx_extraction::ZXPipeline,
};

/// ZX-calculus based optimization pass
#[derive(Debug, Clone)]
pub struct ZXOptimizationPass {
    /// Name of this optimization pass
    name: String,
    /// Whether to print optimization statistics
    verbose: bool,
}

impl Default for ZXOptimizationPass {
    fn default() -> Self {
        Self {
            name: "ZX-Calculus Optimization".to_string(),
            verbose: false,
        }
    }
}

impl ZXOptimizationPass {
    /// Create a new ZX optimization pass
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable verbose output
    #[must_use]
    pub const fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl OptimizationPass for ZXOptimizationPass {
    fn optimize(&self, gates: Vec<Box<dyn GateOp>>) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        if gates.is_empty() {
            return Ok(gates);
        }

        let pipeline = ZXPipeline::new();

        if self.verbose {
            let (original_t, _) = pipeline.compare_t_count(&gates, &gates);
            println!(
                "ZX-Calculus: Processing circuit with {} gates, {} T-gates",
                gates.len(),
                original_t
            );
        }

        let optimized = pipeline.optimize(&gates)?;

        if self.verbose {
            let (original_t, optimized_t) = pipeline.compare_t_count(&gates, &optimized);
            println!(
                "ZX-Calculus: Optimized to {} gates, {} T-gates ({}% reduction)",
                optimized.len(),
                optimized_t,
                if original_t > 0 {
                    ((original_t - optimized_t) * 100) / original_t
                } else {
                    0
                }
            );
        }

        Ok(optimized)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_applicable(&self, gates: &[Box<dyn GateOp>]) -> bool {
        // ZX-calculus is particularly effective for circuits with:
        // 1. Many single-qubit rotations
        // 2. Clifford+T decompositions
        // 3. CNOT and CZ gates

        if gates.is_empty() {
            return false;
        }

        // Check if circuit has supported gates
        let has_supported_gates = gates.iter().any(|g| {
            matches!(
                g.name(),
                "H" | "X" | "Y" | "Z" | "S" | "T" | "RX" | "RY" | "RZ" | "CNOT" | "CZ"
            )
        });

        // Check if circuit would benefit from ZX optimization
        let rotation_count = gates
            .iter()
            .filter(|g| matches!(g.name(), "RX" | "RY" | "RZ" | "T"))
            .count();

        let cnot_count = gates
            .iter()
            .filter(|g| matches!(g.name(), "CNOT" | "CZ"))
            .count();

        // Apply if there are rotations or CNOTs to optimize
        has_supported_gates && (rotation_count > 1 || cnot_count > 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::{multi::*, single::*};
    use crate::qubit::QubitId;

    #[test]
    fn test_zx_optimization_pass() {
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: QubitId(0) }),
            Box::new(RotationZ {
                target: QubitId(0),
                theta: std::f64::consts::PI / 4.0,
            }),
            Box::new(RotationZ {
                target: QubitId(0),
                theta: std::f64::consts::PI / 4.0,
            }),
            Box::new(Hadamard { target: QubitId(0) }),
        ];

        let pass = ZXOptimizationPass::new();
        assert!(pass.is_applicable(&gates));

        let optimized = pass
            .optimize(gates.clone())
            .expect("Failed to optimize circuit with ZX pass");

        // Should optimize the circuit
        assert!(optimized.len() <= gates.len());
    }

    #[test]
    fn test_zx_not_applicable() {
        // Circuit with no supported gates
        let gates: Vec<Box<dyn GateOp>> = vec![];

        let pass = ZXOptimizationPass::new();
        assert!(!pass.is_applicable(&gates));
    }

    #[test]
    fn test_zx_with_optimization_chain() {
        use crate::optimization::OptimizationChain;

        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(RotationZ {
                target: QubitId(0),
                theta: std::f64::consts::PI / 4.0,
            }),
            Box::new(RotationZ {
                target: QubitId(0),
                theta: std::f64::consts::PI / 4.0,
            }),
            Box::new(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            }),
            Box::new(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            }),
        ];

        let chain = OptimizationChain::new().add_pass(Box::new(ZXOptimizationPass::new()));

        let optimized = chain
            .optimize(gates.clone())
            .expect("Failed to optimize circuit with ZX chain");

        // Should optimize both the T gates and CNOT cancellation
        assert!(optimized.len() < gates.len());
    }
}
