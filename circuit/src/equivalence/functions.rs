//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::builder::Circuit;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{
        multi::{CRX, CRY, CRZ},
        single::{RotationX, RotationY, RotationZ},
        GateOp,
    },
    qubit::QubitId,
};

use super::types::{EquivalenceChecker, EquivalenceOptions, EquivalenceResult};

/// Default tolerance for numerical comparisons
pub(crate) const DEFAULT_TOLERANCE: f64 = 1e-10;
/// Enhanced tolerance with `SciRS2` statistical analysis
pub(crate) const SCIRS2_DEFAULT_TOLERANCE: f64 = 1e-12;
/// Tolerance for complex number comparisons
pub(crate) const COMPLEX_TOLERANCE: f64 = 1e-14;
/// Quick check if two circuits are structurally identical
#[must_use]
pub fn circuits_structurally_equal<const N: usize>(
    circuit1: &Circuit<N>,
    circuit2: &Circuit<N>,
) -> bool {
    let checker = EquivalenceChecker::default();
    checker
        .check_structural_equivalence(circuit1, circuit2)
        .map(|result| result.equivalent)
        .unwrap_or(false)
}
/// Quick check if two circuits are equivalent (using default options)
pub fn circuits_equivalent<const N: usize>(
    circuit1: &Circuit<N>,
    circuit2: &Circuit<N>,
) -> QuantRS2Result<bool> {
    let mut checker = EquivalenceChecker::default();
    Ok(checker.check_equivalence(circuit1, circuit2)?.equivalent)
}
/// Check equivalence using `SciRS2` numerical analysis with custom tolerance
pub fn circuits_scirs2_equivalent<const N: usize>(
    circuit1: &Circuit<N>,
    circuit2: &Circuit<N>,
    options: EquivalenceOptions,
) -> QuantRS2Result<EquivalenceResult> {
    let mut checker = EquivalenceChecker::new(options);
    checker.check_equivalence(circuit1, circuit2)
}
/// Quick `SciRS2` numerical equivalence check with default enhanced options
pub fn circuits_scirs2_numerical_equivalent<const N: usize>(
    circuit1: &Circuit<N>,
    circuit2: &Circuit<N>,
) -> QuantRS2Result<EquivalenceResult> {
    let options = EquivalenceOptions {
        enable_adaptive_tolerance: true,
        enable_statistical_analysis: true,
        enable_stability_analysis: true,
        ..Default::default()
    };
    let mut checker = EquivalenceChecker::new(options);
    checker.check_scirs2_numerical_equivalence(circuit1, circuit2)
}
#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::{Hadamard, PauliX, PauliZ};
    #[test]
    fn test_structural_equivalence() {
        let mut circuit1 = Circuit::<2>::new();
        circuit1
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate to circuit1");
        circuit1
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("Failed to add CNOT gate to circuit1");
        let mut circuit2 = Circuit::<2>::new();
        circuit2
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate to circuit2");
        circuit2
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("Failed to add CNOT gate to circuit2");
        let checker = EquivalenceChecker::default();
        let result = checker
            .check_structural_equivalence(&circuit1, &circuit2)
            .expect("Structural equivalence check failed");
        assert!(result.equivalent);
    }
    #[test]
    fn test_structural_non_equivalence() {
        let mut circuit1 = Circuit::<2>::new();
        circuit1
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate to circuit1");
        circuit1
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("Failed to add CNOT gate to circuit1");
        let mut circuit2 = Circuit::<2>::new();
        circuit2
            .add_gate(PauliX { target: QubitId(0) })
            .expect("Failed to add PauliX gate to circuit2");
        circuit2
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("Failed to add CNOT gate to circuit2");
        let checker = EquivalenceChecker::default();
        let result = checker
            .check_structural_equivalence(&circuit1, &circuit2)
            .expect("Structural equivalence check failed");
        assert!(!result.equivalent);
    }
    #[test]
    fn test_different_gate_count() {
        let mut circuit1 = Circuit::<2>::new();
        circuit1
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate to circuit1");
        let mut circuit2 = Circuit::<2>::new();
        circuit2
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate to circuit2");
        circuit2
            .add_gate(PauliZ { target: QubitId(0) })
            .expect("Failed to add PauliZ gate to circuit2");
        let checker = EquivalenceChecker::default();
        let result = checker
            .check_structural_equivalence(&circuit1, &circuit2)
            .expect("Structural equivalence check failed");
        assert!(!result.equivalent);
        assert!(result.details.contains("Different number of gates"));
    }
    #[test]
    fn test_parametric_gate_equivalence_equal() {
        let mut circuit1 = Circuit::<1>::new();
        circuit1
            .add_gate(RotationX {
                target: QubitId(0),
                theta: std::f64::consts::PI / 4.0,
            })
            .expect("Failed to add RotationX gate to circuit1");
        let mut circuit2 = Circuit::<1>::new();
        circuit2
            .add_gate(RotationX {
                target: QubitId(0),
                theta: std::f64::consts::PI / 4.0,
            })
            .expect("Failed to add RotationX gate to circuit2");
        let checker = EquivalenceChecker::default();
        let result = checker
            .check_structural_equivalence(&circuit1, &circuit2)
            .expect("Structural equivalence check failed");
        assert!(result.equivalent);
    }
    #[test]
    fn test_parametric_gate_equivalence_different_params() {
        let mut circuit1 = Circuit::<1>::new();
        circuit1
            .add_gate(RotationX {
                target: QubitId(0),
                theta: std::f64::consts::PI / 4.0,
            })
            .expect("Failed to add RotationX gate to circuit1");
        let mut circuit2 = Circuit::<1>::new();
        circuit2
            .add_gate(RotationX {
                target: QubitId(0),
                theta: std::f64::consts::PI / 2.0,
            })
            .expect("Failed to add RotationX gate to circuit2");
        let checker = EquivalenceChecker::default();
        let result = checker
            .check_structural_equivalence(&circuit1, &circuit2)
            .expect("Structural equivalence check failed");
        assert!(!result.equivalent);
    }
    #[test]
    fn test_parametric_gate_numerical_tolerance() {
        let mut circuit1 = Circuit::<1>::new();
        circuit1
            .add_gate(RotationY {
                target: QubitId(0),
                theta: 1.0,
            })
            .expect("Failed to add RotationY gate to circuit1");
        let mut circuit2 = Circuit::<1>::new();
        circuit2
            .add_gate(RotationY {
                target: QubitId(0),
                theta: 1.0 + 1e-12,
            })
            .expect("Failed to add RotationY gate to circuit2");
        let checker = EquivalenceChecker::default();
        let result = checker
            .check_structural_equivalence(&circuit1, &circuit2)
            .expect("Structural equivalence check failed");
        assert!(result.equivalent);
    }
    #[test]
    fn test_controlled_rotation_equivalence() {
        let mut circuit1 = Circuit::<2>::new();
        circuit1
            .add_gate(CRZ {
                control: QubitId(0),
                target: QubitId(1),
                theta: std::f64::consts::PI,
            })
            .expect("Failed to add CRZ gate to circuit1");
        let mut circuit2 = Circuit::<2>::new();
        circuit2
            .add_gate(CRZ {
                control: QubitId(0),
                target: QubitId(1),
                theta: std::f64::consts::PI,
            })
            .expect("Failed to add CRZ gate to circuit2");
        let checker = EquivalenceChecker::default();
        let result = checker
            .check_structural_equivalence(&circuit1, &circuit2)
            .expect("Structural equivalence check failed");
        assert!(result.equivalent);
    }
}
