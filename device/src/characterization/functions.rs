//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{CircuitResult, DeviceResult};
use quantrs2_circuit::prelude::*;
use quantrs2_core::qubit::QubitId;

use super::types::{DriftTracker, ProcessTomography, RandomizedBenchmarking, StateTomography};

/// Trait for executing characterization circuits
#[async_trait::async_trait]
pub trait CharacterizationExecutor {
    async fn execute_characterization_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        shots: usize,
    ) -> DeviceResult<CircuitResult>;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_process_tomography_circuits() {
        let tomo = ProcessTomography::new(1);
        let prep_circuits = tomo.preparation_circuits();
        let meas_circuits = tomo.measurement_circuits();
        assert_eq!(prep_circuits.len(), 4);
        assert_eq!(meas_circuits.len(), 4);
    }
    #[test]
    fn test_state_tomography_circuits() {
        let tomo = StateTomography::new(2);
        let circuits = tomo.measurement_circuits();
        assert_eq!(circuits.len(), 9);
    }
    #[test]
    #[ignore = "Skipping randomized benchmarking test"]
    fn test_randomized_benchmarking() {
        let rb = RandomizedBenchmarking::new(vec![QubitId::new(0)]);
        let sequence = rb.generate_clifford_sequence(10);
        assert!(!sequence.is_empty());
    }
    #[test]
    fn test_drift_tracking() {
        let mut tracker = DriftTracker::new(vec!["T1".to_string()]);
        for i in 0..20 {
            let value = 50.0 + (i as f64) * 0.1;
            tracker.add_measurement("T1", i as f64, value);
        }
        let drift = tracker.detect_drift("T1", 5);
        assert!(drift.is_some());
        assert!(drift.expect("drift should be Some") > 0.0);
    }
}
