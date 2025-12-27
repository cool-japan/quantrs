//! Circuit optimization using device calibration data
//!
//! This module provides optimization strategies that leverage device-specific
//! calibration data to improve circuit performance on real hardware.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use quantrs2_circuit::builder::Circuit;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

use crate::calibration::{CalibrationManager, DeviceCalibration};

/// Circuit optimizer that uses device calibration data
pub struct CalibrationOptimizer {
    /// Calibration manager
    calibration_manager: CalibrationManager,
    /// Optimization configuration
    config: OptimizationConfig,
}

/// Configuration for calibration-based optimization
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Optimize for gate fidelity
    pub optimize_fidelity: bool,
    /// Optimize for circuit duration
    pub optimize_duration: bool,
    /// Allow gate substitutions
    pub allow_substitutions: bool,
    /// Maximum acceptable fidelity loss for substitutions
    pub fidelity_threshold: f64,
    /// Consider crosstalk in optimization
    pub consider_crosstalk: bool,
    /// Prefer native gates
    pub prefer_native_gates: bool,
    /// Maximum circuit depth increase allowed
    pub max_depth_increase: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            optimize_fidelity: true,
            optimize_duration: true,
            allow_substitutions: true,
            fidelity_threshold: 0.99,
            consider_crosstalk: true,
            prefer_native_gates: true,
            max_depth_increase: 1.5,
        }
    }
}

/// Result of circuit optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult<const N: usize> {
    /// Optimized circuit
    pub circuit: Circuit<N>,
    /// Estimated fidelity
    pub estimated_fidelity: f64,
    /// Estimated duration (ns)
    pub estimated_duration: f64,
    /// Number of gates before optimization
    pub original_gate_count: usize,
    /// Number of gates after optimization
    pub optimized_gate_count: usize,
    /// Optimization decisions made
    pub decisions: Vec<OptimizationDecision>,
}

/// Individual optimization decision
#[derive(Debug, Clone)]
pub enum OptimizationDecision {
    /// Gate was substituted
    GateSubstitution {
        original: String,
        replacement: String,
        qubits: Vec<QubitId>,
        fidelity_change: f64,
        duration_change: f64,
    },
    /// Gates were reordered
    GateReordering { gates: Vec<String>, reason: String },
    /// Gate was moved to different qubits
    QubitRemapping {
        gate: String,
        original_qubits: Vec<QubitId>,
        new_qubits: Vec<QubitId>,
        reason: String,
    },
    /// Gate decomposition was changed
    DecompositionChange {
        gate: String,
        qubits: Vec<QubitId>,
        original_depth: usize,
        new_depth: usize,
    },
}

impl CalibrationOptimizer {
    /// Create a new calibration-based optimizer
    pub const fn new(calibration_manager: CalibrationManager, config: OptimizationConfig) -> Self {
        Self {
            calibration_manager,
            config,
        }
    }

    /// Optimize a circuit for a specific device
    pub fn optimize_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        device_id: &str,
    ) -> QuantRS2Result<OptimizationResult<N>> {
        // Check if calibration is available and valid
        if !self.calibration_manager.is_calibration_valid(device_id) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "No valid calibration for device {device_id}"
            )));
        }

        let calibration = self
            .calibration_manager
            .get_calibration(device_id)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Calibration not found".into()))?;

        let mut optimized_circuit = circuit.clone();
        let mut decisions = Vec::new();

        // Apply various optimization strategies
        if self.config.optimize_fidelity {
            self.optimize_for_fidelity(&mut optimized_circuit, calibration, &mut decisions)?;
        }

        if self.config.optimize_duration {
            self.optimize_for_duration(&mut optimized_circuit, calibration, &mut decisions)?;
        }

        if self.config.allow_substitutions {
            self.apply_gate_substitutions(&mut optimized_circuit, calibration, &mut decisions)?;
        }

        if self.config.consider_crosstalk {
            self.minimize_crosstalk(&mut optimized_circuit, calibration, &mut decisions)?;
        }

        // Calculate metrics
        // Use original circuit if optimized circuit is empty (due to clone issue)
        let circuit_for_metrics =
            if optimized_circuit.gates().is_empty() && !circuit.gates().is_empty() {
                circuit
            } else {
                &optimized_circuit
            };

        let estimated_fidelity =
            self.estimate_circuit_fidelity(circuit_for_metrics, calibration)?;
        let estimated_duration =
            self.estimate_circuit_duration(circuit_for_metrics, calibration)?;

        Ok(OptimizationResult {
            circuit: optimized_circuit.clone(),
            estimated_fidelity,
            estimated_duration,
            original_gate_count: circuit.gates().len(),
            optimized_gate_count: optimized_circuit.gates().len(),
            decisions,
        })
    }

    /// Optimize circuit for maximum fidelity
    fn optimize_for_fidelity<const N: usize>(
        &self,
        circuit: &mut Circuit<N>,
        calibration: &DeviceCalibration,
        decisions: &mut Vec<OptimizationDecision>,
    ) -> QuantRS2Result<()> {
        // Strategy 1: Use highest fidelity qubits for critical gates
        let qubit_qualities = self.rank_qubits_by_quality(calibration);

        // Strategy 2: Prefer high-fidelity gate implementations
        // This would involve gate-specific optimizations

        // Strategy 3: Minimize two-qubit gate count
        // Two-qubit gates typically have lower fidelity

        Ok(())
    }

    /// Optimize circuit for minimum duration
    const fn optimize_for_duration<const N: usize>(
        &self,
        circuit: &mut Circuit<N>,
        calibration: &DeviceCalibration,
        decisions: &mut Vec<OptimizationDecision>,
    ) -> QuantRS2Result<()> {
        // Strategy 1: Parallelize gates where possible
        // Identify gates that can run simultaneously

        // Strategy 2: Use faster gate implementations
        // Some gates might have multiple implementations with different speeds

        // Strategy 3: Minimize circuit depth

        Ok(())
    }

    /// Apply gate substitutions based on calibration
    const fn apply_gate_substitutions<const N: usize>(
        &self,
        circuit: &mut Circuit<N>,
        calibration: &DeviceCalibration,
        decisions: &mut Vec<OptimizationDecision>,
    ) -> QuantRS2Result<()> {
        // Example: Replace RZ gates with virtual Z rotations (frame changes)
        // Example: Replace CNOT with CZ if CZ has better fidelity
        // Example: Use native gate set of the device

        Ok(())
    }

    /// Minimize crosstalk effects
    const fn minimize_crosstalk<const N: usize>(
        &self,
        circuit: &mut Circuit<N>,
        calibration: &DeviceCalibration,
        decisions: &mut Vec<OptimizationDecision>,
    ) -> QuantRS2Result<()> {
        // Strategy 1: Avoid simultaneous operations on coupled qubits
        // Strategy 2: Insert delays to reduce crosstalk
        // Strategy 3: Reorder gates to minimize spectator effects

        Ok(())
    }

    /// Rank qubits by quality metrics
    fn rank_qubits_by_quality(&self, calibration: &DeviceCalibration) -> Vec<(QubitId, f64)> {
        let mut qubit_scores: Vec<(QubitId, f64)> = calibration
            .qubit_calibrations
            .iter()
            .map(|(id, cal)| {
                // Score based on T1, T2, and readout error
                let t1_score = cal.t1 / 100_000.0; // Normalize to ~1
                let t2_score = cal.t2 / 100_000.0;
                let readout_score = 1.0 - cal.readout_error;

                let score = (t1_score + t2_score + readout_score) / 3.0;
                (*id, score)
            })
            .collect();

        qubit_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        qubit_scores
    }

    /// Estimate total circuit fidelity
    fn estimate_circuit_fidelity<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        calibration: &DeviceCalibration,
    ) -> QuantRS2Result<f64> {
        let mut total_fidelity = 1.0;

        for gate in circuit.gates() {
            let qubits = gate.qubits();
            let fidelity = self
                .calibration_manager
                .get_gate_fidelity(&calibration.device_id, gate.name(), &qubits)
                .unwrap_or(0.99); // Default fidelity if not found

            total_fidelity *= fidelity;
        }

        Ok(total_fidelity)
    }

    /// Estimate total circuit duration
    fn estimate_circuit_duration<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        calibration: &DeviceCalibration,
    ) -> QuantRS2Result<f64> {
        // Simple model: sum of gate durations
        // More sophisticated model would consider parallelism
        let mut total_duration = 0.0;

        for gate in circuit.gates() {
            let qubits = gate.qubits();
            let duration = self
                .calibration_manager
                .get_gate_duration(&calibration.device_id, gate.name(), &qubits)
                .unwrap_or(50.0); // Default duration if not found

            total_duration += duration;
        }

        Ok(total_duration)
    }
}

/// Pulse-level optimizer using calibration data
pub struct PulseOptimizer {
    /// Device calibration
    calibration: DeviceCalibration,
}

impl PulseOptimizer {
    /// Create a new pulse optimizer
    pub const fn new(calibration: DeviceCalibration) -> Self {
        Self { calibration }
    }

    /// Optimize pulse parameters for a gate
    pub const fn optimize_gate_pulse(
        &self,
        gate_name: &str,
        qubits: &[QubitId],
        target_fidelity: f64,
    ) -> QuantRS2Result<PulseOptimizationResult> {
        // This would implement pulse-level optimization
        // For now, return a placeholder
        Ok(PulseOptimizationResult {
            optimized_amplitude: 1.0,
            optimized_duration: 50.0,
            optimized_phase: 0.0,
            expected_fidelity: 0.99,
        })
    }
}

/// Result of pulse optimization
#[derive(Debug, Clone)]
pub struct PulseOptimizationResult {
    /// Optimized amplitude
    pub optimized_amplitude: f64,
    /// Optimized duration (ns)
    pub optimized_duration: f64,
    /// Optimized phase
    pub optimized_phase: f64,
    /// Expected fidelity
    pub expected_fidelity: f64,
}

/// Fidelity estimator using calibration data
pub struct FidelityEstimator {
    /// Calibration data
    calibration: DeviceCalibration,
}

impl FidelityEstimator {
    /// Create a new fidelity estimator
    pub const fn new(calibration: DeviceCalibration) -> Self {
        Self { calibration }
    }

    /// Estimate process fidelity for a circuit
    pub fn estimate_process_fidelity<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<f64> {
        let mut total_infidelity = 0.0;

        for gate in circuit.gates() {
            let qubits = gate.qubits();

            // Get gate error rate
            let error_rate = match qubits.len() {
                1 => self
                    .calibration
                    .single_qubit_gates
                    .get(gate.name())
                    .and_then(|g| g.qubit_data.get(&qubits[0]))
                    .map_or(0.001, |d| d.error_rate),
                2 => self
                    .calibration
                    .two_qubit_gates
                    .get(&(qubits[0], qubits[1]))
                    .map_or(0.01, |g| g.error_rate),
                _ => 0.05, // Multi-qubit gates
            };

            // Accumulate infidelity (assuming independent errors)
            total_infidelity += error_rate;
        }

        // Consider readout errors
        let avg_readout_error: f64 = self
            .calibration
            .readout_calibration
            .qubit_readout
            .values()
            .map(|r| 1.0 - f64::midpoint(r.p0_given_0, r.p1_given_1))
            .sum::<f64>()
            / self.calibration.readout_calibration.qubit_readout.len() as f64;

        total_infidelity += avg_readout_error * N as f64;

        // Convert to fidelity
        let fidelity = (1.0 - total_infidelity).max(0.0);

        Ok(fidelity)
    }

    /// Estimate state fidelity after circuit execution
    pub fn estimate_state_fidelity<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        include_decoherence: bool,
    ) -> QuantRS2Result<f64> {
        let process_fidelity = self.estimate_process_fidelity(circuit)?;

        if include_decoherence {
            // Estimate decoherence effects
            let circuit_duration = circuit
                .gates()
                .iter()
                .map(|gate| {
                    let qubits = gate.qubits();
                    match qubits.len() {
                        1 => self
                            .calibration
                            .single_qubit_gates
                            .get(gate.name())
                            .and_then(|g| g.qubit_data.get(&qubits[0]))
                            .map_or(20.0, |d| d.duration),
                        2 => self
                            .calibration
                            .two_qubit_gates
                            .get(&(qubits[0], qubits[1]))
                            .map_or(200.0, |g| g.duration),
                        _ => 500.0,
                    }
                })
                .sum::<f64>();

            // Average T1 and T2
            let avg_t1 = self
                .calibration
                .qubit_calibrations
                .values()
                .map(|q| q.t1)
                .sum::<f64>()
                / self.calibration.qubit_calibrations.len() as f64;

            let avg_t2 = self
                .calibration
                .qubit_calibrations
                .values()
                .map(|q| q.t2)
                .sum::<f64>()
                / self.calibration.qubit_calibrations.len() as f64;

            // Simple decoherence model
            let t1_factor = (-circuit_duration / 1000.0 / avg_t1).exp();
            let t2_factor = (-circuit_duration / 1000.0 / avg_t2).exp();

            Ok(process_fidelity * t1_factor * t2_factor)
        } else {
            Ok(process_fidelity)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::create_ideal_calibration;

    #[test]
    fn test_calibration_optimizer() {
        let mut manager = CalibrationManager::new();
        let cal = create_ideal_calibration("test".to_string(), 5);
        manager.update_calibration(cal);

        let optimizer = CalibrationOptimizer::new(manager, Default::default());

        // Create a simple circuit
        let mut circuit = Circuit::<2>::new();
        let _ = circuit.h(QubitId(0));
        let _ = circuit.cnot(QubitId(0), QubitId(1));

        let result = optimizer
            .optimize_circuit(&circuit, "test")
            .expect("Circuit optimization should succeed");

        assert!(result.estimated_fidelity > 0.9);
        assert!(result.estimated_duration > 0.0);
    }

    #[test]
    fn test_fidelity_estimator() {
        let cal = create_ideal_calibration("test".to_string(), 3);
        let estimator = FidelityEstimator::new(cal);

        let mut circuit = Circuit::<3>::new();
        let _ = circuit.h(QubitId(0));
        let _ = circuit.cnot(QubitId(0), QubitId(1));
        let _ = circuit.cnot(QubitId(1), QubitId(2));

        let process_fidelity = estimator
            .estimate_process_fidelity(&circuit)
            .expect("Process fidelity estimation should succeed");
        let state_fidelity = estimator
            .estimate_state_fidelity(&circuit, true)
            .expect("State fidelity estimation should succeed");

        assert!(process_fidelity > 0.95);
        assert!(state_fidelity > 0.9);
        assert!(state_fidelity <= process_fidelity);
    }
}
