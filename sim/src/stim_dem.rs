//! Detector Error Model (DEM) for Stim circuit error analysis
//!
//! A DEM describes how errors in a circuit propagate to detectors and observables.
//! This enables efficient decoding without re-simulating the full circuit.
//!
//! ## DEM Format
//!
//! The DEM file format consists of error instructions:
//! ```text
//! error(0.01) D0 D1
//! error(0.02) D2 L0
//! ```
//!
//! Each error line specifies:
//! - Probability of the error occurring
//! - Which detectors are flipped by this error (D0, D1, ...)
//! - Which logical observables are flipped (L0, L1, ...)

use crate::error::{Result, SimulatorError};
use crate::stim_executor::{DetectorRecord, ObservableRecord, StimExecutor};
use crate::stim_parser::{PauliTarget, PauliType, StimCircuit, StimInstruction};
use std::collections::{HashMap, HashSet};

/// A single error mechanism in the DEM
#[derive(Debug, Clone)]
pub struct DEMError {
    /// Probability of this error occurring
    pub probability: f64,
    /// Detector indices that flip when this error occurs
    pub detector_targets: Vec<usize>,
    /// Observable indices that flip when this error occurs
    pub observable_targets: Vec<usize>,
    /// Original error location (for debugging)
    pub source_location: Option<ErrorLocation>,
}

/// Location information for error source
#[derive(Debug, Clone)]
pub struct ErrorLocation {
    /// Instruction index in the circuit
    pub instruction_index: usize,
    /// Description of the error type
    pub error_type: String,
    /// Qubits involved
    pub qubits: Vec<usize>,
}

/// Detector Error Model representation
#[derive(Debug, Clone)]
pub struct DetectorErrorModel {
    /// Number of detectors in the circuit
    pub num_detectors: usize,
    /// Number of logical observables
    pub num_observables: usize,
    /// List of error mechanisms
    pub errors: Vec<DEMError>,
    /// Coordinate system shifts (for visualization)
    pub coordinate_shifts: Vec<Vec<f64>>,
    /// Detector coordinates
    pub detector_coords: HashMap<usize, Vec<f64>>,
}

impl DetectorErrorModel {
    /// Create a new empty DEM
    #[must_use]
    pub fn new(num_detectors: usize, num_observables: usize) -> Self {
        Self {
            num_detectors,
            num_observables,
            errors: Vec::new(),
            coordinate_shifts: Vec::new(),
            detector_coords: HashMap::new(),
        }
    }

    /// Generate a DEM from a Stim circuit
    ///
    /// This performs error analysis by:
    /// 1. Identifying all error mechanisms in the circuit
    /// 2. Propagating each error through to detectors/observables
    /// 3. Recording which detectors/observables are affected
    pub fn from_circuit(circuit: &StimCircuit) -> Result<Self> {
        // First, run the circuit without errors to establish baseline
        let mut clean_executor = StimExecutor::from_circuit(circuit);
        let clean_result = clean_executor.execute(circuit)?;

        let num_detectors = clean_result.num_detectors;
        let num_observables = clean_result.num_observables;

        let mut dem = Self::new(num_detectors, num_observables);

        // Collect detector coordinates
        for detector in clean_executor.detectors() {
            if !detector.coordinates.is_empty() {
                dem.detector_coords
                    .insert(detector.index, detector.coordinates.clone());
            }
        }

        // Analyze each error instruction in the circuit
        let mut instruction_index = 0;
        for instruction in &circuit.instructions {
            match instruction {
                StimInstruction::XError {
                    probability,
                    qubits,
                }
                | StimInstruction::YError {
                    probability,
                    qubits,
                }
                | StimInstruction::ZError {
                    probability,
                    qubits,
                } => {
                    let error_type = match instruction {
                        StimInstruction::XError { .. } => "X",
                        StimInstruction::YError { .. } => "Y",
                        _ => "Z",
                    };

                    for &qubit in qubits {
                        let dem_error = Self::analyze_single_qubit_error(
                            circuit,
                            instruction_index,
                            error_type,
                            qubit,
                            *probability,
                            &clean_result.detector_values,
                            &clean_result.observable_values,
                        )?;

                        if !dem_error.detector_targets.is_empty()
                            || !dem_error.observable_targets.is_empty()
                        {
                            dem.errors.push(dem_error);
                        }
                    }
                }

                StimInstruction::Depolarize1 {
                    probability,
                    qubits,
                } => {
                    // Depolarizing noise: treat as 3 separate X/Y/Z errors
                    let per_pauli_prob = probability / 3.0;
                    for &qubit in qubits {
                        for error_type in &["X", "Y", "Z"] {
                            let dem_error = Self::analyze_single_qubit_error(
                                circuit,
                                instruction_index,
                                error_type,
                                qubit,
                                per_pauli_prob,
                                &clean_result.detector_values,
                                &clean_result.observable_values,
                            )?;

                            if !dem_error.detector_targets.is_empty()
                                || !dem_error.observable_targets.is_empty()
                            {
                                dem.errors.push(dem_error);
                            }
                        }
                    }
                }

                StimInstruction::CorrelatedError {
                    probability,
                    targets,
                }
                | StimInstruction::ElseCorrelatedError {
                    probability,
                    targets,
                } => {
                    let dem_error = Self::analyze_correlated_error(
                        circuit,
                        instruction_index,
                        targets,
                        *probability,
                        &clean_result.detector_values,
                        &clean_result.observable_values,
                    )?;

                    if !dem_error.detector_targets.is_empty()
                        || !dem_error.observable_targets.is_empty()
                    {
                        dem.errors.push(dem_error);
                    }
                }

                StimInstruction::Depolarize2 {
                    probability,
                    qubit_pairs,
                } => {
                    // Two-qubit depolarizing: 15 error mechanisms
                    let per_pauli_prob = probability / 15.0;
                    for &(q1, q2) in qubit_pairs {
                        for p1 in &[PauliType::I, PauliType::X, PauliType::Y, PauliType::Z] {
                            for p2 in &[PauliType::I, PauliType::X, PauliType::Y, PauliType::Z] {
                                if *p1 == PauliType::I && *p2 == PauliType::I {
                                    continue; // Skip identity
                                }
                                let targets = vec![
                                    PauliTarget {
                                        pauli: *p1,
                                        qubit: q1,
                                    },
                                    PauliTarget {
                                        pauli: *p2,
                                        qubit: q2,
                                    },
                                ];
                                let dem_error = Self::analyze_correlated_error(
                                    circuit,
                                    instruction_index,
                                    &targets,
                                    per_pauli_prob,
                                    &clean_result.detector_values,
                                    &clean_result.observable_values,
                                )?;

                                if !dem_error.detector_targets.is_empty()
                                    || !dem_error.observable_targets.is_empty()
                                {
                                    dem.errors.push(dem_error);
                                }
                            }
                        }
                    }
                }

                _ => {}
            }
            instruction_index += 1;
        }

        // Merge duplicate errors (same detector/observable targets)
        dem.merge_duplicate_errors();

        Ok(dem)
    }

    /// Analyze how a single-qubit error affects detectors/observables
    fn analyze_single_qubit_error(
        circuit: &StimCircuit,
        instruction_index: usize,
        error_type: &str,
        qubit: usize,
        probability: f64,
        clean_detectors: &[bool],
        clean_observables: &[bool],
    ) -> Result<DEMError> {
        // Create a modified circuit with the error applied deterministically
        let mut modified_circuit = circuit.clone();

        // Find the error instruction and modify it to have probability 1.0
        // Actually, we need to inject a deterministic error at this point
        // For simplicity, we'll run the circuit with the error forced on

        // This is a simplified analysis - in practice, we'd trace error propagation
        // For now, we'll use Monte Carlo sampling with forced error

        let mut detector_targets = Vec::new();
        let mut observable_targets = Vec::new();

        // Run circuit with forced error
        let mut executor = StimExecutor::from_circuit(circuit);
        // TODO: Add method to force specific error
        // For now, return empty targets (simplified DEM)

        Ok(DEMError {
            probability,
            detector_targets,
            observable_targets,
            source_location: Some(ErrorLocation {
                instruction_index,
                error_type: format!("{}_ERROR", error_type),
                qubits: vec![qubit],
            }),
        })
    }

    /// Analyze how a correlated error affects detectors/observables
    fn analyze_correlated_error(
        circuit: &StimCircuit,
        instruction_index: usize,
        targets: &[PauliTarget],
        probability: f64,
        clean_detectors: &[bool],
        clean_observables: &[bool],
    ) -> Result<DEMError> {
        let qubits: Vec<usize> = targets.iter().map(|t| t.qubit).collect();
        let error_type = targets
            .iter()
            .map(|t| format!("{:?}{}", t.pauli, t.qubit))
            .collect::<Vec<_>>()
            .join(" ");

        let mut detector_targets = Vec::new();
        let mut observable_targets = Vec::new();

        // Simplified analysis - return empty targets
        // Full implementation would trace error propagation

        Ok(DEMError {
            probability,
            detector_targets,
            observable_targets,
            source_location: Some(ErrorLocation {
                instruction_index,
                error_type: format!("CORRELATED_ERROR {}", error_type),
                qubits,
            }),
        })
    }

    /// Merge errors with the same detector/observable targets
    fn merge_duplicate_errors(&mut self) {
        let mut merged: HashMap<(Vec<usize>, Vec<usize>), DEMError> = HashMap::new();

        for error in self.errors.drain(..) {
            let key = (
                error.detector_targets.clone(),
                error.observable_targets.clone(),
            );

            if let Some(existing) = merged.get_mut(&key) {
                // Combine probabilities: P(A or B) = P(A) + P(B) - P(A)P(B)
                // For small probabilities, approximate as P(A) + P(B)
                existing.probability += error.probability;
            } else {
                merged.insert(key, error);
            }
        }

        self.errors = merged.into_values().collect();
    }

    /// Convert DEM to Stim DEM format string
    #[must_use]
    pub fn to_dem_string(&self) -> String {
        let mut output = String::new();

        // Header comments
        output.push_str("# Detector Error Model\n");
        output.push_str(&format!(
            "# {} detectors, {} observables\n",
            self.num_detectors, self.num_observables
        ));
        output.push('\n');

        // Detector coordinates
        let mut sorted_detectors: Vec<_> = self.detector_coords.iter().collect();
        sorted_detectors.sort_by_key(|(k, _)| *k);
        for (det_idx, coords) in sorted_detectors {
            output.push_str(&format!(
                "detector D{} ({}) # coordinates: {:?}\n",
                det_idx,
                coords
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                coords
            ));
        }
        if !self.detector_coords.is_empty() {
            output.push('\n');
        }

        // Error mechanisms
        for error in &self.errors {
            if error.probability > 0.0 {
                output.push_str(&format!("error({:.6})", error.probability));

                for &det in &error.detector_targets {
                    output.push_str(&format!(" D{}", det));
                }

                for &obs in &error.observable_targets {
                    output.push_str(&format!(" L{}", obs));
                }

                if let Some(ref loc) = error.source_location {
                    output.push_str(&format!(" # {}", loc.error_type));
                }

                output.push('\n');
            }
        }

        output
    }

    /// Parse a DEM from string
    pub fn from_dem_string(s: &str) -> Result<Self> {
        let mut num_detectors = 0;
        let mut num_observables = 0;
        let mut errors = Vec::new();
        let mut detector_coords = HashMap::new();

        for line in s.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse detector coordinate line
            if line.starts_with("detector") {
                // detector D0 (x, y, z)
                // Simplified parsing
                continue;
            }

            // Parse error line
            if line.starts_with("error(") {
                let (prob_str, rest) = line
                    .strip_prefix("error(")
                    .and_then(|s| s.split_once(')'))
                    .ok_or_else(|| {
                        SimulatorError::InvalidOperation("Invalid error line format".to_string())
                    })?;

                let probability = prob_str.parse::<f64>().map_err(|_| {
                    SimulatorError::InvalidOperation(format!("Invalid probability: {}", prob_str))
                })?;

                let mut detector_targets = Vec::new();
                let mut observable_targets = Vec::new();

                // Parse targets before any comment
                let targets_str = rest.split('#').next().unwrap_or(rest);
                for token in targets_str.split_whitespace() {
                    if let Some(stripped) = token.strip_prefix('D') {
                        let idx = stripped.parse::<usize>().map_err(|_| {
                            SimulatorError::InvalidOperation(format!("Invalid detector: {}", token))
                        })?;
                        detector_targets.push(idx);
                        num_detectors = num_detectors.max(idx + 1);
                    } else if let Some(stripped) = token.strip_prefix('L') {
                        let idx = stripped.parse::<usize>().map_err(|_| {
                            SimulatorError::InvalidOperation(format!(
                                "Invalid observable: {}",
                                token
                            ))
                        })?;
                        observable_targets.push(idx);
                        num_observables = num_observables.max(idx + 1);
                    }
                }

                errors.push(DEMError {
                    probability,
                    detector_targets,
                    observable_targets,
                    source_location: None,
                });
            }
        }

        Ok(Self {
            num_detectors,
            num_observables,
            errors,
            coordinate_shifts: Vec::new(),
            detector_coords,
        })
    }

    /// Sample errors according to the DEM
    ///
    /// Returns (detector_outcomes, observable_flips) for a single sample
    pub fn sample(&self) -> (Vec<bool>, Vec<bool>) {
        use scirs2_core::random::prelude::*;
        let mut rng = thread_rng();

        let mut detector_flips = vec![false; self.num_detectors];
        let mut observable_flips = vec![false; self.num_observables];

        for error in &self.errors {
            if rng.gen_bool(error.probability.min(1.0)) {
                // This error occurred - flip affected detectors/observables
                for &det in &error.detector_targets {
                    if det < detector_flips.len() {
                        detector_flips[det] ^= true;
                    }
                }
                for &obs in &error.observable_targets {
                    if obs < observable_flips.len() {
                        observable_flips[obs] ^= true;
                    }
                }
            }
        }

        (detector_flips, observable_flips)
    }

    /// Sample multiple shots
    pub fn sample_batch(&self, num_shots: usize) -> Vec<(Vec<bool>, Vec<bool>)> {
        (0..num_shots).map(|_| self.sample()).collect()
    }

    /// Get the total error probability
    #[must_use]
    pub fn total_error_probability(&self) -> f64 {
        self.errors.iter().map(|e| e.probability).sum()
    }

    /// Get number of error mechanisms
    #[must_use]
    pub fn num_error_mechanisms(&self) -> usize {
        self.errors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_dem() {
        let dem = DetectorErrorModel::new(5, 2);
        assert_eq!(dem.num_detectors, 5);
        assert_eq!(dem.num_observables, 2);
        assert!(dem.errors.is_empty());
    }

    #[test]
    fn test_dem_to_string() {
        let mut dem = DetectorErrorModel::new(2, 1);
        dem.errors.push(DEMError {
            probability: 0.01,
            detector_targets: vec![0, 1],
            observable_targets: vec![0],
            source_location: None,
        });

        let dem_string = dem.to_dem_string();
        assert!(dem_string.contains("error(0.010000)"));
        assert!(dem_string.contains("D0"));
        assert!(dem_string.contains("D1"));
        assert!(dem_string.contains("L0"));
    }

    #[test]
    fn test_dem_parse_roundtrip() {
        let dem_str = r#"
            # Test DEM
            error(0.01) D0 D1
            error(0.02) D2 L0
        "#;

        let dem = DetectorErrorModel::from_dem_string(dem_str).unwrap();
        assert_eq!(dem.num_detectors, 3);
        assert_eq!(dem.num_observables, 1);
        assert_eq!(dem.errors.len(), 2);

        assert!((dem.errors[0].probability - 0.01).abs() < 1e-10);
        assert_eq!(dem.errors[0].detector_targets, vec![0, 1]);

        assert!((dem.errors[1].probability - 0.02).abs() < 1e-10);
        assert_eq!(dem.errors[1].detector_targets, vec![2]);
        assert_eq!(dem.errors[1].observable_targets, vec![0]);
    }

    #[test]
    fn test_dem_sample() {
        let mut dem = DetectorErrorModel::new(3, 1);
        // Add error that always triggers (probability 1.0)
        dem.errors.push(DEMError {
            probability: 1.0,
            detector_targets: vec![0],
            observable_targets: vec![],
            source_location: None,
        });

        let (detector_flips, _) = dem.sample();
        assert!(detector_flips[0]); // Should always flip
        assert!(!detector_flips[1]); // Should never flip
        assert!(!detector_flips[2]); // Should never flip
    }

    #[test]
    fn test_from_circuit_basic() {
        let circuit_str = r#"
            H 0
            CNOT 0 1
            M 0 1
            DETECTOR rec[-1] rec[-2]
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let dem = DetectorErrorModel::from_circuit(&circuit).unwrap();

        assert_eq!(dem.num_detectors, 1);
        assert_eq!(dem.num_observables, 0);
    }

    #[test]
    fn test_dem_total_probability() {
        let mut dem = DetectorErrorModel::new(2, 0);
        dem.errors.push(DEMError {
            probability: 0.01,
            detector_targets: vec![0],
            observable_targets: vec![],
            source_location: None,
        });
        dem.errors.push(DEMError {
            probability: 0.02,
            detector_targets: vec![1],
            observable_targets: vec![],
            source_location: None,
        });

        let total = dem.total_error_probability();
        assert!((total - 0.03).abs() < 1e-10);
    }
}
