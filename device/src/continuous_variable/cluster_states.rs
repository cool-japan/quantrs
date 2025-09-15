//! Cluster states for continuous variable quantum computing
//!
//! This module implements cluster state generation and manipulation for CV systems,
//! enabling measurement-based quantum computing with continuous variables.

use super::{CVDeviceConfig, CVGateSequence, Complex, GaussianState};
use crate::{DeviceError, DeviceResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

/// Types of CV cluster states
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClusterStateType {
    /// Linear cluster (1D chain)
    Linear,
    /// Square lattice cluster (2D)
    Square,
    /// Hexagonal lattice cluster
    Hexagonal,
    /// Custom graph cluster
    Custom { adjacency_matrix: Vec<Vec<bool>> },
}

/// Cluster state configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStateConfig {
    /// Type of cluster state
    pub cluster_type: ClusterStateType,
    /// Number of modes in the cluster
    pub num_modes: usize,
    /// Squeezing parameter for initial squeezed states
    pub squeezing_parameter: f64,
    /// Entangling gate strength
    pub entangling_strength: f64,
    /// Graph structure (adjacency list)
    pub graph_structure: HashMap<usize, Vec<usize>>,
    /// Enable finite squeezing compensation
    pub finite_squeezing_compensation: bool,
}

impl ClusterStateConfig {
    /// Create a linear cluster configuration
    pub fn linear(num_modes: usize, squeezing: f64) -> Self {
        let mut graph_structure = HashMap::new();

        for i in 0..num_modes {
            let mut neighbors = Vec::new();
            if i > 0 {
                neighbors.push(i - 1);
            }
            if i < num_modes - 1 {
                neighbors.push(i + 1);
            }
            graph_structure.insert(i, neighbors);
        }

        Self {
            cluster_type: ClusterStateType::Linear,
            num_modes,
            squeezing_parameter: squeezing,
            entangling_strength: 1.0,
            graph_structure,
            finite_squeezing_compensation: true,
        }
    }

    /// Create a square lattice cluster configuration
    pub fn square_lattice(width: usize, height: usize, squeezing: f64) -> Self {
        let num_modes = width * height;
        let mut graph_structure = HashMap::new();

        for i in 0..height {
            for j in 0..width {
                let node = i * width + j;
                let mut neighbors = Vec::new();

                // Add horizontal neighbors
                if j > 0 {
                    neighbors.push(i * width + (j - 1));
                }
                if j < width - 1 {
                    neighbors.push(i * width + (j + 1));
                }

                // Add vertical neighbors
                if i > 0 {
                    neighbors.push((i - 1) * width + j);
                }
                if i < height - 1 {
                    neighbors.push((i + 1) * width + j);
                }

                graph_structure.insert(node, neighbors);
            }
        }

        Self {
            cluster_type: ClusterStateType::Square,
            num_modes,
            squeezing_parameter: squeezing,
            entangling_strength: 1.0,
            graph_structure,
            finite_squeezing_compensation: true,
        }
    }

    /// Create a custom cluster configuration
    pub fn custom(adjacency_matrix: Vec<Vec<bool>>, squeezing: f64) -> DeviceResult<Self> {
        let num_modes = adjacency_matrix.len();

        // Validate adjacency matrix
        for row in &adjacency_matrix {
            if row.len() != num_modes {
                return Err(DeviceError::InvalidInput(
                    "Adjacency matrix must be square".to_string(),
                ));
            }
        }

        // Build graph structure from adjacency matrix
        let mut graph_structure = HashMap::new();
        for i in 0..num_modes {
            let mut neighbors = Vec::new();
            for j in 0..num_modes {
                if adjacency_matrix[i][j] {
                    neighbors.push(j);
                }
            }
            graph_structure.insert(i, neighbors);
        }

        Ok(Self {
            cluster_type: ClusterStateType::Custom { adjacency_matrix },
            num_modes,
            squeezing_parameter: squeezing,
            entangling_strength: 1.0,
            graph_structure,
            finite_squeezing_compensation: true,
        })
    }
}

/// Cluster state generator
pub struct ClusterStateGenerator {
    /// Configuration
    config: ClusterStateConfig,
    /// Current cluster state
    cluster_state: Option<GaussianState>,
    /// Generation statistics
    generation_stats: GenerationStatistics,
}

/// Statistics for cluster state generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStatistics {
    /// Total entangling gates applied
    pub total_entangling_gates: usize,
    /// Average entanglement per mode
    pub average_entanglement: f64,
    /// Nullifier violations (finite squeezing effects)
    pub nullifier_violations: f64,
    /// Generation fidelity
    pub generation_fidelity: f64,
    /// Time taken for generation (ms)
    pub generation_time_ms: f64,
}

impl Default for GenerationStatistics {
    fn default() -> Self {
        Self {
            total_entangling_gates: 0,
            average_entanglement: 0.0,
            nullifier_violations: 0.0,
            generation_fidelity: 0.0,
            generation_time_ms: 0.0,
        }
    }
}

impl ClusterStateGenerator {
    /// Create a new cluster state generator
    pub fn new(config: ClusterStateConfig) -> Self {
        Self {
            config,
            cluster_state: None,
            generation_stats: GenerationStatistics::default(),
        }
    }

    /// Generate the cluster state
    pub async fn generate_cluster_state(&mut self) -> DeviceResult<GaussianState> {
        let start_time = std::time::Instant::now();

        println!(
            "Generating {} cluster state with {} modes",
            match self.config.cluster_type {
                ClusterStateType::Linear => "linear",
                ClusterStateType::Square => "square lattice",
                ClusterStateType::Hexagonal => "hexagonal",
                ClusterStateType::Custom { .. } => "custom",
            },
            self.config.num_modes
        );

        // Step 1: Initialize with squeezed vacuum states
        let mut state = self.initialize_squeezed_modes().await?;

        // Step 2: Apply entangling gates according to graph structure
        self.apply_entangling_gates(&mut state).await?;

        // Step 3: Apply finite squeezing compensation if enabled
        if self.config.finite_squeezing_compensation {
            self.apply_squeezing_compensation(&mut state).await?;
        }

        // Step 4: Calculate generation statistics
        self.calculate_generation_statistics(&state);

        let generation_time = start_time.elapsed();
        self.generation_stats.generation_time_ms = generation_time.as_millis() as f64;

        println!(
            "Cluster state generated in {:.2} ms with fidelity {:.3}",
            self.generation_stats.generation_time_ms, self.generation_stats.generation_fidelity
        );

        self.cluster_state = Some(state.clone());
        Ok(state)
    }

    /// Initialize all modes with squeezed vacuum states
    async fn initialize_squeezed_modes(&mut self) -> DeviceResult<GaussianState> {
        let squeezing_params = vec![self.config.squeezing_parameter; self.config.num_modes];
        let squeezing_phases = vec![0.0; self.config.num_modes]; // All in x-quadrature

        GaussianState::squeezed_vacuum_state(
            self.config.num_modes,
            squeezing_params,
            squeezing_phases,
        )
        .map_err(|e| DeviceError::InvalidInput(format!("Failed to create squeezed states: {}", e)))
    }

    /// Apply entangling gates according to the graph structure
    async fn apply_entangling_gates(&mut self, state: &mut GaussianState) -> DeviceResult<()> {
        let mut gate_count = 0;

        // Collect all edges to avoid double-counting
        let mut edges = HashSet::new();
        for (node, neighbors) in &self.config.graph_structure {
            for neighbor in neighbors {
                let edge = if node < neighbor {
                    (*node, *neighbor)
                } else {
                    (*neighbor, *node)
                };
                edges.insert(edge);
            }
        }

        println!("Applying {} entangling gates", edges.len());

        // Apply controlled-Z gates for each edge
        for (mode1, mode2) in edges {
            self.apply_cv_cz_gate(state, mode1, mode2).await?;
            gate_count += 1;

            // Simulate some processing time
            if gate_count % 10 == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
        }

        self.generation_stats.total_entangling_gates = gate_count;
        Ok(())
    }

    /// Apply a continuous variable controlled-Z gate
    async fn apply_cv_cz_gate(
        &self,
        state: &mut GaussianState,
        mode1: usize,
        mode2: usize,
    ) -> DeviceResult<()> {
        // CV controlled-Z gate is implemented via position-momentum coupling
        // This creates correlation between x1 and p2, and p1 and x2

        let strength = self.config.entangling_strength;

        // Get current covariance matrix elements
        let old_covar = state.covariancematrix.clone();

        // Apply CZ transformation to covariance matrix
        // CZ gate: exp(i*g*x1*x2) creates correlations

        let i1_x = 2 * mode1; // x quadrature of mode 1
        let i1_p = 2 * mode1 + 1; // p quadrature of mode 1
        let i2_x = 2 * mode2; // x quadrature of mode 2
        let i2_p = 2 * mode2 + 1; // p quadrature of mode 2

        // CZ gate adds correlations between x1-p2 and p1-x2
        state.covariancematrix[i1_x][i2_p] +=
            strength * old_covar[i1_x][i1_x].sqrt() * old_covar[i2_p][i2_p].sqrt();
        state.covariancematrix[i2_p][i1_x] +=
            strength * old_covar[i1_x][i1_x].sqrt() * old_covar[i2_p][i2_p].sqrt();

        state.covariancematrix[i1_p][i2_x] +=
            strength * old_covar[i1_p][i1_p].sqrt() * old_covar[i2_x][i2_x].sqrt();
        state.covariancematrix[i2_x][i1_p] +=
            strength * old_covar[i1_p][i1_p].sqrt() * old_covar[i2_x][i2_x].sqrt();

        Ok(())
    }

    /// Apply finite squeezing compensation
    async fn apply_squeezing_compensation(
        &mut self,
        state: &mut GaussianState,
    ) -> DeviceResult<()> {
        println!("Applying finite squeezing compensation");

        // For finite squeezing, we need to compensate for the fact that perfect
        // cluster states require infinite squeezing. We apply additional local
        // operations to improve the cluster state fidelity.

        for mode in 0..self.config.num_modes {
            // Calculate required compensation based on current variance
            let var_x = state.covariancematrix[2 * mode][2 * mode];
            let compensation_squeezing = if var_x > 0.1 {
                -0.5 * var_x.ln() // Additional squeezing to reduce variance
            } else {
                0.0
            };

            if compensation_squeezing > 0.1 {
                state.apply_squeezing(mode, compensation_squeezing, 0.0)?;
            }
        }

        Ok(())
    }

    /// Calculate generation statistics
    fn calculate_generation_statistics(&mut self, state: &GaussianState) {
        // Calculate average entanglement
        let entanglement_measures = state.calculate_entanglement_measures();
        self.generation_stats.average_entanglement = entanglement_measures.logarithmic_negativity;

        // Calculate nullifier violations (measure of finite squeezing effects)
        let mut total_violation = 0.0;
        for mode in 0..self.config.num_modes {
            let var_x = state.covariancematrix[2 * mode][2 * mode];
            // Ideal cluster state has zero x-variance
            total_violation += var_x;
        }
        self.generation_stats.nullifier_violations = total_violation / self.config.num_modes as f64;

        // Estimate generation fidelity
        self.generation_stats.generation_fidelity = self.estimate_cluster_state_fidelity(state);
    }

    /// Estimate cluster state fidelity
    fn estimate_cluster_state_fidelity(&self, state: &GaussianState) -> f64 {
        // Simplified fidelity calculation based on nullifier violations
        let ideal_variance = 0.0; // Perfect cluster state has zero x-variance
        let actual_variance = self.generation_stats.nullifier_violations;

        // Fidelity decreases with variance
        let variance_penalty = (-actual_variance / 0.5).exp();

        // Account for entanglement quality
        let entanglement_bonus = (self.generation_stats.average_entanglement / 2.0).tanh();

        (variance_penalty * (0.8 + 0.2 * entanglement_bonus)).clamp(0.0, 1.0)
    }

    /// Perform measurement-based computation on the cluster state
    pub async fn perform_mbqc_sequence(
        &mut self,
        measurement_sequence: Vec<MBQCMeasurement>,
    ) -> DeviceResult<Vec<MBQCResult>> {
        if self.cluster_state.is_none() {
            return Err(DeviceError::InvalidInput(
                "No cluster state generated".to_string(),
            ));
        }

        let mut state = self.cluster_state.as_ref().unwrap().clone();
        let mut results = Vec::new();

        println!(
            "Performing MBQC sequence with {} measurements",
            measurement_sequence.len()
        );

        for (i, measurement) in measurement_sequence.iter().enumerate() {
            let result = self
                .perform_single_mbqc_measurement(&mut state, measurement)
                .await?;
            results.push(result);

            println!(
                "Completed measurement {} of {}",
                i + 1,
                measurement_sequence.len()
            );
        }

        Ok(results)
    }

    /// Perform a single MBQC measurement
    async fn perform_single_mbqc_measurement(
        &self,
        state: &mut GaussianState,
        measurement: &MBQCMeasurement,
    ) -> DeviceResult<MBQCResult> {
        match measurement.measurement_type {
            MBQCMeasurementType::Homodyne { phase } => {
                // Create a simplified measurement config
                let config = CVDeviceConfig::default();
                let measured_value =
                    state.homodyne_measurement(measurement.mode, phase, &config)?;

                Ok(MBQCResult {
                    mode: measurement.mode,
                    measurement_type: measurement.measurement_type.clone(),
                    outcome: measured_value,
                    feedforward_corrections: self
                        .calculate_feedforward(measurement, measured_value),
                    success_probability: 1.0, // Homodyne always succeeds
                })
            }

            MBQCMeasurementType::Projection { target_value } => {
                // Simplified projection measurement
                let config = CVDeviceConfig::default();
                let measured_value = state.homodyne_measurement(measurement.mode, 0.0, &config)?;
                let success_prob =
                    self.calculate_projection_success_probability(measured_value, target_value);

                Ok(MBQCResult {
                    mode: measurement.mode,
                    measurement_type: measurement.measurement_type.clone(),
                    outcome: measured_value,
                    feedforward_corrections: Vec::new(),
                    success_probability: success_prob,
                })
            }
        }
    }

    /// Calculate feedforward corrections
    fn calculate_feedforward(
        &self,
        measurement: &MBQCMeasurement,
        outcome: f64,
    ) -> Vec<FeedforwardCorrection> {
        let mut corrections = Vec::new();

        // Apply feedforward to neighboring modes
        if let Some(neighbors) = self.config.graph_structure.get(&measurement.mode) {
            for &neighbor in neighbors {
                // Simplified feedforward: phase correction proportional to outcome
                let correction_phase = outcome * 0.1; // Simplified calculation
                corrections.push(FeedforwardCorrection {
                    target_mode: neighbor,
                    correction_type: CorrectionType::PhaseShift {
                        phase: correction_phase,
                    },
                });
            }
        }

        corrections
    }

    /// Calculate projection measurement success probability
    fn calculate_projection_success_probability(&self, measured: f64, target: f64) -> f64 {
        let tolerance = 0.5; // Tolerance for projection success
        let deviation = (measured - target).abs();

        if deviation <= tolerance {
            1.0 - deviation / tolerance
        } else {
            0.0
        }
    }

    /// Get current cluster state
    pub fn get_cluster_state(&self) -> Option<&GaussianState> {
        self.cluster_state.as_ref()
    }

    /// Get generation statistics
    pub fn get_generation_statistics(&self) -> &GenerationStatistics {
        &self.generation_stats
    }

    /// Validate cluster state properties
    pub fn validate_cluster_state(&self) -> DeviceResult<ClusterStateValidation> {
        if let Some(state) = &self.cluster_state {
            let mut validation = ClusterStateValidation::default();

            // Check nullifier properties
            validation.nullifier_eigenvalues = self.calculate_nullifier_eigenvalues(state);
            validation.max_nullifier_violation = validation
                .nullifier_eigenvalues
                .iter()
                .fold(0.0, |acc, &x| acc.max(x));

            // Check entanglement structure
            validation.entanglement_spectrum = self.calculate_entanglement_spectrum(state);
            validation.average_entanglement = validation.entanglement_spectrum.iter().sum::<f64>()
                / validation.entanglement_spectrum.len() as f64;

            // Overall validation
            validation.is_valid =
                validation.max_nullifier_violation < 0.1 && validation.average_entanglement > 0.5;

            Ok(validation)
        } else {
            Err(DeviceError::InvalidInput(
                "No cluster state to validate".to_string(),
            ))
        }
    }

    /// Calculate nullifier eigenvalues
    fn calculate_nullifier_eigenvalues(&self, state: &GaussianState) -> Vec<f64> {
        // Simplified calculation of nullifier operator eigenvalues
        let mut eigenvalues = Vec::new();

        for mode in 0..self.config.num_modes {
            let var_x = state.covariancematrix[2 * mode][2 * mode];
            eigenvalues.push(var_x);
        }

        eigenvalues
    }

    /// Calculate entanglement spectrum
    fn calculate_entanglement_spectrum(&self, state: &GaussianState) -> Vec<f64> {
        // Simplified entanglement spectrum calculation
        let mut spectrum = Vec::new();

        // Calculate bipartite entanglement for each possible bipartition
        for i in 0..self.config.num_modes {
            for j in (i + 1)..self.config.num_modes {
                let cov_ij = state.covariancematrix[2 * i][2 * j];
                let entanglement = cov_ij.abs(); // Simplified measure
                spectrum.push(entanglement);
            }
        }

        spectrum
    }
}

/// MBQC measurement specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MBQCMeasurement {
    /// Mode to measure
    pub mode: usize,
    /// Type of measurement
    pub measurement_type: MBQCMeasurementType,
    /// Adaptive measurement (depends on previous results)
    pub is_adaptive: bool,
}

/// Types of MBQC measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MBQCMeasurementType {
    /// Homodyne measurement at specific phase
    Homodyne { phase: f64 },
    /// Projection onto specific value
    Projection { target_value: f64 },
}

/// Result of an MBQC measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MBQCResult {
    /// Measured mode
    pub mode: usize,
    /// Type of measurement performed
    pub measurement_type: MBQCMeasurementType,
    /// Measurement outcome
    pub outcome: f64,
    /// Required feedforward corrections
    pub feedforward_corrections: Vec<FeedforwardCorrection>,
    /// Success probability (for projection measurements)
    pub success_probability: f64,
}

/// Feedforward correction specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedforwardCorrection {
    /// Target mode for correction
    pub target_mode: usize,
    /// Type of correction
    pub correction_type: CorrectionType,
}

/// Types of feedforward corrections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrectionType {
    /// Phase shift correction
    PhaseShift { phase: f64 },
    /// Displacement correction
    Displacement { amplitude: Complex },
    /// Squeezing correction
    Squeezing { parameter: f64, phase: f64 },
}

/// Cluster state validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStateValidation {
    /// Whether the cluster state is valid
    pub is_valid: bool,
    /// Nullifier operator eigenvalues
    pub nullifier_eigenvalues: Vec<f64>,
    /// Maximum nullifier violation
    pub max_nullifier_violation: f64,
    /// Entanglement spectrum
    pub entanglement_spectrum: Vec<f64>,
    /// Average entanglement
    pub average_entanglement: f64,
}

impl Default for ClusterStateValidation {
    fn default() -> Self {
        Self {
            is_valid: false,
            nullifier_eigenvalues: Vec::new(),
            max_nullifier_violation: 0.0,
            entanglement_spectrum: Vec::new(),
            average_entanglement: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_cluster_config() {
        let config = ClusterStateConfig::linear(5, 1.0);
        assert_eq!(config.num_modes, 5);
        assert_eq!(config.cluster_type, ClusterStateType::Linear);
        assert_eq!(config.graph_structure.len(), 5);

        // Check linear connectivity
        assert_eq!(config.graph_structure[&0], vec![1]);
        assert_eq!(config.graph_structure[&2], vec![1, 3]);
        assert_eq!(config.graph_structure[&4], vec![3]);
    }

    #[test]
    fn test_square_lattice_config() {
        let config = ClusterStateConfig::square_lattice(3, 3, 1.0);
        assert_eq!(config.num_modes, 9);
        assert_eq!(config.cluster_type, ClusterStateType::Square);

        // Check corner node (should have 2 neighbors)
        assert_eq!(config.graph_structure[&0].len(), 2);
        // Check center node (should have 4 neighbors)
        assert_eq!(config.graph_structure[&4].len(), 4);
    }

    #[test]
    fn test_custom_cluster_config() {
        let adjacency = vec![
            vec![false, true, false],
            vec![true, false, true],
            vec![false, true, false],
        ];

        let config = ClusterStateConfig::custom(adjacency, 1.0).unwrap();
        assert_eq!(config.num_modes, 3);

        // Check custom connectivity
        assert_eq!(config.graph_structure[&0], vec![1]);
        assert_eq!(config.graph_structure[&1], vec![0, 2]);
        assert_eq!(config.graph_structure[&2], vec![1]);
    }

    #[tokio::test]
    async fn test_cluster_state_generation() {
        let config = ClusterStateConfig::linear(3, 1.0);
        let mut generator = ClusterStateGenerator::new(config);

        let state = generator.generate_cluster_state().await.unwrap();
        assert_eq!(state.num_modes, 3);

        let stats = generator.get_generation_statistics();
        assert!(stats.total_entangling_gates > 0);
        assert!(stats.generation_fidelity > 0.0);
    }

    #[tokio::test]
    async fn test_mbqc_measurement() {
        let config = ClusterStateConfig::linear(3, 1.0);
        let mut generator = ClusterStateGenerator::new(config);
        generator.generate_cluster_state().await.unwrap();

        let measurements = vec![MBQCMeasurement {
            mode: 0,
            measurement_type: MBQCMeasurementType::Homodyne { phase: 0.0 },
            is_adaptive: false,
        }];

        let results = generator.perform_mbqc_sequence(measurements).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].success_probability, 1.0);
    }

    #[test]
    fn test_cluster_validation() {
        let config = ClusterStateConfig::linear(2, 1.0);
        let generator = ClusterStateGenerator::new(config);

        // Should fail validation without generated state
        assert!(generator.validate_cluster_state().is_err());
    }

    #[test]
    fn test_feedforward_calculation() {
        let config = ClusterStateConfig::linear(3, 1.0);
        let generator = ClusterStateGenerator::new(config);

        let measurement = MBQCMeasurement {
            mode: 1,
            measurement_type: MBQCMeasurementType::Homodyne { phase: 0.0 },
            is_adaptive: false,
        };

        let corrections = generator.calculate_feedforward(&measurement, 1.0);
        assert_eq!(corrections.len(), 2); // Mode 1 has 2 neighbors in linear chain
    }
}
