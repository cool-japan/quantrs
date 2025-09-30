//! Energy landscape visualization for quantum annealing problems
//!
//! This module provides tools for visualizing the energy landscapes
//! of Ising models and QUBO problems to understand problem difficulty
//! and solution quality.

use scirs2_core::random::prelude::*;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::random::ChaCha8Rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use thiserror::Error;

use crate::ising::{IsingError, IsingModel};
use crate::simulator::{AnnealingParams, AnnealingSolution, QuantumAnnealingSimulator};

/// Errors that can occur during visualization
#[derive(Error, Debug)]
pub enum VisualizationError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Result type for visualization operations
pub type VisualizationResult<T> = Result<T, VisualizationError>;

/// Energy landscape data point
#[derive(Debug, Clone)]
pub struct LandscapePoint {
    /// Configuration (spin or binary variables)
    pub configuration: Vec<i8>,

    /// Energy of this configuration
    pub energy: f64,

    /// Hamming distance from a reference configuration
    pub hamming_distance: Option<usize>,

    /// Basin index (if clustering is applied)
    pub basin_id: Option<usize>,
}

/// Energy landscape analyzer
pub struct LandscapeAnalyzer {
    /// Reference configuration for distance calculations
    reference_config: Option<Vec<i8>>,

    /// Energy cutoff for sampling
    energy_cutoff: Option<f64>,

    /// Maximum number of samples
    max_samples: usize,

    /// Random seed for reproducibility
    seed: Option<u64>,
}

impl LandscapeAnalyzer {
    /// Create a new landscape analyzer
    pub fn new() -> Self {
        Self {
            reference_config: None,
            energy_cutoff: None,
            max_samples: 10000,
            seed: None,
        }
    }

    /// Set reference configuration for distance calculations
    pub fn with_reference(mut self, config: Vec<i8>) -> Self {
        self.reference_config = Some(config);
        self
    }

    /// Set energy cutoff for sampling
    pub fn with_energy_cutoff(mut self, cutoff: f64) -> Self {
        self.energy_cutoff = Some(cutoff);
        self
    }

    /// Set maximum number of samples
    pub fn with_max_samples(mut self, max_samples: usize) -> Self {
        self.max_samples = max_samples;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sample energy landscape using random sampling
    pub fn sample_landscape(&self, model: &IsingModel) -> VisualizationResult<Vec<LandscapePoint>> {
        if model.num_qubits > 20 {
            return Err(VisualizationError::InvalidParameter(
                "Random sampling only practical for problems with ≤20 qubits".to_string(),
            ));
        }

        let mut rng = match self.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(thread_rng().gen()),
        };

        let mut points = Vec::new();
        let num_samples = std::cmp::min(self.max_samples, 2_usize.pow(model.num_qubits as u32));

        for _ in 0..num_samples {
            let config: Vec<i8> = (0..model.num_qubits)
                .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
                .collect();

            let energy = model.energy(&config)?;

            // Skip if energy exceeds cutoff
            if let Some(cutoff) = self.energy_cutoff {
                if energy > cutoff {
                    continue;
                }
            }

            let hamming_distance = self
                .reference_config
                .as_ref()
                .map(|ref_config| hamming_distance(&config, ref_config));

            points.push(LandscapePoint {
                configuration: config,
                energy,
                hamming_distance,
                basin_id: None,
            });
        }

        Ok(points)
    }

    /// Exhaustively enumerate energy landscape (small problems only)
    pub fn enumerate_landscape(
        &self,
        model: &IsingModel,
    ) -> VisualizationResult<Vec<LandscapePoint>> {
        if model.num_qubits > 16 {
            return Err(VisualizationError::InvalidParameter(
                "Exhaustive enumeration only practical for problems with ≤16 qubits".to_string(),
            ));
        }

        let mut points = Vec::new();
        let total_configs = 2_usize.pow(model.num_qubits as u32);

        for i in 0..total_configs {
            let config: Vec<i8> = (0..model.num_qubits)
                .map(|j| if (i >> j) & 1 == 1 { 1 } else { -1 })
                .collect();

            let energy = model.energy(&config)?;

            // Skip if energy exceeds cutoff
            if let Some(cutoff) = self.energy_cutoff {
                if energy > cutoff {
                    continue;
                }
            }

            let hamming_distance = self
                .reference_config
                .as_ref()
                .map(|ref_config| hamming_distance(&config, ref_config));

            points.push(LandscapePoint {
                configuration: config,
                energy,
                hamming_distance,
                basin_id: None,
            });
        }

        Ok(points)
    }

    /// Sample landscape using annealing trajectories
    pub fn sample_from_annealing(
        &self,
        model: &IsingModel,
        num_trajectories: usize,
    ) -> VisualizationResult<Vec<LandscapePoint>> {
        let mut all_points = Vec::new();

        for i in 0..num_trajectories {
            let params = AnnealingParams {
                seed: self.seed.map(|s| s + i as u64),
                num_sweeps: 1000,
                num_repetitions: 1,
                ..Default::default()
            };

            let mut simulator = QuantumAnnealingSimulator::new(params)
                .map_err(|e| VisualizationError::ComputationError(e.to_string()))?;

            // Collect intermediate states during annealing
            let trajectory = self.collect_annealing_trajectory(&mut simulator, model)?;

            all_points.extend(trajectory);
        }

        Ok(all_points)
    }

    /// Collect trajectory points during annealing
    fn collect_annealing_trajectory(
        &self,
        _simulator: &mut QuantumAnnealingSimulator,
        model: &IsingModel,
    ) -> VisualizationResult<Vec<LandscapePoint>> {
        // For now, generate a simple trajectory
        // In a real implementation, this would collect intermediate states
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.unwrap_or(0));
        let mut points = Vec::new();

        let mut current_config: Vec<i8> = (0..model.num_qubits)
            .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
            .collect();

        for step in 0..100 {
            let energy = model.energy(&current_config)?;

            let hamming_distance = self
                .reference_config
                .as_ref()
                .map(|ref_config| hamming_distance(&current_config, ref_config));

            points.push(LandscapePoint {
                configuration: current_config.clone(),
                energy,
                hamming_distance,
                basin_id: None,
            });

            // Simple random walk for demonstration
            if step % 10 == 0 {
                let flip_idx = rng.gen_range(0..model.num_qubits);
                current_config[flip_idx] *= -1;
            }
        }

        Ok(points)
    }
}

impl Default for LandscapeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Energy landscape statistics
#[derive(Debug, Clone)]
pub struct LandscapeStats {
    /// Number of configurations sampled
    pub num_configurations: usize,

    /// Minimum energy found
    pub min_energy: f64,

    /// Maximum energy found
    pub max_energy: f64,

    /// Average energy
    pub mean_energy: f64,

    /// Energy standard deviation
    pub energy_std: f64,

    /// Number of local minima
    pub num_local_minima: usize,

    /// Energy gap to first excited state
    pub energy_gap: Option<f64>,

    /// Degeneracy of ground state
    pub ground_state_degeneracy: usize,
}

/// Calculate landscape statistics
pub fn calculate_landscape_stats(points: &[LandscapePoint]) -> LandscapeStats {
    if points.is_empty() {
        return LandscapeStats {
            num_configurations: 0,
            min_energy: 0.0,
            max_energy: 0.0,
            mean_energy: 0.0,
            energy_std: 0.0,
            num_local_minima: 0,
            energy_gap: None,
            ground_state_degeneracy: 0,
        };
    }

    let energies: Vec<f64> = points.iter().map(|p| p.energy).collect();

    let min_energy = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let max_energy = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_energy = energies.iter().sum::<f64>() / energies.len() as f64;

    let variance = energies
        .iter()
        .map(|&e| (e - mean_energy).powi(2))
        .sum::<f64>()
        / energies.len() as f64;
    let energy_std = variance.sqrt();

    // Find unique energies and count degeneracies
    let mut energy_counts = HashMap::new();
    for &energy in &energies {
        *energy_counts.entry(OrderedFloat(energy)).or_insert(0) += 1;
    }

    let ground_state_degeneracy = *energy_counts.get(&OrderedFloat(min_energy)).unwrap_or(&0);

    // Find energy gap
    let mut unique_energies: Vec<f64> = energy_counts.keys().map(|&OrderedFloat(e)| e).collect();
    unique_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let energy_gap = if unique_energies.len() > 1 {
        Some(unique_energies[1] - unique_energies[0])
    } else {
        None
    };

    LandscapeStats {
        num_configurations: points.len(),
        min_energy,
        max_energy,
        mean_energy,
        energy_std,
        num_local_minima: 0, // Would require neighbor analysis
        energy_gap,
        ground_state_degeneracy,
    }
}

/// Plot energy landscape to file
pub fn plot_energy_landscape<P: AsRef<Path>>(
    points: &[LandscapePoint],
    output_path: P,
    title: &str,
) -> VisualizationResult<()> {
    let mut file = File::create(output_path)?;

    // Write CSV header
    writeln!(file, "configuration_id,energy,hamming_distance,basin_id")?;

    // Write data points
    for (i, point) in points.iter().enumerate() {
        writeln!(
            file,
            "{},{},{},{}",
            i,
            point.energy,
            point
                .hamming_distance
                .map_or("".to_string(), |d| d.to_string()),
            point.basin_id.map_or("".to_string(), |b| b.to_string())
        )?;
    }

    // Write metadata as comments
    writeln!(file, "# Title: {}", title)?;
    writeln!(file, "# Number of points: {}", points.len())?;

    if !points.is_empty() {
        let stats = calculate_landscape_stats(points);
        writeln!(file, "# Min energy: {:.6}", stats.min_energy)?;
        writeln!(file, "# Max energy: {:.6}", stats.max_energy)?;
        writeln!(file, "# Mean energy: {:.6}", stats.mean_energy)?;
        writeln!(file, "# Energy std: {:.6}", stats.energy_std)?;
    }

    Ok(())
}

/// Plot energy distribution histogram
pub fn plot_energy_histogram<P: AsRef<Path>>(
    points: &[LandscapePoint],
    output_path: P,
    num_bins: usize,
) -> VisualizationResult<()> {
    if points.is_empty() {
        return Err(VisualizationError::InvalidParameter(
            "No points to plot".to_string(),
        ));
    }

    let energies: Vec<f64> = points.iter().map(|p| p.energy).collect();
    let min_energy = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let max_energy = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if min_energy == max_energy {
        return Err(VisualizationError::InvalidParameter(
            "All energies are the same".to_string(),
        ));
    }

    let bin_width = (max_energy - min_energy) / num_bins as f64;
    let mut bins = vec![0; num_bins];

    for &energy in &energies {
        let bin_idx = ((energy - min_energy) / bin_width).floor() as usize;
        let bin_idx = std::cmp::min(bin_idx, num_bins - 1);
        bins[bin_idx] += 1;
    }

    let mut file = File::create(output_path)?;
    writeln!(file, "bin_center,count,frequency")?;

    for (i, &count) in bins.iter().enumerate() {
        let bin_center = min_energy + (i as f64 + 0.5) * bin_width;
        let frequency = count as f64 / energies.len() as f64;
        writeln!(file, "{:.6},{},{:.6}", bin_center, count, frequency)?;
    }

    Ok(())
}

/// Calculate Hamming distance between two configurations
fn hamming_distance(config1: &[i8], config2: &[i8]) -> usize {
    config1
        .iter()
        .zip(config2.iter())
        .filter(|(&a, &b)| a != b)
        .count()
}

/// Wrapper for f64 to make it hashable and orderable
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl std::hash::Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl std::cmp::Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl std::cmp::PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Basin analysis for energy landscape
pub struct BasinAnalyzer {
    /// Energy tolerance for grouping states into basins
    energy_tolerance: f64,

    /// Hamming distance threshold for connectivity
    hamming_threshold: usize,
}

impl BasinAnalyzer {
    /// Create a new basin analyzer
    pub fn new(energy_tolerance: f64, hamming_threshold: usize) -> Self {
        Self {
            energy_tolerance,
            hamming_threshold,
        }
    }

    /// Identify energy basins in the landscape
    pub fn identify_basins(&self, points: &mut [LandscapePoint]) -> VisualizationResult<usize> {
        if points.is_empty() {
            return Ok(0);
        }

        let mut basin_id = 0;

        for i in 0..points.len() {
            if points[i].basin_id.is_some() {
                continue;
            }

            // Start a new basin
            points[i].basin_id = Some(basin_id);
            let mut stack = vec![i];

            while let Some(current_idx) = stack.pop() {
                let current_energy = points[current_idx].energy;
                let current_config = points[current_idx].configuration.clone();

                // Find neighbors
                for j in 0..points.len() {
                    if points[j].basin_id.is_some() {
                        continue;
                    }

                    let energy_diff = (points[j].energy - current_energy).abs();
                    let hamming_dist = hamming_distance(&current_config, &points[j].configuration);

                    if energy_diff <= self.energy_tolerance
                        && hamming_dist <= self.hamming_threshold
                    {
                        points[j].basin_id = Some(basin_id);
                        stack.push(j);
                    }
                }
            }

            basin_id += 1;
        }

        Ok(basin_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_landscape_analyzer() {
        let mut model = IsingModel::new(4);
        model.set_coupling(0, 1, -1.0).unwrap();
        model.set_coupling(1, 2, -1.0).unwrap();
        model.set_coupling(2, 3, -1.0).unwrap();

        let analyzer = LandscapeAnalyzer::new().with_max_samples(100).with_seed(42);

        let points = analyzer.sample_landscape(&model).unwrap();
        assert!(!points.is_empty());
        assert!(points.len() <= 100);
    }

    #[test]
    fn test_landscape_stats() {
        let points = vec![
            LandscapePoint {
                configuration: vec![1, 1, 1],
                energy: -2.0,
                hamming_distance: None,
                basin_id: None,
            },
            LandscapePoint {
                configuration: vec![-1, -1, -1],
                energy: -2.0,
                hamming_distance: None,
                basin_id: None,
            },
            LandscapePoint {
                configuration: vec![1, -1, 1],
                energy: 1.0,
                hamming_distance: None,
                basin_id: None,
            },
        ];

        let stats = calculate_landscape_stats(&points);
        assert_eq!(stats.num_configurations, 3);
        assert_eq!(stats.min_energy, -2.0);
        assert_eq!(stats.max_energy, 1.0);
        assert_eq!(stats.ground_state_degeneracy, 2);
        assert_eq!(stats.energy_gap, Some(3.0));
    }

    #[test]
    fn test_hamming_distance() {
        let config1 = vec![1, -1, 1, -1];
        let config2 = vec![1, 1, -1, -1];
        assert_eq!(hamming_distance(&config1, &config2), 2);

        let config3 = vec![1, -1, 1, -1];
        assert_eq!(hamming_distance(&config1, &config3), 0);
    }

    #[test]
    fn test_basin_analyzer() {
        let mut points = vec![
            LandscapePoint {
                configuration: vec![1, 1],
                energy: -1.0,
                hamming_distance: None,
                basin_id: None,
            },
            LandscapePoint {
                configuration: vec![1, -1],
                energy: -0.9,
                hamming_distance: None,
                basin_id: None,
            },
            LandscapePoint {
                configuration: vec![-1, -1],
                energy: 2.0,
                hamming_distance: None,
                basin_id: None,
            },
        ];

        let analyzer = BasinAnalyzer::new(0.2, 1);
        let num_basins = analyzer.identify_basins(&mut points).unwrap();

        assert_eq!(num_basins, 2);
        assert_eq!(points[0].basin_id, points[1].basin_id);
        assert_ne!(points[0].basin_id, points[2].basin_id);
    }
}
