//! Population annealing with MPI support
//!
//! This module implements population annealing, a parallel sampling algorithm
//! that maintains a population of configurations and periodically resamples
//! based on Boltzmann weights.

use scirs2_core::random::prelude::*;
use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::{Rng, SeedableRng};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::ising::{IsingError, IsingModel};
use crate::simulator::{
    AnnealingParams, AnnealingSolution, TemperatureSchedule, TransverseFieldSchedule,
};

/// Errors that can occur during population annealing
#[derive(Error, Debug)]
pub enum PopulationAnnealingError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// MPI communication error
    #[error("MPI error: {0}")]
    MpiError(String),

    /// Population evolution error
    #[error("Population evolution error: {0}")]
    EvolutionError(String),
}

/// Result type for population annealing operations
pub type PopulationAnnealingResult<T> = Result<T, PopulationAnnealingError>;

/// Configuration for population annealing
#[derive(Debug, Clone)]
pub struct PopulationAnnealingConfig {
    /// Population size
    pub population_size: usize,

    /// Temperature schedule
    pub temperature_schedule: TemperatureSchedule,

    /// Initial temperature
    pub initial_temperature: f64,

    /// Final temperature
    pub final_temperature: f64,

    /// Number of temperature steps
    pub num_temperature_steps: usize,

    /// Number of Monte Carlo sweeps per temperature step
    pub sweeps_per_step: usize,

    /// Resampling frequency (in temperature steps)
    pub resampling_frequency: usize,

    /// Effective sample size threshold for resampling
    pub ess_threshold: f64,

    /// Random seed
    pub seed: Option<u64>,

    /// Maximum runtime
    pub timeout: Option<Duration>,

    /// MPI configuration
    pub mpi_config: Option<MpiConfig>,
}

impl Default for PopulationAnnealingConfig {
    fn default() -> Self {
        Self {
            population_size: 1000,
            temperature_schedule: TemperatureSchedule::Exponential(3.0),
            initial_temperature: 10.0,
            final_temperature: 0.01,
            num_temperature_steps: 100,
            sweeps_per_step: 100,
            resampling_frequency: 5,
            ess_threshold: 0.5,
            seed: None,
            timeout: Some(Duration::from_secs(3600)), // 1 hour
            mpi_config: None,
        }
    }
}

/// MPI configuration for distributed population annealing
#[derive(Debug, Clone)]
pub struct MpiConfig {
    /// Number of MPI processes
    pub num_processes: usize,

    /// Current process rank
    pub rank: usize,

    /// Enable load balancing
    pub load_balancing: bool,

    /// Communication frequency (in temperature steps)
    pub communication_frequency: usize,
}

/// Individual configuration in the population
#[derive(Debug, Clone)]
pub struct PopulationMember {
    /// Spin configuration
    pub configuration: Vec<i8>,

    /// Energy of the configuration
    pub energy: f64,

    /// Weight (for resampling)
    pub weight: f64,

    /// Ancestor lineage (for tracking)
    pub lineage: Vec<usize>,
}

impl PopulationMember {
    /// Create a new population member
    #[must_use]
    pub const fn new(configuration: Vec<i8>, energy: f64) -> Self {
        Self {
            configuration,
            energy,
            weight: 1.0,
            lineage: vec![],
        }
    }
}

/// Population annealing results
#[derive(Debug, Clone)]
pub struct PopulationAnnealingSolution {
    /// Best configuration found
    pub best_configuration: Vec<i8>,

    /// Best energy found
    pub best_energy: f64,

    /// Final population
    pub final_population: Vec<PopulationMember>,

    /// Energy statistics over time
    pub energy_history: Vec<EnergyStatistics>,

    /// Effective sample size history
    pub ess_history: Vec<f64>,

    /// Total runtime
    pub runtime: Duration,

    /// Number of resampling events
    pub num_resamplings: usize,

    /// Additional info
    pub info: String,
}

/// Energy statistics for the population
#[derive(Debug, Clone)]
pub struct EnergyStatistics {
    /// Temperature at this step
    pub temperature: f64,

    /// Minimum energy in population
    pub min_energy: f64,

    /// Maximum energy in population
    pub max_energy: f64,

    /// Mean energy
    pub mean_energy: f64,

    /// Energy standard deviation
    pub energy_std: f64,

    /// Effective sample size
    pub effective_sample_size: f64,
}

/// Population annealing simulator
pub struct PopulationAnnealingSimulator {
    /// Configuration
    config: PopulationAnnealingConfig,

    /// Random number generator
    rng: ChaCha8Rng,

    /// MPI interface (optional)
    mpi_interface: Option<MpiInterface>,
}

impl PopulationAnnealingSimulator {
    /// Create a new population annealing simulator
    pub fn new(config: PopulationAnnealingConfig) -> PopulationAnnealingResult<Self> {
        // Validate configuration
        Self::validate_config(&config)?;

        // Initialize RNG
        let rng = match config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(thread_rng().gen()),
        };

        // Initialize MPI interface if configured
        let mpi_interface = config
            .mpi_config
            .as_ref()
            .map(|mpi_config| MpiInterface::new(mpi_config.clone()))
            .transpose()?;

        Ok(Self {
            config,
            rng,
            mpi_interface,
        })
    }

    /// Validate configuration parameters
    fn validate_config(config: &PopulationAnnealingConfig) -> PopulationAnnealingResult<()> {
        if config.population_size == 0 {
            return Err(PopulationAnnealingError::InvalidParameter(
                "Population size must be positive".to_string(),
            ));
        }

        if config.initial_temperature <= 0.0 || config.final_temperature <= 0.0 {
            return Err(PopulationAnnealingError::InvalidParameter(
                "Temperatures must be positive".to_string(),
            ));
        }

        if config.initial_temperature <= config.final_temperature {
            return Err(PopulationAnnealingError::InvalidParameter(
                "Initial temperature must be greater than final temperature".to_string(),
            ));
        }

        if config.num_temperature_steps == 0 {
            return Err(PopulationAnnealingError::InvalidParameter(
                "Number of temperature steps must be positive".to_string(),
            ));
        }

        if config.ess_threshold <= 0.0 || config.ess_threshold > 1.0 {
            return Err(PopulationAnnealingError::InvalidParameter(
                "ESS threshold must be in (0, 1]".to_string(),
            ));
        }

        Ok(())
    }

    /// Solve an Ising model using population annealing
    pub fn solve(
        &mut self,
        model: &IsingModel,
    ) -> PopulationAnnealingResult<PopulationAnnealingSolution> {
        let start_time = Instant::now();

        // Initialize population
        let mut population = self.initialize_population(model)?;

        // Track statistics
        let mut energy_history = Vec::new();
        let mut ess_history = Vec::new();
        let mut best_energy = f64::INFINITY;
        let mut best_configuration = vec![];
        let mut num_resamplings = 0;

        // Temperature schedule
        let temperatures = self.generate_temperature_schedule();

        // Main annealing loop
        for (step, &temperature) in temperatures.iter().enumerate() {
            // Check timeout
            if let Some(timeout) = self.config.timeout {
                if start_time.elapsed() > timeout {
                    break;
                }
            }

            // Perform Monte Carlo sweeps
            self.monte_carlo_step(model, &mut population, temperature)?;

            // Calculate weights and statistics
            let stats = self.calculate_statistics(&population, temperature);
            energy_history.push(stats.clone());
            ess_history.push(stats.effective_sample_size);

            // Update best solution
            for member in &population {
                if member.energy < best_energy {
                    best_energy = member.energy;
                    best_configuration = member.configuration.clone();
                }
            }

            // Resampling decision
            let should_resample = (step + 1) % self.config.resampling_frequency == 0
                || stats.effective_sample_size
                    < self.config.ess_threshold * self.config.population_size as f64;

            if should_resample {
                self.resample_population(&mut population, temperature)?;
                num_resamplings += 1;

                // MPI communication if enabled
                if let Some(mpi) = &mut self.mpi_interface {
                    mpi.exchange_populations(&mut population)?;
                }
            }
        }

        let runtime = start_time.elapsed();

        Ok(PopulationAnnealingSolution {
            best_configuration,
            best_energy,
            final_population: population,
            energy_history,
            ess_history,
            runtime,
            num_resamplings,
            info: format!(
                "Population annealing completed: {} temperature steps, {} resamplings, runtime: {:?}",
                temperatures.len(), num_resamplings, runtime
            ),
        })
    }

    /// Initialize the population with random configurations
    fn initialize_population(
        &mut self,
        model: &IsingModel,
    ) -> PopulationAnnealingResult<Vec<PopulationMember>> {
        let mut population = Vec::with_capacity(self.config.population_size);

        for _ in 0..self.config.population_size {
            // Generate random configuration
            let configuration: Vec<i8> = (0..model.num_qubits)
                .map(|_| if self.rng.gen_bool(0.5) { 1 } else { -1 })
                .collect();

            // Calculate energy
            let energy = model.energy(&configuration)?;

            population.push(PopulationMember::new(configuration, energy));
        }

        Ok(population)
    }

    /// Generate temperature schedule
    fn generate_temperature_schedule(&self) -> Vec<f64> {
        let mut temperatures = Vec::with_capacity(self.config.num_temperature_steps);

        for i in 0..self.config.num_temperature_steps {
            let t = i as f64 / (self.config.num_temperature_steps - 1) as f64;
            let temperature =
                self.config
                    .temperature_schedule
                    .calculate(t, 1.0, self.config.initial_temperature);

            // Ensure we don't go below final temperature
            let clamped_temp = temperature.max(self.config.final_temperature);
            temperatures.push(clamped_temp);
        }

        temperatures
    }

    /// Perform Monte Carlo step for the entire population
    fn monte_carlo_step(
        &mut self,
        model: &IsingModel,
        population: &mut [PopulationMember],
        temperature: f64,
    ) -> PopulationAnnealingResult<()> {
        for member in population.iter_mut() {
            for _ in 0..self.config.sweeps_per_step {
                // Choose random spin to flip
                let spin_idx = self.rng.gen_range(0..model.num_qubits);

                // Calculate energy change
                let old_spin = member.configuration[spin_idx];
                member.configuration[spin_idx] = -old_spin;

                let new_energy = model.energy(&member.configuration)?;
                let delta_e = new_energy - member.energy;

                // Metropolis acceptance
                let accept = delta_e <= 0.0 || {
                    let prob = (-delta_e / temperature).exp();
                    self.rng.gen::<f64>() < prob
                };

                if accept {
                    member.energy = new_energy;
                } else {
                    // Revert the change
                    member.configuration[spin_idx] = old_spin;
                }
            }
        }

        Ok(())
    }

    /// Calculate population statistics
    fn calculate_statistics(
        &self,
        population: &[PopulationMember],
        temperature: f64,
    ) -> EnergyStatistics {
        let energies: Vec<f64> = population.iter().map(|m| m.energy).collect();

        let min_energy = energies.iter().copied().fold(f64::INFINITY, f64::min);
        let max_energy = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_energy = energies.iter().sum::<f64>() / energies.len() as f64;

        let variance = energies
            .iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f64>()
            / energies.len() as f64;
        let energy_std = variance.sqrt();

        // Calculate effective sample size
        let min_e = min_energy;
        let weights: Vec<f64> = energies
            .iter()
            .map(|&e| (-(e - min_e) / temperature).exp())
            .collect();

        let sum_weights = weights.iter().sum::<f64>();
        let sum_weights_squared = weights.iter().map(|&w| w * w).sum::<f64>();

        let effective_sample_size = if sum_weights_squared > 0.0 {
            sum_weights * sum_weights / sum_weights_squared
        } else {
            population.len() as f64
        };

        EnergyStatistics {
            temperature,
            min_energy,
            max_energy,
            mean_energy,
            energy_std,
            effective_sample_size,
        }
    }

    /// Resample population based on Boltzmann weights
    fn resample_population(
        &mut self,
        population: &mut Vec<PopulationMember>,
        temperature: f64,
    ) -> PopulationAnnealingResult<()> {
        if population.is_empty() {
            return Ok(());
        }

        // Calculate Boltzmann weights
        let min_energy = population
            .iter()
            .map(|m| m.energy)
            .fold(f64::INFINITY, f64::min);
        let weights: Vec<f64> = population
            .iter()
            .map(|m| (-(m.energy - min_energy) / temperature).exp())
            .collect();

        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 {
            return Err(PopulationAnnealingError::EvolutionError(
                "Total weight is zero or negative".to_string(),
            ));
        }

        // Normalize weights
        let normalized_weights: Vec<f64> = weights.iter().map(|w| w / total_weight).collect();

        // Cumulative distribution
        let mut cumulative_weights = vec![0.0; normalized_weights.len()];
        cumulative_weights[0] = normalized_weights[0];
        for i in 1..normalized_weights.len() {
            cumulative_weights[i] = cumulative_weights[i - 1] + normalized_weights[i];
        }

        // Resample
        let mut new_population = Vec::with_capacity(population.len());
        for _ in 0..population.len() {
            let r = self.rng.gen::<f64>();
            let idx = cumulative_weights
                .iter()
                .position(|&w| r <= w)
                .unwrap_or(population.len() - 1);

            let mut new_member = population[idx].clone();
            new_member.lineage.push(idx);
            new_population.push(new_member);
        }

        *population = new_population;
        Ok(())
    }
}

/// MPI interface for distributed population annealing
struct MpiInterface {
    config: MpiConfig,
}

impl MpiInterface {
    const fn new(config: MpiConfig) -> PopulationAnnealingResult<Self> {
        // In a real implementation, this would initialize MPI
        // For now, we just store the config
        Ok(Self { config })
    }

    const fn exchange_populations(
        &self,
        _population: &mut Vec<PopulationMember>,
    ) -> PopulationAnnealingResult<()> {
        // In a real implementation, this would:
        // 1. Gather population statistics from all processes
        // 2. Perform load balancing if needed
        // 3. Exchange population members between processes
        // 4. Redistribute populations based on performance

        // For now, this is a placeholder
        Ok(())
    }
}

/// Convert population annealing result to standard annealing solution
impl From<PopulationAnnealingSolution> for AnnealingSolution {
    fn from(result: PopulationAnnealingSolution) -> Self {
        Self {
            best_spins: result.best_configuration,
            best_energy: result.best_energy,
            repetitions: 1,
            total_sweeps: 0, // Would need to track this
            runtime: result.runtime,
            info: result.info,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_population_annealing_config() {
        let config = PopulationAnnealingConfig::default();
        assert!(PopulationAnnealingSimulator::validate_config(&config).is_ok());

        // Test invalid configuration
        let mut invalid_config = config.clone();
        invalid_config.population_size = 0;
        assert!(PopulationAnnealingSimulator::validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_temperature_schedule() {
        let config = PopulationAnnealingConfig {
            initial_temperature: 10.0,
            final_temperature: 0.1,
            num_temperature_steps: 5,
            ..Default::default()
        };

        let mut simulator =
            PopulationAnnealingSimulator::new(config).expect("failed to create simulator in test");
        let temperatures = simulator.generate_temperature_schedule();

        assert_eq!(temperatures.len(), 5);
        assert!(temperatures[0] >= temperatures[4]);
        assert!(temperatures[4] >= 0.1);
    }

    #[test]
    fn test_population_initialization() {
        let mut model = IsingModel::new(4);
        model
            .set_coupling(0, 1, -1.0)
            .expect("failed to set coupling in test");

        let config = PopulationAnnealingConfig {
            population_size: 10,
            seed: Some(42),
            ..Default::default()
        };

        let mut simulator =
            PopulationAnnealingSimulator::new(config).expect("failed to create simulator in test");
        let population = simulator
            .initialize_population(&model)
            .expect("failed to initialize population in test");

        assert_eq!(population.len(), 10);
        for member in &population {
            assert_eq!(member.configuration.len(), 4);
            assert!(member.configuration.iter().all(|&s| s == 1 || s == -1));
        }
    }

    #[test]
    fn test_simple_population_annealing() {
        let mut model = IsingModel::new(3);
        model
            .set_coupling(0, 1, -1.0)
            .expect("failed to set coupling in test");
        model
            .set_coupling(1, 2, -1.0)
            .expect("failed to set coupling in test");

        let config = PopulationAnnealingConfig {
            population_size: 50,
            num_temperature_steps: 10,
            sweeps_per_step: 10,
            seed: Some(42),
            ..Default::default()
        };

        let mut simulator =
            PopulationAnnealingSimulator::new(config).expect("failed to create simulator in test");
        let result = simulator
            .solve(&model)
            .expect("failed to solve model in test");

        assert!(result.best_energy <= 0.0); // Should find good solution
        assert_eq!(result.final_population.len(), 50);
        assert_eq!(result.energy_history.len(), 10);
    }
}
