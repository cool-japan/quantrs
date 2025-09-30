//! Coherent Ising Machine simulation for photonic quantum annealing
//!
//! This module implements coherent Ising machines (CIMs) based on optical parametric
//! oscillators (OPOs) and photonic networks. CIMs use optical pulses to represent
//! spins and leverage optical interference and parametric amplification to solve
//! optimization problems through a completely different physical mechanism than
//! traditional quantum annealing.

use scirs2_core::random::prelude::*;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::random::ChaCha8Rng;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::ising::{IsingError, IsingModel};

/// Errors that can occur in coherent Ising machine simulation
#[derive(Error, Debug)]
pub enum CimError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Invalid optical parameters
    #[error("Invalid optical parameters: {0}")]
    InvalidOpticalParameters(String),

    /// Simulation error
    #[error("Simulation error: {0}")]
    SimulationError(String),

    /// Network topology error
    #[error("Network topology error: {0}")]
    TopologyError(String),

    /// Convergence error
    #[error("Convergence error: {0}")]
    ConvergenceError(String),
}

/// Result type for CIM operations
pub type CimResult<T> = Result<T, CimError>;

/// Optical parametric oscillator (OPO) representation
#[derive(Debug, Clone)]
pub struct OpticalParametricOscillator {
    /// Oscillator index
    pub index: usize,

    /// Complex amplitude (amplitude and phase)
    pub amplitude: Complex,

    /// Parametric gain
    pub gain: f64,

    /// Linear loss rate
    pub loss: f64,

    /// Nonlinear saturation parameter
    pub saturation: f64,

    /// External injection field
    pub injection: Complex,

    /// Oscillation threshold
    pub threshold: f64,
}

/// Complex number representation for optical fields
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    /// Real part
    pub re: f64,
    /// Imaginary part
    pub im: f64,
}

impl Complex {
    /// Create a new complex number
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Create from polar coordinates
    pub fn from_polar(magnitude: f64, phase: f64) -> Self {
        Self {
            re: magnitude * phase.cos(),
            im: magnitude * phase.sin(),
        }
    }

    /// Get magnitude squared
    pub fn magnitude_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Get magnitude
    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    /// Get phase angle
    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Complex conjugate
    pub fn conjugate(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            re: self.re * scalar,
            im: self.im * scalar,
        }
    }
}

/// Optical coupling between oscillators
#[derive(Debug, Clone)]
pub struct OpticalCoupling {
    /// Source oscillator index
    pub from: usize,

    /// Target oscillator index
    pub to: usize,

    /// Coupling strength (can be complex for phase shifts)
    pub strength: Complex,

    /// Time delay (for large networks)
    pub delay: f64,
}

/// Coherent Ising machine configuration
#[derive(Debug, Clone)]
pub struct CimConfig {
    /// Number of optical oscillators
    pub num_oscillators: usize,

    /// Simulation time step
    pub dt: f64,

    /// Total simulation time
    pub total_time: f64,

    /// Pump power ramping schedule
    pub pump_schedule: PumpSchedule,

    /// Optical network topology
    pub topology: NetworkTopology,

    /// Noise parameters
    pub noise_config: NoiseConfig,

    /// Measurement and detection parameters
    pub measurement_config: MeasurementConfig,

    /// Convergence criteria
    pub convergence_config: ConvergenceConfig,

    /// Random seed
    pub seed: Option<u64>,

    /// Enable detailed logging
    pub detailed_logging: bool,

    /// Output file for optical state evolution
    pub output_file: Option<String>,
}

impl Default for CimConfig {
    fn default() -> Self {
        Self {
            num_oscillators: 4,
            dt: 0.001,
            total_time: 10.0,
            pump_schedule: PumpSchedule::Linear {
                initial_power: 0.5,
                final_power: 2.0,
            },
            topology: NetworkTopology::FullyConnected,
            noise_config: NoiseConfig::default(),
            measurement_config: MeasurementConfig::default(),
            convergence_config: ConvergenceConfig::default(),
            seed: None,
            detailed_logging: false,
            output_file: None,
        }
    }
}

/// Pump power ramping schedules
pub enum PumpSchedule {
    /// Linear increase in pump power
    Linear {
        initial_power: f64,
        final_power: f64,
    },

    /// Exponential approach to final power
    Exponential {
        initial_power: f64,
        final_power: f64,
        time_constant: f64,
    },

    /// S-curve (sigmoid) ramping
    Sigmoid {
        initial_power: f64,
        final_power: f64,
        steepness: f64,
        midpoint: f64,
    },

    /// Custom time-dependent function
    Custom {
        power_function: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    },
}

impl std::fmt::Debug for PumpSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PumpSchedule::Linear {
                initial_power,
                final_power,
            } => f
                .debug_struct("Linear")
                .field("initial_power", initial_power)
                .field("final_power", final_power)
                .finish(),
            PumpSchedule::Exponential {
                initial_power,
                final_power,
                time_constant,
            } => f
                .debug_struct("Exponential")
                .field("initial_power", initial_power)
                .field("final_power", final_power)
                .field("time_constant", time_constant)
                .finish(),
            PumpSchedule::Sigmoid {
                initial_power,
                final_power,
                steepness,
                midpoint,
            } => f
                .debug_struct("Sigmoid")
                .field("initial_power", initial_power)
                .field("final_power", final_power)
                .field("steepness", steepness)
                .field("midpoint", midpoint)
                .finish(),
            PumpSchedule::Custom { .. } => f
                .debug_struct("Custom")
                .field("power_function", &"<function>")
                .finish(),
        }
    }
}

impl Clone for PumpSchedule {
    fn clone(&self) -> Self {
        match self {
            PumpSchedule::Linear {
                initial_power,
                final_power,
            } => PumpSchedule::Linear {
                initial_power: *initial_power,
                final_power: *final_power,
            },
            PumpSchedule::Exponential {
                initial_power,
                final_power,
                time_constant,
            } => PumpSchedule::Exponential {
                initial_power: *initial_power,
                final_power: *final_power,
                time_constant: *time_constant,
            },
            PumpSchedule::Sigmoid {
                initial_power,
                final_power,
                steepness,
                midpoint,
            } => PumpSchedule::Sigmoid {
                initial_power: *initial_power,
                final_power: *final_power,
                steepness: *steepness,
                midpoint: *midpoint,
            },
            PumpSchedule::Custom { .. } => {
                // Cannot clone function pointers, use default linear schedule
                PumpSchedule::Linear {
                    initial_power: 0.0,
                    final_power: 1.0,
                }
            }
        }
    }
}

/// Network topology configurations
#[derive(Debug, Clone)]
pub enum NetworkTopology {
    /// Fully connected network (all-to-all coupling)
    FullyConnected,

    /// Ring topology
    Ring,

    /// Two-dimensional lattice
    Lattice2D { width: usize, height: usize },

    /// Random network with specified connectivity
    Random { connectivity: f64 },

    /// Small-world network
    SmallWorld {
        ring_connectivity: usize,
        rewiring_probability: f64,
    },

    /// Custom topology from coupling matrix
    Custom { couplings: Vec<OpticalCoupling> },
}

/// Noise configuration for realistic simulation
#[derive(Debug, Clone)]
pub struct NoiseConfig {
    /// Quantum noise strength
    pub quantum_noise: f64,

    /// Classical phase noise
    pub phase_noise: f64,

    /// Amplitude noise
    pub amplitude_noise: f64,

    /// Thermal noise temperature
    pub temperature: f64,

    /// Environmental decoherence rate
    pub decoherence_rate: f64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            quantum_noise: 0.01,
            phase_noise: 0.001,
            amplitude_noise: 0.001,
            temperature: 0.01,
            decoherence_rate: 0.001,
        }
    }
}

/// Measurement and detection configuration
#[derive(Debug, Clone)]
pub struct MeasurementConfig {
    /// Homodyne detection efficiency
    pub detection_efficiency: f64,

    /// Measurement time window
    pub measurement_window: f64,

    /// Phase reference for homodyne detection
    pub reference_phase: f64,

    /// Number of measurement repetitions
    pub measurement_repetitions: usize,

    /// Threshold for binary decision
    pub decision_threshold: f64,
}

impl Default for MeasurementConfig {
    fn default() -> Self {
        Self {
            detection_efficiency: 0.95,
            measurement_window: 1.0,
            reference_phase: 0.0,
            measurement_repetitions: 100,
            decision_threshold: 0.0,
        }
    }
}

/// Convergence criteria configuration
#[derive(Debug, Clone)]
pub struct ConvergenceConfig {
    /// Energy tolerance for convergence
    pub energy_tolerance: f64,

    /// Maximum time without improvement
    pub stagnation_time: f64,

    /// Minimum oscillation threshold
    pub oscillation_threshold: f64,

    /// Phase stability requirement
    pub phase_stability: f64,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            energy_tolerance: 1e-6,
            stagnation_time: 1.0,
            oscillation_threshold: 0.1,
            phase_stability: 0.01,
        }
    }
}

/// Results from coherent Ising machine simulation
#[derive(Debug, Clone)]
pub struct CimResults {
    /// Best solution found
    pub best_solution: Vec<i8>,

    /// Best energy achieved
    pub best_energy: f64,

    /// Final optical state
    pub final_optical_state: Vec<Complex>,

    /// Energy evolution over time
    pub energy_history: Vec<f64>,

    /// Optical state evolution (sampled)
    pub optical_evolution: Vec<Vec<Complex>>,

    /// Time points for evolution data
    pub time_points: Vec<f64>,

    /// Convergence achieved
    pub converged: bool,

    /// Convergence time
    pub convergence_time: f64,

    /// Total simulation time
    pub total_simulation_time: Duration,

    /// Optical statistics
    pub optical_stats: OpticalStatistics,

    /// Performance metrics
    pub performance_metrics: CimPerformanceMetrics,
}

/// Optical system statistics
#[derive(Debug, Clone)]
pub struct OpticalStatistics {
    /// Average optical power over time
    pub average_power: f64,

    /// Power variance
    pub power_variance: f64,

    /// Phase coherence measures
    pub phase_coherence: Vec<f64>,

    /// Cross-correlations between oscillators
    pub cross_correlations: Vec<Vec<f64>>,

    /// Oscillation frequencies
    pub oscillation_frequencies: Vec<f64>,

    /// Pump efficiency
    pub pump_efficiency: f64,
}

/// Performance metrics for CIM
#[derive(Debug, Clone)]
pub struct CimPerformanceMetrics {
    /// Solution quality (energy gap from best known)
    pub solution_quality: f64,

    /// Time to convergence
    pub time_to_convergence: f64,

    /// Number of phase transitions
    pub phase_transitions: usize,

    /// Average optical power efficiency
    pub power_efficiency: f64,

    /// Noise resilience score
    pub noise_resilience: f64,
}

/// Coherent Ising machine simulator
pub struct CoherentIsingMachine {
    /// Configuration
    config: CimConfig,

    /// Optical oscillators
    oscillators: Vec<OpticalParametricOscillator>,

    /// Optical couplings
    couplings: Vec<OpticalCoupling>,

    /// Random number generator
    rng: ChaCha8Rng,

    /// Current simulation time
    current_time: f64,

    /// Evolution history (for analysis)
    evolution_history: Vec<(f64, Vec<Complex>)>,

    /// Energy history
    energy_history: Vec<f64>,
}

impl CoherentIsingMachine {
    /// Create a new coherent Ising machine
    pub fn new(config: CimConfig) -> CimResult<Self> {
        let rng = match config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(thread_rng().gen()),
        };

        let mut cim = Self {
            oscillators: Vec::new(),
            couplings: Vec::new(),
            rng,
            current_time: 0.0,
            evolution_history: Vec::new(),
            energy_history: Vec::new(),
            config,
        };

        cim.initialize_system()?;
        Ok(cim)
    }

    /// Initialize the optical system
    fn initialize_system(&mut self) -> CimResult<()> {
        // Initialize oscillators
        for i in 0..self.config.num_oscillators {
            let osc = OpticalParametricOscillator {
                index: i,
                amplitude: Complex::new(
                    self.rng.gen_range(-0.1..0.1),
                    self.rng.gen_range(-0.1..0.1),
                ),
                gain: 1.0,
                loss: 0.1,
                saturation: 1.0,
                injection: Complex::new(0.0, 0.0),
                threshold: 1.0,
            };
            self.oscillators.push(osc);
        }

        // Initialize couplings based on topology
        self.initialize_topology()?;

        Ok(())
    }

    /// Initialize network topology
    fn initialize_topology(&mut self) -> CimResult<()> {
        self.couplings.clear();

        let topology = self.config.topology.clone();
        match topology {
            NetworkTopology::FullyConnected => {
                for i in 0..self.config.num_oscillators {
                    for j in (i + 1)..self.config.num_oscillators {
                        self.couplings.push(OpticalCoupling {
                            from: i,
                            to: j,
                            strength: Complex::new(0.1, 0.0),
                            delay: 0.0,
                        });
                        self.couplings.push(OpticalCoupling {
                            from: j,
                            to: i,
                            strength: Complex::new(0.1, 0.0),
                            delay: 0.0,
                        });
                    }
                }
            }

            NetworkTopology::Ring => {
                for i in 0..self.config.num_oscillators {
                    let j = (i + 1) % self.config.num_oscillators;
                    self.couplings.push(OpticalCoupling {
                        from: i,
                        to: j,
                        strength: Complex::new(0.2, 0.0),
                        delay: 0.0,
                    });
                    self.couplings.push(OpticalCoupling {
                        from: j,
                        to: i,
                        strength: Complex::new(0.2, 0.0),
                        delay: 0.0,
                    });
                }
            }

            NetworkTopology::Lattice2D { width, height } => {
                if width * height != self.config.num_oscillators {
                    return Err(CimError::TopologyError(
                        "Lattice dimensions don't match number of oscillators".to_string(),
                    ));
                }

                // Nearest neighbor couplings
                for i in 0..width {
                    for j in 0..height {
                        let idx = i * height + j;

                        // Right neighbor
                        if i + 1 < width {
                            let neighbor = (i + 1) * height + j;
                            self.add_bidirectional_coupling(idx, neighbor, 0.15);
                        }

                        // Bottom neighbor
                        if j + 1 < height {
                            let neighbor = i * height + (j + 1);
                            self.add_bidirectional_coupling(idx, neighbor, 0.15);
                        }
                    }
                }
            }

            NetworkTopology::Random { connectivity } => {
                let num_possible_edges =
                    self.config.num_oscillators * (self.config.num_oscillators - 1) / 2;
                let num_edges = (num_possible_edges as f64 * connectivity) as usize;

                let mut added_edges = std::collections::HashSet::new();

                while added_edges.len() < num_edges {
                    let i = self.rng.gen_range(0..self.config.num_oscillators);
                    let j = self.rng.gen_range(0..self.config.num_oscillators);

                    if i != j {
                        let edge = if i < j { (i, j) } else { (j, i) };
                        if !added_edges.contains(&edge) {
                            added_edges.insert(edge);
                            let strength = self.rng.gen_range(0.05..0.25);
                            self.add_bidirectional_coupling(edge.0, edge.1, strength);
                        }
                    }
                }
            }

            NetworkTopology::SmallWorld {
                ring_connectivity,
                rewiring_probability,
            } => {
                // Start with ring lattice
                for i in 0..self.config.num_oscillators {
                    for k in 1..=ring_connectivity {
                        let j = (i + k) % self.config.num_oscillators;

                        // Rewire with probability
                        if self.rng.gen_bool(rewiring_probability) {
                            let new_target = self.rng.gen_range(0..self.config.num_oscillators);
                            if new_target != i {
                                self.add_bidirectional_coupling(i, new_target, 0.1);
                            }
                        } else {
                            self.add_bidirectional_coupling(i, j, 0.1);
                        }
                    }
                }
            }

            NetworkTopology::Custom { couplings } => {
                self.couplings = couplings;
            }
        }

        Ok(())
    }

    /// Add bidirectional coupling between two oscillators
    fn add_bidirectional_coupling(&mut self, i: usize, j: usize, strength: f64) {
        self.couplings.push(OpticalCoupling {
            from: i,
            to: j,
            strength: Complex::new(strength, 0.0),
            delay: 0.0,
        });
        self.couplings.push(OpticalCoupling {
            from: j,
            to: i,
            strength: Complex::new(strength, 0.0),
            delay: 0.0,
        });
    }

    /// Solve an Ising problem using the coherent Ising machine
    pub fn solve(&mut self, problem: &IsingModel) -> CimResult<CimResults> {
        if problem.num_qubits != self.config.num_oscillators {
            return Err(CimError::SimulationError(format!(
                "Problem size {} doesn't match CIM size {}",
                problem.num_qubits, self.config.num_oscillators
            )));
        }

        println!("Starting coherent Ising machine simulation...");
        let start_time = Instant::now();

        // Map Ising problem to optical couplings
        self.map_ising_to_optical(problem)?;

        // Reset simulation state
        self.current_time = 0.0;
        self.evolution_history.clear();
        self.energy_history.clear();

        // Main simulation loop
        let num_steps = (self.config.total_time / self.config.dt) as usize;
        let mut best_energy = f64::INFINITY;
        let mut best_solution = vec![0; problem.num_qubits];
        let mut converged = false;
        let mut convergence_time = self.config.total_time;

        for step in 0..num_steps {
            self.current_time = step as f64 * self.config.dt;

            // Update pump power according to schedule
            let pump_power = self.get_pump_power(self.current_time);
            self.update_pump_power(pump_power);

            // Evolve optical system
            self.evolve_system()?;

            // Add noise
            self.add_noise();

            // Record state
            if step % 100 == 0 || step == num_steps - 1 {
                self.record_state();
            }

            // Evaluate current solution
            let current_solution = self.measure_solution()?;
            let current_energy = self.evaluate_energy(problem, &current_solution)?;
            self.energy_history.push(current_energy);

            // Update best solution
            if current_energy < best_energy {
                best_energy = current_energy;
                best_solution = current_solution;
            }

            // Check convergence
            if !converged && self.check_convergence()? {
                converged = true;
                convergence_time = self.current_time;
                if self.config.detailed_logging {
                    println!("Converged at time {:.3}", convergence_time);
                }
            }

            // Logging
            if step % 1000 == 0 && self.config.detailed_logging {
                println!(
                    "Step {}: Time = {:.3}, Energy = {:.6}, Power = {:.3}",
                    step, self.current_time, current_energy, pump_power
                );
            }
        }

        let total_simulation_time = start_time.elapsed();

        // Calculate final statistics
        let optical_stats = self.calculate_optical_statistics();
        let performance_metrics = self.calculate_performance_metrics(best_energy, convergence_time);

        // Prepare results
        let results = CimResults {
            best_solution,
            best_energy,
            final_optical_state: self.oscillators.iter().map(|osc| osc.amplitude).collect(),
            energy_history: self.energy_history.clone(),
            optical_evolution: self
                .evolution_history
                .iter()
                .map(|(_, state)| state.clone())
                .collect(),
            time_points: self.evolution_history.iter().map(|(t, _)| *t).collect(),
            converged,
            convergence_time,
            total_simulation_time,
            optical_stats,
            performance_metrics,
        };

        println!("CIM simulation completed in {:.2?}", total_simulation_time);
        println!("Best energy: {:.6}, Converged: {}", best_energy, converged);

        Ok(results)
    }

    /// Map Ising problem to optical couplings
    fn map_ising_to_optical(&mut self, problem: &IsingModel) -> CimResult<()> {
        // Set injection fields based on biases
        for i in 0..problem.num_qubits {
            let bias = problem.get_bias(i).unwrap_or(0.0);
            self.oscillators[i].injection = Complex::new(bias * 0.1, 0.0);
        }

        // Update coupling strengths based on Ising couplings
        for coupling in &mut self.couplings {
            if let Ok(ising_coupling) = problem.get_coupling(coupling.from, coupling.to) {
                if ising_coupling != 0.0 {
                    // Map Ising coupling to optical coupling
                    let optical_strength = ising_coupling * 0.1;
                    coupling.strength = Complex::new(optical_strength, 0.0);
                }
            }
        }

        Ok(())
    }

    /// Get pump power according to schedule
    fn get_pump_power(&self, time: f64) -> f64 {
        let normalized_time = time / self.config.total_time;

        match &self.config.pump_schedule {
            PumpSchedule::Linear {
                initial_power,
                final_power,
            } => initial_power + (final_power - initial_power) * normalized_time,

            PumpSchedule::Exponential {
                initial_power,
                final_power,
                time_constant,
            } => {
                initial_power
                    + (final_power - initial_power) * (1.0 - (-time / time_constant).exp())
            }

            PumpSchedule::Sigmoid {
                initial_power,
                final_power,
                steepness,
                midpoint,
            } => {
                let sigmoid = 1.0 / (1.0 + (-(normalized_time - midpoint) * steepness).exp());
                initial_power + (final_power - initial_power) * sigmoid
            }

            PumpSchedule::Custom { power_function } => power_function(time),
        }
    }

    /// Update pump power for all oscillators
    fn update_pump_power(&mut self, pump_power: f64) {
        for osc in &mut self.oscillators {
            osc.gain = pump_power;
        }
    }

    /// Evolve the optical system by one time step
    fn evolve_system(&mut self) -> CimResult<()> {
        let dt = self.config.dt;
        let mut new_amplitudes = Vec::new();

        // Calculate evolution for each oscillator
        for i in 0..self.oscillators.len() {
            let osc = &self.oscillators[i];
            let mut derivative = Complex::new(0.0, 0.0);

            // Parametric gain term (above threshold)
            let power = osc.amplitude.magnitude_squared();
            if osc.gain > osc.threshold {
                let net_gain = osc.gain - osc.threshold - osc.loss;
                derivative = derivative + osc.amplitude * net_gain * (1.0 - power / osc.saturation);
            } else {
                // Below threshold: only loss
                derivative = derivative + osc.amplitude * (-osc.loss);
            }

            // Injection field
            derivative = derivative + osc.injection;

            // Coupling terms
            for coupling in &self.couplings {
                if coupling.to == i {
                    let source_amplitude = self.oscillators[coupling.from].amplitude;
                    derivative = derivative + source_amplitude * coupling.strength;
                }
            }

            // Integrate using Euler method
            let new_amplitude = osc.amplitude + derivative * dt;
            new_amplitudes.push(new_amplitude);
        }

        // Update amplitudes
        for (i, new_amplitude) in new_amplitudes.into_iter().enumerate() {
            self.oscillators[i].amplitude = new_amplitude;
        }

        Ok(())
    }

    /// Add noise to the optical system
    fn add_noise(&mut self) {
        let noise_config = &self.config.noise_config;

        for osc in &mut self.oscillators {
            // Quantum noise (white noise)
            let quantum_noise_re = self.rng.gen_range(-1.0..1.0) * noise_config.quantum_noise;
            let quantum_noise_im = self.rng.gen_range(-1.0..1.0) * noise_config.quantum_noise;

            // Phase noise
            let phase_noise = self.rng.gen_range(-1.0..1.0) * noise_config.phase_noise;
            let current_phase = osc.amplitude.phase();
            let new_phase = current_phase + phase_noise;

            // Amplitude noise
            let amplitude_noise =
                1.0 + self.rng.gen_range(-1.0..1.0) * noise_config.amplitude_noise;
            let current_magnitude = osc.amplitude.magnitude();
            let new_magnitude = current_magnitude * amplitude_noise;

            // Apply noise
            osc.amplitude = Complex::from_polar(new_magnitude, new_phase)
                + Complex::new(quantum_noise_re, quantum_noise_im);

            // Decoherence (amplitude damping)
            osc.amplitude = osc.amplitude * (1.0 - noise_config.decoherence_rate * self.config.dt);
        }
    }

    /// Record current state for analysis
    fn record_state(&mut self) {
        let state: Vec<Complex> = self.oscillators.iter().map(|osc| osc.amplitude).collect();
        self.evolution_history.push((self.current_time, state));
    }

    /// Measure solution from optical state
    fn measure_solution(&self) -> CimResult<Vec<i8>> {
        let mut solution = Vec::new();

        for osc in &self.oscillators {
            // Homodyne detection: measure real part (in-phase quadrature)
            let measurement =
                osc.amplitude.re * self.config.measurement_config.detection_efficiency;

            // Binary decision based on threshold
            let spin = if measurement > self.config.measurement_config.decision_threshold {
                1
            } else {
                -1
            };

            solution.push(spin);
        }

        Ok(solution)
    }

    /// Evaluate energy of a solution
    fn evaluate_energy(&self, problem: &IsingModel, solution: &[i8]) -> CimResult<f64> {
        let mut energy = 0.0;

        // Bias terms
        for i in 0..solution.len() {
            energy += problem.get_bias(i).unwrap_or(0.0) * solution[i] as f64;
        }

        // Coupling terms
        for i in 0..solution.len() {
            for j in (i + 1)..solution.len() {
                energy += problem.get_coupling(i, j).unwrap_or(0.0)
                    * solution[i] as f64
                    * solution[j] as f64;
            }
        }

        Ok(energy)
    }

    /// Check convergence criteria
    fn check_convergence(&self) -> CimResult<bool> {
        if self.energy_history.len() < 100 {
            return Ok(false);
        }

        // Check energy stability
        let recent_window = 50;
        let recent_energies = &self.energy_history[self.energy_history.len() - recent_window..];
        let energy_variance = {
            let mean = recent_energies.iter().sum::<f64>() / recent_energies.len() as f64;
            recent_energies
                .iter()
                .map(|&e| (e - mean).powi(2))
                .sum::<f64>()
                / recent_energies.len() as f64
        };

        let energy_stable = energy_variance < self.config.convergence_config.energy_tolerance;

        // Check oscillation stability
        let oscillation_stable = self.oscillators.iter().all(|osc| {
            osc.amplitude.magnitude() > self.config.convergence_config.oscillation_threshold
        });

        Ok(energy_stable && oscillation_stable)
    }

    /// Calculate optical system statistics
    fn calculate_optical_statistics(&self) -> OpticalStatistics {
        let num_oscillators = self.oscillators.len();

        // Average power
        let total_power: f64 = self
            .oscillators
            .iter()
            .map(|osc| osc.amplitude.magnitude_squared())
            .sum();
        let average_power = total_power / num_oscillators as f64;

        // Power variance
        let power_variance = {
            let powers: Vec<f64> = self
                .oscillators
                .iter()
                .map(|osc| osc.amplitude.magnitude_squared())
                .collect();
            let mean = powers.iter().sum::<f64>() / powers.len() as f64;
            powers.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / powers.len() as f64
        };

        // Phase coherence (simplified)
        let phase_coherence: Vec<f64> = self
            .oscillators
            .iter()
            .map(|osc| osc.amplitude.magnitude().min(1.0))
            .collect();

        // Cross-correlations (simplified)
        let mut cross_correlations = vec![vec![0.0; num_oscillators]; num_oscillators];
        for i in 0..num_oscillators {
            for j in 0..num_oscillators {
                if i == j {
                    cross_correlations[i][j] = 1.0;
                } else {
                    // Simplified correlation based on phase difference
                    let phase_diff = (self.oscillators[i].amplitude.phase()
                        - self.oscillators[j].amplitude.phase())
                    .abs();
                    cross_correlations[i][j] = (phase_diff / std::f64::consts::PI).cos().abs();
                }
            }
        }

        // Oscillation frequencies (estimated from evolution if available)
        let oscillation_frequencies = vec![1.0; num_oscillators]; // Placeholder

        OpticalStatistics {
            average_power,
            power_variance,
            phase_coherence,
            cross_correlations,
            oscillation_frequencies,
            pump_efficiency: 0.8, // Placeholder
        }
    }

    /// Calculate performance metrics
    fn calculate_performance_metrics(
        &self,
        best_energy: f64,
        convergence_time: f64,
    ) -> CimPerformanceMetrics {
        // Solution quality (placeholder - would need known ground state)
        let solution_quality = 1.0; // Placeholder

        // Time to convergence
        let time_to_convergence = convergence_time / self.config.total_time;

        // Power efficiency
        let average_power = self
            .oscillators
            .iter()
            .map(|osc| osc.amplitude.magnitude_squared())
            .sum::<f64>()
            / self.oscillators.len() as f64;
        let power_efficiency = 1.0 / (1.0 + average_power);

        CimPerformanceMetrics {
            solution_quality,
            time_to_convergence,
            phase_transitions: 0, // Would need to track phase transitions
            power_efficiency,
            noise_resilience: 0.8, // Placeholder
        }
    }
}

/// Helper functions for creating CIM configurations

/// Create a standard CIM configuration for small problems
pub fn create_standard_cim_config(num_oscillators: usize, simulation_time: f64) -> CimConfig {
    CimConfig {
        num_oscillators,
        dt: 0.001,
        total_time: simulation_time,
        pump_schedule: PumpSchedule::Linear {
            initial_power: 0.5,
            final_power: 1.5,
        },
        topology: NetworkTopology::FullyConnected,
        ..Default::default()
    }
}

/// Create a CIM configuration with low noise for testing
pub fn create_low_noise_cim_config(num_oscillators: usize) -> CimConfig {
    let mut config = create_standard_cim_config(num_oscillators, 5.0);
    config.noise_config = NoiseConfig {
        quantum_noise: 0.001,
        phase_noise: 0.0001,
        amplitude_noise: 0.0001,
        temperature: 0.001,
        decoherence_rate: 0.0001,
    };
    config
}

/// Create a CIM configuration with realistic noise
pub fn create_realistic_cim_config(num_oscillators: usize) -> CimConfig {
    let mut config = create_standard_cim_config(num_oscillators, 10.0);
    config.noise_config = NoiseConfig {
        quantum_noise: 0.05,
        phase_noise: 0.01,
        amplitude_noise: 0.01,
        temperature: 0.1,
        decoherence_rate: 0.01,
    };
    config.detailed_logging = true;
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let c1 = Complex::new(3.0, 4.0);
        let c2 = Complex::new(1.0, 2.0);

        assert_eq!(c1.magnitude(), 5.0);
        assert!((c1.phase() - 4.0_f64.atan2(3.0)).abs() < 1e-10);

        let sum = c1 + c2;
        assert_eq!(sum.re, 4.0);
        assert_eq!(sum.im, 6.0);

        let product = c1 * c2;
        assert_eq!(product.re, -5.0); // 3*1 - 4*2
        assert_eq!(product.im, 10.0); // 3*2 + 4*1
    }

    #[test]
    fn test_cim_config_creation() {
        let config = create_standard_cim_config(4, 5.0);
        assert_eq!(config.num_oscillators, 4);
        assert_eq!(config.total_time, 5.0);

        match config.topology {
            NetworkTopology::FullyConnected => {}
            _ => panic!("Expected fully connected topology"),
        }
    }

    #[test]
    fn test_optical_oscillator() {
        let osc = OpticalParametricOscillator {
            index: 0,
            amplitude: Complex::new(1.0, 0.0),
            gain: 1.5,
            loss: 0.1,
            saturation: 1.0,
            injection: Complex::new(0.1, 0.0),
            threshold: 1.0,
        };

        assert_eq!(osc.amplitude.magnitude(), 1.0);
        assert_eq!(osc.gain, 1.5);
        assert!(osc.gain > osc.threshold);
    }
}
