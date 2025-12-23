//! Quantum Monte Carlo (QMC) simulation methods.
//!
//! This module implements various QMC algorithms for simulating quantum systems,
//! including Variational Monte Carlo (VMC) and Diffusion Monte Carlo (DMC).

use crate::prelude::SimulatorError;
use fastrand;
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::Complex64;

use crate::error::Result;
use crate::trotter::{Hamiltonian, HamiltonianTerm};

/// Walker in QMC simulation
#[derive(Debug, Clone)]
pub struct Walker {
    /// Configuration (bit string representation)
    pub config: Vec<bool>,
    /// Weight/amplitude
    pub weight: Complex64,
    /// Local energy
    pub local_energy: Complex64,
}

impl Walker {
    /// Create a new walker
    pub fn new(num_qubits: usize) -> Self {
        let mut config = vec![false; num_qubits];
        // Random initial configuration
        for bit in &mut config {
            *bit = fastrand::bool();
        }

        Self {
            config,
            weight: Complex64::new(1.0, 0.0),
            local_energy: Complex64::new(0.0, 0.0),
        }
    }

    /// Flip a qubit in the configuration
    pub fn flip(&mut self, qubit: usize) {
        if qubit < self.config.len() {
            self.config[qubit] = !self.config[qubit];
        }
    }

    /// Get configuration as integer
    pub fn as_integer(&self) -> usize {
        let mut result = 0;
        for (i, &bit) in self.config.iter().enumerate() {
            if bit {
                result |= 1 << i;
            }
        }
        result
    }
}

/// Wave function ansatz for VMC
#[derive(Debug, Clone)]
pub enum WaveFunction {
    /// Product state
    Product(Vec<Complex64>),
    /// Jastrow factor
    Jastrow { alpha: f64, beta: f64 },
    /// Neural network quantum state (simplified)
    NeuralNetwork {
        weights: Array2<f64>,
        biases: Array1<f64>,
    },
    /// Matrix product state
    MPS {
        tensors: Vec<Array3<Complex64>>,
        bond_dim: usize,
    },
}

impl WaveFunction {
    /// Evaluate wave function amplitude for a configuration
    pub fn amplitude(&self, config: &[bool]) -> Complex64 {
        match self {
            Self::Product(amps) => {
                let mut result = Complex64::new(1.0, 0.0);
                for (i, &bit) in config.iter().enumerate() {
                    if i < amps.len() {
                        result *= if bit {
                            amps[i]
                        } else {
                            Complex64::new(1.0, 0.0) - amps[i]
                        };
                    }
                }
                result
            }
            Self::Jastrow { alpha, beta } => {
                // Jastrow factor: exp(sum_ij J_ij n_i n_j)
                let mut exponent = 0.0;
                for (i, &n_i) in config.iter().enumerate() {
                    if n_i {
                        exponent += alpha;
                        for (j, &n_j) in config.iter().enumerate() {
                            if i != j && n_j {
                                exponent += beta / (1.0 + (i as f64 - j as f64).abs());
                            }
                        }
                    }
                }
                Complex64::new(exponent.exp(), 0.0)
            }
            Self::NeuralNetwork { weights, biases } => {
                // Simplified RBM-like network
                let input: Vec<f64> = config.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
                let hidden_dim = weights.nrows();

                let mut hidden = vec![0.0; hidden_dim];
                for h in 0..hidden_dim {
                    let mut sum = biases[h];
                    for (v, &x) in input.iter().enumerate() {
                        if v < weights.ncols() {
                            sum += weights[[h, v]] * x;
                        }
                    }
                    hidden[h] = 1.0 / (1.0 + (-sum).exp()); // Sigmoid
                }

                let mut log_psi = 0.0;
                for &h in &hidden {
                    log_psi += h.ln_1p();
                }
                Complex64::new(log_psi.exp(), 0.0)
            }
            Self::MPS { .. } => {
                // Simplified MPS evaluation
                Complex64::new(1.0, 0.0)
            }
        }
    }

    /// Compute log derivative for parameter optimization
    pub fn log_derivative(&self, config: &[bool], param_idx: usize) -> f64 {
        match self {
            Self::Jastrow { alpha, beta } => {
                // Gradient w.r.t. Jastrow parameters
                if param_idx == 0 {
                    // d/d(alpha)
                    config.iter().filter(|&&b| b).count() as f64
                } else {
                    // d/d(beta)
                    let mut sum = 0.0;
                    for (i, &n_i) in config.iter().enumerate() {
                        if n_i {
                            for (j, &n_j) in config.iter().enumerate() {
                                if i != j && n_j {
                                    sum += 1.0 / (1.0 + (i as f64 - j as f64).abs());
                                }
                            }
                        }
                    }
                    sum
                }
            }
            _ => 0.0, // Simplified for other types
        }
    }
}

/// Variational Monte Carlo simulator
pub struct VMC {
    /// Wave function ansatz
    wave_function: WaveFunction,
    /// Number of qubits
    num_qubits: usize,
    /// Hamiltonian
    hamiltonian: Hamiltonian,
}

impl VMC {
    /// Create a new VMC simulator
    pub const fn new(
        num_qubits: usize,
        wave_function: WaveFunction,
        hamiltonian: Hamiltonian,
    ) -> Self {
        Self {
            wave_function,
            num_qubits,
            hamiltonian,
        }
    }

    /// Run VMC simulation
    pub fn run(
        &mut self,
        num_samples: usize,
        num_thermalization: usize,
        optimization_steps: usize,
        learning_rate: f64,
    ) -> Result<VMCResult> {
        let mut energies = Vec::new();
        let mut variances = Vec::new();

        for step in 0..optimization_steps {
            // Thermalization
            let mut walker = Walker::new(self.num_qubits);
            for _ in 0..num_thermalization {
                self.metropolis_step(&mut walker)?;
            }

            // Sampling
            let mut local_energies = Vec::new();
            let mut gradients = [0.0; 2]; // For Jastrow parameters

            for _ in 0..num_samples {
                self.metropolis_step(&mut walker)?;

                // Compute local energy
                let e_loc = self.local_energy(&walker.config)?;
                local_energies.push(e_loc);

                // Compute gradients for optimization
                if let WaveFunction::Jastrow { .. } = &self.wave_function {
                    for p in 0..2 {
                        let deriv = self.wave_function.log_derivative(&walker.config, p);
                        gradients[p] += (e_loc.re
                            - local_energies.iter().map(|e| e.re).sum::<f64>()
                                / local_energies.len() as f64)
                            * deriv;
                    }
                }
            }

            // Statistics
            let mean_energy = local_energies.iter().map(|e| e.re).sum::<f64>() / num_samples as f64;
            let variance = local_energies
                .iter()
                .map(|e| (e.re - mean_energy).powi(2))
                .sum::<f64>()
                / num_samples as f64;

            energies.push(mean_energy);
            variances.push(variance);

            // Parameter update (gradient descent)
            if let WaveFunction::Jastrow {
                ref mut alpha,
                ref mut beta,
            } = &mut self.wave_function
            {
                *alpha -= learning_rate * gradients[0] / num_samples as f64;
                *beta -= learning_rate * gradients[1] / num_samples as f64;
            }

            // Print progress
            if step % 10 == 0 {
                println!(
                    "VMC Step {}: E = {:.6} ± {:.6}",
                    step,
                    mean_energy,
                    variance.sqrt()
                );
            }
        }

        Ok(VMCResult {
            final_energy: *energies.last().unwrap(),
            energy_history: energies,
            variance_history: variances,
        })
    }

    /// Metropolis step
    fn metropolis_step(&self, walker: &mut Walker) -> Result<()> {
        // Propose move: flip random qubit
        let qubit = fastrand::usize(..self.num_qubits);
        let old_config = walker.config.clone();
        walker.flip(qubit);

        // Compute acceptance ratio
        let old_amp = self.wave_function.amplitude(&old_config);
        let new_amp = self.wave_function.amplitude(&walker.config);
        let ratio = (new_amp.norm() / old_amp.norm()).powi(2);

        // Accept or reject
        if fastrand::f64() >= ratio {
            walker.config = old_config; // Reject
        }

        Ok(())
    }

    /// Compute local energy
    fn local_energy(&self, config: &[bool]) -> Result<Complex64> {
        let psi = self.wave_function.amplitude(config);
        if psi.norm() < 1e-15 {
            return Ok(Complex64::new(0.0, 0.0));
        }

        let mut h_psi = Complex64::new(0.0, 0.0);

        // Apply Hamiltonian terms
        for term in &self.hamiltonian.terms {
            match term {
                HamiltonianTerm::SinglePauli {
                    qubit,
                    pauli,
                    coefficient,
                } => {
                    match pauli.as_str() {
                        "Z" => {
                            // Diagonal term
                            let sign = if config[*qubit] { 1.0 } else { -1.0 };
                            h_psi += coefficient * sign * psi;
                        }
                        "X" => {
                            // Off-diagonal: flip qubit
                            let mut flipped = config.to_vec();
                            flipped[*qubit] = !flipped[*qubit];
                            let psi_flipped = self.wave_function.amplitude(&flipped);
                            h_psi += coefficient * psi_flipped;
                        }
                        "Y" => {
                            // Off-diagonal with phase
                            let mut flipped = config.to_vec();
                            flipped[*qubit] = !flipped[*qubit];
                            let psi_flipped = self.wave_function.amplitude(&flipped);
                            let phase = if config[*qubit] {
                                Complex64::new(0.0, -1.0)
                            } else {
                                Complex64::new(0.0, 1.0)
                            };
                            h_psi += coefficient * phase * psi_flipped;
                        }
                        _ => {}
                    }
                }
                HamiltonianTerm::TwoPauli {
                    qubit1,
                    qubit2,
                    pauli1,
                    pauli2,
                    coefficient,
                } => {
                    // Two-qubit terms (simplified)
                    if pauli1 == "Z" && pauli2 == "Z" {
                        let sign1 = if config[*qubit1] { 1.0 } else { -1.0 };
                        let sign2 = if config[*qubit2] { 1.0 } else { -1.0 };
                        h_psi += coefficient * sign1 * sign2 * psi;
                    }
                }
                _ => {} // Other terms not implemented in this simplified version
            }
        }

        Ok(h_psi / psi)
    }
}

/// VMC simulation result
#[derive(Debug)]
pub struct VMCResult {
    /// Final energy
    pub final_energy: f64,
    /// Energy history
    pub energy_history: Vec<f64>,
    /// Variance history
    pub variance_history: Vec<f64>,
}

/// Diffusion Monte Carlo simulator
pub struct DMC {
    /// Reference energy
    reference_energy: f64,
    /// Time step
    tau: f64,
    /// Target walker number
    target_walkers: usize,
    /// Hamiltonian
    hamiltonian: Hamiltonian,
    /// Number of qubits
    num_qubits: usize,
}

impl DMC {
    /// Create a new DMC simulator
    pub const fn new(
        num_qubits: usize,
        hamiltonian: Hamiltonian,
        tau: f64,
        target_walkers: usize,
    ) -> Self {
        Self {
            reference_energy: 0.0,
            tau,
            target_walkers,
            hamiltonian,
            num_qubits,
        }
    }

    /// Run DMC simulation
    pub fn run(&mut self, num_blocks: usize, steps_per_block: usize) -> Result<DMCResult> {
        // Initialize walkers
        let mut walkers: Vec<Walker> = (0..self.target_walkers)
            .map(|_| Walker::new(self.num_qubits))
            .collect();

        let mut energies = Vec::new();
        let mut walker_counts = Vec::new();

        for block in 0..num_blocks {
            let mut block_energy = 0.0;
            let mut total_weight = 0.0;

            for _ in 0..steps_per_block {
                // Propagate walkers
                let mut new_walkers = Vec::new();

                for walker in &walkers {
                    // Diffusion step
                    let mut new_walker = walker.clone();
                    self.diffusion_step(&mut new_walker)?;

                    // Branching
                    let local_e = self.diagonal_energy(&new_walker.config)?;
                    let growth_factor = (-self.tau * (local_e - self.reference_energy)).exp();
                    let num_copies = self.branch(growth_factor);

                    for _ in 0..num_copies {
                        new_walkers.push(new_walker.clone());
                    }

                    block_energy += local_e * walker.weight.norm();
                    total_weight += walker.weight.norm();
                }

                // Ensure at least one walker survives
                if new_walkers.is_empty() {
                    // Keep at least one walker from the previous generation
                    new_walkers.push(walkers[0].clone());
                }

                walkers = new_walkers;

                // Population control
                self.population_control(&mut walkers)?;
            }

            // Record statistics
            let avg_energy = block_energy / total_weight;
            energies.push(avg_energy);
            walker_counts.push(walkers.len());

            // Update reference energy
            self.reference_energy =
                avg_energy - (walkers.len() as f64 - self.target_walkers as f64).ln() / self.tau;

            if block % 10 == 0 {
                println!(
                    "DMC Block {}: E = {:.6}, Walkers = {}",
                    block,
                    avg_energy,
                    walkers.len()
                );
            }
        }

        Ok(DMCResult {
            ground_state_energy: *energies.last().unwrap(),
            energy_history: energies,
            walker_history: walker_counts,
        })
    }

    /// Diffusion step (random walk)
    fn diffusion_step(&self, walker: &mut Walker) -> Result<()> {
        // Simple diffusion: flip random qubits
        let num_flips = fastrand::usize(1..=3.min(self.num_qubits));
        for _ in 0..num_flips {
            let qubit = fastrand::usize(..self.num_qubits);
            walker.flip(qubit);
        }
        Ok(())
    }

    /// Compute diagonal energy
    fn diagonal_energy(&self, config: &[bool]) -> Result<f64> {
        let mut energy = 0.0;

        for term in &self.hamiltonian.terms {
            match term {
                HamiltonianTerm::SinglePauli {
                    qubit,
                    pauli,
                    coefficient,
                } => {
                    if pauli == "Z" {
                        let sign = if config[*qubit] { 1.0 } else { -1.0 };
                        energy += coefficient * sign;
                    }
                }
                HamiltonianTerm::TwoPauli {
                    qubit1,
                    qubit2,
                    pauli1,
                    pauli2,
                    coefficient,
                } => {
                    if pauli1 == "Z" && pauli2 == "Z" {
                        let sign1 = if config[*qubit1] { 1.0 } else { -1.0 };
                        let sign2 = if config[*qubit2] { 1.0 } else { -1.0 };
                        energy += coefficient * sign1 * sign2;
                    }
                }
                _ => {}
            }
        }

        Ok(energy)
    }

    /// Branching process
    fn branch(&self, growth_factor: f64) -> usize {
        // Ensure growth factor is reasonable to prevent walker extinction
        let clamped_factor = growth_factor.clamp(0.1, 3.0);
        let expected = clamped_factor;
        let integer_part = expected.floor() as usize;
        let fractional_part = expected - integer_part as f64;

        if fastrand::f64() < fractional_part {
            integer_part + 1
        } else {
            integer_part
        }
    }

    /// Population control
    fn population_control(&self, walkers: &mut Vec<Walker>) -> Result<()> {
        let current_size = walkers.len();

        if current_size == 0 {
            return Err(SimulatorError::ComputationError(
                "All walkers died".to_string(),
            ));
        }

        // Simple comb method
        if current_size > 2 * self.target_walkers {
            // Remove every other walker
            let mut new_walkers = Vec::new();
            for (i, walker) in walkers.iter().enumerate() {
                if i % 2 == 0 {
                    let mut w = walker.clone();
                    w.weight *= Complex64::new(2.0, 0.0);
                    new_walkers.push(w);
                }
            }
            *walkers = new_walkers;
        } else if current_size < self.target_walkers / 2 {
            // Duplicate walkers
            let mut new_walkers = walkers.clone();
            for walker in walkers.iter() {
                let mut w = walker.clone();
                w.weight *= Complex64::new(0.5, 0.0);
                new_walkers.push(w);
            }
            *walkers = new_walkers;
        }

        Ok(())
    }
}

/// DMC simulation result
#[derive(Debug)]
pub struct DMCResult {
    /// Ground state energy
    pub ground_state_energy: f64,
    /// Energy history
    pub energy_history: Vec<f64>,
    /// Walker count history
    pub walker_history: Vec<usize>,
}

/// Path Integral Monte Carlo
pub struct PIMC {
    /// Number of imaginary time slices
    num_slices: usize,
    /// Inverse temperature
    beta: f64,
    /// Number of qubits
    num_qubits: usize,
    /// Hamiltonian
    hamiltonian: Hamiltonian,
}

impl PIMC {
    /// Create a new PIMC simulator
    pub const fn new(
        num_qubits: usize,
        hamiltonian: Hamiltonian,
        beta: f64,
        num_slices: usize,
    ) -> Self {
        Self {
            num_slices,
            beta,
            num_qubits,
            hamiltonian,
        }
    }

    /// Run PIMC simulation
    pub fn run(&self, num_samples: usize, num_thermalization: usize) -> Result<PIMCResult> {
        // Initialize path (world line configuration)
        let mut path: Vec<Vec<bool>> = (0..self.num_slices)
            .map(|_| (0..self.num_qubits).map(|_| fastrand::bool()).collect())
            .collect();

        let tau = self.beta / self.num_slices as f64;
        let mut energies = Vec::new();
        let mut magnetizations = Vec::new();

        // Thermalization
        for _ in 0..num_thermalization {
            self.update_path(&mut path, tau)?;
        }

        // Sampling
        for _ in 0..num_samples {
            self.update_path(&mut path, tau)?;

            // Measure observables
            let energy = self.measure_energy(&path)?;
            let magnetization = self.measure_magnetization(&path);

            energies.push(energy);
            magnetizations.push(magnetization);
        }

        Ok(PIMCResult {
            average_energy: energies.iter().sum::<f64>() / energies.len() as f64,
            average_magnetization: magnetizations.iter().sum::<f64>() / magnetizations.len() as f64,
            energy_samples: energies,
            magnetization_samples: magnetizations,
        })
    }

    /// Update path configuration
    fn update_path(&self, path: &mut Vec<Vec<bool>>, tau: f64) -> Result<()> {
        // World line updates
        for _ in 0..self.num_qubits * self.num_slices {
            let slice = fastrand::usize(..self.num_slices);
            let qubit = fastrand::usize(..self.num_qubits);

            // Compute action change
            let action_old = self.path_action(path, tau)?;
            path[slice][qubit] = !path[slice][qubit];
            let action_new = self.path_action(path, tau)?;

            // Metropolis acceptance
            if fastrand::f64() >= (-(action_new - action_old)).exp() {
                path[slice][qubit] = !path[slice][qubit]; // Reject
            }
        }

        Ok(())
    }

    /// Compute path action
    fn path_action(&self, path: &[Vec<bool>], tau: f64) -> Result<f64> {
        let mut action = 0.0;

        // Kinetic term (periodic boundary conditions)
        for s in 0..self.num_slices {
            let next_s = (s + 1) % self.num_slices;
            for q in 0..self.num_qubits {
                if path[s][q] != path[next_s][q] {
                    action += -0.5 * tau.ln();
                }
            }
        }

        // Potential term
        for s in 0..self.num_slices {
            action += tau * self.diagonal_energy(&path[s])?;
        }

        Ok(action)
    }

    /// Measure energy
    fn measure_energy(&self, path: &[Vec<bool>]) -> Result<f64> {
        let mut total = 0.0;
        for config in path {
            total += self.diagonal_energy(config)?;
        }
        Ok(total / self.num_slices as f64)
    }

    /// Measure magnetization
    fn measure_magnetization(&self, path: &[Vec<bool>]) -> f64 {
        let mut total = 0.0;
        for config in path {
            let mag: f64 = config.iter().map(|&b| if b { 1.0 } else { -1.0 }).sum();
            total += mag;
        }
        total / (self.num_slices * self.num_qubits) as f64
    }

    /// Compute diagonal energy (same as DMC)
    fn diagonal_energy(&self, config: &[bool]) -> Result<f64> {
        let mut energy = 0.0;

        for term in &self.hamiltonian.terms {
            match term {
                HamiltonianTerm::SinglePauli {
                    qubit,
                    pauli,
                    coefficient,
                } => {
                    if pauli == "Z" {
                        let sign = if config[*qubit] { 1.0 } else { -1.0 };
                        energy += coefficient * sign;
                    }
                }
                HamiltonianTerm::TwoPauli {
                    qubit1,
                    qubit2,
                    pauli1,
                    pauli2,
                    coefficient,
                } => {
                    if pauli1 == "Z" && pauli2 == "Z" {
                        let sign1 = if config[*qubit1] { 1.0 } else { -1.0 };
                        let sign2 = if config[*qubit2] { 1.0 } else { -1.0 };
                        energy += coefficient * sign1 * sign2;
                    }
                }
                _ => {}
            }
        }

        Ok(energy)
    }
}

/// PIMC simulation result
#[derive(Debug)]
pub struct PIMCResult {
    /// Average energy
    pub average_energy: f64,
    /// Average magnetization
    pub average_magnetization: f64,
    /// Energy samples
    pub energy_samples: Vec<f64>,
    /// Magnetization samples
    pub magnetization_samples: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trotter::HamiltonianLibrary;

    #[test]
    fn test_walker() {
        let walker = Walker::new(4);
        assert_eq!(walker.config.len(), 4);
        assert_eq!(walker.weight, Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_wave_function_product() {
        let amps = vec![Complex64::new(0.7, 0.0), Complex64::new(0.6, 0.0)];
        let wf = WaveFunction::Product(amps);

        let config = vec![true, false];
        let amp = wf.amplitude(&config);
        assert!(0.7f64.mul_add(-0.4, amp.norm()).abs() < 1e-10);
    }

    #[test]
    fn test_vmc_ising() {
        let ham = HamiltonianLibrary::transverse_ising_1d(3, 1.0, 0.5, false).unwrap();
        let wf = WaveFunction::Jastrow {
            alpha: 0.5,
            beta: 0.1,
        };
        let mut vmc = VMC::new(3, wf, ham);

        let result = vmc.run(100, 50, 10, 0.01).unwrap();
        assert!(result.final_energy.is_finite());
    }

    #[test]
    fn test_dmc_simple() {
        let ham = HamiltonianLibrary::transverse_ising_1d(2, 1.0, 1.0, false).unwrap();
        // Use larger time step and fewer walkers for more stable test
        let mut dmc = DMC::new(2, ham, 0.1, 50);

        let result = dmc.run(5, 5).unwrap();
        assert!(result.ground_state_energy.is_finite());
    }

    #[test]
    fn test_pimc_thermal() {
        let ham = HamiltonianLibrary::xy_model(3, 1.0, true).unwrap();
        let pimc = PIMC::new(3, ham, 1.0, 10);

        let result = pimc.run(100, 50).unwrap();
        assert!(result.average_energy.is_finite());
    }
}
