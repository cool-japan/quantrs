//! Reverse annealing support for quantum annealing
//!
//! This module implements reverse annealing schedules and techniques for
//! quantum annealing. Reverse annealing starts from a known classical state
//! and briefly increases quantum fluctuations to explore nearby states before
//! annealing forward again.

use crate::ising::{IsingModel, IsingResult, IsingError};
use crate::simulator::{AnnealingError, AnnealingResult, AnnealingSolution};
use rand::{Rng, SeedableRng, thread_rng};
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};

/// Reverse annealing schedule configuration
#[derive(Debug, Clone)]
pub struct ReverseAnnealingSchedule {
    /// Starting s-parameter (0 = full quantum, 1 = full classical)
    pub s_start: f64,
    /// Target s-parameter for reversal (how far back to reverse)
    pub s_target: f64,
    /// Pause duration at s_target (as fraction of total time)
    pub pause_duration: f64,
    /// Quench rate for forward annealing
    pub quench_rate: f64,
    /// Hold time at the end (as fraction of total time)
    pub hold_duration: f64,
}

impl Default for ReverseAnnealingSchedule {
    fn default() -> Self {
        Self {
            s_start: 1.0,      // Start from classical state
            s_target: 0.45,    // Reverse to 45% classical
            pause_duration: 0.1, // Pause for 10% of total time
            quench_rate: 1.0,   // Normal quench rate
            hold_duration: 0.0,  // No hold at end
        }
    }
}

impl ReverseAnnealingSchedule {
    /// Create a new reverse annealing schedule
    pub fn new(s_target: f64, pause_duration: f64) -> AnnealingResult<Self> {
        if !(0.0..=1.0).contains(&s_target) {
            return Err(AnnealingError::InvalidSchedule(
                format!("s_target must be in [0,1], got {}", s_target)
            ));
        }
        if !(0.0..=1.0).contains(&pause_duration) {
            return Err(AnnealingError::InvalidSchedule(
                format!("pause_duration must be in [0,1], got {}", pause_duration)
            ));
        }
        
        Ok(Self {
            s_start: 1.0,
            s_target,
            pause_duration,
            quench_rate: 1.0,
            hold_duration: 0.0,
        })
    }

    /// Calculate s(t) for a given normalized time
    pub fn s_of_t(&self, t_normalized: f64) -> f64 {
        // t_normalized is in [0, 1]
        let t1 = (1.0 - self.pause_duration - self.hold_duration) / 2.0;
        let t2 = t1 + self.pause_duration;
        let t3 = 1.0 - self.hold_duration;
        
        if t_normalized <= t1 {
            // Reverse annealing phase
            self.s_start + (self.s_target - self.s_start) * (t_normalized / t1)
        } else if t_normalized <= t2 {
            // Pause phase
            self.s_target
        } else if t_normalized <= t3 {
            // Forward annealing phase
            let forward_progress = (t_normalized - t2) / (t3 - t2);
            self.s_target + (1.0 - self.s_target) * forward_progress * self.quench_rate
        } else {
            // Hold phase
            1.0
        }
    }

    /// Calculate transverse field strength A(s)
    pub fn transverse_field(&self, s: f64) -> f64 {
        // Standard D-Wave schedule: A(s) = A(0) * (1 - s)
        let a_max = 2.0; // Maximum transverse field strength
        a_max * (1.0 - s)
    }

    /// Calculate problem Hamiltonian strength B(s)
    pub fn problem_strength(&self, s: f64) -> f64 {
        // Standard D-Wave schedule: B(s) = B(1) * s
        let b_max = 1.0; // Maximum problem strength
        b_max * s
    }
}

/// Reverse annealing parameters
#[derive(Debug, Clone)]
pub struct ReverseAnnealingParams {
    /// Reverse annealing schedule
    pub schedule: ReverseAnnealingSchedule,
    /// Initial state (classical solution to start from)
    pub initial_state: Vec<i8>,
    /// Number of Monte Carlo sweeps
    pub num_sweeps: usize,
    /// Number of repetitions
    pub num_repetitions: usize,
    /// Random seed
    pub seed: Option<u64>,
    /// Reinitialize fraction (fraction of spins to randomize)
    pub reinitialize_fraction: f64,
    /// Local search radius (for targeted reverse annealing)
    pub local_search_radius: Option<usize>,
}

impl ReverseAnnealingParams {
    /// Create new reverse annealing parameters
    pub fn new(initial_state: Vec<i8>) -> Self {
        Self {
            schedule: ReverseAnnealingSchedule::default(),
            initial_state,
            num_sweeps: 1000,
            num_repetitions: 10,
            seed: None,
            reinitialize_fraction: 0.0,
            local_search_radius: None,
        }
    }

    /// Set targeted reverse annealing with local search
    pub fn with_local_search(mut self, radius: usize) -> Self {
        self.local_search_radius = Some(radius);
        self
    }

    /// Set partial reinitialization
    pub fn with_reinitialization(mut self, fraction: f64) -> Self {
        self.reinitialize_fraction = fraction.clamp(0.0, 1.0);
        self
    }
}

/// Reverse annealing simulator
pub struct ReverseAnnealingSimulator {
    params: ReverseAnnealingParams,
    rng: ChaCha8Rng,
}

impl ReverseAnnealingSimulator {
    /// Create a new reverse annealing simulator
    pub fn new(params: ReverseAnnealingParams) -> AnnealingResult<Self> {
        let rng = match params.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(thread_rng().gen()),
        };

        Ok(Self { params, rng })
    }

    /// Solve an Ising model using reverse annealing
    pub fn solve(&mut self, model: &IsingModel) -> AnnealingResult<AnnealingSolution> {
        let start_time = Instant::now();
        let num_qubits = model.num_qubits;
        
        // Validate initial state
        if self.params.initial_state.len() != num_qubits {
            return Err(AnnealingError::InvalidParameter(
                format!("Initial state length {} doesn't match model size {}", 
                    self.params.initial_state.len(), num_qubits)
            ));
        }

        let mut best_solution = self.params.initial_state.clone();
        let mut best_energy = model.energy(&best_solution)
            .map_err(AnnealingError::IsingError)?;
        
        let mut all_solutions = Vec::new();
        let mut all_energies = Vec::new();

        for rep in 0..self.params.num_repetitions {
            // Initialize state
            let initial_state = self.params.initial_state.clone();
            let mut state = self.prepare_initial_state(&initial_state);
            
            // Run reverse annealing
            let solution = self.run_reverse_annealing(model, &mut state)?;
            let energy = model.energy(&solution)
                .map_err(AnnealingError::IsingError)?;
            
            all_solutions.push(solution.clone());
            all_energies.push(energy);
            
            if energy < best_energy {
                best_energy = energy;
                best_solution = solution;
            }
        }

        let elapsed = start_time.elapsed();

        Ok(AnnealingSolution {
            best_spins: best_solution,
            best_energy,
            repetitions: self.params.num_repetitions,
            total_sweeps: self.params.num_sweeps * self.params.num_repetitions,
            runtime: elapsed,
            info: format!(
                "Reverse annealing with {} repetitions, {} sweeps each, s_target={}",
                self.params.num_repetitions,
                self.params.num_sweeps,
                self.params.schedule.s_target
            ),
        })
    }

    /// Prepare initial state with optional reinitialization
    fn prepare_initial_state(&mut self, base_state: &[i8]) -> Vec<i8> {
        let mut state = base_state.to_vec();
        
        // Apply partial reinitialization
        if self.params.reinitialize_fraction > 0.0 {
            let num_to_reinit = (state.len() as f64 * self.params.reinitialize_fraction) as usize;
            for _ in 0..num_to_reinit {
                let idx = self.rng.gen_range(0..state.len());
                state[idx] = if self.rng.gen_bool(0.5) { 1 } else { -1 };
            }
        }
        
        // Apply local search mask if specified
        if let Some(radius) = self.params.local_search_radius {
            self.apply_local_search_mask(&mut state, radius);
        }
        
        state
    }

    /// Apply local search mask for targeted reverse annealing
    fn apply_local_search_mask(&mut self, state: &mut [i8], radius: usize) {
        // In targeted reverse annealing, only spins within radius are allowed to change
        // This is typically implemented using anneal_offsets in hardware
        // For simulation, we'll mark which spins can be updated
        
        // For now, we'll implement a simple version where we select random centers
        let num_centers = (state.len() as f64 * 0.1).max(1.0) as usize;
        let mut can_update = vec![false; state.len()];
        
        for _ in 0..num_centers {
            let center = self.rng.gen_range(0..state.len());
            for i in 0..state.len() {
                if (i as i32 - center as i32).abs() <= radius as i32 {
                    can_update[i] = true;
                }
            }
        }
        
        // Store mask for use during annealing
        // (In a full implementation, this would affect the update selection)
    }

    /// Run the reverse annealing process
    fn run_reverse_annealing(
        &mut self, 
        model: &IsingModel, 
        state: &mut Vec<i8>
    ) -> AnnealingResult<Vec<i8>> {
        let schedule = &self.params.schedule;
        
        for sweep in 0..self.params.num_sweeps {
            // Calculate normalized time
            let t_norm = sweep as f64 / self.params.num_sweeps as f64;
            
            // Get s-parameter from schedule
            let s = schedule.s_of_t(t_norm);
            
            // Calculate effective fields
            let transverse_field = schedule.transverse_field(s);
            let problem_strength = schedule.problem_strength(s);
            
            // Perform Monte Carlo updates
            for _ in 0..model.num_qubits {
                let i = self.rng.gen_range(0..model.num_qubits);
                
                // Calculate local field
                let mut h_local = 0.0;
                
                // Add bias term
                if let Ok(bias) = model.get_bias(i) {
                    h_local += bias * problem_strength;
                }
                
                // Add coupling terms
                for j in 0..model.num_qubits {
                    if i != j {
                        if let Ok(coupling) = model.get_coupling(i, j) {
                            h_local += coupling * state[j] as f64 * problem_strength;
                        }
                    }
                }
                
                // Add transverse field term (simplified)
                let quantum_term = transverse_field;
                
                // Calculate energy difference for flip
                let delta_e = 2.0 * state[i] as f64 * h_local;
                
                // Metropolis acceptance with quantum fluctuations
                let effective_temp = 0.1 + quantum_term * 0.5; // Simplified
                let accept_prob = (-delta_e / effective_temp).exp().min(1.0);
                
                if self.rng.gen_bool(accept_prob) {
                    state[i] *= -1;
                }
            }
        }
        
        Ok(state.clone())
    }
}

/// Builder for reverse annealing schedules
pub struct ReverseAnnealingScheduleBuilder {
    s_target: f64,
    pause_duration: f64,
    quench_rate: f64,
    hold_duration: f64,
}

impl ReverseAnnealingScheduleBuilder {
    /// Create a new schedule builder
    pub fn new() -> Self {
        Self {
            s_target: 0.45,
            pause_duration: 0.1,
            quench_rate: 1.0,
            hold_duration: 0.0,
        }
    }

    /// Set the target s-parameter for reversal
    pub fn s_target(mut self, s: f64) -> Self {
        self.s_target = s;
        self
    }

    /// Set the pause duration
    pub fn pause_duration(mut self, duration: f64) -> Self {
        self.pause_duration = duration;
        self
    }

    /// Set the quench rate
    pub fn quench_rate(mut self, rate: f64) -> Self {
        self.quench_rate = rate;
        self
    }

    /// Set the hold duration
    pub fn hold_duration(mut self, duration: f64) -> Self {
        self.hold_duration = duration;
        self
    }

    /// Build the schedule
    pub fn build(self) -> AnnealingResult<ReverseAnnealingSchedule> {
        if !(0.0..=1.0).contains(&self.s_target) {
            return Err(AnnealingError::InvalidSchedule(
                "s_target must be in [0,1]".to_string()
            ));
        }
        
        Ok(ReverseAnnealingSchedule {
            s_start: 1.0,
            s_target: self.s_target,
            pause_duration: self.pause_duration,
            quench_rate: self.quench_rate,
            hold_duration: self.hold_duration,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_schedule_creation() {
        let schedule = ReverseAnnealingSchedule::new(0.45, 0.1).unwrap();
        assert_eq!(schedule.s_start, 1.0);
        assert_eq!(schedule.s_target, 0.45);
    }

    #[test]
    fn test_schedule_s_of_t() {
        let schedule = ReverseAnnealingSchedule::default();
        
        // At t=0, should be at s_start
        assert!((schedule.s_of_t(0.0) - 1.0).abs() < 1e-6);
        
        // At midpoint of reversal, should be between s_start and s_target
        let mid = 0.45 / 2.0;
        let s_mid = schedule.s_of_t(mid);
        assert!(s_mid > schedule.s_target && s_mid < schedule.s_start);
        
        // At end, should be back at 1.0
        assert!((schedule.s_of_t(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reverse_annealing_params() {
        let initial_state = vec![1, -1, 1, -1];
        let params = ReverseAnnealingParams::new(initial_state.clone())
            .with_local_search(2)
            .with_reinitialization(0.25);
        
        assert_eq!(params.initial_state, initial_state);
        assert_eq!(params.local_search_radius, Some(2));
        assert_eq!(params.reinitialize_fraction, 0.25);
    }
}