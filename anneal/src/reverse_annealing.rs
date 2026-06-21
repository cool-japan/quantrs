//! Reverse annealing support for quantum annealing
//!
//! This module implements reverse annealing schedules and techniques for
//! quantum annealing. Reverse annealing starts from a known classical state
//! and briefly increases quantum fluctuations to explore nearby states before
//! annealing forward again.

use crate::ising::{IsingError, IsingModel, IsingResult};
use crate::simulator::{AnnealingError, AnnealingResult, AnnealingSolution};
use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::{thread_rng, Rng, RngExt, SeedableRng};
use std::time::{Duration, Instant};

/// Reverse annealing schedule configuration
#[derive(Debug, Clone)]
pub struct ReverseAnnealingSchedule {
    /// Starting s-parameter (0 = full quantum, 1 = full classical)
    pub s_start: f64,
    /// Target s-parameter for reversal (how far back to reverse)
    pub s_target: f64,
    /// Pause duration at `s_target` (as fraction of total time)
    pub pause_duration: f64,
    /// Quench rate for forward annealing
    pub quench_rate: f64,
    /// Hold time at the end (as fraction of total time)
    pub hold_duration: f64,
}

impl Default for ReverseAnnealingSchedule {
    fn default() -> Self {
        Self {
            s_start: 1.0,        // Start from classical state
            s_target: 0.45,      // Reverse to 45% classical
            pause_duration: 0.1, // Pause for 10% of total time
            quench_rate: 1.0,    // Normal quench rate
            hold_duration: 0.0,  // No hold at end
        }
    }
}

impl ReverseAnnealingSchedule {
    /// Create a new reverse annealing schedule
    pub fn new(s_target: f64, pause_duration: f64) -> AnnealingResult<Self> {
        if !(0.0..=1.0).contains(&s_target) {
            return Err(AnnealingError::InvalidSchedule(format!(
                "s_target must be in [0,1], got {s_target}"
            )));
        }
        if !(0.0..=1.0).contains(&pause_duration) {
            return Err(AnnealingError::InvalidSchedule(format!(
                "pause_duration must be in [0,1], got {pause_duration}"
            )));
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
    #[must_use]
    pub fn s_of_t(&self, t_normalized: f64) -> f64 {
        // t_normalized is in [0, 1]
        let t1 = (1.0 - self.pause_duration - self.hold_duration) / 2.0;
        let t2 = t1 + self.pause_duration;
        let t3 = 1.0 - self.hold_duration;

        if t_normalized <= t1 {
            // Reverse annealing phase
            (self.s_target - self.s_start).mul_add(t_normalized / t1, self.s_start)
        } else if t_normalized <= t2 {
            // Pause phase
            self.s_target
        } else if t_normalized <= t3 {
            // Forward annealing phase
            let forward_progress = (t_normalized - t2) / (t3 - t2);
            ((1.0 - self.s_target) * forward_progress).mul_add(self.quench_rate, self.s_target)
        } else {
            // Hold phase
            1.0
        }
    }

    /// Calculate transverse field strength A(s)
    #[must_use]
    pub fn transverse_field(&self, s: f64) -> f64 {
        // Standard D-Wave schedule: A(s) = A(0) * (1 - s)
        let a_max = 2.0; // Maximum transverse field strength
        a_max * (1.0 - s)
    }

    /// Calculate problem Hamiltonian strength B(s)
    #[must_use]
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
    /// Base thermal temperature for the Metropolis acceptance criterion.
    ///
    /// The effective temperature governing spin flips is
    /// `base_temperature + A(s)`, where `A(s)` is the schedule's transverse
    /// field. This couples the acceptance to both thermal and quantum
    /// fluctuations rather than to an arbitrary constant.
    pub base_temperature: f64,
}

impl ReverseAnnealingParams {
    /// Create new reverse annealing parameters
    #[must_use]
    pub fn new(initial_state: Vec<i8>) -> Self {
        Self {
            schedule: ReverseAnnealingSchedule::default(),
            initial_state,
            num_sweeps: 1000,
            num_repetitions: 10,
            seed: None,
            reinitialize_fraction: 0.0,
            local_search_radius: None,
            base_temperature: 0.1,
        }
    }

    /// Set targeted reverse annealing with local search
    #[must_use]
    pub const fn with_local_search(mut self, radius: usize) -> Self {
        self.local_search_radius = Some(radius);
        self
    }

    /// Set partial reinitialization
    #[must_use]
    pub const fn with_reinitialization(mut self, fraction: f64) -> Self {
        self.reinitialize_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Set the base thermal temperature used for the Metropolis acceptance.
    ///
    /// During reverse annealing the *effective* fluctuation scale is the sum of
    /// this base thermal temperature and the (Schedule-derived) transverse
    /// field, so that flips are driven by both thermal and quantum
    /// fluctuations. The value must be strictly positive.
    #[must_use]
    pub fn with_base_temperature(mut self, temperature: f64) -> Self {
        if temperature.is_finite() && temperature > 0.0 {
            self.base_temperature = temperature;
        }
        self
    }
}

/// Reverse annealing simulator
pub struct ReverseAnnealingSimulator {
    params: ReverseAnnealingParams,
    rng: ChaCha8Rng,
    /// Optional per-qubit update mask for targeted (local-search) reverse
    /// annealing. When `Some`, only qubits whose entry is `true` are allowed to
    /// flip during annealing; the rest are frozen at their initial value. This
    /// emulates hardware anneal-offsets used for targeted reverse annealing.
    update_mask: Option<Vec<bool>>,
}

impl ReverseAnnealingSimulator {
    /// Create a new reverse annealing simulator
    pub fn new(params: ReverseAnnealingParams) -> AnnealingResult<Self> {
        let rng = match params.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(thread_rng().random()),
        };

        Ok(Self {
            params,
            rng,
            update_mask: None,
        })
    }

    /// Solve an Ising model using reverse annealing
    pub fn solve(&mut self, model: &IsingModel) -> AnnealingResult<AnnealingSolution> {
        let start_time = Instant::now();
        let num_qubits = model.num_qubits;

        // Validate initial state
        if self.params.initial_state.len() != num_qubits {
            return Err(AnnealingError::InvalidParameter(format!(
                "Initial state length {} doesn't match model size {}",
                self.params.initial_state.len(),
                num_qubits
            )));
        }

        let mut best_solution = self.params.initial_state.clone();
        let mut best_energy = model
            .energy(&best_solution)
            .map_err(AnnealingError::IsingError)?;

        let mut all_solutions = Vec::new();
        let mut all_energies = Vec::new();

        for rep in 0..self.params.num_repetitions {
            // Initialize state
            let initial_state = self.params.initial_state.clone();
            let mut state = self.prepare_initial_state(model, &initial_state);

            // Run reverse annealing
            let solution = self.run_reverse_annealing(model, &mut state)?;
            let energy = model
                .energy(&solution)
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
                self.params.num_repetitions, self.params.num_sweeps, self.params.schedule.s_target
            ),
        })
    }

    /// Prepare initial state with optional reinitialization
    fn prepare_initial_state(&mut self, model: &IsingModel, base_state: &[i8]) -> Vec<i8> {
        let mut state = base_state.to_vec();

        // Apply partial reinitialization
        if self.params.reinitialize_fraction > 0.0 {
            let num_to_reinit = (state.len() as f64 * self.params.reinitialize_fraction) as usize;
            for _ in 0..num_to_reinit {
                let idx = self.rng.random_range(0..state.len());
                state[idx] = if self.rng.random_bool(0.5) { 1 } else { -1 };
            }
        }

        // Build and store the local-search update mask if a radius is specified.
        if let Some(radius) = self.params.local_search_radius {
            self.update_mask = Some(self.build_local_search_mask(model, &state, radius));
        } else {
            self.update_mask = None;
        }

        state
    }

    /// Build the local-search update mask for targeted reverse annealing.
    ///
    /// In targeted reverse annealing only spins in the neighbourhood of a set of
    /// "active" centers are allowed to change (hardware realizes this via
    /// anneal-offsets). Rather than choosing centers at random, the centers are
    /// the *most frustrated* spins of the incoming state: those whose local
    /// energy contribution
    /// `e_i = s_i (h_i + Σ_j J_ij s_j)`
    /// is largest (most positive), i.e. the spins that most want to flip given
    /// the current configuration. Starting from each center we mark every spin
    /// reachable within `radius` hops *along the coupling graph* (a breadth-first
    /// expansion), which is the problem-structure analogue of "within `radius`
    /// of the center". All other spins are frozen at their initial value. The
    /// returned mask is consumed by [`run_reverse_annealing`], so the
    /// restriction genuinely affects the Monte-Carlo update selection.
    fn build_local_search_mask(
        &self,
        model: &IsingModel,
        state: &[i8],
        radius: usize,
    ) -> Vec<bool> {
        let num_qubits = model.num_qubits;
        if num_qubits == 0 {
            return Vec::new();
        }

        // Build the adjacency list once from the (sparse) coupling list.
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); num_qubits];
        for coupling in model.couplings() {
            if coupling.i < num_qubits && coupling.j < num_qubits {
                adjacency[coupling.i].push(coupling.j);
                adjacency[coupling.j].push(coupling.i);
            }
        }

        // Per-spin local energy contribution under the current state. A large
        // positive value means the spin is frustrated and flipping it lowers the
        // energy, so its neighbourhood is the most useful to relax.
        let mut frustration: Vec<(usize, f64)> = Vec::with_capacity(num_qubits);
        for i in 0..num_qubits {
            let mut local_field = model.get_bias(i).unwrap_or(0.0);
            for &j in &adjacency[i] {
                if let Ok(coupling) = model.get_coupling(i, j) {
                    local_field += coupling * f64::from(state[j]);
                }
            }
            let energy_contribution = f64::from(state[i]) * local_field;
            frustration.push((i, energy_contribution));
        }

        // Sort by descending frustration; pick the most frustrated spins as the
        // seeds. The seed count scales with the problem size but is at least one.
        frustration.sort_by(|a, b| b.1.total_cmp(&a.1));
        let num_centers = ((num_qubits as f64 * 0.1).ceil() as usize).clamp(1, num_qubits);

        let mut can_update = vec![false; num_qubits];
        for &(center, _) in frustration.iter().take(num_centers) {
            // Breadth-first expansion to `radius` hops along the coupling graph.
            let mut frontier = vec![center];
            can_update[center] = true;
            for _ in 0..radius {
                let mut next_frontier = Vec::new();
                for &node in &frontier {
                    for &neighbor in &adjacency[node] {
                        if !can_update[neighbor] {
                            can_update[neighbor] = true;
                            next_frontier.push(neighbor);
                        }
                    }
                }
                if next_frontier.is_empty() {
                    break;
                }
                frontier = next_frontier;
            }
        }

        can_update
    }

    /// Run the reverse annealing process
    #[must_use]
    fn run_reverse_annealing(
        &mut self,
        model: &IsingModel,
        state: &mut Vec<i8>,
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
                let i = self.rng.random_range(0..model.num_qubits);

                // Respect the targeted local-search mask: frozen qubits never
                // flip during annealing.
                if let Some(mask) = &self.update_mask {
                    if matches!(mask.get(i), Some(false)) {
                        continue;
                    }
                }

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
                            h_local += coupling * f64::from(state[j]) * problem_strength;
                        }
                    }
                }

                // Energy difference for flipping spin i. With s_i -> -s_i the
                // longitudinal contribution changes by -2 s_i (h_i + Σ_j J_ij
                // s_j), so the energy change is +2 s_i * h_local.
                let delta_e = 2.0 * f64::from(state[i]) * h_local;

                // Mean-field reverse-annealing acceptance. The effective
                // fluctuation scale is the base thermal temperature plus the
                // schedule's transverse field A(s): early in the reversal A(s)
                // is large (many flips, broad exploration) and it shrinks back
                // to the thermal floor as the system re-anneals towards s=1.
                // This ties the dynamics to the real schedule rather than to a
                // hand-tuned constant.
                let effective_temp = self.params.base_temperature + transverse_field;
                let accept_prob = if delta_e <= 0.0 {
                    1.0
                } else {
                    (-delta_e / effective_temp).exp()
                };

                if self.rng.random_bool(accept_prob) {
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
    #[must_use]
    pub const fn new() -> Self {
        Self {
            s_target: 0.45,
            pause_duration: 0.1,
            quench_rate: 1.0,
            hold_duration: 0.0,
        }
    }

    /// Set the target s-parameter for reversal
    #[must_use]
    pub const fn s_target(mut self, s: f64) -> Self {
        self.s_target = s;
        self
    }

    /// Set the pause duration
    #[must_use]
    pub const fn pause_duration(mut self, duration: f64) -> Self {
        self.pause_duration = duration;
        self
    }

    /// Set the quench rate
    #[must_use]
    pub const fn quench_rate(mut self, rate: f64) -> Self {
        self.quench_rate = rate;
        self
    }

    /// Set the hold duration
    #[must_use]
    pub const fn hold_duration(mut self, duration: f64) -> Self {
        self.hold_duration = duration;
        self
    }

    /// Build the schedule
    pub fn build(self) -> AnnealingResult<ReverseAnnealingSchedule> {
        if !(0.0..=1.0).contains(&self.s_target) {
            return Err(AnnealingError::InvalidSchedule(
                "s_target must be in [0,1]".to_string(),
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
        let schedule =
            ReverseAnnealingSchedule::new(0.45, 0.1).expect("Schedule creation should succeed");
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
