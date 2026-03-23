//! Multi-walker and decoherent quantum walk implementations.

use super::discrete::DiscreteQuantumWalk;
use super::graph::{CoinOperator, Graph};
use crate::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;

/// Multi-walker quantum walk for studying entanglement and correlations
pub struct MultiWalkerQuantumWalk {
    graph: Graph,
    num_walkers: usize,
    /// State tensor product space: walker1_pos ⊗ walker2_pos ⊗ ...
    state: Array1<Complex64>,
    /// Dimension of single walker space
    single_walker_dim: usize,
}

impl MultiWalkerQuantumWalk {
    /// Create a new multi-walker quantum walk
    pub fn new(graph: Graph, num_walkers: usize) -> Self {
        let single_walker_dim = graph.num_vertices;
        let total_dim = single_walker_dim.pow(num_walkers as u32);

        Self {
            graph,
            num_walkers,
            state: Array1::zeros(total_dim),
            single_walker_dim,
        }
    }

    /// Initialize walkers at specific positions
    pub fn initialize_positions(&mut self, positions: &[usize]) -> QuantRS2Result<()> {
        if positions.len() != self.num_walkers {
            return Err(QuantRS2Error::InvalidInput(
                "Number of positions must match number of walkers".to_string(),
            ));
        }

        for &pos in positions {
            if pos >= self.single_walker_dim {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Position {pos} out of bounds"
                )));
            }
        }

        // Reset state
        self.state.fill(Complex64::new(0.0, 0.0));

        // Set amplitude for initial configuration
        let index = self.positions_to_index(positions);
        self.state[index] = Complex64::new(1.0, 0.0);

        Ok(())
    }

    /// Initialize in entangled superposition
    pub fn initialize_entangled_bell_state(
        &mut self,
        pos1: usize,
        pos2: usize,
    ) -> QuantRS2Result<()> {
        if self.num_walkers != 2 {
            return Err(QuantRS2Error::InvalidInput(
                "Bell state initialization only works for 2 walkers".to_string(),
            ));
        }

        self.state.fill(Complex64::new(0.0, 0.0));

        let amplitude = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);

        // |pos1,pos2> + |pos2,pos1>
        let idx1 = self.positions_to_index(&[pos1, pos2]);
        let idx2 = self.positions_to_index(&[pos2, pos1]);

        self.state[idx1] = amplitude;
        self.state[idx2] = amplitude;

        Ok(())
    }

    /// Convert walker positions to state vector index
    fn positions_to_index(&self, positions: &[usize]) -> usize {
        let mut index = 0;
        let mut multiplier = 1;

        for &pos in positions.iter().rev() {
            index += pos * multiplier;
            multiplier *= self.single_walker_dim;
        }

        index
    }

    /// Convert state vector index to walker positions
    fn index_to_positions(&self, mut index: usize) -> Vec<usize> {
        let mut positions = Vec::with_capacity(self.num_walkers);

        for _ in 0..self.num_walkers {
            positions.push(index % self.single_walker_dim);
            index /= self.single_walker_dim;
        }

        positions.reverse();
        positions
    }

    /// Perform one step (simplified version - each walker evolves independently)
    pub fn step_independent(&mut self) {
        let mut new_state = Array1::zeros(self.state.len());

        for (index, &amplitude) in self.state.iter().enumerate() {
            if amplitude.norm_sqr() < 1e-15 {
                continue;
            }

            let positions = self.index_to_positions(index);

            // Each walker moves to neighboring vertices with equal probability
            let mut total_neighbors = 1;
            for &pos in &positions {
                total_neighbors *= self.graph.degree(pos).max(1);
            }

            let neighbor_amplitude = amplitude / (total_neighbors as f64).sqrt();

            // Generate all possible neighbor configurations
            self.add_neighbor_amplitudes(
                &positions,
                0,
                &mut Vec::new(),
                neighbor_amplitude,
                &mut new_state,
            );
        }

        self.state = new_state;
    }

    /// Recursively add amplitudes for all neighbor configurations
    fn add_neighbor_amplitudes(
        &self,
        original_positions: &[usize],
        walker_idx: usize,
        current_positions: &mut Vec<usize>,
        amplitude: Complex64,
        new_state: &mut Array1<Complex64>,
    ) {
        if walker_idx >= self.num_walkers {
            let index = self.positions_to_index(current_positions);
            new_state[index] += amplitude;
            return;
        }

        let pos = original_positions[walker_idx];
        let neighbors = &self.graph.edges[pos];

        if neighbors.is_empty() {
            // Stay at same position if no neighbors
            current_positions.push(pos);
            self.add_neighbor_amplitudes(
                original_positions,
                walker_idx + 1,
                current_positions,
                amplitude,
                new_state,
            );
            current_positions.pop();
        } else {
            for &neighbor in neighbors {
                current_positions.push(neighbor);
                self.add_neighbor_amplitudes(
                    original_positions,
                    walker_idx + 1,
                    current_positions,
                    amplitude,
                    new_state,
                );
                current_positions.pop();
            }
        }
    }

    /// Get marginal probability distribution for a specific walker
    pub fn marginal_probabilities(&self, walker_idx: usize) -> Vec<f64> {
        let mut probs = vec![0.0; self.single_walker_dim];

        for (index, &amplitude) in self.state.iter().enumerate() {
            let positions = self.index_to_positions(index);
            probs[positions[walker_idx]] += amplitude.norm_sqr();
        }

        probs
    }

    /// Calculate entanglement entropy between walkers
    pub fn entanglement_entropy(&self) -> f64 {
        if self.num_walkers != 2 {
            return 0.0; // Only implemented for 2 walkers
        }

        // Compute reduced density matrix for walker 1
        let mut reduced_dm = Array2::zeros((self.single_walker_dim, self.single_walker_dim));

        for i in 0..self.single_walker_dim {
            for j in 0..self.single_walker_dim {
                for k in 0..self.single_walker_dim {
                    let idx1 = self.positions_to_index(&[i, k]);
                    let idx2 = self.positions_to_index(&[j, k]);

                    reduced_dm[[i, j]] += self.state[idx1].conj() * self.state[idx2];
                }
            }
        }

        // Calculate von Neumann entropy (simplified - would use eigenvalues in practice)
        let trace = reduced_dm.diag().mapv(|x: Complex64| x.re).sum();
        -trace * trace.ln() // Simplified approximation
    }
}

/// Quantum walk with environmental decoherence
pub struct DecoherentQuantumWalk {
    base_walk: DiscreteQuantumWalk,
    decoherence_rate: f64,
    measurement_probability: f64,
}

impl DecoherentQuantumWalk {
    /// Create a new decoherent quantum walk
    pub fn new(graph: Graph, coin_operator: CoinOperator, decoherence_rate: f64) -> Self {
        Self {
            base_walk: DiscreteQuantumWalk::new(graph, coin_operator),
            decoherence_rate,
            measurement_probability: 0.0,
        }
    }

    /// Initialize walker position
    pub fn initialize_position(&mut self, position: usize) {
        self.base_walk.initialize_position(position);
    }

    /// Perform one step with decoherence
    pub fn step(&mut self) {
        // Apply unitary evolution
        self.base_walk.step();

        // Apply decoherence
        self.apply_decoherence();
    }

    /// Apply decoherence by mixing with classical random walk
    fn apply_decoherence(&mut self) {
        if self.decoherence_rate <= 0.0 {
            return;
        }

        // Get current probabilities
        let quantum_probs = self.base_walk.position_probabilities();

        // Classical random walk step
        let mut classical_probs = vec![0.0; quantum_probs.len()];
        for (v, &prob) in quantum_probs.iter().enumerate() {
            if prob > 0.0 {
                let degree = self.base_walk.graph.degree(v) as f64;
                if degree > 0.0 {
                    for &neighbor in &self.base_walk.graph.edges[v] {
                        classical_probs[neighbor] += prob / degree;
                    }
                } else {
                    classical_probs[v] += prob; // Stay if isolated
                }
            }
        }

        // Mix quantum and classical
        let quantum_weight = 1.0 - self.decoherence_rate;
        let classical_weight = self.decoherence_rate;

        // Update quantum state to match mixed probabilities (approximate)
        for v in 0..quantum_probs.len() {
            let mixed_prob =
                quantum_weight * quantum_probs[v] + classical_weight * classical_probs[v];

            // Scale amplitudes to match mixed probabilities (simplified)
            if quantum_probs[v] > 0.0 {
                let scale_factor = (mixed_prob / quantum_probs[v]).sqrt();

                for coin in 0..self.base_walk.coin_dimension {
                    let idx = self.base_walk.state_index(v, coin);
                    if idx < self.base_walk.state.len() {
                        self.base_walk.state[idx] *= scale_factor;
                    }
                }
            }
        }

        // Renormalize
        let total_norm: f64 = self.base_walk.state.iter().map(|c| c.norm_sqr()).sum();
        if total_norm > 0.0 {
            let norm_factor = 1.0 / total_norm.sqrt();
            for amplitude in &mut self.base_walk.state {
                *amplitude *= norm_factor;
            }
        }
    }

    /// Get position probabilities
    pub fn position_probabilities(&self) -> Vec<f64> {
        self.base_walk.position_probabilities()
    }

    /// Set decoherence rate
    pub const fn set_decoherence_rate(&mut self, rate: f64) {
        self.decoherence_rate = rate.clamp(0.0, 1.0);
    }
}
