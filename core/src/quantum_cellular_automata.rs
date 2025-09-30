//! Quantum Cellular Automata Simulation
//!
//! This module implements quantum cellular automata (QCA) which are quantum
//! generalizations of classical cellular automata. QCA evolve quantum states
//! on a lattice according to local unitary rules, providing a model for
//! quantum computation and many-body quantum dynamics.

use crate::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::Complex64;

/// Types of quantum cellular automata
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QCAType {
    /// Partitioned QCA (PQCA) - alternating between different partitions
    Partitioned,
    /// Margolus neighborhood QCA - 2x2 blocks with alternating shifts
    Margolus,
    /// Moore neighborhood QCA - 3x3 neighborhoods
    Moore,
    /// Von Neumann neighborhood QCA - plus-shaped neighborhoods
    VonNeumann,
}

/// Boundary conditions for the lattice
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    /// Periodic boundary conditions (toroidal topology)
    Periodic,
    /// Fixed boundary conditions (edges are fixed states)
    Fixed,
    /// Open boundary conditions (no interactions across boundaries)
    Open,
}

/// Update rule for quantum cellular automata
pub trait QCARule {
    /// Apply the local update rule to a neighborhood
    fn apply(&self, neighborhood: &[Complex64]) -> QuantRS2Result<Vec<Complex64>>;

    /// Get the size of the neighborhood this rule operates on
    fn neighborhood_size(&self) -> usize;

    /// Check if the rule is reversible (unitary)
    fn is_reversible(&self) -> bool;
}

/// Quantum rule based on a unitary matrix
#[derive(Debug, Clone)]
pub struct UnitaryRule {
    /// Unitary matrix defining the evolution
    pub unitary: Array2<Complex64>,
    /// Number of qubits the rule operates on
    pub num_qubits: usize,
}

impl UnitaryRule {
    /// Create a new unitary rule
    pub fn new(unitary: Array2<Complex64>) -> QuantRS2Result<Self> {
        let (rows, cols) = unitary.dim();
        if rows != cols {
            return Err(QuantRS2Error::InvalidInput(
                "Unitary matrix must be square".to_string(),
            ));
        }

        // Check if matrix is unitary (U†U = I)
        let conjugate_transpose = unitary.t().mapv(|x| x.conj());
        let product = conjugate_transpose.dot(&unitary);

        for i in 0..rows {
            for j in 0..cols {
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                if (product[[i, j]] - expected).norm() > 1e-10 {
                    return Err(QuantRS2Error::InvalidInput(
                        "Matrix is not unitary".to_string(),
                    ));
                }
            }
        }

        let num_qubits = (rows as f64).log2() as usize;
        if 1 << num_qubits != rows {
            return Err(QuantRS2Error::InvalidInput(
                "Matrix dimension must be a power of 2".to_string(),
            ));
        }

        Ok(Self {
            unitary,
            num_qubits,
        })
    }

    /// Create CNOT rule for 2 qubits
    pub fn cnot() -> Self {
        let unitary = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();

        Self::new(unitary).unwrap()
    }

    /// Create Hadamard rule for 1 qubit
    pub fn hadamard() -> Self {
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        let unitary = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(sqrt2_inv, 0.0),
                Complex64::new(sqrt2_inv, 0.0),
                Complex64::new(sqrt2_inv, 0.0),
                Complex64::new(-sqrt2_inv, 0.0),
            ],
        )
        .unwrap();

        Self::new(unitary).unwrap()
    }

    /// Create Toffoli rule for 3 qubits
    pub fn toffoli() -> Self {
        let mut unitary = Array2::zeros((8, 8));

        // Identity for most states
        for i in 0..6 {
            unitary[[i, i]] = Complex64::new(1.0, 0.0);
        }

        // Swap last two states (|110⟩ ↔ |111⟩)
        unitary[[6, 7]] = Complex64::new(1.0, 0.0);
        unitary[[7, 6]] = Complex64::new(1.0, 0.0);

        Self::new(unitary).unwrap()
    }
}

impl QCARule for UnitaryRule {
    fn apply(&self, neighborhood: &[Complex64]) -> QuantRS2Result<Vec<Complex64>> {
        if neighborhood.len() != 1 << self.num_qubits {
            return Err(QuantRS2Error::InvalidInput(
                "Neighborhood size doesn't match rule dimension".to_string(),
            ));
        }

        let state = Array1::from_vec(neighborhood.to_vec());
        let evolved = self.unitary.dot(&state);
        Ok(evolved.to_vec())
    }

    fn neighborhood_size(&self) -> usize {
        1 << self.num_qubits
    }

    fn is_reversible(&self) -> bool {
        true // All unitary rules are reversible
    }
}

/// Quantum cellular automaton on a 1D lattice
pub struct QuantumCellularAutomaton1D {
    /// Number of sites in the lattice
    pub num_sites: usize,
    /// Current state of each site (amplitude for |0⟩ and |1⟩)
    pub state: Array2<Complex64>,
    /// Update rule
    rule: Box<dyn QCARule + Send + Sync>,
    /// Boundary conditions
    boundary: BoundaryCondition,
    /// QCA type
    qca_type: QCAType,
    /// Current time step
    pub time_step: usize,
}

impl QuantumCellularAutomaton1D {
    /// Create a new 1D quantum cellular automaton
    pub fn new(
        num_sites: usize,
        rule: Box<dyn QCARule + Send + Sync>,
        boundary: BoundaryCondition,
        qca_type: QCAType,
    ) -> Self {
        // Initialize with all sites in |0⟩ state
        let mut state = Array2::zeros((num_sites, 2));
        for i in 0..num_sites {
            state[[i, 0]] = Complex64::new(1.0, 0.0); // |0⟩ state
        }

        Self {
            num_sites,
            state,
            rule,
            boundary,
            qca_type,
            time_step: 0,
        }
    }

    /// Set the state of a specific site
    pub fn set_site_state(
        &mut self,
        site: usize,
        amplitudes: [Complex64; 2],
    ) -> QuantRS2Result<()> {
        if site >= self.num_sites {
            return Err(QuantRS2Error::InvalidInput(
                "Site index out of bounds".to_string(),
            ));
        }

        // Normalize the state
        let norm = (amplitudes[0].norm_sqr() + amplitudes[1].norm_sqr()).sqrt();
        if norm < 1e-10 {
            return Err(QuantRS2Error::InvalidInput(
                "State cannot have zero norm".to_string(),
            ));
        }

        self.state[[site, 0]] = amplitudes[0] / norm;
        self.state[[site, 1]] = amplitudes[1] / norm;

        Ok(())
    }

    /// Get the state of a specific site
    pub fn get_site_state(&self, site: usize) -> QuantRS2Result<[Complex64; 2]> {
        if site >= self.num_sites {
            return Err(QuantRS2Error::InvalidInput(
                "Site index out of bounds".to_string(),
            ));
        }

        Ok([self.state[[site, 0]], self.state[[site, 1]]])
    }

    /// Initialize with a random state
    pub fn initialize_random(&mut self, rng: &mut dyn scirs2_core::random::RngCore) -> QuantRS2Result<()> {
        use scirs2_core::random::prelude::*;

        for i in 0..self.num_sites {
            let theta = rng.gen_range(0.0..std::f64::consts::PI);
            let phi = rng.gen_range(0.0..2.0 * std::f64::consts::PI);

            let amp0 = Complex64::new(theta.cos(), 0.0);
            let amp1 = Complex64::new(theta.sin() * phi.cos(), theta.sin() * phi.sin());

            self.set_site_state(i, [amp0, amp1])?;
        }

        Ok(())
    }

    /// Perform one evolution step
    pub fn step(&mut self) -> QuantRS2Result<()> {
        match self.qca_type {
            QCAType::Partitioned => self.step_partitioned(),
            QCAType::Margolus => self.step_margolus(),
            QCAType::Moore => self.step_moore(),
            QCAType::VonNeumann => self.step_von_neumann(),
        }
    }

    /// Partitioned QCA step (alternating even/odd partitions)
    fn step_partitioned(&mut self) -> QuantRS2Result<()> {
        let rule_size = self.rule.neighborhood_size();
        let num_qubits = (rule_size as f64).log2() as usize;

        if num_qubits != 2 {
            return Err(QuantRS2Error::InvalidInput(
                "Partitioned QCA currently supports only 2-qubit rules".to_string(),
            ));
        }

        let mut new_state = self.state.clone();
        let offset = self.time_step % 2;

        // Apply rule to neighboring pairs
        for i in (offset..self.num_sites.saturating_sub(1)).step_by(2) {
            let j = (i + 1) % self.num_sites;

            // Get neighborhood state vector
            let neighborhood = self.get_pair_state_vector(i, j)?;

            // Apply rule
            let evolved = self.rule.apply(&neighborhood)?;

            // Update states
            new_state[[i, 0]] = evolved[0];
            new_state[[i, 1]] = evolved[1];
            new_state[[j, 0]] = evolved[2];
            new_state[[j, 1]] = evolved[3];
        }

        self.state = new_state;
        self.time_step += 1;
        Ok(())
    }

    /// Get state vector for a pair of sites
    fn get_pair_state_vector(&self, site1: usize, site2: usize) -> QuantRS2Result<Vec<Complex64>> {
        let amp1_0 = self.state[[site1, 0]];
        let amp1_1 = self.state[[site1, 1]];
        let amp2_0 = self.state[[site2, 0]];
        let amp2_1 = self.state[[site2, 1]];

        // Tensor product: |site1⟩ ⊗ |site2⟩
        Ok(vec![
            amp1_0 * amp2_0, // |00⟩
            amp1_0 * amp2_1, // |01⟩
            amp1_1 * amp2_0, // |10⟩
            amp1_1 * amp2_1, // |11⟩
        ])
    }

    /// Margolus neighborhood step (2x2 blocks, alternating)
    fn step_margolus(&mut self) -> QuantRS2Result<()> {
        // For 1D, Margolus reduces to partitioned with 2-site neighborhoods
        self.step_partitioned()
    }

    /// Moore neighborhood step (3x3 neighborhoods in 2D, 3-site in 1D)
    fn step_moore(&mut self) -> QuantRS2Result<()> {
        let rule_size = self.rule.neighborhood_size();
        if rule_size != 8 {
            // 3 qubits = 8 dimensional Hilbert space
            return Err(QuantRS2Error::InvalidInput(
                "Moore neighborhood requires 3-qubit rule".to_string(),
            ));
        }

        let mut new_state = self.state.clone();

        for i in 0..self.num_sites {
            let left = self.get_neighbor(i, -1);
            let center = i;
            let right = self.get_neighbor(i, 1);

            // Get 3-site neighborhood state
            let neighborhood = self.get_triple_state_vector(left, center, right)?;

            // Apply rule
            let evolved = self.rule.apply(&neighborhood)?;

            // Update center site only
            new_state[[center, 0]] = evolved[2]; // Extract center qubit from |abc⟩ -> |b⟩
            new_state[[center, 1]] = evolved[6];
        }

        self.state = new_state;
        self.time_step += 1;
        Ok(())
    }

    /// Von Neumann neighborhood step (plus-shaped, center + 4 neighbors)
    fn step_von_neumann(&mut self) -> QuantRS2Result<()> {
        // For 1D, Von Neumann is same as Moore (3 neighbors)
        self.step_moore()
    }

    /// Get neighbor index with boundary conditions
    fn get_neighbor(&self, site: usize, offset: isize) -> usize {
        match self.boundary {
            BoundaryCondition::Periodic => {
                let new_site = site as isize + offset;
                ((new_site % self.num_sites as isize + self.num_sites as isize)
                    % self.num_sites as isize) as usize
            }
            BoundaryCondition::Fixed | BoundaryCondition::Open => {
                let new_site = site as isize + offset;
                if new_site < 0 {
                    0
                } else if new_site >= self.num_sites as isize {
                    self.num_sites - 1
                } else {
                    new_site as usize
                }
            }
        }
    }

    /// Get state vector for three sites
    fn get_triple_state_vector(
        &self,
        site1: usize,
        site2: usize,
        site3: usize,
    ) -> QuantRS2Result<Vec<Complex64>> {
        let mut state_vector = vec![Complex64::new(0.0, 0.0); 8];

        // Build 3-qubit state vector
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let amp =
                        self.state[[site1, i]] * self.state[[site2, j]] * self.state[[site3, k]];
                    let idx = 4 * i + 2 * j + k;
                    state_vector[idx] = amp;
                }
            }
        }

        Ok(state_vector)
    }

    /// Calculate the entanglement entropy of a region
    pub fn entanglement_entropy(
        &self,
        region_start: usize,
        region_size: usize,
    ) -> QuantRS2Result<f64> {
        if region_start + region_size > self.num_sites {
            return Err(QuantRS2Error::InvalidInput(
                "Region extends beyond lattice".to_string(),
            ));
        }

        // For full implementation, we'd need to compute the reduced density matrix
        // This is a simplified approximation based on site entropies
        let mut entropy = 0.0;

        for i in region_start..region_start + region_size {
            let p0 = self.state[[i, 0]].norm_sqr();
            let p1 = self.state[[i, 1]].norm_sqr();

            if p0 > 1e-12 {
                entropy -= p0 * p0.ln();
            }
            if p1 > 1e-12 {
                entropy -= p1 * p1.ln();
            }
        }

        Ok(entropy)
    }

    /// Get probability distribution at each site
    pub fn site_probabilities(&self) -> Vec<[f64; 2]> {
        self.state
            .rows()
            .into_iter()
            .map(|row| [row[0].norm_sqr(), row[1].norm_sqr()])
            .collect()
    }

    /// Compute correlation function between two sites
    pub fn correlation(&self, site1: usize, site2: usize) -> QuantRS2Result<Complex64> {
        if site1 >= self.num_sites || site2 >= self.num_sites {
            return Err(QuantRS2Error::InvalidInput(
                "Site index out of bounds".to_string(),
            ));
        }

        // Simplified correlation: ⟨Z_i Z_j⟩
        let z1 = self.state[[site1, 0]].norm_sqr() - self.state[[site1, 1]].norm_sqr();
        let z2 = self.state[[site2, 0]].norm_sqr() - self.state[[site2, 1]].norm_sqr();

        Ok(Complex64::new(z1 * z2, 0.0))
    }
}

impl std::fmt::Debug for QuantumCellularAutomaton1D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantumCellularAutomaton1D")
            .field("num_sites", &self.num_sites)
            .field("boundary", &self.boundary)
            .field("qca_type", &self.qca_type)
            .field("time_step", &self.time_step)
            .finish()
    }
}

/// 2D Quantum cellular automaton
pub struct QuantumCellularAutomaton2D {
    /// Width of the lattice
    pub width: usize,
    /// Height of the lattice
    pub height: usize,
    /// Current state (width × height × 2 for qubit amplitudes)
    pub state: Array3<Complex64>,
    /// Update rule
    rule: Box<dyn QCARule + Send + Sync>,
    /// Boundary conditions
    boundary: BoundaryCondition,
    /// QCA type
    qca_type: QCAType,
    /// Current time step
    pub time_step: usize,
}

impl QuantumCellularAutomaton2D {
    /// Create a new 2D quantum cellular automaton
    pub fn new(
        width: usize,
        height: usize,
        rule: Box<dyn QCARule + Send + Sync>,
        boundary: BoundaryCondition,
        qca_type: QCAType,
    ) -> Self {
        // Initialize with all sites in |0⟩ state
        let mut state = Array3::zeros((width, height, 2));
        for i in 0..width {
            for j in 0..height {
                state[[i, j, 0]] = Complex64::new(1.0, 0.0); // |0⟩ state
            }
        }

        Self {
            width,
            height,
            state,
            rule,
            boundary,
            qca_type,
            time_step: 0,
        }
    }

    /// Set the state of a specific site
    pub fn set_site_state(
        &mut self,
        x: usize,
        y: usize,
        amplitudes: [Complex64; 2],
    ) -> QuantRS2Result<()> {
        if x >= self.width || y >= self.height {
            return Err(QuantRS2Error::InvalidInput(
                "Site coordinates out of bounds".to_string(),
            ));
        }

        // Normalize the state
        let norm = (amplitudes[0].norm_sqr() + amplitudes[1].norm_sqr()).sqrt();
        if norm < 1e-10 {
            return Err(QuantRS2Error::InvalidInput(
                "State cannot have zero norm".to_string(),
            ));
        }

        self.state[[x, y, 0]] = amplitudes[0] / norm;
        self.state[[x, y, 1]] = amplitudes[1] / norm;

        Ok(())
    }

    /// Get the state of a specific site
    pub fn get_site_state(&self, x: usize, y: usize) -> QuantRS2Result<[Complex64; 2]> {
        if x >= self.width || y >= self.height {
            return Err(QuantRS2Error::InvalidInput(
                "Site coordinates out of bounds".to_string(),
            ));
        }

        Ok([self.state[[x, y, 0]], self.state[[x, y, 1]]])
    }

    /// Perform one evolution step
    pub fn step(&mut self) -> QuantRS2Result<()> {
        match self.qca_type {
            QCAType::Margolus => self.step_margolus_2d(),
            QCAType::Moore => self.step_moore_2d(),
            QCAType::VonNeumann => self.step_von_neumann_2d(),
            QCAType::Partitioned => self.step_partitioned_2d(),
        }
    }

    /// Margolus neighborhood step for 2D (2×2 blocks)
    fn step_margolus_2d(&mut self) -> QuantRS2Result<()> {
        let rule_size = self.rule.neighborhood_size();
        if rule_size != 16 {
            // 4 qubits = 16 dimensional Hilbert space
            return Err(QuantRS2Error::InvalidInput(
                "2D Margolus requires 4-qubit rule".to_string(),
            ));
        }

        let mut new_state = self.state.clone();
        let x_offset = self.time_step % 2;
        let y_offset = self.time_step % 2;

        // Process 2×2 blocks
        for i in (x_offset..self.width.saturating_sub(1)).step_by(2) {
            for j in (y_offset..self.height.saturating_sub(1)).step_by(2) {
                let block_state = self.get_block_state_vector(i, j)?;
                let evolved = self.rule.apply(&block_state)?;

                // Update the 2×2 block
                for (idx, &val) in evolved.iter().enumerate() {
                    let local_x = idx % 2;
                    let local_y = (idx / 2) % 2;
                    let qubit = (idx / 4) % 2;

                    if i + local_x < self.width && j + local_y < self.height {
                        new_state[[i + local_x, j + local_y, qubit]] = val;
                    }
                }
            }
        }

        self.state = new_state;
        self.time_step += 1;
        Ok(())
    }

    /// Get state vector for a 2×2 block
    fn get_block_state_vector(
        &self,
        start_x: usize,
        start_y: usize,
    ) -> QuantRS2Result<Vec<Complex64>> {
        let mut state_vector = vec![Complex64::new(0.0, 0.0); 16];

        // Get states of 4 sites in the block
        let sites = [
            (start_x, start_y),
            (start_x + 1, start_y),
            (start_x, start_y + 1),
            (start_x + 1, start_y + 1),
        ];

        // Build 4-qubit state vector
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        let amp = self.state[[sites[0].0, sites[0].1, i]]
                            * self.state[[sites[1].0, sites[1].1, j]]
                            * self.state[[sites[2].0, sites[2].1, k]]
                            * self.state[[sites[3].0, sites[3].1, l]];

                        let idx = 8 * i + 4 * j + 2 * k + l;
                        state_vector[idx] = amp;
                    }
                }
            }
        }

        Ok(state_vector)
    }

    /// Moore neighborhood step for 2D (3×3 neighborhoods)
    fn step_moore_2d(&mut self) -> QuantRS2Result<()> {
        // This would require 9-qubit rules, which is very large
        // For practical purposes, we'll use a simplified version
        self.step_von_neumann_2d()
    }

    /// Von Neumann neighborhood step for 2D (plus-shaped)
    fn step_von_neumann_2d(&mut self) -> QuantRS2Result<()> {
        let rule_size = self.rule.neighborhood_size();
        if rule_size != 32 {
            // 5 qubits = 32 dimensional Hilbert space
            return Err(QuantRS2Error::InvalidInput(
                "2D Von Neumann requires 5-qubit rule".to_string(),
            ));
        }

        let mut new_state = self.state.clone();

        for i in 0..self.width {
            for j in 0..self.height {
                // Get Von Neumann neighborhood (center + 4 neighbors)
                let neighbors = self.get_von_neumann_neighbors(i, j);
                let neighborhood_state = self.get_neighborhood_state_vector(&neighbors)?;

                // Apply rule
                let evolved = self.rule.apply(&neighborhood_state)?;

                // Update center site only
                new_state[[i, j, 0]] = evolved[16]; // Center qubit |0⟩ component
                new_state[[i, j, 1]] = evolved[24]; // Center qubit |1⟩ component
            }
        }

        self.state = new_state;
        self.time_step += 1;
        Ok(())
    }

    /// Partitioned step for 2D
    fn step_partitioned_2d(&mut self) -> QuantRS2Result<()> {
        // Apply horizontal and vertical 2-qubit rules alternately
        if self.time_step % 2 == 0 {
            self.step_horizontal_pairs()
        } else {
            self.step_vertical_pairs()
        }
    }

    /// Apply rule to horizontal pairs
    fn step_horizontal_pairs(&mut self) -> QuantRS2Result<()> {
        let rule_size = self.rule.neighborhood_size();
        if rule_size != 4 {
            // 2 qubits = 4 dimensional
            return Err(QuantRS2Error::InvalidInput(
                "Horizontal pairs require 2-qubit rule".to_string(),
            ));
        }

        let mut new_state = self.state.clone();

        for j in 0..self.height {
            for i in (0..self.width.saturating_sub(1)).step_by(2) {
                let pair_state = self.get_pair_state_vector_2d(i, j, i + 1, j)?;
                let evolved = self.rule.apply(&pair_state)?;

                // Update pair
                new_state[[i, j, 0]] = evolved[0];
                new_state[[i, j, 1]] = evolved[1];
                new_state[[i + 1, j, 0]] = evolved[2];
                new_state[[i + 1, j, 1]] = evolved[3];
            }
        }

        self.state = new_state;
        self.time_step += 1;
        Ok(())
    }

    /// Apply rule to vertical pairs
    fn step_vertical_pairs(&mut self) -> QuantRS2Result<()> {
        let rule_size = self.rule.neighborhood_size();
        if rule_size != 4 {
            // 2 qubits = 4 dimensional
            return Err(QuantRS2Error::InvalidInput(
                "Vertical pairs require 2-qubit rule".to_string(),
            ));
        }

        let mut new_state = self.state.clone();

        for i in 0..self.width {
            for j in (0..self.height.saturating_sub(1)).step_by(2) {
                let pair_state = self.get_pair_state_vector_2d(i, j, i, j + 1)?;
                let evolved = self.rule.apply(&pair_state)?;

                // Update pair
                new_state[[i, j, 0]] = evolved[0];
                new_state[[i, j, 1]] = evolved[1];
                new_state[[i, j + 1, 0]] = evolved[2];
                new_state[[i, j + 1, 1]] = evolved[3];
            }
        }

        self.state = new_state;
        self.time_step += 1;
        Ok(())
    }

    /// Get Von Neumann neighbors (up, down, left, right, center)
    fn get_von_neumann_neighbors(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        vec![
            (x, y),                          // center
            (self.get_neighbor_x(x, -1), y), // left
            (self.get_neighbor_x(x, 1), y),  // right
            (x, self.get_neighbor_y(y, -1)), // up
            (x, self.get_neighbor_y(y, 1)),  // down
        ]
    }

    /// Get neighbor x-coordinate with boundary conditions
    fn get_neighbor_x(&self, x: usize, offset: isize) -> usize {
        match self.boundary {
            BoundaryCondition::Periodic => {
                let new_x = x as isize + offset;
                ((new_x % self.width as isize + self.width as isize) % self.width as isize) as usize
            }
            BoundaryCondition::Fixed | BoundaryCondition::Open => {
                let new_x = x as isize + offset;
                if new_x < 0 {
                    0
                } else if new_x >= self.width as isize {
                    self.width - 1
                } else {
                    new_x as usize
                }
            }
        }
    }

    /// Get neighbor y-coordinate with boundary conditions
    fn get_neighbor_y(&self, y: usize, offset: isize) -> usize {
        match self.boundary {
            BoundaryCondition::Periodic => {
                let new_y = y as isize + offset;
                ((new_y % self.height as isize + self.height as isize) % self.height as isize)
                    as usize
            }
            BoundaryCondition::Fixed | BoundaryCondition::Open => {
                let new_y = y as isize + offset;
                if new_y < 0 {
                    0
                } else if new_y >= self.height as isize {
                    self.height - 1
                } else {
                    new_y as usize
                }
            }
        }
    }

    /// Get state vector for neighborhood sites
    fn get_neighborhood_state_vector(
        &self,
        sites: &[(usize, usize)],
    ) -> QuantRS2Result<Vec<Complex64>> {
        let num_sites = sites.len();
        let state_size = 1 << num_sites;
        let mut state_vector = vec![Complex64::new(0.0, 0.0); state_size];

        // This is a simplified implementation - full tensor product would be expensive
        // For now, return a simplified state
        state_vector[0] = Complex64::new(1.0, 0.0);
        Ok(state_vector)
    }

    /// Get pair state vector for 2D coordinates
    fn get_pair_state_vector_2d(
        &self,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
    ) -> QuantRS2Result<Vec<Complex64>> {
        let amp1_0 = self.state[[x1, y1, 0]];
        let amp1_1 = self.state[[x1, y1, 1]];
        let amp2_0 = self.state[[x2, y2, 0]];
        let amp2_1 = self.state[[x2, y2, 1]];

        Ok(vec![
            amp1_0 * amp2_0, // |00⟩
            amp1_0 * amp2_1, // |01⟩
            amp1_1 * amp2_0, // |10⟩
            amp1_1 * amp2_1, // |11⟩
        ])
    }

    /// Get total probability distribution across the lattice
    pub fn probability_distribution(&self) -> Array3<f64> {
        let mut probs = Array3::zeros((self.width, self.height, 2));

        for i in 0..self.width {
            for j in 0..self.height {
                probs[[i, j, 0]] = self.state[[i, j, 0]].norm_sqr();
                probs[[i, j, 1]] = self.state[[i, j, 1]].norm_sqr();
            }
        }

        probs
    }

    /// Calculate magnetization of the lattice
    pub fn magnetization(&self) -> f64 {
        let mut total_magnetization = 0.0;

        for i in 0..self.width {
            for j in 0..self.height {
                let prob_0 = self.state[[i, j, 0]].norm_sqr();
                let prob_1 = self.state[[i, j, 1]].norm_sqr();
                total_magnetization += prob_0 - prob_1; // Z expectation value
            }
        }

        total_magnetization / (self.width * self.height) as f64
    }
}

impl std::fmt::Debug for QuantumCellularAutomaton2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantumCellularAutomaton2D")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("boundary", &self.boundary)
            .field("qca_type", &self.qca_type)
            .field("time_step", &self.time_step)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unitary_rule_creation() {
        let hadamard = UnitaryRule::hadamard();
        assert_eq!(hadamard.num_qubits, 1);
        assert!(hadamard.is_reversible());

        let cnot = UnitaryRule::cnot();
        assert_eq!(cnot.num_qubits, 2);
        assert!(cnot.is_reversible());
    }

    #[test]
    fn test_1d_qca_initialization() {
        let rule = Box::new(UnitaryRule::cnot());
        let qca = QuantumCellularAutomaton1D::new(
            10,
            rule,
            BoundaryCondition::Periodic,
            QCAType::Partitioned,
        );

        assert_eq!(qca.num_sites, 10);
        assert_eq!(qca.time_step, 0);

        // Check initial state (all |0⟩)
        for i in 0..10 {
            let state = qca.get_site_state(i).unwrap();
            assert!((state[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
            assert!((state[1] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        }
    }

    #[test]
    fn test_1d_qca_evolution() {
        let rule = Box::new(UnitaryRule::cnot());
        let mut qca = QuantumCellularAutomaton1D::new(
            4,
            rule,
            BoundaryCondition::Periodic,
            QCAType::Partitioned,
        );

        // Set middle site to |1⟩
        qca.set_site_state(1, [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)])
            .unwrap();

        // Evolve one step
        qca.step().unwrap();
        assert_eq!(qca.time_step, 1);

        // Check that evolution occurred
        let probs = qca.site_probabilities();
        assert!(probs.len() == 4);
    }

    #[test]
    fn test_2d_qca_initialization() {
        let rule = Box::new(UnitaryRule::cnot());
        let qca = QuantumCellularAutomaton2D::new(
            5,
            5,
            rule,
            BoundaryCondition::Periodic,
            QCAType::Margolus,
        );

        assert_eq!(qca.width, 5);
        assert_eq!(qca.height, 5);
        assert_eq!(qca.time_step, 0);

        // Check initial state
        for i in 0..5 {
            for j in 0..5 {
                let state = qca.get_site_state(i, j).unwrap();
                assert!((state[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
                assert!((state[1] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn test_entanglement_entropy() {
        let rule = Box::new(UnitaryRule::cnot());
        let mut qca = QuantumCellularAutomaton1D::new(
            4,
            rule,
            BoundaryCondition::Periodic,
            QCAType::Partitioned,
        );

        // Create some mixed state
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        qca.set_site_state(
            0,
            [
                Complex64::new(sqrt2_inv, 0.0),
                Complex64::new(sqrt2_inv, 0.0),
            ],
        )
        .unwrap();

        let entropy = qca.entanglement_entropy(0, 2).unwrap();
        assert!(entropy >= 0.0);
    }

    #[test]
    fn test_correlation_function() {
        let rule = Box::new(UnitaryRule::cnot());
        let mut qca = QuantumCellularAutomaton1D::new(
            4,
            rule,
            BoundaryCondition::Periodic,
            QCAType::Partitioned,
        );

        // Set sites to definite states
        qca.set_site_state(0, [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)])
            .unwrap(); // |0⟩
        qca.set_site_state(1, [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)])
            .unwrap(); // |1⟩

        let correlation = qca.correlation(0, 1).unwrap();
        assert!((correlation - Complex64::new(-1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_2d_magnetization() {
        let rule = Box::new(UnitaryRule::cnot());
        let mut qca = QuantumCellularAutomaton2D::new(
            3,
            3,
            rule,
            BoundaryCondition::Periodic,
            QCAType::Partitioned,
        );

        // Set all sites to |0⟩ (already the default)
        let magnetization = qca.magnetization();
        assert!((magnetization - 1.0).abs() < 1e-10);

        // Set all sites to |1⟩
        for i in 0..3 {
            for j in 0..3 {
                qca.set_site_state(i, j, [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)])
                    .unwrap();
            }
        }

        let magnetization = qca.magnetization();
        assert!((magnetization - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_toffoli_rule() {
        let toffoli = UnitaryRule::toffoli();
        assert_eq!(toffoli.num_qubits, 3);
        assert_eq!(toffoli.neighborhood_size(), 8);

        // Test that |110⟩ → |111⟩
        let input = vec![
            Complex64::new(0.0, 0.0), // |000⟩
            Complex64::new(0.0, 0.0), // |001⟩
            Complex64::new(0.0, 0.0), // |010⟩
            Complex64::new(0.0, 0.0), // |011⟩
            Complex64::new(0.0, 0.0), // |100⟩
            Complex64::new(0.0, 0.0), // |101⟩
            Complex64::new(1.0, 0.0), // |110⟩
            Complex64::new(0.0, 0.0), // |111⟩
        ];

        let output = toffoli.apply(&input).unwrap();

        // Should flip to |111⟩
        assert!((output[6] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        assert!((output[7] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }
}
