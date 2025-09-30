//! Quantum Cellular Automata (QCA) simulation for novel quantum algorithms.
//!
//! This module implements quantum cellular automata, which are quantum analogues
//! of classical cellular automata. QCA evolve quantum states according to local
//! unitary rules, enabling exploration of novel quantum algorithms, cryptographic
//! applications, and quantum computation models with inherent locality properties.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;

/// Quantum cellular automaton configuration
#[derive(Debug, Clone)]
pub struct QCAConfig {
    /// Lattice dimensions (1D, 2D, or 3D)
    pub dimensions: Vec<usize>,
    /// Boundary conditions
    pub boundary_conditions: BoundaryConditions,
    /// Local neighborhood type
    pub neighborhood: NeighborhoodType,
    /// Evolution rule type
    pub rule_type: QCARuleType,
    /// Number of evolution steps
    pub evolution_steps: usize,
    /// Enable parallel evolution
    pub parallel_evolution: bool,
    /// Measurement strategy
    pub measurement_strategy: MeasurementStrategy,
}

impl Default for QCAConfig {
    fn default() -> Self {
        Self {
            dimensions: vec![16], // 1D lattice with 16 cells
            boundary_conditions: BoundaryConditions::Periodic,
            neighborhood: NeighborhoodType::Moore,
            rule_type: QCARuleType::Partitioned,
            evolution_steps: 100,
            parallel_evolution: true,
            measurement_strategy: MeasurementStrategy::Probabilistic,
        }
    }
}

/// Boundary conditions for the lattice
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryConditions {
    /// Periodic boundaries (toroidal topology)
    Periodic,
    /// Fixed boundaries (cells on boundary are fixed)
    Fixed,
    /// Open boundaries (no interaction beyond boundary)
    Open,
    /// Reflective boundaries
    Reflective,
}

/// Neighborhood types for local evolution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborhoodType {
    /// Von Neumann neighborhood (nearest neighbors only)
    VonNeumann,
    /// Moore neighborhood (includes diagonal neighbors)
    Moore,
    /// Custom neighborhood pattern
    Custom,
    /// Extended neighborhood (next-nearest neighbors)
    Extended,
}

/// QCA rule types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QCARuleType {
    /// Partitioned QCA (update disjoint partitions simultaneously)
    Partitioned,
    /// Global QCA (single global unitary evolution)
    Global,
    /// Sequential QCA (update cells one by one)
    Sequential,
    /// Margolus neighborhood QCA
    Margolus,
    /// Custom rule implementation
    Custom,
}

/// Measurement strategy for QCA observation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementStrategy {
    /// Probabilistic measurements
    Probabilistic,
    /// Deterministic state observation
    Deterministic,
    /// Partial measurements (subset of cells)
    Partial,
    /// Continuous weak measurements
    Continuous,
}

/// Quantum cellular automaton simulator
pub struct QuantumCellularAutomaton {
    /// Configuration
    config: QCAConfig,
    /// Current quantum state of the lattice
    state: Array1<Complex64>,
    /// Evolution rules for each partition/neighborhood
    evolution_rules: Vec<QCARule>,
    /// Lattice cell mapping
    cell_mapping: CellMapping,
    /// SciRS2 backend for optimization
    backend: Option<SciRS2Backend>,
    /// Evolution history
    evolution_history: Vec<QCASnapshot>,
    /// Statistics
    stats: QCAStats,
}

/// QCA evolution rule
#[derive(Debug, Clone)]
pub struct QCARule {
    /// Rule identifier
    pub id: String,
    /// Unitary matrix for local evolution
    pub unitary: Array2<Complex64>,
    /// Cells affected by this rule
    pub affected_cells: Vec<usize>,
    /// Neighborhood pattern
    pub neighborhood_pattern: Vec<(i32, i32, i32)>, // Relative positions (x, y, z)
    /// Rule parameters
    pub parameters: HashMap<String, f64>,
}

/// Cell mapping for different lattice geometries
#[derive(Debug, Clone)]
pub struct CellMapping {
    /// Total number of cells
    pub total_cells: usize,
    /// Dimension sizes
    pub dimensions: Vec<usize>,
    /// Cell coordinates to linear index mapping
    pub coord_to_index: HashMap<Vec<usize>, usize>,
    /// Linear index to coordinates mapping
    pub index_to_coord: HashMap<usize, Vec<usize>>,
    /// Neighbor mappings for each cell
    pub neighbors: HashMap<usize, Vec<usize>>,
}

/// QCA state snapshot
#[derive(Debug, Clone)]
pub struct QCASnapshot {
    /// Time step
    pub time_step: usize,
    /// Quantum state at this time
    pub state: Array1<Complex64>,
    /// Measurement results (if any)
    pub measurements: Option<Vec<f64>>,
    /// Entropy measures
    pub entanglement_entropy: Option<f64>,
    /// Local observables
    pub local_observables: HashMap<String, f64>,
}

/// QCA simulation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QCAStats {
    /// Total evolution steps performed
    pub evolution_steps: usize,
    /// Total evolution time
    pub total_evolution_time_ms: f64,
    /// Average step time
    pub avg_step_time_ms: f64,
    /// Number of measurements performed
    pub measurements_performed: usize,
    /// Entropy evolution statistics
    pub entropy_stats: EntropyStats,
    /// Rule application counts
    pub rule_applications: HashMap<String, usize>,
}

/// Entropy statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntropyStats {
    /// Maximum entropy observed
    pub max_entropy: f64,
    /// Minimum entropy observed
    pub min_entropy: f64,
    /// Average entropy
    pub avg_entropy: f64,
    /// Entropy variance
    pub entropy_variance: f64,
}

impl QuantumCellularAutomaton {
    /// Create new quantum cellular automaton
    pub fn new(config: QCAConfig) -> Result<Self> {
        let cell_mapping = Self::create_cell_mapping(&config)?;
        let total_cells = cell_mapping.total_cells;

        // Initialize quantum state |0...0⟩
        let state_size = 1 << total_cells;
        let mut state = Array1::zeros(state_size);
        state[0] = Complex64::new(1.0, 0.0);

        let evolution_rules = Self::create_default_rules(&config, &cell_mapping)?;

        Ok(Self {
            config,
            state,
            evolution_rules,
            cell_mapping,
            backend: None,
            evolution_history: Vec::new(),
            stats: QCAStats::default(),
        })
    }

    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }

    /// Set initial state
    pub fn set_initial_state(&mut self, initial_state: Array1<Complex64>) -> Result<()> {
        if initial_state.len() != self.state.len() {
            return Err(SimulatorError::InvalidInput(format!(
                "Initial state size {} doesn't match expected size {}",
                initial_state.len(),
                self.state.len()
            )));
        }

        self.state = initial_state;
        self.evolution_history.clear();
        self.stats = QCAStats::default();

        Ok(())
    }

    /// Add custom evolution rule
    pub fn add_rule(&mut self, rule: QCARule) -> Result<()> {
        // Validate rule unitary
        if !Self::is_unitary(&rule.unitary) {
            return Err(SimulatorError::InvalidInput(
                "Rule matrix is not unitary".to_string(),
            ));
        }

        self.evolution_rules.push(rule);
        Ok(())
    }

    /// Evolve the QCA for specified number of steps
    pub fn evolve(&mut self, steps: usize) -> Result<QCAEvolutionResult> {
        let start_time = std::time::Instant::now();

        for step in 0..steps {
            let step_start = std::time::Instant::now();

            // Apply evolution rules based on rule type
            match self.config.rule_type {
                QCARuleType::Partitioned => self.evolve_partitioned()?,
                QCARuleType::Global => self.evolve_global()?,
                QCARuleType::Sequential => self.evolve_sequential()?,
                QCARuleType::Margolus => self.evolve_margolus()?,
                QCARuleType::Custom => self.evolve_custom()?,
            }

            // Take snapshot if requested
            if step % 10 == 0 || step == steps - 1 {
                let snapshot = self.take_snapshot(step)?;
                self.evolution_history.push(snapshot);
            }

            // Apply measurements if configured
            if let MeasurementStrategy::Continuous = self.config.measurement_strategy {
                self.apply_weak_measurements()?;
            }

            let step_time = step_start.elapsed().as_secs_f64() * 1000.0;
            self.stats.avg_step_time_ms =
                (self.stats.avg_step_time_ms * self.stats.evolution_steps as f64 + step_time)
                    / (self.stats.evolution_steps + 1) as f64;
            self.stats.evolution_steps += 1;
        }

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.total_evolution_time_ms += total_time;

        Ok(QCAEvolutionResult {
            final_state: self.state.clone(),
            evolution_history: self.evolution_history.clone(),
            total_steps: steps,
            total_time_ms: total_time,
            final_entropy: self.calculate_entanglement_entropy()?,
        })
    }

    /// Evolve using partitioned updates
    fn evolve_partitioned(&mut self) -> Result<()> {
        match self.config.dimensions.len() {
            1 => self.evolve_1d_partitioned(),
            2 => self.evolve_2d_partitioned(),
            3 => self.evolve_3d_partitioned(),
            _ => Err(SimulatorError::UnsupportedOperation(
                "QCA supports only 1D, 2D, and 3D lattices".to_string(),
            )),
        }
    }

    /// 1D partitioned evolution
    fn evolve_1d_partitioned(&mut self) -> Result<()> {
        let lattice_size = self.config.dimensions[0];

        // Create partitions for parallel update
        let mut partitions = Vec::new();
        let partition_size = 2; // Update pairs of neighboring cells

        for start in (0..lattice_size).step_by(partition_size) {
            let end = (start + partition_size).min(lattice_size);
            partitions.push((start, end));
        }

        // Apply evolution to each partition
        for (start, end) in partitions {
            for cell in start..end - 1 {
                let neighbor = match self.config.boundary_conditions {
                    BoundaryConditions::Periodic => (cell + 1) % lattice_size,
                    BoundaryConditions::Fixed | BoundaryConditions::Open => {
                        if cell + 1 < lattice_size {
                            cell + 1
                        } else {
                            continue;
                        }
                    }
                    BoundaryConditions::Reflective => {
                        if cell + 1 < lattice_size {
                            cell + 1
                        } else {
                            lattice_size - 1
                        }
                    }
                };

                self.apply_two_cell_rule(cell, neighbor)?;
            }
        }

        Ok(())
    }

    /// 2D partitioned evolution
    fn evolve_2d_partitioned(&mut self) -> Result<()> {
        let (width, height) = (self.config.dimensions[0], self.config.dimensions[1]);

        // Use checkerboard pattern for 2D partitioned updates
        for parity in 0..2 {
            for y in 0..height {
                for x in (parity..width).step_by(2) {
                    let cell = y * width + x;
                    let neighbors = self.get_neighbors_2d(x, y, width, height);

                    if !neighbors.is_empty() {
                        self.apply_neighborhood_rule(cell, &neighbors)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// 3D partitioned evolution
    fn evolve_3d_partitioned(&mut self) -> Result<()> {
        let (width, height, depth) = (
            self.config.dimensions[0],
            self.config.dimensions[1],
            self.config.dimensions[2],
        );

        // Use 3D checkerboard pattern
        for parity in 0..2 {
            for z in 0..depth {
                for y in 0..height {
                    for x in (parity..width).step_by(2) {
                        let cell = z * width * height + y * width + x;
                        let neighbors = self.get_neighbors_3d(x, y, z, width, height, depth);

                        if !neighbors.is_empty() {
                            self.apply_neighborhood_rule(cell, &neighbors)?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Global evolution (single unitary on entire system)
    fn evolve_global(&mut self) -> Result<()> {
        if let Some(global_rule) = self.evolution_rules.first() {
            // Apply global unitary evolution
            let new_state = global_rule.unitary.dot(&self.state);
            self.state = new_state;

            // Update rule application statistics
            let rule_id = global_rule.id.clone();
            *self.stats.rule_applications.entry(rule_id).or_insert(0) += 1;
        }

        Ok(())
    }

    /// Sequential evolution (update one cell at a time)
    fn evolve_sequential(&mut self) -> Result<()> {
        let total_cells = self.cell_mapping.total_cells;
        for cell in 0..total_cells {
            let neighbors = self
                .cell_mapping
                .neighbors
                .get(&cell)
                .cloned()
                .unwrap_or_default();
            if !neighbors.is_empty() {
                self.apply_neighborhood_rule(cell, &neighbors)?;
            }
        }
        Ok(())
    }

    /// Margolus neighborhood evolution
    fn evolve_margolus(&mut self) -> Result<()> {
        // Margolus neighborhood: 2x2 blocks that alternate between offset configurations
        if self.config.dimensions.len() != 2 {
            return Err(SimulatorError::UnsupportedOperation(
                "Margolus QCA only supports 2D lattices".to_string(),
            ));
        }

        let (width, height) = (self.config.dimensions[0], self.config.dimensions[1]);
        let offset = self.stats.evolution_steps % 2; // Alternate between two configurations

        for y in (offset..height - 1).step_by(2) {
            for x in (offset..width - 1).step_by(2) {
                // 2x2 Margolus block
                let cells = vec![
                    y * width + x,
                    y * width + (x + 1),
                    (y + 1) * width + x,
                    (y + 1) * width + (x + 1),
                ];

                self.apply_margolus_rule(&cells)?;
            }
        }

        Ok(())
    }

    /// Custom evolution rule
    fn evolve_custom(&mut self) -> Result<()> {
        // Apply all custom rules in sequence
        for rule in &self.evolution_rules.clone() {
            if rule.affected_cells.len() == 1 {
                self.apply_single_cell_rule(rule.affected_cells[0], &rule.unitary)?;
            } else if rule.affected_cells.len() == 2 {
                self.apply_two_cell_rule(rule.affected_cells[0], rule.affected_cells[1])?;
            } else {
                self.apply_neighborhood_rule(rule.affected_cells[0], &rule.affected_cells[1..])?;
            }

            // Update statistics
            *self
                .stats
                .rule_applications
                .entry(rule.id.clone())
                .or_insert(0) += 1;
        }

        Ok(())
    }

    /// Apply rule to two neighboring cells
    fn apply_two_cell_rule(&mut self, cell1: usize, cell2: usize) -> Result<()> {
        // Find appropriate two-cell rule and clone what we need
        let rule_data = self
            .evolution_rules
            .iter()
            .find(|r| r.unitary.dim() == (4, 4))
            .map(|r| (r.unitary.clone(), r.id.clone()))
            .ok_or_else(|| {
                SimulatorError::UnsupportedOperation("No two-cell rule available".to_string())
            })?;

        // Apply two-qubit unitary to the pair
        self.apply_two_qubit_unitary(cell1, cell2, &rule_data.0)?;

        // Update statistics
        *self.stats.rule_applications.entry(rule_data.1).or_insert(0) += 1;

        Ok(())
    }

    /// Apply rule to a neighborhood of cells
    fn apply_neighborhood_rule(&mut self, center_cell: usize, neighbors: &[usize]) -> Result<()> {
        // For simplicity, apply pairwise interactions between center and each neighbor
        for &neighbor in neighbors {
            self.apply_two_cell_rule(center_cell, neighbor)?;
        }
        Ok(())
    }

    /// Apply Margolus rule to a 2x2 block
    fn apply_margolus_rule(&mut self, cells: &[usize]) -> Result<()> {
        if cells.len() != 4 {
            return Err(SimulatorError::InvalidInput(
                "Margolus rule requires exactly 4 cells".to_string(),
            ));
        }

        // Find appropriate 4-cell rule or create default
        let rule_unitary = self
            .evolution_rules
            .iter()
            .find(|r| r.unitary.dim() == (16, 16))
            .map(|r| r.unitary.clone())
            .unwrap_or_else(Self::create_margolus_rotation_unitary);

        // Apply four-qubit unitary
        self.apply_four_qubit_unitary(cells, &rule_unitary)?;

        Ok(())
    }

    /// Apply single-cell rule
    fn apply_single_cell_rule(&mut self, cell: usize, unitary: &Array2<Complex64>) -> Result<()> {
        self.apply_single_qubit_unitary(cell, unitary)
    }

    /// Apply unitary to single qubit
    fn apply_single_qubit_unitary(
        &mut self,
        qubit: usize,
        unitary: &Array2<Complex64>,
    ) -> Result<()> {
        let state_size = self.state.len();
        let qubit_mask = 1 << qubit;

        let mut new_state = self.state.clone();

        for i in 0..state_size {
            if i & qubit_mask == 0 {
                let j = i | qubit_mask;
                if j < state_size {
                    let amp_0 = self.state[i];
                    let amp_1 = self.state[j];

                    new_state[i] = unitary[[0, 0]] * amp_0 + unitary[[0, 1]] * amp_1;
                    new_state[j] = unitary[[1, 0]] * amp_0 + unitary[[1, 1]] * amp_1;
                }
            }
        }

        self.state = new_state;
        Ok(())
    }

    /// Apply unitary to two qubits
    fn apply_two_qubit_unitary(
        &mut self,
        qubit1: usize,
        qubit2: usize,
        unitary: &Array2<Complex64>,
    ) -> Result<()> {
        let state_size = self.state.len();
        let mask1 = 1 << qubit1;
        let mask2 = 1 << qubit2;

        let mut new_state = self.state.clone();

        for i in 0..state_size {
            let i00 = i & !(mask1 | mask2);
            let i01 = i00 | mask2;
            let i10 = i00 | mask1;
            let i11 = i00 | mask1 | mask2;

            if i == i00 {
                let amp_00 = self.state[i00];
                let amp_01 = self.state[i01];
                let amp_10 = self.state[i10];
                let amp_11 = self.state[i11];

                new_state[i00] = unitary[[0, 0]] * amp_00
                    + unitary[[0, 1]] * amp_01
                    + unitary[[0, 2]] * amp_10
                    + unitary[[0, 3]] * amp_11;
                new_state[i01] = unitary[[1, 0]] * amp_00
                    + unitary[[1, 1]] * amp_01
                    + unitary[[1, 2]] * amp_10
                    + unitary[[1, 3]] * amp_11;
                new_state[i10] = unitary[[2, 0]] * amp_00
                    + unitary[[2, 1]] * amp_01
                    + unitary[[2, 2]] * amp_10
                    + unitary[[2, 3]] * amp_11;
                new_state[i11] = unitary[[3, 0]] * amp_00
                    + unitary[[3, 1]] * amp_01
                    + unitary[[3, 2]] * amp_10
                    + unitary[[3, 3]] * amp_11;
            }
        }

        self.state = new_state;
        Ok(())
    }

    /// Apply unitary to four qubits
    fn apply_four_qubit_unitary(
        &mut self,
        qubits: &[usize],
        unitary: &Array2<Complex64>,
    ) -> Result<()> {
        if qubits.len() != 4 {
            return Err(SimulatorError::InvalidInput(
                "Four qubits required".to_string(),
            ));
        }

        let state_size = self.state.len();
        let mut new_state = self.state.clone();

        // Create masks for the four qubits
        let masks: Vec<usize> = qubits.iter().map(|&q| 1 << q).collect();

        for i in 0..state_size {
            // Extract base index (all qubits in |0⟩ state)
            let base = i & !(masks[0] | masks[1] | masks[2] | masks[3]);

            // Only process if this is the base index for this group
            if i == base {
                // Collect amplitudes for all 16 basis states
                let mut amplitudes = Vec::new();
                for state_idx in 0..16 {
                    let full_idx = base
                        | (if state_idx & 1 != 0 { masks[0] } else { 0 })
                        | (if state_idx & 2 != 0 { masks[1] } else { 0 })
                        | (if state_idx & 4 != 0 { masks[2] } else { 0 })
                        | (if state_idx & 8 != 0 { masks[3] } else { 0 });
                    amplitudes.push(self.state[full_idx]);
                }

                // Apply unitary transformation
                for out_state in 0..16 {
                    let mut new_amplitude = Complex64::new(0.0, 0.0);
                    for (in_state, &amplitude) in amplitudes.iter().enumerate() {
                        new_amplitude += unitary[[out_state, in_state]] * amplitude;
                    }

                    let full_idx = base
                        | (if out_state & 1 != 0 { masks[0] } else { 0 })
                        | (if out_state & 2 != 0 { masks[1] } else { 0 })
                        | (if out_state & 4 != 0 { masks[2] } else { 0 })
                        | (if out_state & 8 != 0 { masks[3] } else { 0 });
                    new_state[full_idx] = new_amplitude;
                }
            }
        }

        self.state = new_state;
        Ok(())
    }

    /// Get 2D neighbors
    fn get_neighbors_2d(&self, x: usize, y: usize, width: usize, height: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();

        let deltas: &[(i32, i32)] = match self.config.neighborhood {
            NeighborhoodType::VonNeumann => &[(-1, 0), (1, 0), (0, -1), (0, 1)],
            NeighborhoodType::Moore => &[
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ],
            _ => &[(-1, 0), (1, 0), (0, -1), (0, 1)], // Default to Von Neumann
        };

        for (dx, dy) in deltas {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;

            let (nx, ny) = match self.config.boundary_conditions {
                BoundaryConditions::Periodic => {
                    let nx = ((nx % width as i32) + width as i32) % width as i32;
                    let ny = ((ny % height as i32) + height as i32) % height as i32;
                    (nx as usize, ny as usize)
                }
                BoundaryConditions::Fixed | BoundaryConditions::Open => {
                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        (nx as usize, ny as usize)
                    } else {
                        continue;
                    }
                }
                BoundaryConditions::Reflective => {
                    let nx = if nx < 0 {
                        0
                    } else if nx >= width as i32 {
                        width - 1
                    } else {
                        nx as usize
                    };
                    let ny = if ny < 0 {
                        0
                    } else if ny >= height as i32 {
                        height - 1
                    } else {
                        ny as usize
                    };
                    (nx, ny)
                }
            };

            neighbors.push(ny * width + nx);
        }

        neighbors
    }

    /// Get 3D neighbors
    fn get_neighbors_3d(
        &self,
        x: usize,
        y: usize,
        z: usize,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Vec<usize> {
        let mut neighbors = Vec::new();

        // Von Neumann neighborhood in 3D (6 neighbors)
        let deltas = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ];

        for (dx, dy, dz) in deltas {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let nz = z as i32 + dz;

            let (nx, ny, nz) = match self.config.boundary_conditions {
                BoundaryConditions::Periodic => {
                    let nx = ((nx % width as i32) + width as i32) % width as i32;
                    let ny = ((ny % height as i32) + height as i32) % height as i32;
                    let nz = ((nz % depth as i32) + depth as i32) % depth as i32;
                    (nx as usize, ny as usize, nz as usize)
                }
                BoundaryConditions::Fixed | BoundaryConditions::Open => {
                    if nx >= 0
                        && nx < width as i32
                        && ny >= 0
                        && ny < height as i32
                        && nz >= 0
                        && nz < depth as i32
                    {
                        (nx as usize, ny as usize, nz as usize)
                    } else {
                        continue;
                    }
                }
                BoundaryConditions::Reflective => {
                    let nx = if nx < 0 {
                        0
                    } else if nx >= width as i32 {
                        width - 1
                    } else {
                        nx as usize
                    };
                    let ny = if ny < 0 {
                        0
                    } else if ny >= height as i32 {
                        height - 1
                    } else {
                        ny as usize
                    };
                    let nz = if nz < 0 {
                        0
                    } else if nz >= depth as i32 {
                        depth - 1
                    } else {
                        nz as usize
                    };
                    (nx, ny, nz)
                }
            };

            neighbors.push(nz * width * height + ny * width + nx);
        }

        neighbors
    }

    /// Apply weak measurements for continuous monitoring
    fn apply_weak_measurements(&mut self) -> Result<()> {
        // Implement weak measurement on a subset of cells
        let measurement_strength = 0.01; // Weak measurement parameter
        let cells_to_measure = (0..self.cell_mapping.total_cells)
            .step_by(4)
            .collect::<Vec<_>>();

        for &cell in &cells_to_measure {
            // Apply weak Z measurement
            let measurement_operator =
                self.create_weak_measurement_operator(cell, measurement_strength);
            self.state = measurement_operator.dot(&self.state);

            // Renormalize
            let norm: f64 = self.state.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            if norm > 1e-15 {
                self.state.mapv_inplace(|x| x / norm);
            }
        }

        self.stats.measurements_performed += cells_to_measure.len();
        Ok(())
    }

    /// Create weak measurement operator
    fn create_weak_measurement_operator(&self, cell: usize, strength: f64) -> Array2<Complex64> {
        let state_size = self.state.len();
        let mut operator = Array2::eye(state_size);
        let cell_mask = 1 << cell;

        for i in 0..state_size {
            if i & cell_mask != 0 {
                // Apply weak dephasing to |1⟩ states
                let factor = Complex64::new((1.0 - strength).sqrt(), 0.0);
                operator[[i, i]] = factor;
            }
        }

        operator
    }

    /// Take a snapshot of the current state
    fn take_snapshot(&mut self, time_step: usize) -> Result<QCASnapshot> {
        let entropy = self.calculate_entanglement_entropy()?;

        // Update entropy statistics
        if self.stats.entropy_stats.max_entropy == 0.0 {
            self.stats.entropy_stats.max_entropy = entropy;
            self.stats.entropy_stats.min_entropy = entropy;
        } else {
            self.stats.entropy_stats.max_entropy =
                self.stats.entropy_stats.max_entropy.max(entropy);
            self.stats.entropy_stats.min_entropy =
                self.stats.entropy_stats.min_entropy.min(entropy);
        }

        let snapshot_count = self.evolution_history.len() as f64;
        self.stats.entropy_stats.avg_entropy =
            (self.stats.entropy_stats.avg_entropy * snapshot_count + entropy)
                / (snapshot_count + 1.0);

        Ok(QCASnapshot {
            time_step,
            state: self.state.clone(),
            measurements: None,
            entanglement_entropy: Some(entropy),
            local_observables: self.calculate_local_observables()?,
        })
    }

    /// Calculate entanglement entropy
    fn calculate_entanglement_entropy(&self) -> Result<f64> {
        // For simplicity, calculate von Neumann entropy of reduced density matrix for first half of system
        let half_size = self.cell_mapping.total_cells / 2;
        if half_size == 0 {
            return Ok(0.0);
        }

        let reduced_dm = self.compute_reduced_density_matrix(half_size)?;
        let eigenvalues = self.compute_eigenvalues(&reduced_dm)?;

        let entropy = eigenvalues
            .iter()
            .filter(|&&lambda| lambda > 1e-15)
            .map(|&lambda| -lambda * lambda.ln())
            .sum();

        Ok(entropy)
    }

    /// Compute reduced density matrix for first n qubits
    fn compute_reduced_density_matrix(&self, n_qubits: usize) -> Result<Array2<f64>> {
        let reduced_size = 1 << n_qubits;
        let mut reduced_dm = Array2::zeros((reduced_size, reduced_size));

        let total_qubits = self.cell_mapping.total_cells;
        let env_size = 1 << (total_qubits - n_qubits);

        for i in 0..reduced_size {
            for j in 0..reduced_size {
                let mut element = 0.0;

                for k in 0..env_size {
                    let full_i = i | (k << n_qubits);
                    let full_j = j | (k << n_qubits);

                    if full_i < self.state.len() && full_j < self.state.len() {
                        element += (self.state[full_i] * self.state[full_j].conj()).re;
                    }
                }

                reduced_dm[[i, j]] = element;
            }
        }

        Ok(reduced_dm)
    }

    /// Compute eigenvalues of a real symmetric matrix
    fn compute_eigenvalues(&self, matrix: &Array2<f64>) -> Result<Vec<f64>> {
        // Simplified eigenvalue computation - in practice would use LAPACK
        let mut eigenvalues = Vec::new();

        // For small matrices, use power iteration or analytical solutions
        if matrix.dim().0 <= 4 {
            // Analytical or simple numerical methods
            for i in 0..matrix.dim().0 {
                eigenvalues.push(matrix[[i, i]]); // Approximation using diagonal elements
            }
        } else {
            // For larger matrices, would need proper eigenvalue solver
            for i in 0..matrix.dim().0 {
                eigenvalues.push(matrix[[i, i]]);
            }
        }

        Ok(eigenvalues)
    }

    /// Calculate local observables
    fn calculate_local_observables(&self) -> Result<HashMap<String, f64>> {
        let mut observables = HashMap::new();

        // Calculate local magnetization (average Z measurement) for each cell
        for cell in 0..self.cell_mapping.total_cells {
            let magnetization = self.calculate_local_magnetization(cell)?;
            observables.insert(format!("magnetization_{}", cell), magnetization);
        }

        // Calculate correlation functions for nearest neighbors
        for cell in 0..self.cell_mapping.total_cells {
            if let Some(neighbors) = self.cell_mapping.neighbors.get(&cell) {
                for &neighbor in neighbors {
                    if neighbor > cell {
                        // Avoid double counting
                        let correlation = self.calculate_correlation(cell, neighbor)?;
                        observables
                            .insert(format!("correlation_{}_{}", cell, neighbor), correlation);
                    }
                }
            }
        }

        Ok(observables)
    }

    /// Calculate local magnetization (Z expectation value)
    fn calculate_local_magnetization(&self, cell: usize) -> Result<f64> {
        let cell_mask = 1 << cell;
        let mut expectation = 0.0;

        for (i, &amplitude) in self.state.iter().enumerate() {
            let prob = amplitude.norm_sqr();
            let z_value = if i & cell_mask != 0 { -1.0 } else { 1.0 };
            expectation += prob * z_value;
        }

        Ok(expectation)
    }

    /// Calculate correlation between two cells
    fn calculate_correlation(&self, cell1: usize, cell2: usize) -> Result<f64> {
        let mask1 = 1 << cell1;
        let mask2 = 1 << cell2;
        let mut correlation = 0.0;

        for (i, &amplitude) in self.state.iter().enumerate() {
            let prob = amplitude.norm_sqr();
            let z1 = if i & mask1 != 0 { -1.0 } else { 1.0 };
            let z2 = if i & mask2 != 0 { -1.0 } else { 1.0 };
            correlation += prob * z1 * z2;
        }

        Ok(correlation)
    }

    /// Measure specific cells
    pub fn measure_cells(&mut self, cells: &[usize]) -> Result<Vec<bool>> {
        let mut results = Vec::new();

        for &cell in cells {
            let prob_one = self.calculate_measurement_probability(cell)?;
            let result = fastrand::f64() < prob_one;
            results.push(result);

            // Apply measurement collapse
            self.collapse_state_after_measurement(cell, result)?;
        }

        self.stats.measurements_performed += cells.len();
        Ok(results)
    }

    /// Calculate measurement probability for a cell
    fn calculate_measurement_probability(&self, cell: usize) -> Result<f64> {
        let cell_mask = 1 << cell;
        let mut prob_one = 0.0;

        for (i, &amplitude) in self.state.iter().enumerate() {
            if i & cell_mask != 0 {
                prob_one += amplitude.norm_sqr();
            }
        }

        Ok(prob_one)
    }

    /// Collapse state after measurement
    fn collapse_state_after_measurement(&mut self, cell: usize, result: bool) -> Result<()> {
        let cell_mask = 1 << cell;
        let mut norm = 0.0;

        // Zero out incompatible amplitudes and calculate new norm
        for (i, amplitude) in self.state.iter_mut().enumerate() {
            let cell_value = (i & cell_mask) != 0;
            if cell_value != result {
                *amplitude = Complex64::new(0.0, 0.0);
            } else {
                norm += amplitude.norm_sqr();
            }
        }

        // Renormalize
        if norm > 1e-15 {
            let norm_factor = 1.0 / norm.sqrt();
            self.state.mapv_inplace(|x| x * norm_factor);
        }

        Ok(())
    }

    /// Get current state
    pub fn get_state(&self) -> &Array1<Complex64> {
        &self.state
    }

    /// Get evolution history
    pub fn get_evolution_history(&self) -> &[QCASnapshot] {
        &self.evolution_history
    }

    /// Get statistics
    pub fn get_stats(&self) -> &QCAStats {
        &self.stats
    }

    /// Reset the QCA to initial state
    pub fn reset(&mut self) -> Result<()> {
        let state_size = self.state.len();
        self.state = Array1::zeros(state_size);
        self.state[0] = Complex64::new(1.0, 0.0);
        self.evolution_history.clear();
        self.stats = QCAStats::default();
        Ok(())
    }

    /// Helper methods
    fn create_cell_mapping(config: &QCAConfig) -> Result<CellMapping> {
        let total_cells = config.dimensions.iter().product();
        let mut coord_to_index = HashMap::new();
        let mut index_to_coord = HashMap::new();
        let mut neighbors = HashMap::new();

        match config.dimensions.len() {
            1 => {
                let size = config.dimensions[0];
                for i in 0..size {
                    coord_to_index.insert(vec![i], i);
                    index_to_coord.insert(i, vec![i]);

                    let mut cell_neighbors = Vec::new();
                    match config.boundary_conditions {
                        BoundaryConditions::Periodic => {
                            cell_neighbors.push((i + size - 1) % size);
                            cell_neighbors.push((i + 1) % size);
                        }
                        _ => {
                            if i > 0 {
                                cell_neighbors.push(i - 1);
                            }
                            if i < size - 1 {
                                cell_neighbors.push(i + 1);
                            }
                        }
                    }
                    neighbors.insert(i, cell_neighbors);
                }
            }
            2 => {
                let (width, height) = (config.dimensions[0], config.dimensions[1]);
                for y in 0..height {
                    for x in 0..width {
                        let index = y * width + x;
                        coord_to_index.insert(vec![x, y], index);
                        index_to_coord.insert(index, vec![x, y]);
                        // Neighbors will be computed dynamically
                        neighbors.insert(index, Vec::new());
                    }
                }
            }
            3 => {
                let (width, height, depth) = (
                    config.dimensions[0],
                    config.dimensions[1],
                    config.dimensions[2],
                );
                for z in 0..depth {
                    for y in 0..height {
                        for x in 0..width {
                            let index = z * width * height + y * width + x;
                            coord_to_index.insert(vec![x, y, z], index);
                            index_to_coord.insert(index, vec![x, y, z]);
                            neighbors.insert(index, Vec::new());
                        }
                    }
                }
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(
                    "QCA supports only 1D, 2D, and 3D lattices".to_string(),
                ))
            }
        }

        Ok(CellMapping {
            total_cells,
            dimensions: config.dimensions.clone(),
            coord_to_index,
            index_to_coord,
            neighbors,
        })
    }

    fn create_default_rules(
        config: &QCAConfig,
        cell_mapping: &CellMapping,
    ) -> Result<Vec<QCARule>> {
        let mut rules = Vec::new();

        match config.rule_type {
            QCARuleType::Global => {
                // Create random unitary for global evolution
                let state_size = 1 << cell_mapping.total_cells.min(10); // Limit for practicality
                let unitary = Self::create_random_unitary(state_size);
                rules.push(QCARule {
                    id: "global_evolution".to_string(),
                    unitary,
                    affected_cells: (0..cell_mapping.total_cells).collect(),
                    neighborhood_pattern: Vec::new(),
                    parameters: HashMap::new(),
                });
            }
            _ => {
                // Create local rules
                rules.push(QCARule {
                    id: "two_cell_interaction".to_string(),
                    unitary: Self::create_cnot_unitary(),
                    affected_cells: Vec::new(),
                    neighborhood_pattern: vec![(0, 0, 0), (1, 0, 0)],
                    parameters: HashMap::new(),
                });

                rules.push(QCARule {
                    id: "single_cell_rotation".to_string(),
                    unitary: Self::create_rotation_unitary(std::f64::consts::PI / 4.0),
                    affected_cells: Vec::new(),
                    neighborhood_pattern: vec![(0, 0, 0)],
                    parameters: HashMap::new(),
                });
            }
        }

        Ok(rules)
    }

    fn create_cnot_unitary() -> Array2<Complex64> {
        Array2::from_shape_vec(
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
        .unwrap()
    }

    fn create_rotation_unitary(angle: f64) -> Array2<Complex64> {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(cos_half, 0.0),
                Complex64::new(0.0, -sin_half),
                Complex64::new(0.0, -sin_half),
                Complex64::new(cos_half, 0.0),
            ],
        )
        .unwrap()
    }

    fn create_margolus_rotation_unitary() -> Array2<Complex64> {
        // 4-cell Margolus rotation unitary (rotates the 2x2 block)
        let mut unitary = Array2::zeros((16, 16));

        // Identity for most states, with specific rotations for the 2x2 block
        for i in 0..16 {
            // Apply rotation pattern: |ab⟩|cd⟩ → |da⟩|bc⟩
            let a = (i >> 3) & 1;
            let b = (i >> 2) & 1;
            let c = (i >> 1) & 1;
            let d = i & 1;

            let rotated = (d << 3) | (a << 2) | (b << 1) | c;
            unitary[[rotated, i]] = Complex64::new(1.0, 0.0);
        }

        unitary
    }

    fn create_random_unitary(size: usize) -> Array2<Complex64> {
        // Create a random unitary matrix using Gram-Schmidt process
        let mut matrix = Array2::zeros((size, size));

        // Start with random complex matrix
        for i in 0..size {
            for j in 0..size {
                matrix[[i, j]] = Complex64::new(fastrand::f64() - 0.5, fastrand::f64() - 0.5);
            }
        }

        // Apply Gram-Schmidt orthogonalization
        for j in 0..size {
            // Normalize column j
            let mut norm_sq = 0.0;
            for i in 0..size {
                norm_sq += matrix[[i, j]].norm_sqr();
            }
            let norm = norm_sq.sqrt();

            if norm > 1e-15 {
                for i in 0..size {
                    matrix[[i, j]] /= norm;
                }
            }

            // Orthogonalize subsequent columns
            for k in j + 1..size {
                let mut inner_product = Complex64::new(0.0, 0.0);
                for i in 0..size {
                    inner_product += matrix[[i, j]].conj() * matrix[[i, k]];
                }

                for i in 0..size {
                    let correction = inner_product * matrix[[i, j]];
                    matrix[[i, k]] -= correction;
                }
            }
        }

        matrix
    }

    fn is_unitary(matrix: &Array2<Complex64>) -> bool {
        let (rows, cols) = matrix.dim();
        if rows != cols {
            return false;
        }

        // Check if U†U = I (approximately)
        let mut identity_check = Array2::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..rows {
                    sum += matrix[[k, i]].conj() * matrix[[k, j]];
                }
                identity_check[[i, j]] = sum;
            }
        }

        // Check if close to identity
        for i in 0..rows {
            for j in 0..cols {
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (identity_check[[i, j]] - expected).norm();
                if diff > 1e-10 {
                    return false;
                }
            }
        }

        true
    }
}

/// QCA evolution result
#[derive(Debug, Clone)]
pub struct QCAEvolutionResult {
    /// Final quantum state
    pub final_state: Array1<Complex64>,
    /// Evolution history snapshots
    pub evolution_history: Vec<QCASnapshot>,
    /// Total evolution steps
    pub total_steps: usize,
    /// Total evolution time
    pub total_time_ms: f64,
    /// Final entanglement entropy
    pub final_entropy: f64,
}

/// QCA utilities
pub struct QCAUtils;

impl QCAUtils {
    /// Create a predefined QCA configuration
    pub fn create_predefined_config(config_type: &str, size: usize) -> QCAConfig {
        match config_type {
            "game_of_life" => QCAConfig {
                dimensions: vec![size, size],
                boundary_conditions: BoundaryConditions::Periodic,
                neighborhood: NeighborhoodType::Moore,
                rule_type: QCARuleType::Partitioned,
                evolution_steps: 50,
                parallel_evolution: true,
                measurement_strategy: MeasurementStrategy::Probabilistic,
            },
            "elementary_ca" => QCAConfig {
                dimensions: vec![size],
                boundary_conditions: BoundaryConditions::Periodic,
                neighborhood: NeighborhoodType::VonNeumann,
                rule_type: QCARuleType::Sequential,
                evolution_steps: 100,
                parallel_evolution: false,
                measurement_strategy: MeasurementStrategy::Deterministic,
            },
            "margolus_ca" => QCAConfig {
                dimensions: vec![size, size],
                boundary_conditions: BoundaryConditions::Periodic,
                neighborhood: NeighborhoodType::Custom,
                rule_type: QCARuleType::Margolus,
                evolution_steps: 25,
                parallel_evolution: true,
                measurement_strategy: MeasurementStrategy::Partial,
            },
            _ => QCAConfig::default(),
        }
    }

    /// Create initial pattern for QCA
    pub fn create_initial_pattern(pattern_type: &str, dimensions: &[usize]) -> Array1<Complex64> {
        let total_cells = dimensions.iter().product::<usize>();
        let state_size = 1 << total_cells;
        let mut state = Array1::zeros(state_size);

        match pattern_type {
            "random" => {
                // Random superposition
                for i in 0..state_size {
                    state[i] = Complex64::new(fastrand::f64() - 0.5, fastrand::f64() - 0.5);
                }
                // Normalize
                let norm: f64 = state.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                state.mapv_inplace(|x| x / norm);
            }
            "glider" if dimensions.len() == 2 => {
                // Conway's Game of Life glider pattern
                let width = dimensions[0];
                let height = dimensions[1];

                if width >= 3 && height >= 3 {
                    // Glider pattern: positions (1,0), (2,1), (0,2), (1,2), (2,2)
                    let glider_positions = [
                        width,
                        2 * width + 1,
                        2, // 0 * width + 2, simplified to avoid clippy warning
                        width + 2,
                        2 * width + 2,
                    ];

                    let mut glider_state = 0;
                    for &pos in &glider_positions {
                        if pos < total_cells {
                            glider_state |= 1 << pos;
                        }
                    }

                    if glider_state < state_size {
                        state[glider_state] = Complex64::new(1.0, 0.0);
                    }
                }
            }
            "uniform" => {
                // Uniform superposition
                let amplitude = 1.0 / (state_size as f64).sqrt();
                state.fill(Complex64::new(amplitude, 0.0));
            }
            _ => {
                // Default: |0...0⟩ state
                state[0] = Complex64::new(1.0, 0.0);
            }
        }

        state
    }

    /// Benchmark QCA performance
    pub fn benchmark_qca() -> Result<QCABenchmarkResults> {
        let mut results = QCABenchmarkResults::default();

        let configs = vec![
            (
                "1d_elementary",
                QCAUtils::create_predefined_config("elementary_ca", 8),
            ),
            (
                "2d_game_of_life",
                QCAUtils::create_predefined_config("game_of_life", 4),
            ),
            (
                "2d_margolus",
                QCAUtils::create_predefined_config("margolus_ca", 4),
            ),
        ];

        for (name, mut config) in configs {
            config.evolution_steps = 20; // Limit for benchmarking

            let mut qca = QuantumCellularAutomaton::new(config)?;
            let initial_state = QCAUtils::create_initial_pattern("random", &qca.config.dimensions);
            qca.set_initial_state(initial_state)?;

            let start = std::time::Instant::now();
            let _result = qca.evolve(20)?;
            let time = start.elapsed().as_secs_f64() * 1000.0;

            results.benchmark_times.push((name.to_string(), time));
            results
                .qca_stats
                .insert(name.to_string(), qca.get_stats().clone());
        }

        Ok(results)
    }
}

/// QCA benchmark results
#[derive(Debug, Clone, Default)]
pub struct QCABenchmarkResults {
    /// Benchmark times by configuration
    pub benchmark_times: Vec<(String, f64)>,
    /// QCA statistics by configuration
    pub qca_stats: HashMap<String, QCAStats>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_qca_creation() {
        let config = QCAConfig::default();
        let qca = QuantumCellularAutomaton::new(config);
        assert!(qca.is_ok());
    }

    #[test]
    fn test_qca_1d_evolution() {
        let config = QCAConfig {
            dimensions: vec![4],
            evolution_steps: 5,
            ..Default::default()
        };

        let mut qca = QuantumCellularAutomaton::new(config).unwrap();
        let result = qca.evolve(5);
        assert!(result.is_ok());

        let evolution_result = result.unwrap();
        assert_eq!(evolution_result.total_steps, 5);
        assert!(evolution_result.total_time_ms > 0.0);
    }

    #[test]
    fn test_qca_2d_evolution() {
        let config = QCAConfig {
            dimensions: vec![3, 3],
            evolution_steps: 3,
            rule_type: QCARuleType::Partitioned,
            ..Default::default()
        };

        let mut qca = QuantumCellularAutomaton::new(config).unwrap();
        let result = qca.evolve(3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_qca_measurement() {
        let config = QCAConfig {
            dimensions: vec![3],
            ..Default::default()
        };

        let mut qca = QuantumCellularAutomaton::new(config).unwrap();

        // Measure all cells
        let results = qca.measure_cells(&[0, 1, 2]);
        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 3);
    }

    #[test]
    fn test_boundary_conditions() {
        let configs = vec![
            BoundaryConditions::Periodic,
            BoundaryConditions::Fixed,
            BoundaryConditions::Open,
            BoundaryConditions::Reflective,
        ];

        for boundary in configs {
            let config = QCAConfig {
                dimensions: vec![4],
                boundary_conditions: boundary,
                evolution_steps: 2,
                ..Default::default()
            };

            let mut qca = QuantumCellularAutomaton::new(config).unwrap();
            let result = qca.evolve(2);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_neighborhood_types() {
        let neighborhoods = vec![NeighborhoodType::VonNeumann, NeighborhoodType::Moore];

        for neighborhood in neighborhoods {
            let config = QCAConfig {
                dimensions: vec![3, 3],
                neighborhood,
                evolution_steps: 2,
                ..Default::default()
            };

            let mut qca = QuantumCellularAutomaton::new(config).unwrap();
            let result = qca.evolve(2);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_predefined_configs() {
        let config_types = vec!["game_of_life", "elementary_ca", "margolus_ca"];

        for config_type in config_types {
            let config = QCAUtils::create_predefined_config(config_type, 4);
            let qca = QuantumCellularAutomaton::new(config);
            assert!(qca.is_ok());
        }
    }

    #[test]
    fn test_initial_patterns() {
        let dimensions = vec![4, 4];
        let patterns = vec!["random", "glider", "uniform"];

        for pattern in patterns {
            let state = QCAUtils::create_initial_pattern(pattern, &dimensions);
            assert_eq!(state.len(), 1 << 16); // 2^16 for 16 cells

            // Check normalization
            let norm: f64 = state.iter().map(|x| x.norm_sqr()).sum();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_entanglement_entropy_calculation() {
        let config = QCAConfig {
            dimensions: vec![4],
            ..Default::default()
        };

        let mut qca = QuantumCellularAutomaton::new(config).unwrap();

        // Create entangled state
        let state_size = qca.state.len();
        qca.state = Array1::zeros(state_size);
        qca.state[0] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0); // |0000⟩
        qca.state[15] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0); // |1111⟩

        let entropy = qca.calculate_entanglement_entropy().unwrap();
        assert!(entropy >= 0.0);
    }

    #[test]
    fn test_local_observables() {
        let config = QCAConfig {
            dimensions: vec![3],
            ..Default::default()
        };

        let qca = QuantumCellularAutomaton::new(config).unwrap();
        let observables = qca.calculate_local_observables().unwrap();

        // Should have magnetization for each cell
        assert!(observables.contains_key("magnetization_0"));
        assert!(observables.contains_key("magnetization_1"));
        assert!(observables.contains_key("magnetization_2"));
    }

    #[test]
    fn test_unitary_check() {
        let cnot = QuantumCellularAutomaton::create_cnot_unitary();
        assert!(QuantumCellularAutomaton::is_unitary(&cnot));

        let rotation =
            QuantumCellularAutomaton::create_rotation_unitary(std::f64::consts::PI / 4.0);
        assert!(QuantumCellularAutomaton::is_unitary(&rotation));
    }
}
