//! Measurement-based quantum computing (MBQC)
//!
//! This module provides implementations for one-way quantum computing using
//! cluster states, graph states, and measurement patterns.

use crate::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

/// Measurement basis for MBQC
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeasurementBasis {
    /// Computational basis (Z)
    Computational,
    /// X basis
    X,
    /// Y basis
    Y,
    /// XY-plane measurement at angle θ
    XY(f64),
    /// XZ-plane measurement at angle θ
    XZ(f64),
    /// YZ-plane measurement at angle θ
    YZ(f64),
}

impl MeasurementBasis {
    /// Get the measurement operator for this basis
    pub fn operator(&self) -> Array2<Complex64> {
        match self {
            MeasurementBasis::Computational => {
                // |0⟩⟨0|
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                    ],
                )
                .unwrap()
            }
            MeasurementBasis::X => {
                // |+⟩⟨+|
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(0.5, 0.0),
                        Complex64::new(0.5, 0.0),
                        Complex64::new(0.5, 0.0),
                        Complex64::new(0.5, 0.0),
                    ],
                )
                .unwrap()
            }
            MeasurementBasis::Y => {
                // |i⟩⟨i| where |i⟩ = (|0⟩ + i|1⟩)/√2
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(0.5, 0.0),
                        Complex64::new(0.0, -0.5),
                        Complex64::new(0.0, 0.5),
                        Complex64::new(0.5, 0.0),
                    ],
                )
                .unwrap()
            }
            MeasurementBasis::XY(theta) => {
                // |θ⟩⟨θ| where |θ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
                let c = (theta / 2.0).cos();
                let s = (theta / 2.0).sin();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(c * c, 0.0),
                        Complex64::new(c * s, 0.0),
                        Complex64::new(c * s, 0.0),
                        Complex64::new(s * s, 0.0),
                    ],
                )
                .unwrap()
            }
            MeasurementBasis::XZ(theta) => {
                // Rotation in XZ plane
                let c = (theta / 2.0).cos();
                let s = (theta / 2.0).sin();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(c * c, 0.0),
                        Complex64::new(c, 0.0) * Complex64::new(0.0, -s),
                        Complex64::new(c, 0.0) * Complex64::new(0.0, s),
                        Complex64::new(s * s, 0.0),
                    ],
                )
                .unwrap()
            }
            MeasurementBasis::YZ(theta) => {
                // Rotation in YZ plane
                let c = (theta / 2.0).cos();
                let s = (theta / 2.0).sin();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(c * c, 0.0),
                        Complex64::new(s, 0.0) * Complex64::new(1.0, 0.0),
                        Complex64::new(s, 0.0) * Complex64::new(1.0, 0.0),
                        Complex64::new(s * s, 0.0),
                    ],
                )
                .unwrap()
            }
        }
    }
}

/// Graph structure for graph states
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of vertices
    pub num_vertices: usize,
    /// Adjacency list representation
    pub edges: HashMap<usize, HashSet<usize>>,
}

impl Graph {
    /// Create a new empty graph
    pub fn new(num_vertices: usize) -> Self {
        let mut edges = HashMap::new();
        for i in 0..num_vertices {
            edges.insert(i, HashSet::new());
        }

        Self {
            num_vertices,
            edges,
        }
    }

    /// Add an edge between two vertices
    pub fn add_edge(&mut self, u: usize, v: usize) -> QuantRS2Result<()> {
        if u >= self.num_vertices || v >= self.num_vertices {
            return Err(QuantRS2Error::InvalidInput(
                "Vertex index out of bounds".to_string(),
            ));
        }

        if u != v {
            self.edges.get_mut(&u).unwrap().insert(v);
            self.edges.get_mut(&v).unwrap().insert(u);
        }

        Ok(())
    }

    /// Get neighbors of a vertex
    pub fn neighbors(&self, v: usize) -> Option<&HashSet<usize>> {
        self.edges.get(&v)
    }

    /// Create a linear cluster (1D chain)
    pub fn linear_cluster(n: usize) -> Self {
        let mut graph = Self::new(n);
        for i in 0..n - 1 {
            graph.add_edge(i, i + 1).unwrap();
        }
        graph
    }

    /// Create a 2D rectangular cluster
    pub fn rectangular_cluster(rows: usize, cols: usize) -> Self {
        let n = rows * cols;
        let mut graph = Self::new(n);

        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;

                // Horizontal edges
                if c < cols - 1 {
                    graph.add_edge(idx, idx + 1).unwrap();
                }

                // Vertical edges
                if r < rows - 1 {
                    graph.add_edge(idx, idx + cols).unwrap();
                }
            }
        }

        graph
    }

    /// Create a complete graph
    pub fn complete(n: usize) -> Self {
        let mut graph = Self::new(n);
        for i in 0..n {
            for j in i + 1..n {
                graph.add_edge(i, j).unwrap();
            }
        }
        graph
    }

    /// Create a star graph (one central node connected to all others)
    pub fn star(n: usize) -> Self {
        let mut graph = Self::new(n);
        for i in 1..n {
            graph.add_edge(0, i).unwrap();
        }
        graph
    }
}

/// Measurement pattern for MBQC
#[derive(Debug, Clone)]
pub struct MeasurementPattern {
    /// Measurement basis for each qubit
    pub measurements: HashMap<usize, MeasurementBasis>,
    /// Measurement order (important for adaptivity)
    pub order: Vec<usize>,
    /// Corrections to apply based on measurement outcomes
    pub x_corrections: HashMap<usize, Vec<(usize, bool)>>, // (source, sign)
    pub z_corrections: HashMap<usize, Vec<(usize, bool)>>,
    /// Input qubits (not measured)
    pub inputs: HashSet<usize>,
    /// Output qubits (measured last or not measured)
    pub outputs: HashSet<usize>,
}

impl MeasurementPattern {
    /// Create a new measurement pattern
    pub fn new() -> Self {
        Self {
            measurements: HashMap::new(),
            order: Vec::new(),
            x_corrections: HashMap::new(),
            z_corrections: HashMap::new(),
            inputs: HashSet::new(),
            outputs: HashSet::new(),
        }
    }

    /// Add a measurement
    pub fn add_measurement(&mut self, qubit: usize, basis: MeasurementBasis) {
        self.measurements.insert(qubit, basis);
        if !self.order.contains(&qubit) {
            self.order.push(qubit);
        }
    }

    /// Add X correction dependency
    pub fn add_x_correction(&mut self, target: usize, source: usize, sign: bool) {
        self.x_corrections
            .entry(target)
            .or_insert_with(Vec::new)
            .push((source, sign));
    }

    /// Add Z correction dependency
    pub fn add_z_correction(&mut self, target: usize, source: usize, sign: bool) {
        self.z_corrections
            .entry(target)
            .or_insert_with(Vec::new)
            .push((source, sign));
    }

    /// Set input qubits
    pub fn set_inputs(&mut self, inputs: Vec<usize>) {
        self.inputs = inputs.into_iter().collect();
    }

    /// Set output qubits
    pub fn set_outputs(&mut self, outputs: Vec<usize>) {
        self.outputs = outputs.into_iter().collect();
    }

    /// Create pattern for single-qubit rotation
    pub fn single_qubit_rotation(angle: f64) -> Self {
        let mut pattern = Self::new();

        // Three qubits: input (0), auxiliary (1), output (2)
        pattern.set_inputs(vec![0]);
        pattern.set_outputs(vec![2]);

        // Measure auxiliary qubit at angle
        pattern.add_measurement(1, MeasurementBasis::XY(angle));

        // Measure input qubit in X basis
        pattern.add_measurement(0, MeasurementBasis::X);

        // Corrections
        pattern.add_x_correction(2, 0, true);
        pattern.add_z_correction(2, 1, true);

        pattern
    }

    /// Create pattern for CNOT gate
    pub fn cnot() -> Self {
        let mut pattern = Self::new();

        // 15 qubits in standard CNOT pattern
        // Inputs: control (0), target (1)
        // Outputs: control (13), target (14)
        pattern.set_inputs(vec![0, 1]);
        pattern.set_outputs(vec![13, 14]);

        // Measurement order and bases (simplified)
        for i in 2..13 {
            pattern.add_measurement(i, MeasurementBasis::XY(PI / 2.0));
        }

        // Add corrections (simplified - full pattern is complex)
        pattern.add_x_correction(13, 0, true);
        pattern.add_x_correction(14, 1, true);

        pattern
    }
}

impl Default for MeasurementPattern {
    fn default() -> Self {
        Self::new()
    }
}

/// Cluster state for MBQC
pub struct ClusterState {
    /// Underlying graph structure
    pub graph: Graph,
    /// State vector (2^n complex amplitudes)
    pub state: Array1<Complex64>,
    /// Measured qubits and their outcomes
    pub measurements: HashMap<usize, bool>,
}

impl ClusterState {
    /// Create a cluster state from a graph
    pub fn from_graph(graph: Graph) -> QuantRS2Result<Self> {
        let n = graph.num_vertices;
        let dim = 1 << n;

        // Initialize all qubits in |+⟩ state
        let mut state = Array1::zeros(dim);
        state[0] = Complex64::new(1.0, 0.0);

        // Apply Hadamard to all qubits
        for i in 0..n {
            state = Self::apply_hadamard(&state, i, n)?;
        }

        // Apply CZ gates for each edge
        for (u, neighbors) in &graph.edges {
            for &v in neighbors {
                if u < &v {
                    state = Self::apply_cz(&state, *u, v, n)?;
                }
            }
        }

        // Normalize
        let norm = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        state = state / Complex64::new(norm, 0.0);

        Ok(Self {
            graph,
            state,
            measurements: HashMap::new(),
        })
    }

    /// Apply Hadamard gate to a qubit in the state vector
    fn apply_hadamard(
        state: &Array1<Complex64>,
        qubit: usize,
        n: usize,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = 1 << n;
        let mut new_state = Array1::zeros(dim);
        let h_factor = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            if bit == 0 {
                // |0⟩ -> |+⟩
                new_state[i] += h_factor * state[i];
                new_state[i | (1 << qubit)] += h_factor * state[i];
            } else {
                // |1⟩ -> |−⟩
                new_state[i & !(1 << qubit)] += h_factor * state[i];
                new_state[i] -= h_factor * state[i];
            }
        }

        Ok(new_state)
    }

    /// Apply CZ gate between two qubits
    fn apply_cz(
        state: &Array1<Complex64>,
        q1: usize,
        q2: usize,
        n: usize,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = 1 << n;
        let mut new_state = state.clone();

        for i in 0..dim {
            let bit1 = (i >> q1) & 1;
            let bit2 = (i >> q2) & 1;
            if bit1 == 1 && bit2 == 1 {
                new_state[i] *= -1.0;
            }
        }

        Ok(new_state)
    }

    /// Measure a qubit in a given basis
    pub fn measure(&mut self, qubit: usize, basis: MeasurementBasis) -> QuantRS2Result<bool> {
        if qubit >= self.graph.num_vertices {
            return Err(QuantRS2Error::InvalidInput(
                "Qubit index out of bounds".to_string(),
            ));
        }

        if self.measurements.contains_key(&qubit) {
            return Err(QuantRS2Error::InvalidInput(
                "Qubit already measured".to_string(),
            ));
        }

        // Apply basis rotation if needed
        let state = match basis {
            MeasurementBasis::Computational => self.state.clone(),
            MeasurementBasis::X => {
                Self::apply_hadamard(&self.state, qubit, self.graph.num_vertices)?
            }
            MeasurementBasis::Y => {
                // Apply S† then H
                let mut state = self.state.clone();
                for i in 0..state.len() {
                    if (i >> qubit) & 1 == 1 {
                        state[i] *= Complex64::new(0.0, -1.0);
                    }
                }
                Self::apply_hadamard(&state, qubit, self.graph.num_vertices)?
            }
            MeasurementBasis::XY(theta) => {
                // Apply rotation R_z(-θ) then H
                let mut state = self.state.clone();
                for i in 0..state.len() {
                    if (i >> qubit) & 1 == 1 {
                        state[i] *= Complex64::from_polar(1.0, -theta);
                    }
                }
                Self::apply_hadamard(&state, qubit, self.graph.num_vertices)?
            }
            _ => {
                return Err(QuantRS2Error::UnsupportedOperation(
                    "Measurement basis not yet implemented".to_string(),
                ));
            }
        };

        // Calculate probabilities
        let mut prob_0 = 0.0;
        let mut prob_1 = 0.0;

        for i in 0..state.len() {
            let bit = (i >> qubit) & 1;
            let prob = state[i].norm_sqr();
            if bit == 0 {
                prob_0 += prob;
            } else {
                prob_1 += prob;
            }
        }

        // Randomly choose outcome
        use scirs2_core::random::prelude::*;
        let outcome = if thread_rng().gen::<f64>() < prob_0 / (prob_0 + prob_1) {
            false
        } else {
            true
        };

        // Project state
        let norm = if outcome {
            prob_1.sqrt()
        } else {
            prob_0.sqrt()
        };
        let mut new_state = Array1::zeros(state.len());

        for i in 0..state.len() {
            let bit = (i >> qubit) & 1;
            if (bit == 1) == outcome {
                new_state[i] = state[i] / norm;
            }
        }

        self.state = new_state;
        self.measurements.insert(qubit, outcome);

        Ok(outcome)
    }

    /// Apply Pauli corrections based on measurement outcomes
    pub fn apply_corrections(
        &mut self,
        x_corrections: &HashMap<usize, Vec<(usize, bool)>>,
        z_corrections: &HashMap<usize, Vec<(usize, bool)>>,
    ) -> QuantRS2Result<()> {
        let _n = self.graph.num_vertices;

        // Apply X corrections
        for (target, sources) in x_corrections {
            let mut apply_x = false;
            for (source, sign) in sources {
                if let Some(&outcome) = self.measurements.get(source) {
                    if outcome && *sign {
                        apply_x = !apply_x;
                    }
                }
            }

            if apply_x && !self.measurements.contains_key(target) {
                // Apply X gate
                for i in 0..self.state.len() {
                    let bit = (i >> target) & 1;
                    if bit == 0 {
                        let j = i | (1 << target);
                        self.state.swap(i, j);
                    }
                }
            }
        }

        // Apply Z corrections
        for (target, sources) in z_corrections {
            let mut apply_z = false;
            for (source, sign) in sources {
                if let Some(&outcome) = self.measurements.get(source) {
                    if outcome && *sign {
                        apply_z = !apply_z;
                    }
                }
            }

            if apply_z && !self.measurements.contains_key(target) {
                // Apply Z gate
                for i in 0..self.state.len() {
                    if (i >> target) & 1 == 1 {
                        self.state[i] *= -1.0;
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the reduced state of unmeasured qubits
    pub fn get_output_state(&self, output_qubits: &[usize]) -> QuantRS2Result<Array1<Complex64>> {
        let n_out = output_qubits.len();
        let dim_out = 1 << n_out;
        let mut output_state = Array1::zeros(dim_out);

        // Map output qubits to indices
        let mut qubit_map = HashMap::new();
        for (i, &q) in output_qubits.iter().enumerate() {
            qubit_map.insert(q, i);
        }

        // Trace out measured qubits
        for i in 0..self.state.len() {
            let mut out_idx = 0;
            let mut valid = true;

            // Check measured qubits match their outcomes
            for (&q, &outcome) in &self.measurements {
                let bit = (i >> q) & 1;
                if (bit == 1) != outcome {
                    valid = false;
                    break;
                }
            }

            if valid {
                // Extract output qubit values
                for (j, &q) in output_qubits.iter().enumerate() {
                    if (i >> q) & 1 == 1 {
                        out_idx |= 1 << j;
                    }
                }

                output_state[out_idx] += self.state[i];
            }
        }

        // Normalize
        let norm = output_state
            .iter()
            .map(|c: &Complex64| c.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            output_state = output_state / Complex64::new(norm, 0.0);
        }

        Ok(output_state)
    }
}

/// MBQC computation flow
pub struct MBQCComputation {
    /// Cluster state
    pub cluster: ClusterState,
    /// Measurement pattern
    pub pattern: MeasurementPattern,
    /// Current step in computation
    pub current_step: usize,
}

impl MBQCComputation {
    /// Create a new MBQC computation
    pub fn new(graph: Graph, pattern: MeasurementPattern) -> QuantRS2Result<Self> {
        let cluster = ClusterState::from_graph(graph)?;

        Ok(Self {
            cluster,
            pattern,
            current_step: 0,
        })
    }

    /// Execute one measurement step
    pub fn step(&mut self) -> QuantRS2Result<Option<(usize, bool)>> {
        if self.current_step >= self.pattern.order.len() {
            return Ok(None);
        }

        let qubit = self.pattern.order[self.current_step];
        self.current_step += 1;

        // Skip if this is an output qubit that shouldn't be measured
        if self.pattern.outputs.contains(&qubit) && self.current_step == self.pattern.order.len() {
            return self.step();
        }

        // Get measurement basis
        let basis = self
            .pattern
            .measurements
            .get(&qubit)
            .copied()
            .unwrap_or(MeasurementBasis::Computational);

        // Perform measurement
        let outcome = self.cluster.measure(qubit, basis)?;

        // Apply corrections
        self.cluster
            .apply_corrections(&self.pattern.x_corrections, &self.pattern.z_corrections)?;

        Ok(Some((qubit, outcome)))
    }

    /// Execute all measurements
    pub fn run(&mut self) -> QuantRS2Result<HashMap<usize, bool>> {
        while self.step()?.is_some() {}
        Ok(self.cluster.measurements.clone())
    }

    /// Get the final output state
    pub fn output_state(&self) -> QuantRS2Result<Array1<Complex64>> {
        let outputs: Vec<usize> = self.pattern.outputs.iter().copied().collect();
        self.cluster.get_output_state(&outputs)
    }
}

/// Convert a quantum circuit to MBQC pattern
pub struct CircuitToMBQC {
    /// Qubit mapping from circuit to cluster
    #[allow(dead_code)]
    qubit_map: HashMap<usize, usize>,
    /// Current cluster size
    #[allow(dead_code)]
    cluster_size: usize,
}

impl CircuitToMBQC {
    /// Create a new converter
    pub fn new() -> Self {
        Self {
            qubit_map: HashMap::new(),
            cluster_size: 0,
        }
    }

    /// Convert a single-qubit gate to measurement pattern
    pub fn convert_single_qubit_gate(
        &mut self,
        _qubit: usize,
        angle: f64,
    ) -> (Graph, MeasurementPattern) {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(1, 2).unwrap();

        let pattern = MeasurementPattern::single_qubit_rotation(angle);

        (graph, pattern)
    }

    /// Convert CNOT gate to measurement pattern
    pub fn convert_cnot(&mut self, _control: usize, _target: usize) -> (Graph, MeasurementPattern) {
        // Standard 15-qubit CNOT pattern
        let mut graph = Graph::new(15);

        // Build the brickwork pattern
        for i in 0..5 {
            for j in 0..3 {
                let idx = i * 3 + j;
                if j < 2 {
                    graph.add_edge(idx, idx + 1).unwrap();
                }
                if i < 4 {
                    graph.add_edge(idx, idx + 3).unwrap();
                }
            }
        }

        let pattern = MeasurementPattern::cnot();

        (graph, pattern)
    }
}

impl Default for CircuitToMBQC {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_construction() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(1, 2).unwrap();
        graph.add_edge(2, 3).unwrap();

        assert_eq!(graph.neighbors(1).unwrap().len(), 2);
        assert!(graph.neighbors(1).unwrap().contains(&0));
        assert!(graph.neighbors(1).unwrap().contains(&2));
    }

    #[test]
    fn test_linear_cluster() {
        let graph = Graph::linear_cluster(5);
        assert_eq!(graph.num_vertices, 5);
        assert_eq!(graph.neighbors(2).unwrap().len(), 2);
        assert_eq!(graph.neighbors(0).unwrap().len(), 1);
        assert_eq!(graph.neighbors(4).unwrap().len(), 1);
    }

    #[test]
    fn test_rectangular_cluster() {
        let graph = Graph::rectangular_cluster(3, 3);
        assert_eq!(graph.num_vertices, 9);

        // Corner vertex has 2 neighbors
        assert_eq!(graph.neighbors(0).unwrap().len(), 2);

        // Center vertex has 4 neighbors
        assert_eq!(graph.neighbors(4).unwrap().len(), 4);
    }

    #[test]
    fn test_cluster_state_creation() {
        let graph = Graph::linear_cluster(3);
        let cluster = ClusterState::from_graph(graph).unwrap();

        // Check state is normalized
        let norm: f64 = cluster.state.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);

        // Check dimension
        assert_eq!(cluster.state.len(), 8); // 2^3
    }

    #[test]
    fn test_measurement_pattern() {
        let mut pattern = MeasurementPattern::new();
        pattern.add_measurement(0, MeasurementBasis::X);
        pattern.add_measurement(1, MeasurementBasis::XY(PI / 4.0));
        pattern.add_x_correction(2, 0, true);
        pattern.add_z_correction(2, 1, true);

        assert_eq!(pattern.measurements.len(), 2);
        assert_eq!(pattern.order.len(), 2);
        assert!(pattern.x_corrections.contains_key(&2));
    }

    #[test]
    fn test_single_qubit_measurement() {
        let graph = Graph::new(1);
        let mut cluster = ClusterState::from_graph(graph).unwrap();

        // Measure in X basis
        let outcome = cluster.measure(0, MeasurementBasis::X).unwrap();

        // Check qubit is marked as measured
        assert!(cluster.measurements.contains_key(&0));
        assert_eq!(cluster.measurements[&0], outcome);
    }

    #[test]
    fn test_mbqc_computation() {
        let graph = Graph::linear_cluster(3);
        let pattern = MeasurementPattern::single_qubit_rotation(PI / 4.0);

        let mut computation = MBQCComputation::new(graph, pattern).unwrap();

        // Run computation
        let outcomes = computation.run().unwrap();

        // Check measurements were performed
        assert!(outcomes.contains_key(&0));
        assert!(outcomes.contains_key(&1));
    }

    #[test]
    fn test_circuit_conversion() {
        let mut converter = CircuitToMBQC::new();

        // Convert single-qubit gate
        let (graph, pattern) = converter.convert_single_qubit_gate(0, PI / 2.0);
        assert_eq!(graph.num_vertices, 3);
        assert_eq!(pattern.measurements.len(), 2);

        // Convert CNOT
        let (graph, _pattern) = converter.convert_cnot(0, 1);
        assert_eq!(graph.num_vertices, 15);
    }
}
