//! Quantum Low-Density Parity-Check (LDPC) Codes with Belief Propagation Decoding
//!
//! This module implements quantum LDPC codes, which are a class of quantum error correction
//! codes characterized by sparse parity-check matrices. LDPC codes offer excellent error
//! correction performance with efficient decoding algorithms, making them practical for
//! large-scale quantum error correction.
//!
//! Key features:
//! - Sparse parity-check matrix construction and optimization
//! - Belief propagation decoding with message passing
//! - Support for both CSS and non-CSS LDPC codes
//! - Tanner graph representation for efficient processing
//! - Syndrome decoding with iterative refinement
//! - Performance analysis and threshold estimation
//! - Hardware-aware optimization for different architectures

use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::circuit_interfaces::{
    CircuitInterface, InterfaceCircuit, InterfaceGate, InterfaceGateType,
};
use crate::error::Result;

/// LDPC code construction method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LDPCConstructionMethod {
    /// Random regular LDPC codes
    RandomRegular,
    /// Progressive edge-growth (PEG) construction
    ProgressiveEdgeGrowth,
    /// Gallager construction
    Gallager,
    /// `MacKay` construction
    MacKay,
    /// Quantum-specific constructions
    QuantumBicycle,
    /// Surface code as LDPC
    SurfaceCode,
}

/// Belief propagation decoding algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefPropagationAlgorithm {
    /// Sum-product algorithm
    SumProduct,
    /// Min-sum algorithm (simplified)
    MinSum,
    /// Normalized min-sum
    NormalizedMinSum,
    /// Offset min-sum
    OffsetMinSum,
    /// Layered belief propagation
    LayeredBP,
}

/// LDPC code configuration
#[derive(Debug, Clone)]
pub struct LDPCConfig {
    /// Number of information bits
    pub k: usize,
    /// Code length (number of physical qubits)
    pub n: usize,
    /// Number of parity checks
    pub m: usize,
    /// Variable node degree (for regular codes)
    pub dv: usize,
    /// Check node degree (for regular codes)
    pub dc: usize,
    /// Construction method
    pub construction_method: LDPCConstructionMethod,
    /// Belief propagation algorithm
    pub bp_algorithm: BeliefPropagationAlgorithm,
    /// Maximum BP iterations
    pub max_bp_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Damping factor for message updates
    pub damping_factor: f64,
    /// Enable early termination
    pub early_termination: bool,
    /// Noise variance for channel
    pub noise_variance: f64,
}

impl Default for LDPCConfig {
    fn default() -> Self {
        Self {
            k: 10,
            n: 20,
            m: 10,
            dv: 3,
            dc: 6,
            construction_method: LDPCConstructionMethod::ProgressiveEdgeGrowth,
            bp_algorithm: BeliefPropagationAlgorithm::SumProduct,
            max_bp_iterations: 50,
            convergence_threshold: 1e-6,
            damping_factor: 0.8,
            early_termination: true,
            noise_variance: 0.1,
        }
    }
}

/// Tanner graph representation of LDPC code
#[derive(Debug, Clone)]
pub struct TannerGraph {
    /// Variable nodes (corresponding to qubits)
    pub variable_nodes: Vec<VariableNode>,
    /// Check nodes (corresponding to stabilizers)
    pub check_nodes: Vec<CheckNode>,
    /// Adjacency matrix (variable to check connections)
    pub adjacency_matrix: Array2<bool>,
    /// Parity check matrix H
    pub parity_check_matrix: Array2<u8>,
}

/// Variable node in Tanner graph
#[derive(Debug, Clone)]
pub struct VariableNode {
    /// Node index
    pub id: usize,
    /// Connected check nodes
    pub connected_checks: Vec<usize>,
    /// Current belief (log-likelihood ratio)
    pub belief: f64,
    /// Channel information
    pub channel_llr: f64,
    /// Incoming messages from check nodes
    pub incoming_messages: HashMap<usize, f64>,
}

/// Check node in Tanner graph
#[derive(Debug, Clone)]
pub struct CheckNode {
    /// Node index
    pub id: usize,
    /// Connected variable nodes
    pub connected_variables: Vec<usize>,
    /// Incoming messages from variable nodes
    pub incoming_messages: HashMap<usize, f64>,
    /// Syndrome value
    pub syndrome: bool,
}

impl VariableNode {
    /// Create new variable node
    #[must_use]
    pub fn new(id: usize) -> Self {
        Self {
            id,
            connected_checks: Vec::new(),
            belief: 0.0,
            channel_llr: 0.0,
            incoming_messages: HashMap::new(),
        }
    }

    /// Update belief based on incoming messages
    pub fn update_belief(&mut self) {
        self.belief = self.channel_llr + self.incoming_messages.values().sum::<f64>();
    }

    /// Compute outgoing message to specific check node
    #[must_use]
    pub fn compute_outgoing_message(&self, check_id: usize) -> f64 {
        let message_sum: f64 = self
            .incoming_messages
            .iter()
            .filter(|(&id, _)| id != check_id)
            .map(|(_, &value)| value)
            .sum();
        self.channel_llr + message_sum
    }
}

impl CheckNode {
    /// Create new check node
    #[must_use]
    pub fn new(id: usize) -> Self {
        Self {
            id,
            connected_variables: Vec::new(),
            incoming_messages: HashMap::new(),
            syndrome: false,
        }
    }

    /// Compute outgoing message to specific variable node using sum-product
    #[must_use]
    pub fn compute_outgoing_message_sum_product(&self, var_id: usize) -> f64 {
        let product: f64 = self
            .incoming_messages
            .iter()
            .filter(|(&id, _)| id != var_id)
            .map(|(_, &msg)| msg.tanh() / 2.0)
            .product();

        2.0 * product.atanh()
    }

    /// Compute outgoing message using min-sum algorithm
    #[must_use]
    pub fn compute_outgoing_message_min_sum(&self, var_id: usize) -> f64 {
        let other_messages: Vec<f64> = self
            .incoming_messages
            .iter()
            .filter(|(&id, _)| id != var_id)
            .map(|(_, &msg)| msg)
            .collect();

        if other_messages.is_empty() {
            return 0.0;
        }

        let sign_product: f64 = other_messages
            .iter()
            .map(|&msg| if msg >= 0.0 { 1.0 } else { -1.0 })
            .product();

        let min_magnitude = other_messages
            .iter()
            .map(|&msg| msg.abs())
            .fold(f64::INFINITY, f64::min);

        sign_product * min_magnitude
    }
}

/// Quantum LDPC code implementation
pub struct QuantumLDPCCode {
    /// Configuration
    config: LDPCConfig,
    /// Tanner graph representation
    tanner_graph: TannerGraph,
    /// X stabilizer generators
    #[allow(dead_code)]
    x_stabilizers: Array2<u8>,
    /// Z stabilizer generators
    #[allow(dead_code)]
    z_stabilizers: Array2<u8>,
    /// Logical X operators
    #[allow(dead_code)]
    logical_x_ops: Array2<u8>,
    /// Logical Z operators
    #[allow(dead_code)]
    logical_z_ops: Array2<u8>,
    /// Circuit interface
    #[allow(dead_code)]
    circuit_interface: CircuitInterface,
    /// Performance statistics
    stats: LDPCStats,
}

/// LDPC performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LDPCStats {
    /// Total decoding attempts
    pub total_decodings: usize,
    /// Successful decodings
    pub successful_decodings: usize,
    /// Average BP iterations
    pub avg_bp_iterations: f64,
    /// Block error rate
    pub block_error_rate: f64,
    /// Bit error rate
    pub bit_error_rate: f64,
    /// Average decoding time
    pub avg_decoding_time_ms: f64,
    /// Threshold estimate
    pub threshold_estimate: f64,
    /// Convergence rate
    pub convergence_rate: f64,
}

impl LDPCStats {
    /// Update statistics after decoding
    pub fn update_after_decoding(&mut self, success: bool, iterations: usize, time_ms: f64) {
        self.total_decodings += 1;
        if success {
            self.successful_decodings += 1;
        }

        let prev_avg_iter = self.avg_bp_iterations;
        self.avg_bp_iterations = prev_avg_iter
            .mul_add((self.total_decodings - 1) as f64, iterations as f64)
            / self.total_decodings as f64;

        let prev_avg_time = self.avg_decoding_time_ms;
        self.avg_decoding_time_ms = prev_avg_time
            .mul_add((self.total_decodings - 1) as f64, time_ms)
            / self.total_decodings as f64;

        self.block_error_rate =
            1.0 - (self.successful_decodings as f64 / self.total_decodings as f64);
        self.convergence_rate = self.successful_decodings as f64 / self.total_decodings as f64;
    }
}

/// Belief propagation decoding result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPDecodingResult {
    /// Decoded codeword
    pub decoded_bits: Vec<bool>,
    /// Final syndrome
    pub final_syndrome: Vec<bool>,
    /// Number of BP iterations
    pub iterations: usize,
    /// Converged successfully
    pub converged: bool,
    /// Final log-likelihood ratios
    pub final_llrs: Vec<f64>,
    /// Decoding time in milliseconds
    pub decoding_time_ms: f64,
    /// Error correction success
    pub success: bool,
}

impl QuantumLDPCCode {
    /// Create new quantum LDPC code
    pub fn new(config: LDPCConfig) -> Result<Self> {
        let circuit_interface = CircuitInterface::new(Default::default())?;

        // Construct Tanner graph
        let tanner_graph = Self::construct_tanner_graph(&config)?;

        // Generate stabilizer matrices
        let (x_stabilizers, z_stabilizers) =
            Self::generate_stabilizer_matrices(&config, &tanner_graph)?;

        // Generate logical operators
        let (logical_x_ops, logical_z_ops) =
            Self::generate_logical_operators(&config, &x_stabilizers, &z_stabilizers)?;

        Ok(Self {
            config,
            tanner_graph,
            x_stabilizers,
            z_stabilizers,
            logical_x_ops,
            logical_z_ops,
            circuit_interface,
            stats: LDPCStats::default(),
        })
    }

    /// Construct Tanner graph based on configuration
    fn construct_tanner_graph(config: &LDPCConfig) -> Result<TannerGraph> {
        match config.construction_method {
            LDPCConstructionMethod::RandomRegular => Self::construct_random_regular(config),
            LDPCConstructionMethod::ProgressiveEdgeGrowth => Self::construct_peg(config),
            LDPCConstructionMethod::Gallager => Self::construct_gallager(config),
            LDPCConstructionMethod::MacKay => Self::construct_mackay(config),
            LDPCConstructionMethod::QuantumBicycle => Self::construct_quantum_bicycle(config),
            LDPCConstructionMethod::SurfaceCode => Self::construct_surface_code(config),
        }
    }

    /// Construct random regular LDPC code
    fn construct_random_regular(config: &LDPCConfig) -> Result<TannerGraph> {
        let mut variable_nodes = Vec::with_capacity(config.n);
        let mut check_nodes = Vec::with_capacity(config.m);
        let mut adjacency_matrix = Array2::from_elem((config.n, config.m), false);

        // Initialize nodes
        for i in 0..config.n {
            variable_nodes.push(VariableNode::new(i));
        }
        for i in 0..config.m {
            check_nodes.push(CheckNode::new(i));
        }

        // Create edges for regular LDPC code
        let mut edges = Vec::new();

        // Add edges from variable nodes
        for var_id in 0..config.n {
            for _ in 0..config.dv {
                edges.push(var_id);
            }
        }

        // Shuffle edges for randomness
        for i in 0..edges.len() {
            let j = fastrand::usize(i..edges.len());
            edges.swap(i, j);
        }

        // Assign edges to check nodes
        let mut edge_idx = 0;
        for check_id in 0..config.m {
            for _ in 0..config.dc {
                if edge_idx < edges.len() {
                    let var_id = edges[edge_idx];

                    // Add connection
                    variable_nodes[var_id].connected_checks.push(check_id);
                    check_nodes[check_id].connected_variables.push(var_id);
                    adjacency_matrix[[var_id, check_id]] = true;

                    edge_idx += 1;
                }
            }
        }

        // Create parity check matrix
        let mut parity_check_matrix = Array2::zeros((config.m, config.n));
        for (var_id, check_id) in adjacency_matrix.indexed_iter() {
            if *check_id {
                parity_check_matrix[[var_id.1, var_id.0]] = 1;
            }
        }

        Ok(TannerGraph {
            variable_nodes,
            check_nodes,
            adjacency_matrix,
            parity_check_matrix,
        })
    }

    /// Construct LDPC code using Progressive Edge Growth (PEG)
    fn construct_peg(config: &LDPCConfig) -> Result<TannerGraph> {
        let mut variable_nodes = Vec::with_capacity(config.n);
        let mut check_nodes = Vec::with_capacity(config.m);
        let mut adjacency_matrix = Array2::from_elem((config.n, config.m), false);

        // Initialize nodes
        for i in 0..config.n {
            variable_nodes.push(VariableNode::new(i));
        }
        for i in 0..config.m {
            check_nodes.push(CheckNode::new(i));
        }

        // PEG algorithm: for each variable node, connect to checks that maximize girth
        for var_id in 0..config.n {
            let mut connected_checks = HashSet::new();

            for _ in 0..config.dv {
                let best_check = Self::find_best_check_for_peg(
                    var_id,
                    &connected_checks,
                    &variable_nodes,
                    &check_nodes,
                    &adjacency_matrix,
                    config.m,
                );

                if let Some(check_id) = best_check {
                    // Add connection
                    variable_nodes[var_id].connected_checks.push(check_id);
                    check_nodes[check_id].connected_variables.push(var_id);
                    adjacency_matrix[[var_id, check_id]] = true;
                    connected_checks.insert(check_id);
                }
            }
        }

        // Create parity check matrix
        let mut parity_check_matrix = Array2::zeros((config.m, config.n));
        for (var_id, check_id) in adjacency_matrix.indexed_iter() {
            if *check_id {
                parity_check_matrix[[var_id.1, var_id.0]] = 1;
            }
        }

        Ok(TannerGraph {
            variable_nodes,
            check_nodes,
            adjacency_matrix,
            parity_check_matrix,
        })
    }

    /// Find best check node for PEG construction
    fn find_best_check_for_peg(
        var_id: usize,
        connected_checks: &HashSet<usize>,
        _variable_nodes: &[VariableNode],
        _check_nodes: &[CheckNode],
        adjacency_matrix: &Array2<bool>,
        num_checks: usize,
    ) -> Option<usize> {
        let mut best_check = None;
        let mut best_girth = 0;

        for check_id in 0..num_checks {
            if connected_checks.contains(&check_id) {
                continue;
            }

            // Calculate local girth if we connect to this check
            let girth = Self::calculate_local_girth(var_id, check_id, adjacency_matrix);

            if girth > best_girth {
                best_girth = girth;
                best_check = Some(check_id);
            }
        }

        best_check
    }

    /// Calculate local girth around a potential edge
    fn calculate_local_girth(
        var_id: usize,
        check_id: usize,
        adjacency_matrix: &Array2<bool>,
    ) -> usize {
        // Simplified girth calculation - BFS to find shortest cycle
        let mut visited_vars = HashSet::new();
        let mut visited_checks = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back((var_id, 0, true)); // (node_id, distance, is_variable)
        visited_vars.insert(var_id);

        while let Some((node_id, dist, is_var)) = queue.pop_front() {
            if dist > 6 {
                // Limit search depth
                break;
            }

            if is_var {
                // Variable node - explore connected checks
                for (check_idx, &connected) in adjacency_matrix.row(node_id).indexed_iter() {
                    if connected && check_idx != check_id {
                        if visited_checks.contains(&check_idx) {
                            return dist * 2; // Found cycle
                        }
                        visited_checks.insert(check_idx);
                        queue.push_back((check_idx, dist + 1, false));
                    }
                }
            } else {
                // Check node - explore connected variables
                for (var_idx, &connected) in adjacency_matrix.column(node_id).indexed_iter() {
                    if connected && var_idx != var_id {
                        if visited_vars.contains(&var_idx) {
                            return dist * 2 + 1; // Found cycle
                        }
                        visited_vars.insert(var_idx);
                        queue.push_back((var_idx, dist + 1, true));
                    }
                }
            }
        }

        12 // Default large girth if no cycle found
    }

    /// Construct Gallager LDPC code
    fn construct_gallager(config: &LDPCConfig) -> Result<TannerGraph> {
        // For simplicity, use random regular construction
        // Real Gallager construction would be more sophisticated
        Self::construct_random_regular(config)
    }

    /// Construct `MacKay` LDPC code
    fn construct_mackay(config: &LDPCConfig) -> Result<TannerGraph> {
        // For simplicity, use PEG construction
        // Real MacKay construction would use specific algorithms
        Self::construct_peg(config)
    }

    /// Construct quantum bicycle code
    fn construct_quantum_bicycle(config: &LDPCConfig) -> Result<TannerGraph> {
        // Simplified bicycle code construction
        let mut variable_nodes = Vec::with_capacity(config.n);
        let mut check_nodes = Vec::with_capacity(config.m);
        let mut adjacency_matrix = Array2::from_elem((config.n, config.m), false);

        // Initialize nodes
        for i in 0..config.n {
            variable_nodes.push(VariableNode::new(i));
        }
        for i in 0..config.m {
            check_nodes.push(CheckNode::new(i));
        }

        // Bicycle code structure: cyclic connections
        let l = config.n / 2;
        for i in 0..l {
            for j in 0..config.dv {
                let check_id = (i + j * l / config.dv) % config.m;

                // Connect variable i to check_id
                variable_nodes[i].connected_checks.push(check_id);
                check_nodes[check_id].connected_variables.push(i);
                adjacency_matrix[[i, check_id]] = true;

                // Connect variable i+l to check_id (bicycle structure)
                if i + l < config.n {
                    variable_nodes[i + l].connected_checks.push(check_id);
                    check_nodes[check_id].connected_variables.push(i + l);
                    adjacency_matrix[[i + l, check_id]] = true;
                }
            }
        }

        // Create parity check matrix
        let mut parity_check_matrix = Array2::zeros((config.m, config.n));
        for (var_id, check_id) in adjacency_matrix.indexed_iter() {
            if *check_id {
                parity_check_matrix[[var_id.1, var_id.0]] = 1;
            }
        }

        Ok(TannerGraph {
            variable_nodes,
            check_nodes,
            adjacency_matrix,
            parity_check_matrix,
        })
    }

    /// Construct surface code as LDPC
    fn construct_surface_code(config: &LDPCConfig) -> Result<TannerGraph> {
        // Simplified surface code construction
        let d = (config.n as f64).sqrt() as usize; // Assume square lattice
        let mut variable_nodes = Vec::with_capacity(config.n);
        let mut check_nodes = Vec::with_capacity(config.m);
        let mut adjacency_matrix = Array2::from_elem((config.n, config.m), false);

        // Initialize nodes
        for i in 0..config.n {
            variable_nodes.push(VariableNode::new(i));
        }
        for i in 0..config.m {
            check_nodes.push(CheckNode::new(i));
        }

        // Surface code structure: each stabilizer connects to 4 neighboring qubits
        for check_id in 0..config.m {
            let row = check_id / d;
            let col = check_id % d;

            // Connect to neighboring qubits (up, down, left, right)
            let neighbors = [
                (row.wrapping_sub(1), col),
                (row + 1, col),
                (row, col.wrapping_sub(1)),
                (row, col + 1),
            ];

            for (r, c) in &neighbors {
                if *r < d && *c < d {
                    let var_id = r * d + c;
                    if var_id < config.n {
                        variable_nodes[var_id].connected_checks.push(check_id);
                        check_nodes[check_id].connected_variables.push(var_id);
                        adjacency_matrix[[var_id, check_id]] = true;
                    }
                }
            }
        }

        // Create parity check matrix
        let mut parity_check_matrix = Array2::zeros((config.m, config.n));
        for (var_id, check_id) in adjacency_matrix.indexed_iter() {
            if *check_id {
                parity_check_matrix[[var_id.1, var_id.0]] = 1;
            }
        }

        Ok(TannerGraph {
            variable_nodes,
            check_nodes,
            adjacency_matrix,
            parity_check_matrix,
        })
    }

    /// Generate stabilizer matrices for quantum LDPC code
    fn generate_stabilizer_matrices(
        _config: &LDPCConfig,
        tanner_graph: &TannerGraph,
    ) -> Result<(Array2<u8>, Array2<u8>)> {
        // For CSS codes, X and Z stabilizers are independent
        let x_stabilizers = tanner_graph.parity_check_matrix.clone();
        let z_stabilizers = tanner_graph.parity_check_matrix.clone();

        Ok((x_stabilizers, z_stabilizers))
    }

    /// Generate logical operators
    fn generate_logical_operators(
        config: &LDPCConfig,
        _x_stabilizers: &Array2<u8>,
        _z_stabilizers: &Array2<u8>,
    ) -> Result<(Array2<u8>, Array2<u8>)> {
        let k = config.k;
        let n = config.n;

        // Simplified logical operator generation
        let mut logical_x_ops = Array2::zeros((k, n));
        let mut logical_z_ops = Array2::zeros((k, n));

        // For demonstration, create simple logical operators
        for i in 0..k.min(n) {
            logical_x_ops[[i, i]] = 1;
            logical_z_ops[[i, i]] = 1;
        }

        Ok((logical_x_ops, logical_z_ops))
    }

    /// Perform belief propagation decoding
    pub fn decode_belief_propagation(
        &mut self,
        received_llrs: &[f64],
        syndrome: &[bool],
    ) -> Result<BPDecodingResult> {
        let start_time = std::time::Instant::now();

        // Initialize channel LLRs
        for (i, &llr) in received_llrs.iter().enumerate() {
            if i < self.tanner_graph.variable_nodes.len() {
                self.tanner_graph.variable_nodes[i].channel_llr = llr;
            }
        }

        // Set syndromes
        for (i, &syn) in syndrome.iter().enumerate() {
            if i < self.tanner_graph.check_nodes.len() {
                self.tanner_graph.check_nodes[i].syndrome = syn;
            }
        }

        let mut converged = false;
        let mut iteration = 0;

        // Belief propagation iterations
        while iteration < self.config.max_bp_iterations && !converged {
            // Variable to check messages
            self.update_variable_to_check_messages();

            // Check to variable messages
            self.update_check_to_variable_messages();

            // Update variable beliefs
            self.update_variable_beliefs();

            // Check convergence
            converged = self.check_convergence();

            if self.config.early_termination && converged {
                break;
            }

            iteration += 1;
        }

        // Extract final decoded bits
        let decoded_bits = self.extract_decoded_bits();

        // Calculate final syndrome
        let final_syndrome = self.calculate_syndrome(&decoded_bits);

        // Check if decoding was successful
        let success = final_syndrome.iter().all(|&s| !s);

        let decoding_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        // Update statistics
        self.stats
            .update_after_decoding(success, iteration, decoding_time_ms);

        // Extract final LLRs
        let final_llrs: Vec<f64> = self
            .tanner_graph
            .variable_nodes
            .iter()
            .map(|node| node.belief)
            .collect();

        Ok(BPDecodingResult {
            decoded_bits,
            final_syndrome,
            iterations: iteration,
            converged,
            final_llrs,
            decoding_time_ms,
            success,
        })
    }

    /// Update variable to check messages
    fn update_variable_to_check_messages(&mut self) {
        for var_node in &mut self.tanner_graph.variable_nodes {
            for &check_id in &var_node.connected_checks.clone() {
                let message = var_node.compute_outgoing_message(check_id);

                // Apply damping
                let old_message = self.tanner_graph.check_nodes[check_id]
                    .incoming_messages
                    .get(&var_node.id)
                    .unwrap_or(&0.0);

                let damped_message = self
                    .config
                    .damping_factor
                    .mul_add(message, (1.0 - self.config.damping_factor) * old_message);

                self.tanner_graph.check_nodes[check_id]
                    .incoming_messages
                    .insert(var_node.id, damped_message);
            }
        }
    }

    /// Update check to variable messages
    fn update_check_to_variable_messages(&mut self) {
        for check_node in &mut self.tanner_graph.check_nodes {
            for &var_id in &check_node.connected_variables.clone() {
                let message = match self.config.bp_algorithm {
                    BeliefPropagationAlgorithm::SumProduct => {
                        check_node.compute_outgoing_message_sum_product(var_id)
                    }
                    BeliefPropagationAlgorithm::MinSum => {
                        check_node.compute_outgoing_message_min_sum(var_id)
                    }
                    BeliefPropagationAlgorithm::NormalizedMinSum => {
                        let min_sum_msg = check_node.compute_outgoing_message_min_sum(var_id);
                        min_sum_msg * 0.75 // Normalization factor
                    }
                    BeliefPropagationAlgorithm::OffsetMinSum => {
                        let min_sum_msg = check_node.compute_outgoing_message_min_sum(var_id);
                        let offset = 0.1;
                        if min_sum_msg.abs() > offset {
                            min_sum_msg.signum() * (min_sum_msg.abs() - offset)
                        } else {
                            0.0
                        }
                    }
                    BeliefPropagationAlgorithm::LayeredBP => {
                        // Simplified layered BP
                        check_node.compute_outgoing_message_sum_product(var_id)
                    }
                };

                // Apply damping
                let old_message = self.tanner_graph.variable_nodes[var_id]
                    .incoming_messages
                    .get(&check_node.id)
                    .unwrap_or(&0.0);

                let damped_message = self
                    .config
                    .damping_factor
                    .mul_add(message, (1.0 - self.config.damping_factor) * old_message);

                self.tanner_graph.variable_nodes[var_id]
                    .incoming_messages
                    .insert(check_node.id, damped_message);
            }
        }
    }

    /// Update variable node beliefs
    fn update_variable_beliefs(&mut self) {
        for var_node in &mut self.tanner_graph.variable_nodes {
            var_node.update_belief();
        }
    }

    /// Check convergence of belief propagation
    fn check_convergence(&self) -> bool {
        // Check if syndrome is satisfied
        let decoded_bits = self.extract_decoded_bits();
        let syndrome = self.calculate_syndrome(&decoded_bits);

        syndrome.iter().all(|&s| !s)
    }

    /// Extract decoded bits from current beliefs
    fn extract_decoded_bits(&self) -> Vec<bool> {
        self.tanner_graph
            .variable_nodes
            .iter()
            .map(|node| node.belief < 0.0)
            .collect()
    }

    /// Calculate syndrome for given codeword
    fn calculate_syndrome(&self, codeword: &[bool]) -> Vec<bool> {
        let mut syndrome = vec![false; self.tanner_graph.check_nodes.len()];

        for (check_id, check_node) in self.tanner_graph.check_nodes.iter().enumerate() {
            let mut parity = false;
            for &var_id in &check_node.connected_variables {
                if var_id < codeword.len() && codeword[var_id] {
                    parity = !parity;
                }
            }
            syndrome[check_id] = parity;
        }

        syndrome
    }

    /// Generate syndrome extraction circuit
    pub fn syndrome_circuit(&self) -> Result<InterfaceCircuit> {
        let num_data_qubits = self.config.n;
        let num_syndrome_qubits = self.config.m;
        let total_qubits = num_data_qubits + num_syndrome_qubits;

        let mut circuit = InterfaceCircuit::new(total_qubits, num_syndrome_qubits);

        // Add syndrome extraction gates based on parity check matrix
        for (check_id, check_node) in self.tanner_graph.check_nodes.iter().enumerate() {
            let syndrome_qubit = num_data_qubits + check_id;

            for &var_id in &check_node.connected_variables {
                if var_id < num_data_qubits {
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::CNOT,
                        vec![var_id, syndrome_qubit],
                    ));
                }
            }
        }

        Ok(circuit)
    }

    /// Estimate error threshold
    pub fn estimate_threshold(
        &mut self,
        noise_range: (f64, f64),
        num_trials: usize,
    ) -> Result<f64> {
        let (min_noise, max_noise) = noise_range;
        let mut threshold = f64::midpoint(min_noise, max_noise);
        let mut search_range = max_noise - min_noise;

        // Binary search for threshold
        while search_range > 0.001 {
            self.config.noise_variance = threshold;

            let mut successes = 0;
            for _ in 0..num_trials {
                // Generate random errors
                let errors: Vec<bool> = (0..self.config.n)
                    .map(|_| fastrand::f64() < threshold)
                    .collect();

                // Generate LLRs from errors
                let llrs: Vec<f64> = errors
                    .iter()
                    .map(|&error| {
                        if error {
                            -2.0 / self.config.noise_variance
                        } else {
                            2.0 / self.config.noise_variance
                        }
                    })
                    .collect();

                // Calculate syndrome
                let syndrome = self.calculate_syndrome(&errors);

                // Attempt decoding
                if let Ok(result) = self.decode_belief_propagation(&llrs, &syndrome) {
                    if result.success {
                        successes += 1;
                    }
                }
            }

            let success_rate = f64::from(successes) / num_trials as f64;

            if success_rate > 0.5 {
                threshold = f64::midpoint(threshold, max_noise);
            } else {
                threshold = f64::midpoint(min_noise, threshold);
            }

            search_range /= 2.0;
        }

        self.stats.threshold_estimate = threshold;
        Ok(threshold)
    }

    /// Get current statistics
    #[must_use]
    pub const fn get_stats(&self) -> &LDPCStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = LDPCStats::default();
    }

    /// Get code parameters
    #[must_use]
    pub const fn get_parameters(&self) -> (usize, usize, usize) {
        (self.config.n, self.config.k, self.config.m)
    }

    /// Get Tanner graph
    #[must_use]
    pub const fn get_tanner_graph(&self) -> &TannerGraph {
        &self.tanner_graph
    }
}

/// Benchmark quantum LDPC codes performance
pub fn benchmark_quantum_ldpc_codes() -> Result<HashMap<String, f64>> {
    let mut results = HashMap::new();

    // Test different LDPC configurations
    let configs = vec![
        LDPCConfig {
            k: 10,
            n: 20,
            m: 10,
            construction_method: LDPCConstructionMethod::RandomRegular,
            bp_algorithm: BeliefPropagationAlgorithm::SumProduct,
            ..Default::default()
        },
        LDPCConfig {
            k: 15,
            n: 30,
            m: 15,
            construction_method: LDPCConstructionMethod::ProgressiveEdgeGrowth,
            bp_algorithm: BeliefPropagationAlgorithm::MinSum,
            ..Default::default()
        },
        LDPCConfig {
            k: 20,
            n: 40,
            m: 20,
            construction_method: LDPCConstructionMethod::QuantumBicycle,
            bp_algorithm: BeliefPropagationAlgorithm::NormalizedMinSum,
            ..Default::default()
        },
    ];

    for (i, config) in configs.into_iter().enumerate() {
        let start = std::time::Instant::now();

        let mut ldpc_code = QuantumLDPCCode::new(config)?;

        // Perform multiple decoding tests
        for _ in 0..50 {
            // Generate test errors
            let errors: Vec<bool> = (0..ldpc_code.config.n)
                .map(|_| fastrand::f64() < 0.05)
                .collect();

            // Generate LLRs
            let llrs: Vec<f64> = errors
                .iter()
                .map(|&error| if error { -1.0 } else { 1.0 })
                .collect();

            // Calculate syndrome
            let syndrome = ldpc_code.calculate_syndrome(&errors);

            // Decode
            let _result = ldpc_code.decode_belief_propagation(&llrs, &syndrome)?;
        }

        let time = start.elapsed().as_secs_f64() * 1000.0;
        results.insert(format!("config_{i}"), time);

        // Add performance metrics
        let stats = ldpc_code.get_stats();
        results.insert(format!("config_{i}_success_rate"), stats.convergence_rate);
        results.insert(
            format!("config_{i}_avg_iterations"),
            stats.avg_bp_iterations,
        );
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_ldpc_code_creation() {
        let config = LDPCConfig::default();
        let ldpc_code = QuantumLDPCCode::new(config);
        assert!(ldpc_code.is_ok());
    }

    #[test]
    fn test_tanner_graph_construction() {
        let config = LDPCConfig {
            k: 5,
            n: 10,
            m: 5,
            dv: 2,
            dc: 4,
            ..Default::default()
        };

        let tanner_graph = QuantumLDPCCode::construct_random_regular(&config);
        assert!(tanner_graph.is_ok());

        let graph = tanner_graph.expect("tanner_graph construction should succeed");
        assert_eq!(graph.variable_nodes.len(), 10);
        assert_eq!(graph.check_nodes.len(), 5);
    }

    #[test]
    fn test_variable_node_operations() {
        let mut var_node = VariableNode::new(0);
        var_node.channel_llr = 1.0;
        var_node.incoming_messages.insert(1, 0.5);
        var_node.incoming_messages.insert(2, -0.3);

        var_node.update_belief();
        assert_abs_diff_eq!(var_node.belief, 1.2, epsilon = 1e-10);

        let outgoing = var_node.compute_outgoing_message(1);
        assert_abs_diff_eq!(outgoing, 0.7, epsilon = 1e-10);
    }

    #[test]
    fn test_check_node_operations() {
        let mut check_node = CheckNode::new(0);
        check_node.incoming_messages.insert(1, 0.8);
        check_node.incoming_messages.insert(2, -0.6);
        check_node.incoming_messages.insert(3, 1.2);

        let sum_product_msg = check_node.compute_outgoing_message_sum_product(1);
        assert!(sum_product_msg.is_finite());

        let min_sum_msg = check_node.compute_outgoing_message_min_sum(1);
        assert_abs_diff_eq!(min_sum_msg.abs(), 0.6, epsilon = 1e-10);
    }

    #[test]
    fn test_belief_propagation_decoding() {
        let config = LDPCConfig {
            k: 3,
            n: 6,
            m: 3,
            max_bp_iterations: 10,
            ..Default::default()
        };

        let mut ldpc_code =
            QuantumLDPCCode::new(config).expect("LDPC code creation should succeed");

        let llrs = vec![1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
        let syndrome = vec![false, true, false];

        let result = ldpc_code.decode_belief_propagation(&llrs, &syndrome);
        assert!(result.is_ok());

        let bp_result = result.expect("decode_belief_propagation should succeed");
        assert_eq!(bp_result.decoded_bits.len(), 6);
        assert!(bp_result.iterations <= 10);
    }

    #[test]
    fn test_syndrome_calculation() {
        let config = LDPCConfig {
            k: 2,
            n: 4,
            m: 2,
            ..Default::default()
        };

        let ldpc_code = QuantumLDPCCode::new(config).expect("LDPC code creation should succeed");

        let codeword = vec![false, true, false, true];
        let syndrome = ldpc_code.calculate_syndrome(&codeword);

        assert_eq!(syndrome.len(), 2);
    }

    #[test]
    fn test_syndrome_circuit_generation() {
        let config = LDPCConfig {
            k: 3,
            n: 6,
            m: 3,
            ..Default::default()
        };

        let ldpc_code = QuantumLDPCCode::new(config).expect("LDPC code creation should succeed");
        let circuit = ldpc_code.syndrome_circuit();

        assert!(circuit.is_ok());
        let syndrome_circuit = circuit.expect("syndrome_circuit should succeed");
        assert_eq!(syndrome_circuit.num_qubits, 9); // 6 data + 3 syndrome
    }

    #[test]
    fn test_different_construction_methods() {
        let base_config = LDPCConfig {
            k: 3,
            n: 6,
            m: 3,
            ..Default::default()
        };

        let methods = vec![
            LDPCConstructionMethod::RandomRegular,
            LDPCConstructionMethod::ProgressiveEdgeGrowth,
            LDPCConstructionMethod::QuantumBicycle,
            LDPCConstructionMethod::SurfaceCode,
        ];

        for method in methods {
            let mut config = base_config.clone();
            config.construction_method = method;

            let ldpc_code = QuantumLDPCCode::new(config);
            assert!(ldpc_code.is_ok(), "Failed for method: {method:?}");
        }
    }

    #[test]
    fn test_different_bp_algorithms() {
        let base_config = LDPCConfig {
            k: 3,
            n: 6,
            m: 3,
            ..Default::default()
        };

        let algorithms = vec![
            BeliefPropagationAlgorithm::SumProduct,
            BeliefPropagationAlgorithm::MinSum,
            BeliefPropagationAlgorithm::NormalizedMinSum,
            BeliefPropagationAlgorithm::OffsetMinSum,
        ];

        for algorithm in algorithms {
            let mut config = base_config.clone();
            config.bp_algorithm = algorithm;

            let mut ldpc_code =
                QuantumLDPCCode::new(config).expect("LDPC code creation should succeed");

            let llrs = vec![1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
            let syndrome = vec![false, true, false];

            let result = ldpc_code.decode_belief_propagation(&llrs, &syndrome);
            assert!(result.is_ok(), "Failed for algorithm: {algorithm:?}");
        }
    }

    #[test]
    fn test_stats_updates() {
        let mut stats = LDPCStats::default();

        stats.update_after_decoding(true, 5, 10.0);
        assert_eq!(stats.total_decodings, 1);
        assert_eq!(stats.successful_decodings, 1);
        assert_abs_diff_eq!(stats.avg_bp_iterations, 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.block_error_rate, 0.0, epsilon = 1e-10);

        stats.update_after_decoding(false, 8, 15.0);
        assert_eq!(stats.total_decodings, 2);
        assert_eq!(stats.successful_decodings, 1);
        assert_abs_diff_eq!(stats.avg_bp_iterations, 6.5, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.block_error_rate, 0.5, epsilon = 1e-10);
    }
}
