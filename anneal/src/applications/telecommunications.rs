//! Telecommunications Industry Optimization
//!
//! This module provides optimization solutions for the telecommunications industry,
//! including network topology optimization, traffic routing, infrastructure placement,
//! spectrum allocation, and quality of service optimization.

use std::collections::HashMap;
use crate::qubo::{QuboFormulation, QuboBuilder};
use crate::ising::IsingModel;
use crate::simulator::{AnnealingParams, ClassicalAnnealingSimulator};
use super::{ApplicationError, ApplicationResult, OptimizationProblem, IndustrySolution, IndustryConstraint, IndustryObjective};

/// Network Topology Optimization Problem
#[derive(Debug, Clone)]
pub struct NetworkTopologyOptimization {
    /// Number of network nodes
    pub num_nodes: usize,
    /// Potential connections between nodes
    pub potential_connections: Vec<(usize, usize)>,
    /// Connection costs
    pub connection_costs: Vec<f64>,
    /// Connection capacities
    pub connection_capacities: Vec<f64>,
    /// Traffic demand matrix [source][destination]
    pub traffic_demands: Vec<Vec<f64>>,
    /// Reliability requirements for connections
    pub reliability_requirements: Vec<f64>,
    /// Maximum network latency allowed
    pub max_latency: f64,
    /// Redundancy requirements
    pub redundancy_level: usize,
    /// Quality of Service constraints
    pub qos_constraints: Vec<IndustryConstraint>,
}

impl NetworkTopologyOptimization {
    /// Create a new network topology optimization problem
    pub fn new(
        num_nodes: usize,
        potential_connections: Vec<(usize, usize)>,
        connection_costs: Vec<f64>,
        traffic_demands: Vec<Vec<f64>>,
    ) -> ApplicationResult<Self> {
        if connection_costs.len() != potential_connections.len() {
            return Err(ApplicationError::InvalidConfiguration(
                "Connection costs must match number of potential connections".to_string()
            ));
        }
        
        if traffic_demands.len() != num_nodes {
            return Err(ApplicationError::InvalidConfiguration(
                "Traffic demand matrix dimension mismatch".to_string()
            ));
        }
        
        for row in &traffic_demands {
            if row.len() != num_nodes {
                return Err(ApplicationError::InvalidConfiguration(
                    "Traffic demand matrix is not square".to_string()
                ));
            }
        }
        
        Ok(Self {
            num_nodes,
            potential_connections: potential_connections.clone(),
            connection_costs,
            connection_capacities: vec![100.0; potential_connections.len()], // Default capacity
            traffic_demands,
            reliability_requirements: vec![0.99; potential_connections.len()], // 99% reliability
            max_latency: 100.0, // 100ms max latency
            redundancy_level: 2, // Dual redundancy
            qos_constraints: Vec::new(),
        })
    }
    
    /// Set connection capacities
    pub fn set_connection_capacities(&mut self, capacities: Vec<f64>) -> ApplicationResult<()> {
        if capacities.len() != self.potential_connections.len() {
            return Err(ApplicationError::InvalidConfiguration(
                "Capacities must match number of potential connections".to_string()
            ));
        }
        
        self.connection_capacities = capacities;
        Ok(())
    }
    
    /// Calculate network connectivity
    pub fn calculate_connectivity(&self, topology: &NetworkTopology) -> f64 {
        let mut connectivity_matrix = vec![vec![false; self.num_nodes]; self.num_nodes];
        
        // Mark direct connections
        for (i, &active) in topology.active_connections.iter().enumerate() {
            if active {
                let (u, v) = self.potential_connections[i];
                connectivity_matrix[u][v] = true;
                connectivity_matrix[v][u] = true;
            }
        }
        
        // Floyd-Warshall to find all-pairs connectivity
        for k in 0..self.num_nodes {
            for i in 0..self.num_nodes {
                for j in 0..self.num_nodes {
                    connectivity_matrix[i][j] |= connectivity_matrix[i][k] && connectivity_matrix[k][j];
                }
            }
        }
        
        // Calculate connectivity ratio
        let mut connected_pairs = 0;
        let total_pairs = self.num_nodes * (self.num_nodes - 1);
        
        for i in 0..self.num_nodes {
            for j in 0..self.num_nodes {
                if i != j && connectivity_matrix[i][j] {
                    connected_pairs += 1;
                }
            }
        }
        
        connected_pairs as f64 / total_pairs as f64
    }
    
    /// Calculate network latency
    pub fn calculate_network_latency(&self, topology: &NetworkTopology) -> f64 {
        // Simplified latency calculation based on path lengths
        let mut distance_matrix = vec![vec![f64::INFINITY; self.num_nodes]; self.num_nodes];
        
        // Initialize distances
        for i in 0..self.num_nodes {
            distance_matrix[i][i] = 0.0;
        }
        
        for (i, &active) in topology.active_connections.iter().enumerate() {
            if active {
                let (u, v) = self.potential_connections[i];
                let latency = 1.0; // Simplified: 1ms per hop
                distance_matrix[u][v] = latency;
                distance_matrix[v][u] = latency;
            }
        }
        
        // Floyd-Warshall for shortest paths
        for k in 0..self.num_nodes {
            for i in 0..self.num_nodes {
                for j in 0..self.num_nodes {
                    if distance_matrix[i][k] + distance_matrix[k][j] < distance_matrix[i][j] {
                        distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j];
                    }
                }
            }
        }
        
        // Average latency for all connected pairs
        let mut total_latency = 0.0;
        let mut connected_pairs = 0;
        
        for i in 0..self.num_nodes {
            for j in 0..self.num_nodes {
                if i != j && distance_matrix[i][j] < f64::INFINITY {
                    total_latency += distance_matrix[i][j];
                    connected_pairs += 1;
                }
            }
        }
        
        if connected_pairs > 0 {
            total_latency / connected_pairs as f64
        } else {
            f64::INFINITY
        }
    }
    
    /// Calculate total network cost
    pub fn calculate_total_cost(&self, topology: &NetworkTopology) -> f64 {
        topology.active_connections.iter().enumerate()
            .filter(|(_, &active)| active)
            .map(|(i, _)| self.connection_costs[i])
            .sum()
    }
}

impl OptimizationProblem for NetworkTopologyOptimization {
    type Solution = NetworkTopology;
    type ObjectiveValue = f64;
    
    fn description(&self) -> String {
        format!(
            "Network topology optimization with {} nodes and {} potential connections",
            self.num_nodes,
            self.potential_connections.len()
        )
    }
    
    fn size_metrics(&self) -> HashMap<String, usize> {
        let mut metrics = HashMap::new();
        metrics.insert("num_nodes".to_string(), self.num_nodes);
        metrics.insert("num_potential_connections".to_string(), self.potential_connections.len());
        metrics.insert("num_qos_constraints".to_string(), self.qos_constraints.len());
        metrics
    }
    
    fn validate(&self) -> ApplicationResult<()> {
        if self.num_nodes < 2 {
            return Err(ApplicationError::DataValidationError(
                "At least 2 nodes required".to_string()
            ));
        }
        
        if self.potential_connections.is_empty() {
            return Err(ApplicationError::DataValidationError(
                "At least one potential connection required".to_string()
            ));
        }
        
        // Validate node indices in connections
        for &(u, v) in &self.potential_connections {
            if u >= self.num_nodes || v >= self.num_nodes {
                return Err(ApplicationError::DataValidationError(
                    "Connection references invalid node index".to_string()
                ));
            }
        }
        
        // Check positive costs and capacities
        for &cost in &self.connection_costs {
            if cost < 0.0 {
                return Err(ApplicationError::DataValidationError(
                    "Connection costs must be non-negative".to_string()
                ));
            }
        }
        
        for &capacity in &self.connection_capacities {
            if capacity <= 0.0 {
                return Err(ApplicationError::DataValidationError(
                    "Connection capacities must be positive".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    fn to_qubo(&self) -> ApplicationResult<QuboFormulation> {
        let mut builder = QuboBuilder::new();
        let num_connections = self.potential_connections.len();
        
        // Binary variables: x[i] = 1 if connection i is active
        for i in 0..num_connections {
            // Cost term
            builder.add_bias(i, self.connection_costs[i]);
            
            // Connectivity benefit (negative to encourage connections)
            let connectivity_benefit = -50.0; // Encourage network connectivity
            builder.add_bias(i, connectivity_benefit);
        }
        
        // Redundancy constraints: ensure multiple paths between critical nodes
        let redundancy_penalty = 1000.0;
        for node in 0..self.num_nodes {
            let mut adjacent_connections = Vec::new();
            
            for (conn_idx, &(u, v)) in self.potential_connections.iter().enumerate() {
                if u == node || v == node {
                    adjacent_connections.push(conn_idx);
                }
            }
            
            // Penalty for having too few connections per node
            if adjacent_connections.len() >= self.redundancy_level {
                for i in 0..adjacent_connections.len() {
                    for j in (i + 1)..adjacent_connections.len() {
                        let var1 = adjacent_connections[i];
                        let var2 = adjacent_connections[j];
                        // Encourage having multiple connections per node
                        builder.add_coupling(var1, var2, -redundancy_penalty / 2.0);
                    }
                }
            }
        }
        
        // Traffic capacity constraints
        let capacity_penalty = 500.0;
        for (conn_idx, &(u, v)) in self.potential_connections.iter().enumerate() {
            let traffic_demand = self.traffic_demands[u][v] + self.traffic_demands[v][u];
            let capacity = self.connection_capacities[conn_idx];
            
            if traffic_demand > capacity {
                // Penalty for insufficient capacity
                builder.add_bias(conn_idx, capacity_penalty * (traffic_demand - capacity));
            }
        }
        
        builder.build()
    }
    
    fn evaluate_solution(&self, solution: &Self::Solution) -> ApplicationResult<Self::ObjectiveValue> {
        let total_cost = self.calculate_total_cost(solution);
        let connectivity = self.calculate_connectivity(solution);
        let latency = self.calculate_network_latency(solution);
        
        // Multi-objective: minimize cost and latency, maximize connectivity
        let connectivity_weight = 100.0;
        let latency_penalty = 10.0;
        
        Ok(total_cost + latency_penalty * latency - connectivity_weight * connectivity)
    }
    
    fn is_feasible(&self, solution: &Self::Solution) -> bool {
        // Check connectivity requirements
        let connectivity = self.calculate_connectivity(solution);
        if connectivity < 0.8 { // At least 80% connectivity
            return false;
        }
        
        // Check latency constraints
        let latency = self.calculate_network_latency(solution);
        if latency > self.max_latency {
            return false;
        }
        
        // Check traffic capacity constraints
        for (conn_idx, &active) in solution.active_connections.iter().enumerate() {
            if active {
                let (u, v) = self.potential_connections[conn_idx];
                let traffic_demand = self.traffic_demands[u][v] + self.traffic_demands[v][u];
                let capacity = self.connection_capacities[conn_idx];
                
                if traffic_demand > capacity * 1.1 { // Allow 10% overutilization
                    return false;
                }
            }
        }
        
        true
    }
}

/// Network Topology Solution
#[derive(Debug, Clone)]
pub struct NetworkTopology {
    /// Which connections are active
    pub active_connections: Vec<bool>,
    /// Total network cost
    pub total_cost: f64,
    /// Network connectivity ratio
    pub connectivity: f64,
    /// Average network latency
    pub average_latency: f64,
    /// Network performance metrics
    pub performance_metrics: TelecomMetrics,
}

/// Telecommunications performance metrics
#[derive(Debug, Clone)]
pub struct TelecomMetrics {
    /// Network throughput (Gbps)
    pub throughput: f64,
    /// Packet loss rate
    pub packet_loss_rate: f64,
    /// Jitter (ms)
    pub jitter: f64,
    /// Network availability
    pub availability: f64,
    /// Mean time between failures (hours)
    pub mtbf: f64,
    /// Coverage area (km²)
    pub coverage_area: f64,
}

impl IndustrySolution for NetworkTopology {
    type Problem = NetworkTopologyOptimization;
    
    fn from_binary(problem: &Self::Problem, binary_solution: &[i8]) -> ApplicationResult<Self> {
        let num_connections = problem.potential_connections.len();
        let mut active_connections = vec![false; num_connections];
        
        // Decode binary solution
        for i in 0..num_connections.min(binary_solution.len()) {
            active_connections[i] = binary_solution[i] == 1;
        }
        
        let total_cost = problem.calculate_total_cost(&NetworkTopology {
            active_connections: active_connections.clone(),
            total_cost: 0.0,
            connectivity: 0.0,
            average_latency: 0.0,
            performance_metrics: TelecomMetrics {
                throughput: 0.0,
                packet_loss_rate: 0.0,
                jitter: 0.0,
                availability: 0.0,
                mtbf: 0.0,
                coverage_area: 0.0,
            },
        });
        
        let connectivity = problem.calculate_connectivity(&NetworkTopology {
            active_connections: active_connections.clone(),
            total_cost: 0.0,
            connectivity: 0.0,
            average_latency: 0.0,
            performance_metrics: TelecomMetrics {
                throughput: 0.0,
                packet_loss_rate: 0.0,
                jitter: 0.0,
                availability: 0.0,
                mtbf: 0.0,
                coverage_area: 0.0,
            },
        });
        
        let average_latency = problem.calculate_network_latency(&NetworkTopology {
            active_connections: active_connections.clone(),
            total_cost: 0.0,
            connectivity: 0.0,
            average_latency: 0.0,
            performance_metrics: TelecomMetrics {
                throughput: 0.0,
                packet_loss_rate: 0.0,
                jitter: 0.0,
                availability: 0.0,
                mtbf: 0.0,
                coverage_area: 0.0,
            },
        });
        
        // Calculate performance metrics
        let num_active = active_connections.iter().filter(|&&x| x).count();
        let performance_metrics = TelecomMetrics {
            throughput: num_active as f64 * 10.0, // 10 Gbps per connection
            packet_loss_rate: 0.001, // 0.1% packet loss
            jitter: 2.0, // 2ms jitter
            availability: 0.999, // 99.9% availability
            mtbf: 8760.0, // 1 year MTBF
            coverage_area: num_active as f64 * 100.0, // 100 km² per connection
        };
        
        Ok(NetworkTopology {
            active_connections,
            total_cost,
            connectivity,
            average_latency,
            performance_metrics,
        })
    }
    
    fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("type".to_string(), "Network Topology".to_string());
        summary.insert("total_cost".to_string(), format!("${:.2}", self.total_cost));
        summary.insert("connectivity".to_string(), format!("{:.1}%", self.connectivity * 100.0));
        summary.insert("average_latency".to_string(), format!("{:.1} ms", self.average_latency));
        summary.insert("throughput".to_string(), format!("{:.1} Gbps", self.performance_metrics.throughput));
        summary.insert("availability".to_string(), format!("{:.3}%", self.performance_metrics.availability * 100.0));
        
        let num_active = self.active_connections.iter().filter(|&&x| x).count();
        summary.insert("active_connections".to_string(), num_active.to_string());
        
        summary
    }
    
    fn metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("total_cost".to_string(), self.total_cost);
        metrics.insert("connectivity".to_string(), self.connectivity);
        metrics.insert("average_latency".to_string(), self.average_latency);
        metrics.insert("throughput".to_string(), self.performance_metrics.throughput);
        metrics.insert("packet_loss_rate".to_string(), self.performance_metrics.packet_loss_rate);
        metrics.insert("jitter".to_string(), self.performance_metrics.jitter);
        metrics.insert("availability".to_string(), self.performance_metrics.availability);
        metrics.insert("mtbf".to_string(), self.performance_metrics.mtbf);
        metrics.insert("coverage_area".to_string(), self.performance_metrics.coverage_area);
        
        let num_active = self.active_connections.iter().filter(|&&x| x).count();
        metrics.insert("active_connections".to_string(), num_active as f64);
        
        metrics
    }
    
    fn export_format(&self) -> ApplicationResult<String> {
        let mut output = String::new();
        output.push_str("# Network Topology Optimization Report\n\n");
        
        output.push_str("## Network Summary\n");
        output.push_str(&format!("Total Cost: ${:.2}\n", self.total_cost));
        output.push_str(&format!("Connectivity: {:.1}%\n", self.connectivity * 100.0));
        output.push_str(&format!("Average Latency: {:.1} ms\n", self.average_latency));
        
        output.push_str("\n## Performance Metrics\n");
        output.push_str(&format!("Throughput: {:.1} Gbps\n", self.performance_metrics.throughput));
        output.push_str(&format!("Packet Loss Rate: {:.3}%\n", self.performance_metrics.packet_loss_rate * 100.0));
        output.push_str(&format!("Jitter: {:.1} ms\n", self.performance_metrics.jitter));
        output.push_str(&format!("Availability: {:.3}%\n", self.performance_metrics.availability * 100.0));
        output.push_str(&format!("MTBF: {:.1} hours\n", self.performance_metrics.mtbf));
        output.push_str(&format!("Coverage Area: {:.1} km²\n", self.performance_metrics.coverage_area));
        
        output.push_str("\n## Active Connections\n");
        for (i, &active) in self.active_connections.iter().enumerate() {
            if active {
                output.push_str(&format!("Connection {}: Active\n", i + 1));
            }
        }
        
        Ok(output)
    }
}

/// Spectrum Allocation Optimization Problem
#[derive(Debug, Clone)]
pub struct SpectrumAllocation {
    /// Number of frequency bands
    pub num_bands: usize,
    /// Number of geographic regions
    pub num_regions: usize,
    /// Interference matrix between bands
    pub interference_matrix: Vec<Vec<f64>>,
    /// Service demand by region and band
    pub service_demands: Vec<Vec<f64>>,
    /// Band availability by region
    pub band_availability: Vec<Vec<bool>>,
    /// Regulatory constraints
    pub regulatory_constraints: Vec<IndustryConstraint>,
    /// Quality of Service requirements
    pub qos_requirements: Vec<f64>,
}

impl SpectrumAllocation {
    /// Create new spectrum allocation problem
    pub fn new(
        num_bands: usize,
        num_regions: usize,
        interference_matrix: Vec<Vec<f64>>,
        service_demands: Vec<Vec<f64>>,
    ) -> ApplicationResult<Self> {
        if interference_matrix.len() != num_bands {
            return Err(ApplicationError::InvalidConfiguration(
                "Interference matrix dimension mismatch".to_string()
            ));
        }
        
        Ok(Self {
            num_bands,
            num_regions,
            interference_matrix,
            service_demands,
            band_availability: vec![vec![true; num_bands]; num_regions],
            regulatory_constraints: Vec::new(),
            qos_requirements: vec![0.95; num_regions],
        })
    }
    
    /// Calculate interference for a given allocation
    pub fn calculate_interference(&self, allocation: &SpectrumSolution) -> f64 {
        let mut total_interference = 0.0;
        
        for region in 0..self.num_regions {
            for band1 in 0..self.num_bands {
                for band2 in 0..self.num_bands {
                    if band1 != band2 && 
                       allocation.band_assignments[region][band1] && 
                       allocation.band_assignments[region][band2] {
                        total_interference += self.interference_matrix[band1][band2];
                    }
                }
            }
        }
        
        total_interference
    }
}

/// Spectrum Allocation Solution
#[derive(Debug, Clone)]
pub struct SpectrumSolution {
    /// Band assignments [region][band] = assigned
    pub band_assignments: Vec<Vec<bool>>,
    /// Total interference
    pub total_interference: f64,
    /// Service coverage achieved
    pub service_coverage: Vec<f64>,
    /// Spectrum utilization efficiency
    pub spectrum_efficiency: f64,
}

/// Create benchmark telecommunications problems
pub fn create_benchmark_problems(size: usize) -> ApplicationResult<Vec<Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>>> {
    let mut problems = Vec::new();
    
    // Problem 1: Small network topology optimization
    let potential_connections = vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)];
    let connection_costs = vec![10.0, 15.0, 8.0, 12.0, 9.0];
    let traffic_demands = vec![
        vec![0.0, 5.0, 3.0, 2.0],
        vec![5.0, 0.0, 4.0, 6.0],
        vec![3.0, 4.0, 0.0, 3.0],
        vec![2.0, 6.0, 3.0, 0.0],
    ];
    
    let network_problem = NetworkTopologyOptimization::new(
        4, // nodes
        potential_connections,
        connection_costs,
        traffic_demands,
    )?;
    
    problems.push(Box::new(network_problem) as Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>);
    
    // Problem 2: Larger network for more complex scenarios
    if size >= 6 {
        let mut large_connections = Vec::new();
        let mut large_costs = Vec::new();
        
        // Create a more connected network
        for i in 0..size {
            for j in (i + 1)..size {
                large_connections.push((i, j));
                large_costs.push(5.0 + (i + j) as f64 * 1.5);
            }
        }
        
        let large_demands = vec![vec![2.0; size]; size];
        
        let large_network = NetworkTopologyOptimization::new(
            size,
            large_connections,
            large_costs,
            large_demands,
        )?;
        
        problems.push(Box::new(large_network) as Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>);
    }
    
    Ok(problems)
}

/// Solve network topology optimization using quantum annealing
pub fn solve_network_topology(
    problem: &NetworkTopologyOptimization,
    params: Option<AnnealingParams>,
) -> ApplicationResult<NetworkTopology> {
    // Convert to QUBO
    let qubo = problem.to_qubo()?;
    
    // Convert to Ising
    let ising = IsingModel::from_qubo(&qubo)?;
    
    // Set up annealing parameters
    let annealing_params = params.unwrap_or_else(|| {
        let mut p = AnnealingParams::default();
        p.num_sweeps = 25000;
        p.num_repetitions = 40;
        p.initial_temperature = 5.0;
        p.final_temperature = 0.001;
        p
    });
    
    // Solve with classical annealing
    let simulator = ClassicalAnnealingSimulator::new(annealing_params)
        .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;
    
    let result = simulator.solve(&ising)
        .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;
    
    // Convert solution back to network topology
    NetworkTopology::from_binary(problem, &result.best_spins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_topology_creation() {
        let connections = vec![(0, 1), (1, 2), (0, 2)];
        let costs = vec![10.0, 15.0, 12.0];
        let demands = vec![
            vec![0.0, 5.0, 3.0],
            vec![5.0, 0.0, 4.0],
            vec![3.0, 4.0, 0.0],
        ];
        
        let network = NetworkTopologyOptimization::new(3, connections, costs, demands).unwrap();
        assert_eq!(network.num_nodes, 3);
        assert_eq!(network.potential_connections.len(), 3);
    }
    
    #[test]
    fn test_network_connectivity_calculation() {
        let connections = vec![(0, 1), (1, 2)];
        let costs = vec![10.0, 15.0];
        let demands = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ];
        
        let network = NetworkTopologyOptimization::new(3, connections, costs, demands).unwrap();
        
        let topology = NetworkTopology {
            active_connections: vec![true, true],
            total_cost: 25.0,
            connectivity: 0.0,
            average_latency: 0.0,
            performance_metrics: TelecomMetrics {
                throughput: 0.0,
                packet_loss_rate: 0.0,
                jitter: 0.0,
                availability: 0.0,
                mtbf: 0.0,
                coverage_area: 0.0,
            },
        };
        
        let connectivity = network.calculate_connectivity(&topology);
        assert_eq!(connectivity, 1.0); // All nodes connected
    }
    
    #[test]
    fn test_network_cost_calculation() {
        let connections = vec![(0, 1), (1, 2)];
        let costs = vec![10.0, 15.0];
        let demands = vec![vec![0.0; 3]; 3];
        
        let network = NetworkTopologyOptimization::new(3, connections, costs, demands).unwrap();
        
        let topology = NetworkTopology {
            active_connections: vec![true, false],
            total_cost: 0.0,
            connectivity: 0.0,
            average_latency: 0.0,
            performance_metrics: TelecomMetrics {
                throughput: 0.0,
                packet_loss_rate: 0.0,
                jitter: 0.0,
                availability: 0.0,
                mtbf: 0.0,
                coverage_area: 0.0,
            },
        };
        
        let cost = network.calculate_total_cost(&topology);
        assert_eq!(cost, 10.0);
    }
    
    #[test]
    fn test_spectrum_allocation_creation() {
        let interference_matrix = vec![
            vec![0.0, 0.3, 0.1],
            vec![0.3, 0.0, 0.4],
            vec![0.1, 0.4, 0.0],
        ];
        let demands = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 1.0, 2.0],
        ];
        
        let spectrum = SpectrumAllocation::new(3, 2, interference_matrix, demands).unwrap();
        assert_eq!(spectrum.num_bands, 3);
        assert_eq!(spectrum.num_regions, 2);
    }
    
    #[test]
    fn test_network_validation() {
        let connections = vec![(0, 1), (1, 2)];
        let costs = vec![10.0, 15.0];
        let demands = vec![vec![0.0; 3]; 3];
        
        let network = NetworkTopologyOptimization::new(3, connections, costs, demands).unwrap();
        assert!(network.validate().is_ok());
        
        // Test invalid network (node index out of bounds)
        let invalid_connections = vec![(0, 1), (1, 5)]; // Node 5 doesn't exist
        let invalid_costs = vec![10.0, 15.0];
        let invalid_network = NetworkTopologyOptimization::new(3, invalid_connections, invalid_costs, demands);
        assert!(invalid_network.is_ok()); // Created successfully
        assert!(invalid_network.unwrap().validate().is_err()); // But validation fails
    }
    
    #[test]
    fn test_benchmark_problems() {
        let problems = create_benchmark_problems(5).unwrap();
        assert_eq!(problems.len(), 2);
        
        for problem in &problems {
            assert!(problem.validate().is_ok());
        }
    }
}