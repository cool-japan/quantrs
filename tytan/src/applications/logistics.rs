//! Logistics applications: Route optimization toolkit.
//!
//! This module provides quantum optimization tools for logistics applications
//! including vehicle routing, supply chain optimization, and warehouse management.

// Sampler types available for logistics applications
#![allow(dead_code)]

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::prelude::*;
use std::collections::{HashMap, HashSet};

/// Vehicle Routing Problem (VRP) optimizer
pub struct VehicleRoutingOptimizer {
    /// Distance matrix between locations
    distance_matrix: Array2<f64>,
    /// Vehicle capacity
    vehicle_capacity: f64,
    /// Demand at each location
    demands: Array1<f64>,
    /// Time windows for each location
    time_windows: Option<Vec<TimeWindow>>,
    /// Number of vehicles
    num_vehicles: usize,
    /// Depot location
    depot: usize,
    /// Problem variant
    variant: VRPVariant,
}

#[derive(Debug, Clone)]
pub struct TimeWindow {
    /// Earliest arrival time
    start: f64,
    /// Latest arrival time
    end: f64,
    /// Service time at location
    service_time: f64,
}

#[derive(Debug, Clone)]
pub enum VRPVariant {
    /// Capacitated VRP
    CVRP,
    /// VRP with Time Windows
    VRPTW,
    /// Multi-Depot VRP
    MDVRP { depots: Vec<usize> },
    /// Pickup and Delivery
    VRPPD {
        pickups: Vec<usize>,
        deliveries: Vec<usize>,
    },
    /// VRP with Backhauls
    VRPB { backhaul_customers: Vec<usize> },
    /// Heterogeneous Fleet VRP
    HVRP { vehicle_types: Vec<VehicleType> },
}

#[derive(Debug, Clone)]
pub struct VehicleType {
    /// Vehicle capacity
    capacity: f64,
    /// Fixed cost
    fixed_cost: f64,
    /// Cost per distance
    distance_cost: f64,
    /// Maximum distance
    max_distance: Option<f64>,
    /// Speed factor
    speed_factor: f64,
}

impl VehicleRoutingOptimizer {
    /// Create new VRP optimizer
    pub fn new(
        distance_matrix: Array2<f64>,
        vehicle_capacity: f64,
        demands: Array1<f64>,
        num_vehicles: usize,
    ) -> Result<Self, String> {
        if distance_matrix.shape()[0] != distance_matrix.shape()[1] {
            return Err("Distance matrix must be square".to_string());
        }

        if distance_matrix.shape()[0] != demands.len() {
            return Err("Distance matrix and demands size mismatch".to_string());
        }

        Ok(Self {
            distance_matrix,
            vehicle_capacity,
            demands,
            time_windows: None,
            num_vehicles,
            depot: 0,
            variant: VRPVariant::CVRP,
        })
    }

    /// Set problem variant
    pub fn with_variant(mut self, variant: VRPVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Set time windows
    pub fn with_time_windows(mut self, time_windows: Vec<TimeWindow>) -> Self {
        self.time_windows = Some(time_windows);
        self.variant = VRPVariant::VRPTW;
        self
    }

    /// Build QUBO formulation
    pub fn build_qubo(&self) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        let n_locations = self.distance_matrix.shape()[0];
        let _n_customers = n_locations - 1; // Excluding depot

        // Variables: x_{v,i,j} = 1 if vehicle v travels from i to j
        let n_vars = self.num_vehicles * n_locations * n_locations;

        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();

        // Create variable mapping
        let mut var_idx = 0;
        for v in 0..self.num_vehicles {
            for i in 0..n_locations {
                for j in 0..n_locations {
                    let var_name = format!("x_{v}_{i}_{j}");
                    var_map.insert(var_name, var_idx);
                    var_idx += 1;
                }
            }
        }

        // Add objective: minimize total distance
        self.add_distance_objective(&mut qubo, &var_map)?;

        // Add constraints based on variant
        match &self.variant {
            VRPVariant::CVRP => {
                self.add_cvrp_constraints(&mut qubo, &var_map)?;
            }
            VRPVariant::VRPTW => {
                self.add_cvrp_constraints(&mut qubo, &var_map)?;
                self.add_time_window_constraints(&mut qubo, &var_map)?;
            }
            VRPVariant::MDVRP { depots } => {
                self.add_mdvrp_constraints(&mut qubo, &var_map, depots)?;
            }
            _ => {
                self.add_cvrp_constraints(&mut qubo, &var_map)?;
            }
        }

        Ok((qubo, var_map))
    }

    /// Add distance objective
    fn add_distance_objective(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        for v in 0..self.num_vehicles {
            for i in 0..self.distance_matrix.shape()[0] {
                for j in 0..self.distance_matrix.shape()[1] {
                    if i != j {
                        let var_name = format!("x_{v}_{i}_{j}");
                        if let Some(&var_idx) = var_map.get(&var_name) {
                            qubo[[var_idx, var_idx]] += self.distance_matrix[[i, j]];
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Add CVRP constraints
    fn add_cvrp_constraints(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        let penalty = 1000.0;
        let n_locations = self.distance_matrix.shape()[0];

        // 1. Each customer visited exactly once
        for j in 1..n_locations {
            // Skip depot
            // (sum_{v,i} x_{v,i,j} - 1)^2
            for v1 in 0..self.num_vehicles {
                for i1 in 0..n_locations {
                    if i1 != j {
                        let var1 = format!("x_{v1}_{i1}_{j}");
                        if let Some(&idx1) = var_map.get(&var1) {
                            // Linear term
                            qubo[[idx1, idx1]] -= 2.0 * penalty;

                            // Quadratic terms
                            for v2 in 0..self.num_vehicles {
                                for i2 in 0..n_locations {
                                    if i2 != j {
                                        let var2 = format!("x_{v2}_{i2}_{j}");
                                        if let Some(&idx2) = var_map.get(&var2) {
                                            qubo[[idx1, idx2]] += penalty;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Flow conservation
        for v in 0..self.num_vehicles {
            for i in 0..n_locations {
                // sum_j x_{v,i,j} = sum_j x_{v,j,i}
                for j1 in 0..n_locations {
                    if j1 != i {
                        let var_out = format!("x_{v}_{i}_{j1}");
                        if let Some(&idx_out) = var_map.get(&var_out) {
                            for j2 in 0..n_locations {
                                if j2 != i {
                                    let var_in = format!("x_{v}_{j2}_{i}");
                                    if let Some(&idx_in) = var_map.get(&var_in) {
                                        qubo[[idx_out, idx_in]] -= penalty;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. Capacity constraints (simplified)
        self.add_capacity_constraints(qubo, var_map, penalty)?;

        // 4. Each vehicle starts and ends at depot
        for v in 0..self.num_vehicles {
            // Start at depot
            for j in 1..n_locations {
                let var = format!("x_{}_{}_{}", v, 0, j);
                if let Some(&idx) = var_map.get(&var) {
                    qubo[[idx, idx]] -= penalty * 0.1; // Encourage depot start
                }
            }
        }

        Ok(())
    }

    /// Add capacity constraints
    fn add_capacity_constraints(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        penalty: f64,
    ) -> Result<(), String> {
        // Simplified: penalize routes that exceed capacity
        let n_locations = self.distance_matrix.shape()[0];

        for v in 0..self.num_vehicles {
            // Approximate capacity usage
            let route_demand = 0.0;

            for i in 0..n_locations {
                for j in 1..n_locations {
                    // Skip depot
                    let var = format!("x_{v}_{i}_{j}");
                    if let Some(&idx) = var_map.get(&var) {
                        // Penalize if demand exceeds capacity
                        if route_demand + self.demands[j] > self.vehicle_capacity {
                            qubo[[idx, idx]] += penalty * 10.0;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Add time window constraints
    fn add_time_window_constraints(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        if let Some(time_windows) = &self.time_windows {
            let penalty = 1000.0;
            let n_locations = self.distance_matrix.shape()[0];

            // Simplified: penalize violations of time windows
            for v in 0..self.num_vehicles {
                for i in 0..n_locations {
                    for j in 0..n_locations {
                        if i != j {
                            let var = format!("x_{v}_{i}_{j}");
                            if let Some(&idx) = var_map.get(&var) {
                                // Travel time
                                let travel_time = self.distance_matrix[[i, j]];

                                // Check if arrival at j violates time window
                                if j < time_windows.len() {
                                    let earliest_arrival = if i < time_windows.len() {
                                        time_windows[i].start
                                            + time_windows[i].service_time
                                            + travel_time
                                    } else {
                                        travel_time
                                    };

                                    if earliest_arrival > time_windows[j].end {
                                        qubo[[idx, idx]] += penalty * 5.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Add multi-depot constraints
    fn add_mdvrp_constraints(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        depots: &[usize],
    ) -> Result<(), String> {
        // Each vehicle assigned to exactly one depot
        let penalty = 1000.0;

        for v in 0..self.num_vehicles {
            // Vehicle must start from one of the depots
            for &depot in depots {
                for j in 0..self.distance_matrix.shape()[0] {
                    if !depots.contains(&j) {
                        let var = format!("x_{v}_{depot}_{j}");
                        if let Some(&idx) = var_map.get(&var) {
                            qubo[[idx, idx]] -= penalty * 0.1; // Encourage depot start
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Decode solution to routes
    pub fn decode_solution(&self, solution: &HashMap<String, bool>) -> Vec<Route> {
        let mut routes = Vec::new();
        let n_locations = self.distance_matrix.shape()[0];

        for v in 0..self.num_vehicles {
            let mut route = Route {
                vehicle_id: v,
                path: vec![self.depot],
                total_distance: 0.0,
                total_demand: 0.0,
                arrival_times: vec![0.0],
            };

            let mut current = self.depot;
            let mut visited = HashSet::new();
            visited.insert(self.depot);

            // Build route by following edges
            loop {
                let mut next_location = None;

                for j in 0..n_locations {
                    if !visited.contains(&j) {
                        let var = format!("x_{v}_{current}_{j}");
                        if *solution.get(&var).unwrap_or(&false) {
                            next_location = Some(j);
                            break;
                        }
                    }
                }

                if let Some(next) = next_location {
                    route.path.push(next);
                    route.total_distance += self.distance_matrix[[current, next]];
                    route.total_demand += self.demands[next];

                    let arrival_time = route.arrival_times.last().copied().unwrap_or(0.0)
                        + self.distance_matrix[[current, next]];
                    route.arrival_times.push(arrival_time);

                    visited.insert(next);
                    current = next;
                } else {
                    break;
                }
            }

            // Return to depot if route is not empty
            if route.path.len() > 1 {
                route.path.push(self.depot);
                route.total_distance += self.distance_matrix[[current, self.depot]];
                route.arrival_times.push(
                    route.arrival_times.last().copied().unwrap_or(0.0)
                        + self.distance_matrix[[current, self.depot]],
                );
                routes.push(route);
            }
        }

        routes
    }

    /// Validate solution
    pub fn validate_solution(&self, routes: &[Route]) -> ValidationResult {
        let mut violations = Vec::new();
        let mut visited_customers = HashSet::new();

        // Check each route
        for route in routes {
            // Capacity check
            if route.total_demand > self.vehicle_capacity {
                violations.push(ConstraintViolation::CapacityExceeded {
                    vehicle: route.vehicle_id,
                    demand: route.total_demand,
                    capacity: self.vehicle_capacity,
                });
            }

            // Time window checks
            if let Some(time_windows) = &self.time_windows {
                for (i, &loc) in route.path.iter().enumerate() {
                    if loc < time_windows.len()
                        && i < route.arrival_times.len()
                        && route.arrival_times[i] > time_windows[loc].end
                    {
                        violations.push(ConstraintViolation::TimeWindowViolation {
                            location: loc,
                            arrival: route.arrival_times[i],
                            window_end: time_windows[loc].end,
                        });
                    }
                }
            }

            // Track visited customers
            for &loc in &route.path {
                if loc != self.depot {
                    visited_customers.insert(loc);
                }
            }
        }

        // Check all customers visited
        let n_customers = self.distance_matrix.shape()[0] - 1;
        if visited_customers.len() < n_customers {
            violations.push(ConstraintViolation::CustomersNotVisited {
                missing: n_customers - visited_customers.len(),
            });
        }

        ValidationResult {
            is_valid: violations.is_empty(),
            violations,
            total_distance: routes.iter().map(|r| r.total_distance).sum(),
            num_vehicles_used: routes.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Route {
    pub vehicle_id: usize,
    pub path: Vec<usize>,
    pub total_distance: f64,
    pub total_demand: f64,
    pub arrival_times: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum ConstraintViolation {
    CapacityExceeded {
        vehicle: usize,
        demand: f64,
        capacity: f64,
    },
    TimeWindowViolation {
        location: usize,
        arrival: f64,
        window_end: f64,
    },
    CustomersNotVisited {
        missing: usize,
    },
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub violations: Vec<ConstraintViolation>,
    pub total_distance: f64,
    pub num_vehicles_used: usize,
}

/// Simplified optimization problem trait for binary VRP
pub trait OptimizationProblem {
    type Solution;

    /// Evaluate the objective function
    fn evaluate(&self, solution: &Self::Solution) -> f64;
}

/// Vehicle Routing Problem for optimization
pub struct VehicleRoutingProblem {
    pub optimizer: VehicleRoutingOptimizer,
}

impl VehicleRoutingProblem {
    pub const fn new(optimizer: VehicleRoutingOptimizer) -> Self {
        Self { optimizer }
    }

    /// Evaluate floating point solution
    pub fn evaluate_continuous(&self, x: &Array1<f64>) -> f64 {
        // Convert continuous solution to binary decisions
        let n_locations = self.optimizer.distance_matrix.shape()[0];
        let n_vars = self.optimizer.num_vehicles * n_locations * n_locations;

        if x.len() != n_vars {
            return f64::INFINITY; // Invalid solution
        }

        let mut energy = 0.0;

        // Calculate distance objective
        let mut var_idx = 0;
        for _v in 0..self.optimizer.num_vehicles {
            for i in 0..n_locations {
                for j in 0..n_locations {
                    if i != j {
                        let decision = if x[var_idx] > 0.5 { 1.0 } else { 0.0 };
                        energy += decision * self.optimizer.distance_matrix[[i, j]];
                    }
                    var_idx += 1;
                }
            }
        }

        // Add constraint penalties
        energy += self.calculate_constraint_penalties(x);

        energy
    }

    fn calculate_constraint_penalties(&self, x: &Array1<f64>) -> f64 {
        let penalty = 1000.0;
        let mut total_penalty = 0.0;

        let n_locations = self.optimizer.distance_matrix.shape()[0];

        // Customer visit constraint: each customer visited exactly once
        for j in 1..n_locations {
            // Skip depot
            let mut visits = 0.0;
            let mut var_idx = 0;

            for _v in 0..self.optimizer.num_vehicles {
                for i in 0..n_locations {
                    if i != j {
                        let decision = if x[var_idx + i * n_locations + j] > 0.5 {
                            1.0
                        } else {
                            0.0
                        };
                        visits += decision;
                    }
                }
                var_idx += n_locations * n_locations;
            }

            total_penalty += penalty * (visits - 1.0f64).abs();
        }

        total_penalty
    }
}

/// Binary Vehicle Routing Problem wrapper
pub struct BinaryVehicleRoutingProblem {
    inner: VehicleRoutingProblem,
}

impl BinaryVehicleRoutingProblem {
    pub const fn new(optimizer: VehicleRoutingOptimizer) -> Self {
        Self {
            inner: VehicleRoutingProblem::new(optimizer),
        }
    }

    /// Get the number of variables needed for binary representation
    pub fn num_variables(&self) -> usize {
        let n_locations = self.inner.optimizer.distance_matrix.shape()[0];
        self.inner.optimizer.num_vehicles * n_locations * n_locations
    }

    /// Evaluate binary solution directly
    pub fn evaluate_binary(&self, solution: &[i8]) -> f64 {
        // Convert i8 binary solution to f64 array for the inner problem
        let x: Array1<f64> = solution.iter().map(|&b| b as f64).collect();
        self.inner.evaluate_continuous(&x)
    }

    /// Create random binary solution
    pub fn random_solution(&self) -> Vec<i8> {
        let mut rng = thread_rng();
        let n_vars = self.num_variables();
        (0..n_vars)
            .map(|_| i8::from(rng.gen::<f64>() > 0.8))
            .collect()
    }

    /// Convert binary solution to routes
    pub fn decode_binary_solution(&self, solution: &[i8]) -> Vec<Route> {
        let mut bool_solution = HashMap::new();
        let n_locations = self.inner.optimizer.distance_matrix.shape()[0];

        let mut var_idx = 0;
        for v in 0..self.inner.optimizer.num_vehicles {
            for i in 0..n_locations {
                for j in 0..n_locations {
                    let var_name = format!("x_{v}_{i}_{j}");
                    bool_solution.insert(var_name, solution[var_idx] == 1);
                    var_idx += 1;
                }
            }
        }

        self.inner.optimizer.decode_solution(&bool_solution)
    }
}

impl OptimizationProblem for BinaryVehicleRoutingProblem {
    type Solution = Vec<i8>;

    fn evaluate(&self, solution: &Self::Solution) -> f64 {
        self.evaluate_binary(solution)
    }
}

/// Create benchmark problems for testing
pub fn create_benchmark_problems() -> Vec<BinaryVehicleRoutingProblem> {
    let mut problems = Vec::new();

    // Small VRP problem
    let small_distances = Array2::from_shape_vec(
        (4, 4),
        vec![
            0.0, 10.0, 15.0, 20.0, 10.0, 0.0, 25.0, 30.0, 15.0, 25.0, 0.0, 35.0, 20.0, 30.0, 35.0,
            0.0,
        ],
    )
    .expect("Small benchmark distance matrix has valid shape");

    let small_demands = Array1::from_vec(vec![0.0, 10.0, 15.0, 20.0]);

    let small_optimizer =
        VehicleRoutingOptimizer::new(small_distances.clone(), 50.0, small_demands.clone(), 2)
            .expect("Small benchmark VRP has valid configuration");

    problems.push(BinaryVehicleRoutingProblem::new(small_optimizer));

    // Medium VRP problem
    let medium_distances = Array2::from_shape_vec(
        (6, 6),
        vec![
            0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 10.0, 0.0, 25.0, 30.0, 35.0, 40.0, 15.0, 25.0, 0.0,
            35.0, 40.0, 45.0, 20.0, 30.0, 35.0, 0.0, 45.0, 50.0, 25.0, 35.0, 40.0, 45.0, 0.0, 55.0,
            30.0, 40.0, 45.0, 50.0, 55.0, 0.0,
        ],
    )
    .expect("Medium benchmark distance matrix has valid shape");

    let medium_demands = Array1::from_vec(vec![0.0, 12.0, 18.0, 22.0, 16.0, 14.0]);

    let medium_optimizer = VehicleRoutingOptimizer::new(medium_distances, 60.0, medium_demands, 3)
        .expect("Medium benchmark VRP has valid configuration");

    problems.push(BinaryVehicleRoutingProblem::new(medium_optimizer));

    // Capacitated VRP with time windows
    let cvrptw_optimizer = VehicleRoutingOptimizer::new(
        small_distances,
        40.0, // Tighter capacity
        small_demands,
        2,
    )
    .expect("CVRPTW benchmark VRP has valid configuration")
    .with_time_windows(vec![
        TimeWindow {
            start: 0.0,
            end: 100.0,
            service_time: 5.0,
        }, // Depot
        TimeWindow {
            start: 10.0,
            end: 50.0,
            service_time: 10.0,
        }, // Customer 1
        TimeWindow {
            start: 20.0,
            end: 60.0,
            service_time: 8.0,
        }, // Customer 2
        TimeWindow {
            start: 30.0,
            end: 80.0,
            service_time: 12.0,
        }, // Customer 3
    ]);

    problems.push(BinaryVehicleRoutingProblem::new(cvrptw_optimizer));

    problems
}

/// Traveling Salesman Problem (TSP) optimizer
pub struct TSPOptimizer {
    /// Distance matrix
    distance_matrix: Array2<f64>,
    /// Problem variant
    variant: TSPVariant,
    /// Subtour elimination method
    subtour_method: SubtourElimination,
}

#[derive(Debug, Clone)]
pub enum TSPVariant {
    /// Standard TSP
    Standard,
    /// Asymmetric TSP
    ATSP,
    /// TSP with time windows
    TSPTW { time_windows: Vec<TimeWindow> },
    /// Multiple TSP
    MTSP { num_salesmen: usize },
    /// Prize-collecting TSP
    PCTSP { prizes: Vec<f64>, min_prize: f64 },
}

#[derive(Debug, Clone)]
pub enum SubtourElimination {
    /// MTZ constraints
    MillerTuckerZemlin,
    /// DFJ constraints
    DantzigFulkersonJohnson,
    /// Flow-based
    FlowBased,
    /// Lazy constraints
    Lazy,
}

impl TSPOptimizer {
    /// Create new TSP optimizer
    pub fn new(distance_matrix: Array2<f64>) -> Result<Self, String> {
        if distance_matrix.shape()[0] != distance_matrix.shape()[1] {
            return Err("Distance matrix must be square".to_string());
        }

        Ok(Self {
            distance_matrix,
            variant: TSPVariant::Standard,
            subtour_method: SubtourElimination::MillerTuckerZemlin,
        })
    }

    /// Set variant
    pub fn with_variant(mut self, variant: TSPVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Build QUBO formulation
    pub fn build_qubo(&self) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        let n = self.distance_matrix.shape()[0];

        match &self.variant {
            TSPVariant::Standard => self.build_standard_tsp_qubo(n),
            TSPVariant::MTSP { num_salesmen } => self.build_mtsp_qubo(n, *num_salesmen),
            _ => self.build_standard_tsp_qubo(n),
        }
    }

    /// Build standard TSP QUBO
    fn build_standard_tsp_qubo(
        &self,
        n: usize,
    ) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        // Variables: x_{i,t} = 1 if city i is visited at time t
        let n_vars = n * n;
        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();

        // Create variable mapping
        for i in 0..n {
            for t in 0..n {
                let var_name = format!("x_{i}_{t}");
                var_map.insert(var_name, i * n + t);
            }
        }

        let penalty = 1000.0;

        // Objective: minimize distance
        for t in 0..n - 1 {
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let var1 = i * n + t;
                        let var2 = j * n + (t + 1);
                        qubo[[var1, var2]] += self.distance_matrix[[i, j]];
                    }
                }
            }
        }

        // Constraint 1: Each city visited exactly once
        for i in 0..n {
            // (sum_t x_{i,t} - 1)^2
            for t1 in 0..n {
                let idx1 = i * n + t1;
                qubo[[idx1, idx1]] -= 2.0 * penalty;

                for t2 in 0..n {
                    let idx2 = i * n + t2;
                    qubo[[idx1, idx2]] += penalty;
                }
            }
        }

        // Constraint 2: Each time slot has exactly one city
        for t in 0..n {
            // (sum_i x_{i,t} - 1)^2
            for i1 in 0..n {
                let idx1 = i1 * n + t;
                qubo[[idx1, idx1]] -= 2.0 * penalty;

                for i2 in 0..n {
                    let idx2 = i2 * n + t;
                    qubo[[idx1, idx2]] += penalty;
                }
            }
        }

        Ok((qubo, var_map))
    }

    /// Build Multiple TSP QUBO
    fn build_mtsp_qubo(
        &self,
        n: usize,
        num_salesmen: usize,
    ) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        // Variables: x_{s,i,t} = 1 if salesman s visits city i at time t
        let n_vars = num_salesmen * n * n;
        let qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();

        // Create variable mapping
        for s in 0..num_salesmen {
            for i in 0..n {
                for t in 0..n {
                    let var_name = format!("x_{s}_{i}_{t}");
                    var_map.insert(var_name, s * n * n + i * n + t);
                }
            }
        }

        // Add objective and constraints
        // Similar to standard TSP but for each salesman

        Ok((qubo, var_map))
    }

    /// Decode TSP solution
    pub fn decode_solution(&self, solution: &HashMap<String, bool>) -> Vec<usize> {
        let n = self.distance_matrix.shape()[0];
        let mut tour = vec![0; n];

        for i in 0..n {
            for t in 0..n {
                let var_name = format!("x_{i}_{t}");
                if *solution.get(&var_name).unwrap_or(&false) {
                    tour[t] = i;
                }
            }
        }

        tour
    }
}

/// Supply chain optimizer
pub struct SupplyChainOptimizer {
    /// Network structure
    network: SupplyChainNetwork,
    /// Optimization objectives
    objectives: Vec<SupplyChainObjective>,
    /// Constraints
    constraints: SupplyChainConstraints,
    /// Time horizon
    time_horizon: usize,
}

#[derive(Debug, Clone)]
pub struct SupplyChainNetwork {
    /// Suppliers
    pub suppliers: Vec<Supplier>,
    /// Warehouses
    pub warehouses: Vec<Warehouse>,
    /// Distribution centers
    pub distribution_centers: Vec<DistributionCenter>,
    /// Customers
    pub customers: Vec<Customer>,
    /// Transportation links
    pub links: Vec<TransportLink>,
}

#[derive(Debug, Clone)]
pub struct Supplier {
    pub id: usize,
    pub capacity: f64,
    pub cost_per_unit: f64,
    pub lead_time: usize,
    pub reliability: f64,
}

#[derive(Debug, Clone)]
pub struct Warehouse {
    pub id: usize,
    pub capacity: f64,
    pub holding_cost: f64,
    pub fixed_cost: f64,
    pub location: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct DistributionCenter {
    pub id: usize,
    pub capacity: f64,
    pub processing_cost: f64,
    pub location: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct Customer {
    pub id: usize,
    pub demand: Array1<f64>, // Demand over time
    pub priority: f64,
    pub location: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct TransportLink {
    pub from_type: NodeType,
    pub from_id: usize,
    pub to_type: NodeType,
    pub to_id: usize,
    pub capacity: f64,
    pub cost_per_unit: f64,
    pub lead_time: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeType {
    Supplier,
    Warehouse,
    DistributionCenter,
    Customer,
}

#[derive(Debug, Clone)]
pub enum SupplyChainObjective {
    /// Minimize total cost
    MinimizeCost,
    /// Minimize delivery time
    MinimizeDeliveryTime,
    /// Maximize service level
    MaximizeServiceLevel,
    /// Minimize inventory
    MinimizeInventory,
    /// Balance workload
    BalanceWorkload,
}

#[derive(Debug, Clone)]
pub struct SupplyChainConstraints {
    /// Capacity constraints
    pub enforce_capacity: bool,
    /// Service level requirements
    pub min_service_level: f64,
    /// Maximum lead time
    pub max_lead_time: Option<usize>,
    /// Safety stock requirements
    pub safety_stock: HashMap<usize, f64>,
    /// Budget constraint
    pub max_budget: Option<f64>,
}

impl SupplyChainOptimizer {
    /// Create new supply chain optimizer
    pub fn new(network: SupplyChainNetwork, time_horizon: usize) -> Self {
        Self {
            network,
            objectives: vec![SupplyChainObjective::MinimizeCost],
            constraints: SupplyChainConstraints {
                enforce_capacity: true,
                min_service_level: 0.95,
                max_lead_time: None,
                safety_stock: HashMap::new(),
                max_budget: None,
            },
            time_horizon,
        }
    }

    /// Add objective
    pub fn add_objective(mut self, objective: SupplyChainObjective) -> Self {
        self.objectives.push(objective);
        self
    }

    /// Set constraints
    pub fn with_constraints(mut self, constraints: SupplyChainConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Build QUBO formulation
    pub fn build_qubo(&self) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        // Variables:
        // - x_{s,w,t}: flow from supplier s to warehouse w at time t
        // - y_{w,d,t}: flow from warehouse w to DC d at time t
        // - z_{d,c,t}: flow from DC d to customer c at time t
        // - I_{w,t}: inventory at warehouse w at time t

        let mut var_map = HashMap::new();
        let mut var_idx = 0;

        // Create variables for flows
        for t in 0..self.time_horizon {
            // Supplier to warehouse
            for s in &self.network.suppliers {
                for w in &self.network.warehouses {
                    let var_name = format!("x_{}_{}_{}", s.id, w.id, t);
                    var_map.insert(var_name, var_idx);
                    var_idx += 1;
                }
            }

            // Warehouse to DC
            for w in &self.network.warehouses {
                for d in &self.network.distribution_centers {
                    let var_name = format!("y_{}_{}_{}", w.id, d.id, t);
                    var_map.insert(var_name, var_idx);
                    var_idx += 1;
                }
            }

            // DC to customer
            for d in &self.network.distribution_centers {
                for c in &self.network.customers {
                    let var_name = format!("z_{}_{}_{}", d.id, c.id, t);
                    var_map.insert(var_name, var_idx);
                    var_idx += 1;
                }
            }

            // Inventory variables
            for w in &self.network.warehouses {
                let var_name = format!("I_{}_{}", w.id, t);
                var_map.insert(var_name, var_idx);
                var_idx += 1;
            }
        }

        let n_vars = var_idx;
        let mut qubo = Array2::zeros((n_vars, n_vars));

        // Add objectives
        for objective in &self.objectives {
            match objective {
                SupplyChainObjective::MinimizeCost => {
                    self.add_cost_objective(&mut qubo, &var_map)?;
                }
                SupplyChainObjective::MinimizeInventory => {
                    self.add_inventory_objective(&mut qubo, &var_map)?;
                }
                _ => {}
            }
        }

        // Add constraints
        self.add_flow_conservation_constraints(&mut qubo, &var_map)?;
        self.add_capacity_constraints_sc(&mut qubo, &var_map)?;
        self.add_demand_constraints(&mut qubo, &var_map)?;

        Ok((qubo, var_map))
    }

    /// Add cost objective
    fn add_cost_objective(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        // Transportation costs
        for link in &self.network.links {
            for t in 0..self.time_horizon {
                let var_name = match (&link.from_type, &link.to_type) {
                    (NodeType::Supplier, NodeType::Warehouse) => {
                        format!("x_{}_{}_{}", link.from_id, link.to_id, t)
                    }
                    (NodeType::Warehouse, NodeType::DistributionCenter) => {
                        format!("y_{}_{}_{}", link.from_id, link.to_id, t)
                    }
                    (NodeType::DistributionCenter, NodeType::Customer) => {
                        format!("z_{}_{}_{}", link.from_id, link.to_id, t)
                    }
                    _ => continue,
                };

                if let Some(&idx) = var_map.get(&var_name) {
                    qubo[[idx, idx]] += link.cost_per_unit;
                }
            }
        }

        // Holding costs
        for w in &self.network.warehouses {
            for t in 0..self.time_horizon {
                let var_name = format!("I_{}_{}", w.id, t);
                if let Some(&idx) = var_map.get(&var_name) {
                    qubo[[idx, idx]] += w.holding_cost;
                }
            }
        }

        Ok(())
    }

    /// Add inventory objective
    fn add_inventory_objective(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        for w in &self.network.warehouses {
            for t in 0..self.time_horizon {
                let var_name = format!("I_{}_{}", w.id, t);
                if let Some(&idx) = var_map.get(&var_name) {
                    qubo[[idx, idx]] += 1.0; // Minimize inventory
                }
            }
        }

        Ok(())
    }

    /// Add flow conservation constraints
    fn add_flow_conservation_constraints(
        &self,
        _qubo: &mut Array2<f64>,
        _var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        let _penalty = 1000.0;

        // Warehouse flow conservation
        for _w in &self.network.warehouses {
            for _t in 1..self.time_horizon {
                // I_{w,t} = I_{w,t-1} + sum_s x_{s,w,t} - sum_d y_{w,d,t}

                // This is a complex constraint, simplified here
                // Penalize imbalanced flows
            }
        }

        Ok(())
    }

    /// Add capacity constraints
    fn add_capacity_constraints_sc(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        let penalty = 1000.0;

        // Supplier capacity
        for s in &self.network.suppliers {
            for t in 0..self.time_horizon {
                // sum_w x_{s,w,t} <= capacity_s
                // Simplified: penalize excessive flow
                for w in &self.network.warehouses {
                    let var_name = format!("x_{}_{}_{}", s.id, w.id, t);
                    if let Some(&idx) = var_map.get(&var_name) {
                        if s.capacity > 0.0 {
                            qubo[[idx, idx]] += penalty / s.capacity;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Add demand constraints
    fn add_demand_constraints(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        let penalty = 1000.0;

        // Customer demand satisfaction
        for c in &self.network.customers {
            for t in 0..self.time_horizon.min(c.demand.len()) {
                // sum_d z_{d,c,t} = demand_{c,t}
                // Simplified: encourage meeting demand
                for d in &self.network.distribution_centers {
                    let var_name = format!("z_{}_{}_{}", d.id, c.id, t);
                    if let Some(&idx) = var_map.get(&var_name) {
                        qubo[[idx, idx]] -= penalty * c.demand[t] * c.priority;
                    }
                }
            }
        }

        Ok(())
    }
}

/// Warehouse optimization
pub struct WarehouseOptimizer {
    /// Warehouse layout
    layout: WarehouseLayout,
    /// Storage policies
    policies: StoragePolicies,
    /// Order data
    orders: Vec<Order>,
    /// Optimization goals
    goals: WarehouseGoals,
}

#[derive(Debug, Clone)]
pub struct WarehouseLayout {
    /// Grid dimensions
    pub rows: usize,
    pub cols: usize,
    pub levels: usize,
    /// Storage locations
    pub locations: Vec<StorageLocation>,
    /// Picking stations
    pub picking_stations: Vec<(usize, usize)>,
    /// Distance function
    pub distance_type: DistanceType,
}

#[derive(Debug, Clone)]
pub struct StorageLocation {
    pub id: usize,
    pub position: (usize, usize, usize), // row, col, level
    pub capacity: f64,
    pub item_type: Option<String>,
    pub accessibility: f64,
}

#[derive(Debug, Clone)]
pub enum DistanceType {
    Manhattan,
    Euclidean,
    Rectilinear,
    Custom,
}

#[derive(Debug, Clone)]
pub struct StoragePolicies {
    /// Storage assignment policy
    pub assignment: AssignmentPolicy,
    /// Replenishment policy
    pub replenishment: ReplenishmentPolicy,
    /// Picking policy
    pub picking: PickingPolicy,
}

#[derive(Debug, Clone)]
pub enum AssignmentPolicy {
    /// Random storage
    Random,
    /// ABC classification
    ABC {
        a_locations: Vec<usize>,
        b_locations: Vec<usize>,
        c_locations: Vec<usize>,
    },
    /// Dedicated storage
    Dedicated,
    /// Class-based storage
    ClassBased,
}

#[derive(Debug, Clone)]
pub enum ReplenishmentPolicy {
    /// Fixed order quantity
    FixedQuantity { quantity: f64 },
    /// Reorder point
    ReorderPoint { level: f64 },
    /// Periodic review
    Periodic { interval: usize },
}

#[derive(Debug, Clone)]
pub enum PickingPolicy {
    /// Single order picking
    Single,
    /// Batch picking
    Batch { size: usize },
    /// Zone picking
    Zone { zones: Vec<Vec<usize>> },
    /// Wave picking
    Wave { interval: usize },
}

#[derive(Debug, Clone)]
pub struct Order {
    pub id: usize,
    pub items: Vec<OrderItem>,
    pub priority: f64,
    pub due_time: f64,
}

#[derive(Debug, Clone)]
pub struct OrderItem {
    pub sku: String,
    pub quantity: usize,
    pub location: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct WarehouseGoals {
    /// Minimize picking distance
    pub minimize_distance: bool,
    /// Minimize order completion time
    pub minimize_time: bool,
    /// Balance workload
    pub balance_workload: bool,
    /// Maximize space utilization
    pub maximize_utilization: bool,
}

impl WarehouseOptimizer {
    /// Create new warehouse optimizer
    pub const fn new(
        layout: WarehouseLayout,
        policies: StoragePolicies,
        orders: Vec<Order>,
    ) -> Self {
        Self {
            layout,
            policies,
            orders,
            goals: WarehouseGoals {
                minimize_distance: true,
                minimize_time: false,
                balance_workload: false,
                maximize_utilization: false,
            },
        }
    }

    /// Optimize order picking
    pub fn optimize_picking(&self) -> Result<PickingPlan, String> {
        match &self.policies.picking {
            PickingPolicy::Batch { size } => self.optimize_batch_picking(*size),
            _ => self.optimize_single_picking(),
        }
    }

    /// Optimize batch picking
    fn optimize_batch_picking(&self, batch_size: usize) -> Result<PickingPlan, String> {
        let mut batches = Vec::new();
        let mut remaining_orders = self.orders.clone();

        while !remaining_orders.is_empty() {
            let batch_orders: Vec<_> = remaining_orders
                .drain(..batch_size.min(remaining_orders.len()))
                .collect();

            let route = self.optimize_picking_route(&batch_orders)?;

            let estimated_time = self.estimate_picking_time(&route);
            batches.push(Batch {
                orders: batch_orders,
                route,
                estimated_time,
            });
        }

        let total_distance = batches.iter().map(|b| b.route.total_distance).sum();
        let total_time = batches.iter().map(|b| b.estimated_time).sum();

        Ok(PickingPlan {
            batches,
            total_distance,
            total_time,
        })
    }

    /// Optimize single order picking
    fn optimize_single_picking(&self) -> Result<PickingPlan, String> {
        let mut batches = Vec::new();

        for order in &self.orders {
            let route = self.optimize_picking_route(&[order.clone()])?;
            let estimated_time = self.estimate_picking_time(&route);

            batches.push(Batch {
                orders: vec![order.clone()],
                route,
                estimated_time,
            });
        }

        let total_distance = batches.iter().map(|b| b.route.total_distance).sum();
        let total_time = batches.iter().map(|b| b.estimated_time).sum();

        Ok(PickingPlan {
            batches,
            total_distance,
            total_time,
        })
    }

    /// Optimize picking route for orders
    fn optimize_picking_route(&self, orders: &[Order]) -> Result<PickingRoute, String> {
        // Collect all pick locations
        let mut pick_locations = Vec::new();

        for order in orders {
            for item in &order.items {
                if let Some(loc) = item.location {
                    pick_locations.push(loc);
                }
            }
        }

        // Remove duplicates
        pick_locations.sort_unstable();
        pick_locations.dedup();

        // Build distance matrix including picking station
        let n = pick_locations.len() + 1; // +1 for picking station
        let mut distances = Array2::zeros((n, n));

        // Distance from picking station to locations
        let station = self.layout.picking_stations[0]; // Use first station
        for (i, &loc) in pick_locations.iter().enumerate() {
            let loc_pos = self.layout.locations[loc].position;
            distances[[0, i + 1]] = self.calculate_distance((station.0, station.1, 0), loc_pos);
            distances[[i + 1, 0]] = distances[[0, i + 1]];
        }

        // Distance between locations
        for (i, &loc1) in pick_locations.iter().enumerate() {
            for (j, &loc2) in pick_locations.iter().enumerate() {
                if i != j {
                    let pos1 = self.layout.locations[loc1].position;
                    let pos2 = self.layout.locations[loc2].position;
                    distances[[i + 1, j + 1]] = self.calculate_distance(pos1, pos2);
                }
            }
        }

        // Solve TSP
        let tsp = TSPOptimizer::new(distances)?;
        let (_qubo, _var_map) = tsp.build_qubo()?;

        // Simplified: return S-shaped route
        let sequence = (0..pick_locations.len()).collect();
        Ok(PickingRoute {
            locations: pick_locations,
            sequence,
            total_distance: 0.0, // Would calculate from solution
        })
    }

    /// Calculate distance between positions
    fn calculate_distance(&self, pos1: (usize, usize, usize), pos2: (usize, usize, usize)) -> f64 {
        match self.layout.distance_type {
            DistanceType::Manhattan => {
                ((pos1.0 as i32 - pos2.0 as i32).abs()
                    + (pos1.1 as i32 - pos2.1 as i32).abs()
                    + (pos1.2 as i32 - pos2.2 as i32).abs()) as f64
            }
            DistanceType::Euclidean => (pos1.2 as f64 - pos2.2 as f64)
                .mul_add(
                    pos1.2 as f64 - pos2.2 as f64,
                    (pos1.1 as f64 - pos2.1 as f64).mul_add(
                        pos1.1 as f64 - pos2.1 as f64,
                        (pos1.0 as f64 - pos2.0 as f64).powi(2),
                    ),
                )
                .sqrt(),
            _ => 0.0,
        }
    }

    /// Estimate picking time for route
    fn estimate_picking_time(&self, route: &PickingRoute) -> f64 {
        // Simplified: distance * speed + pick time
        let travel_time = route.total_distance / 1.0; // 1 unit/time
        let pick_time = route.locations.len() as f64 * 10.0; // 10 time units per pick

        travel_time + pick_time
    }
}

#[derive(Debug, Clone)]
pub struct PickingPlan {
    pub batches: Vec<Batch>,
    pub total_distance: f64,
    pub total_time: f64,
}

#[derive(Debug, Clone)]
pub struct Batch {
    pub orders: Vec<Order>,
    pub route: PickingRoute,
    pub estimated_time: f64,
}

#[derive(Debug, Clone)]
pub struct PickingRoute {
    pub locations: Vec<usize>,
    pub sequence: Vec<usize>,
    pub total_distance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vrp_optimizer() {
        let distances = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 10.0, 15.0, 20.0, 10.0, 0.0, 25.0, 30.0, 15.0, 25.0, 0.0, 35.0, 20.0, 30.0,
                35.0, 0.0,
            ],
        )
        .expect("Test distance matrix has valid shape");

        let demands = Array1::from_vec(vec![0.0, 10.0, 15.0, 20.0]);

        let optimizer = VehicleRoutingOptimizer::new(distances, 50.0, demands, 2)
            .expect("Test VRP optimizer should be created with valid inputs");

        let (_qubo, var_map) = optimizer
            .build_qubo()
            .expect("VRP QUBO should build successfully");
        assert!(!var_map.is_empty());
    }

    #[test]
    fn test_tsp_optimizer() {
        let distances = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 10.0, 15.0, 20.0, 10.0, 0.0, 25.0, 30.0, 15.0, 25.0, 0.0, 35.0, 20.0, 30.0,
                35.0, 0.0,
            ],
        )
        .expect("Test distance matrix has valid shape");

        let optimizer =
            TSPOptimizer::new(distances).expect("TSP optimizer should be created with valid input");
        let (_qubo, var_map) = optimizer
            .build_qubo()
            .expect("TSP QUBO should build successfully");

        assert_eq!(var_map.len(), 16); // 4 cities * 4 time slots
    }

    #[test]
    fn test_supply_chain() {
        let network = SupplyChainNetwork {
            suppliers: vec![Supplier {
                id: 0,
                capacity: 100.0,
                cost_per_unit: 10.0,
                lead_time: 2,
                reliability: 0.95,
            }],
            warehouses: vec![Warehouse {
                id: 0,
                capacity: 200.0,
                holding_cost: 1.0,
                fixed_cost: 1000.0,
                location: (0.0, 0.0),
            }],
            distribution_centers: vec![DistributionCenter {
                id: 0,
                capacity: 150.0,
                processing_cost: 2.0,
                location: (10.0, 10.0),
            }],
            customers: vec![Customer {
                id: 0,
                demand: Array1::from_vec(vec![20.0, 25.0, 30.0]),
                priority: 1.0,
                location: (20.0, 20.0),
            }],
            links: vec![],
        };

        let optimizer = SupplyChainOptimizer::new(network, 3);
        let (_qubo, var_map) = optimizer
            .build_qubo()
            .expect("Supply chain QUBO should build successfully");

        assert!(!var_map.is_empty());
    }

    #[test]
    fn test_binary_vrp_wrapper() {
        let distances = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 10.0, 15.0, 20.0, 10.0, 0.0, 25.0, 30.0, 15.0, 25.0, 0.0, 35.0, 20.0, 30.0,
                35.0, 0.0,
            ],
        )
        .expect("Test distance matrix has valid shape");

        let demands = Array1::from_vec(vec![0.0, 10.0, 15.0, 20.0]);

        let optimizer = VehicleRoutingOptimizer::new(distances, 50.0, demands, 2)
            .expect("Test VRP optimizer should be created with valid inputs");

        let binary_vrp = BinaryVehicleRoutingProblem::new(optimizer);

        // Test number of variables
        assert_eq!(binary_vrp.num_variables(), 32); // 2 vehicles * 4 locations * 4 locations

        // Test random solution generation
        let solution = binary_vrp.random_solution();
        assert_eq!(solution.len(), 32);

        // Test solution evaluation
        let energy = binary_vrp.evaluate_binary(&solution);
        assert!(energy.is_finite());

        // Test solution decoding
        let routes = binary_vrp.decode_binary_solution(&solution);
        assert!(routes.len() <= 2); // Maximum 2 vehicles
    }

    #[test]
    fn test_create_benchmark_problems() {
        let problems = create_benchmark_problems();
        assert_eq!(problems.len(), 3); // Small, medium, and CVRPTW problems

        // Test each problem
        for (i, problem) in problems.iter().enumerate() {
            let solution = problem.random_solution();
            let energy = problem.evaluate_binary(&solution);
            assert!(energy.is_finite(), "Problem {i} should have finite energy");

            let routes = problem.decode_binary_solution(&solution);
            // Each problem should be able to decode solutions
            assert!(
                routes.len() <= 3,
                "Problem {i} should have at most 3 routes"
            );
        }
    }
}
