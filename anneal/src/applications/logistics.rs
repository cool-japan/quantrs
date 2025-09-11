//! Logistics Industry Optimization
//!
//! This module provides optimization solutions for the logistics industry,
//! including vehicle routing, supply chain optimization, warehouse management,
//! and delivery scheduling.

use super::{
    ApplicationError, ApplicationResult, IndustryConstraint, IndustryObjective, IndustrySolution,
    OptimizationProblem,
};
use crate::ising::IsingModel;
use crate::qubo::{QuboBuilder, QuboFormulation};
use crate::simulator::{AnnealingParams, ClassicalAnnealingSimulator};
use std::collections::HashMap;

/// Vehicle Routing Problem (VRP) with time windows and capacity constraints
#[derive(Debug, Clone)]
pub struct VehicleRoutingProblem {
    /// Number of vehicles available
    pub num_vehicles: usize,
    /// Vehicle capacities
    pub vehicle_capacities: Vec<f64>,
    /// Customer locations (including depot at index 0)
    pub locations: Vec<(f64, f64)>,
    /// Customer demands
    pub demands: Vec<f64>,
    /// Distance matrix between locations
    pub distance_matrix: Vec<Vec<f64>>,
    /// Time windows for each location (start_time, end_time)
    pub time_windows: Vec<(f64, f64)>,
    /// Service times at each location
    pub service_times: Vec<f64>,
    /// Vehicle operating costs per unit distance
    pub vehicle_costs: Vec<f64>,
    /// Maximum route duration
    pub max_route_duration: f64,
    /// Penalty for violating constraints
    pub constraint_penalty: f64,
}

impl VehicleRoutingProblem {
    /// Create a new vehicle routing problem
    pub fn new(
        num_vehicles: usize,
        vehicle_capacities: Vec<f64>,
        locations: Vec<(f64, f64)>,
        demands: Vec<f64>,
    ) -> ApplicationResult<Self> {
        if vehicle_capacities.len() != num_vehicles {
            return Err(ApplicationError::InvalidConfiguration(
                "Vehicle capacities must match number of vehicles".to_string(),
            ));
        }

        if locations.len() != demands.len() {
            return Err(ApplicationError::InvalidConfiguration(
                "Number of locations must match number of demands".to_string(),
            ));
        }

        let num_locations = locations.len();
        let distance_matrix = Self::calculate_distance_matrix(&locations);

        Ok(Self {
            num_vehicles,
            vehicle_capacities,
            locations,
            demands,
            distance_matrix,
            time_windows: vec![(0.0, 1440.0); num_locations], // Default: full day
            service_times: vec![10.0; num_locations],         // Default: 10 minutes
            vehicle_costs: vec![1.0; num_vehicles],           // Default: unit cost
            max_route_duration: 480.0,                        // Default: 8 hours
            constraint_penalty: 1000.0,
        })
    }

    /// Calculate Euclidean distance matrix
    fn calculate_distance_matrix(locations: &[(f64, f64)]) -> Vec<Vec<f64>> {
        let n = locations.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dx = locations[i].0 - locations[j].0;
                    let dy = locations[i].1 - locations[j].1;
                    matrix[i][j] = (dx * dx + dy * dy).sqrt();
                }
            }
        }

        matrix
    }

    /// Set time windows for locations
    pub fn set_time_windows(&mut self, time_windows: Vec<(f64, f64)>) -> ApplicationResult<()> {
        if time_windows.len() != self.locations.len() {
            return Err(ApplicationError::InvalidConfiguration(
                "Time windows must match number of locations".to_string(),
            ));
        }

        self.time_windows = time_windows;
        Ok(())
    }

    /// Set service times
    pub fn set_service_times(&mut self, service_times: Vec<f64>) -> ApplicationResult<()> {
        if service_times.len() != self.locations.len() {
            return Err(ApplicationError::InvalidConfiguration(
                "Service times must match number of locations".to_string(),
            ));
        }

        self.service_times = service_times;
        Ok(())
    }

    /// Calculate total route distance
    pub fn calculate_route_distance(&self, route: &[usize]) -> f64 {
        if route.len() < 2 {
            return 0.0;
        }

        route
            .windows(2)
            .map(|pair| self.distance_matrix[pair[0]][pair[1]])
            .sum()
    }

    /// Calculate total route duration
    pub fn calculate_route_duration(&self, route: &[usize]) -> f64 {
        if route.len() < 2 {
            return 0.0;
        }

        let travel_time: f64 = route
            .windows(2)
            .map(|pair| self.distance_matrix[pair[0]][pair[1]])
            .sum();

        let service_time: f64 = route.iter().map(|&loc| self.service_times[loc]).sum();

        travel_time + service_time
    }

    /// Check if route satisfies capacity constraint
    pub fn check_capacity_constraint(&self, route: &[usize], vehicle_idx: usize) -> bool {
        let total_demand: f64 = route.iter().map(|&loc| self.demands[loc]).sum();

        total_demand <= self.vehicle_capacities[vehicle_idx]
    }

    /// Check if route satisfies time window constraints
    pub fn check_time_windows(&self, route: &[usize]) -> bool {
        if route.len() < 2 {
            return true;
        }

        let mut current_time = 0.0;

        for i in 1..route.len() {
            let prev_loc = route[i - 1];
            let curr_loc = route[i];

            // Travel time
            current_time += self.distance_matrix[prev_loc][curr_loc];

            // Check if we arrive within time window
            let (earliest, latest) = self.time_windows[curr_loc];
            if current_time > latest {
                return false;
            }

            // Wait if we arrive early
            current_time = current_time.max(earliest);

            // Add service time
            current_time += self.service_times[curr_loc];
        }

        true
    }
}

impl OptimizationProblem for VehicleRoutingProblem {
    type Solution = VehicleRoutingSolution;
    type ObjectiveValue = f64;

    fn description(&self) -> String {
        format!(
            "Vehicle Routing Problem with {} vehicles and {} customers",
            self.num_vehicles,
            self.locations.len() - 1 // Subtract depot
        )
    }

    fn size_metrics(&self) -> HashMap<String, usize> {
        let mut metrics = HashMap::new();
        metrics.insert("num_vehicles".to_string(), self.num_vehicles);
        metrics.insert("num_customers".to_string(), self.locations.len() - 1);
        metrics.insert("num_locations".to_string(), self.locations.len());
        metrics
    }

    fn validate(&self) -> ApplicationResult<()> {
        if self.num_vehicles == 0 {
            return Err(ApplicationError::DataValidationError(
                "At least one vehicle required".to_string(),
            ));
        }

        if self.locations.len() < 2 {
            return Err(ApplicationError::DataValidationError(
                "At least depot and one customer required".to_string(),
            ));
        }

        // Check that depot has zero demand
        if self.demands[0] != 0.0 {
            return Err(ApplicationError::DataValidationError(
                "Depot must have zero demand".to_string(),
            ));
        }

        // Check positive vehicle capacities
        for (i, &capacity) in self.vehicle_capacities.iter().enumerate() {
            if capacity <= 0.0 {
                return Err(ApplicationError::DataValidationError(format!(
                    "Vehicle {} has non-positive capacity",
                    i
                )));
            }
        }

        Ok(())
    }

    fn to_qubo(&self) -> ApplicationResult<(crate::ising::QuboModel, HashMap<String, usize>)> {
        let num_customers = self.locations.len() - 1;
        let mut builder = QuboBuilder::new();

        // Binary variables: x[v][i][j] = 1 if vehicle v travels from i to j
        let mut var_counter = 0;
        let mut var_map = HashMap::new();
        let mut string_var_map = HashMap::new();

        for v in 0..self.num_vehicles {
            for i in 0..self.locations.len() {
                for j in 0..self.locations.len() {
                    if i != j {
                        let var_name = format!("x_{}_{}__{}", v, i, j);
                        var_map.insert((v, i, j), var_counter);
                        string_var_map.insert(var_name, var_counter);
                        var_counter += 1;
                    }
                }
            }
        }

        // Objective: minimize total distance
        for v in 0..self.num_vehicles {
            for i in 0..self.locations.len() {
                for j in 0..self.locations.len() {
                    if i != j {
                        let var_idx = var_map[&(v, i, j)];
                        let distance = self.distance_matrix[i][j];
                        let cost = distance * self.vehicle_costs[v];
                        builder.add_bias(var_idx, cost);
                    }
                }
            }
        }

        // Constraint: each customer visited exactly once
        for customer in 1..self.locations.len() {
            let mut constraint_vars = Vec::new();

            for v in 0..self.num_vehicles {
                for i in 0..self.locations.len() {
                    if i != customer {
                        if let Some(&var_idx) = var_map.get(&(v, i, customer)) {
                            constraint_vars.push(var_idx);
                        }
                    }
                }
            }

            // Add penalty for not visiting exactly once
            for &var1 in &constraint_vars {
                builder.add_bias(var1, -self.constraint_penalty);
                for &var2 in &constraint_vars {
                    if var1 != var2 {
                        builder.add_coupling(var1, var2, self.constraint_penalty);
                    }
                }
            }
        }

        // Constraint: flow conservation
        for v in 0..self.num_vehicles {
            for loc in 0..self.locations.len() {
                let mut in_vars = Vec::new();
                let mut out_vars = Vec::new();

                // Incoming edges
                for i in 0..self.locations.len() {
                    if i != loc {
                        if let Some(&var_idx) = var_map.get(&(v, i, loc)) {
                            in_vars.push(var_idx);
                        }
                    }
                }

                // Outgoing edges
                for j in 0..self.locations.len() {
                    if j != loc {
                        if let Some(&var_idx) = var_map.get(&(v, loc, j)) {
                            out_vars.push(var_idx);
                        }
                    }
                }

                // Flow conservation: in = out
                for &in_var in &in_vars {
                    for &out_var in &out_vars {
                        builder.add_coupling(in_var, out_var, -self.constraint_penalty);
                    }
                }
            }
        }

        Ok((builder.build(), string_var_map))
    }

    fn evaluate_solution(
        &self,
        solution: &Self::Solution,
    ) -> ApplicationResult<Self::ObjectiveValue> {
        let mut total_cost = 0.0;

        for (vehicle_idx, route) in solution.routes.iter().enumerate() {
            if route.len() > 1 {
                let distance = self.calculate_route_distance(route);
                total_cost += distance * self.vehicle_costs[vehicle_idx];
            }
        }

        Ok(total_cost)
    }

    fn is_feasible(&self, solution: &Self::Solution) -> bool {
        // Check that all customers are visited exactly once
        let mut visited = vec![false; self.locations.len()];
        visited[0] = true; // Depot doesn't need to be visited by customers

        for route in &solution.routes {
            for &location in route {
                if visited[location] {
                    return false; // Location visited more than once
                }
                visited[location] = true;
            }
        }

        // Check all customers are visited
        if !visited[1..].iter().all(|&v| v) {
            return false;
        }

        // Check capacity and time constraints for each route
        for (vehicle_idx, route) in solution.routes.iter().enumerate() {
            if !self.check_capacity_constraint(route, vehicle_idx) {
                return false;
            }

            if !self.check_time_windows(route) {
                return false;
            }

            if self.calculate_route_duration(route) > self.max_route_duration {
                return false;
            }
        }

        true
    }
}

/// Binary wrapper for vehicle routing optimization that works with `Vec<i8>` solutions
#[derive(Debug, Clone)]
pub struct BinaryVehicleRoutingProblem {
    inner: VehicleRoutingProblem,
}

impl BinaryVehicleRoutingProblem {
    pub fn new(inner: VehicleRoutingProblem) -> Self {
        Self { inner }
    }
}

impl OptimizationProblem for BinaryVehicleRoutingProblem {
    type Solution = Vec<i8>;
    type ObjectiveValue = f64;

    fn description(&self) -> String {
        self.inner.description()
    }

    fn size_metrics(&self) -> HashMap<String, usize> {
        self.inner.size_metrics()
    }

    fn validate(&self) -> ApplicationResult<()> {
        self.inner.validate()
    }

    fn to_qubo(&self) -> ApplicationResult<(crate::ising::QuboModel, HashMap<String, usize>)> {
        self.inner.to_qubo()
    }

    fn evaluate_solution(
        &self,
        solution: &Self::Solution,
    ) -> ApplicationResult<Self::ObjectiveValue> {
        // Simplified VRP evaluation: treat binary solution as edge selection
        let num_locations = self.inner.locations.len();
        let num_vehicles = self.inner.num_vehicles;

        // Extract selected edges from binary solution
        let mut total_cost = 0.0;
        let mut vehicle_loads = vec![0.0; num_vehicles];

        let edge_vars_per_vehicle = num_locations * num_locations;

        for vehicle in 0..num_vehicles {
            let vehicle_offset = vehicle * edge_vars_per_vehicle;

            for i in 0..num_locations {
                for j in 0..num_locations {
                    if i != j {
                        let var_idx = vehicle_offset + i * num_locations + j;
                        if var_idx < solution.len() && solution[var_idx] == 1 {
                            // This edge is selected
                            total_cost += self.inner.distance_matrix[i][j]
                                * self.inner.vehicle_costs[vehicle];

                            // Add demand if going to a customer location (not depot)
                            if j > 0 && j < self.inner.demands.len() {
                                vehicle_loads[vehicle] += self.inner.demands[j];
                            }
                        }
                    }
                }
            }

            // Capacity penalty
            if vehicle_loads[vehicle] > self.inner.vehicle_capacities[vehicle] {
                total_cost +=
                    1000.0 * (vehicle_loads[vehicle] - self.inner.vehicle_capacities[vehicle]);
            }
        }

        // Minimize cost (negative for maximization algorithms)
        Ok(-total_cost)
    }

    fn is_feasible(&self, solution: &Self::Solution) -> bool {
        // Basic feasibility: at least some edges must be selected
        solution.iter().any(|&x| x == 1)
    }
}

/// Solution for Vehicle Routing Problem
#[derive(Debug, Clone)]
pub struct VehicleRoutingSolution {
    /// Routes for each vehicle (sequence of location indices)
    pub routes: Vec<Vec<usize>>,
    /// Total distance traveled
    pub total_distance: f64,
    /// Total cost
    pub total_cost: f64,
    /// Route statistics
    pub route_stats: Vec<RouteStatistics>,
}

/// Statistics for individual routes
#[derive(Debug, Clone)]
pub struct RouteStatistics {
    /// Vehicle index
    pub vehicle_id: usize,
    /// Route distance
    pub distance: f64,
    /// Route duration
    pub duration: f64,
    /// Capacity utilization
    pub capacity_utilization: f64,
    /// Number of customers served
    pub customers_served: usize,
    /// Time window violations
    pub time_violations: usize,
}

impl IndustrySolution for VehicleRoutingSolution {
    type Problem = VehicleRoutingProblem;

    fn from_binary(problem: &Self::Problem, binary_solution: &[i8]) -> ApplicationResult<Self> {
        // Simplified binary solution interpretation
        // In practice, this would involve complex routing construction
        let num_customers = problem.locations.len() - 1;
        let mut routes = vec![Vec::new(); problem.num_vehicles];

        // Simple greedy assignment for demonstration
        for customer in 1..problem.locations.len() {
            let vehicle_idx = customer % problem.num_vehicles;
            routes[vehicle_idx].push(customer);
        }

        // Add depot at start and end of each route
        for route in &mut routes {
            if !route.is_empty() {
                route.insert(0, 0); // Start at depot
                route.push(0); // Return to depot
            }
        }

        let mut total_distance = 0.0;
        let mut total_cost = 0.0;
        let mut route_stats = Vec::new();

        for (vehicle_idx, route) in routes.iter().enumerate() {
            let distance = problem.calculate_route_distance(route);
            let duration = problem.calculate_route_duration(route);
            let cost = distance * problem.vehicle_costs[vehicle_idx];

            let demand: f64 = route.iter().map(|&loc| problem.demands[loc]).sum();
            let capacity_utilization = demand / problem.vehicle_capacities[vehicle_idx];

            let customers_served = route.len().saturating_sub(2); // Exclude depot visits

            route_stats.push(RouteStatistics {
                vehicle_id: vehicle_idx,
                distance,
                duration,
                capacity_utilization,
                customers_served,
                time_violations: 0, // Simplified
            });

            total_distance += distance;
            total_cost += cost;
        }

        Ok(Self {
            routes,
            total_distance,
            total_cost,
            route_stats,
        })
    }

    fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("type".to_string(), "Vehicle Routing".to_string());
        summary.insert("num_routes".to_string(), self.routes.len().to_string());
        summary.insert(
            "total_distance".to_string(),
            format!("{:.2}", self.total_distance),
        );
        summary.insert("total_cost".to_string(), format!("{:.2}", self.total_cost));

        let active_vehicles = self.routes.iter().filter(|route| route.len() > 2).count();
        summary.insert("active_vehicles".to_string(), active_vehicles.to_string());

        let total_customers: usize = self
            .route_stats
            .iter()
            .map(|stats| stats.customers_served)
            .sum();
        summary.insert("customers_served".to_string(), total_customers.to_string());

        let avg_utilization = self
            .route_stats
            .iter()
            .map(|stats| stats.capacity_utilization)
            .sum::<f64>()
            / self.route_stats.len() as f64;
        summary.insert(
            "avg_capacity_utilization".to_string(),
            format!("{:.1}%", avg_utilization * 100.0),
        );

        summary
    }

    fn metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("total_distance".to_string(), self.total_distance);
        metrics.insert("total_cost".to_string(), self.total_cost);

        if !self.route_stats.is_empty() {
            let avg_distance = self
                .route_stats
                .iter()
                .map(|stats| stats.distance)
                .sum::<f64>()
                / self.route_stats.len() as f64;
            metrics.insert("avg_route_distance".to_string(), avg_distance);

            let max_distance = self
                .route_stats
                .iter()
                .map(|stats| stats.distance)
                .fold(0.0, f64::max);
            metrics.insert("max_route_distance".to_string(), max_distance);

            let avg_utilization = self
                .route_stats
                .iter()
                .map(|stats| stats.capacity_utilization)
                .sum::<f64>()
                / self.route_stats.len() as f64;
            metrics.insert("avg_capacity_utilization".to_string(), avg_utilization);

            let route_balance = self
                .route_stats
                .iter()
                .map(|stats| stats.customers_served as f64)
                .collect::<Vec<_>>();
            let mean_customers = route_balance.iter().sum::<f64>() / route_balance.len() as f64;
            let variance = route_balance
                .iter()
                .map(|&x| (x - mean_customers).powi(2))
                .sum::<f64>()
                / route_balance.len() as f64;
            metrics.insert("route_balance_variance".to_string(), variance);
        }

        metrics
    }

    fn export_format(&self) -> ApplicationResult<String> {
        let mut output = String::new();
        output.push_str("# Vehicle Routing Solution\n\n");

        output.push_str("## Solution Summary\n");
        output.push_str(&format!("Total Distance: {:.2} km\n", self.total_distance));
        output.push_str(&format!("Total Cost: ${:.2}\n", self.total_cost));
        output.push_str(&format!(
            "Active Vehicles: {}\n",
            self.routes.iter().filter(|route| route.len() > 2).count()
        ));

        output.push_str("\n## Route Details\n");
        for (i, route) in self.routes.iter().enumerate() {
            if route.len() > 2 {
                // Skip empty routes
                let stats = &self.route_stats[i];
                output.push_str(&format!("\n### Vehicle {} Route\n", i + 1));
                output.push_str(&format!("Sequence: {:?}\n", route));
                output.push_str(&format!("Distance: {:.2} km\n", stats.distance));
                output.push_str(&format!("Duration: {:.1} minutes\n", stats.duration));
                output.push_str(&format!("Customers: {}\n", stats.customers_served));
                output.push_str(&format!(
                    "Capacity Utilization: {:.1}%\n",
                    stats.capacity_utilization * 100.0
                ));
            }
        }

        output.push_str("\n## Performance Metrics\n");
        for (key, value) in self.metrics() {
            output.push_str(&format!("{}: {:.3}\n", key, value));
        }

        Ok(output)
    }
}

/// Supply Chain Optimization Problem
#[derive(Debug, Clone)]
pub struct SupplyChainOptimization {
    /// Number of suppliers
    pub num_suppliers: usize,
    /// Number of distribution centers
    pub num_distribution_centers: usize,
    /// Number of retail locations
    pub num_retailers: usize,
    /// Supplier capacities
    pub supplier_capacities: Vec<f64>,
    /// Distribution center capacities
    pub dc_capacities: Vec<f64>,
    /// Retailer demands
    pub retailer_demands: Vec<f64>,
    /// Transportation costs (supplier -> DC)
    pub supplier_dc_costs: Vec<Vec<f64>>,
    /// Transportation costs (DC -> retailer)
    pub dc_retailer_costs: Vec<Vec<f64>>,
    /// Fixed costs for opening DCs
    pub dc_fixed_costs: Vec<f64>,
    /// Service level requirements
    pub service_levels: Vec<f64>,
}

impl SupplyChainOptimization {
    /// Create a new supply chain optimization problem
    pub fn new(
        num_suppliers: usize,
        num_distribution_centers: usize,
        num_retailers: usize,
        supplier_capacities: Vec<f64>,
        dc_capacities: Vec<f64>,
        retailer_demands: Vec<f64>,
    ) -> ApplicationResult<Self> {
        // Initialize cost matrices with default values
        let supplier_dc_costs = vec![vec![1.0; num_distribution_centers]; num_suppliers];
        let dc_retailer_costs = vec![vec![1.0; num_retailers]; num_distribution_centers];
        let dc_fixed_costs = vec![100.0; num_distribution_centers];
        let service_levels = vec![0.95; num_retailers];

        Ok(Self {
            num_suppliers,
            num_distribution_centers,
            num_retailers,
            supplier_capacities,
            dc_capacities,
            retailer_demands,
            supplier_dc_costs,
            dc_retailer_costs,
            dc_fixed_costs,
            service_levels,
        })
    }

    /// Set transportation costs
    pub fn set_transportation_costs(
        &mut self,
        supplier_dc_costs: Vec<Vec<f64>>,
        dc_retailer_costs: Vec<Vec<f64>>,
    ) -> ApplicationResult<()> {
        if supplier_dc_costs.len() != self.num_suppliers {
            return Err(ApplicationError::InvalidConfiguration(
                "Supplier-DC cost matrix size mismatch".to_string(),
            ));
        }

        self.supplier_dc_costs = supplier_dc_costs;
        self.dc_retailer_costs = dc_retailer_costs;
        Ok(())
    }

    /// Calculate total transportation cost
    pub fn calculate_transportation_cost(&self, solution: &SupplyChainSolution) -> f64 {
        let mut total_cost = 0.0;

        // Supplier to DC costs
        for (s, supplier_flows) in solution.supplier_dc_flows.iter().enumerate() {
            for (dc, &flow) in supplier_flows.iter().enumerate() {
                total_cost += flow * self.supplier_dc_costs[s][dc];
            }
        }

        // DC to retailer costs
        for (dc, dc_flows) in solution.dc_retailer_flows.iter().enumerate() {
            for (r, &flow) in dc_flows.iter().enumerate() {
                total_cost += flow * self.dc_retailer_costs[dc][r];
            }
        }

        total_cost
    }
}

/// Supply Chain Solution
#[derive(Debug, Clone)]
pub struct SupplyChainSolution {
    /// Flow from suppliers to distribution centers
    pub supplier_dc_flows: Vec<Vec<f64>>,
    /// Flow from distribution centers to retailers
    pub dc_retailer_flows: Vec<Vec<f64>>,
    /// Which distribution centers are open
    pub dc_open: Vec<bool>,
    /// Total cost
    pub total_cost: f64,
    /// Service level achieved
    pub service_level_achieved: Vec<f64>,
}

/// Create benchmark logistics problems
pub fn create_benchmark_problems(
    size: usize,
) -> ApplicationResult<Vec<Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>>>
{
    let mut problems = Vec::new();

    // Problem 1: Small VRP
    let locations = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.5), (1.5, 2.0), (0.5, 1.5)];
    let demands = vec![0.0, 10.0, 15.0, 8.0, 12.0];
    let capacities = vec![30.0, 25.0];

    let vrp = VehicleRoutingProblem::new(2, capacities, locations, demands)?;
    problems.push(Box::new(BinaryVehicleRoutingProblem::new(vrp))
        as Box<
            dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>,
        >);

    // Problem 2: Medium VRP with larger instance
    if size >= 8 {
        let mut large_locations = vec![(0.0, 0.0)]; // Depot
        let mut large_demands = vec![0.0]; // Depot demand

        for i in 0..size {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / size as f64;
            let radius = 1.0 + (i as f64 * 0.1);
            large_locations.push((radius * angle.cos(), radius * angle.sin()));
            large_demands.push(5.0 + (i as f64 * 2.0));
        }

        let large_capacities = vec![50.0; 3];
        let large_vrp =
            VehicleRoutingProblem::new(3, large_capacities, large_locations, large_demands)?;
        problems.push(Box::new(BinaryVehicleRoutingProblem::new(large_vrp))
            as Box<
                dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>,
            >);
    }

    Ok(problems)
}

/// Solve VRP using quantum annealing
pub fn solve_vehicle_routing(
    problem: &VehicleRoutingProblem,
    params: Option<AnnealingParams>,
) -> ApplicationResult<VehicleRoutingSolution> {
    // Convert to QUBO
    let (qubo, _var_map) = problem.to_qubo()?;

    // Convert to Ising
    let ising = IsingModel::from_qubo(&qubo);

    // Set up annealing parameters
    let annealing_params = params.unwrap_or_else(|| {
        let mut p = AnnealingParams::default();
        p.num_sweeps = 15000;
        p.num_repetitions = 25;
        p.initial_temperature = 3.0;
        p.final_temperature = 0.005;
        p
    });

    // Solve with classical annealing
    let simulator = ClassicalAnnealingSimulator::new(annealing_params)
        .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;

    let result = simulator
        .solve(&ising)
        .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;

    // Convert solution back to routing solution
    VehicleRoutingSolution::from_binary(problem, &result.best_spins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vrp_creation() {
        let locations = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)];
        let demands = vec![0.0, 10.0, 15.0];
        let capacities = vec![30.0];

        let vrp = VehicleRoutingProblem::new(1, capacities, locations, demands).unwrap();
        assert_eq!(vrp.num_vehicles, 1);
        assert_eq!(vrp.locations.len(), 3);
    }

    #[test]
    fn test_distance_calculation() {
        let locations = vec![(0.0, 0.0), (3.0, 4.0)]; // 3-4-5 triangle
        let demands = vec![0.0, 10.0];
        let capacities = vec![20.0];

        let vrp = VehicleRoutingProblem::new(1, capacities, locations, demands).unwrap();
        assert_eq!(vrp.distance_matrix[0][1], 5.0);
        assert_eq!(vrp.distance_matrix[1][0], 5.0);
    }

    #[test]
    fn test_capacity_constraint() {
        let locations = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)];
        let demands = vec![0.0, 10.0, 15.0];
        let capacities = vec![20.0];

        let vrp = VehicleRoutingProblem::new(1, capacities, locations, demands).unwrap();

        let route1 = vec![0, 1, 0]; // Demand: 10
        let route2 = vec![0, 1, 2, 0]; // Demand: 25

        assert!(vrp.check_capacity_constraint(&route1, 0));
        assert!(!vrp.check_capacity_constraint(&route2, 0));
    }

    #[test]
    fn test_time_windows() {
        let locations = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)];
        let demands = vec![0.0, 10.0, 15.0];
        let capacities = vec![30.0];

        let mut vrp = VehicleRoutingProblem::new(1, capacities, locations, demands).unwrap();

        // Set tight time windows
        vrp.set_time_windows(vec![(0.0, 100.0), (10.0, 20.0), (50.0, 60.0)])
            .unwrap();
        vrp.set_service_times(vec![0.0, 5.0, 5.0]).unwrap();

        let route = vec![0, 1, 2, 0];
        assert!(vrp.check_time_windows(&route));
    }

    #[test]
    fn test_supply_chain_creation() {
        let supply_chain = SupplyChainOptimization::new(
            2,                            // suppliers
            3,                            // DCs
            4,                            // retailers
            vec![100.0, 150.0],           // supplier capacities
            vec![80.0, 90.0, 70.0],       // DC capacities
            vec![20.0, 25.0, 30.0, 15.0], // retailer demands
        )
        .unwrap();

        assert_eq!(supply_chain.num_suppliers, 2);
        assert_eq!(supply_chain.num_distribution_centers, 3);
        assert_eq!(supply_chain.num_retailers, 4);
    }

    #[test]
    fn test_benchmark_problems() {
        let problems = create_benchmark_problems(8).unwrap();
        assert_eq!(problems.len(), 2);

        for problem in &problems {
            assert!(problem.validate().is_ok());
        }
    }
}
