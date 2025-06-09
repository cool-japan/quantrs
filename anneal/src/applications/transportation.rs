//! Transportation and Logistics Optimization
//!
//! This module provides optimization solutions for the transportation industry,
//! including traffic flow optimization, vehicle routing problems, logistics planning,
//! smart city infrastructure optimization, and autonomous vehicle coordination.

use std::collections::HashMap;
use crate::qubo::{QuboFormulation, QuboBuilder};
use crate::ising::IsingModel;
use crate::simulator::{AnnealingParams, ClassicalAnnealingSimulator};
use super::{ApplicationError, ApplicationResult, OptimizationProblem, IndustrySolution, IndustryConstraint, IndustryObjective};

/// Vehicle Routing Problem with Time Windows
#[derive(Debug, Clone)]
pub struct VehicleRoutingProblem {
    /// Number of vehicles
    pub num_vehicles: usize,
    /// Number of customer locations
    pub num_customers: usize,
    /// Distance matrix between locations (includes depot at index 0)
    pub distance_matrix: Vec<Vec<f64>>,
    /// Customer demands
    pub customer_demands: Vec<f64>,
    /// Vehicle capacities
    pub vehicle_capacities: Vec<f64>,
    /// Time windows for each customer [start_time, end_time]
    pub time_windows: Vec<(f64, f64)>,
    /// Service times at each location
    pub service_times: Vec<f64>,
    /// Maximum route duration
    pub max_route_duration: f64,
    /// Vehicle operating costs per unit distance
    pub vehicle_costs: Vec<f64>,
    /// Priority weights for customers
    pub customer_priorities: Vec<f64>,
}

impl VehicleRoutingProblem {
    /// Create a new vehicle routing problem
    pub fn new(
        num_vehicles: usize,
        num_customers: usize,
        distance_matrix: Vec<Vec<f64>>,
        customer_demands: Vec<f64>,
        vehicle_capacities: Vec<f64>,
    ) -> ApplicationResult<Self> {
        let total_locations = num_customers + 1; // +1 for depot
        
        if distance_matrix.len() != total_locations {
            return Err(ApplicationError::InvalidConfiguration(
                "Distance matrix dimension mismatch".to_string()
            ));
        }
        
        if customer_demands.len() != num_customers {
            return Err(ApplicationError::InvalidConfiguration(
                "Customer demands length mismatch".to_string()
            ));
        }
        
        if vehicle_capacities.len() != num_vehicles {
            return Err(ApplicationError::InvalidConfiguration(
                "Vehicle capacities length mismatch".to_string()
            ));
        }
        
        Ok(Self {
            num_vehicles,
            num_customers,
            distance_matrix,
            customer_demands,
            vehicle_capacities,
            time_windows: vec![(0.0, 1000.0); num_customers + 1], // Default wide windows
            service_times: vec![5.0; num_customers + 1], // Default 5 minute service
            max_route_duration: 480.0, // 8 hours
            vehicle_costs: vec![1.0; num_vehicles], // Default cost per distance unit
            customer_priorities: vec![1.0; num_customers], // Equal priority
        })
    }
    
    /// Set time windows for customers
    pub fn set_time_windows(&mut self, time_windows: Vec<(f64, f64)>) -> ApplicationResult<()> {
        if time_windows.len() != self.num_customers + 1 {
            return Err(ApplicationError::InvalidConfiguration(
                "Time windows length mismatch".to_string()
            ));
        }
        self.time_windows = time_windows;
        Ok(())
    }
    
    /// Calculate total route distance
    pub fn calculate_route_distance(&self, route: &[usize]) -> f64 {
        if route.len() < 2 {
            return 0.0;
        }
        
        let mut total_distance = 0.0;
        for window in route.windows(2) {
            total_distance += self.distance_matrix[window[0]][window[1]];
        }
        total_distance
    }
    
    /// Check if route satisfies capacity constraints
    pub fn check_capacity_constraint(&self, route: &[usize], vehicle_idx: usize) -> bool {
        let total_demand: f64 = route.iter()
            .skip(1) // Skip depot
            .take(route.len() - 2) // Skip return to depot
            .map(|&customer| self.customer_demands[customer - 1]) // Adjust for depot offset
            .sum();
        
        total_demand <= self.vehicle_capacities[vehicle_idx]
    }
    
    /// Check time window feasibility
    pub fn check_time_windows(&self, route: &[usize]) -> bool {
        let mut current_time = 0.0;
        
        for i in 1..route.len() {
            let prev_location = route[i - 1];
            let current_location = route[i];
            
            // Travel time
            current_time += self.distance_matrix[prev_location][current_location] / 50.0; // Assume 50 km/h
            
            // Check time window
            let (earliest, latest) = self.time_windows[current_location];
            if current_time > latest {
                return false; // Arrived too late
            }
            
            // Wait if arrived too early
            if current_time < earliest {
                current_time = earliest;
            }
            
            // Add service time
            current_time += self.service_times[current_location];
        }
        
        current_time <= self.max_route_duration
    }
}

impl OptimizationProblem for VehicleRoutingProblem {
    type Solution = VehicleRoutingSolution;
    type ObjectiveValue = f64;
    
    fn description(&self) -> String {
        format!(
            "Vehicle routing problem with {} vehicles and {} customers",
            self.num_vehicles,
            self.num_customers
        )
    }
    
    fn size_metrics(&self) -> HashMap<String, usize> {
        let mut metrics = HashMap::new();
        metrics.insert("num_vehicles".to_string(), self.num_vehicles);
        metrics.insert("num_customers".to_string(), self.num_customers);
        metrics.insert("total_locations".to_string(), self.num_customers + 1);
        metrics
    }
    
    fn validate(&self) -> ApplicationResult<()> {
        if self.num_vehicles == 0 {
            return Err(ApplicationError::DataValidationError(
                "At least one vehicle required".to_string()
            ));
        }
        
        if self.num_customers == 0 {
            return Err(ApplicationError::DataValidationError(
                "At least one customer required".to_string()
            ));
        }
        
        // Check positive demands
        for &demand in &self.customer_demands {
            if demand < 0.0 {
                return Err(ApplicationError::DataValidationError(
                    "Customer demands must be non-negative".to_string()
                ));
            }
        }
        
        // Check positive capacities
        for &capacity in &self.vehicle_capacities {
            if capacity <= 0.0 {
                return Err(ApplicationError::DataValidationError(
                    "Vehicle capacities must be positive".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    fn to_qubo(&self) -> ApplicationResult<QuboFormulation> {
        let mut builder = QuboBuilder::new();
        
        // Binary variables: x[i][j][k] = 1 if vehicle k travels from location i to j
        let total_locations = self.num_customers + 1;
        let num_vars = total_locations * total_locations * self.num_vehicles;
        
        // Helper function to get variable index
        let var_index = |i: usize, j: usize, k: usize| -> usize {
            i * total_locations * self.num_vehicles + j * self.num_vehicles + k
        };
        
        // Objective: minimize total distance and costs
        for i in 0..total_locations {
            for j in 0..total_locations {
                if i != j {
                    for k in 0..self.num_vehicles {
                        let var_idx = var_index(i, j, k);
                        let cost = self.distance_matrix[i][j] * self.vehicle_costs[k];
                        builder.add_bias(var_idx, cost);
                    }
                }
            }
        }
        
        // Constraint: each customer visited exactly once
        let visit_penalty = 1000.0;
        for customer in 1..total_locations {
            for i in 0..total_locations {
                if i != customer {
                    for k1 in 0..self.num_vehicles {
                        let var1 = var_index(i, customer, k1);
                        builder.add_bias(var1, -visit_penalty);
                        
                        for j in 0..total_locations {
                            if j != customer {
                                for k2 in 0..self.num_vehicles {
                                    if (i, k1) != (j, k2) {
                                        let var2 = var_index(j, customer, k2);
                                        builder.add_coupling(var1, var2, visit_penalty);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Constraint: flow conservation (what comes in must go out)
        let flow_penalty = 800.0;
        for location in 0..total_locations {
            for k in 0..self.num_vehicles {
                let mut in_vars = Vec::new();
                let mut out_vars = Vec::new();
                
                for i in 0..total_locations {
                    if i != location {
                        in_vars.push(var_index(i, location, k));
                    }
                }
                
                for j in 0..total_locations {
                    if j != location {
                        out_vars.push(var_index(location, j, k));
                    }
                }
                
                // Add flow conservation penalty
                for &in_var in &in_vars {
                    for &out_var in &out_vars {
                        builder.add_coupling(in_var, out_var, -flow_penalty);
                    }
                }
            }
        }
        
        // Capacity constraints (simplified penalty)
        let capacity_penalty = 500.0;
        for k in 0..self.num_vehicles {
            for customer in 1..total_locations {
                let demand = self.customer_demands[customer - 1];
                let capacity_violation = (demand / self.vehicle_capacities[k]).max(0.0);
                
                for i in 0..total_locations {
                    if i != customer {
                        let var_idx = var_index(i, customer, k);
                        builder.add_bias(var_idx, capacity_penalty * capacity_violation);
                    }
                }
            }
        }
        
        builder.build()
    }
    
    fn evaluate_solution(&self, solution: &Self::Solution) -> ApplicationResult<Self::ObjectiveValue> {
        let mut total_cost = 0.0;
        
        for (vehicle_idx, route) in solution.routes.iter().enumerate() {
            if !route.is_empty() {
                let distance = self.calculate_route_distance(route);
                total_cost += distance * self.vehicle_costs[vehicle_idx];
                
                // Add penalty for constraint violations
                if !self.check_capacity_constraint(route, vehicle_idx) {
                    total_cost += 10000.0; // Large penalty
                }
                
                if !self.check_time_windows(route) {
                    total_cost += 5000.0; // Time window penalty
                }
            }
        }
        
        Ok(total_cost)
    }
    
    fn is_feasible(&self, solution: &Self::Solution) -> bool {
        // Check that all customers are visited exactly once
        let mut visited_customers = vec![false; self.num_customers];
        
        for route in &solution.routes {
            for &location in route {
                if location > 0 && location <= self.num_customers {
                    if visited_customers[location - 1] {
                        return false; // Customer visited more than once
                    }
                    visited_customers[location - 1] = true;
                }
            }
        }
        
        // Check all customers are visited
        if !visited_customers.iter().all(|&visited| visited) {
            return false;
        }
        
        // Check capacity and time window constraints for each route
        for (vehicle_idx, route) in solution.routes.iter().enumerate() {
            if !route.is_empty() {
                if !self.check_capacity_constraint(route, vehicle_idx) {
                    return false;
                }
                
                if !self.check_time_windows(route) {
                    return false;
                }
            }
        }
        
        true
    }
}

/// Vehicle routing solution
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
    /// Transportation metrics
    pub metrics: TransportationMetrics,
}

/// Statistics for individual routes
#[derive(Debug, Clone)]
pub struct RouteStatistics {
    /// Route distance
    pub distance: f64,
    /// Route duration
    pub duration: f64,
    /// Total demand served
    pub total_demand: f64,
    /// Number of customers served
    pub num_customers: usize,
    /// Capacity utilization
    pub capacity_utilization: f64,
    /// Time window compliance
    pub time_window_compliance: bool,
}

/// Transportation performance metrics
#[derive(Debug, Clone)]
pub struct TransportationMetrics {
    /// Fleet utilization (fraction of vehicles used)
    pub fleet_utilization: f64,
    /// Average route distance
    pub avg_route_distance: f64,
    /// Average route duration
    pub avg_route_duration: f64,
    /// Total fuel consumption estimate
    pub fuel_consumption: f64,
    /// CO2 emissions estimate (kg)
    pub co2_emissions: f64,
    /// Service level (customers served on time)
    pub service_level: f64,
    /// Cost per customer served
    pub cost_per_customer: f64,
}

impl IndustrySolution for VehicleRoutingSolution {
    type Problem = VehicleRoutingProblem;
    
    fn from_binary(problem: &Self::Problem, binary_solution: &[i8]) -> ApplicationResult<Self> {
        let total_locations = problem.num_customers + 1;
        let num_vars = total_locations * total_locations * problem.num_vehicles;
        
        // Decode binary solution to routes
        let mut routes = vec![Vec::new(); problem.num_vehicles];
        
        // Helper function to get variable index
        let var_index = |i: usize, j: usize, k: usize| -> usize {
            i * total_locations * problem.num_vehicles + j * total_locations + k
        };
        
        // Build routes from binary variables
        for k in 0..problem.num_vehicles {
            let mut current_location = 0; // Start at depot
            let mut route = vec![0]; // Include depot
            let mut visited = vec![false; total_locations];
            visited[0] = true;
            
            // Follow the path indicated by the binary solution
            while route.len() < total_locations && route.len() < 10 { // Prevent infinite loops
                let mut next_location = None;
                
                for j in 0..total_locations {
                    if !visited[j] {
                        let var_idx = var_index(current_location, j, k);
                        if var_idx < binary_solution.len() && binary_solution[var_idx] > 0 {
                            next_location = Some(j);
                            break;
                        }
                    }
                }
                
                if let Some(next) = next_location {
                    route.push(next);
                    visited[next] = true;
                    current_location = next;
                } else {
                    break;
                }
            }
            
            // Return to depot if we visited customers
            if route.len() > 1 {
                route.push(0);
            }
            
            routes[k] = route;
        }
        
        // Calculate statistics
        let mut total_distance = 0.0;
        let mut route_stats = Vec::new();
        let mut vehicles_used = 0;
        
        for (vehicle_idx, route) in routes.iter().enumerate() {
            if route.len() > 2 { // More than just depot->depot
                vehicles_used += 1;
                let distance = problem.calculate_route_distance(route);
                total_distance += distance;
                
                let total_demand: f64 = route.iter()
                    .skip(1)
                    .take(route.len() - 2)
                    .map(|&customer| problem.customer_demands[customer - 1])
                    .sum();
                
                let capacity_utilization = total_demand / problem.vehicle_capacities[vehicle_idx];
                let duration = distance / 50.0 + route.len() as f64 * 5.0; // Travel + service time
                
                route_stats.push(RouteStatistics {
                    distance,
                    duration,
                    total_demand,
                    num_customers: route.len().saturating_sub(2),
                    capacity_utilization,
                    time_window_compliance: problem.check_time_windows(route),
                });
            } else {
                route_stats.push(RouteStatistics {
                    distance: 0.0,
                    duration: 0.0,
                    total_demand: 0.0,
                    num_customers: 0,
                    capacity_utilization: 0.0,
                    time_window_compliance: true,
                });
            }
        }
        
        let total_cost = total_distance + (problem.num_vehicles - vehicles_used) as f64 * 100.0; // Penalty for unused vehicles
        
        // Calculate transportation metrics
        let fleet_utilization = vehicles_used as f64 / problem.num_vehicles as f64;
        let avg_route_distance = if vehicles_used > 0 { total_distance / vehicles_used as f64 } else { 0.0 };
        let avg_route_duration = route_stats.iter()
            .filter(|s| s.num_customers > 0)
            .map(|s| s.duration)
            .sum::<f64>() / vehicles_used.max(1) as f64;
        
        let fuel_consumption = total_distance * 0.1; // 0.1 L/km estimate
        let co2_emissions = fuel_consumption * 2.3; // 2.3 kg CO2/L estimate
        
        let customers_on_time = route_stats.iter()
            .filter(|s| s.time_window_compliance)
            .map(|s| s.num_customers)
            .sum::<usize>();
        let service_level = customers_on_time as f64 / problem.num_customers as f64;
        
        let cost_per_customer = if problem.num_customers > 0 {
            total_cost / problem.num_customers as f64
        } else {
            0.0
        };
        
        let metrics = TransportationMetrics {
            fleet_utilization,
            avg_route_distance,
            avg_route_duration,
            fuel_consumption,
            co2_emissions,
            service_level,
            cost_per_customer,
        };
        
        Ok(VehicleRoutingSolution {
            routes,
            total_distance,
            total_cost,
            route_stats,
            metrics,
        })
    }
    
    fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("type".to_string(), "Vehicle Routing".to_string());
        summary.insert("total_distance".to_string(), format!("{:.1} km", self.total_distance));
        summary.insert("total_cost".to_string(), format!("${:.2}", self.total_cost));
        summary.insert("fleet_utilization".to_string(), format!("{:.1}%", self.metrics.fleet_utilization * 100.0));
        summary.insert("service_level".to_string(), format!("{:.1}%", self.metrics.service_level * 100.0));
        summary.insert("fuel_consumption".to_string(), format!("{:.1} L", self.metrics.fuel_consumption));
        summary.insert("co2_emissions".to_string(), format!("{:.1} kg", self.metrics.co2_emissions));
        
        let vehicles_used = self.routes.iter().filter(|r| r.len() > 2).count();
        summary.insert("vehicles_used".to_string(), vehicles_used.to_string());
        
        summary
    }
    
    fn metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("total_distance".to_string(), self.total_distance);
        metrics.insert("total_cost".to_string(), self.total_cost);
        metrics.insert("fleet_utilization".to_string(), self.metrics.fleet_utilization);
        metrics.insert("avg_route_distance".to_string(), self.metrics.avg_route_distance);
        metrics.insert("avg_route_duration".to_string(), self.metrics.avg_route_duration);
        metrics.insert("fuel_consumption".to_string(), self.metrics.fuel_consumption);
        metrics.insert("co2_emissions".to_string(), self.metrics.co2_emissions);
        metrics.insert("service_level".to_string(), self.metrics.service_level);
        metrics.insert("cost_per_customer".to_string(), self.metrics.cost_per_customer);
        
        let vehicles_used = self.routes.iter().filter(|r| r.len() > 2).count();
        metrics.insert("vehicles_used".to_string(), vehicles_used as f64);
        
        metrics
    }
    
    fn export_format(&self) -> ApplicationResult<String> {
        let mut output = String::new();
        output.push_str("# Vehicle Routing Optimization Report\n\n");
        
        output.push_str("## Summary\n");
        output.push_str(&format!("Total Distance: {:.1} km\n", self.total_distance));
        output.push_str(&format!("Total Cost: ${:.2}\n", self.total_cost));
        output.push_str(&format!("Fleet Utilization: {:.1}%\n", self.metrics.fleet_utilization * 100.0));
        output.push_str(&format!("Service Level: {:.1}%\n", self.metrics.service_level * 100.0));
        
        output.push_str("\n## Environmental Impact\n");
        output.push_str(&format!("Fuel Consumption: {:.1} L\n", self.metrics.fuel_consumption));
        output.push_str(&format!("CO2 Emissions: {:.1} kg\n", self.metrics.co2_emissions));
        
        output.push_str("\n## Route Details\n");
        for (i, (route, stats)) in self.routes.iter().zip(&self.route_stats).enumerate() {
            if route.len() > 2 {
                output.push_str(&format!("Vehicle {}: ", i + 1));
                output.push_str(&format!("Route {:?}, ", route));
                output.push_str(&format!("Distance: {:.1} km, ", stats.distance));
                output.push_str(&format!("Duration: {:.1} h, ", stats.duration / 60.0));
                output.push_str(&format!("Customers: {}, ", stats.num_customers));
                output.push_str(&format!("Capacity: {:.1}%\n", stats.capacity_utilization * 100.0));
            }
        }
        
        Ok(output)
    }
}

/// Traffic Flow Optimization Problem
#[derive(Debug, Clone)]
pub struct TrafficFlowOptimization {
    /// Number of intersections
    pub num_intersections: usize,
    /// Number of road segments
    pub num_segments: usize,
    /// Traffic flow rates between intersections
    pub flow_matrix: Vec<Vec<f64>>,
    /// Road segment capacities
    pub segment_capacities: Vec<f64>,
    /// Traffic light cycle times (seconds)
    pub cycle_times: Vec<f64>,
    /// Green time allocations for each direction
    pub green_time_allocations: Vec<Vec<f64>>,
    /// Priority routes (emergency, public transport)
    pub priority_routes: Vec<PriorityRoute>,
    /// Environmental constraints
    pub emission_limits: Vec<f64>,
}

/// Priority route definition
#[derive(Debug, Clone)]
pub struct PriorityRoute {
    /// Route path (sequence of intersections)
    pub path: Vec<usize>,
    /// Priority level (1-10, higher is more important)
    pub priority: usize,
    /// Maximum allowed delay (seconds)
    pub max_delay: f64,
    /// Route type
    pub route_type: RouteType,
}

/// Types of priority routes
#[derive(Debug, Clone, PartialEq)]
pub enum RouteType {
    /// Emergency services
    Emergency,
    /// Public transportation
    PublicTransport,
    /// Commercial delivery
    Commercial,
    /// Regular traffic
    Regular,
}

impl TrafficFlowOptimization {
    /// Create a new traffic flow optimization problem
    pub fn new(
        num_intersections: usize,
        num_segments: usize,
        flow_matrix: Vec<Vec<f64>>,
        segment_capacities: Vec<f64>,
    ) -> ApplicationResult<Self> {
        if flow_matrix.len() != num_intersections {
            return Err(ApplicationError::InvalidConfiguration(
                "Flow matrix dimension mismatch".to_string()
            ));
        }
        
        if segment_capacities.len() != num_segments {
            return Err(ApplicationError::InvalidConfiguration(
                "Segment capacities length mismatch".to_string()
            ));
        }
        
        Ok(Self {
            num_intersections,
            num_segments,
            flow_matrix,
            segment_capacities,
            cycle_times: vec![90.0; num_intersections], // Default 90 second cycles
            green_time_allocations: vec![vec![30.0, 30.0, 15.0, 15.0]; num_intersections], // 4-direction default
            priority_routes: Vec::new(),
            emission_limits: vec![100.0; num_intersections], // kg CO2/hour
        })
    }
    
    /// Add priority route
    pub fn add_priority_route(&mut self, route: PriorityRoute) {
        self.priority_routes.push(route);
    }
    
    /// Calculate total travel time for given signal timing
    pub fn calculate_total_travel_time(&self, solution: &TrafficFlowSolution) -> f64 {
        let mut total_time = 0.0;
        
        for i in 0..self.num_intersections {
            for j in 0..self.num_intersections {
                if i != j && self.flow_matrix[i][j] > 0.0 {
                    let flow = self.flow_matrix[i][j];
                    let delay = self.calculate_intersection_delay(i, &solution.signal_timings[i]);
                    total_time += flow * delay;
                }
            }
        }
        
        total_time
    }
    
    /// Calculate delay at intersection based on signal timing
    fn calculate_intersection_delay(&self, intersection: usize, timing: &IntersectionTiming) -> f64 {
        // Simplified delay calculation using Webster's formula
        let cycle_time = timing.cycle_time;
        let effective_green = timing.green_times.iter().sum::<f64>();
        let flow_ratio = 0.8; // Simplified
        
        if effective_green <= 0.0 {
            return 1000.0; // Very high delay for invalid timing
        }
        
        let delay = (cycle_time * (1.0 - effective_green / cycle_time).powi(2)) / 
                   (2.0 * (1.0 - flow_ratio * effective_green / cycle_time));
        
        delay.max(0.0).min(1000.0) // Cap at reasonable values
    }
}

/// Traffic flow solution
#[derive(Debug, Clone)]
pub struct TrafficFlowSolution {
    /// Signal timing for each intersection
    pub signal_timings: Vec<IntersectionTiming>,
    /// Total travel time
    pub total_travel_time: f64,
    /// Average delay per vehicle
    pub average_delay: f64,
    /// Throughput metrics
    pub throughput_metrics: ThroughputMetrics,
    /// Environmental impact
    pub environmental_impact: EnvironmentalImpact,
}

/// Signal timing for an intersection
#[derive(Debug, Clone)]
pub struct IntersectionTiming {
    /// Total cycle time (seconds)
    pub cycle_time: f64,
    /// Green times for each direction (seconds)
    pub green_times: Vec<f64>,
    /// Offset from master clock (seconds)
    pub offset: f64,
    /// Coordination with adjacent intersections
    pub coordination_factor: f64,
}

/// Traffic throughput metrics
#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    /// Total vehicles processed per hour
    pub vehicles_per_hour: f64,
    /// Average speed (km/h)
    pub average_speed: f64,
    /// Capacity utilization
    pub capacity_utilization: f64,
    /// Queue lengths
    pub average_queue_length: f64,
    /// Level of service grade
    pub level_of_service: String,
}

/// Environmental impact metrics
#[derive(Debug, Clone)]
pub struct EnvironmentalImpact {
    /// CO2 emissions (kg/hour)
    pub co2_emissions: f64,
    /// NOx emissions (kg/hour)
    pub nox_emissions: f64,
    /// Fuel consumption (L/hour)
    pub fuel_consumption: f64,
    /// Noise level (dB)
    pub noise_level: f64,
}

/// Create benchmark transportation problems
pub fn create_benchmark_problems(size: usize) -> ApplicationResult<Vec<Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>>> {
    let mut problems = Vec::new();
    
    // Problem 1: Small vehicle routing problem
    let distance_matrix = vec![
        vec![0.0, 10.0, 15.0, 20.0],  // Depot distances
        vec![10.0, 0.0, 8.0, 12.0],   // Customer 1
        vec![15.0, 8.0, 0.0, 9.0],    // Customer 2
        vec![20.0, 12.0, 9.0, 0.0],   // Customer 3
    ];
    
    let customer_demands = vec![5.0, 8.0, 3.0];
    let vehicle_capacities = vec![15.0, 12.0];
    
    let vrp = VehicleRoutingProblem::new(
        2, // 2 vehicles
        3, // 3 customers
        distance_matrix,
        customer_demands,
        vehicle_capacities,
    )?;
    
    problems.push(Box::new(vrp) as Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>);
    
    // Problem 2: Larger VRP for complex scenarios
    if size >= 6 {
        let mut large_distance_matrix = vec![vec![0.0; size + 1]; size + 1];
        
        // Generate random but realistic distances
        for i in 0..=size {
            for j in 0..=size {
                if i != j {
                    large_distance_matrix[i][j] = (i as f64 - j as f64).abs() * 5.0 + 10.0;
                }
            }
        }
        
        let large_demands = vec![5.0; size];
        let large_capacities = vec![20.0; size / 2];
        
        let large_vrp = VehicleRoutingProblem::new(
            size / 2,
            size,
            large_distance_matrix,
            large_demands,
            large_capacities,
        )?;
        
        problems.push(Box::new(large_vrp) as Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>);
    }
    
    Ok(problems)
}

/// Solve vehicle routing problem using quantum annealing
pub fn solve_vehicle_routing(
    problem: &VehicleRoutingProblem,
    params: Option<AnnealingParams>,
) -> ApplicationResult<VehicleRoutingSolution> {
    // Convert to QUBO
    let qubo = problem.to_qubo()?;
    
    // Convert to Ising
    let ising = IsingModel::from_qubo(&qubo)?;
    
    // Set up annealing parameters
    let annealing_params = params.unwrap_or_else(|| {
        let mut p = AnnealingParams::default();
        p.num_sweeps = 30000;
        p.num_repetitions = 50;
        p.initial_temperature = 8.0;
        p.final_temperature = 0.001;
        p
    });
    
    // Solve with classical annealing
    let simulator = ClassicalAnnealingSimulator::new(annealing_params)
        .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;
    
    let result = simulator.solve(&ising)
        .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;
    
    // Convert solution back to vehicle routing solution
    VehicleRoutingSolution::from_binary(problem, &result.best_spins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vehicle_routing_creation() {
        let distance_matrix = vec![
            vec![0.0, 10.0, 15.0],
            vec![10.0, 0.0, 8.0],
            vec![15.0, 8.0, 0.0],
        ];
        let demands = vec![5.0, 8.0];
        let capacities = vec![15.0];
        
        let vrp = VehicleRoutingProblem::new(1, 2, distance_matrix, demands, capacities).unwrap();
        assert_eq!(vrp.num_vehicles, 1);
        assert_eq!(vrp.num_customers, 2);
    }
    
    #[test]
    fn test_route_distance_calculation() {
        let distance_matrix = vec![
            vec![0.0, 10.0, 15.0],
            vec![10.0, 0.0, 8.0],
            vec![15.0, 8.0, 0.0],
        ];
        let demands = vec![5.0, 8.0];
        let capacities = vec![15.0];
        
        let vrp = VehicleRoutingProblem::new(1, 2, distance_matrix, demands, capacities).unwrap();
        
        let route = vec![0, 1, 2, 0]; // Depot -> Customer 1 -> Customer 2 -> Depot
        let distance = vrp.calculate_route_distance(&route);
        assert_eq!(distance, 10.0 + 8.0 + 15.0); // 33.0
    }
    
    #[test]
    fn test_capacity_constraint() {
        let distance_matrix = vec![vec![0.0; 3]; 3];
        let demands = vec![5.0, 8.0];
        let capacities = vec![10.0]; // Small capacity
        
        let vrp = VehicleRoutingProblem::new(1, 2, distance_matrix, demands, capacities).unwrap();
        
        let route = vec![0, 1, 2, 0]; // Both customers
        assert!(!vrp.check_capacity_constraint(&route, 0)); // Should exceed capacity
        
        let route2 = vec![0, 1, 0]; // Only customer 1
        assert!(vrp.check_capacity_constraint(&route2, 0)); // Should be within capacity
    }
    
    #[test]
    fn test_traffic_flow_creation() {
        let flow_matrix = vec![
            vec![0.0, 100.0, 50.0],
            vec![80.0, 0.0, 120.0],
            vec![60.0, 90.0, 0.0],
        ];
        let capacities = vec![200.0, 180.0, 150.0];
        
        let traffic = TrafficFlowOptimization::new(3, 3, flow_matrix, capacities).unwrap();
        assert_eq!(traffic.num_intersections, 3);
        assert_eq!(traffic.num_segments, 3);
    }
    
    #[test]
    fn test_vrp_validation() {
        let distance_matrix = vec![vec![0.0; 3]; 3];
        let demands = vec![5.0, 8.0];
        let capacities = vec![15.0];
        
        let vrp = VehicleRoutingProblem::new(1, 2, distance_matrix, demands, capacities).unwrap();
        assert!(vrp.validate().is_ok());
        
        // Test invalid VRP (zero vehicles)
        let invalid_vrp = VehicleRoutingProblem::new(0, 2, distance_matrix, demands, capacities);
        // Should create successfully but fail validation
        // Note: Constructor might return error before we get to validate
    }
    
    #[test]
    fn test_benchmark_problems() {
        let problems = create_benchmark_problems(4).unwrap();
        assert_eq!(problems.len(), 1); // Size 4 gets only the small problem
        
        let larger_problems = create_benchmark_problems(8).unwrap();
        assert_eq!(larger_problems.len(), 2); // Size 8 gets both problems
        
        for problem in &larger_problems {
            assert!(problem.validate().is_ok());
        }
    }
}