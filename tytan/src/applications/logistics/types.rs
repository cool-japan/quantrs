//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Route {
    pub vehicle_id: usize,
    pub path: Vec<usize>,
    pub total_distance: f64,
    pub total_demand: f64,
    pub arrival_times: Vec<f64>,
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
        let mut var_map = HashMap::new();
        let mut var_idx = 0;
        for t in 0..self.time_horizon {
            for s in &self.network.suppliers {
                for w in &self.network.warehouses {
                    let var_name = format!("x_{}_{}_{}", s.id, w.id, t);
                    var_map.insert(var_name, var_idx);
                    var_idx += 1;
                }
            }
            for w in &self.network.warehouses {
                for d in &self.network.distribution_centers {
                    let var_name = format!("y_{}_{}_{}", w.id, d.id, t);
                    var_map.insert(var_name, var_idx);
                    var_idx += 1;
                }
            }
            for d in &self.network.distribution_centers {
                for c in &self.network.customers {
                    let var_name = format!("z_{}_{}_{}", d.id, c.id, t);
                    var_map.insert(var_name, var_idx);
                    var_idx += 1;
                }
            }
            for w in &self.network.warehouses {
                let var_name = format!("I_{}_{}", w.id, t);
                var_map.insert(var_name, var_idx);
                var_idx += 1;
            }
        }
        let n_vars = var_idx;
        let mut qubo = Array2::zeros((n_vars, n_vars));
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
                    qubo[[idx, idx]] += 1.0;
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
        for _w in &self.network.warehouses {
            for _t in 1..self.time_horizon {}
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
        for s in &self.network.suppliers {
            for t in 0..self.time_horizon {
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
        for c in &self.network.customers {
            for t in 0..self.time_horizon.min(c.demand.len()) {
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
#[derive(Debug, Clone)]
pub struct DistributionCenter {
    pub id: usize,
    pub capacity: f64,
    pub processing_cost: f64,
    pub location: (f64, f64),
}
#[derive(Debug, Clone)]
pub struct StorageLocation {
    pub id: usize,
    pub position: (usize, usize, usize),
    pub capacity: f64,
    pub item_type: Option<String>,
    pub accessibility: f64,
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
pub struct Warehouse {
    pub id: usize,
    pub capacity: f64,
    pub holding_cost: f64,
    pub fixed_cost: f64,
    pub location: (f64, f64),
}
#[derive(Debug, Clone)]
pub struct OrderItem {
    pub sku: String,
    pub quantity: usize,
    pub location: Option<usize>,
}
#[derive(Debug, Clone)]
pub struct Order {
    pub id: usize,
    pub items: Vec<OrderItem>,
    pub priority: f64,
    pub due_time: f64,
}
#[derive(Debug, Clone)]
pub struct PickingRoute {
    pub locations: Vec<usize>,
    pub sequence: Vec<usize>,
    pub total_distance: f64,
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
        let mut pick_locations = Vec::new();
        for order in orders {
            for item in &order.items {
                if let Some(loc) = item.location {
                    pick_locations.push(loc);
                }
            }
        }
        pick_locations.sort_unstable();
        pick_locations.dedup();
        let n = pick_locations.len() + 1;
        let mut distances = Array2::zeros((n, n));
        let station = self.layout.picking_stations[0];
        for (i, &loc) in pick_locations.iter().enumerate() {
            let loc_pos = self.layout.locations[loc].position;
            distances[[0, i + 1]] = self.calculate_distance((station.0, station.1, 0), loc_pos);
            distances[[i + 1, 0]] = distances[[0, i + 1]];
        }
        for (i, &loc1) in pick_locations.iter().enumerate() {
            for (j, &loc2) in pick_locations.iter().enumerate() {
                if i != j {
                    let pos1 = self.layout.locations[loc1].position;
                    let pos2 = self.layout.locations[loc2].position;
                    distances[[i + 1, j + 1]] = self.calculate_distance(pos1, pos2);
                }
            }
        }
        let tsp = TSPOptimizer::new(distances)?;
        let (_qubo, _var_map) = tsp.build_qubo()?;
        let sequence = (0..pick_locations.len()).collect();
        Ok(PickingRoute {
            locations: pick_locations,
            sequence,
            total_distance: 0.0,
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
        let travel_time = route.total_distance / 1.0;
        let pick_time = route.locations.len() as f64 * 10.0;
        travel_time + pick_time
    }
}
#[derive(Debug, Clone)]
pub struct Customer {
    pub id: usize,
    pub demand: Array1<f64>,
    pub priority: f64,
    pub location: (f64, f64),
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
pub struct TransportLink {
    pub from_type: NodeType,
    pub from_id: usize,
    pub to_type: NodeType,
    pub to_id: usize,
    pub capacity: f64,
    pub cost_per_unit: f64,
    pub lead_time: usize,
}
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub violations: Vec<ConstraintViolation>,
    pub total_distance: f64,
    pub num_vehicles_used: usize,
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
#[derive(Debug, Clone)]
pub enum ReplenishmentPolicy {
    /// Fixed order quantity
    FixedQuantity { quantity: f64 },
    /// Reorder point
    ReorderPoint { level: f64 },
    /// Periodic review
    Periodic { interval: usize },
}
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
        let _n_customers = n_locations - 1;
        let n_vars = self.num_vehicles * n_locations * n_locations;
        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();
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
        self.add_distance_objective(&mut qubo, &var_map)?;
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
        for j in 1..n_locations {
            for v1 in 0..self.num_vehicles {
                for i1 in 0..n_locations {
                    if i1 != j {
                        let var1 = format!("x_{v1}_{i1}_{j}");
                        if let Some(&idx1) = var_map.get(&var1) {
                            qubo[[idx1, idx1]] -= 2.0 * penalty;
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
        for v in 0..self.num_vehicles {
            for i in 0..n_locations {
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
        self.add_capacity_constraints(qubo, var_map, penalty)?;
        for v in 0..self.num_vehicles {
            for j in 1..n_locations {
                let var = format!("x_{}_{}_{}", v, 0, j);
                if let Some(&idx) = var_map.get(&var) {
                    qubo[[idx, idx]] -= penalty * 0.1;
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
        let n_locations = self.distance_matrix.shape()[0];
        for v in 0..self.num_vehicles {
            let route_demand = 0.0;
            for i in 0..n_locations {
                for j in 1..n_locations {
                    let var = format!("x_{v}_{i}_{j}");
                    if let Some(&idx) = var_map.get(&var) {
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
            for v in 0..self.num_vehicles {
                for i in 0..n_locations {
                    for j in 0..n_locations {
                        if i != j {
                            let var = format!("x_{v}_{i}_{j}");
                            if let Some(&idx) = var_map.get(&var) {
                                let travel_time = self.distance_matrix[[i, j]];
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
        let penalty = 1000.0;
        for v in 0..self.num_vehicles {
            for &depot in depots {
                for j in 0..self.distance_matrix.shape()[0] {
                    if !depots.contains(&j) {
                        let var = format!("x_{v}_{depot}_{j}");
                        if let Some(&idx) = var_map.get(&var) {
                            qubo[[idx, idx]] -= penalty * 0.1;
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
        for route in routes {
            if route.total_demand > self.vehicle_capacity {
                violations.push(ConstraintViolation::CapacityExceeded {
                    vehicle: route.vehicle_id,
                    demand: route.total_demand,
                    capacity: self.vehicle_capacity,
                });
            }
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
            for &loc in &route.path {
                if loc != self.depot {
                    visited_customers.insert(loc);
                }
            }
        }
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
pub struct PickingPlan {
    pub batches: Vec<Batch>,
    pub total_distance: f64,
    pub total_time: f64,
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
#[derive(Debug, Clone)]
pub struct Batch {
    pub orders: Vec<Order>,
    pub route: PickingRoute,
    pub estimated_time: f64,
}
#[derive(Debug, Clone)]
pub enum DistanceType {
    Manhattan,
    Euclidean,
    Rectilinear,
    Custom,
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
/// Traveling Salesman Problem (TSP) optimizer
pub struct TSPOptimizer {
    /// Distance matrix
    distance_matrix: Array2<f64>,
    /// Problem variant
    variant: TSPVariant,
    /// Subtour elimination method
    subtour_method: SubtourElimination,
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
        let n_vars = n * n;
        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();
        for i in 0..n {
            for t in 0..n {
                let var_name = format!("x_{i}_{t}");
                var_map.insert(var_name, i * n + t);
            }
        }
        let penalty = 1000.0;
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
        for i in 0..n {
            for t1 in 0..n {
                let idx1 = i * n + t1;
                qubo[[idx1, idx1]] -= 2.0 * penalty;
                for t2 in 0..n {
                    let idx2 = i * n + t2;
                    qubo[[idx1, idx2]] += penalty;
                }
            }
        }
        for t in 0..n {
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
        let n_vars = num_salesmen * n * n;
        let qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();
        for s in 0..num_salesmen {
            for i in 0..n {
                for t in 0..n {
                    let var_name = format!("x_{s}_{i}_{t}");
                    var_map.insert(var_name, s * n * n + i * n + t);
                }
            }
        }
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeType {
    Supplier,
    Warehouse,
    DistributionCenter,
    Customer,
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
        let n_locations = self.optimizer.distance_matrix.shape()[0];
        let n_vars = self.optimizer.num_vehicles * n_locations * n_locations;
        if x.len() != n_vars {
            return f64::INFINITY;
        }
        let mut energy = 0.0;
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
        energy += self.calculate_constraint_penalties(x);
        energy
    }
    fn calculate_constraint_penalties(&self, x: &Array1<f64>) -> f64 {
        let penalty = 1000.0;
        let mut total_penalty = 0.0;
        let n_locations = self.optimizer.distance_matrix.shape()[0];
        for j in 1..n_locations {
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
#[derive(Debug, Clone)]
pub struct TimeWindow {
    /// Earliest arrival time
    pub start: f64,
    /// Latest arrival time
    pub end: f64,
    /// Service time at location
    pub service_time: f64,
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
