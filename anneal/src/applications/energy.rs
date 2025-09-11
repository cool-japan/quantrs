//! Energy Industry Optimization
//!
//! This module provides optimization solutions for the energy industry,
//! including grid optimization, renewable energy management, load balancing,
//! and energy trading optimization.

use super::{
    ApplicationError, ApplicationResult, IndustryConstraint, IndustryObjective, IndustrySolution,
    OptimizationProblem,
};
use crate::ising::IsingModel;
use crate::qubo::{QuboBuilder, QuboFormulation};
use crate::simulator::{AnnealingParams, ClassicalAnnealingSimulator};
use std::collections::HashMap;

/// Smart Grid Optimization Problem
#[derive(Debug, Clone)]
pub struct SmartGridOptimization {
    /// Number of generators
    pub num_generators: usize,
    /// Number of demand nodes
    pub num_demand_nodes: usize,
    /// Generator capacities (MW)
    pub generator_capacities: Vec<f64>,
    /// Generator marginal costs ($/MWh)
    pub generator_costs: Vec<f64>,
    /// Generator ramp rates (MW/hour)
    pub generator_ramp_rates: Vec<f64>,
    /// Demand forecasts by node and time period
    pub demand_forecasts: Vec<Vec<f64>>,
    /// Transmission line capacities
    pub transmission_capacities: HashMap<(usize, usize), f64>,
    /// Transmission line costs
    pub transmission_costs: HashMap<(usize, usize), f64>,
    /// Time periods to optimize
    pub num_time_periods: usize,
    /// Renewable energy availability
    pub renewable_availability: Vec<Vec<f64>>,
    /// Energy storage systems
    pub storage_systems: Vec<EnergyStorageSystem>,
    /// Grid constraints
    pub grid_constraints: Vec<IndustryConstraint>,
}

/// Energy Storage System
#[derive(Debug, Clone)]
pub struct EnergyStorageSystem {
    /// Storage capacity (MWh)
    pub capacity: f64,
    /// Maximum charge/discharge rate (MW)
    pub max_power: f64,
    /// Round-trip efficiency
    pub efficiency: f64,
    /// Location (node index)
    pub location: usize,
    /// Operating cost ($/MWh)
    pub operating_cost: f64,
}

impl SmartGridOptimization {
    /// Create a new smart grid optimization problem
    pub fn new(
        num_generators: usize,
        num_demand_nodes: usize,
        generator_capacities: Vec<f64>,
        generator_costs: Vec<f64>,
        demand_forecasts: Vec<Vec<f64>>,
        num_time_periods: usize,
    ) -> ApplicationResult<Self> {
        if generator_capacities.len() != num_generators {
            return Err(ApplicationError::InvalidConfiguration(
                "Generator capacities must match number of generators".to_string(),
            ));
        }

        if generator_costs.len() != num_generators {
            return Err(ApplicationError::InvalidConfiguration(
                "Generator costs must match number of generators".to_string(),
            ));
        }

        if demand_forecasts.len() != num_demand_nodes {
            return Err(ApplicationError::InvalidConfiguration(
                "Demand forecasts must match number of demand nodes".to_string(),
            ));
        }

        Ok(Self {
            num_generators,
            num_demand_nodes,
            generator_capacities,
            generator_costs,
            generator_ramp_rates: vec![50.0; num_generators], // Default ramp rate
            demand_forecasts,
            transmission_capacities: HashMap::new(),
            transmission_costs: HashMap::new(),
            num_time_periods,
            renewable_availability: vec![vec![0.0; num_time_periods]; num_generators],
            storage_systems: Vec::new(),
            grid_constraints: Vec::new(),
        })
    }

    /// Add transmission line
    pub fn add_transmission_line(
        &mut self,
        from_node: usize,
        to_node: usize,
        capacity: f64,
        cost: f64,
    ) -> ApplicationResult<()> {
        if from_node >= self.num_demand_nodes || to_node >= self.num_demand_nodes {
            return Err(ApplicationError::InvalidConfiguration(
                "Node indices out of bounds".to_string(),
            ));
        }

        self.transmission_capacities
            .insert((from_node, to_node), capacity);
        self.transmission_costs.insert((from_node, to_node), cost);
        Ok(())
    }

    /// Add energy storage system
    pub fn add_storage_system(&mut self, storage: EnergyStorageSystem) -> ApplicationResult<()> {
        if storage.location >= self.num_demand_nodes {
            return Err(ApplicationError::InvalidConfiguration(
                "Storage location out of bounds".to_string(),
            ));
        }

        self.storage_systems.push(storage);
        Ok(())
    }

    /// Set renewable availability profiles
    pub fn set_renewable_availability(
        &mut self,
        availability: Vec<Vec<f64>>,
    ) -> ApplicationResult<()> {
        if availability.len() != self.num_generators {
            return Err(ApplicationError::InvalidConfiguration(
                "Renewable availability must match number of generators".to_string(),
            ));
        }

        self.renewable_availability = availability;
        Ok(())
    }

    /// Calculate total demand for time period
    pub fn calculate_total_demand(&self, time_period: usize) -> f64 {
        if time_period >= self.num_time_periods {
            return 0.0;
        }

        self.demand_forecasts
            .iter()
            .map(|node_demands| node_demands.get(time_period).unwrap_or(&0.0))
            .sum()
    }

    /// Calculate generation cost
    pub fn calculate_generation_cost(&self, solution: &GridSolution) -> f64 {
        let mut total_cost = 0.0;

        for t in 0..self.num_time_periods {
            for g in 0..self.num_generators {
                let generation = solution.generation_schedule[g][t];
                total_cost += generation * self.generator_costs[g];
            }
        }

        total_cost
    }

    /// Calculate transmission cost
    pub fn calculate_transmission_cost(&self, solution: &GridSolution) -> f64 {
        let mut total_cost = 0.0;

        for t in 0..self.num_time_periods {
            for ((from, to), &cost) in &self.transmission_costs {
                if let Some(flow) = solution.transmission_flows.get(&(*from, *to)) {
                    if let Some(time_flows) = flow.get(t) {
                        total_cost += time_flows * cost;
                    }
                }
            }
        }

        total_cost
    }
}

impl OptimizationProblem for SmartGridOptimization {
    type Solution = GridSolution;
    type ObjectiveValue = f64;

    fn description(&self) -> String {
        format!(
            "Smart Grid Optimization with {} generators, {} nodes, {} time periods",
            self.num_generators, self.num_demand_nodes, self.num_time_periods
        )
    }

    fn size_metrics(&self) -> HashMap<String, usize> {
        let mut metrics = HashMap::new();
        metrics.insert("num_generators".to_string(), self.num_generators);
        metrics.insert("num_demand_nodes".to_string(), self.num_demand_nodes);
        metrics.insert("num_time_periods".to_string(), self.num_time_periods);
        metrics.insert(
            "num_transmission_lines".to_string(),
            self.transmission_capacities.len(),
        );
        metrics.insert(
            "num_storage_systems".to_string(),
            self.storage_systems.len(),
        );
        metrics
    }

    fn validate(&self) -> ApplicationResult<()> {
        if self.num_generators == 0 {
            return Err(ApplicationError::DataValidationError(
                "At least one generator required".to_string(),
            ));
        }

        if self.num_demand_nodes == 0 {
            return Err(ApplicationError::DataValidationError(
                "At least one demand node required".to_string(),
            ));
        }

        if self.num_time_periods == 0 {
            return Err(ApplicationError::DataValidationError(
                "At least one time period required".to_string(),
            ));
        }

        // Check that total generation capacity exceeds peak demand
        let total_capacity: f64 = self.generator_capacities.iter().sum();
        let peak_demand = (0..self.num_time_periods)
            .map(|t| self.calculate_total_demand(t))
            .fold(0.0, f64::max);

        if total_capacity < peak_demand {
            return Err(ApplicationError::DataValidationError(
                "Insufficient generation capacity to meet peak demand".to_string(),
            ));
        }

        Ok(())
    }

    fn to_qubo(&self) -> ApplicationResult<(crate::ising::QuboModel, HashMap<String, usize>)> {
        let mut builder = QuboBuilder::new();
        let precision = 10; // Discretization for continuous variables

        // Binary variables: g[gen][time][level] = 1 if generator gen at time time produces level
        let mut var_counter = 0;
        let mut gen_vars = HashMap::new();
        let mut string_var_map = HashMap::new();

        for g in 0..self.num_generators {
            for t in 0..self.num_time_periods {
                for level in 0..precision {
                    let var_name = format!("g_{}_{}_{}", g, t, level);
                    gen_vars.insert((g, t, level), var_counter);
                    string_var_map.insert(var_name, var_counter);
                    var_counter += 1;
                }
            }
        }

        // Objective: minimize generation cost
        for g in 0..self.num_generators {
            for t in 0..self.num_time_periods {
                for level in 0..precision {
                    let generation =
                        (level as f64) * self.generator_capacities[g] / (precision as f64);
                    let cost = generation * self.generator_costs[g];
                    let var_idx = gen_vars[&(g, t, level)];
                    builder.add_bias(var_idx, cost);
                }
            }
        }

        // Constraint: exactly one generation level per generator per time
        let constraint_penalty = 10000.0;
        for g in 0..self.num_generators {
            for t in 0..self.num_time_periods {
                let mut level_vars = Vec::new();
                for level in 0..precision {
                    level_vars.push(gen_vars[&(g, t, level)]);
                }

                // Add penalty for not selecting exactly one level
                for &var1 in &level_vars {
                    builder.add_bias(var1, -constraint_penalty);
                    for &var2 in &level_vars {
                        if var1 != var2 {
                            builder.add_coupling(var1, var2, constraint_penalty);
                        }
                    }
                }
            }
        }

        // Constraint: supply meets demand
        for t in 0..self.num_time_periods {
            let total_demand = self.calculate_total_demand(t);
            let mut supply_vars = Vec::new();
            let mut supply_coeffs = Vec::new();

            for g in 0..self.num_generators {
                for level in 0..precision {
                    let generation =
                        (level as f64) * self.generator_capacities[g] / (precision as f64);
                    supply_vars.push(gen_vars[&(g, t, level)]);
                    supply_coeffs.push(generation);
                }
            }

            // Penalty for supply-demand mismatch
            let mismatch_penalty = 50000.0;
            for (i, &var1) in supply_vars.iter().enumerate() {
                let coeff1 = supply_coeffs[i];
                builder.add_bias(var1, -2.0 * mismatch_penalty * total_demand * coeff1);

                for (j, &var2) in supply_vars.iter().enumerate() {
                    let coeff2 = supply_coeffs[j];
                    builder.add_coupling(var1, var2, mismatch_penalty * coeff1 * coeff2);
                }
            }
        }

        Ok((builder.build(), string_var_map))
    }

    fn evaluate_solution(
        &self,
        solution: &Self::Solution,
    ) -> ApplicationResult<Self::ObjectiveValue> {
        let generation_cost = self.calculate_generation_cost(solution);
        let transmission_cost = self.calculate_transmission_cost(solution);

        Ok(generation_cost + transmission_cost)
    }

    fn is_feasible(&self, solution: &Self::Solution) -> bool {
        // Check supply-demand balance
        for t in 0..self.num_time_periods {
            let total_generation: f64 = solution
                .generation_schedule
                .iter()
                .map(|gen_schedule| gen_schedule.get(t).unwrap_or(&0.0))
                .sum();
            let total_demand = self.calculate_total_demand(t);

            if (total_generation - total_demand).abs() > 1e-6 {
                return false;
            }
        }

        // Check generator capacity constraints
        for (g, gen_schedule) in solution.generation_schedule.iter().enumerate() {
            for &generation in gen_schedule {
                if generation > self.generator_capacities[g] {
                    return false;
                }
            }
        }

        // Check transmission capacity constraints
        for ((from, to), flows) in &solution.transmission_flows {
            if let Some(&capacity) = self.transmission_capacities.get(&(*from, *to)) {
                for &flow in flows {
                    if flow > capacity {
                        return false;
                    }
                }
            }
        }

        true
    }
}

/// Grid optimization solution
#[derive(Debug, Clone)]
pub struct GridSolution {
    /// Generation schedule \[generator\]\[time\]
    pub generation_schedule: Vec<Vec<f64>>,
    /// Transmission flows \[line\]\[time\]
    pub transmission_flows: HashMap<(usize, usize), Vec<f64>>,
    /// Storage operations \[storage\]\[time\]
    pub storage_operations: Vec<Vec<f64>>,
    /// Total cost
    pub total_cost: f64,
    /// Grid performance metrics
    pub grid_metrics: GridMetrics,
}

/// Grid performance metrics
#[derive(Debug, Clone)]
pub struct GridMetrics {
    /// Load factor
    pub load_factor: f64,
    /// Peak demand (MW)
    pub peak_demand: f64,
    /// Total energy generated (MWh)
    pub total_energy: f64,
    /// Renewable penetration
    pub renewable_penetration: f64,
    /// Grid efficiency
    pub efficiency: f64,
    /// Reliability metrics
    pub reliability_score: f64,
}

impl IndustrySolution for GridSolution {
    type Problem = SmartGridOptimization;

    fn from_binary(problem: &Self::Problem, binary_solution: &[i8]) -> ApplicationResult<Self> {
        let precision = 10;
        let mut generation_schedule =
            vec![vec![0.0; problem.num_time_periods]; problem.num_generators];
        let mut var_idx = 0;

        // Decode generation schedule
        for g in 0..problem.num_generators {
            for t in 0..problem.num_time_periods {
                for level in 0..precision {
                    if var_idx < binary_solution.len() && binary_solution[var_idx] == 1 {
                        generation_schedule[g][t] =
                            (level as f64) * problem.generator_capacities[g] / (precision as f64);
                        break;
                    }
                    var_idx += 1;
                }
            }
        }

        // Simple transmission flow calculation (would be more complex in practice)
        let mut transmission_flows = HashMap::new();
        for ((from, to), _) in &problem.transmission_capacities {
            transmission_flows.insert((*from, *to), vec![0.0; problem.num_time_periods]);
        }

        // Storage operations (simplified)
        let storage_operations =
            vec![vec![0.0; problem.num_time_periods]; problem.storage_systems.len()];

        // Calculate metrics
        let total_energy: f64 = generation_schedule.iter().flat_map(|gen| gen.iter()).sum();

        let peak_demand = (0..problem.num_time_periods)
            .map(|t| problem.calculate_total_demand(t))
            .fold(0.0, f64::max);

        let average_demand = total_energy / problem.num_time_periods as f64;
        let load_factor = if peak_demand > 0.0 {
            average_demand / peak_demand
        } else {
            0.0
        };

        let grid_metrics = GridMetrics {
            load_factor,
            peak_demand,
            total_energy,
            renewable_penetration: 0.0, // Simplified
            efficiency: 0.95,           // Simplified
            reliability_score: 0.99,    // Simplified
        };

        let total_cost = problem.calculate_generation_cost(&Self {
            generation_schedule: generation_schedule.clone(),
            transmission_flows: transmission_flows.clone(),
            storage_operations: storage_operations.clone(),
            total_cost: 0.0,
            grid_metrics: grid_metrics.clone(),
        });

        Ok(Self {
            generation_schedule,
            transmission_flows,
            storage_operations,
            total_cost,
            grid_metrics,
        })
    }

    fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("type".to_string(), "Smart Grid Optimization".to_string());
        summary.insert("total_cost".to_string(), format!("${:.2}", self.total_cost));
        summary.insert(
            "peak_demand".to_string(),
            format!("{:.1} MW", self.grid_metrics.peak_demand),
        );
        summary.insert(
            "total_energy".to_string(),
            format!("{:.1} MWh", self.grid_metrics.total_energy),
        );
        summary.insert(
            "load_factor".to_string(),
            format!("{:.1}%", self.grid_metrics.load_factor * 100.0),
        );
        summary.insert(
            "efficiency".to_string(),
            format!("{:.1}%", self.grid_metrics.efficiency * 100.0),
        );
        summary.insert(
            "reliability".to_string(),
            format!("{:.2}%", self.grid_metrics.reliability_score * 100.0),
        );
        summary
    }

    fn metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("total_cost".to_string(), self.total_cost);
        metrics.insert("load_factor".to_string(), self.grid_metrics.load_factor);
        metrics.insert("peak_demand".to_string(), self.grid_metrics.peak_demand);
        metrics.insert("total_energy".to_string(), self.grid_metrics.total_energy);
        metrics.insert(
            "renewable_penetration".to_string(),
            self.grid_metrics.renewable_penetration,
        );
        metrics.insert("efficiency".to_string(), self.grid_metrics.efficiency);
        metrics.insert(
            "reliability_score".to_string(),
            self.grid_metrics.reliability_score,
        );

        // Calculate additional operational metrics
        let num_active_generators = self
            .generation_schedule
            .iter()
            .filter(|schedule| schedule.iter().any(|&gen| gen > 1e-6))
            .count();
        metrics.insert(
            "active_generators".to_string(),
            num_active_generators as f64,
        );

        let max_generation: f64 = self
            .generation_schedule
            .iter()
            .flat_map(|schedule| schedule.iter())
            .fold(0.0, |a, &b| a.max(b));
        metrics.insert("max_generation".to_string(), max_generation);

        metrics
    }

    fn export_format(&self) -> ApplicationResult<String> {
        let mut output = String::new();
        output.push_str("# Smart Grid Optimization Solution\n\n");

        output.push_str("## Solution Summary\n");
        output.push_str(&format!("Total Cost: ${:.2}\n", self.total_cost));
        output.push_str(&format!(
            "Peak Demand: {:.1} MW\n",
            self.grid_metrics.peak_demand
        ));
        output.push_str(&format!(
            "Total Energy: {:.1} MWh\n",
            self.grid_metrics.total_energy
        ));
        output.push_str(&format!(
            "Load Factor: {:.1}%\n",
            self.grid_metrics.load_factor * 100.0
        ));
        output.push_str(&format!(
            "Grid Efficiency: {:.1}%\n",
            self.grid_metrics.efficiency * 100.0
        ));

        output.push_str("\n## Generation Schedule\n");
        for (g, schedule) in self.generation_schedule.iter().enumerate() {
            output.push_str(&format!("Generator {}: ", g + 1));
            for (t, &generation) in schedule.iter().enumerate() {
                if generation > 1e-6 {
                    output.push_str(&format!("T{}: {:.1}MW ", t + 1, generation));
                }
            }
            output.push('\n');
        }

        output.push_str("\n## Grid Performance Metrics\n");
        for (key, value) in self.metrics() {
            output.push_str(&format!("{}: {:.3}\n", key, value));
        }

        Ok(output)
    }
}

/// Renewable Energy Integration Problem
#[derive(Debug, Clone)]
pub struct RenewableEnergyIntegration {
    /// Solar farm locations and capacities
    pub solar_farms: Vec<SolarFarm>,
    /// Wind farm locations and capacities
    pub wind_farms: Vec<WindFarm>,
    /// Battery storage systems
    pub battery_systems: Vec<BatterySystem>,
    /// Demand profiles
    pub demand_profile: Vec<f64>,
    /// Grid connection costs
    pub connection_costs: Vec<f64>,
    /// Curtailment penalty
    pub curtailment_penalty: f64,
}

/// Solar farm specification
#[derive(Debug, Clone)]
pub struct SolarFarm {
    /// Capacity (MW)
    pub capacity: f64,
    /// Solar irradiance profile
    pub irradiance_profile: Vec<f64>,
    /// Efficiency
    pub efficiency: f64,
    /// Installation cost ($/MW)
    pub installation_cost: f64,
}

/// Wind farm specification
#[derive(Debug, Clone)]
pub struct WindFarm {
    /// Capacity (MW)
    pub capacity: f64,
    /// Wind speed profile
    pub wind_profile: Vec<f64>,
    /// Power curve coefficients
    pub power_curve: Vec<f64>,
    /// Installation cost ($/MW)
    pub installation_cost: f64,
}

/// Battery energy storage system
#[derive(Debug, Clone)]
pub struct BatterySystem {
    /// Energy capacity (MWh)
    pub energy_capacity: f64,
    /// Power capacity (MW)
    pub power_capacity: f64,
    /// Round-trip efficiency
    pub efficiency: f64,
    /// Installation cost ($/MWh)
    pub installation_cost: f64,
    /// Cycle life
    pub cycle_life: usize,
}

/// Binary wrapper for Smart Grid Optimization that works with binary solutions
#[derive(Debug, Clone)]
pub struct BinarySmartGridOptimization {
    inner: SmartGridOptimization,
}

impl BinarySmartGridOptimization {
    pub fn new(inner: SmartGridOptimization) -> Self {
        Self { inner }
    }
}

impl OptimizationProblem for BinarySmartGridOptimization {
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
        // Convert binary solution to GridSolution for evaluation
        let grid_solution = GridSolution::from_binary(&self.inner, solution)?;
        self.inner.evaluate_solution(&grid_solution)
    }

    fn is_feasible(&self, solution: &Self::Solution) -> bool {
        // Convert binary solution to GridSolution for feasibility check
        if let Ok(grid_solution) = GridSolution::from_binary(&self.inner, solution) {
            self.inner.is_feasible(&grid_solution)
        } else {
            false
        }
    }
}

/// Create benchmark energy problems
pub fn create_benchmark_problems(
    size: usize,
) -> ApplicationResult<Vec<Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>>>
{
    let mut problems = Vec::new();

    // Problem 1: Small grid optimization
    let generator_capacities = vec![100.0, 150.0, 200.0];
    let generator_costs = vec![30.0, 25.0, 40.0]; // $/MWh
    let demand_forecasts = vec![
        vec![50.0, 80.0, 120.0, 100.0], // Node 1
        vec![30.0, 60.0, 90.0, 70.0],   // Node 2
    ];

    let grid_problem = SmartGridOptimization::new(
        3, // generators
        2, // demand nodes
        generator_capacities,
        generator_costs,
        demand_forecasts,
        4, // time periods
    )?;

    problems.push(Box::new(BinarySmartGridOptimization::new(grid_problem))
        as Box<
            dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>,
        >);

    // Problem 2: Larger grid with storage
    if size >= 5 {
        let large_capacities = (0..size).map(|i| 50.0 + i as f64 * 20.0).collect();
        let large_costs = (0..size).map(|i| 20.0 + i as f64 * 5.0).collect();
        let large_demands = (0..3)
            .map(|_| (0..6).map(|t| 30.0 + t as f64 * 15.0).collect())
            .collect();

        let mut large_grid =
            SmartGridOptimization::new(size, 3, large_capacities, large_costs, large_demands, 6)?;

        // Add storage systems
        for i in 0..2 {
            large_grid.add_storage_system(EnergyStorageSystem {
                capacity: 100.0,
                max_power: 50.0,
                efficiency: 0.90,
                location: i,
                operating_cost: 5.0,
            })?;
        }

        problems.push(Box::new(BinarySmartGridOptimization::new(large_grid))
            as Box<
                dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>,
            >);
    }

    Ok(problems)
}

/// Solve grid optimization using quantum annealing
pub fn solve_grid_optimization(
    problem: &SmartGridOptimization,
    params: Option<AnnealingParams>,
) -> ApplicationResult<GridSolution> {
    // Convert to QUBO
    let (qubo, _var_map) = problem.to_qubo()?;

    // Convert to Ising
    let ising = IsingModel::from_qubo(&qubo);

    // Set up annealing parameters
    let annealing_params = params.unwrap_or_else(|| {
        let mut p = AnnealingParams::default();
        p.num_sweeps = 20000;
        p.num_repetitions = 30;
        p.initial_temperature = 4.0;
        p.final_temperature = 0.001;
        p
    });

    // Solve with classical annealing
    let simulator = ClassicalAnnealingSimulator::new(annealing_params)
        .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;

    let result = simulator
        .solve(&ising)
        .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;

    // Convert solution back to grid solution
    GridSolution::from_binary(problem, &result.best_spins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_optimization_creation() {
        let generator_capacities = vec![100.0, 150.0];
        let generator_costs = vec![30.0, 25.0];
        let demand_forecasts = vec![vec![50.0, 80.0], vec![30.0, 60.0]];

        let grid = SmartGridOptimization::new(
            2,
            2,
            generator_capacities,
            generator_costs,
            demand_forecasts,
            2,
        )
        .unwrap();

        assert_eq!(grid.num_generators, 2);
        assert_eq!(grid.num_demand_nodes, 2);
        assert_eq!(grid.num_time_periods, 2);
    }

    #[test]
    fn test_transmission_line_addition() {
        let mut grid = SmartGridOptimization::new(
            2,
            3,
            vec![100.0, 150.0],
            vec![30.0, 25.0],
            vec![vec![50.0], vec![30.0], vec![20.0]],
            1,
        )
        .unwrap();

        assert!(grid.add_transmission_line(0, 1, 100.0, 2.0).is_ok());
        assert!(grid.add_transmission_line(1, 2, 150.0, 3.0).is_ok());
        assert_eq!(grid.transmission_capacities.len(), 2);
    }

    #[test]
    fn test_storage_system_addition() {
        let mut grid = SmartGridOptimization::new(
            2,
            2,
            vec![100.0, 150.0],
            vec![30.0, 25.0],
            vec![vec![50.0], vec![30.0]],
            1,
        )
        .unwrap();

        let storage = EnergyStorageSystem {
            capacity: 100.0,
            max_power: 50.0,
            efficiency: 0.90,
            location: 0,
            operating_cost: 5.0,
        };

        assert!(grid.add_storage_system(storage).is_ok());
        assert_eq!(grid.storage_systems.len(), 1);
    }

    #[test]
    fn test_demand_calculation() {
        let grid = SmartGridOptimization::new(
            2,
            2,
            vec![100.0, 150.0],
            vec![30.0, 25.0],
            vec![vec![50.0, 80.0], vec![30.0, 60.0]],
            2,
        )
        .unwrap();

        assert_eq!(grid.calculate_total_demand(0), 80.0); // 50 + 30
        assert_eq!(grid.calculate_total_demand(1), 140.0); // 80 + 60
    }

    #[test]
    fn test_grid_validation() {
        let grid = SmartGridOptimization::new(
            2,
            2,
            vec![100.0, 150.0],
            vec![30.0, 25.0],
            vec![vec![50.0, 80.0], vec![30.0, 60.0]],
            2,
        )
        .unwrap();

        assert!(grid.validate().is_ok());

        // Test insufficient capacity
        let insufficient_grid = SmartGridOptimization::new(
            1,
            2,
            vec![50.0], // Not enough for peak demand of 140
            vec![30.0],
            vec![vec![50.0, 80.0], vec![30.0, 60.0]],
            2,
        )
        .unwrap();

        assert!(insufficient_grid.validate().is_err());
    }

    #[test]
    fn test_benchmark_problems() {
        let problems = create_benchmark_problems(5).unwrap();
        assert_eq!(problems.len(), 2);

        for problem in &problems {
            assert!(problem.validate().is_ok());
            let metrics = problem.size_metrics();
            assert!(metrics.contains_key("num_generators"));
            assert!(metrics.contains_key("num_demand_nodes"));
        }
    }

    #[test]
    fn test_solar_farm() {
        let solar = SolarFarm {
            capacity: 100.0,
            irradiance_profile: vec![0.0, 0.3, 0.8, 1.0, 0.9, 0.5, 0.0],
            efficiency: 0.20,
            installation_cost: 1000.0,
        };

        assert_eq!(solar.capacity, 100.0);
        assert_eq!(solar.irradiance_profile.len(), 7);
    }

    #[test]
    fn test_battery_system() {
        let battery = BatterySystem {
            energy_capacity: 50.0,
            power_capacity: 25.0,
            efficiency: 0.95,
            installation_cost: 500.0,
            cycle_life: 5000,
        };

        assert_eq!(battery.energy_capacity, 50.0);
        assert_eq!(battery.efficiency, 0.95);
    }
}
