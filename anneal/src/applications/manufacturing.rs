//! Manufacturing Industry Optimization
//!
//! This module provides optimization solutions for the manufacturing industry,
//! including production scheduling, resource allocation, quality control,
//! and supply chain integration.

use std::collections::HashMap;
use crate::qubo::{QuboFormulation, QuboBuilder};
use crate::ising::IsingModel;
use crate::simulator::{AnnealingParams, ClassicalAnnealingSimulator};
use super::{ApplicationError, ApplicationResult, OptimizationProblem, IndustrySolution, IndustryConstraint, IndustryObjective};

/// Production Scheduling Problem
#[derive(Debug, Clone)]
pub struct ProductionScheduling {
    /// Number of jobs to schedule
    pub num_jobs: usize,
    /// Number of machines available
    pub num_machines: usize,
    /// Processing times for each job on each machine
    pub processing_times: Vec<Vec<f64>>,
    /// Setup times between jobs on machines
    pub setup_times: Vec<Vec<Vec<f64>>>,
    /// Job priorities
    pub job_priorities: Vec<f64>,
    /// Due dates for jobs
    pub due_dates: Vec<f64>,
    /// Machine capabilities (which jobs can run on which machines)
    pub machine_capabilities: Vec<Vec<bool>>,
    /// Resource requirements for each job
    pub resource_requirements: Vec<HashMap<String, f64>>,
    /// Available resources
    pub available_resources: HashMap<String, f64>,
    /// Quality constraints
    pub quality_constraints: Vec<IndustryConstraint>,
}

impl ProductionScheduling {
    /// Create a new production scheduling problem
    pub fn new(
        num_jobs: usize,
        num_machines: usize,
        processing_times: Vec<Vec<f64>>,
        due_dates: Vec<f64>,
    ) -> ApplicationResult<Self> {
        if processing_times.len() != num_jobs {
            return Err(ApplicationError::InvalidConfiguration(
                "Processing times must match number of jobs".to_string()
            ));
        }
        
        for (i, times) in processing_times.iter().enumerate() {
            if times.len() != num_machines {
                return Err(ApplicationError::InvalidConfiguration(
                    format!("Processing times for job {} must match number of machines", i)
                ));
            }
        }
        
        if due_dates.len() != num_jobs {
            return Err(ApplicationError::InvalidConfiguration(
                "Due dates must match number of jobs".to_string()
            ));
        }
        
        Ok(Self {
            num_jobs,
            num_machines,
            processing_times,
            setup_times: vec![vec![vec![0.0; num_jobs]; num_jobs]; num_machines],
            job_priorities: vec![1.0; num_jobs],
            due_dates,
            machine_capabilities: vec![vec![true; num_machines]; num_jobs],
            resource_requirements: vec![HashMap::new(); num_jobs],
            available_resources: HashMap::new(),
            quality_constraints: Vec::new(),
        })
    }
    
    /// Set setup times between jobs on machines
    pub fn set_setup_times(&mut self, setup_times: Vec<Vec<Vec<f64>>>) -> ApplicationResult<()> {
        if setup_times.len() != self.num_machines {
            return Err(ApplicationError::InvalidConfiguration(
                "Setup times must match number of machines".to_string()
            ));
        }
        
        self.setup_times = setup_times;
        Ok(())
    }
    
    /// Set machine capabilities
    pub fn set_machine_capabilities(&mut self, capabilities: Vec<Vec<bool>>) -> ApplicationResult<()> {
        if capabilities.len() != self.num_jobs {
            return Err(ApplicationError::InvalidConfiguration(
                "Machine capabilities must match number of jobs".to_string()
            ));
        }
        
        self.machine_capabilities = capabilities;
        Ok(())
    }
    
    /// Add resource requirement for a job
    pub fn add_resource_requirement(&mut self, job: usize, resource: String, amount: f64) -> ApplicationResult<()> {
        if job >= self.num_jobs {
            return Err(ApplicationError::InvalidConfiguration(
                "Job index out of bounds".to_string()
            ));
        }
        
        self.resource_requirements[job].insert(resource, amount);
        Ok(())
    }
    
    /// Set available resource capacity
    pub fn set_resource_capacity(&mut self, resource: String, capacity: f64) {
        self.available_resources.insert(resource, capacity);
    }
    
    /// Calculate makespan for a schedule
    pub fn calculate_makespan(&self, schedule: &ProductionSchedule) -> f64 {
        let mut machine_finish_times = vec![0.0; self.num_machines];
        
        for assignment in &schedule.job_assignments {
            let job = assignment.job_id;
            let machine = assignment.machine_id;
            let start_time = assignment.start_time;
            
            let processing_time = self.processing_times[job][machine];
            let finish_time = start_time + processing_time;
            
            machine_finish_times[machine] = machine_finish_times[machine].max(finish_time);
        }
        
        machine_finish_times.iter().fold(0.0, |a, &b| a.max(b))
    }
    
    /// Calculate total tardiness
    pub fn calculate_tardiness(&self, schedule: &ProductionSchedule) -> f64 {
        let mut total_tardiness = 0.0;
        
        for assignment in &schedule.job_assignments {
            let job = assignment.job_id;
            let completion_time = assignment.start_time + self.processing_times[job][assignment.machine_id];
            let tardiness = (completion_time - self.due_dates[job]).max(0.0);
            total_tardiness += tardiness * self.job_priorities[job];
        }
        
        total_tardiness
    }
    
    /// Calculate resource utilization
    pub fn calculate_resource_utilization(&self, schedule: &ProductionSchedule) -> HashMap<String, f64> {
        let mut utilization = HashMap::new();
        
        for (resource, &capacity) in &self.available_resources {
            let mut total_usage = 0.0;
            
            for assignment in &schedule.job_assignments {
                let job = assignment.job_id;
                if let Some(&usage) = self.resource_requirements[job].get(resource) {
                    total_usage += usage;
                }
            }
            
            utilization.insert(resource.clone(), total_usage / capacity);
        }
        
        utilization
    }
}

impl OptimizationProblem for ProductionScheduling {
    type Solution = ProductionSchedule;
    type ObjectiveValue = f64;
    
    fn description(&self) -> String {
        format!(
            "Production scheduling with {} jobs and {} machines",
            self.num_jobs,
            self.num_machines
        )
    }
    
    fn size_metrics(&self) -> HashMap<String, usize> {
        let mut metrics = HashMap::new();
        metrics.insert("num_jobs".to_string(), self.num_jobs);
        metrics.insert("num_machines".to_string(), self.num_machines);
        metrics.insert("num_resources".to_string(), self.available_resources.len());
        metrics.insert("num_constraints".to_string(), self.quality_constraints.len());
        metrics
    }
    
    fn validate(&self) -> ApplicationResult<()> {
        if self.num_jobs == 0 {
            return Err(ApplicationError::DataValidationError(
                "At least one job required".to_string()
            ));
        }
        
        if self.num_machines == 0 {
            return Err(ApplicationError::DataValidationError(
                "At least one machine required".to_string()
            ));
        }
        
        // Check that each job can be processed on at least one machine
        for (job, capabilities) in self.machine_capabilities.iter().enumerate() {
            if !capabilities.iter().any(|&capable| capable) {
                return Err(ApplicationError::DataValidationError(
                    format!("Job {} cannot be processed on any machine", job)
                ));
            }
        }
        
        // Check positive processing times
        for (job, times) in self.processing_times.iter().enumerate() {
            for (machine, &time) in times.iter().enumerate() {
                if time < 0.0 {
                    return Err(ApplicationError::DataValidationError(
                        format!("Negative processing time for job {} on machine {}", job, machine)
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    fn to_qubo(&self) -> ApplicationResult<QuboFormulation> {
        let mut builder = QuboBuilder::new();
        let time_horizon = 100; // Discretized time slots
        
        // Binary variables: x[j][m][t] = 1 if job j starts on machine m at time t
        let mut var_counter = 0;
        let mut var_map = HashMap::new();
        
        for job in 0..self.num_jobs {
            for machine in 0..self.num_machines {
                if self.machine_capabilities[job][machine] {
                    for time in 0..time_horizon {
                        var_map.insert((job, machine, time), var_counter);
                        var_counter += 1;
                    }
                }
            }
        }
        
        // Objective: minimize makespan + weighted tardiness
        for job in 0..self.num_jobs {
            for machine in 0..self.num_machines {
                if self.machine_capabilities[job][machine] {
                    for time in 0..time_horizon {
                        let var_idx = var_map[&(job, machine, time)];
                        let processing_time = self.processing_times[job][machine];
                        let completion_time = time as f64 + processing_time;
                        
                        // Makespan penalty
                        let makespan_penalty = completion_time * 0.1;
                        builder.add_bias(var_idx, makespan_penalty);
                        
                        // Tardiness penalty
                        let tardiness = (completion_time - self.due_dates[job]).max(0.0);
                        let tardiness_penalty = tardiness * self.job_priorities[job];
                        builder.add_bias(var_idx, tardiness_penalty);
                    }
                }
            }
        }
        
        // Constraint: each job scheduled exactly once
        let constraint_penalty = 10000.0;
        for job in 0..self.num_jobs {
            let mut job_vars = Vec::new();
            
            for machine in 0..self.num_machines {
                if self.machine_capabilities[job][machine] {
                    for time in 0..time_horizon {
                        job_vars.push(var_map[&(job, machine, time)]);
                    }
                }
            }
            
            // Penalty for not scheduling exactly once
            for &var1 in &job_vars {
                builder.add_bias(var1, -constraint_penalty);
                for &var2 in &job_vars {
                    if var1 != var2 {
                        builder.add_coupling(var1, var2, constraint_penalty);
                    }
                }
            }
        }
        
        // Constraint: no overlapping jobs on same machine
        for machine in 0..self.num_machines {
            for time in 0..time_horizon {
                let mut overlapping_vars = Vec::new();
                
                for job in 0..self.num_jobs {
                    if self.machine_capabilities[job][machine] {
                        let processing_time = self.processing_times[job][machine] as usize;
                        
                        // Check if job would be running at this time
                        for start_time in 0..=time {
                            if start_time + processing_time > time {
                                if let Some(&var_idx) = var_map.get(&(job, machine, start_time)) {
                                    overlapping_vars.push(var_idx);
                                }
                            }
                        }
                    }
                }
                
                // Penalty for multiple jobs running simultaneously
                if overlapping_vars.len() > 1 {
                    for &var1 in &overlapping_vars {
                        for &var2 in &overlapping_vars {
                            if var1 != var2 {
                                builder.add_coupling(var1, var2, constraint_penalty);
                            }
                        }
                    }
                }
            }
        }
        
        builder.build()
    }
    
    fn evaluate_solution(&self, solution: &Self::Solution) -> ApplicationResult<Self::ObjectiveValue> {
        let makespan = self.calculate_makespan(solution);
        let tardiness = self.calculate_tardiness(solution);
        
        // Combined objective: makespan + weighted tardiness
        Ok(makespan + tardiness)
    }
    
    fn is_feasible(&self, solution: &Self::Solution) -> bool {
        // Check that all jobs are assigned
        if solution.job_assignments.len() != self.num_jobs {
            return false;
        }
        
        // Check machine capabilities
        for assignment in &solution.job_assignments {
            if !self.machine_capabilities[assignment.job_id][assignment.machine_id] {
                return false;
            }
        }
        
        // Check for overlapping jobs on same machine
        let mut machine_schedules: Vec<Vec<(f64, f64)>> = vec![Vec::new(); self.num_machines];
        
        for assignment in &solution.job_assignments {
            let start = assignment.start_time;
            let end = start + self.processing_times[assignment.job_id][assignment.machine_id];
            machine_schedules[assignment.machine_id].push((start, end));
        }
        
        for schedule in &machine_schedules {
            let mut sorted_schedule = schedule.clone();
            sorted_schedule.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            
            for i in 1..sorted_schedule.len() {
                if sorted_schedule[i].0 < sorted_schedule[i-1].1 {
                    return false; // Overlap detected
                }
            }
        }
        
        // Check resource constraints
        let utilization = self.calculate_resource_utilization(solution);
        for (_, &util) in &utilization {
            if util > 1.0 {
                return false;
            }
        }
        
        true
    }
}

/// Production schedule solution
#[derive(Debug, Clone)]
pub struct ProductionSchedule {
    /// Job assignments
    pub job_assignments: Vec<JobAssignment>,
    /// Total makespan
    pub makespan: f64,
    /// Total tardiness
    pub total_tardiness: f64,
    /// Resource utilization
    pub resource_utilization: HashMap<String, f64>,
    /// Performance metrics
    pub metrics: SchedulingMetrics,
}

/// Individual job assignment
#[derive(Debug, Clone)]
pub struct JobAssignment {
    /// Job identifier
    pub job_id: usize,
    /// Assigned machine
    pub machine_id: usize,
    /// Start time
    pub start_time: f64,
    /// Priority
    pub priority: f64,
}

/// Scheduling performance metrics
#[derive(Debug, Clone)]
pub struct SchedulingMetrics {
    /// Average flow time
    pub avg_flow_time: f64,
    /// Machine utilization rates
    pub machine_utilization: Vec<f64>,
    /// On-time delivery rate
    pub on_time_rate: f64,
    /// Total setup time
    pub total_setup_time: f64,
    /// Efficiency score
    pub efficiency_score: f64,
}

impl IndustrySolution for ProductionSchedule {
    type Problem = ProductionScheduling;
    
    fn from_binary(problem: &Self::Problem, binary_solution: &[i8]) -> ApplicationResult<Self> {
        let time_horizon = 100;
        let mut job_assignments = Vec::new();
        let mut var_idx = 0;
        
        // Decode job assignments from binary solution
        for job in 0..problem.num_jobs {
            for machine in 0..problem.num_machines {
                if problem.machine_capabilities[job][machine] {
                    for time in 0..time_horizon {
                        if var_idx < binary_solution.len() && binary_solution[var_idx] == 1 {
                            job_assignments.push(JobAssignment {
                                job_id: job,
                                machine_id: machine,
                                start_time: time as f64,
                                priority: problem.job_priorities[job],
                            });
                        }
                        var_idx += 1;
                    }
                }
            }
        }
        
        // Calculate metrics
        let makespan = problem.calculate_makespan(&ProductionSchedule {
            job_assignments: job_assignments.clone(),
            makespan: 0.0,
            total_tardiness: 0.0,
            resource_utilization: HashMap::new(),
            metrics: SchedulingMetrics {
                avg_flow_time: 0.0,
                machine_utilization: Vec::new(),
                on_time_rate: 0.0,
                total_setup_time: 0.0,
                efficiency_score: 0.0,
            },
        });
        
        let total_tardiness = problem.calculate_tardiness(&ProductionSchedule {
            job_assignments: job_assignments.clone(),
            makespan,
            total_tardiness: 0.0,
            resource_utilization: HashMap::new(),
            metrics: SchedulingMetrics {
                avg_flow_time: 0.0,
                machine_utilization: Vec::new(),
                on_time_rate: 0.0,
                total_setup_time: 0.0,
                efficiency_score: 0.0,
            },
        });
        
        // Calculate machine utilization
        let mut machine_utilization = vec![0.0; problem.num_machines];
        for assignment in &job_assignments {
            let processing_time = problem.processing_times[assignment.job_id][assignment.machine_id];
            machine_utilization[assignment.machine_id] += processing_time;
        }
        
        for util in &mut machine_utilization {
            *util /= makespan.max(1.0);
        }
        
        // Calculate on-time delivery rate
        let mut on_time_count = 0;
        for assignment in &job_assignments {
            let completion_time = assignment.start_time + 
                problem.processing_times[assignment.job_id][assignment.machine_id];
            if completion_time <= problem.due_dates[assignment.job_id] {
                on_time_count += 1;
            }
        }
        let on_time_rate = on_time_count as f64 / problem.num_jobs as f64;
        
        let resource_utilization = problem.calculate_resource_utilization(&ProductionSchedule {
            job_assignments: job_assignments.clone(),
            makespan,
            total_tardiness,
            resource_utilization: HashMap::new(),
            metrics: SchedulingMetrics {
                avg_flow_time: 0.0,
                machine_utilization: machine_utilization.clone(),
                on_time_rate,
                total_setup_time: 0.0,
                efficiency_score: 0.0,
            },
        });
        
        let metrics = SchedulingMetrics {
            avg_flow_time: makespan / problem.num_jobs as f64,
            machine_utilization,
            on_time_rate,
            total_setup_time: 0.0, // Simplified
            efficiency_score: on_time_rate * (1.0 - total_tardiness / (makespan * problem.num_jobs as f64)),
        };
        
        Ok(ProductionSchedule {
            job_assignments,
            makespan,
            total_tardiness,
            resource_utilization,
            metrics,
        })
    }
    
    fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("type".to_string(), "Production Scheduling".to_string());
        summary.insert("num_jobs".to_string(), self.job_assignments.len().to_string());
        summary.insert("makespan".to_string(), format!("{:.2} hours", self.makespan));
        summary.insert("total_tardiness".to_string(), format!("{:.2} hours", self.total_tardiness));
        summary.insert("on_time_rate".to_string(), format!("{:.1}%", self.metrics.on_time_rate * 100.0));
        summary.insert("efficiency_score".to_string(), format!("{:.3}", self.metrics.efficiency_score));
        
        let avg_machine_util = self.metrics.machine_utilization.iter().sum::<f64>() / 
            self.metrics.machine_utilization.len() as f64;
        summary.insert("avg_machine_utilization".to_string(), format!("{:.1}%", avg_machine_util * 100.0));
        
        summary
    }
    
    fn metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("makespan".to_string(), self.makespan);
        metrics.insert("total_tardiness".to_string(), self.total_tardiness);
        metrics.insert("avg_flow_time".to_string(), self.metrics.avg_flow_time);
        metrics.insert("on_time_rate".to_string(), self.metrics.on_time_rate);
        metrics.insert("efficiency_score".to_string(), self.metrics.efficiency_score);
        
        for (i, &util) in self.metrics.machine_utilization.iter().enumerate() {
            metrics.insert(format!("machine_{}_utilization", i), util);
        }
        
        for (resource, &util) in &self.resource_utilization {
            metrics.insert(format!("resource_{}_utilization", resource), util);
        }
        
        metrics
    }
    
    fn export_format(&self) -> ApplicationResult<String> {
        let mut output = String::new();
        output.push_str("# Production Schedule Report\n\n");
        
        output.push_str("## Schedule Summary\n");
        output.push_str(&format!("Makespan: {:.2} hours\n", self.makespan));
        output.push_str(&format!("Total Tardiness: {:.2} hours\n", self.total_tardiness));
        output.push_str(&format!("On-time Delivery Rate: {:.1}%\n", self.metrics.on_time_rate * 100.0));
        output.push_str(&format!("Efficiency Score: {:.3}\n", self.metrics.efficiency_score));
        
        output.push_str("\n## Job Assignments\n");
        let mut sorted_assignments = self.job_assignments.clone();
        sorted_assignments.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());
        
        for assignment in &sorted_assignments {
            output.push_str(&format!(
                "Job {}: Machine {} at {:.1}h (Priority: {:.1})\n",
                assignment.job_id,
                assignment.machine_id,
                assignment.start_time,
                assignment.priority
            ));
        }
        
        output.push_str("\n## Machine Utilization\n");
        for (i, &util) in self.metrics.machine_utilization.iter().enumerate() {
            output.push_str(&format!("Machine {}: {:.1}%\n", i, util * 100.0));
        }
        
        output.push_str("\n## Resource Utilization\n");
        for (resource, &util) in &self.resource_utilization {
            output.push_str(&format!("{}: {:.1}%\n", resource, util * 100.0));
        }
        
        Ok(output)
    }
}

/// Quality Control Optimization Problem
#[derive(Debug, Clone)]
pub struct QualityControlOptimization {
    /// Number of inspection stations
    pub num_stations: usize,
    /// Number of quality parameters
    pub num_parameters: usize,
    /// Inspection costs per station
    pub inspection_costs: Vec<f64>,
    /// Detection probabilities for each parameter at each station
    pub detection_probabilities: Vec<Vec<f64>>,
    /// Defect costs if not caught
    pub defect_costs: Vec<f64>,
    /// Station capacities
    pub station_capacities: Vec<f64>,
    /// Quality targets
    pub quality_targets: Vec<f64>,
    /// Process flow constraints
    pub flow_constraints: Vec<IndustryConstraint>,
}

impl QualityControlOptimization {
    /// Create new quality control optimization problem
    pub fn new(
        num_stations: usize,
        num_parameters: usize,
        inspection_costs: Vec<f64>,
        detection_probabilities: Vec<Vec<f64>>,
        defect_costs: Vec<f64>,
    ) -> ApplicationResult<Self> {
        if detection_probabilities.len() != num_stations {
            return Err(ApplicationError::InvalidConfiguration(
                "Detection probabilities must match number of stations".to_string()
            ));
        }
        
        Ok(Self {
            num_stations,
            num_parameters,
            inspection_costs,
            detection_probabilities,
            defect_costs,
            station_capacities: vec![100.0; num_stations],
            quality_targets: vec![0.95; num_parameters],
            flow_constraints: Vec::new(),
        })
    }
    
    /// Calculate total quality score
    pub fn calculate_quality_score(&self, allocation: &[bool]) -> f64 {
        let mut total_score = 0.0;
        
        for param in 0..self.num_parameters {
            let mut detection_prob = 0.0;
            
            for (station, &allocated) in allocation.iter().enumerate() {
                if allocated {
                    detection_prob += self.detection_probabilities[station][param];
                }
            }
            
            detection_prob = detection_prob.min(1.0);
            total_score += detection_prob * self.quality_targets[param];
        }
        
        total_score / self.num_parameters as f64
    }
}

/// Manufacturing Resource Planning (MRP) Problem
#[derive(Debug, Clone)]
pub struct ManufacturingResourcePlanning {
    /// Bill of materials
    pub bill_of_materials: HashMap<String, Vec<(String, f64)>>,
    /// Lead times for materials
    pub lead_times: HashMap<String, f64>,
    /// Inventory levels
    pub inventory_levels: HashMap<String, f64>,
    /// Demand forecast
    pub demand_forecast: HashMap<String, Vec<f64>>,
    /// Supplier capabilities
    pub supplier_capabilities: HashMap<String, Vec<String>>,
    /// Cost structures
    pub cost_structure: HashMap<String, f64>,
}

/// Create benchmark manufacturing problems
pub fn create_benchmark_problems(size: usize) -> ApplicationResult<Vec<Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>>> {
    let mut problems = Vec::new();
    
    // Problem 1: Small production scheduling
    let processing_times = vec![
        vec![5.0, 8.0, 3.0], // Job 0
        vec![7.0, 4.0, 6.0], // Job 1
        vec![6.0, 9.0, 2.0], // Job 2
    ];
    let due_dates = vec![15.0, 20.0, 18.0];
    
    let mut small_scheduling = ProductionScheduling::new(3, 3, processing_times, due_dates)?;
    small_scheduling.job_priorities = vec![1.0, 2.0, 1.5];
    
    problems.push(Box::new(small_scheduling) as Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>);
    
    // Problem 2: Larger scheduling problem
    if size >= 5 {
        let large_processing_times: Vec<Vec<f64>> = (0..size)
            .map(|i| (0..3).map(|j| 3.0 + ((i + j) as f64 * 2.0) % 8.0).collect())
            .collect();
        let large_due_dates: Vec<f64> = (0..size)
            .map(|i| 20.0 + (i as f64 * 5.0))
            .collect();
        
        let large_scheduling = ProductionScheduling::new(size, 3, large_processing_times, large_due_dates)?;
        problems.push(Box::new(large_scheduling) as Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>);
    }
    
    // Problem 3: Quality control optimization
    let inspection_costs = vec![50.0, 75.0, 100.0];
    let detection_probs = vec![
        vec![0.8, 0.6, 0.9], // Station 0
        vec![0.7, 0.9, 0.7], // Station 1
        vec![0.9, 0.8, 0.8], // Station 2
    ];
    let defect_costs = vec![1000.0, 1500.0, 800.0];
    
    let quality_control = QualityControlOptimization::new(3, 3, inspection_costs, detection_probs, defect_costs)?;
    // Note: QualityControlOptimization would need to implement OptimizationProblem trait
    
    Ok(problems)
}

/// Solve production scheduling using quantum annealing
pub fn solve_production_scheduling(
    problem: &ProductionScheduling,
    params: Option<AnnealingParams>,
) -> ApplicationResult<ProductionSchedule> {
    // Convert to QUBO
    let qubo = problem.to_qubo()?;
    
    // Convert to Ising
    let ising = IsingModel::from_qubo(&qubo)?;
    
    // Set up annealing parameters
    let annealing_params = params.unwrap_or_else(|| {
        let mut p = AnnealingParams::default();
        p.num_sweeps = 25000;
        p.num_repetitions = 30;
        p.initial_temperature = 5.0;
        p.final_temperature = 0.001;
        p
    });
    
    // Solve with classical annealing
    let simulator = ClassicalAnnealingSimulator::new(annealing_params)
        .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;
    
    let result = simulator.solve(&ising)
        .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;
    
    // Convert solution back to production schedule
    ProductionSchedule::from_binary(problem, &result.best_spins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_scheduling_creation() {
        let processing_times = vec![
            vec![5.0, 8.0],
            vec![7.0, 4.0],
        ];
        let due_dates = vec![15.0, 20.0];
        
        let scheduling = ProductionScheduling::new(2, 2, processing_times, due_dates).unwrap();
        assert_eq!(scheduling.num_jobs, 2);
        assert_eq!(scheduling.num_machines, 2);
    }
    
    #[test]
    fn test_makespan_calculation() {
        let processing_times = vec![
            vec![5.0, 8.0],
            vec![7.0, 4.0],
        ];
        let due_dates = vec![15.0, 20.0];
        
        let scheduling = ProductionScheduling::new(2, 2, processing_times, due_dates).unwrap();
        
        let schedule = ProductionSchedule {
            job_assignments: vec![
                JobAssignment { job_id: 0, machine_id: 0, start_time: 0.0, priority: 1.0 },
                JobAssignment { job_id: 1, machine_id: 1, start_time: 0.0, priority: 1.0 },
            ],
            makespan: 0.0,
            total_tardiness: 0.0,
            resource_utilization: HashMap::new(),
            metrics: SchedulingMetrics {
                avg_flow_time: 0.0,
                machine_utilization: Vec::new(),
                on_time_rate: 0.0,
                total_setup_time: 0.0,
                efficiency_score: 0.0,
            },
        };
        
        let makespan = scheduling.calculate_makespan(&schedule);
        assert_eq!(makespan, 7.0); // max(5.0, 7.0)
    }
    
    #[test]
    fn test_tardiness_calculation() {
        let processing_times = vec![
            vec![10.0],
            vec![15.0],
        ];
        let due_dates = vec![8.0, 12.0];
        
        let mut scheduling = ProductionScheduling::new(2, 1, processing_times, due_dates).unwrap();
        scheduling.job_priorities = vec![1.0, 2.0];
        
        let schedule = ProductionSchedule {
            job_assignments: vec![
                JobAssignment { job_id: 0, machine_id: 0, start_time: 0.0, priority: 1.0 },
                JobAssignment { job_id: 1, machine_id: 0, start_time: 10.0, priority: 2.0 },
            ],
            makespan: 0.0,
            total_tardiness: 0.0,
            resource_utilization: HashMap::new(),
            metrics: SchedulingMetrics {
                avg_flow_time: 0.0,
                machine_utilization: Vec::new(),
                on_time_rate: 0.0,
                total_setup_time: 0.0,
                efficiency_score: 0.0,
            },
        };
        
        let tardiness = scheduling.calculate_tardiness(&schedule);
        // Job 0: completion at 10, due at 8, tardiness = 2 * 1.0 = 2.0
        // Job 1: completion at 25, due at 12, tardiness = 13 * 2.0 = 26.0
        // Total: 28.0
        assert_eq!(tardiness, 28.0);
    }
    
    #[test]
    fn test_quality_control_creation() {
        let costs = vec![50.0, 75.0];
        let detection_probs = vec![
            vec![0.8, 0.6],
            vec![0.7, 0.9],
        ];
        let defect_costs = vec![1000.0, 1500.0];
        
        let quality_control = QualityControlOptimization::new(2, 2, costs, detection_probs, defect_costs).unwrap();
        assert_eq!(quality_control.num_stations, 2);
        assert_eq!(quality_control.num_parameters, 2);
    }
    
    #[test]
    fn test_quality_score_calculation() {
        let costs = vec![50.0, 75.0];
        let detection_probs = vec![
            vec![0.8, 0.6],
            vec![0.7, 0.9],
        ];
        let defect_costs = vec![1000.0, 1500.0];
        
        let quality_control = QualityControlOptimization::new(2, 2, costs, detection_probs, defect_costs).unwrap();
        
        let allocation = vec![true, false]; // Only use station 0
        let score = quality_control.calculate_quality_score(&allocation);
        
        // Expected: (0.8 * 0.95 + 0.6 * 0.95) / 2 = (0.76 + 0.57) / 2 = 0.665
        assert!((score - 0.665).abs() < 1e-6);
    }
    
    #[test]
    fn test_benchmark_problems() {
        let problems = create_benchmark_problems(4).unwrap();
        assert_eq!(problems.len(), 2);
        
        for problem in &problems {
            assert!(problem.validate().is_ok());
            let metrics = problem.size_metrics();
            assert!(metrics.contains_key("num_jobs"));
            assert!(metrics.contains_key("num_machines"));
        }
    }
}