//! Healthcare Industry Optimization
//!
//! This module provides optimization solutions for the healthcare industry,
//! including resource allocation, patient scheduling, treatment planning,
//! and supply chain optimization for medical facilities.

use std::collections::HashMap;
use crate::qubo::{QuboFormulation, QuboBuilder};
use crate::ising::IsingModel;
use crate::simulator::{AnnealingParams, ClassicalAnnealingSimulator};
use super::{ApplicationError, ApplicationResult, OptimizationProblem, IndustrySolution, IndustryConstraint, IndustryObjective};

/// Medical Resource Allocation Problem
#[derive(Debug, Clone)]
pub struct MedicalResourceAllocation {
    /// Number of hospitals/clinics
    pub num_facilities: usize,
    /// Number of medical departments
    pub num_departments: usize,
    /// Number of resource types (staff, equipment, beds)
    pub num_resource_types: usize,
    /// Available resources at each facility
    pub available_resources: Vec<Vec<f64>>,
    /// Resource requirements by department
    pub resource_requirements: Vec<Vec<f64>>,
    /// Patient demand forecasts by facility and department
    pub patient_demands: Vec<Vec<f64>>,
    /// Resource utilization costs
    pub resource_costs: Vec<Vec<f64>>,
    /// Quality of care metrics
    pub quality_targets: Vec<f64>,
    /// Emergency capacity requirements
    pub emergency_reserves: Vec<f64>,
    /// Resource sharing constraints between facilities
    pub sharing_constraints: Vec<IndustryConstraint>,
}

impl MedicalResourceAllocation {
    /// Create a new medical resource allocation problem
    pub fn new(
        num_facilities: usize,
        num_departments: usize,
        num_resource_types: usize,
        available_resources: Vec<Vec<f64>>,
        patient_demands: Vec<Vec<f64>>,
    ) -> ApplicationResult<Self> {
        if available_resources.len() != num_facilities {
            return Err(ApplicationError::InvalidConfiguration(
                "Available resources must match number of facilities".to_string()
            ));
        }
        
        if patient_demands.len() != num_facilities {
            return Err(ApplicationError::InvalidConfiguration(
                "Patient demands must match number of facilities".to_string()
            ));
        }
        
        Ok(Self {
            num_facilities,
            num_departments,
            num_resource_types,
            available_resources,
            resource_requirements: vec![vec![1.0; num_resource_types]; num_departments],
            patient_demands,
            resource_costs: vec![vec![1.0; num_resource_types]; num_facilities],
            quality_targets: vec![0.95; num_departments],
            emergency_reserves: vec![0.1; num_resource_types], // 10% emergency reserve
            sharing_constraints: Vec::new(),
        })
    }
    
    /// Set resource requirements for departments
    pub fn set_resource_requirements(&mut self, requirements: Vec<Vec<f64>>) -> ApplicationResult<()> {
        if requirements.len() != self.num_departments {
            return Err(ApplicationError::InvalidConfiguration(
                "Resource requirements must match number of departments".to_string()
            ));
        }
        
        self.resource_requirements = requirements;
        Ok(())
    }
    
    /// Calculate resource utilization rate
    pub fn calculate_utilization(&self, allocation: &MedicalResourceSolution) -> Vec<f64> {
        let mut utilization = vec![0.0; self.num_resource_types];
        
        for facility in 0..self.num_facilities {
            for dept in 0..self.num_departments {
                for resource in 0..self.num_resource_types {
                    let allocated = allocation.allocations[facility][dept][resource];
                    let available = self.available_resources[facility][resource];
                    
                    if available > 0.0 {
                        utilization[resource] += allocated / available;
                    }
                }
            }
        }
        
        // Average utilization across facilities
        for util in &mut utilization {
            *util /= self.num_facilities as f64;
        }
        
        utilization
    }
    
    /// Calculate patient satisfaction score
    pub fn calculate_patient_satisfaction(&self, allocation: &MedicalResourceSolution) -> f64 {
        let mut total_satisfaction = 0.0;
        let mut total_patients = 0.0;
        
        for facility in 0..self.num_facilities {
            for dept in 0..self.num_departments {
                let demand = self.patient_demands[facility][dept];
                let capacity = self.calculate_department_capacity(facility, dept, allocation);
                
                let service_ratio = if demand > 0.0 {
                    (capacity / demand).min(1.0)
                } else {
                    1.0
                };
                
                total_satisfaction += service_ratio * demand;
                total_patients += demand;
            }
        }
        
        if total_patients > 0.0 {
            total_satisfaction / total_patients
        } else {
            1.0
        }
    }
    
    /// Calculate department capacity based on resource allocation
    fn calculate_department_capacity(&self, facility: usize, dept: usize, allocation: &MedicalResourceSolution) -> f64 {
        let mut capacity = f64::INFINITY;
        
        for resource in 0..self.num_resource_types {
            let allocated = allocation.allocations[facility][dept][resource];
            let required = self.resource_requirements[dept][resource];
            
            if required > 0.0 {
                capacity = capacity.min(allocated / required);
            }
        }
        
        if capacity == f64::INFINITY {
            0.0
        } else {
            capacity
        }
    }
}

impl OptimizationProblem for MedicalResourceAllocation {
    type Solution = MedicalResourceSolution;
    type ObjectiveValue = f64;
    
    fn description(&self) -> String {
        format!(
            "Medical resource allocation for {} facilities, {} departments, {} resource types",
            self.num_facilities,
            self.num_departments,
            self.num_resource_types
        )
    }
    
    fn size_metrics(&self) -> HashMap<String, usize> {
        let mut metrics = HashMap::new();
        metrics.insert("num_facilities".to_string(), self.num_facilities);
        metrics.insert("num_departments".to_string(), self.num_departments);
        metrics.insert("num_resource_types".to_string(), self.num_resource_types);
        metrics.insert("num_constraints".to_string(), self.sharing_constraints.len());
        metrics
    }
    
    fn validate(&self) -> ApplicationResult<()> {
        if self.num_facilities == 0 {
            return Err(ApplicationError::DataValidationError(
                "At least one facility required".to_string()
            ));
        }
        
        if self.num_departments == 0 {
            return Err(ApplicationError::DataValidationError(
                "At least one department required".to_string()
            ));
        }
        
        if self.num_resource_types == 0 {
            return Err(ApplicationError::DataValidationError(
                "At least one resource type required".to_string()
            ));
        }
        
        // Check positive resource availability
        for facility in &self.available_resources {
            for &resource in facility {
                if resource < 0.0 {
                    return Err(ApplicationError::DataValidationError(
                        "Available resources must be non-negative".to_string()
                    ));
                }
            }
        }
        
        // Check positive patient demands
        for facility in &self.patient_demands {
            for &demand in facility {
                if demand < 0.0 {
                    return Err(ApplicationError::DataValidationError(
                        "Patient demands must be non-negative".to_string()
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    fn to_qubo(&self) -> ApplicationResult<QuboFormulation> {
        let mut builder = QuboBuilder::new();
        let precision = 20; // Discretization precision
        
        // Binary variables: x[f][d][r][l] = 1 if facility f allocates level l of resource r to department d
        let mut var_counter = 0;
        let mut var_map = HashMap::new();
        
        for facility in 0..self.num_facilities {
            for dept in 0..self.num_departments {
                for resource in 0..self.num_resource_types {
                    for level in 0..precision {
                        var_map.insert((facility, dept, resource, level), var_counter);
                        var_counter += 1;
                    }
                }
            }
        }
        
        // Objective: maximize patient satisfaction - cost penalty
        for facility in 0..self.num_facilities {
            for dept in 0..self.num_departments {
                let demand = self.patient_demands[facility][dept];
                
                for resource in 0..self.num_resource_types {
                    for level in 0..precision {
                        let allocation = (level as f64) / (precision as f64);
                        let var_idx = var_map[&(facility, dept, resource, level)];
                        
                        // Service benefit (maximize patient care)
                        let required = self.resource_requirements[dept][resource];
                        let capacity_contribution = if required > 0.0 {
                            allocation / required
                        } else {
                            0.0
                        };
                        let service_benefit = -demand * capacity_contribution * 100.0; // Negative for maximization
                        builder.add_bias(var_idx, service_benefit);
                        
                        // Cost penalty
                        let cost = allocation * self.available_resources[facility][resource] * 
                            self.resource_costs[facility][resource];
                        builder.add_bias(var_idx, cost);
                    }
                }
            }
        }
        
        // Constraint: each resource-department pair gets exactly one allocation level
        let constraint_penalty = 10000.0;
        for facility in 0..self.num_facilities {
            for dept in 0..self.num_departments {
                for resource in 0..self.num_resource_types {
                    let mut constraint_vars = Vec::new();
                    
                    for level in 0..precision {
                        constraint_vars.push(var_map[&(facility, dept, resource, level)]);
                    }
                    
                    // Penalty for not selecting exactly one level
                    for &var1 in &constraint_vars {
                        builder.add_bias(var1, -constraint_penalty);
                        for &var2 in &constraint_vars {
                            if var1 != var2 {
                                builder.add_coupling(var1, var2, constraint_penalty);
                            }
                        }
                    }
                }
            }
        }
        
        // Constraint: don't exceed available resources
        for facility in 0..self.num_facilities {
            for resource in 0..self.num_resource_types {
                let available = self.available_resources[facility][resource] * 
                    (1.0 - self.emergency_reserves[resource]); // Reserve for emergencies
                
                for level1 in 0..precision {
                    for dept1 in 0..self.num_departments {
                        for level2 in 0..precision {
                            for dept2 in (dept1 + 1)..self.num_departments {
                                let alloc1 = (level1 as f64) / (precision as f64) * available;
                                let alloc2 = (level2 as f64) / (precision as f64) * available;
                                
                                if alloc1 + alloc2 > available {
                                    let var1 = var_map[&(facility, dept1, resource, level1)];
                                    let var2 = var_map[&(facility, dept2, resource, level2)];
                                    builder.add_coupling(var1, var2, constraint_penalty);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        builder.build()
    }
    
    fn evaluate_solution(&self, solution: &Self::Solution) -> ApplicationResult<Self::ObjectiveValue> {
        let satisfaction = self.calculate_patient_satisfaction(solution);
        let total_cost = solution.total_cost;
        
        // Objective: maximize satisfaction while minimizing cost
        Ok(satisfaction * 1000.0 - total_cost)
    }
    
    fn is_feasible(&self, solution: &Self::Solution) -> bool {
        // Check resource constraints
        for facility in 0..self.num_facilities {
            for resource in 0..self.num_resource_types {
                let total_allocated: f64 = (0..self.num_departments)
                    .map(|dept| solution.allocations[facility][dept][resource])
                    .sum();
                
                let available = self.available_resources[facility][resource] * 
                    (1.0 - self.emergency_reserves[resource]);
                
                if total_allocated > available + 1e-6 {
                    return false;
                }
            }
        }
        
        // Check minimum quality targets
        let satisfaction = self.calculate_patient_satisfaction(solution);
        if satisfaction < 0.8 { // Minimum 80% patient satisfaction
            return false;
        }
        
        true
    }
}

/// Solution for Medical Resource Allocation
#[derive(Debug, Clone)]
pub struct MedicalResourceSolution {
    /// Resource allocations [facility][department][resource]
    pub allocations: Vec<Vec<Vec<f64>>>,
    /// Total allocation cost
    pub total_cost: f64,
    /// Patient satisfaction score
    pub patient_satisfaction: f64,
    /// Resource utilization rates
    pub utilization_rates: Vec<f64>,
    /// Quality metrics
    pub quality_metrics: HealthcareMetrics,
}

/// Healthcare performance metrics
#[derive(Debug, Clone)]
pub struct HealthcareMetrics {
    /// Average patient wait time
    pub avg_wait_time: f64,
    /// Bed occupancy rate
    pub bed_occupancy: f64,
    /// Staff utilization rate
    pub staff_utilization: f64,
    /// Equipment utilization rate
    pub equipment_utilization: f64,
    /// Emergency response time
    pub emergency_response_time: f64,
    /// Quality of care score
    pub quality_score: f64,
}

impl IndustrySolution for MedicalResourceSolution {
    type Problem = MedicalResourceAllocation;
    
    fn from_binary(problem: &Self::Problem, binary_solution: &[i8]) -> ApplicationResult<Self> {
        let precision = 20;
        let mut allocations = vec![vec![vec![0.0; problem.num_resource_types]; problem.num_departments]; problem.num_facilities];
        let mut var_idx = 0;
        
        // Decode allocations from binary solution
        for facility in 0..problem.num_facilities {
            for dept in 0..problem.num_departments {
                for resource in 0..problem.num_resource_types {
                    for level in 0..precision {
                        if var_idx < binary_solution.len() && binary_solution[var_idx] == 1 {
                            let allocation_fraction = (level as f64) / (precision as f64);
                            let available = problem.available_resources[facility][resource] * 
                                (1.0 - problem.emergency_reserves[resource]);
                            allocations[facility][dept][resource] = allocation_fraction * available;
                            break;
                        }
                        var_idx += 1;
                    }
                }
            }
        }
        
        // Calculate total cost
        let mut total_cost = 0.0;
        for facility in 0..problem.num_facilities {
            for dept in 0..problem.num_departments {
                for resource in 0..problem.num_resource_types {
                    total_cost += allocations[facility][dept][resource] * 
                        problem.resource_costs[facility][resource];
                }
            }
        }
        
        let solution = MedicalResourceSolution {
            allocations,
            total_cost,
            patient_satisfaction: 0.0,
            utilization_rates: Vec::new(),
            quality_metrics: HealthcareMetrics {
                avg_wait_time: 0.0,
                bed_occupancy: 0.0,
                staff_utilization: 0.0,
                equipment_utilization: 0.0,
                emergency_response_time: 0.0,
                quality_score: 0.0,
            },
        };
        
        // Calculate metrics
        let patient_satisfaction = problem.calculate_patient_satisfaction(&solution);
        let utilization_rates = problem.calculate_utilization(&solution);
        
        let quality_metrics = HealthcareMetrics {
            avg_wait_time: 25.0, // Simplified calculation
            bed_occupancy: utilization_rates.get(0).copied().unwrap_or(0.0),
            staff_utilization: utilization_rates.get(1).copied().unwrap_or(0.0),
            equipment_utilization: utilization_rates.get(2).copied().unwrap_or(0.0),
            emergency_response_time: 8.0, // Simplified
            quality_score: patient_satisfaction,
        };
        
        Ok(MedicalResourceSolution {
            allocations: solution.allocations,
            total_cost,
            patient_satisfaction,
            utilization_rates,
            quality_metrics,
        })
    }
    
    fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("type".to_string(), "Medical Resource Allocation".to_string());
        summary.insert("total_cost".to_string(), format!("${:.2}", self.total_cost));
        summary.insert("patient_satisfaction".to_string(), format!("{:.1}%", self.patient_satisfaction * 100.0));
        summary.insert("avg_wait_time".to_string(), format!("{:.1} minutes", self.quality_metrics.avg_wait_time));
        summary.insert("bed_occupancy".to_string(), format!("{:.1}%", self.quality_metrics.bed_occupancy * 100.0));
        summary.insert("staff_utilization".to_string(), format!("{:.1}%", self.quality_metrics.staff_utilization * 100.0));
        summary.insert("quality_score".to_string(), format!("{:.3}", self.quality_metrics.quality_score));
        summary
    }
    
    fn metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("total_cost".to_string(), self.total_cost);
        metrics.insert("patient_satisfaction".to_string(), self.patient_satisfaction);
        metrics.insert("avg_wait_time".to_string(), self.quality_metrics.avg_wait_time);
        metrics.insert("bed_occupancy".to_string(), self.quality_metrics.bed_occupancy);
        metrics.insert("staff_utilization".to_string(), self.quality_metrics.staff_utilization);
        metrics.insert("equipment_utilization".to_string(), self.quality_metrics.equipment_utilization);
        metrics.insert("emergency_response_time".to_string(), self.quality_metrics.emergency_response_time);
        metrics.insert("quality_score".to_string(), self.quality_metrics.quality_score);
        
        for (i, &util) in self.utilization_rates.iter().enumerate() {
            metrics.insert(format!("resource_{}_utilization", i), util);
        }
        
        metrics
    }
    
    fn export_format(&self) -> ApplicationResult<String> {
        let mut output = String::new();
        output.push_str("# Medical Resource Allocation Report\n\n");
        
        output.push_str("## Allocation Summary\n");
        output.push_str(&format!("Total Cost: ${:.2}\n", self.total_cost));
        output.push_str(&format!("Patient Satisfaction: {:.1}%\n", self.patient_satisfaction * 100.0));
        output.push_str(&format!("Quality Score: {:.3}\n", self.quality_metrics.quality_score));
        
        output.push_str("\n## Resource Utilization\n");
        output.push_str(&format!("Bed Occupancy: {:.1}%\n", self.quality_metrics.bed_occupancy * 100.0));
        output.push_str(&format!("Staff Utilization: {:.1}%\n", self.quality_metrics.staff_utilization * 100.0));
        output.push_str(&format!("Equipment Utilization: {:.1}%\n", self.quality_metrics.equipment_utilization * 100.0));
        
        output.push_str("\n## Performance Metrics\n");
        output.push_str(&format!("Average Wait Time: {:.1} minutes\n", self.quality_metrics.avg_wait_time));
        output.push_str(&format!("Emergency Response Time: {:.1} minutes\n", self.quality_metrics.emergency_response_time));
        
        output.push_str("\n## Resource Allocations\n");
        for (facility, facility_allocs) in self.allocations.iter().enumerate() {
            output.push_str(&format!("### Facility {}\n", facility + 1));
            for (dept, dept_allocs) in facility_allocs.iter().enumerate() {
                output.push_str(&format!("Department {}: ", dept + 1));
                for (resource, &allocation) in dept_allocs.iter().enumerate() {
                    output.push_str(&format!("R{}: {:.1} ", resource, allocation));
                }
                output.push_str("\n");
            }
        }
        
        Ok(output)
    }
}

/// Treatment Planning Optimization Problem
#[derive(Debug, Clone)]
pub struct TreatmentPlanningOptimization {
    /// Number of patients
    pub num_patients: usize,
    /// Number of available treatments
    pub num_treatments: usize,
    /// Treatment efficacy matrix [patient][treatment]
    pub treatment_efficacy: Vec<Vec<f64>>,
    /// Treatment costs
    pub treatment_costs: Vec<f64>,
    /// Treatment durations
    pub treatment_durations: Vec<f64>,
    /// Patient priority scores
    pub patient_priorities: Vec<f64>,
    /// Treatment compatibility constraints
    pub compatibility_constraints: Vec<Vec<bool>>,
    /// Resource requirements for treatments
    pub resource_requirements: Vec<HashMap<String, f64>>,
    /// Available resources
    pub available_resources: HashMap<String, f64>,
}

impl TreatmentPlanningOptimization {
    /// Create new treatment planning problem
    pub fn new(
        num_patients: usize,
        num_treatments: usize,
        treatment_efficacy: Vec<Vec<f64>>,
        treatment_costs: Vec<f64>,
    ) -> ApplicationResult<Self> {
        if treatment_efficacy.len() != num_patients {
            return Err(ApplicationError::InvalidConfiguration(
                "Treatment efficacy matrix dimension mismatch".to_string()
            ));
        }
        
        Ok(Self {
            num_patients,
            num_treatments,
            treatment_efficacy,
            treatment_costs,
            treatment_durations: vec![1.0; num_treatments],
            patient_priorities: vec![1.0; num_patients],
            compatibility_constraints: vec![vec![true; num_treatments]; num_patients],
            resource_requirements: vec![HashMap::new(); num_treatments],
            available_resources: HashMap::new(),
        })
    }
    
    /// Calculate total treatment benefit
    pub fn calculate_benefit(&self, plan: &TreatmentPlan) -> f64 {
        let mut total_benefit = 0.0;
        
        for (patient, assignments) in plan.patient_treatments.iter().enumerate() {
            for (treatment, &assigned) in assignments.iter().enumerate() {
                if assigned {
                    let efficacy = self.treatment_efficacy[patient][treatment];
                    let priority = self.patient_priorities[patient];
                    total_benefit += efficacy * priority;
                }
            }
        }
        
        total_benefit
    }
}

/// Treatment Plan Solution
#[derive(Debug, Clone)]
pub struct TreatmentPlan {
    /// Treatment assignments [patient][treatment] = assigned
    pub patient_treatments: Vec<Vec<bool>>,
    /// Total cost
    pub total_cost: f64,
    /// Total benefit
    pub total_benefit: f64,
    /// Resource utilization
    pub resource_utilization: HashMap<String, f64>,
}

/// Create benchmark healthcare problems
pub fn create_benchmark_problems(size: usize) -> ApplicationResult<Vec<Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>>> {
    let mut problems = Vec::new();
    
    // Problem 1: Small hospital resource allocation
    let available_resources = vec![
        vec![50.0, 20.0, 10.0], // Facility 1: beds, staff, equipment
        vec![30.0, 15.0, 8.0],  // Facility 2
    ];
    let patient_demands = vec![
        vec![40.0, 25.0, 15.0], // Facility 1 demands by department
        vec![20.0, 18.0, 12.0], // Facility 2 demands by department
    ];
    
    let hospital_allocation = MedicalResourceAllocation::new(
        2, // facilities
        3, // departments  
        3, // resource types
        available_resources,
        patient_demands,
    )?;
    
    problems.push(Box::new(hospital_allocation) as Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>);
    
    // Problem 2: Larger healthcare network
    if size >= 4 {
        let large_resources = vec![vec![100.0, 50.0, 25.0, 15.0]; size];
        let large_demands = vec![vec![80.0, 40.0, 30.0, 20.0]; size];
        
        let large_allocation = MedicalResourceAllocation::new(
            size,
            4, // departments
            4, // resource types
            large_resources,
            large_demands,
        )?;
        
        problems.push(Box::new(large_allocation) as Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>);
    }
    
    Ok(problems)
}

/// Solve medical resource allocation using quantum annealing
pub fn solve_medical_resource_allocation(
    problem: &MedicalResourceAllocation,
    params: Option<AnnealingParams>,
) -> ApplicationResult<MedicalResourceSolution> {
    // Convert to QUBO
    let qubo = problem.to_qubo()?;
    
    // Convert to Ising
    let ising = IsingModel::from_qubo(&qubo)?;
    
    // Set up annealing parameters
    let annealing_params = params.unwrap_or_else(|| {
        let mut p = AnnealingParams::default();
        p.num_sweeps = 20000;
        p.num_repetitions = 30;
        p.initial_temperature = 4.0;
        p.final_temperature = 0.01;
        p
    });
    
    // Solve with classical annealing
    let simulator = ClassicalAnnealingSimulator::new(annealing_params)
        .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;
    
    let result = simulator.solve(&ising)
        .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;
    
    // Convert solution back to medical resource allocation
    MedicalResourceSolution::from_binary(problem, &result.best_spins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_medical_resource_allocation_creation() {
        let available_resources = vec![
            vec![50.0, 20.0, 10.0],
            vec![30.0, 15.0, 8.0],
        ];
        let patient_demands = vec![
            vec![40.0, 25.0, 15.0],
            vec![20.0, 18.0, 12.0],
        ];
        
        let allocation = MedicalResourceAllocation::new(2, 3, 3, available_resources, patient_demands).unwrap();
        assert_eq!(allocation.num_facilities, 2);
        assert_eq!(allocation.num_departments, 3);
        assert_eq!(allocation.num_resource_types, 3);
    }
    
    #[test]
    fn test_patient_satisfaction_calculation() {
        let available_resources = vec![vec![100.0, 50.0]];
        let patient_demands = vec![vec![80.0, 40.0]];
        
        let problem = MedicalResourceAllocation::new(1, 2, 2, available_resources, patient_demands).unwrap();
        
        let solution = MedicalResourceSolution {
            allocations: vec![vec![vec![80.0, 20.0], vec![20.0, 30.0]]],
            total_cost: 1000.0,
            patient_satisfaction: 0.0,
            utilization_rates: Vec::new(),
            quality_metrics: HealthcareMetrics {
                avg_wait_time: 0.0,
                bed_occupancy: 0.0,
                staff_utilization: 0.0,
                equipment_utilization: 0.0,
                emergency_response_time: 0.0,
                quality_score: 0.0,
            },
        };
        
        let satisfaction = problem.calculate_patient_satisfaction(&solution);
        assert!(satisfaction >= 0.0 && satisfaction <= 1.0);
    }
    
    #[test]
    fn test_treatment_planning_creation() {
        let efficacy = vec![
            vec![0.8, 0.6, 0.9],
            vec![0.7, 0.9, 0.5],
        ];
        let costs = vec![1000.0, 1500.0, 800.0];
        
        let planning = TreatmentPlanningOptimization::new(2, 3, efficacy, costs).unwrap();
        assert_eq!(planning.num_patients, 2);
        assert_eq!(planning.num_treatments, 3);
    }
    
    #[test]
    fn test_benchmark_problems() {
        let problems = create_benchmark_problems(3).unwrap();
        assert_eq!(problems.len(), 2);
        
        for problem in &problems {
            assert!(problem.validate().is_ok());
        }
    }
}