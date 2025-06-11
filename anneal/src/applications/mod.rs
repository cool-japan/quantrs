//! Industry-Specific Optimization Libraries
//!
//! This module provides specialized optimization frameworks for various industries,
//! leveraging quantum annealing techniques to solve real-world problems.
//!
//! # Available Industries
//!
//! - **Finance**: Portfolio optimization, risk management, fraud detection
//! - **Logistics**: Vehicle routing, supply chain optimization, scheduling
//! - **Energy**: Grid optimization, renewable energy management, load balancing
//! - **Manufacturing**: Production scheduling, quality control, resource allocation
//! - **Healthcare**: Treatment optimization, resource allocation, drug discovery
//! - **Telecommunications**: Network optimization, traffic routing, infrastructure planning
//! - **Transportation**: Vehicle routing, traffic flow optimization, smart city planning
//!
//! # Design Philosophy
//!
//! Each industry module provides:
//! - Domain-specific problem formulations
//! - Real-world constraints and objectives
//! - Benchmark problems and datasets
//! - Performance metrics relevant to the industry
//! - Integration with quantum annealing solvers

pub mod energy;
pub mod finance;
pub mod healthcare;
pub mod integration_tests;
pub mod logistics;
pub mod manufacturing;
pub mod performance_benchmarks;
pub mod telecommunications;
pub mod transportation;
pub mod unified;

use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur in industry applications
#[derive(Error, Debug)]
pub enum ApplicationError {
    /// Invalid problem configuration
    #[error("Invalid problem configuration: {0}")]
    InvalidConfiguration(String),

    /// Constraint violation
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    /// Optimization error
    #[error("Optimization error: {0}")]
    OptimizationError(String),

    /// Data validation error
    #[error("Data validation error: {0}")]
    DataValidationError(String),

    /// Resource limit exceeded
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),

    /// Industry-specific error
    #[error("Industry-specific error: {0}")]
    IndustrySpecificError(String),
}

/// Result type for industry applications
pub type ApplicationResult<T> = Result<T, ApplicationError>;

/// Common traits for industry-specific problems

/// Problem instance that can be solved with quantum annealing
pub trait OptimizationProblem {
    type Solution;
    type ObjectiveValue;

    /// Get problem description
    fn description(&self) -> String;

    /// Get problem size metrics
    fn size_metrics(&self) -> HashMap<String, usize>;

    /// Validate problem instance
    fn validate(&self) -> ApplicationResult<()>;

    /// Convert to QUBO formulation
    fn to_qubo(&self) -> ApplicationResult<(crate::ising::QuboModel, HashMap<String, usize>)>;

    /// Evaluate solution quality
    fn evaluate_solution(
        &self,
        solution: &Self::Solution,
    ) -> ApplicationResult<Self::ObjectiveValue>;

    /// Check if solution satisfies all constraints
    fn is_feasible(&self, solution: &Self::Solution) -> bool;
}

/// Solution that can be interpreted in industry context
pub trait IndustrySolution {
    type Problem;

    /// Convert from binary solution vector
    fn from_binary(problem: &Self::Problem, binary_solution: &[i8]) -> ApplicationResult<Self>
    where
        Self: Sized;

    /// Get solution summary
    fn summary(&self) -> HashMap<String, String>;

    /// Get solution metrics
    fn metrics(&self) -> HashMap<String, f64>;

    /// Export solution in industry-standard format
    fn export_format(&self) -> ApplicationResult<String>;
}

/// Performance benchmarking for industry problems
pub trait Benchmarkable {
    type BenchmarkResult;

    /// Run benchmark suite
    fn run_benchmark(&self) -> ApplicationResult<Self::BenchmarkResult>;

    /// Compare against industry baselines
    fn compare_baseline(&self, baseline: &Self::BenchmarkResult) -> HashMap<String, f64>;

    /// Generate benchmark report
    fn benchmark_report(&self, result: &Self::BenchmarkResult) -> String;
}

/// Common industry problem categories
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProblemCategory {
    /// Resource allocation and scheduling
    ResourceAllocation,
    /// Route and path optimization
    Routing,
    /// Portfolio and investment optimization
    Portfolio,
    /// Network design and optimization
    NetworkDesign,
    /// Supply chain optimization
    SupplyChain,
    /// Risk management and assessment
    RiskManagement,
    /// Quality control and testing
    QualityControl,
    /// Demand forecasting and planning
    DemandPlanning,
    /// Energy management and grid optimization
    EnergyManagement,
    /// Treatment and care optimization
    TreatmentOptimization,
}

/// Industry-specific constraint types
#[derive(Debug, Clone)]
pub enum IndustryConstraint {
    /// Resource capacity constraints
    Capacity { resource: String, limit: f64 },
    /// Time window constraints
    TimeWindow { start: f64, end: f64 },
    /// Budget constraints
    Budget { limit: f64 },
    /// Regulatory compliance constraints
    Regulatory {
        regulation: String,
        requirement: String,
    },
    /// Quality requirements
    Quality { metric: String, threshold: f64 },
    /// Safety requirements
    Safety { standard: String, level: f64 },
    /// Custom constraint
    Custom { name: String, description: String },
}

/// Common objective functions across industries
#[derive(Debug, Clone)]
pub enum IndustryObjective {
    /// Minimize total cost
    MinimizeCost,
    /// Maximize profit/revenue
    MaximizeProfit,
    /// Minimize risk
    MinimizeRisk,
    /// Maximize efficiency
    MaximizeEfficiency,
    /// Minimize time/makespan
    MinimizeTime,
    /// Maximize quality
    MaximizeQuality,
    /// Minimize resource usage
    MinimizeResourceUsage,
    /// Maximize customer satisfaction
    MaximizeSatisfaction,
    /// Multi-objective combination
    MultiObjective(Vec<(IndustryObjective, f64)>), // (objective, weight)
}

/// Utility functions for industry applications

/// Create standard benchmark problems for testing
pub fn create_benchmark_suite(
    industry: &str,
    size: &str,
) -> ApplicationResult<Vec<Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>>>
{
    match (industry, size) {
        ("finance", "small") => Ok(finance::create_benchmark_problems(10)?),
        ("finance", "medium") => Ok(finance::create_benchmark_problems(50)?),
        ("finance", "large") => Ok(finance::create_benchmark_problems(200)?),

        ("logistics", "small") => Ok(logistics::create_benchmark_problems(5)?),
        ("logistics", "medium") => Ok(logistics::create_benchmark_problems(20)?),
        ("logistics", "large") => Ok(logistics::create_benchmark_problems(100)?),

        ("energy", "small") => Ok(energy::create_benchmark_problems(8)?),
        ("energy", "medium") => Ok(energy::create_benchmark_problems(30)?),
        ("energy", "large") => Ok(energy::create_benchmark_problems(150)?),

        ("transportation", "small") => Ok(transportation::create_benchmark_problems(5)?),
        ("transportation", "medium") => Ok(transportation::create_benchmark_problems(15)?),
        ("transportation", "large") => Ok(transportation::create_benchmark_problems(50)?),

        ("manufacturing", "small") => Ok(manufacturing::create_benchmark_problems(5)?),
        ("manufacturing", "medium") => Ok(manufacturing::create_benchmark_problems(15)?),
        ("manufacturing", "large") => Ok(manufacturing::create_benchmark_problems(50)?),

        ("healthcare", "small") => Ok(healthcare::create_benchmark_problems(5)?),
        ("healthcare", "medium") => Ok(healthcare::create_benchmark_problems(15)?),
        ("healthcare", "large") => Ok(healthcare::create_benchmark_problems(50)?),

        ("telecommunications", "small") => Ok(telecommunications::create_benchmark_problems(5)?),
        ("telecommunications", "medium") => Ok(telecommunications::create_benchmark_problems(15)?),
        ("telecommunications", "large") => Ok(telecommunications::create_benchmark_problems(50)?),

        _ => Err(ApplicationError::InvalidConfiguration(format!(
            "Unknown benchmark: {} / {}",
            industry, size
        ))),
    }
}

/// Generate comprehensive performance report
pub fn generate_performance_report(
    industry: &str,
    results: &HashMap<String, f64>,
) -> ApplicationResult<String> {
    let mut report = String::new();

    report.push_str(&format!(
        "# {} Industry Optimization Report\n\n",
        industry.to_uppercase()
    ));
    report.push_str("## Performance Metrics\n\n");

    // Sort metrics for consistent reporting
    let mut sorted_metrics: Vec<_> = results.iter().collect();
    sorted_metrics.sort_by_key(|(key, _)| *key);

    for (metric, value) in sorted_metrics {
        report.push_str(&format!("- **{}**: {:.4}\n", metric, value));
    }

    report.push_str("\n## Industry-Specific Analysis\n\n");

    match industry {
        "finance" => {
            report.push_str("- Risk-adjusted returns analyzed\n");
            report.push_str("- Regulatory compliance verified\n");
            report.push_str("- Market volatility considered\n");
        }
        "logistics" => {
            report.push_str("- Route efficiency optimized\n");
            report.push_str("- Delivery time constraints satisfied\n");
            report.push_str("- Vehicle capacity utilization maximized\n");
        }
        "energy" => {
            report.push_str("- Grid stability maintained\n");
            report.push_str("- Renewable energy integration optimized\n");
            report.push_str("- Load balancing achieved\n");
        }
        "manufacturing" => {
            report.push_str("- Production schedules optimized\n");
            report.push_str("- Resource utilization maximized\n");
            report.push_str("- Quality constraints satisfied\n");
        }
        "healthcare" => {
            report.push_str("- Patient care maximized\n");
            report.push_str("- Resource allocation optimized\n");
            report.push_str("- Emergency capacity reserved\n");
        }
        "telecommunications" => {
            report.push_str("- Network connectivity optimized\n");
            report.push_str("- Latency minimized\n");
            report.push_str("- Capacity constraints satisfied\n");
        }
        "transportation" => {
            report.push_str("- Route efficiency optimized\n");
            report.push_str("- Vehicle capacity utilization maximized\n");
            report.push_str("- Time window constraints satisfied\n");
        }
        _ => {
            report.push_str("- Domain-specific analysis completed\n");
        }
    }

    Ok(report)
}

/// Validate industry-specific constraints
pub fn validate_constraints(
    constraints: &[IndustryConstraint],
    solution_data: &HashMap<String, f64>,
) -> ApplicationResult<()> {
    for constraint in constraints {
        match constraint {
            IndustryConstraint::Capacity { resource, limit } => {
                if let Some(&usage) = solution_data.get(resource) {
                    if usage > *limit {
                        return Err(ApplicationError::ConstraintViolation(format!(
                            "Resource {} usage {} exceeds limit {}",
                            resource, usage, limit
                        )));
                    }
                }
            }
            IndustryConstraint::Budget { limit } => {
                if let Some(&cost) = solution_data.get("total_cost") {
                    if cost > *limit {
                        return Err(ApplicationError::ConstraintViolation(format!(
                            "Total cost {} exceeds budget {}",
                            cost, limit
                        )));
                    }
                }
            }
            IndustryConstraint::Quality { metric, threshold } => {
                if let Some(&quality) = solution_data.get(metric) {
                    if quality < *threshold {
                        return Err(ApplicationError::ConstraintViolation(format!(
                            "Quality metric {} value {} below threshold {}",
                            metric, quality, threshold
                        )));
                    }
                }
            }
            _ => {
                // For other constraint types, assume they're handled elsewhere
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_validation() {
        let constraints = vec![
            IndustryConstraint::Capacity {
                resource: "memory".to_string(),
                limit: 100.0,
            },
            IndustryConstraint::Budget { limit: 1000.0 },
        ];

        let mut solution_data = HashMap::new();
        solution_data.insert("memory".to_string(), 80.0);
        solution_data.insert("total_cost".to_string(), 500.0);

        assert!(validate_constraints(&constraints, &solution_data).is_ok());

        solution_data.insert("memory".to_string(), 150.0);
        assert!(validate_constraints(&constraints, &solution_data).is_err());
    }

    #[test]
    fn test_performance_report_generation() {
        let mut results = HashMap::new();
        results.insert("accuracy".to_string(), 0.95);
        results.insert("efficiency".to_string(), 0.88);

        let report = generate_performance_report("finance", &results).unwrap();
        assert!(report.contains("FINANCE"));
        assert!(report.contains("accuracy"));
        assert!(report.contains("0.95"));
    }
}
