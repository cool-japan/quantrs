//! Unified Problem Interface and Solver Factory
//!
//! This module provides a comprehensive unified interface for all industry-specific
//! optimization problems, along with a solver factory that can automatically
//! select and configure appropriate solvers based on problem characteristics.

use super::{
    energy, finance, healthcare, logistics, manufacturing, telecommunications, ApplicationError,
    ApplicationResult, IndustryConstraint, IndustryObjective, IndustrySolution,
    OptimizationProblem, ProblemCategory,
};
use crate::ising::IsingModel;
use crate::qubo::{QuboBuilder, QuboFormulation};
use crate::simulator::{
    AnnealingParams, AnnealingResult, ClassicalAnnealingSimulator, QuantumAnnealingSimulator,
};
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;

/// Unified solution type that can represent different kinds of optimization solutions
#[derive(Debug, Clone)]
pub enum UnifiedSolution {
    /// Binary solution vector (for discrete optimization)
    Binary(Vec<i8>),
    /// Continuous solution vector (for continuous optimization)
    Continuous(Vec<f64>),
    /// Mixed binary-continuous solution
    Mixed {
        binary: Vec<i8>,
        continuous: Vec<f64>,
    },
    /// Industry-specific solution data
    Custom(serde_json::Value),
}

impl UnifiedSolution {
    /// Get the binary part of the solution if available
    #[must_use]
    pub const fn binary(&self) -> Option<&Vec<i8>> {
        match self {
            Self::Binary(b) => Some(b),
            Self::Mixed { binary, .. } => Some(binary),
            _ => None,
        }
    }

    /// Get the continuous part of the solution if available
    #[must_use]
    pub const fn continuous(&self) -> Option<&Vec<f64>> {
        match self {
            Self::Continuous(c) => Some(c),
            Self::Mixed { continuous, .. } => Some(continuous),
            _ => None,
        }
    }
}

/// Unified problem interface that extends the base optimization problem trait
pub trait UnifiedProblem:
    OptimizationProblem<Solution = UnifiedSolution, ObjectiveValue = f64> + Debug + Send + Sync
{
    /// Get the industry category of this problem
    fn category(&self) -> ProblemCategory;

    /// Get the industry name
    fn industry(&self) -> &'static str;

    /// Get problem complexity estimate (small, medium, large)
    fn complexity(&self) -> ProblemComplexity;

    /// Get recommended solver configuration
    fn recommended_solver_config(&self) -> SolverConfiguration;

    /// Get problem-specific constraints
    fn constraints(&self) -> Vec<IndustryConstraint>;

    /// Get problem objective
    fn objective(&self) -> IndustryObjective;

    /// Get expected solution quality bounds
    fn solution_bounds(&self) -> Option<(f64, f64)>; // (lower_bound, upper_bound)

    /// Convert to Any for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Clone into a boxed unified problem
    fn clone_unified(&self) -> Box<dyn UnifiedProblem>;
}

/// Problem complexity categories for solver selection
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProblemComplexity {
    /// Small problems (< 100 variables)
    Small,
    /// Medium problems (100-1000 variables)
    Medium,
    /// Large problems (1000-10_000 variables)
    Large,
    /// Extra large problems (> `10_000` variables)
    ExtraLarge,
}

/// Solver configuration recommendations
#[derive(Debug, Clone)]
pub struct SolverConfiguration {
    /// Preferred solver type
    pub solver_type: SolverType,
    /// Annealing parameters
    pub annealing_params: AnnealingParams,
    /// Hardware requirements
    pub hardware_requirements: HardwareRequirements,
    /// Optimization hints
    pub optimization_hints: Vec<OptimizationHint>,
}

/// Available solver types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum SolverType {
    #[default]
    /// Classical simulated annealing
    Classical,
    /// Quantum annealing simulator
    QuantumSimulator,
    /// D-Wave quantum annealer
    DWave,
    /// Hybrid classical-quantum
    Hybrid,
    /// Problem-specific heuristic
    Heuristic,
}

/// Hardware requirements for solving
#[derive(Debug, Clone)]
pub struct HardwareRequirements {
    /// Minimum memory (GB)
    pub min_memory_gb: f64,
    /// Recommended CPU cores
    pub recommended_cores: usize,
    /// GPU acceleration beneficial
    pub gpu_acceleration: bool,
    /// Quantum hardware required
    pub quantum_hardware: bool,
}

/// Optimization hints for solver configuration
#[derive(Debug, Clone)]
pub enum OptimizationHint {
    /// Problem has sparse structure
    SparseStructure,
    /// Problem has many local minima
    MultiModal,
    /// Problem is highly constrained
    HighlyConstrained,
    /// Real-time solving required
    RealTime,
    /// High precision required
    HighPrecision,
    /// Approximate solution acceptable
    ApproximateOk,
}

/// Comprehensive solver factory for all problem types
#[derive(Debug, Clone)]
pub struct UnifiedSolverFactory {
    /// Default solver configurations by category
    default_configs: HashMap<ProblemCategory, SolverConfiguration>,
    /// Available solver backends
    available_solvers: Vec<SolverType>,
    /// Performance cache for solver selection
    performance_cache: HashMap<String, f64>,
}

impl Default for UnifiedSolverFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl UnifiedSolverFactory {
    /// Create a new solver factory with default configurations
    #[must_use]
    pub fn new() -> Self {
        let mut factory = Self {
            default_configs: HashMap::new(),
            available_solvers: vec![SolverType::Classical, SolverType::QuantumSimulator],
            performance_cache: HashMap::new(),
        };

        factory.initialize_default_configs();
        factory
    }

    /// Initialize default configurations for different problem categories
    fn initialize_default_configs(&mut self) {
        // Finance problems - typically medium complexity, precision important
        self.default_configs.insert(
            ProblemCategory::Portfolio,
            SolverConfiguration {
                solver_type: SolverType::Classical,
                annealing_params: AnnealingParams {
                    num_sweeps: 10_000,
                    num_repetitions: 20,
                    initial_temperature: 2.0,
                    final_temperature: 0.01,
                    temperature_schedule: crate::simulator::TemperatureSchedule::Linear,
                    ..Default::default()
                },
                hardware_requirements: HardwareRequirements {
                    min_memory_gb: 2.0,
                    recommended_cores: 4,
                    gpu_acceleration: false,
                    quantum_hardware: false,
                },
                optimization_hints: vec![OptimizationHint::HighPrecision],
            },
        );

        // Logistics problems - complex routing, heuristics often helpful
        self.default_configs.insert(
            ProblemCategory::Routing,
            SolverConfiguration {
                solver_type: SolverType::Hybrid,
                annealing_params: AnnealingParams {
                    num_sweeps: 15_000,
                    num_repetitions: 25,
                    initial_temperature: 3.0,
                    final_temperature: 0.005,
                    temperature_schedule: crate::simulator::TemperatureSchedule::Exponential(0.95),
                    ..Default::default()
                },
                hardware_requirements: HardwareRequirements {
                    min_memory_gb: 4.0,
                    recommended_cores: 8,
                    gpu_acceleration: true,
                    quantum_hardware: false,
                },
                optimization_hints: vec![
                    OptimizationHint::MultiModal,
                    OptimizationHint::HighlyConstrained,
                ],
            },
        );

        // Network design - large sparse problems
        self.default_configs.insert(
            ProblemCategory::NetworkDesign,
            SolverConfiguration {
                solver_type: SolverType::QuantumSimulator,
                annealing_params: AnnealingParams {
                    num_sweeps: 25_000,
                    num_repetitions: 40,
                    initial_temperature: 5.0,
                    final_temperature: 0.001,
                    temperature_schedule: crate::simulator::TemperatureSchedule::Linear,
                    ..Default::default()
                },
                hardware_requirements: HardwareRequirements {
                    min_memory_gb: 8.0,
                    recommended_cores: 16,
                    gpu_acceleration: true,
                    quantum_hardware: false,
                },
                optimization_hints: vec![OptimizationHint::SparseStructure],
            },
        );

        // Add more default configurations as needed
        self.add_remaining_default_configs();
    }

    /// Add remaining default configurations
    fn add_remaining_default_configs(&mut self) {
        // Resource allocation problems
        self.default_configs.insert(
            ProblemCategory::ResourceAllocation,
            SolverConfiguration {
                solver_type: SolverType::Classical,
                annealing_params: AnnealingParams::default(),
                hardware_requirements: HardwareRequirements {
                    min_memory_gb: 1.0,
                    recommended_cores: 2,
                    gpu_acceleration: false,
                    quantum_hardware: false,
                },
                optimization_hints: vec![OptimizationHint::HighlyConstrained],
            },
        );

        // Supply chain optimization
        self.default_configs.insert(
            ProblemCategory::SupplyChain,
            SolverConfiguration {
                solver_type: SolverType::Hybrid,
                annealing_params: AnnealingParams {
                    num_sweeps: 20_000,
                    num_repetitions: 30,
                    initial_temperature: 4.0,
                    final_temperature: 0.01,
                    ..Default::default()
                },
                hardware_requirements: HardwareRequirements {
                    min_memory_gb: 6.0,
                    recommended_cores: 12,
                    gpu_acceleration: true,
                    quantum_hardware: false,
                },
                optimization_hints: vec![OptimizationHint::MultiModal],
            },
        );
    }

    /// Create a problem from industry and type specifications
    pub fn create_problem(
        &self,
        industry: &str,
        problem_type: &str,
        config: HashMap<String, serde_json::Value>,
    ) -> ApplicationResult<Box<dyn UnifiedProblem>> {
        match (industry, problem_type) {
            ("finance", "portfolio") => {
                let problem = self.create_finance_portfolio(config)?;
                Ok(Box::new(problem) as Box<dyn UnifiedProblem>)
            }
            ("logistics", "vrp") => {
                let problem = self.create_logistics_vrp(config)?;
                Ok(Box::new(problem) as Box<dyn UnifiedProblem>)
            }
            ("telecommunications", "network") => {
                let problem = self.create_telecom_network(config)?;
                Ok(Box::new(problem) as Box<dyn UnifiedProblem>)
            }
            ("energy", "grid") => {
                let problem = self.create_energy_grid(config)?;
                Ok(Box::new(problem) as Box<dyn UnifiedProblem>)
            }
            ("manufacturing", "scheduling") => {
                let problem = self.create_manufacturing_scheduling(config)?;
                Ok(Box::new(problem) as Box<dyn UnifiedProblem>)
            }
            ("healthcare", "resource") => {
                let problem = self.create_healthcare_resource(config)?;
                Ok(Box::new(problem) as Box<dyn UnifiedProblem>)
            }
            _ => Err(ApplicationError::InvalidConfiguration(format!(
                "Unknown problem type: {industry} / {problem_type}"
            ))),
        }
    }

    /// Solve any unified problem with automatic solver selection
    pub fn solve_problem(
        &self,
        problem: &dyn UnifiedProblem,
        custom_config: Option<SolverConfiguration>,
    ) -> ApplicationResult<UnifiedSolution> {
        // Get solver configuration (custom or default)
        let config = custom_config.unwrap_or_else(|| self.get_recommended_config(problem));

        // Validate problem
        problem.validate()?;

        // Convert to QUBO
        let (qubo_model, _var_map) = problem.to_qubo()?;

        // Solve based on configuration
        let result = match config.solver_type {
            SolverType::Classical => self.solve_classical(&qubo_model, &config.annealing_params)?,
            SolverType::QuantumSimulator => {
                self.solve_quantum_simulator(&qubo_model, &config.annealing_params)?
            }
            SolverType::Hybrid => self.solve_hybrid(&qubo_model, &config.annealing_params)?,
            _ => {
                return Err(ApplicationError::OptimizationError(
                    "Solver type not yet implemented".to_string(),
                ))
            }
        };

        // Create unified solution using the enum variant
        Ok(UnifiedSolution::Binary(result.best_spins))
    }

    /// Get recommended configuration for a problem
    fn get_recommended_config(&self, problem: &dyn UnifiedProblem) -> SolverConfiguration {
        // Use category defaults if available, otherwise problem's own recommendation
        let mut config = self
            .default_configs
            .get(&problem.category())
            .cloned()
            .unwrap_or_else(|| problem.recommended_solver_config());

        // Adjust based on problem complexity
        self.adjust_config_for_complexity(&mut config, problem.complexity());

        config
    }

    /// Adjust configuration based on problem complexity
    fn adjust_config_for_complexity(
        &self,
        config: &mut SolverConfiguration,
        complexity: ProblemComplexity,
    ) {
        match complexity {
            ProblemComplexity::Small => {
                config.annealing_params.num_sweeps = config.annealing_params.num_sweeps.min(5000);
                config.annealing_params.num_repetitions =
                    config.annealing_params.num_repetitions.min(10);
            }
            ProblemComplexity::Medium => {
                // Keep default values
            }
            ProblemComplexity::Large => {
                config.annealing_params.num_sweeps = config.annealing_params.num_sweeps.max(20_000);
                config.annealing_params.num_repetitions =
                    config.annealing_params.num_repetitions.max(30);
                config.hardware_requirements.min_memory_gb *= 2.0;
            }
            ProblemComplexity::ExtraLarge => {
                config.annealing_params.num_sweeps = config.annealing_params.num_sweeps.max(50_000);
                config.annealing_params.num_repetitions =
                    config.annealing_params.num_repetitions.max(50);
                config.hardware_requirements.min_memory_gb *= 4.0;
                config.solver_type = SolverType::Hybrid; // Force hybrid for very large problems
            }
        }
    }

    /// Solve using classical annealing
    fn solve_classical(
        &self,
        qubo_model: &crate::ising::QuboModel,
        params: &AnnealingParams,
    ) -> ApplicationResult<crate::simulator::AnnealingSolution> {
        let ising = IsingModel::from_qubo(qubo_model);

        let simulator = ClassicalAnnealingSimulator::new(params.clone())
            .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;

        simulator
            .solve(&ising)
            .map_err(|e| ApplicationError::OptimizationError(e.to_string()))
    }

    /// Solve using quantum simulator
    fn solve_quantum_simulator(
        &self,
        qubo_model: &crate::ising::QuboModel,
        params: &AnnealingParams,
    ) -> ApplicationResult<crate::simulator::AnnealingSolution> {
        let ising = IsingModel::from_qubo(qubo_model);

        let simulator = QuantumAnnealingSimulator::new(params.clone())
            .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;

        simulator
            .solve(&ising)
            .map_err(|e| ApplicationError::OptimizationError(e.to_string()))
    }

    /// Solve using hybrid approach
    fn solve_hybrid(
        &self,
        qubo_model: &crate::ising::QuboModel,
        params: &AnnealingParams,
    ) -> ApplicationResult<crate::simulator::AnnealingSolution> {
        // For now, use classical as fallback
        // In future, implement actual hybrid solver
        self.solve_classical(qubo_model, params)
    }

    /// Create finance portfolio problem from configuration
    fn create_finance_portfolio(
        &self,
        config: HashMap<String, serde_json::Value>,
    ) -> ApplicationResult<UnifiedPortfolioOptimization> {
        // Extract configuration parameters
        let num_assets = config
            .get("num_assets")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(10) as usize;

        let budget = config
            .get("budget")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1_000_000.0);

        let risk_tolerance = config
            .get("risk_tolerance")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.5);

        // Generate sample data
        let expected_returns: Vec<f64> = (0..num_assets)
            .map(|i| 0.03 + 0.07 * (i as f64) / (num_assets as f64))
            .collect();

        let covariance_matrix = self.create_sample_covariance_matrix(num_assets, 0.2);

        let portfolio = finance::PortfolioOptimization::new(
            expected_returns,
            covariance_matrix,
            budget,
            risk_tolerance,
        )?;

        Ok(UnifiedPortfolioOptimization { inner: portfolio })
    }

    /// Create logistics VRP problem from configuration
    fn create_logistics_vrp(
        &self,
        config: HashMap<String, serde_json::Value>,
    ) -> ApplicationResult<UnifiedVehicleRoutingProblem> {
        let num_vehicles = config
            .get("num_vehicles")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(3) as usize;

        let num_customers = config
            .get("num_customers")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(10) as usize;

        // Generate sample problem
        let mut locations = vec![(0.0, 0.0)]; // Depot
        let mut demands = vec![0.0]; // Depot demand

        for i in 0..num_customers {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / num_customers as f64;
            let radius = (i as f64).mul_add(0.1, 1.0);
            locations.push((radius * angle.cos(), radius * angle.sin()));
            demands.push((i as f64).mul_add(2.0, 5.0));
        }

        let capacities = vec![50.0; num_vehicles];

        let vrp =
            logistics::VehicleRoutingProblem::new(num_vehicles, capacities, locations, demands)?;

        Ok(UnifiedVehicleRoutingProblem { inner: vrp })
    }

    /// Create telecommunications network problem from configuration
    fn create_telecom_network(
        &self,
        config: HashMap<String, serde_json::Value>,
    ) -> ApplicationResult<UnifiedNetworkTopologyOptimization> {
        let num_nodes = config
            .get("num_nodes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(6) as usize;

        // Generate fully connected potential connections
        let mut potential_connections = Vec::new();
        let mut connection_costs = Vec::new();

        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                potential_connections.push((i, j));
                connection_costs.push(((i + j) as f64).mul_add(1.5, 5.0));
            }
        }

        let traffic_demands = vec![vec![2.0; num_nodes]; num_nodes];

        let network = telecommunications::NetworkTopologyOptimization::new(
            num_nodes,
            potential_connections,
            connection_costs,
            traffic_demands,
        )?;

        Ok(UnifiedNetworkTopologyOptimization { inner: network })
    }

    /// Create energy grid problem from configuration
    fn create_energy_grid(
        &self,
        _config: HashMap<String, serde_json::Value>,
    ) -> ApplicationResult<UnifiedEnergyGridOptimization> {
        // Placeholder - would create actual energy grid problem
        Err(ApplicationError::InvalidConfiguration(
            "Energy grid problem creation not yet implemented".to_string(),
        ))
    }

    /// Create manufacturing scheduling problem from configuration
    fn create_manufacturing_scheduling(
        &self,
        _config: HashMap<String, serde_json::Value>,
    ) -> ApplicationResult<UnifiedManufacturingScheduling> {
        // Placeholder - would create actual manufacturing problem
        Err(ApplicationError::InvalidConfiguration(
            "Manufacturing scheduling problem creation not yet implemented".to_string(),
        ))
    }

    /// Create healthcare resource problem from configuration
    fn create_healthcare_resource(
        &self,
        _config: HashMap<String, serde_json::Value>,
    ) -> ApplicationResult<UnifiedHealthcareResourceOptimization> {
        // Placeholder - would create actual healthcare problem
        Err(ApplicationError::InvalidConfiguration(
            "Healthcare resource problem creation not yet implemented".to_string(),
        ))
    }

    /// Helper to create sample covariance matrix
    fn create_sample_covariance_matrix(&self, n: usize, base_volatility: f64) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[i][j] =
                        base_volatility * base_volatility * (1.0 + 0.5 * (i as f64) / (n as f64));
                } else {
                    let correlation = 0.1 * (1.0 - (i as f64 - j as f64).abs() / (n as f64));
                    let vol_i = (matrix[i][i]).sqrt();
                    let vol_j = (matrix[j][j]).sqrt();
                    matrix[i][j] = correlation * vol_i * vol_j;
                }
            }
        }

        matrix
    }
}

// Removed duplicate UnifiedSolution struct definition - using enum version instead

/// Convergence information for the solving process
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Whether the solver converged
    pub converged: bool,
    /// Final energy/objective value
    pub final_energy: f64,
    /// Energy variance across repetitions
    pub energy_variance: f64,
    /// Acceptance rate during annealing
    pub acceptance_rate: f64,
}

/// Information about the problem that was solved
#[derive(Debug, Clone)]
pub struct ProblemInfo {
    /// Industry name
    pub industry: String,
    /// Problem category
    pub category: ProblemCategory,
    /// Problem complexity
    pub complexity: ProblemComplexity,
    /// Number of variables
    pub num_variables: usize,
    /// Number of constraints
    pub num_constraints: usize,
}

// Wrapper types for existing problems to implement UnifiedProblem
// These would be expanded for all problem types

/// Unified wrapper for portfolio optimization
#[derive(Debug, Clone)]
pub struct UnifiedPortfolioOptimization {
    inner: finance::PortfolioOptimization,
}

/// Unified wrapper for vehicle routing
#[derive(Debug, Clone)]
pub struct UnifiedVehicleRoutingProblem {
    inner: logistics::VehicleRoutingProblem,
}

/// Unified wrapper for network topology optimization
#[derive(Debug, Clone)]
pub struct UnifiedNetworkTopologyOptimization {
    inner: telecommunications::NetworkTopologyOptimization,
}

// Placeholder types for other industries
#[derive(Debug, Clone)]
pub struct UnifiedEnergyGridOptimization {
    // Would contain actual energy problem
}

#[derive(Debug, Clone)]
pub struct UnifiedManufacturingScheduling {
    // Would contain actual manufacturing problem
}

#[derive(Debug, Clone)]
pub struct UnifiedHealthcareResourceOptimization {
    // Would contain actual healthcare problem
}

// Implement UnifiedProblem for all wrapper types
// This is where the unified interface comes together

impl OptimizationProblem for UnifiedPortfolioOptimization {
    type Solution = UnifiedSolution;
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
        match solution {
            UnifiedSolution::Binary(binary_sol) => {
                // Convert binary solution to PortfolioSolution for evaluation
                // Create equal weights for selected assets (simplified conversion)
                let num_assets = binary_sol.len();
                let weights: Vec<f64> = binary_sol
                    .iter()
                    .map(|&x| if x == 1 { 1.0 / num_assets as f64 } else { 0.0 })
                    .collect();
                let portfolio_sol = finance::PortfolioSolution {
                    weights,
                    metrics: finance::PortfolioMetrics {
                        expected_return: 0.0,
                        volatility: 0.0,
                        sharpe_ratio: 0.0,
                        max_drawdown: 0.0,
                        var_95: 0.0,
                        cvar_95: 0.0,
                    },
                };
                self.inner.evaluate_solution(&portfolio_sol)
            }
            UnifiedSolution::Custom(_) => {
                // Custom solutions not supported for portfolio optimization
                Err(ApplicationError::OptimizationError(
                    "Custom solution format not supported for portfolio optimization".to_string(),
                ))
            }
            _ => Err(ApplicationError::OptimizationError(
                "Unsupported solution type for portfolio optimization".to_string(),
            )),
        }
    }

    fn is_feasible(&self, solution: &Self::Solution) -> bool {
        match solution {
            UnifiedSolution::Binary(binary_sol) => {
                // Convert binary solution to PortfolioSolution for feasibility check
                let num_assets = binary_sol.len();
                let weights: Vec<f64> = binary_sol
                    .iter()
                    .map(|&x| if x == 1 { 1.0 / num_assets as f64 } else { 0.0 })
                    .collect();
                let portfolio_sol = finance::PortfolioSolution {
                    weights,
                    metrics: finance::PortfolioMetrics {
                        expected_return: 0.0,
                        volatility: 0.0,
                        sharpe_ratio: 0.0,
                        max_drawdown: 0.0,
                        var_95: 0.0,
                        cvar_95: 0.0,
                    },
                };
                self.inner.is_feasible(&portfolio_sol)
            }
            UnifiedSolution::Custom(_) => {
                // Custom solution format not supported for portfolio optimization
                false
            }
            _ => false,
        }
    }
}

impl UnifiedProblem for UnifiedPortfolioOptimization {
    fn category(&self) -> ProblemCategory {
        ProblemCategory::Portfolio
    }

    fn industry(&self) -> &'static str {
        "finance"
    }

    fn complexity(&self) -> ProblemComplexity {
        let num_assets = self.inner.expected_returns.len();
        match num_assets {
            0..=10 => ProblemComplexity::Small,
            11..=50 => ProblemComplexity::Medium,
            51..=200 => ProblemComplexity::Large,
            _ => ProblemComplexity::ExtraLarge,
        }
    }

    fn recommended_solver_config(&self) -> SolverConfiguration {
        SolverConfiguration {
            solver_type: SolverType::Classical,
            annealing_params: AnnealingParams {
                num_sweeps: 10_000,
                num_repetitions: 20,
                initial_temperature: 2.0,
                final_temperature: 0.01,
                ..Default::default()
            },
            hardware_requirements: HardwareRequirements {
                min_memory_gb: 2.0,
                recommended_cores: 4,
                gpu_acceleration: false,
                quantum_hardware: false,
            },
            optimization_hints: vec![OptimizationHint::HighPrecision],
        }
    }

    fn constraints(&self) -> Vec<IndustryConstraint> {
        let mut constraints = vec![IndustryConstraint::Budget {
            limit: self.inner.budget,
        }];

        // Add regulatory constraints if any
        constraints.extend(self.inner.regulatory_constraints.clone());

        constraints
    }

    fn objective(&self) -> IndustryObjective {
        IndustryObjective::MultiObjective(vec![
            (IndustryObjective::MaximizeProfit, 1.0),
            (IndustryObjective::MinimizeRisk, self.inner.risk_tolerance),
        ])
    }

    fn solution_bounds(&self) -> Option<(f64, f64)> {
        // Rough bounds based on expected returns and risk
        let max_return: f64 = self.inner.expected_returns.iter().sum();
        let min_return = self
            .inner
            .expected_returns
            .iter()
            .fold(0.0f64, |a, &b| a.min(b));
        Some((min_return, max_return))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_unified(&self) -> Box<dyn UnifiedProblem> {
        Box::new(self.clone())
    }
}

// Implement OptimizationProblem for other wrapper types
impl OptimizationProblem for UnifiedVehicleRoutingProblem {
    type Solution = UnifiedSolution;
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
        match solution {
            UnifiedSolution::Binary(binary_sol) => {
                // Simple conversion for VRP - treat as route assignments
                let route: Vec<usize> = binary_sol
                    .iter()
                    .enumerate()
                    .filter(|(_, &x)| x == 1)
                    .map(|(i, _)| i)
                    .collect();
                let route_stats = logistics::RouteStatistics {
                    vehicle_id: 0,
                    distance: 0.0,
                    duration: 0.0,
                    capacity_utilization: 0.5,
                    customers_served: route.len(),
                    time_violations: 0,
                };
                let route_sol = logistics::VehicleRoutingSolution {
                    routes: vec![route],
                    total_distance: 0.0,
                    total_cost: 0.0,
                    route_stats: vec![route_stats],
                };
                self.inner.evaluate_solution(&route_sol)
            }
            _ => Err(ApplicationError::OptimizationError(
                "Unsupported solution type for VRP".to_string(),
            )),
        }
    }

    fn is_feasible(&self, solution: &Self::Solution) -> bool {
        match solution {
            UnifiedSolution::Binary(binary_sol) => {
                let route: Vec<usize> = binary_sol
                    .iter()
                    .enumerate()
                    .filter(|(_, &x)| x == 1)
                    .map(|(i, _)| i)
                    .collect();
                let route_stats = logistics::RouteStatistics {
                    vehicle_id: 0,
                    distance: 0.0,
                    duration: 0.0,
                    capacity_utilization: 0.5,
                    customers_served: route.len(),
                    time_violations: 0,
                };
                let route_sol = logistics::VehicleRoutingSolution {
                    routes: vec![route],
                    total_distance: 0.0,
                    total_cost: 0.0,
                    route_stats: vec![route_stats],
                };
                self.inner.is_feasible(&route_sol)
            }
            _ => false,
        }
    }
}

impl UnifiedProblem for UnifiedVehicleRoutingProblem {
    fn category(&self) -> ProblemCategory {
        ProblemCategory::Routing
    }

    fn industry(&self) -> &'static str {
        "logistics"
    }

    fn complexity(&self) -> ProblemComplexity {
        let num_locations = self.inner.locations.len();
        match num_locations {
            0..=20 => ProblemComplexity::Small,
            21..=100 => ProblemComplexity::Medium,
            101..=500 => ProblemComplexity::Large,
            _ => ProblemComplexity::ExtraLarge,
        }
    }

    fn recommended_solver_config(&self) -> SolverConfiguration {
        SolverConfiguration {
            solver_type: SolverType::Hybrid,
            annealing_params: AnnealingParams::default(),
            hardware_requirements: HardwareRequirements {
                min_memory_gb: 4.0,
                recommended_cores: 8,
                gpu_acceleration: true,
                quantum_hardware: false,
            },
            optimization_hints: vec![OptimizationHint::MultiModal],
        }
    }

    fn constraints(&self) -> Vec<IndustryConstraint> {
        vec![IndustryConstraint::Capacity {
            resource: "vehicle".to_string(),
            limit: 100.0,
        }]
    }

    fn objective(&self) -> IndustryObjective {
        IndustryObjective::MinimizeCost
    }

    fn solution_bounds(&self) -> Option<(f64, f64)> {
        Some((0.0, 1000.0)) // Rough bounds for VRP cost
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_unified(&self) -> Box<dyn UnifiedProblem> {
        Box::new(self.clone())
    }
}

// Implement for other wrapper types with minimal implementations
impl OptimizationProblem for UnifiedNetworkTopologyOptimization {
    type Solution = UnifiedSolution;
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
        match solution {
            UnifiedSolution::Binary(binary_sol) => {
                let network_topology = telecommunications::NetworkTopology {
                    active_connections: binary_sol.iter().map(|&x| x == 1).collect(),
                    total_cost: 0.0,
                    connectivity: 0.8,
                    average_latency: 10.0,
                    performance_metrics: telecommunications::TelecomMetrics {
                        throughput: 10.0,
                        packet_loss_rate: 0.01,
                        jitter: 5.0,
                        availability: 0.99,
                        mtbf: 1000.0,
                        coverage_area: 100.0,
                    },
                };
                self.inner.evaluate_solution(&network_topology)
            }
            _ => Err(ApplicationError::OptimizationError(
                "Unsupported solution type for network optimization".to_string(),
            )),
        }
    }

    fn is_feasible(&self, solution: &Self::Solution) -> bool {
        match solution {
            UnifiedSolution::Binary(binary_sol) => {
                let network_topology = telecommunications::NetworkTopology {
                    active_connections: binary_sol.iter().map(|&x| x == 1).collect(),
                    total_cost: 0.0,
                    connectivity: 0.8,
                    average_latency: 10.0,
                    performance_metrics: telecommunications::TelecomMetrics {
                        throughput: 10.0,
                        packet_loss_rate: 0.01,
                        jitter: 5.0,
                        availability: 0.99,
                        mtbf: 1000.0,
                        coverage_area: 100.0,
                    },
                };
                self.inner.is_feasible(&network_topology)
            }
            _ => false,
        }
    }
}

impl UnifiedProblem for UnifiedNetworkTopologyOptimization {
    fn category(&self) -> ProblemCategory {
        ProblemCategory::NetworkDesign
    }

    fn industry(&self) -> &'static str {
        "telecommunications"
    }

    fn complexity(&self) -> ProblemComplexity {
        ProblemComplexity::Medium // Default complexity
    }

    fn recommended_solver_config(&self) -> SolverConfiguration {
        SolverConfiguration {
            solver_type: SolverType::QuantumSimulator,
            annealing_params: AnnealingParams::default(),
            hardware_requirements: HardwareRequirements {
                min_memory_gb: 8.0,
                recommended_cores: 16,
                gpu_acceleration: true,
                quantum_hardware: false,
            },
            optimization_hints: vec![OptimizationHint::SparseStructure],
        }
    }

    fn constraints(&self) -> Vec<IndustryConstraint> {
        vec![]
    }

    fn objective(&self) -> IndustryObjective {
        IndustryObjective::MinimizeCost
    }

    fn solution_bounds(&self) -> Option<(f64, f64)> {
        Some((0.0, 500.0))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_unified(&self) -> Box<dyn UnifiedProblem> {
        Box::new(self.clone())
    }
}

// Placeholder implementations for other wrapper types
impl OptimizationProblem for UnifiedEnergyGridOptimization {
    type Solution = UnifiedSolution;
    type ObjectiveValue = f64;

    fn description(&self) -> String {
        "Energy grid optimization problem".to_string()
    }

    fn size_metrics(&self) -> HashMap<String, usize> {
        HashMap::new()
    }

    fn validate(&self) -> ApplicationResult<()> {
        Ok(())
    }

    fn to_qubo(&self) -> ApplicationResult<(crate::ising::QuboModel, HashMap<String, usize>)> {
        Err(ApplicationError::OptimizationError(
            "Energy grid QUBO not implemented".to_string(),
        ))
    }

    fn evaluate_solution(
        &self,
        _solution: &Self::Solution,
    ) -> ApplicationResult<Self::ObjectiveValue> {
        Ok(0.0)
    }

    fn is_feasible(&self, _solution: &Self::Solution) -> bool {
        true
    }
}

impl UnifiedProblem for UnifiedEnergyGridOptimization {
    fn category(&self) -> ProblemCategory {
        ProblemCategory::ResourceAllocation
    }

    fn industry(&self) -> &'static str {
        "energy"
    }

    fn complexity(&self) -> ProblemComplexity {
        ProblemComplexity::Medium
    }

    fn recommended_solver_config(&self) -> SolverConfiguration {
        SolverConfiguration {
            solver_type: SolverType::Classical,
            annealing_params: AnnealingParams::default(),
            hardware_requirements: HardwareRequirements {
                min_memory_gb: 4.0,
                recommended_cores: 8,
                gpu_acceleration: false,
                quantum_hardware: false,
            },
            optimization_hints: vec![],
        }
    }

    fn constraints(&self) -> Vec<IndustryConstraint> {
        vec![]
    }

    fn objective(&self) -> IndustryObjective {
        IndustryObjective::MinimizeCost
    }

    fn solution_bounds(&self) -> Option<(f64, f64)> {
        Some((0.0, 100.0))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_unified(&self) -> Box<dyn UnifiedProblem> {
        Box::new(self.clone())
    }
}

impl OptimizationProblem for UnifiedManufacturingScheduling {
    type Solution = UnifiedSolution;
    type ObjectiveValue = f64;

    fn description(&self) -> String {
        "Manufacturing scheduling problem".to_string()
    }

    fn size_metrics(&self) -> HashMap<String, usize> {
        HashMap::new()
    }

    fn validate(&self) -> ApplicationResult<()> {
        Ok(())
    }

    fn to_qubo(&self) -> ApplicationResult<(crate::ising::QuboModel, HashMap<String, usize>)> {
        Err(ApplicationError::OptimizationError(
            "Manufacturing QUBO not implemented".to_string(),
        ))
    }

    fn evaluate_solution(
        &self,
        _solution: &Self::Solution,
    ) -> ApplicationResult<Self::ObjectiveValue> {
        Ok(0.0)
    }

    fn is_feasible(&self, _solution: &Self::Solution) -> bool {
        true
    }
}

impl UnifiedProblem for UnifiedManufacturingScheduling {
    fn category(&self) -> ProblemCategory {
        ProblemCategory::ResourceAllocation
    }

    fn industry(&self) -> &'static str {
        "manufacturing"
    }

    fn complexity(&self) -> ProblemComplexity {
        ProblemComplexity::Medium
    }

    fn recommended_solver_config(&self) -> SolverConfiguration {
        SolverConfiguration {
            solver_type: SolverType::Classical,
            annealing_params: AnnealingParams::default(),
            hardware_requirements: HardwareRequirements {
                min_memory_gb: 2.0,
                recommended_cores: 4,
                gpu_acceleration: false,
                quantum_hardware: false,
            },
            optimization_hints: vec![],
        }
    }

    fn constraints(&self) -> Vec<IndustryConstraint> {
        vec![]
    }

    fn objective(&self) -> IndustryObjective {
        IndustryObjective::MaximizeEfficiency
    }

    fn solution_bounds(&self) -> Option<(f64, f64)> {
        Some((0.0, 100.0))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_unified(&self) -> Box<dyn UnifiedProblem> {
        Box::new(self.clone())
    }
}

impl OptimizationProblem for UnifiedHealthcareResourceOptimization {
    type Solution = UnifiedSolution;
    type ObjectiveValue = f64;

    fn description(&self) -> String {
        "Healthcare resource optimization problem".to_string()
    }

    fn size_metrics(&self) -> HashMap<String, usize> {
        HashMap::new()
    }

    fn validate(&self) -> ApplicationResult<()> {
        Ok(())
    }

    fn to_qubo(&self) -> ApplicationResult<(crate::ising::QuboModel, HashMap<String, usize>)> {
        Err(ApplicationError::OptimizationError(
            "Healthcare QUBO not implemented".to_string(),
        ))
    }

    fn evaluate_solution(
        &self,
        _solution: &Self::Solution,
    ) -> ApplicationResult<Self::ObjectiveValue> {
        Ok(0.0)
    }

    fn is_feasible(&self, _solution: &Self::Solution) -> bool {
        true
    }
}

impl UnifiedProblem for UnifiedHealthcareResourceOptimization {
    fn category(&self) -> ProblemCategory {
        ProblemCategory::ResourceAllocation
    }

    fn industry(&self) -> &'static str {
        "healthcare"
    }

    fn complexity(&self) -> ProblemComplexity {
        ProblemComplexity::Medium
    }

    fn recommended_solver_config(&self) -> SolverConfiguration {
        SolverConfiguration {
            solver_type: SolverType::Classical,
            annealing_params: AnnealingParams::default(),
            hardware_requirements: HardwareRequirements {
                min_memory_gb: 2.0,
                recommended_cores: 4,
                gpu_acceleration: false,
                quantum_hardware: false,
            },
            optimization_hints: vec![],
        }
    }

    fn constraints(&self) -> Vec<IndustryConstraint> {
        vec![]
    }

    fn objective(&self) -> IndustryObjective {
        IndustryObjective::MaximizeEfficiency
    }

    fn solution_bounds(&self) -> Option<(f64, f64)> {
        Some((0.0, 100.0))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_unified(&self) -> Box<dyn UnifiedProblem> {
        Box::new(self.clone())
    }
}

/// Performance benchmarking for the unified solver
pub fn run_unified_benchmark(
    factory: &UnifiedSolverFactory,
    industry: &str,
    problem_sizes: Vec<usize>,
) -> ApplicationResult<UnifiedBenchmarkResults> {
    let mut results = UnifiedBenchmarkResults::new();

    for size in problem_sizes {
        // Create benchmark problems
        let problems = super::create_benchmark_suite(industry, &format!("{size}"))?;

        for (i, problem) in problems.iter().enumerate() {
            let start_time = std::time::Instant::now();

            // This would require converting to UnifiedProblem
            // For now, placeholder measurement
            let solve_time = start_time.elapsed().as_secs_f64() * 1000.0;

            results.add_result(
                format!("{industry}_{size}_problem_{i}"),
                solve_time,
                0.0,
                true,
            );
        }
    }

    Ok(results)
}

/// Benchmark results for unified solver performance
#[derive(Debug, Clone)]
pub struct UnifiedBenchmarkResults {
    /// Results by problem identifier
    pub results: HashMap<String, BenchmarkResult>,
    /// Overall statistics
    pub statistics: BenchmarkStatistics,
}

impl UnifiedBenchmarkResults {
    fn new() -> Self {
        Self {
            results: HashMap::new(),
            statistics: BenchmarkStatistics::default(),
        }
    }

    fn add_result(
        &mut self,
        problem_id: String,
        solve_time: f64,
        objective_value: f64,
        converged: bool,
    ) {
        let result = BenchmarkResult {
            solve_time_ms: solve_time,
            objective_value,
            converged,
        };

        self.results.insert(problem_id, result);
        self.update_statistics();
    }

    fn update_statistics(&mut self) {
        let solve_times: Vec<f64> = self.results.values().map(|r| r.solve_time_ms).collect();
        let convergence_rate = self.results.values().filter(|r| r.converged).count() as f64
            / self.results.len() as f64;

        self.statistics = BenchmarkStatistics {
            avg_solve_time: solve_times.iter().sum::<f64>() / solve_times.len() as f64,
            min_solve_time: solve_times.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max_solve_time: solve_times.iter().fold(0.0, |a, &b| a.max(b)),
            convergence_rate,
            total_problems_solved: self.results.len(),
        };
    }
}

/// Individual benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub solve_time_ms: f64,
    pub objective_value: f64,
    pub converged: bool,
}

/// Overall benchmark statistics
#[derive(Debug, Clone, Default)]
pub struct BenchmarkStatistics {
    pub avg_solve_time: f64,
    pub min_solve_time: f64,
    pub max_solve_time: f64,
    pub convergence_rate: f64,
    pub total_problems_solved: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_solver_factory_creation() {
        let factory = UnifiedSolverFactory::new();
        assert!(!factory.default_configs.is_empty());
        assert!(factory.available_solvers.contains(&SolverType::Classical));
    }

    #[test]
    fn test_problem_creation() {
        let factory = UnifiedSolverFactory::new();
        let config = HashMap::from([
            ("num_assets".to_string(), json!(5)),
            ("budget".to_string(), json!(100_000.0)),
            ("risk_tolerance".to_string(), json!(0.3)),
        ]);

        let problem = factory.create_problem("finance", "portfolio", config);
        assert!(problem.is_ok());

        let unified_problem = problem.expect("problem creation should succeed");
        assert_eq!(unified_problem.industry(), "finance");
        assert_eq!(unified_problem.category(), ProblemCategory::Portfolio);
    }

    #[test]
    fn test_complexity_classification() {
        let factory = UnifiedSolverFactory::new();
        let config = HashMap::from([("num_assets".to_string(), json!(5))]);

        let problem = factory
            .create_problem("finance", "portfolio", config)
            .expect("problem creation should succeed");
        assert_eq!(problem.complexity(), ProblemComplexity::Small);
    }

    #[test]
    fn test_solver_configuration_adjustment() {
        let factory = UnifiedSolverFactory::new();
        let mut config = SolverConfiguration {
            solver_type: SolverType::Classical,
            annealing_params: AnnealingParams::default(),
            hardware_requirements: HardwareRequirements {
                min_memory_gb: 1.0,
                recommended_cores: 2,
                gpu_acceleration: false,
                quantum_hardware: false,
            },
            optimization_hints: vec![],
        };

        factory.adjust_config_for_complexity(&mut config, ProblemComplexity::Large);
        assert!(config.annealing_params.num_sweeps >= 20_000);
        assert!(config.hardware_requirements.min_memory_gb >= 2.0);
    }
}
