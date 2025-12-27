//! Problem-specific annealing schedules
//!
//! This module provides optimized annealing schedules for specific problem types.
//! Different optimization problems benefit from different temperature and field
//! schedules based on their energy landscape characteristics.

use crate::ising::{IsingModel, IsingResult};
use crate::simulator::{AnnealingParams, TemperatureSchedule, TransverseFieldSchedule};
use std::collections::HashMap;

/// Problem types with specialized annealing schedules
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProblemType {
    /// Traveling Salesman Problem
    TSP,
    /// Maximum Cut problem
    MaxCut,
    /// Graph Coloring
    GraphColoring,
    /// Quadratic Assignment Problem
    QAP,
    /// Number Partitioning
    NumberPartitioning,
    /// Satisfiability (SAT)
    SAT,
    /// Portfolio Optimization
    Portfolio,
    /// Job Shop Scheduling
    JobShop,
    /// Spin Glass
    SpinGlass,
    /// Custom problem type
    Custom(String),
}

/// Schedule optimizer that creates problem-specific annealing parameters
pub struct ProblemSpecificScheduler {
    /// Problem characteristics analyzer
    analyzer: ProblemAnalyzer,
    /// Schedule templates for different problem types
    templates: HashMap<ProblemType, ScheduleTemplate>,
}

/// Template for annealing schedules
#[derive(Debug, Clone)]
pub struct ScheduleTemplate {
    /// Initial temperature factor (multiplied by problem scale)
    pub initial_temp_factor: f64,
    /// Final temperature factor
    pub final_temp_factor: f64,
    /// Temperature schedule type
    pub temp_schedule: TemperatureSchedule,
    /// Initial transverse field strength
    pub initial_field: f64,
    /// Transverse field schedule
    pub field_schedule: TransverseFieldSchedule,
    /// Number of sweeps factor (multiplied by problem size)
    pub sweeps_factor: f64,
    /// Number of repetitions
    pub num_repetitions: usize,
    /// Use parallel tempering
    pub use_parallel_tempering: bool,
    /// Custom parameters
    pub custom_params: HashMap<String, f64>,
}

impl Default for ScheduleTemplate {
    fn default() -> Self {
        Self {
            initial_temp_factor: 2.0,
            final_temp_factor: 0.01,
            temp_schedule: TemperatureSchedule::Exponential(3.0),
            initial_field: 2.0,
            field_schedule: TransverseFieldSchedule::Linear,
            sweeps_factor: 100.0,
            num_repetitions: 10,
            use_parallel_tempering: false,
            custom_params: HashMap::new(),
        }
    }
}

/// Analyzes problem characteristics to inform schedule optimization
pub struct ProblemAnalyzer {
    /// Coupling strength statistics
    coupling_stats: CouplingStatistics,
    /// Graph properties
    graph_properties: GraphProperties,
}

#[derive(Debug, Clone, Default)]
struct CouplingStatistics {
    mean_coupling: f64,
    std_coupling: f64,
    max_coupling: f64,
    min_coupling: f64,
    sparsity: f64,
}

#[derive(Debug, Clone, Default)]
struct GraphProperties {
    num_variables: usize,
    num_couplings: usize,
    avg_degree: f64,
    max_degree: usize,
    clustering_coefficient: f64,
    is_bipartite: bool,
}

impl ProblemSpecificScheduler {
    /// Create a new problem-specific scheduler
    #[must_use]
    pub fn new() -> Self {
        let mut scheduler = Self {
            analyzer: ProblemAnalyzer::new(),
            templates: HashMap::new(),
        };

        // Initialize with default templates
        scheduler.init_default_templates();
        scheduler
    }

    /// Initialize default schedule templates for known problem types
    fn init_default_templates(&mut self) {
        // TSP: Needs careful temperature control for constraint satisfaction
        self.templates.insert(
            ProblemType::TSP,
            ScheduleTemplate {
                initial_temp_factor: 5.0, // Higher initial temperature
                final_temp_factor: 0.001,
                temp_schedule: TemperatureSchedule::Geometric(0.95, 10.0),
                initial_field: 3.0,
                field_schedule: TransverseFieldSchedule::Exponential(2.0),
                sweeps_factor: 200.0, // More sweeps needed
                num_repetitions: 20,
                use_parallel_tempering: true,
                custom_params: [("constraint_weight".to_string(), 10.0)].into(),
            },
        );

        // MaxCut: Benefits from aggressive cooling
        self.templates.insert(
            ProblemType::MaxCut,
            ScheduleTemplate {
                initial_temp_factor: 1.5,
                final_temp_factor: 0.01,
                temp_schedule: TemperatureSchedule::Exponential(4.0),
                initial_field: 2.0,
                field_schedule: TransverseFieldSchedule::Linear,
                sweeps_factor: 50.0,
                num_repetitions: 10,
                use_parallel_tempering: false,
                custom_params: HashMap::new(),
            },
        );

        // Graph Coloring: Needs exploration of discrete states
        self.templates.insert(
            ProblemType::GraphColoring,
            ScheduleTemplate {
                initial_temp_factor: 3.0,
                final_temp_factor: 0.1, // Higher final temperature
                temp_schedule: TemperatureSchedule::Geometric(0.98, 5.0),
                initial_field: 2.5,
                field_schedule: TransverseFieldSchedule::Custom(|t, t_f| {
                    // Step-like decrease to maintain quantum fluctuations
                    if t < 0.7 * t_f {
                        2.5
                    } else {
                        0.5
                    }
                }),
                sweeps_factor: 150.0,
                num_repetitions: 15,
                use_parallel_tempering: true,
                custom_params: HashMap::new(),
            },
        );

        // Number Partitioning: Highly frustrated, needs careful approach
        self.templates.insert(
            ProblemType::NumberPartitioning,
            ScheduleTemplate {
                initial_temp_factor: 10.0, // Very high initial temperature
                final_temp_factor: 0.001,
                temp_schedule: TemperatureSchedule::Custom(|t, t_f| {
                    // Two-stage cooling
                    let progress = t / t_f;
                    if progress < 0.5 {
                        10.0 * (2.0 * progress).mul_add(-0.8, 1.0)
                    } else {
                        2.0 * 2.0f64.mul_add(-progress, 2.0).powi(2)
                    }
                }),
                initial_field: 4.0,
                field_schedule: TransverseFieldSchedule::Exponential(3.0),
                sweeps_factor: 300.0,
                num_repetitions: 30,
                use_parallel_tempering: true,
                custom_params: HashMap::new(),
            },
        );

        // Portfolio Optimization: Continuous-like problem
        self.templates.insert(
            ProblemType::Portfolio,
            ScheduleTemplate {
                initial_temp_factor: 1.0,
                final_temp_factor: 0.01,
                temp_schedule: TemperatureSchedule::Linear,
                initial_field: 1.5,
                field_schedule: TransverseFieldSchedule::Linear,
                sweeps_factor: 75.0,
                num_repetitions: 10,
                use_parallel_tempering: false,
                custom_params: [("risk_weight".to_string(), 1.0)].into(),
            },
        );

        // Spin Glass: Complex energy landscape
        self.templates.insert(
            ProblemType::SpinGlass,
            ScheduleTemplate {
                initial_temp_factor: 2.0,
                final_temp_factor: 0.001,
                temp_schedule: TemperatureSchedule::Exponential(2.5),
                initial_field: 3.0,
                field_schedule: TransverseFieldSchedule::Custom(|t, t_f| {
                    // Non-monotonic schedule for spin glasses
                    let progress = t / t_f;
                    3.0 * (1.0 - progress) * 0.3f64.mul_add((10.0 * progress).sin(), 1.0)
                }),
                sweeps_factor: 200.0,
                num_repetitions: 25,
                use_parallel_tempering: true,
                custom_params: HashMap::new(),
            },
        );
    }

    /// Create optimized annealing parameters for a specific problem type
    pub fn create_schedule(
        &mut self,
        model: &IsingModel,
        problem_type: ProblemType,
    ) -> IsingResult<AnnealingParams> {
        // Analyze the problem
        self.analyzer.analyze(model)?;

        // Get the template or use default
        let template = self
            .templates
            .get(&problem_type)
            .cloned()
            .unwrap_or_else(ScheduleTemplate::default);

        // Adapt template based on problem analysis
        let adapted = self.adapt_template(&template, &self.analyzer);

        // Create annealing parameters
        Ok(self.template_to_params(&adapted, model.num_qubits))
    }

    /// Automatically detect problem type from model structure
    pub fn detect_problem_type(&mut self, model: &IsingModel) -> IsingResult<ProblemType> {
        self.analyzer.analyze(model)?;

        // Simple heuristics for problem detection
        let props = &self.analyzer.graph_properties;
        let stats = &self.analyzer.coupling_stats;

        // Check for specific patterns
        if props.is_bipartite && stats.min_coupling < 0.0 && stats.max_coupling < 0.0 {
            return Ok(ProblemType::MaxCut);
        }

        if props.avg_degree as f64 > 0.8 * props.num_variables as f64 {
            // Dense problem, might be QAP or TSP
            if stats.std_coupling > stats.mean_coupling.abs() * 2.0 {
                return Ok(ProblemType::QAP);
            }
            return Ok(ProblemType::TSP);
        }

        if stats.sparsity < 0.1 && props.clustering_coefficient < 0.1 {
            return Ok(ProblemType::NumberPartitioning);
        }

        // Default to spin glass for unknown patterns
        Ok(ProblemType::SpinGlass)
    }

    /// Adapt template based on problem analysis
    fn adapt_template(
        &self,
        template: &ScheduleTemplate,
        analyzer: &ProblemAnalyzer,
    ) -> ScheduleTemplate {
        let mut adapted = template.clone();
        let props = &analyzer.graph_properties;
        let stats = &analyzer.coupling_stats;

        // Scale initial temperature based on coupling strengths
        let energy_scale = stats
            .max_coupling
            .abs()
            .max(stats.mean_coupling.abs() * 3.0);
        if energy_scale > 0.0 {
            adapted.initial_temp_factor *= energy_scale;
        }
        // If no couplings, keep the original factor

        // Adjust sweeps based on problem size and connectivity
        let size_factor = (props.num_variables as f64).sqrt();
        let connectivity_factor = (props.avg_degree / 4.0).max(1.0);
        adapted.sweeps_factor *= size_factor * connectivity_factor;

        // Enable parallel tempering for highly frustrated problems
        if stats.std_coupling > stats.mean_coupling.abs() {
            adapted.use_parallel_tempering = true;
        }

        adapted
    }

    /// Convert template to annealing parameters
    fn template_to_params(
        &self,
        template: &ScheduleTemplate,
        num_qubits: usize,
    ) -> AnnealingParams {
        let mut params = AnnealingParams::new();

        // Calculate base temperature from problem characteristics
        let coupling_scale = self.analyzer.coupling_stats.std_coupling.abs();
        let base_temp = if coupling_scale > 0.0 {
            coupling_scale
        } else {
            1.0
        };

        params.initial_temperature = template.initial_temp_factor * base_temp;
        params.final_temperature = template.final_temp_factor * base_temp;
        params.temperature_schedule = template.temp_schedule.clone();
        params.initial_transverse_field = template.initial_field;
        params.transverse_field_schedule = template.field_schedule.clone();
        params.num_sweeps = (template.sweeps_factor * num_qubits as f64) as usize;
        params.num_repetitions = template.num_repetitions;

        // Set updates per sweep based on problem size
        params.updates_per_sweep = Some(num_qubits * 10);

        params
    }

    /// Add custom schedule template
    pub fn add_custom_template(&mut self, name: String, template: ScheduleTemplate) {
        self.templates.insert(ProblemType::Custom(name), template);
    }
}

impl ProblemAnalyzer {
    /// Create a new problem analyzer
    fn new() -> Self {
        Self {
            coupling_stats: CouplingStatistics::default(),
            graph_properties: GraphProperties::default(),
        }
    }

    /// Analyze problem characteristics
    fn analyze(&mut self, model: &IsingModel) -> IsingResult<()> {
        // Reset statistics
        self.coupling_stats = CouplingStatistics::default();
        self.graph_properties = GraphProperties::default();

        // Basic properties
        self.graph_properties.num_variables = model.num_qubits;

        // Analyze couplings
        let couplings = model.couplings();
        self.graph_properties.num_couplings = couplings.len();

        if !couplings.is_empty() {
            let strengths: Vec<f64> = couplings.iter().map(|c| c.strength).collect();

            self.coupling_stats.mean_coupling =
                strengths.iter().sum::<f64>() / strengths.len() as f64;
            self.coupling_stats.max_coupling =
                strengths.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            self.coupling_stats.min_coupling =
                strengths.iter().copied().fold(f64::INFINITY, f64::min);

            // Calculate standard deviation
            let variance = strengths
                .iter()
                .map(|&x| (x - self.coupling_stats.mean_coupling).powi(2))
                .sum::<f64>()
                / strengths.len() as f64;
            self.coupling_stats.std_coupling = variance.sqrt();
        }

        // Calculate graph properties
        let mut degrees = vec![0; model.num_qubits];
        for coupling in &couplings {
            degrees[coupling.i] += 1;
            degrees[coupling.j] += 1;
        }

        self.graph_properties.avg_degree =
            degrees.iter().sum::<usize>() as f64 / model.num_qubits as f64;
        self.graph_properties.max_degree = degrees.iter().copied().max().unwrap_or(0);

        // Sparsity
        let max_edges = model.num_qubits * (model.num_qubits - 1) / 2;
        self.coupling_stats.sparsity = if max_edges > 0 {
            couplings.len() as f64 / max_edges as f64
        } else {
            0.0
        };

        // Simple bipartite check (can be improved)
        self.graph_properties.is_bipartite = self.check_bipartite(model, &couplings);

        Ok(())
    }

    /// Check if the coupling graph is bipartite
    fn check_bipartite(&self, model: &IsingModel, couplings: &[crate::ising::Coupling]) -> bool {
        let mut colors = vec![None; model.num_qubits];
        let mut queue = Vec::new();

        // Try to 2-color the graph
        for start in 0..model.num_qubits {
            if colors[start].is_some() {
                continue;
            }

            colors[start] = Some(0);
            queue.push(start);

            while let Some(node) = queue.pop() {
                let node_color = colors[node].expect("node should have a color assigned");

                for coupling in couplings {
                    let neighbor = if coupling.i == node {
                        coupling.j
                    } else if coupling.j == node {
                        coupling.i
                    } else {
                        continue;
                    };

                    match colors[neighbor] {
                        None => {
                            colors[neighbor] = Some(1 - node_color);
                            queue.push(neighbor);
                        }
                        Some(color) if color == node_color => return false,
                        _ => {}
                    }
                }
            }
        }

        true
    }
}

/// Schedule optimization based on runtime performance
pub struct AdaptiveScheduleOptimizer {
    /// History of schedule performance
    performance_history: Vec<SchedulePerformance>,
    /// Learning rate for schedule adaptation
    learning_rate: f64,
}

#[derive(Debug, Clone)]
struct SchedulePerformance {
    problem_type: ProblemType,
    schedule_params: HashMap<String, f64>,
    final_energy: f64,
    convergence_time: f64,
    success_rate: f64,
}

impl AdaptiveScheduleOptimizer {
    /// Create a new adaptive optimizer
    #[must_use]
    pub const fn new(learning_rate: f64) -> Self {
        Self {
            performance_history: Vec::new(),
            learning_rate,
        }
    }

    /// Record performance of a schedule
    pub fn record_performance(
        &mut self,
        problem_type: ProblemType,
        params: &AnnealingParams,
        energy: f64,
        time: f64,
        success: bool,
    ) {
        let schedule_params = HashMap::from([
            ("initial_temp".to_string(), params.initial_temperature),
            ("initial_field".to_string(), params.initial_transverse_field),
            ("num_sweeps".to_string(), params.num_sweeps as f64),
        ]);

        self.performance_history.push(SchedulePerformance {
            problem_type,
            schedule_params,
            final_energy: energy,
            convergence_time: time,
            success_rate: if success { 1.0 } else { 0.0 },
        });
    }

    /// Suggest improved schedule based on history
    #[must_use]
    pub fn suggest_schedule(
        &self,
        problem_type: ProblemType,
        base_params: &AnnealingParams,
    ) -> AnnealingParams {
        let mut params = base_params.clone();

        // Find similar successful runs
        let similar_runs: Vec<_> = self
            .performance_history
            .iter()
            .filter(|p| p.problem_type == problem_type && p.success_rate > 0.8)
            .collect();

        if !similar_runs.is_empty() {
            // Average successful parameters
            let avg_temp = similar_runs
                .iter()
                .filter_map(|p| p.schedule_params.get("initial_temp"))
                .sum::<f64>()
                / similar_runs.len() as f64;

            let avg_field = similar_runs
                .iter()
                .filter_map(|p| p.schedule_params.get("initial_field"))
                .sum::<f64>()
                / similar_runs.len() as f64;

            // Blend with current parameters
            params.initial_temperature = params
                .initial_temperature
                .mul_add(1.0 - self.learning_rate, avg_temp * self.learning_rate);
            params.initial_transverse_field = params
                .initial_transverse_field
                .mul_add(1.0 - self.learning_rate, avg_field * self.learning_rate);
        }

        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_creation() {
        let mut scheduler = ProblemSpecificScheduler::new();
        let model = IsingModel::new(10);

        let params = scheduler
            .create_schedule(&model, ProblemType::MaxCut)
            .expect("failed to create schedule in test");
        assert!(params.initial_temperature > 0.0);
        assert!(params.num_sweeps > 0);
    }

    #[test]
    fn test_problem_detection() {
        let mut scheduler = ProblemSpecificScheduler::new();
        let mut model = IsingModel::new(4);

        // Create a simple bipartite structure
        model
            .set_coupling(0, 2, -1.0)
            .expect("failed to set coupling in test");
        model
            .set_coupling(0, 3, -1.0)
            .expect("failed to set coupling in test");
        model
            .set_coupling(1, 2, -1.0)
            .expect("failed to set coupling in test");
        model
            .set_coupling(1, 3, -1.0)
            .expect("failed to set coupling in test");

        let detected = scheduler
            .detect_problem_type(&model)
            .expect("failed to detect problem type in test");
        assert_eq!(detected, ProblemType::MaxCut);
    }

    #[test]
    fn test_adaptive_optimizer() {
        let mut optimizer = AdaptiveScheduleOptimizer::new(0.3);
        let params = AnnealingParams::new();

        // Record some performance data
        optimizer.record_performance(ProblemType::TSP, &params, -100.0, 1.5, true);

        // Get suggested parameters
        let suggested = optimizer.suggest_schedule(ProblemType::TSP, &params);
        assert!(suggested.initial_temperature > 0.0);
    }
}
