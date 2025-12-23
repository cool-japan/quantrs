//! Solution debugger for quantum optimization.
//!
//! This module provides comprehensive debugging tools for analyzing
//! quantum optimization solutions, constraint violations, and solution quality.

pub mod analysis;
pub mod comparison;
pub mod config;
pub mod constraint_analyzer;
pub mod energy_analyzer;
pub mod reporting;
pub mod types;
pub mod visualization;

#[cfg(feature = "dwave")]
use crate::compile::{Compile, CompiledModel};

// Re-export main types
pub use analysis::*;
pub use comparison::*;
pub use config::*;
pub use constraint_analyzer::*;
pub use energy_analyzer::*;
pub use reporting::*;
pub use types::*;
pub use visualization::*;

/// Solution debugger
pub struct SolutionDebugger {
    /// Configuration
    config: config::DebuggerConfig,
    /// Problem information
    problem_info: types::ProblemInfo,
    /// Constraint analyzer
    constraint_analyzer: constraint_analyzer::ConstraintAnalyzer,
    /// Energy analyzer
    energy_analyzer: energy_analyzer::EnergyAnalyzer,
    /// Solution comparator
    comparator: comparison::SolutionComparator,
    /// Visualization engine
    visualizer: visualization::SolutionVisualizer,
}

impl SolutionDebugger {
    /// Create new debugger
    pub fn new(problem_info: types::ProblemInfo, config: config::DebuggerConfig) -> Self {
        Self {
            config,
            problem_info,
            constraint_analyzer: constraint_analyzer::ConstraintAnalyzer::new(1e-6),
            energy_analyzer: energy_analyzer::EnergyAnalyzer::new(2),
            comparator: comparison::SolutionComparator::new(),
            visualizer: visualization::SolutionVisualizer::new(),
        }
    }

    /// Debug solution
    pub fn debug_solution(&mut self, solution: &types::Solution) -> reporting::DebugReport {
        let mut report = reporting::DebugReport {
            solution: solution.clone(),
            constraint_analysis: None,
            energy_analysis: None,
            comparison_results: Vec::new(),
            visualizations: Vec::new(),
            issues: Vec::new(),
            suggestions: Vec::new(),
            summary: reporting::DebugSummary::default(),
        };

        // Analyze constraints
        if self.config.check_constraints {
            report.constraint_analysis = Some(self.analyze_constraints(solution));
        }

        // Analyze energy
        if self.config.analyze_energy {
            report.energy_analysis = Some(self.analyze_energy(solution));
        }

        // Compare with known solutions
        if self.config.compare_solutions {
            if let Some(ref optimal) = self.problem_info.optimal_solution {
                report
                    .comparison_results
                    .push(self.compare_solutions(solution, optimal));
            }
        }

        // Generate visualizations
        if self.config.generate_visuals {
            report.visualizations = self.generate_visualizations(solution);
        }

        // Identify issues
        report.issues = self.identify_issues(&report);

        // Generate suggestions
        report.suggestions = self.generate_suggestions(&report);

        // Generate summary
        report.summary = self.generate_summary(&report);

        report
    }

    /// Analyze constraints
    fn analyze_constraints(&mut self, solution: &types::Solution) -> analysis::ConstraintAnalysis {
        let violations = self
            .constraint_analyzer
            .analyze(&self.problem_info.constraints, &solution.assignments);

        let satisfied_count = self.problem_info.constraints.len() - violations.len();
        let satisfaction_rate = satisfied_count as f64 / self.problem_info.constraints.len() as f64;

        analysis::ConstraintAnalysis {
            total_constraints: self.problem_info.constraints.len(),
            satisfied: satisfied_count,
            violated: violations.len(),
            satisfaction_rate,
            penalty_incurred: violations
                .iter()
                .map(|v| v.constraint.penalty * v.violation_amount)
                .sum(),
            violations,
        }
    }

    /// Analyze energy
    fn analyze_energy(&mut self, solution: &types::Solution) -> analysis::EnergyAnalysis {
        let breakdown = self.energy_analyzer.analyze(
            &self.problem_info.qubo,
            &solution.assignments,
            &self.problem_info.var_map,
        );

        // Find critical variables
        let mut critical_vars: Vec<_> = breakdown
            .variable_contributions
            .iter()
            .map(|(var, contrib)| (var.clone(), contrib.abs()))
            .collect();
        critical_vars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        critical_vars.truncate(10);

        // Find critical interactions
        let mut critical_interactions: Vec<_> = breakdown
            .interaction_contributions
            .iter()
            .map(|(vars, contrib)| (vars.clone(), contrib.abs()))
            .collect();
        critical_interactions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        critical_interactions.truncate(10);

        analysis::EnergyAnalysis {
            total_energy: breakdown.total_energy,
            breakdown: breakdown.clone(),
            critical_variables: critical_vars,
            critical_interactions,
            improvement_potential: self.estimate_improvement_potential(&breakdown),
        }
    }

    /// Compare solutions
    fn compare_solutions(
        &self,
        sol1: &types::Solution,
        sol2: &types::Solution,
    ) -> comparison::ComparisonResult {
        self.comparator.compare(sol1, sol2, &self.problem_info)
    }

    /// Generate visualizations
    fn generate_visualizations(
        &self,
        solution: &types::Solution,
    ) -> Vec<visualization::Visualization> {
        let mut visualizations = Vec::new();

        // Solution matrix visualization
        visualizations.push(
            self.visualizer
                .visualize_solution_matrix(solution, &self.problem_info),
        );

        // Energy landscape visualization
        visualizations.push(
            self.visualizer
                .visualize_energy_landscape(solution, &self.problem_info),
        );

        // Constraint graph visualization
        visualizations.push(
            self.visualizer
                .visualize_constraint_graph(solution, &self.problem_info),
        );

        visualizations
    }

    /// Identify issues
    fn identify_issues(&self, report: &reporting::DebugReport) -> Vec<reporting::Issue> {
        let mut issues = Vec::new();

        // Check constraint violations
        if let Some(ref constraint_analysis) = report.constraint_analysis {
            if constraint_analysis.satisfaction_rate < 0.95 {
                issues.push(reporting::Issue {
                    severity: reporting::IssueSeverity::High,
                    category: "Constraints".to_string(),
                    description: format!(
                        "Only {:.1}% of constraints satisfied",
                        constraint_analysis.satisfaction_rate * 100.0
                    ),
                    location: "Constraint analysis".to_string(),
                    suggested_action: "Review constraint violations and adjust solution"
                        .to_string(),
                });
            }
        }

        // Check energy analysis
        if let Some(ref energy_analysis) = report.energy_analysis {
            if energy_analysis.improvement_potential > 0.1 {
                issues.push(reporting::Issue {
                    severity: reporting::IssueSeverity::Medium,
                    category: "Energy".to_string(),
                    description: "Significant energy improvement potential detected".to_string(),
                    location: "Energy analysis".to_string(),
                    suggested_action: "Consider local optimization or different sampling strategy"
                        .to_string(),
                });
            }
        }

        issues
    }

    /// Generate suggestions
    fn generate_suggestions(&self, report: &reporting::DebugReport) -> Vec<reporting::Suggestion> {
        let mut suggestions = Vec::new();

        // Suggestions based on constraint violations
        if let Some(ref constraint_analysis) = report.constraint_analysis {
            for violation in &constraint_analysis.violations {
                if !violation.suggested_fixes.is_empty() {
                    suggestions.push(reporting::Suggestion {
                        category: "Constraint Fix".to_string(),
                        description: format!(
                            "Fix violation in constraint: {}",
                            violation
                                .constraint
                                .name
                                .as_ref()
                                .unwrap_or(&"unnamed".to_string())
                        ),
                        impact: violation.violation_amount,
                        feasibility: 0.8,
                        action_steps: violation
                            .suggested_fixes
                            .iter()
                            .map(|fix| fix.description.clone())
                            .collect(),
                    });
                }
            }
        }

        suggestions
    }

    /// Generate summary
    fn generate_summary(&self, report: &reporting::DebugReport) -> reporting::DebugSummary {
        let mut summary = reporting::DebugSummary::default();

        summary.total_issues = report.issues.len();
        summary.critical_issues = report
            .issues
            .iter()
            .filter(|i| matches!(i.severity, reporting::IssueSeverity::Critical))
            .count();
        summary.suggestions_count = report.suggestions.len();

        if let Some(ref constraint_analysis) = report.constraint_analysis {
            summary.constraint_satisfaction_rate = constraint_analysis.satisfaction_rate;
        }

        if let Some(ref energy_analysis) = report.energy_analysis {
            summary.total_energy = energy_analysis.total_energy;
            summary.improvement_potential = energy_analysis.improvement_potential;
        }

        // Calculate overall score
        summary.overall_score = self.calculate_overall_score(&summary);

        summary
    }

    /// Calculate overall solution score
    fn calculate_overall_score(&self, summary: &reporting::DebugSummary) -> f64 {
        let mut score = 1.0;

        // Penalty for constraint violations
        score *= summary.constraint_satisfaction_rate;

        // Penalty for critical issues
        if summary.critical_issues > 0 {
            score *= 0.5;
        }

        // Penalty for high improvement potential
        score *= (1.0 - summary.improvement_potential).max(0.0);

        score.max(0.0).min(1.0)
    }

    /// Estimate improvement potential
    fn estimate_improvement_potential(&self, breakdown: &energy_analyzer::EnergyBreakdown) -> f64 {
        // Simple heuristic: if energy landscape shows nearby local minima with lower energy
        let current_energy = breakdown.total_energy;
        let best_nearby = breakdown
            .energy_landscape
            .local_minima
            .iter()
            .map(|m| m.energy)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(current_energy);

        if best_nearby < current_energy {
            ((current_energy - best_nearby) / current_energy.abs()).min(1.0)
        } else {
            0.0
        }
    }
}

/// Interactive debugger for real-time solution analysis
pub struct InteractiveDebugger {
    /// Problem information
    problem_info: types::ProblemInfo,
    /// Current solution being debugged
    pub current_solution: Option<types::Solution>,
    /// Debugger instance
    debugger: SolutionDebugger,
    /// Watch variables
    pub watch_variables: Vec<String>,
}

impl InteractiveDebugger {
    /// Create new interactive debugger
    pub fn new(problem_info: types::ProblemInfo) -> Self {
        let config = config::DebuggerConfig {
            detailed_analysis: true,
            check_constraints: true,
            analyze_energy: true,
            compare_solutions: false,
            generate_visuals: false,
            output_format: config::DebugOutputFormat::Console,
            verbosity: config::VerbosityLevel::Normal,
        };

        Self {
            debugger: SolutionDebugger::new(problem_info.clone(), config),
            problem_info,
            current_solution: None,
            watch_variables: Vec::new(),
        }
    }

    /// Load a solution for debugging
    pub fn load_solution(&mut self, solution: types::Solution) {
        self.current_solution = Some(solution);
    }

    /// Add a variable to watch list
    pub fn add_watch(&mut self, variable: String) {
        if !self.watch_variables.contains(&variable) {
            self.watch_variables.push(variable);
        }
    }

    /// Execute a debug command
    pub fn execute_command(&mut self, command: &str) -> String {
        match command {
            "help" => "Available commands: help, energy, constraints, watch".to_string(),
            "energy" => {
                if let Some(ref solution) = self.current_solution {
                    format!("Solution energy: {}", solution.energy)
                } else {
                    "No solution loaded".to_string()
                }
            }
            "constraints" => {
                if let Some(ref solution) = self.current_solution {
                    let report = self.debugger.debug_solution(solution);
                    if let Some(constraint_analysis) = report.constraint_analysis {
                        format!(
                            "Constraint satisfaction rate: {:.2}%",
                            constraint_analysis.satisfaction_rate * 100.0
                        )
                    } else {
                        "No constraint analysis available".to_string()
                    }
                } else {
                    "No solution loaded".to_string()
                }
            }
            "watch" => {
                format!("Watched variables: {:?}", self.watch_variables)
            }
            _ => format!("Unknown command: {command}"),
        }
    }
}
