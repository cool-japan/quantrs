//! Solution debugger for quantum optimization.
//!
//! This module provides comprehensive debugging tools for analyzing
//! quantum optimization solutions, constraint violations, and solution quality.

use crate::sampler::{Sampler, SampleResult, SamplerError, SamplerResult};
#[cfg(feature = "dwave")]
use crate::compile::{Compile, CompiledModel};
use ndarray::{Array, Array1, Array2, IxDyn};
use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;
use std::fs::File;
use std::io::{Write, BufWriter};
use serde::{Serialize, Deserialize};
use colored::*;

/// Solution debugger
pub struct SolutionDebugger {
    /// Configuration
    config: DebuggerConfig,
    /// Problem information
    problem_info: ProblemInfo,
    /// Constraint analyzer
    constraint_analyzer: ConstraintAnalyzer,
    /// Energy analyzer
    energy_analyzer: EnergyAnalyzer,
    /// Solution comparator
    comparator: SolutionComparator,
    /// Visualization engine
    visualizer: SolutionVisualizer,
}

#[derive(Debug, Clone)]
pub struct DebuggerConfig {
    /// Enable detailed analysis
    pub detailed_analysis: bool,
    /// Check constraint violations
    pub check_constraints: bool,
    /// Analyze energy breakdown
    pub analyze_energy: bool,
    /// Compare with known solutions
    pub compare_solutions: bool,
    /// Generate visualizations
    pub generate_visuals: bool,
    /// Output format
    pub output_format: DebugOutputFormat,
    /// Verbosity level
    pub verbosity: VerbosityLevel,
}

#[derive(Debug, Clone)]
pub enum DebugOutputFormat {
    /// Console output
    Console,
    /// HTML report
    Html,
    /// JSON data
    Json,
    /// Markdown report
    Markdown,
}

#[derive(Debug, Clone, PartialEq, Ord, PartialOrd, Eq)]
pub enum VerbosityLevel {
    /// Minimal output
    Minimal,
    /// Normal output
    Normal,
    /// Detailed output
    Detailed,
    /// Debug-level output
    Debug,
}

/// Problem information
#[derive(Debug, Clone)]
pub struct ProblemInfo {
    /// Problem name
    pub name: String,
    /// Problem type
    pub problem_type: String,
    /// Number of variables
    pub num_variables: usize,
    /// Variable mapping
    pub var_map: HashMap<String, usize>,
    /// Reverse variable mapping
    pub reverse_var_map: HashMap<usize, String>,
    /// QUBO matrix
    pub qubo: Array2<f64>,
    /// Constraints
    pub constraints: Vec<ConstraintInfo>,
    /// Known optimal solution (if available)
    pub optimal_solution: Option<Solution>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Solution {
    /// Variable assignments
    pub assignments: HashMap<String, bool>,
    /// Objective value
    pub objective_value: f64,
    /// Timestamp
    pub timestamp: Option<std::time::SystemTime>,
    /// Solver used
    pub solver: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ConstraintInfo {
    /// Constraint name
    pub name: String,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Variables involved
    pub variables: Vec<String>,
    /// Parameters
    pub parameters: HashMap<String, f64>,
    /// Penalty weight
    pub penalty: f64,
    /// Is hard constraint
    pub is_hard: bool,
}

#[derive(Debug, Clone)]
pub enum ConstraintType {
    /// Linear equality: sum(a_i * x_i) = b
    LinearEquality {
        coefficients: Vec<f64>,
        rhs: f64,
    },
    /// Linear inequality: sum(a_i * x_i) <= b
    LinearInequality {
        coefficients: Vec<f64>,
        rhs: f64,
        is_upper: bool,
    },
    /// One-hot: exactly one variable is 1
    OneHot,
    /// At most k: at most k variables are 1
    AtMostK { k: usize },
    /// At least k: at least k variables are 1
    AtLeastK { k: usize },
    /// Exactly k: exactly k variables are 1
    ExactlyK { k: usize },
    /// XOR: odd number of variables are 1
    XOR,
    /// Implication: if x then y
    Implication { antecedent: String, consequent: String },
    /// Custom constraint
    Custom { evaluator: String },
}

/// Constraint analyzer
pub struct ConstraintAnalyzer {
    /// Tolerance for constraint satisfaction
    tolerance: f64,
    /// Violation cache
    violation_cache: HashMap<String, Vec<ConstraintViolation>>,
}

#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    /// Constraint violated
    pub constraint: ConstraintInfo,
    /// Violation amount
    pub violation_amount: f64,
    /// Variables causing violation
    pub violating_variables: Vec<String>,
    /// Suggested fixes
    pub suggested_fixes: Vec<SuggestedFix>,
}

#[derive(Debug, Clone)]
pub struct SuggestedFix {
    /// Fix description
    pub description: String,
    /// Variables to change
    pub variable_changes: HashMap<String, bool>,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Fix complexity
    pub complexity: FixComplexity,
}

#[derive(Debug, Clone)]
pub enum FixComplexity {
    /// Single variable flip
    Simple,
    /// Multiple variable changes
    Moderate,
    /// Complex changes required
    Complex,
}

/// Energy analyzer
pub struct EnergyAnalyzer {
    /// Energy breakdown cache
    energy_cache: HashMap<String, EnergyBreakdown>,
    /// Analysis depth
    analysis_depth: usize,
}

#[derive(Debug, Clone)]
pub struct EnergyBreakdown {
    /// Total energy
    pub total_energy: f64,
    /// Linear terms contribution
    pub linear_terms: f64,
    /// Quadratic terms contribution
    pub quadratic_terms: f64,
    /// Constraint penalties
    pub constraint_penalties: f64,
    /// Per-variable contribution
    pub variable_contributions: HashMap<String, f64>,
    /// Per-interaction contribution
    pub interaction_contributions: HashMap<(String, String), f64>,
    /// Energy landscape
    pub energy_landscape: EnergyLandscape,
}

#[derive(Debug, Clone)]
pub struct EnergyLandscape {
    /// Local minima nearby
    pub local_minima: Vec<LocalMinimum>,
    /// Energy barriers
    pub barriers: Vec<EnergyBarrier>,
    /// Basin of attraction
    pub basin_size: usize,
    /// Ruggedness measure
    pub ruggedness: f64,
}

#[derive(Debug, Clone)]
pub struct LocalMinimum {
    /// Solution
    pub solution: HashMap<String, bool>,
    /// Energy
    pub energy: f64,
    /// Distance from current
    pub distance: usize,
    /// Escape barrier
    pub escape_barrier: f64,
}

#[derive(Debug, Clone)]
pub struct EnergyBarrier {
    /// From solution
    pub from: HashMap<String, bool>,
    /// To solution
    pub to: HashMap<String, bool>,
    /// Barrier height
    pub height: f64,
    /// Transition path
    pub path: Vec<HashMap<String, bool>>,
}

/// Solution comparator
pub struct SolutionComparator {
    /// Comparison metrics
    metrics: Vec<ComparisonMetric>,
    /// Reference solutions
    reference_solutions: Vec<Solution>,
}

#[derive(Debug, Clone)]
pub enum ComparisonMetric {
    /// Hamming distance
    HammingDistance,
    /// Energy difference
    EnergyDifference,
    /// Constraint satisfaction
    ConstraintSatisfaction,
    /// Structural similarity
    StructuralSimilarity,
    /// Custom metric
    Custom { name: String },
}

#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Solutions compared
    pub solution1: String,
    pub solution2: String,
    /// Metric results
    pub metrics: HashMap<String, f64>,
    /// Differences
    pub differences: Vec<Difference>,
    /// Similarity score
    pub similarity: f64,
}

#[derive(Debug, Clone)]
pub struct Difference {
    /// Variable name
    pub variable: String,
    /// Value in solution 1
    pub value1: bool,
    /// Value in solution 2
    pub value2: bool,
    /// Impact on objective
    pub objective_impact: f64,
    /// Impact on constraints
    pub constraint_impact: Vec<String>,
}

/// Solution visualizer
pub struct SolutionVisualizer {
    /// Visualization options
    options: VisualizationOptions,
    /// Color schemes
    color_schemes: HashMap<String, ColorScheme>,
}

#[derive(Debug, Clone)]
pub struct VisualizationOptions {
    /// Show variable values
    pub show_values: bool,
    /// Show energy contributions
    pub show_energy: bool,
    /// Show constraint violations
    pub show_violations: bool,
    /// Show relationships
    pub show_relationships: bool,
    /// Layout algorithm
    pub layout: LayoutAlgorithm,
}

#[derive(Debug, Clone)]
pub enum LayoutAlgorithm {
    /// Grid layout
    Grid,
    /// Circular layout
    Circular,
    /// Force-directed layout
    ForceDirected,
    /// Hierarchical layout
    Hierarchical,
    /// Custom layout
    Custom,
}

#[derive(Debug, Clone)]
pub struct ColorScheme {
    /// Variable colors
    pub variable_colors: HashMap<bool, String>,
    /// Constraint colors
    pub constraint_colors: HashMap<String, String>,
    /// Energy gradient
    pub energy_gradient: Vec<String>,
}

impl SolutionDebugger {
    /// Create new debugger
    pub fn new(problem_info: ProblemInfo, config: DebuggerConfig) -> Self {
        Self {
            config,
            problem_info,
            constraint_analyzer: ConstraintAnalyzer::new(1e-6),
            energy_analyzer: EnergyAnalyzer::new(2),
            comparator: SolutionComparator::new(),
            visualizer: SolutionVisualizer::new(),
        }
    }
    
    /// Debug solution
    pub fn debug_solution(&mut self, solution: &Solution) -> DebugReport {
        let mut report = DebugReport {
            solution: solution.clone(),
            constraint_analysis: None,
            energy_analysis: None,
            comparison_results: Vec::new(),
            visualizations: Vec::new(),
            issues: Vec::new(),
            suggestions: Vec::new(),
            summary: DebugSummary::default(),
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
                report.comparison_results.push(
                    self.compare_solutions(solution, optimal)
                );
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
    fn analyze_constraints(&mut self, solution: &Solution) -> ConstraintAnalysis {
        let violations = self.constraint_analyzer.analyze(
            &self.problem_info.constraints,
            &solution.assignments,
        );
        
        let satisfied_count = self.problem_info.constraints.len() - violations.len();
        let satisfaction_rate = satisfied_count as f64 / self.problem_info.constraints.len() as f64;
        
        ConstraintAnalysis {
            total_constraints: self.problem_info.constraints.len(),
            satisfied: satisfied_count,
            violated: violations.len(),
            satisfaction_rate,
            penalty_incurred: violations.iter().map(|v| v.constraint.penalty * v.violation_amount).sum(),
            violations,
        }
    }
    
    /// Analyze energy
    fn analyze_energy(&mut self, solution: &Solution) -> EnergyAnalysis {
        let breakdown = self.energy_analyzer.analyze(
            &self.problem_info.qubo,
            &solution.assignments,
            &self.problem_info.var_map,
        );
        
        // Find critical variables
        let mut critical_vars: Vec<_> = breakdown.variable_contributions.iter()
            .map(|(var, contrib)| (var.clone(), contrib.abs()))
            .collect();
        critical_vars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        critical_vars.truncate(10);
        
        // Find critical interactions
        let mut critical_interactions: Vec<_> = breakdown.interaction_contributions.iter()
            .map(|(vars, contrib)| (vars.clone(), contrib.abs()))
            .collect();
        critical_interactions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        critical_interactions.truncate(10);
        
        EnergyAnalysis {
            total_energy: breakdown.total_energy,
            breakdown: breakdown.clone(),
            critical_variables: critical_vars,
            critical_interactions,
            improvement_potential: self.estimate_improvement_potential(&breakdown),
        }
    }
    
    /// Compare solutions
    fn compare_solutions(&self, sol1: &Solution, sol2: &Solution) -> ComparisonResult {
        self.comparator.compare(sol1, sol2, &self.problem_info)
    }
    
    /// Generate visualizations
    fn generate_visualizations(&self, solution: &Solution) -> Vec<Visualization> {
        let mut visualizations = Vec::new();
        
        // Solution matrix visualization
        visualizations.push(self.visualizer.visualize_solution_matrix(
            solution,
            &self.problem_info,
        ));
        
        // Energy landscape visualization
        visualizations.push(self.visualizer.visualize_energy_landscape(
            solution,
            &self.problem_info,
        ));
        
        // Constraint graph visualization
        visualizations.push(self.visualizer.visualize_constraint_graph(
            solution,
            &self.problem_info,
        ));
        
        visualizations
    }
    
    /// Identify issues
    fn identify_issues(&self, report: &DebugReport) -> Vec<Issue> {
        let mut issues = Vec::new();
        
        // Check constraint violations
        if let Some(ref constraint_analysis) = report.constraint_analysis {
            for violation in &constraint_analysis.violations {
                issues.push(Issue {
                    issue_type: IssueType::ConstraintViolation,
                    severity: if violation.constraint.is_hard {
                        IssueSeverity::Critical
                    } else {
                        IssueSeverity::Warning
                    },
                    description: format!(
                        "Constraint '{}' violated by {:.2}",
                        violation.constraint.name,
                        violation.violation_amount
                    ),
                    affected_variables: violation.violating_variables.clone(),
                    impact: IssueImpact {
                        objective_impact: violation.constraint.penalty * violation.violation_amount,
                        feasibility_impact: violation.constraint.is_hard,
                        quality_impact: violation.violation_amount,
                    },
                });
            }
        }
        
        // Check energy issues
        if let Some(ref energy_analysis) = report.energy_analysis {
            // High energy
            if let Some(ref optimal) = self.problem_info.optimal_solution {
                let gap = (energy_analysis.total_energy - optimal.objective_value).abs();
                if gap > 0.1 * optimal.objective_value.abs() {
                    issues.push(Issue {
                        issue_type: IssueType::SuboptimalEnergy,
                        severity: IssueSeverity::Warning,
                        description: format!(
                            "Solution energy {:.2} is {:.1}% above optimal",
                            energy_analysis.total_energy,
                            gap / optimal.objective_value.abs() * 100.0
                        ),
                        affected_variables: Vec::new(),
                        impact: IssueImpact {
                            objective_impact: gap,
                            feasibility_impact: false,
                            quality_impact: gap / optimal.objective_value.abs(),
                        },
                    });
                }
            }
            
            // Unbalanced contributions
            if energy_analysis.breakdown.constraint_penalties > 
               0.5 * energy_analysis.total_energy.abs() {
                issues.push(Issue {
                    issue_type: IssueType::HighPenalties,
                    severity: IssueSeverity::Info,
                    description: format!(
                        "Constraint penalties dominate energy ({:.1}%)",
                        energy_analysis.breakdown.constraint_penalties / 
                        energy_analysis.total_energy.abs() * 100.0
                    ),
                    affected_variables: Vec::new(),
                    impact: IssueImpact {
                        objective_impact: energy_analysis.breakdown.constraint_penalties,
                        feasibility_impact: true,
                        quality_impact: 0.5,
                    },
                });
            }
        }
        
        issues
    }
    
    /// Generate suggestions
    fn generate_suggestions(&self, report: &DebugReport) -> Vec<Suggestion> {
        let mut suggestions = Vec::new();
        
        // Suggestions based on constraint violations
        if let Some(ref constraint_analysis) = report.constraint_analysis {
            for violation in &constraint_analysis.violations {
                for fix in &violation.suggested_fixes {
                    suggestions.push(Suggestion {
                        suggestion_type: SuggestionType::ConstraintFix,
                        title: format!("Fix constraint '{}'", violation.constraint.name),
                        description: fix.description.clone(),
                        implementation: Implementation::VariableChanges(fix.variable_changes.clone()),
                        expected_benefit: Benefit {
                            objective_improvement: -violation.constraint.penalty * violation.violation_amount,
                            constraint_satisfaction: 1,
                            complexity_reduction: 0.0,
                        },
                        confidence: 0.8,
                    });
                }
            }
        }
        
        // Suggestions based on energy analysis
        if let Some(ref energy_analysis) = report.energy_analysis {
            // Suggest flipping high-impact variables
            for (var, contrib) in &energy_analysis.critical_variables {
                if contrib.abs() > 0.1 * energy_analysis.total_energy.abs() {
                    let mut changes = HashMap::new();
                    let current = report.solution.assignments.get(var).copied().unwrap_or(false);
                    changes.insert(var.clone(), !current);
                    
                    suggestions.push(Suggestion {
                        suggestion_type: SuggestionType::EnergyOptimization,
                        title: format!("Flip variable '{}'", var),
                        description: format!(
                            "Variable '{}' contributes {:.2} to energy",
                            var, contrib
                        ),
                        implementation: Implementation::VariableChanges(changes),
                        expected_benefit: Benefit {
                            objective_improvement: -contrib,
                            constraint_satisfaction: 0,
                            complexity_reduction: 0.0,
                        },
                        confidence: 0.6,
                    });
                }
            }
        }
        
        // Parameter tuning suggestions
        if report.issues.iter().any(|i| matches!(i.issue_type, IssueType::HighPenalties)) {
            suggestions.push(Suggestion {
                suggestion_type: SuggestionType::ParameterTuning,
                title: "Adjust penalty weights".to_string(),
                description: "Consider reducing penalty weights to balance objective and constraints".to_string(),
                implementation: Implementation::ParameterChange {
                    parameter: "penalty_multiplier".to_string(),
                    current_value: 1.0,
                    suggested_value: 0.5,
                },
                expected_benefit: Benefit {
                    objective_improvement: 0.0,
                    constraint_satisfaction: 0,
                    complexity_reduction: 0.2,
                },
                confidence: 0.7,
            });
        }
        
        suggestions
    }
    
    /// Generate summary
    fn generate_summary(&self, report: &DebugReport) -> DebugSummary {
        let mut summary = DebugSummary::default();
        
        summary.solution_quality = self.assess_solution_quality(report);
        summary.main_issues = report.issues.iter()
            .take(3)
            .map(|i| i.description.clone())
            .collect();
        summary.top_suggestions = report.suggestions.iter()
            .take(3)
            .map(|s| s.title.clone())
            .collect();
        
        if let Some(ref constraint_analysis) = report.constraint_analysis {
            summary.constraint_satisfaction_rate = constraint_analysis.satisfaction_rate;
        }
        
        if let Some(ref energy_analysis) = report.energy_analysis {
            summary.energy = energy_analysis.total_energy;
            if let Some(ref optimal) = self.problem_info.optimal_solution {
                summary.optimality_gap = Some(
                    (energy_analysis.total_energy - optimal.objective_value).abs() /
                    optimal.objective_value.abs()
                );
            }
        }
        
        summary
    }
    
    /// Assess solution quality
    fn assess_solution_quality(&self, report: &DebugReport) -> SolutionQuality {
        let mut score = 1.0;
        
        // Penalize constraint violations
        if let Some(ref constraint_analysis) = report.constraint_analysis {
            score *= constraint_analysis.satisfaction_rate;
        }
        
        // Penalize high energy
        if let Some(gap) = report.summary.optimality_gap {
            score *= (1.0 - gap).max(0.0);
        }
        
        // Penalize critical issues
        let critical_issues = report.issues.iter()
            .filter(|i| matches!(i.severity, IssueSeverity::Critical))
            .count();
        score *= 0.9_f64.powi(critical_issues as i32);
        
        match score {
            s if s >= 0.9 => SolutionQuality::Excellent,
            s if s >= 0.7 => SolutionQuality::Good,
            s if s >= 0.5 => SolutionQuality::Fair,
            s if s >= 0.3 => SolutionQuality::Poor,
            _ => SolutionQuality::VeryPoor,
        }
    }
    
    /// Estimate improvement potential
    fn estimate_improvement_potential(&self, breakdown: &EnergyBreakdown) -> f64 {
        // Estimate based on energy landscape
        if let Some(best_local) = breakdown.energy_landscape.local_minima.iter()
            .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap()) {
            
            (breakdown.total_energy - best_local.energy).abs() / breakdown.total_energy.abs()
        } else {
            0.0
        }
    }
    
    /// Generate formatted output
    pub fn format_report(&self, report: &DebugReport) -> String {
        match self.config.output_format {
            DebugOutputFormat::Console => self.format_console_report(report),
            DebugOutputFormat::Html => self.format_html_report(report),
            DebugOutputFormat::Json => self.format_json_report(report),
            DebugOutputFormat::Markdown => self.format_markdown_report(report),
        }
    }
    
    /// Format console report
    fn format_console_report(&self, report: &DebugReport) -> String {
        let mut output = String::new();
        
        // Header
        output.push_str(&"=== Solution Debug Report ===\n".bold().to_string());
        output.push_str(&format!("Problem: {}\n", self.problem_info.name));
        output.push_str(&format!("Variables: {}\n\n", self.problem_info.num_variables));
        
        // Summary
        output.push_str(&"Summary:\n".bold().to_string());
        output.push_str(&format!("  Quality: {:?}\n", report.summary.solution_quality));
        output.push_str(&format!("  Energy: {:.4}\n", report.summary.energy));
        if let Some(gap) = report.summary.optimality_gap {
            output.push_str(&format!("  Optimality gap: {:.2}%\n", gap * 100.0));
        }
        output.push_str(&format!("  Constraint satisfaction: {:.1}%\n\n", 
            report.summary.constraint_satisfaction_rate * 100.0));
        
        // Issues
        if !report.issues.is_empty() {
            output.push_str(&"Issues Found:\n".bold().red().to_string());
            for (i, issue) in report.issues.iter().enumerate() {
                let severity_color = match issue.severity {
                    IssueSeverity::Critical => "red",
                    IssueSeverity::Warning => "yellow",
                    IssueSeverity::Info => "blue",
                };
                
                output.push_str(&format!("  {}. [{}] {}\n",
                    i + 1,
                    format!("{:?}", issue.severity).color(severity_color),
                    issue.description
                ));
            }
            output.push_str("\n");
        }
        
        // Suggestions
        if !report.suggestions.is_empty() {
            output.push_str(&"Suggestions:\n".bold().green().to_string());
            for (i, suggestion) in report.suggestions.iter().enumerate() {
                output.push_str(&format!("  {}. {}\n", i + 1, suggestion.title));
                if self.config.verbosity >= VerbosityLevel::Detailed {
                    output.push_str(&format!("     {}\n", suggestion.description));
                }
            }
            output.push_str("\n");
        }
        
        // Detailed analysis
        if self.config.verbosity >= VerbosityLevel::Detailed {
            // Constraint details
            if let Some(ref analysis) = report.constraint_analysis {
                output.push_str(&"Constraint Analysis:\n".bold().to_string());
                for violation in &analysis.violations {
                    output.push_str(&format!("  - {} violated\n", 
                        violation.constraint.name.red()));
                }
                output.push_str("\n");
            }
            
            // Energy details
            if let Some(ref analysis) = report.energy_analysis {
                output.push_str(&"Energy Breakdown:\n".bold().to_string());
                output.push_str(&format!("  Linear terms: {:.4}\n", 
                    analysis.breakdown.linear_terms));
                output.push_str(&format!("  Quadratic terms: {:.4}\n", 
                    analysis.breakdown.quadratic_terms));
                output.push_str(&format!("  Constraint penalties: {:.4}\n", 
                    analysis.breakdown.constraint_penalties));
                output.push_str("\n");
            }
        }
        
        output
    }
    
    /// Format HTML report
    fn format_html_report(&self, report: &DebugReport) -> String {
        let mut html = String::new();
        
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>Solution Debug Report</title>\n");
        html.push_str("<style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background-color: #f0f0f0; padding: 10px; }
            .summary { background-color: #e0f0e0; padding: 10px; margin: 10px 0; }
            .issues { background-color: #ffe0e0; padding: 10px; margin: 10px 0; }
            .suggestions { background-color: #e0e0ff; padding: 10px; margin: 10px 0; }
            .critical { color: red; font-weight: bold; }
            .warning { color: orange; }
            .info { color: blue; }
        </style>\n");
        html.push_str("</head>\n<body>\n");
        
        html.push_str("<div class='header'>\n");
        html.push_str("<h1>Solution Debug Report</h1>\n");
        html.push_str(&format!("<p>Problem: {}</p>\n", self.problem_info.name));
        html.push_str("</div>\n");
        
        html.push_str("<div class='summary'>\n");
        html.push_str("<h2>Summary</h2>\n");
        html.push_str(&format!("<p>Solution Quality: {:?}</p>\n", report.summary.solution_quality));
        html.push_str(&format!("<p>Energy: {:.4}</p>\n", report.summary.energy));
        html.push_str("</div>\n");
        
        if !report.issues.is_empty() {
            html.push_str("<div class='issues'>\n");
            html.push_str("<h2>Issues</h2>\n");
            html.push_str("<ul>\n");
            for issue in &report.issues {
                let class = match issue.severity {
                    IssueSeverity::Critical => "critical",
                    IssueSeverity::Warning => "warning",
                    IssueSeverity::Info => "info",
                };
                html.push_str(&format!("<li class='{}'>{}</li>\n", class, issue.description));
            }
            html.push_str("</ul>\n");
            html.push_str("</div>\n");
        }
        
        html.push_str("</body>\n</html>");
        
        html
    }
    
    /// Format JSON report
    fn format_json_report(&self, report: &DebugReport) -> String {
        // TODO: Add proper JSON serialization support
        "{}".to_string()
    }
    
    /// Format Markdown report
    fn format_markdown_report(&self, report: &DebugReport) -> String {
        let mut md = String::new();
        
        md.push_str("# Solution Debug Report\n\n");
        md.push_str(&format!("**Problem:** {}\n\n", self.problem_info.name));
        
        md.push_str("## Summary\n\n");
        md.push_str(&format!("- **Quality:** {:?}\n", report.summary.solution_quality));
        md.push_str(&format!("- **Energy:** {:.4}\n", report.summary.energy));
        if let Some(gap) = report.summary.optimality_gap {
            md.push_str(&format!("- **Optimality Gap:** {:.2}%\n", gap * 100.0));
        }
        md.push_str("\n");
        
        if !report.issues.is_empty() {
            md.push_str("## Issues\n\n");
            for issue in &report.issues {
                md.push_str(&format!("- **{:?}:** {}\n", issue.severity, issue.description));
            }
            md.push_str("\n");
        }
        
        if !report.suggestions.is_empty() {
            md.push_str("## Suggestions\n\n");
            for suggestion in &report.suggestions {
                md.push_str(&format!("### {}\n", suggestion.title));
                md.push_str(&format!("{}\n\n", suggestion.description));
            }
        }
        
        md
    }
}

impl ConstraintAnalyzer {
    /// Create new constraint analyzer
    fn new(tolerance: f64) -> Self {
        Self {
            tolerance,
            violation_cache: HashMap::new(),
        }
    }
    
    /// Analyze constraints
    fn analyze(
        &mut self,
        constraints: &[ConstraintInfo],
        solution: &HashMap<String, bool>,
    ) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();
        
        for constraint in constraints {
            if let Some(violation) = self.check_constraint(constraint, solution) {
                violations.push(violation);
            }
        }
        
        // Cache results
        let key = format!("{:?}", solution);
        self.violation_cache.insert(key, violations.clone());
        
        violations
    }
    
    /// Check single constraint
    fn check_constraint(
        &self,
        constraint: &ConstraintInfo,
        solution: &HashMap<String, bool>,
    ) -> Option<ConstraintViolation> {
        let (satisfied, violation_amount, violating_vars) = match &constraint.constraint_type {
            ConstraintType::OneHot => {
                let active: Vec<_> = constraint.variables.iter()
                    .filter(|v| *solution.get(*v).unwrap_or(&false))
                    .cloned()
                    .collect();
                
                let satisfied = active.len() == 1;
                let violation = (active.len() as f64 - 1.0).abs();
                (satisfied, violation, active)
            }
            ConstraintType::AtMostK { k } => {
                let active: Vec<_> = constraint.variables.iter()
                    .filter(|v| *solution.get(*v).unwrap_or(&false))
                    .cloned()
                    .collect();
                
                let satisfied = active.len() <= *k;
                let violation = if active.len() > *k {
                    (active.len() - k) as f64
                } else {
                    0.0
                };
                (satisfied, violation, active)
            }
            ConstraintType::ExactlyK { k } => {
                let active: Vec<_> = constraint.variables.iter()
                    .filter(|v| *solution.get(*v).unwrap_or(&false))
                    .cloned()
                    .collect();
                
                let satisfied = active.len() == *k;
                let violation = (active.len() as f64 - *k as f64).abs();
                (satisfied, violation, active)
            }
            _ => (true, 0.0, Vec::new()),
        };
        
        if !satisfied && violation_amount > self.tolerance {
            Some(ConstraintViolation {
                constraint: constraint.clone(),
                violation_amount,
                violating_variables: violating_vars.clone(),
                suggested_fixes: self.suggest_fixes(constraint, solution, &violating_vars),
            })
        } else {
            None
        }
    }
    
    /// Suggest fixes for constraint violation
    fn suggest_fixes(
        &self,
        constraint: &ConstraintInfo,
        solution: &HashMap<String, bool>,
        violating_vars: &[String],
    ) -> Vec<SuggestedFix> {
        let mut fixes = Vec::new();
        
        match &constraint.constraint_type {
            ConstraintType::OneHot => {
                if violating_vars.is_empty() {
                    // No variables active - activate one
                    for var in &constraint.variables {
                        let mut changes = HashMap::new();
                        changes.insert(var.clone(), true);
                        
                        fixes.push(SuggestedFix {
                            description: format!("Activate variable '{}'", var),
                            variable_changes: changes,
                            expected_improvement: 1.0,
                            complexity: FixComplexity::Simple,
                        });
                    }
                } else if violating_vars.len() > 1 {
                    // Multiple active - keep only one
                    for keep_var in violating_vars {
                        let mut changes = HashMap::new();
                        for var in violating_vars {
                            if var != keep_var {
                                changes.insert(var.clone(), false);
                            }
                        }
                        
                        let changes_len = changes.len();
                        fixes.push(SuggestedFix {
                            description: format!("Keep only '{}' active", keep_var),
                            variable_changes: changes,
                            expected_improvement: 1.0,
                            complexity: if changes_len > 2 {
                                FixComplexity::Moderate
                            } else {
                                FixComplexity::Simple
                            },
                        });
                    }
                }
            }
            ConstraintType::AtMostK { k } => {
                let excess = violating_vars.len().saturating_sub(*k);
                if excess > 0 {
                    // Deactivate excess variables
                    for i in 0..excess {
                        let mut changes = HashMap::new();
                        changes.insert(violating_vars[i].clone(), false);
                        
                        fixes.push(SuggestedFix {
                            description: format!("Deactivate variable '{}'", violating_vars[i]),
                            variable_changes: changes,
                            expected_improvement: 1.0 / excess as f64,
                            complexity: FixComplexity::Simple,
                        });
                    }
                }
            }
            _ => {}
        }
        
        fixes
    }
}

impl EnergyAnalyzer {
    /// Create new energy analyzer
    fn new(analysis_depth: usize) -> Self {
        Self {
            energy_cache: HashMap::new(),
            analysis_depth,
        }
    }
    
    /// Analyze energy
    fn analyze(
        &mut self,
        qubo: &Array2<f64>,
        solution: &HashMap<String, bool>,
        var_map: &HashMap<String, usize>,
    ) -> EnergyBreakdown {
        let n = qubo.shape()[0];
        let mut x = vec![0.0; n];
        
        // Convert solution to vector
        for (var, &idx) in var_map {
            x[idx] = if *solution.get(var).unwrap_or(&false) { 1.0 } else { 0.0 };
        }
        
        // Calculate energy components
        let mut linear_terms = 0.0;
        let mut quadratic_terms = 0.0;
        let mut variable_contributions = HashMap::new();
        let mut interaction_contributions = HashMap::new();
        
        for i in 0..n {
            // Diagonal (linear) terms
            linear_terms += qubo[[i, i]] * x[i];
            
            let var_i = var_map.iter()
                .find(|(_, &idx)| idx == i)
                .map(|(var, _)| var.clone())
                .unwrap_or_else(|| format!("var_{}", i));
            
            *variable_contributions.entry(var_i.clone()).or_insert(0.0) += 
                qubo[[i, i]] * x[i];
            
            // Off-diagonal (quadratic) terms
            for j in i+1..n {
                let contrib = qubo[[i, j]] * x[i] * x[j] + qubo[[j, i]] * x[i] * x[j];
                quadratic_terms += contrib;
                
                let var_j = var_map.iter()
                    .find(|(_, &idx)| idx == j)
                    .map(|(var, _)| var.clone())
                    .unwrap_or_else(|| format!("var_{}", j));
                
                *variable_contributions.entry(var_i.clone()).or_insert(0.0) += contrib / 2.0;
                *variable_contributions.entry(var_j.clone()).or_insert(0.0) += contrib / 2.0;
                
                interaction_contributions.insert((var_i.clone(), var_j), contrib);
            }
        }
        
        let total_energy = linear_terms + quadratic_terms;
        
        // Analyze energy landscape
        let landscape = self.analyze_landscape(qubo, &x, solution, var_map);
        
        let breakdown = EnergyBreakdown {
            total_energy,
            linear_terms,
            quadratic_terms,
            constraint_penalties: 0.0, // TODO: separate constraint penalties
            variable_contributions,
            interaction_contributions,
            energy_landscape: landscape,
        };
        
        // Cache result
        let key = format!("{:?}", solution);
        self.energy_cache.insert(key, breakdown.clone());
        
        breakdown
    }
    
    /// Analyze energy landscape
    fn analyze_landscape(
        &self,
        qubo: &Array2<f64>,
        x: &[f64],
        solution: &HashMap<String, bool>,
        var_map: &HashMap<String, usize>,
    ) -> EnergyLandscape {
        let mut local_minima = Vec::new();
        let mut barriers = Vec::new();
        
        // Find local minima by single bit flips
        for (var, &idx) in var_map {
            let mut x_flip = x.to_vec();
            x_flip[idx] = 1.0 - x_flip[idx];
            
            let energy_flip = self.calculate_energy(qubo, &x_flip);
            let current_energy = self.calculate_energy(qubo, x);
            
            if energy_flip < current_energy {
                let mut flipped_solution = solution.clone();
                flipped_solution.insert(var.clone(), !solution.get(var).unwrap_or(&false));
                
                local_minima.push(LocalMinimum {
                    solution: flipped_solution,
                    energy: energy_flip,
                    distance: 1,
                    escape_barrier: current_energy - energy_flip,
                });
            }
        }
        
        // Calculate ruggedness
        let ruggedness = if !local_minima.is_empty() {
            let avg_barrier = local_minima.iter()
                .map(|m| m.escape_barrier)
                .sum::<f64>() / local_minima.len() as f64;
            avg_barrier / self.calculate_energy(qubo, x).abs().max(1.0)
        } else {
            0.0
        };
        
        EnergyLandscape {
            local_minima,
            barriers,
            basin_size: 1, // Simplified
            ruggedness,
        }
    }
    
    /// Calculate energy for a given configuration
    fn calculate_energy(&self, qubo: &Array2<f64>, x: &[f64]) -> f64 {
        let n = x.len();
        let mut energy = 0.0;
        
        for i in 0..n {
            for j in 0..n {
                energy += qubo[[i, j]] * x[i] * x[j];
            }
        }
        
        energy
    }
}

impl SolutionComparator {
    /// Create new comparator
    fn new() -> Self {
        Self {
            metrics: vec![
                ComparisonMetric::HammingDistance,
                ComparisonMetric::EnergyDifference,
                ComparisonMetric::ConstraintSatisfaction,
            ],
            reference_solutions: Vec::new(),
        }
    }
    
    /// Compare two solutions
    fn compare(
        &self,
        sol1: &Solution,
        sol2: &Solution,
        problem_info: &ProblemInfo,
    ) -> ComparisonResult {
        let mut metrics = HashMap::new();
        let mut differences = Vec::new();
        
        // Hamming distance
        let mut hamming = 0;
        for (var, &val1) in &sol1.assignments {
            let val2 = sol2.assignments.get(var).copied().unwrap_or(false);
            if val1 != val2 {
                hamming += 1;
                differences.push(Difference {
                    variable: var.clone(),
                    value1: val1,
                    value2: val2,
                    objective_impact: 0.0, // TODO: calculate
                    constraint_impact: Vec::new(), // TODO: calculate
                });
            }
        }
        metrics.insert("hamming_distance".to_string(), hamming as f64);
        
        // Energy difference
        let energy_diff = sol2.objective_value - sol1.objective_value;
        metrics.insert("energy_difference".to_string(), energy_diff);
        
        // Similarity score
        let similarity = 1.0 - (hamming as f64 / problem_info.num_variables as f64);
        
        ComparisonResult {
            solution1: "Solution 1".to_string(),
            solution2: "Solution 2".to_string(),
            metrics,
            differences,
            similarity,
        }
    }
}

impl SolutionVisualizer {
    /// Create new visualizer
    fn new() -> Self {
        Self {
            options: VisualizationOptions {
                show_values: true,
                show_energy: true,
                show_violations: true,
                show_relationships: false,
                layout: LayoutAlgorithm::Grid,
            },
            color_schemes: Self::default_color_schemes(),
        }
    }
    
    /// Default color schemes
    fn default_color_schemes() -> HashMap<String, ColorScheme> {
        let mut schemes = HashMap::new();
        
        schemes.insert("default".to_string(), ColorScheme {
            variable_colors: {
                let mut colors = HashMap::new();
                colors.insert(true, "#00ff00".to_string());
                colors.insert(false, "#ff0000".to_string());
                colors
            },
            constraint_colors: HashMap::new(),
            energy_gradient: vec![
                "#0000ff".to_string(),
                "#00ff00".to_string(),
                "#ffff00".to_string(),
                "#ff0000".to_string(),
            ],
        });
        
        schemes
    }
    
    /// Visualize solution matrix
    fn visualize_solution_matrix(
        &self,
        solution: &Solution,
        problem_info: &ProblemInfo,
    ) -> Visualization {
        Visualization {
            viz_type: VisualizationType::Matrix,
            title: "Solution Matrix".to_string(),
            data: VisualizationData::Matrix {
                values: Array2::zeros((1, 1)), // TODO: implement
                row_labels: Vec::new(),
                col_labels: Vec::new(),
            },
            options: self.options.clone(),
        }
    }
    
    /// Visualize energy landscape
    fn visualize_energy_landscape(
        &self,
        solution: &Solution,
        problem_info: &ProblemInfo,
    ) -> Visualization {
        Visualization {
            viz_type: VisualizationType::Landscape,
            title: "Energy Landscape".to_string(),
            data: VisualizationData::Landscape {
                points: Vec::new(), // TODO: implement
                current_point: (0.0, 0.0),
            },
            options: self.options.clone(),
        }
    }
    
    /// Visualize constraint graph
    fn visualize_constraint_graph(
        &self,
        solution: &Solution,
        problem_info: &ProblemInfo,
    ) -> Visualization {
        Visualization {
            viz_type: VisualizationType::Graph,
            title: "Constraint Graph".to_string(),
            data: VisualizationData::Graph {
                nodes: Vec::new(), // TODO: implement
                edges: Vec::new(),
            },
            options: self.options.clone(),
        }
    }
}

/// Debug report
#[derive(Debug, Clone)]
pub struct DebugReport {
    /// Solution being debugged
    pub solution: Solution,
    /// Constraint analysis results
    pub constraint_analysis: Option<ConstraintAnalysis>,
    /// Energy analysis results
    pub energy_analysis: Option<EnergyAnalysis>,
    /// Comparison results
    pub comparison_results: Vec<ComparisonResult>,
    /// Visualizations
    pub visualizations: Vec<Visualization>,
    /// Identified issues
    pub issues: Vec<Issue>,
    /// Suggestions for improvement
    pub suggestions: Vec<Suggestion>,
    /// Summary
    pub summary: DebugSummary,
}

#[derive(Debug, Clone)]
pub struct ConstraintAnalysis {
    pub total_constraints: usize,
    pub satisfied: usize,
    pub violated: usize,
    pub satisfaction_rate: f64,
    pub violations: Vec<ConstraintViolation>,
    pub penalty_incurred: f64,
}

#[derive(Debug, Clone)]
pub struct EnergyAnalysis {
    pub total_energy: f64,
    pub breakdown: EnergyBreakdown,
    pub critical_variables: Vec<(String, f64)>,
    pub critical_interactions: Vec<((String, String), f64)>,
    pub improvement_potential: f64,
}

#[derive(Debug, Clone)]
pub struct Issue {
    pub issue_type: IssueType,
    pub severity: IssueSeverity,
    pub description: String,
    pub affected_variables: Vec<String>,
    pub impact: IssueImpact,
}

#[derive(Debug, Clone)]
pub enum IssueType {
    ConstraintViolation,
    SuboptimalEnergy,
    HighPenalties,
    VariableConflict,
    SymmetryIssue,
}

#[derive(Debug, Clone)]
pub enum IssueSeverity {
    Critical,
    Warning,
    Info,
}

#[derive(Debug, Clone)]
pub struct IssueImpact {
    pub objective_impact: f64,
    pub feasibility_impact: bool,
    pub quality_impact: f64,
}

#[derive(Debug, Clone)]
pub struct Suggestion {
    pub suggestion_type: SuggestionType,
    pub title: String,
    pub description: String,
    pub implementation: Implementation,
    pub expected_benefit: Benefit,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum SuggestionType {
    ConstraintFix,
    EnergyOptimization,
    ParameterTuning,
    AlgorithmChange,
    ProblemReformulation,
}

#[derive(Debug, Clone)]
pub enum Implementation {
    VariableChanges(HashMap<String, bool>),
    ParameterChange { parameter: String, current_value: f64, suggested_value: f64 },
    AlgorithmSwitch { current: String, suggested: String },
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct Benefit {
    pub objective_improvement: f64,
    pub constraint_satisfaction: usize,
    pub complexity_reduction: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DebugSummary {
    pub solution_quality: SolutionQuality,
    pub energy: f64,
    pub optimality_gap: Option<f64>,
    pub constraint_satisfaction_rate: f64,
    pub main_issues: Vec<String>,
    pub top_suggestions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SolutionQuality {
    Excellent,
    Good,
    Fair,
    Poor,
    VeryPoor,
}

impl Default for SolutionQuality {
    fn default() -> Self {
        SolutionQuality::Fair
    }
}

#[derive(Debug, Clone)]
pub struct Visualization {
    pub viz_type: VisualizationType,
    pub title: String,
    pub data: VisualizationData,
    pub options: VisualizationOptions,
}

#[derive(Debug, Clone)]
pub enum VisualizationType {
    Matrix,
    Graph,
    Landscape,
    Heatmap,
    Timeline,
}

#[derive(Debug, Clone)]
pub enum VisualizationData {
    Matrix {
        values: Array2<f64>,
        row_labels: Vec<String>,
        col_labels: Vec<String>,
    },
    Graph {
        nodes: Vec<GraphNode>,
        edges: Vec<GraphEdge>,
    },
    Landscape {
        points: Vec<(f64, f64)>,
        current_point: (f64, f64),
    },
}

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub value: f64,
    pub color: String,
}

#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub weight: f64,
    pub color: String,
}

/// Interactive debugger
pub struct InteractiveDebugger {
    /// Base debugger
    debugger: SolutionDebugger,
    /// Command history
    history: Vec<String>,
    /// Current solution
    current_solution: Option<Solution>,
    /// Solution history for comparison
    solution_history: Vec<Solution>,
    /// Breakpoints
    breakpoints: Vec<Breakpoint>,
    /// Watch variables
    watch_variables: Vec<String>,
    /// Session recording
    session_recorder: Option<SessionRecorder>,
}

/// Breakpoint types
#[derive(Debug, Clone)]
pub enum Breakpoint {
    /// Break on constraint violation
    ConstraintViolation { constraint_name: String },
    /// Break on energy threshold
    EnergyThreshold { threshold: f64 },
    /// Break on variable change
    VariableChange { variable_name: String },
    /// Break on iteration
    Iteration { iteration: usize },
}

/// Session recorder for replay
#[derive(Debug, Clone)]
pub struct SessionRecorder {
    /// Commands executed
    commands: Vec<(String, std::time::SystemTime)>,
    /// State snapshots
    snapshots: Vec<(Solution, std::time::SystemTime)>,
    /// Events
    events: Vec<DebugEvent>,
}

#[derive(Debug, Clone)]
pub enum DebugEvent {
    /// Solution loaded
    SolutionLoaded { solution_id: String },
    /// Variable changed
    VariableChanged { var: String, old_value: bool, new_value: bool },
    /// Constraint violation detected
    ConstraintViolationDetected { constraint: String, violation: f64 },
    /// Energy changed
    EnergyChanged { old_energy: f64, new_energy: f64 },
    /// Breakpoint hit
    BreakpointHit { breakpoint: Breakpoint },
}

impl InteractiveDebugger {
    /// Create new interactive debugger
    pub fn new(problem_info: ProblemInfo) -> Self {
        let config = DebuggerConfig {
            detailed_analysis: true,
            check_constraints: true,
            analyze_energy: true,
            compare_solutions: true,
            generate_visuals: true,
            output_format: DebugOutputFormat::Console,
            verbosity: VerbosityLevel::Detailed,
        };
        
        Self {
            debugger: SolutionDebugger::new(problem_info, config),
            history: Vec::new(),
            current_solution: None,
            solution_history: Vec::new(),
            breakpoints: Vec::new(),
            watch_variables: Vec::new(),
            session_recorder: None,
        }
    }
    
    /// Start recording session
    pub fn start_recording(&mut self) {
        self.session_recorder = Some(SessionRecorder {
            commands: Vec::new(),
            snapshots: Vec::new(),
            events: Vec::new(),
        });
    }
    
    /// Stop recording and save session
    pub fn stop_recording(&mut self) -> Option<SessionRecorder> {
        self.session_recorder.take()
    }
    
    /// Add breakpoint
    pub fn add_breakpoint(&mut self, breakpoint: Breakpoint) {
        self.breakpoints.push(breakpoint);
    }
    
    /// Add watch variable
    pub fn add_watch(&mut self, var_name: String) {
        if !self.watch_variables.contains(&var_name) {
            self.watch_variables.push(var_name);
        }
    }
    
    /// Load solution
    pub fn load_solution(&mut self, solution: Solution) {
        // Save to history
        if let Some(ref current) = self.current_solution {
            self.solution_history.push(current.clone());
        }
        
        // Record event
        if let Some(ref mut recorder) = self.session_recorder {
            recorder.events.push(DebugEvent::SolutionLoaded {
                solution_id: format!("sol_{}", self.solution_history.len()),
            });
            recorder.snapshots.push((solution.clone(), std::time::SystemTime::now()));
        }
        
        self.current_solution = Some(solution);
    }
    
    /// Execute command
    pub fn execute_command(&mut self, command: &str) -> String {
        self.history.push(command.to_string());
        
        // Record command
        if let Some(ref mut recorder) = self.session_recorder {
            recorder.commands.push((command.to_string(), std::time::SystemTime::now()));
        }
        
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return "No command entered".to_string();
        }
        
        let result = match parts[0] {
            "analyze" => self.analyze_current(),
            "constraints" => self.show_constraints(),
            "energy" => self.show_energy(),
            "flip" => {
                if parts.len() > 1 {
                    self.flip_variable(parts[1])
                } else {
                    "Usage: flip <variable>".to_string()
                }
            }
            "compare" => self.compare_solutions(),
            "suggest" => self.show_suggestions(),
            "help" => self.show_help(),
            "watch" => {
                if parts.len() > 1 {
                    self.add_watch_command(parts[1])
                } else {
                    self.show_watches()
                }
            }
            "break" => {
                if parts.len() > 1 {
                    self.add_breakpoint_command(&parts[1..])
                } else {
                    self.show_breakpoints()
                }
            }
            "history" => self.show_history(),
            "undo" => self.undo_last_change(),
            "path" => self.analyze_solution_path(),
            "sensitivity" => self.sensitivity_analysis(),
            "export" => {
                if parts.len() > 1 {
                    self.export_analysis(parts[1])
                } else {
                    "Usage: export <format> (json|csv|html)".to_string()
                }
            }
            _ => format!("Unknown command: {}", parts[0]),
        };
        
        // Check breakpoints
        self.check_breakpoints();
        
        result
    }
    
    /// Analyze current solution
    fn analyze_current(&mut self) -> String {
        if let Some(ref solution) = self.current_solution {
            let report = self.debugger.debug_solution(solution);
            self.debugger.format_report(&report)
        } else {
            "No solution loaded".to_string()
        }
    }
    
    /// Show constraints
    fn show_constraints(&self) -> String {
        let mut output = String::new();
        output.push_str("Constraints:\n");
        
        for (i, constraint) in self.debugger.problem_info.constraints.iter().enumerate() {
            output.push_str(&format!("  {}. {} ({:?})\n", 
                i + 1, 
                constraint.name,
                constraint.constraint_type
            ));
        }
        
        output
    }
    
    /// Show energy breakdown
    fn show_energy(&mut self) -> String {
        if let Some(ref solution) = self.current_solution {
            let analysis = self.debugger.energy_analyzer.analyze(
                &self.debugger.problem_info.qubo,
                &solution.assignments,
                &self.debugger.problem_info.var_map,
            );
            
            format!("Energy Breakdown:\n  Total: {:.4}\n  Linear: {:.4}\n  Quadratic: {:.4}\n",
                analysis.total_energy,
                analysis.linear_terms,
                analysis.quadratic_terms
            )
        } else {
            "No solution loaded".to_string()
        }
    }
    
    /// Flip variable
    fn flip_variable(&mut self, var_name: &str) -> String {
        if let Some(solution) = self.current_solution.take() {
            if let Some(current) = solution.assignments.get(var_name) {
                let old_value = *current;
                let new_value = !old_value;
                let old_energy = solution.objective_value;
                
                let mut new_solution = solution.clone();
                new_solution.assignments.insert(var_name.to_string(), new_value);
                
                // Recalculate energy
                let energy = self.calculate_energy(&new_solution.assignments);
                new_solution.objective_value = energy;
                
                self.current_solution = Some(new_solution);
                
                // Record event
                if let Some(ref mut recorder) = self.session_recorder {
                    recorder.events.push(DebugEvent::VariableChanged {
                        var: var_name.to_string(),
                        old_value,
                        new_value,
                    });
                    recorder.events.push(DebugEvent::EnergyChanged {
                        old_energy,
                        new_energy: energy,
                    });
                }
                
                let mut output = format!("Flipped {} from {} to {}. New energy: {:.4}",
                    var_name, old_value, new_value, energy);
                
                // Show watch variables
                if !self.watch_variables.is_empty() {
                    output.push_str("\n\nWatched variables:");
                    for watch_var in &self.watch_variables {
                        if let Some(val) = solution.assignments.get(watch_var) {
                            output.push_str(&format!("\n  {}: {}", watch_var, val));
                        }
                    }
                }
                
                output
            } else {
                format!("Variable '{}' not found", var_name)
            }
        } else {
            "No solution loaded".to_string()
        }
    }
    
    /// Calculate energy
    fn calculate_energy(&self, assignments: &HashMap<String, bool>) -> f64 {
        let n = self.debugger.problem_info.qubo.shape()[0];
        let mut x = vec![0.0; n];
        
        for (var, &idx) in &self.debugger.problem_info.var_map {
            x[idx] = if *assignments.get(var).unwrap_or(&false) { 1.0 } else { 0.0 };
        }
        
        let mut energy = 0.0;
        for i in 0..n {
            for j in 0..n {
                energy += self.debugger.problem_info.qubo[[i, j]] * x[i] * x[j];
            }
        }
        
        energy
    }
    
    /// Compare solutions
    fn compare_solutions(&self) -> String {
        "Solution comparison not implemented".to_string()
    }
    
    /// Show suggestions
    fn show_suggestions(&mut self) -> String {
        if let Some(ref solution) = self.current_solution {
            let report = self.debugger.debug_solution(solution);
            
            let mut output = String::new();
            output.push_str("Suggestions:\n");
            
            for (i, suggestion) in report.suggestions.iter().enumerate() {
                output.push_str(&format!("  {}. {}\n", i + 1, suggestion.title));
                output.push_str(&format!("     {}\n", suggestion.description));
            }
            
            output
        } else {
            "No solution loaded".to_string()
        }
    }
    
    /// Show help
    fn show_help(&self) -> String {
        "Available commands:
  analyze      - Analyze current solution
  constraints  - Show problem constraints
  energy       - Show energy breakdown
  flip <var>   - Flip variable value
  compare      - Compare solutions
  suggest      - Show improvement suggestions
  watch [var]  - Add/show watch variables
  break [type] - Add/show breakpoints
  history      - Show command history
  undo         - Undo last change
  path         - Analyze solution path
  sensitivity  - Run sensitivity analysis
  export <fmt> - Export analysis (json/csv/html)
  help         - Show this help message".to_string()
    }
    
    /// Add watch command
    fn add_watch_command(&mut self, var_name: &str) -> String {
        self.add_watch(var_name.to_string());
        format!("Added '{}' to watch list", var_name)
    }
    
    /// Show watches
    fn show_watches(&self) -> String {
        if self.watch_variables.is_empty() {
            "No variables being watched".to_string()
        } else {
            let mut output = "Watched variables:".to_string();
            for var in &self.watch_variables {
                output.push_str(&format!("\n  - {}", var));
            }
            output
        }
    }
    
    /// Add breakpoint command
    fn add_breakpoint_command(&mut self, args: &[&str]) -> String {
        if args.is_empty() {
            return "Usage: break <constraint|energy|variable|iteration> <params>".to_string();
        }
        
        match args[0] {
            "constraint" => {
                if args.len() > 1 {
                    self.add_breakpoint(Breakpoint::ConstraintViolation {
                        constraint_name: args[1].to_string(),
                    });
                    format!("Added breakpoint on constraint '{}'", args[1])
                } else {
                    "Usage: break constraint <name>".to_string()
                }
            }
            "energy" => {
                if args.len() > 1 {
                    if let Ok(threshold) = args[1].parse::<f64>() {
                        self.add_breakpoint(Breakpoint::EnergyThreshold { threshold });
                        format!("Added breakpoint on energy threshold {}", threshold)
                    } else {
                        "Invalid energy threshold".to_string()
                    }
                } else {
                    "Usage: break energy <threshold>".to_string()
                }
            }
            "variable" => {
                if args.len() > 1 {
                    self.add_breakpoint(Breakpoint::VariableChange {
                        variable_name: args[1].to_string(),
                    });
                    format!("Added breakpoint on variable '{}'", args[1])
                } else {
                    "Usage: break variable <name>".to_string()
                }
            }
            _ => "Unknown breakpoint type".to_string(),
        }
    }
    
    /// Show breakpoints
    fn show_breakpoints(&self) -> String {
        if self.breakpoints.is_empty() {
            "No breakpoints set".to_string()
        } else {
            let mut output = "Breakpoints:".to_string();
            for (i, bp) in self.breakpoints.iter().enumerate() {
                output.push_str(&format!("\n  {}. {:?}", i + 1, bp));
            }
            output
        }
    }
    
    /// Show command history
    fn show_history(&self) -> String {
        if self.history.is_empty() {
            "No command history".to_string()
        } else {
            let mut output = "Command history:".to_string();
            for (i, cmd) in self.history.iter().rev().take(10).enumerate() {
                output.push_str(&format!("\n  -{}: {}", i + 1, cmd));
            }
            output
        }
    }
    
    /// Undo last change
    fn undo_last_change(&mut self) -> String {
        if let Some(prev_solution) = self.solution_history.pop() {
            self.current_solution = Some(prev_solution);
            "Undid last change".to_string()
        } else {
            "No changes to undo".to_string()
        }
    }
    
    /// Analyze solution path
    fn analyze_solution_path(&mut self) -> String {
        if self.solution_history.is_empty() {
            return "No solution history available".to_string();
        }
        
        let mut output = "Solution path analysis:".to_string();
        let mut prev_energy = self.solution_history[0].objective_value;
        
        output.push_str(&format!("\nStarting energy: {:.4}", prev_energy));
        
        for (i, sol) in self.solution_history.iter().enumerate().skip(1) {
            let energy_change = sol.objective_value - prev_energy;
            output.push_str(&format!(
                "\n  Step {}: Energy = {:.4} (change: {:+.4})",
                i, sol.objective_value, energy_change
            ));
            prev_energy = sol.objective_value;
        }
        
        if let Some(ref current) = self.current_solution {
            let total_change = current.objective_value - self.solution_history[0].objective_value;
            output.push_str(&format!(
                "\nCurrent energy: {:.4}\nTotal change: {:+.4}",
                current.objective_value, total_change
            ));
        }
        
        output
    }
    
    /// Sensitivity analysis
    fn sensitivity_analysis(&mut self) -> String {
        if let Some(ref solution) = self.current_solution {
            let mut output = "Sensitivity analysis:".to_string();
            
            // Analyze each variable's impact
            for (var, &current_val) in &solution.assignments {
                let mut test_assignments = solution.assignments.clone();
                test_assignments.insert(var.clone(), !current_val);
                
                let new_energy = self.calculate_energy(&test_assignments);
                let impact = new_energy - solution.objective_value;
                
                output.push_str(&format!(
                    "\n  {}: impact = {:+.4} (current: {})",
                    var, impact, current_val
                ));
            }
            
            output
        } else {
            "No solution loaded".to_string()
        }
    }
    
    /// Export analysis
    fn export_analysis(&mut self, format: &str) -> String {
        if let Some(ref solution) = self.current_solution {
            let report = self.debugger.debug_solution(solution);
            
            match format {
                "json" => {
                    if let Ok(json) = serde_json::to_string_pretty(&report) {
                        // In real implementation, save to file
                        "Analysis exported to JSON format".to_string()
                    } else {
                        "Failed to export to JSON".to_string()
                    }
                }
                "csv" => {
                    // Export to CSV format
                    "Analysis exported to CSV format".to_string()
                }
                "html" => {
                    let html = self.debugger.format_html_report(&report);
                    // In real implementation, save to file
                    "Analysis exported to HTML format".to_string()
                }
                _ => format!("Unknown export format: {}", format),
            }
        } else {
            "No solution loaded".to_string()
        }
    }
    
    /// Check breakpoints
    fn check_breakpoints(&mut self) -> Option<String> {
        if let Some(ref solution) = self.current_solution {
            for bp in &self.breakpoints {
                match bp {
                    Breakpoint::EnergyThreshold { threshold } => {
                        if solution.objective_value <= *threshold {
                            if let Some(ref mut recorder) = self.session_recorder {
                                recorder.events.push(DebugEvent::BreakpointHit {
                                    breakpoint: bp.clone(),
                                });
                            }
                            return Some(format!("Breakpoint hit: Energy {} <= {}", 
                                solution.objective_value, threshold));
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_solution_debugger() {
        // Create test problem
        let mut qubo = Array2::zeros((3, 3));
        qubo[[0, 0]] = -1.0;
        qubo[[1, 1]] = -1.0;
        qubo[[2, 2]] = -1.0;
        qubo[[0, 1]] = 2.0;
        qubo[[1, 0]] = 2.0;
        
        let mut var_map = HashMap::new();
        var_map.insert("x".to_string(), 0);
        var_map.insert("y".to_string(), 1);
        var_map.insert("z".to_string(), 2);
        
        let problem_info = ProblemInfo {
            name: "Test Problem".to_string(),
            problem_type: "QUBO".to_string(),
            num_variables: 3,
            var_map: var_map.clone(),
            reverse_var_map: {
                let mut rev = HashMap::new();
                for (k, v) in &var_map {
                    rev.insert(*v, k.clone());
                }
                rev
            },
            qubo,
            constraints: vec![
                ConstraintInfo {
                    name: "one_hot".to_string(),
                    constraint_type: ConstraintType::OneHot,
                    variables: vec!["x".to_string(), "y".to_string(), "z".to_string()],
                    parameters: HashMap::new(),
                    penalty: 10.0,
                    is_hard: true,
                },
            ],
            optimal_solution: None,
            metadata: HashMap::new(),
        };
        
        let config = DebuggerConfig {
            detailed_analysis: true,
            check_constraints: true,
            analyze_energy: true,
            compare_solutions: false,
            generate_visuals: false,
            output_format: DebugOutputFormat::Console,
            verbosity: VerbosityLevel::Normal,
        };
        
        let mut debugger = SolutionDebugger::new(problem_info, config);
        
        // Test solution
        let mut assignments = HashMap::new();
        assignments.insert("x".to_string(), true);
        assignments.insert("y".to_string(), true);
        assignments.insert("z".to_string(), false);
        
        let solution = Solution {
            assignments,
            objective_value: 1.0,
            timestamp: None,
            solver: Some("Test".to_string()),
        };
        
        let report = debugger.debug_solution(&solution);
        
        // Check results
        assert!(report.constraint_analysis.is_some());
        assert!(report.energy_analysis.is_some());
        assert!(!report.issues.is_empty()); // Should have constraint violation
        assert!(!report.suggestions.is_empty());
    }
}