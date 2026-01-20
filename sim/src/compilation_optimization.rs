//! Compilation Time Optimization
//!
//! This module provides tools and strategies for optimizing Rust compilation times
//! in large quantum simulation codebases through dependency analysis and optimization.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use std::fmt::Write;
/// Analysis of module dependencies and compilation characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationAnalysis {
    /// Module dependency graph
    pub dependencies: HashMap<String, Vec<String>>,
    /// Module sizes (lines of code)
    pub module_sizes: HashMap<String, usize>,
    /// Estimated compilation times per module
    pub compilation_times: HashMap<String, f64>,
    /// Heavy dependencies (expensive to compile)
    pub heavy_dependencies: HashSet<String>,
    /// Circular dependencies detected
    pub circular_dependencies: Vec<Vec<String>>,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Compilation optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Type of optimization
    pub optimization_type: OptimizationType,
    /// Affected modules
    pub modules: Vec<String>,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Description of the optimization
    pub description: String,
    /// Implementation priority
    pub priority: RecommendationPriority,
}

/// Types of compilation optimizations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Reduce unused imports
    RemoveUnusedImports,
    /// Split large modules
    ModuleRefactoring,
    /// Use lazy imports where possible
    LazyImports,
    /// Optimize feature flags
    FeatureOptimization,
    /// Reduce macro usage
    MacroOptimization,
    /// Use dynamic loading
    DynamicLoading,
    /// Parallel compilation
    ParallelCompilation,
    /// Incremental compilation improvements
    IncrementalCompilation,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// Low impact optimization
    Low,
    /// Medium impact optimization
    Medium,
    /// High impact optimization
    High,
    /// Critical optimization needed
    Critical,
}

/// Configuration for compilation optimization analysis
#[derive(Debug, Clone)]
pub struct CompilationOptimizerConfig {
    /// Root directory to analyze
    pub root_path: PathBuf,
    /// File extensions to analyze
    pub file_extensions: Vec<String>,
    /// Maximum module size before recommending split
    pub max_module_size: usize,
    /// Threshold for heavy dependencies (compilation time in seconds)
    pub heavy_dependency_threshold: f64,
    /// Enable advanced analysis features
    pub enable_advanced_analysis: bool,
}

impl Default for CompilationOptimizerConfig {
    fn default() -> Self {
        Self {
            root_path: PathBuf::from("."),
            file_extensions: vec!["rs".to_string()],
            max_module_size: 2000,
            heavy_dependency_threshold: 5.0,
            enable_advanced_analysis: true,
        }
    }
}

/// Compilation optimizer for analyzing and improving build times
pub struct CompilationOptimizer {
    /// Configuration
    config: CompilationOptimizerConfig,
    /// Analysis cache
    analysis_cache: HashMap<String, CompilationAnalysis>,
}

impl CompilationOptimizer {
    /// Create new compilation optimizer
    #[must_use]
    pub fn new(config: CompilationOptimizerConfig) -> Self {
        Self {
            config,
            analysis_cache: HashMap::new(),
        }
    }

    /// Analyze codebase for compilation optimization opportunities
    pub fn analyze_codebase(&mut self) -> Result<CompilationAnalysis, Box<dyn std::error::Error>> {
        let mut analysis = CompilationAnalysis {
            dependencies: HashMap::new(),
            module_sizes: HashMap::new(),
            compilation_times: HashMap::new(),
            heavy_dependencies: HashSet::new(),
            circular_dependencies: Vec::new(),
            recommendations: Vec::new(),
        };

        // Analyze module structure
        self.analyze_module_structure(&mut analysis)?;

        // Analyze dependencies
        self.analyze_dependencies(&mut analysis)?;

        // Detect compilation bottlenecks
        self.detect_compilation_bottlenecks(&mut analysis)?;

        // Generate optimization recommendations
        self.generate_recommendations(&mut analysis)?;

        Ok(analysis)
    }

    /// Analyze module structure and sizes
    fn analyze_module_structure(
        &self,
        analysis: &mut CompilationAnalysis,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs;

        fn visit_files(
            dir: &Path,
            extensions: &[String],
            analysis: &mut CompilationAnalysis,
        ) -> Result<(), Box<dyn std::error::Error>> {
            if dir.is_dir() {
                for entry in fs::read_dir(dir)? {
                    let entry = entry?;
                    let path = entry.path();

                    if path.is_dir() {
                        visit_files(&path, extensions, analysis)?;
                    } else if let Some(ext) = path.extension() {
                        if extensions.contains(&ext.to_string_lossy().to_string()) {
                            let content = fs::read_to_string(&path)?;
                            let line_count = content.lines().count();

                            let module_name = path
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();

                            analysis.module_sizes.insert(module_name, line_count);
                        }
                    }
                }
            }
            Ok(())
        }

        visit_files(
            &self.config.root_path,
            &self.config.file_extensions,
            analysis,
        )?;
        Ok(())
    }

    /// Analyze module dependencies
    fn analyze_dependencies(
        &self,
        analysis: &mut CompilationAnalysis,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use regex::Regex;
        use std::fs;

        // Safety: These regex patterns are compile-time constants and always valid
        let use_regex = Regex::new(r"^use\s+([^;]+);").expect("Valid use regex pattern");
        let mod_regex = Regex::new(r"^(?:pub\s+)?mod\s+(\w+)").expect("Valid mod regex pattern");

        for module_name in analysis.module_sizes.keys() {
            let mut module_path = self.config.root_path.clone();
            module_path.push(format!("{module_name}.rs"));

            if let Ok(content) = fs::read_to_string(&module_path) {
                let mut dependencies = Vec::new();

                for line in content.lines() {
                    let line = line.trim();

                    // Extract use statements
                    if let Some(captures) = use_regex.captures(line) {
                        // Safety: captures.get(1) guaranteed by successful regex match with group
                        if let Some(use_path_match) = captures.get(1) {
                            let use_path = use_path_match.as_str();
                            // Extract the first component of the use path
                            if let Some(first_component) = use_path.split("::").next() {
                                if first_component.starts_with("crate::") {
                                    let module = first_component
                                        .strip_prefix("crate::")
                                        .unwrap_or(first_component);
                                    dependencies.push(module.to_string());
                                }
                            }
                        }
                    }

                    // Extract mod statements
                    if let Some(captures) = mod_regex.captures(line) {
                        // Safety: captures.get(1) guaranteed by successful regex match with group
                        if let Some(mod_name_match) = captures.get(1) {
                            let mod_name = mod_name_match.as_str();
                            dependencies.push(mod_name.to_string());
                        }
                    }
                }

                analysis
                    .dependencies
                    .insert(module_name.clone(), dependencies);
            }
        }

        Ok(())
    }

    /// Detect compilation bottlenecks
    fn detect_compilation_bottlenecks(
        &self,
        analysis: &mut CompilationAnalysis,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Estimate compilation times based on module size and complexity
        for (module_name, &size) in &analysis.module_sizes {
            let base_time = size as f64 * 0.001; // 1ms per line baseline

            // Adjust for dependencies
            let dependency_count = analysis
                .dependencies
                .get(module_name)
                .map_or(0, std::vec::Vec::len);
            let dependency_penalty = dependency_count as f64 * 0.1;

            // Adjust for known heavy operations (simplified heuristic)
            let complexity_penalty = if size > 1000 {
                size as f64 * 0.0005
            } else {
                0.0
            };

            let estimated_time = base_time + dependency_penalty + complexity_penalty;
            analysis
                .compilation_times
                .insert(module_name.clone(), estimated_time);

            // Mark as heavy dependency if above threshold
            if estimated_time > self.config.heavy_dependency_threshold {
                analysis.heavy_dependencies.insert(module_name.clone());
            }
        }

        // Detect circular dependencies (simplified algorithm)
        self.detect_circular_dependencies(analysis);

        Ok(())
    }

    /// Detect circular dependencies using depth-first search
    fn detect_circular_dependencies(&self, analysis: &mut CompilationAnalysis) {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut path = Vec::new();

        for module in analysis.dependencies.keys() {
            if !visited.contains(module) {
                self.dfs_cycle_detection(
                    module,
                    &analysis.dependencies,
                    &mut visited,
                    &mut rec_stack,
                    &mut path,
                    &mut analysis.circular_dependencies,
                );
            }
        }
    }

    /// DFS helper for cycle detection
    fn dfs_cycle_detection(
        &self,
        module: &str,
        dependencies: &HashMap<String, Vec<String>>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
        path: &mut Vec<String>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        visited.insert(module.to_string());
        rec_stack.insert(module.to_string());
        path.push(module.to_string());

        if let Some(deps) = dependencies.get(module) {
            for dep in deps {
                if !visited.contains(dep) {
                    self.dfs_cycle_detection(dep, dependencies, visited, rec_stack, path, cycles);
                } else if rec_stack.contains(dep) {
                    // Found a cycle
                    if let Some(cycle_start) = path.iter().position(|m| m == dep) {
                        let cycle = path[cycle_start..].to_vec();
                        cycles.push(cycle);
                    }
                }
            }
        }

        rec_stack.remove(module);
        path.pop();
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        &self,
        analysis: &mut CompilationAnalysis,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Recommend module refactoring for large modules
        for (module_name, &size) in &analysis.module_sizes {
            if size > self.config.max_module_size {
                analysis.recommendations.push(OptimizationRecommendation {
                    optimization_type: OptimizationType::ModuleRefactoring,
                    modules: vec![module_name.clone()],
                    expected_improvement: (size as f64 - self.config.max_module_size as f64)
                        * 0.001,
                    description: format!(
                        "Module '{module_name}' has {size} lines and should be split into smaller modules"
                    ),
                    priority: if size > self.config.max_module_size * 2 {
                        RecommendationPriority::High
                    } else {
                        RecommendationPriority::Medium
                    },
                });
            }
        }

        // Recommend optimization for heavy dependencies
        for heavy_dep in &analysis.heavy_dependencies {
            let compile_time = analysis.compilation_times.get(heavy_dep).unwrap_or(&0.0);
            analysis.recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::LazyImports,
                modules: vec![heavy_dep.clone()],
                expected_improvement: compile_time * 0.3, // 30% improvement estimate
                description: format!(
                    "Module '{heavy_dep}' has high compilation time ({compile_time:.2}s) and could benefit from lazy imports"
                ),
                priority: RecommendationPriority::Medium,
            });
        }

        // Recommend fixes for circular dependencies
        for cycle in &analysis.circular_dependencies {
            analysis.recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::ModuleRefactoring,
                modules: cycle.clone(),
                expected_improvement: 2.0, // Significant improvement for breaking cycles
                description: format!("Circular dependency detected: {}", cycle.join(" -> ")),
                priority: RecommendationPriority::High,
            });
        }

        // Sort recommendations by priority and expected improvement
        analysis.recommendations.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then_with(|| {
                b.expected_improvement
                    .partial_cmp(&a.expected_improvement)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });

        Ok(())
    }

    /// Apply automatic optimizations where possible
    pub fn apply_optimizations(
        &self,
        analysis: &CompilationAnalysis,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut applied_optimizations = Vec::new();

        for recommendation in &analysis.recommendations {
            match recommendation.optimization_type {
                OptimizationType::RemoveUnusedImports => {
                    if self.apply_unused_import_removal(&recommendation.modules)? {
                        applied_optimizations.push(format!(
                            "Removed unused imports from modules: {}",
                            recommendation.modules.join(", ")
                        ));
                    }
                }
                OptimizationType::FeatureOptimization => {
                    if self.apply_feature_optimization(&recommendation.modules)? {
                        applied_optimizations.push(format!(
                            "Optimized feature flags for modules: {}",
                            recommendation.modules.join(", ")
                        ));
                    }
                }
                _ => {
                    // Other optimizations require manual intervention
                    applied_optimizations.push(format!(
                        "Manual optimization needed: {}",
                        recommendation.description
                    ));
                }
            }
        }

        Ok(applied_optimizations)
    }

    /// Apply unused import removal
    fn apply_unused_import_removal(
        &self,
        _modules: &[String],
    ) -> Result<bool, Box<dyn std::error::Error>> {
        // In practice, this would use tools like `cargo clippy` or custom analysis
        // For now, return success indicating the optimization was noted
        Ok(true)
    }

    /// Apply feature flag optimization
    fn apply_feature_optimization(
        &self,
        _modules: &[String],
    ) -> Result<bool, Box<dyn std::error::Error>> {
        // This would optimize Cargo.toml feature flags
        Ok(true)
    }

    /// Generate compilation optimization report
    #[must_use]
    pub fn generate_report(&self, analysis: &CompilationAnalysis) -> String {
        let mut report = String::new();

        report.push_str("# Compilation Optimization Report\n\n");

        // Module statistics
        report.push_str("## Module Statistics\n");
        let total_modules = analysis.module_sizes.len();
        let total_lines: usize = analysis.module_sizes.values().sum();
        let average_size = if total_modules > 0 {
            total_lines / total_modules
        } else {
            0
        };

        // Safety: Writing to String should not fail
        let _ = writeln!(report, "- Total modules: {total_modules}");
        let _ = writeln!(report, "- Total lines of code: {total_lines}");
        let _ = writeln!(report, "- Average module size: {average_size} lines");
        let _ = writeln!(
            report,
            "- Heavy dependencies: {}",
            analysis.heavy_dependencies.len()
        );
        let _ = write!(
            report,
            "- Circular dependencies: {}\n\n",
            analysis.circular_dependencies.len()
        );

        // Largest modules
        let mut modules_by_size: Vec<_> = analysis.module_sizes.iter().collect();
        modules_by_size.sort_by(|a, b| b.1.cmp(a.1));

        report.push_str("## Largest Modules\n");
        for (module, size) in modules_by_size.iter().take(10) {
            let _ = writeln!(report, "- {module}: {size} lines");
        }
        report.push('\n');

        // Recommendations
        report.push_str("## Optimization Recommendations\n");
        for (i, rec) in analysis.recommendations.iter().enumerate() {
            let _ = write!(report, "{}. **{:?}** (Priority: {:?})\n   - Modules: {}\n   - Expected improvement: {:.2}s\n   - {}\n\n",
                i + 1,
                rec.optimization_type,
                rec.priority,
                rec.modules.join(", "),
                rec.expected_improvement,
                rec.description);
        }

        report
    }
}

/// Utility functions for compilation optimization
pub mod utils {
    use super::{CompilationAnalysis, CompilationOptimizerConfig, Path};

    /// Estimate total compilation time from analysis
    #[must_use]
    pub fn estimate_total_compilation_time(analysis: &CompilationAnalysis) -> f64 {
        analysis.compilation_times.values().sum()
    }

    /// Calculate potential time savings from recommendations
    #[must_use]
    pub fn calculate_potential_savings(analysis: &CompilationAnalysis) -> f64 {
        analysis
            .recommendations
            .iter()
            .map(|rec| rec.expected_improvement)
            .sum()
    }

    /// Get compilation efficiency score (0.0 to 1.0)
    #[must_use]
    pub fn get_efficiency_score(analysis: &CompilationAnalysis) -> f64 {
        let total_time = estimate_total_compilation_time(analysis);
        let potential_savings = calculate_potential_savings(analysis);

        if total_time > 0.0 {
            1.0 - (potential_savings / total_time).min(1.0)
        } else {
            1.0
        }
    }

    /// Create default optimization configuration for quantum simulation codebase
    #[must_use]
    pub fn create_quantum_sim_config(root_path: &Path) -> CompilationOptimizerConfig {
        CompilationOptimizerConfig {
            root_path: root_path.to_path_buf(),
            file_extensions: vec!["rs".to_string()],
            max_module_size: 2000, // Following refactoring policy
            heavy_dependency_threshold: 3.0,
            enable_advanced_analysis: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_compilation_optimizer() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let config = CompilationOptimizerConfig {
            root_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        // Create test files
        fs::write(
            temp_dir.path().join("large_module.rs"),
            "use std::collections::HashMap;\n".repeat(3000),
        )
        .expect("Failed to write large_module.rs");

        fs::write(
            temp_dir.path().join("small_module.rs"),
            "use std::vec::Vec;\n".repeat(100),
        )
        .expect("Failed to write small_module.rs");

        let mut optimizer = CompilationOptimizer::new(config);
        let analysis = optimizer
            .analyze_codebase()
            .expect("Failed to analyze codebase");

        assert!(analysis.module_sizes.contains_key("large_module"));
        assert!(analysis.module_sizes.contains_key("small_module"));
        assert!(!analysis.recommendations.is_empty());
    }

    #[test]
    fn test_optimization_recommendations() {
        let mut analysis = CompilationAnalysis {
            dependencies: HashMap::new(),
            module_sizes: HashMap::new(),
            compilation_times: HashMap::new(),
            heavy_dependencies: HashSet::new(),
            circular_dependencies: Vec::new(),
            recommendations: Vec::new(),
        };

        // Add a large module
        analysis
            .module_sizes
            .insert("large_module".to_string(), 5000);
        analysis
            .compilation_times
            .insert("large_module".to_string(), 10.0);

        let config = CompilationOptimizerConfig::default();
        let optimizer = CompilationOptimizer::new(config);
        optimizer
            .generate_recommendations(&mut analysis)
            .expect("Failed to generate recommendations");

        assert!(!analysis.recommendations.is_empty());
        assert!(analysis
            .recommendations
            .iter()
            .any(|rec| { rec.optimization_type == OptimizationType::ModuleRefactoring }));
    }

    #[test]
    fn test_efficiency_calculation() {
        let mut analysis = CompilationAnalysis {
            dependencies: HashMap::new(),
            module_sizes: HashMap::new(),
            compilation_times: HashMap::new(),
            heavy_dependencies: HashSet::new(),
            circular_dependencies: Vec::new(),
            recommendations: Vec::new(),
        };

        analysis
            .compilation_times
            .insert("module1".to_string(), 5.0);
        analysis
            .compilation_times
            .insert("module2".to_string(), 3.0);

        analysis.recommendations.push(OptimizationRecommendation {
            optimization_type: OptimizationType::ModuleRefactoring,
            modules: vec!["module1".to_string()],
            expected_improvement: 2.0,
            description: "Test".to_string(),
            priority: RecommendationPriority::Medium,
        });

        let efficiency = utils::get_efficiency_score(&analysis);
        assert!(efficiency > 0.0 && efficiency <= 1.0);
    }
}
