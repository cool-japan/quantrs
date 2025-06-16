//! Benchmark analysis and reporting

use crate::benchmark::{
    metrics::{aggregation, BenchmarkMetrics},
    runner::BenchmarkResult,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Performance analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Report metadata
    pub metadata: ReportMetadata,
    /// Summary statistics
    pub summary: SummaryStatistics,
    /// Detailed analysis by backend
    pub backend_analysis: HashMap<String, BackendAnalysis>,
    /// Detailed analysis by sampler
    pub sampler_analysis: HashMap<String, SamplerAnalysis>,
    /// Scaling analysis
    pub scaling_analysis: ScalingAnalysis,
    /// Comparative analysis
    pub comparison: ComparativeAnalysis,
    /// Recommendations
    pub recommendations: Vec<Recommendation>,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub generated_at: std::time::SystemTime,
    pub total_benchmarks: usize,
    pub total_duration: std::time::Duration,
    pub platform_info: PlatformInfo,
}

/// Platform information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    pub os: String,
    pub cpu_cores: usize,
    pub cpu_model: String,
    pub memory_gb: f64,
    pub rust_version: String,
}

/// Summary statistics across all benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryStatistics {
    pub total_samples: usize,
    pub best_time_per_sample: std::time::Duration,
    pub best_energy_found: f64,
    pub most_efficient_backend: String,
    pub most_efficient_sampler: String,
    pub overall_metrics: BenchmarkMetrics,
}

/// Analysis for a specific backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendAnalysis {
    pub name: String,
    pub num_benchmarks: usize,
    pub success_rate: f64,
    pub avg_time_per_sample: std::time::Duration,
    pub avg_memory_usage: usize,
    pub best_problem_size: usize,
    pub efficiency_by_size: HashMap<usize, f64>,
    pub efficiency_by_density: HashMap<String, f64>,
}

/// Analysis for a specific sampler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerAnalysis {
    pub name: String,
    pub num_benchmarks: usize,
    pub avg_solution_quality: f64,
    pub convergence_rate: f64,
    pub best_parameters: HashMap<String, f64>,
    pub performance_by_problem_type: HashMap<String, f64>,
}

/// Scaling analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingAnalysis {
    pub time_complexity: ComplexityEstimate,
    pub memory_complexity: ComplexityEstimate,
    pub weak_scaling_efficiency: f64,
    pub strong_scaling_efficiency: f64,
    pub optimal_problem_sizes: Vec<usize>,
}

/// Complexity estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityEstimate {
    pub order: String, // e.g., "O(n)", "O(n²)", "O(n log n)"
    pub coefficient: f64,
    pub r_squared: f64, // Goodness of fit
}

/// Comparative analysis between backends/samplers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeAnalysis {
    pub speedup_matrix: HashMap<(String, String), f64>,
    pub quality_comparison: HashMap<String, f64>,
    pub efficiency_ranking: Vec<(String, f64)>,
    pub pareto_frontier: Vec<ParetoPoint>,
}

/// Point on Pareto frontier (quality vs performance trade-off)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPoint {
    pub configuration: String,
    pub quality_score: f64,
    pub performance_score: f64,
}

/// Recommendation based on analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub category: RecommendationCategory,
    pub message: String,
    pub impact: ImpactLevel,
    pub details: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Configuration,
    Hardware,
    Algorithm,
    Optimization,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ImpactLevel {
    High,
    Medium,
    Low,
}

impl PerformanceReport {
    /// Generate report from benchmark results
    pub fn from_results(results: &[BenchmarkResult]) -> Result<Self, Box<dyn std::error::Error>> {
        if results.is_empty() {
            return Err("No benchmark results to analyze".into());
        }

        let start_time = results.first().unwrap().timestamp;
        let end_time = results.last().unwrap().timestamp;
        let total_duration = end_time
            .duration_since(start_time)
            .unwrap_or(std::time::Duration::ZERO);

        let metadata = ReportMetadata {
            generated_at: std::time::SystemTime::now(),
            total_benchmarks: results.len(),
            total_duration,
            platform_info: Self::get_platform_info(),
        };

        let summary = Self::calculate_summary(results);
        let backend_analysis = Self::analyze_backends(results);
        let sampler_analysis = Self::analyze_samplers(results);
        let scaling_analysis = Self::analyze_scaling(results);
        let comparison = Self::comparative_analysis(results);
        let recommendations = Self::generate_recommendations(&summary, &scaling_analysis);

        Ok(Self {
            metadata,
            summary,
            backend_analysis,
            sampler_analysis,
            scaling_analysis,
            comparison,
            recommendations,
        })
    }

    /// Get platform information
    fn get_platform_info() -> PlatformInfo {
        PlatformInfo {
            os: std::env::consts::OS.to_string(),
            cpu_cores: num_cpus::get(),
            cpu_model: "Unknown".to_string(), // Would need system-specific code
            memory_gb: 0.0,                   // Would need system-specific code
            rust_version: std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()),
        }
    }

    /// Calculate summary statistics
    fn calculate_summary(results: &[BenchmarkResult]) -> SummaryStatistics {
        let total_samples: usize = results
            .iter()
            .map(|r| r.metrics.quality.unique_solutions)
            .sum();

        let best_time_per_sample = results
            .iter()
            .map(|r| r.metrics.timings.time_per_sample)
            .min()
            .unwrap_or(std::time::Duration::ZERO);

        let best_energy_found = results
            .iter()
            .map(|r| r.metrics.quality.best_energy)
            .fold(f64::INFINITY, f64::min);

        // Find most efficient configurations
        let mut backend_efficiency: HashMap<String, f64> = HashMap::new();
        let mut sampler_efficiency: HashMap<String, f64> = HashMap::new();

        for result in results {
            let efficiency = result.metrics.calculate_efficiency();

            backend_efficiency
                .entry(result.backend_name.clone())
                .and_modify(|e| *e += efficiency.samples_per_second)
                .or_insert(efficiency.samples_per_second);

            sampler_efficiency
                .entry(result.sampler_name.clone())
                .and_modify(|e| *e += efficiency.samples_per_second)
                .or_insert(efficiency.samples_per_second);
        }

        let most_efficient_backend = backend_efficiency
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k.clone())
            .unwrap_or_default();

        let most_efficient_sampler = sampler_efficiency
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k.clone())
            .unwrap_or_default();

        // Create aggregate metrics
        let metrics_vec: Vec<_> = results.iter().map(|r| r.metrics.clone()).collect();
        let aggregated = aggregation::aggregate_metrics(&metrics_vec);

        SummaryStatistics {
            total_samples,
            best_time_per_sample,
            best_energy_found,
            most_efficient_backend,
            most_efficient_sampler,
            overall_metrics: BenchmarkMetrics::new(
                aggregated.problem_sizes.iter().sum::<usize>() / aggregated.problem_sizes.len(),
                0.5, // Average density
            ),
        }
    }

    /// Analyze performance by backend
    fn analyze_backends(results: &[BenchmarkResult]) -> HashMap<String, BackendAnalysis> {
        let mut analysis = HashMap::new();

        // Group results by backend
        let mut by_backend: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
        for result in results {
            by_backend
                .entry(result.backend_name.clone())
                .or_default()
                .push(result);
        }

        for (backend_name, backend_results) in by_backend {
            let num_benchmarks = backend_results.len();
            let success_rate = 1.0; // All completed successfully

            let avg_time_per_sample = backend_results
                .iter()
                .map(|r| r.metrics.timings.time_per_sample.as_secs_f64())
                .sum::<f64>()
                / backend_results.len() as f64;

            let avg_memory_usage = backend_results
                .iter()
                .map(|r| r.metrics.memory.peak_memory)
                .sum::<usize>()
                / backend_results.len();

            // Find best problem size
            let mut size_performance: HashMap<usize, Vec<f64>> = HashMap::new();
            for result in &backend_results {
                let efficiency = result.metrics.calculate_efficiency();
                size_performance
                    .entry(result.problem_size)
                    .or_default()
                    .push(efficiency.samples_per_second);
            }

            let best_problem_size = size_performance
                .iter()
                .map(|(size, perfs)| {
                    let avg_perf = perfs.iter().sum::<f64>() / perfs.len() as f64;
                    (*size, avg_perf)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(size, _)| size)
                .unwrap_or(0);

            // Calculate efficiency by size
            let efficiency_by_size: HashMap<usize, f64> = size_performance
                .iter()
                .map(|(size, perfs)| {
                    let avg = perfs.iter().sum::<f64>() / perfs.len() as f64;
                    (*size, avg)
                })
                .collect();

            // Calculate efficiency by density
            let mut density_performance: HashMap<String, Vec<f64>> = HashMap::new();
            for result in &backend_results {
                let density_str = format!("{:.1}", result.problem_density);
                let efficiency = result.metrics.calculate_efficiency();
                density_performance
                    .entry(density_str)
                    .or_default()
                    .push(efficiency.samples_per_second);
            }

            let efficiency_by_density: HashMap<String, f64> = density_performance
                .iter()
                .map(|(density, perfs)| {
                    let avg = perfs.iter().sum::<f64>() / perfs.len() as f64;
                    (density.clone(), avg)
                })
                .collect();

            analysis.insert(
                backend_name.clone(),
                BackendAnalysis {
                    name: backend_name,
                    num_benchmarks,
                    success_rate,
                    avg_time_per_sample: std::time::Duration::from_secs_f64(avg_time_per_sample),
                    avg_memory_usage,
                    best_problem_size,
                    efficiency_by_size,
                    efficiency_by_density,
                },
            );
        }

        analysis
    }

    /// Analyze performance by sampler
    fn analyze_samplers(results: &[BenchmarkResult]) -> HashMap<String, SamplerAnalysis> {
        let mut analysis = HashMap::new();

        // Group results by sampler
        let mut by_sampler: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
        for result in results {
            by_sampler
                .entry(result.sampler_name.clone())
                .or_default()
                .push(result);
        }

        for (sampler_name, sampler_results) in by_sampler {
            let num_benchmarks = sampler_results.len();

            let avg_solution_quality = sampler_results
                .iter()
                .map(|r| r.metrics.quality.best_energy)
                .sum::<f64>()
                / sampler_results.len() as f64;

            // Simple convergence rate estimate
            let convergence_rate = sampler_results
                .iter()
                .filter_map(|r| r.metrics.quality.time_to_target)
                .map(|t| 1.0 / t.as_secs_f64())
                .sum::<f64>()
                / sampler_results.len() as f64;

            // Placeholder for best parameters
            let best_parameters = HashMap::new();

            // Performance by problem type (density)
            let mut problem_type_performance: HashMap<String, Vec<f64>> = HashMap::new();
            for result in &sampler_results {
                let problem_type = if result.problem_density < 0.3 {
                    "sparse"
                } else if result.problem_density < 0.7 {
                    "medium"
                } else {
                    "dense"
                };

                problem_type_performance
                    .entry(problem_type.to_string())
                    .or_default()
                    .push(result.metrics.quality.best_energy);
            }

            let performance_by_problem_type: HashMap<String, f64> = problem_type_performance
                .iter()
                .map(|(ptype, energies)| {
                    let avg = energies.iter().sum::<f64>() / energies.len() as f64;
                    (ptype.clone(), avg)
                })
                .collect();

            analysis.insert(
                sampler_name.clone(),
                SamplerAnalysis {
                    name: sampler_name,
                    num_benchmarks,
                    avg_solution_quality,
                    convergence_rate,
                    best_parameters,
                    performance_by_problem_type,
                },
            );
        }

        analysis
    }

    /// Analyze scaling behavior
    fn analyze_scaling(results: &[BenchmarkResult]) -> ScalingAnalysis {
        // Extract time vs problem size data
        let mut time_data: Vec<(f64, f64)> = Vec::new();
        let mut memory_data: Vec<(f64, f64)> = Vec::new();

        for result in results {
            let size = result.problem_size as f64;
            let time = result.metrics.timings.compute_time.as_secs_f64();
            let memory = result.metrics.memory.peak_memory as f64;

            time_data.push((size, time));
            memory_data.push((size, memory));
        }

        // Fit complexity models
        let time_complexity = Self::fit_complexity_model(&time_data);
        let memory_complexity = Self::fit_complexity_model(&memory_data);

        // Calculate scaling efficiencies (simplified)
        let weak_scaling_efficiency = 0.8; // Placeholder
        let strong_scaling_efficiency = 0.7; // Placeholder

        // Find optimal problem sizes based on efficiency
        let mut size_efficiencies: HashMap<usize, f64> = HashMap::new();
        for result in results {
            let efficiency = result.metrics.calculate_efficiency();
            size_efficiencies
                .entry(result.problem_size)
                .and_modify(|e| *e += efficiency.scalability_factor)
                .or_insert(efficiency.scalability_factor);
        }

        let mut optimal_sizes: Vec<(usize, f64)> = size_efficiencies.into_iter().collect();
        optimal_sizes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let optimal_problem_sizes: Vec<usize> = optimal_sizes
            .into_iter()
            .take(3)
            .map(|(size, _)| size)
            .collect();

        ScalingAnalysis {
            time_complexity,
            memory_complexity,
            weak_scaling_efficiency,
            strong_scaling_efficiency,
            optimal_problem_sizes,
        }
    }

    /// Fit complexity model to data
    fn fit_complexity_model(data: &[(f64, f64)]) -> ComplexityEstimate {
        // Simple linear regression (in practice would fit various models)
        if data.is_empty() {
            return ComplexityEstimate {
                order: "O(1)".to_string(),
                coefficient: 0.0,
                r_squared: 0.0,
            };
        }

        let n = data.len() as f64;
        let sum_x: f64 = data.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = data.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = data.iter().map(|(x, _)| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Calculate R²
        let mean_y = sum_y / n;
        let ss_tot: f64 = data.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = data
            .iter()
            .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
            .sum();
        let r_squared = 1.0 - (ss_res / ss_tot);

        // Determine complexity order based on slope
        let order = if slope < 0.1 {
            "O(1)"
        } else if slope < 1.5 {
            "O(n)"
        } else if slope < 2.5 {
            "O(n²)"
        } else {
            "O(n³)"
        }
        .to_string();

        ComplexityEstimate {
            order,
            coefficient: slope,
            r_squared,
        }
    }

    /// Comparative analysis
    fn comparative_analysis(results: &[BenchmarkResult]) -> ComparativeAnalysis {
        let mut speedup_matrix = HashMap::new();
        let mut quality_comparison = HashMap::new();
        let mut efficiency_scores: HashMap<String, f64> = HashMap::new();

        // Calculate average performance for each configuration
        let mut config_performance: HashMap<String, (f64, f64)> = HashMap::new();
        for result in results {
            let config = format!("{}-{}", result.backend_name, result.sampler_name);
            let efficiency = result.metrics.calculate_efficiency();
            let quality = result.metrics.quality.best_energy;

            config_performance
                .entry(config.clone())
                .and_modify(|(perf, qual)| {
                    *perf += efficiency.samples_per_second;
                    *qual = qual.min(quality);
                })
                .or_insert((efficiency.samples_per_second, quality));

            efficiency_scores
                .entry(config)
                .and_modify(|e| *e += efficiency.samples_per_second)
                .or_insert(efficiency.samples_per_second);
        }

        // Calculate speedup matrix
        let configs: Vec<String> = config_performance.keys().cloned().collect();
        for config1 in &configs {
            for config2 in &configs {
                if let (Some((perf1, _)), Some((perf2, _))) = (
                    config_performance.get(config1),
                    config_performance.get(config2),
                ) {
                    let speedup = perf1 / perf2;
                    speedup_matrix.insert((config1.clone(), config2.clone()), speedup);
                }
            }
        }

        // Quality comparison
        for (config, (_, quality)) in &config_performance {
            quality_comparison.insert(config.clone(), *quality);
        }

        // Efficiency ranking
        let mut efficiency_ranking: Vec<(String, f64)> = efficiency_scores.into_iter().collect();
        efficiency_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Calculate Pareto frontier
        let mut pareto_points: Vec<ParetoPoint> = Vec::new();
        for (config, (performance, quality)) in config_performance {
            let quality_score = -quality; // Negative because lower is better
            let performance_score = performance;

            // Check if dominated
            let is_dominated = pareto_points.iter().any(|p| {
                p.quality_score >= quality_score
                    && p.performance_score >= performance_score
                    && (p.quality_score > quality_score || p.performance_score > performance_score)
            });

            if !is_dominated {
                // Remove dominated points
                pareto_points.retain(|p| {
                    !(quality_score >= p.quality_score
                        && performance_score >= p.performance_score
                        && (quality_score > p.quality_score
                            || performance_score > p.performance_score))
                });

                pareto_points.push(ParetoPoint {
                    configuration: config,
                    quality_score,
                    performance_score,
                });
            }
        }

        ComparativeAnalysis {
            speedup_matrix,
            quality_comparison,
            efficiency_ranking,
            pareto_frontier: pareto_points,
        }
    }

    /// Generate recommendations
    fn generate_recommendations(
        summary: &SummaryStatistics,
        scaling: &ScalingAnalysis,
    ) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Configuration recommendations
        if !scaling.optimal_problem_sizes.is_empty() {
            recommendations.push(Recommendation {
                category: RecommendationCategory::Configuration,
                message: format!(
                    "Optimal problem sizes for this system: {:?}",
                    scaling.optimal_problem_sizes
                ),
                impact: ImpactLevel::High,
                details: HashMap::new(),
            });
        }

        // Hardware recommendations
        if scaling.time_complexity.order.contains("³")
            || scaling.time_complexity.order.contains("⁴")
        {
            recommendations.push(Recommendation {
                category: RecommendationCategory::Hardware,
                message: "Consider GPU acceleration for large problem instances".to_string(),
                impact: ImpactLevel::High,
                details: HashMap::from([(
                    "reason".to_string(),
                    format!("Time complexity is {}", scaling.time_complexity.order),
                )]),
            });
        }

        // Algorithm recommendations
        if summary.best_energy_found > -100.0 {
            // Arbitrary threshold
            recommendations.push(Recommendation {
                category: RecommendationCategory::Algorithm,
                message: "Consider hybrid algorithms for better solution quality".to_string(),
                impact: ImpactLevel::Medium,
                details: HashMap::new(),
            });
        }

        // Optimization recommendations
        if scaling.weak_scaling_efficiency < 0.7 {
            recommendations.push(Recommendation {
                category: RecommendationCategory::Optimization,
                message: "Parallel efficiency is low - consider optimizing communication patterns"
                    .to_string(),
                impact: ImpactLevel::High,
                details: HashMap::from([(
                    "efficiency".to_string(),
                    scaling.weak_scaling_efficiency.to_string(),
                )]),
            });
        }

        recommendations
    }

    /// Save report to file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// Generate human-readable summary
impl PerformanceReport {
    pub fn generate_summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str(&format!("# Performance Benchmark Report\n\n"));
        summary.push_str(&format!("Generated: {:?}\n", self.metadata.generated_at));
        summary.push_str(&format!(
            "Total benchmarks: {}\n",
            self.metadata.total_benchmarks
        ));
        summary.push_str(&format!("Duration: {:?}\n\n", self.metadata.total_duration));

        summary.push_str("## Summary\n");
        summary.push_str(&format!(
            "- Best time per sample: {:?}\n",
            self.summary.best_time_per_sample
        ));
        summary.push_str(&format!(
            "- Best energy found: {:.4}\n",
            self.summary.best_energy_found
        ));
        summary.push_str(&format!(
            "- Most efficient backend: {}\n",
            self.summary.most_efficient_backend
        ));
        summary.push_str(&format!(
            "- Most efficient sampler: {}\n\n",
            self.summary.most_efficient_sampler
        ));

        summary.push_str("## Recommendations\n");
        for rec in &self.recommendations {
            summary.push_str(&format!(
                "- [{}] {}\n",
                match rec.impact {
                    ImpactLevel::High => "HIGH",
                    ImpactLevel::Medium => "MEDIUM",
                    ImpactLevel::Low => "LOW",
                },
                rec.message
            ));
        }

        summary
    }
}
