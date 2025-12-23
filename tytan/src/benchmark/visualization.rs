//! Benchmark visualization utilities

use crate::benchmark::analysis::PerformanceReport;

/// Benchmark visualizer for generating plots and charts
pub struct BenchmarkVisualizer {
    report: PerformanceReport,
}

impl BenchmarkVisualizer {
    /// Create new visualizer from performance report
    pub const fn new(report: PerformanceReport) -> Self {
        Self { report }
    }

    /// Generate all visualizations
    pub fn generate_all(&self, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(output_dir)?;

        // Generate various plots
        self.generate_scaling_plot(output_dir)?;
        self.generate_efficiency_heatmap(output_dir)?;
        self.generate_pareto_plot(output_dir)?;
        self.generate_comparison_chart(output_dir)?;

        // Generate HTML report
        self.generate_html_report(output_dir)?;

        Ok(())
    }

    /// Generate scaling plot
    fn generate_scaling_plot(&self, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "scirs")]
        {
            use crate::scirs_stub::scirs2_plot::{Line, Plot, Scatter};

            let mut plot = Plot::new();

            // Extract scaling data
            for (backend_name, analysis) in &self.report.backend_analysis {
                let mut sizes: Vec<f64> = Vec::new();
                let mut times: Vec<f64> = Vec::new();

                for (size, efficiency) in &analysis.efficiency_by_size {
                    sizes.push(*size as f64);
                    times.push(1.0 / efficiency); // Convert to time
                }

                // Sort by size
                let mut pairs: Vec<_> = sizes.into_iter().zip(times).collect();
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                let (sizes, times): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();

                plot.add_trace(Line::new(sizes, times).name(backend_name));
            }

            plot.set_title("Scaling Analysis");
            plot.set_xlabel("Problem Size");
            plot.set_ylabel("Time per Sample (s)");
            plot.save(&format!("{output_dir}/scaling_plot.html"))?;
        }

        // Fallback: generate CSV data
        self.generate_scaling_csv(output_dir)?;

        Ok(())
    }

    /// Generate CSV data for scaling plot
    fn generate_scaling_csv(&self, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        let mut csv_content = String::from("backend,problem_size,time_per_sample,efficiency\n");

        for (backend_name, analysis) in &self.report.backend_analysis {
            for (size, efficiency) in &analysis.efficiency_by_size {
                csv_content.push_str(&format!(
                    "{},{},{},{}\n",
                    backend_name,
                    size,
                    1.0 / efficiency,
                    efficiency
                ));
            }
        }

        let mut file = std::fs::File::create(format!("{output_dir}/scaling_data.csv"))?;
        file.write_all(csv_content.as_bytes())?;

        Ok(())
    }

    /// Generate efficiency heatmap
    fn generate_efficiency_heatmap(
        &self,
        output_dir: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "scirs")]
        {
            use crate::scirs_stub::scirs2_plot::{Heatmap, Plot};

            // Create efficiency matrix
            let backends: Vec<String> = self.report.backend_analysis.keys().cloned().collect();
            let samplers: Vec<String> = self.report.sampler_analysis.keys().cloned().collect();

            let mut z_data = vec![vec![0.0; samplers.len()]; backends.len()];

            // Fill matrix from speedup data
            for (i, backend) in backends.iter().enumerate() {
                for (j, sampler) in samplers.iter().enumerate() {
                    let mut config = format!("{backend}-{sampler}");

                    // Find efficiency for this configuration
                    if let Some(efficiency) = self
                        .report
                        .comparison
                        .efficiency_ranking
                        .iter()
                        .find(|(c, _)| c == &config)
                        .map(|(_, e)| *e)
                    {
                        z_data[i][j] = efficiency;
                    }
                }
            }

            let heatmap = Heatmap::new(z_data)
                .x_labels(samplers)
                .y_labels(backends)
                .colorscale("Viridis");

            let mut plot = Plot::new();
            plot.add_trace(heatmap);
            plot.set_title("Backend-Sampler Efficiency Heatmap");
            plot.save(&format!("{output_dir}/efficiency_heatmap.html"))?;
        }

        // Fallback: generate CSV
        self.generate_efficiency_csv(output_dir)?;

        Ok(())
    }

    /// Generate CSV data for efficiency heatmap
    fn generate_efficiency_csv(&self, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        let mut csv_content = String::from("configuration,efficiency,rank\n");

        for (i, (config, efficiency)) in
            self.report.comparison.efficiency_ranking.iter().enumerate()
        {
            csv_content.push_str(&format!("{},{},{}\n", config, efficiency, i + 1));
        }

        let mut file = std::fs::File::create(format!("{output_dir}/efficiency_ranking.csv"))?;
        file.write_all(csv_content.as_bytes())?;

        Ok(())
    }

    /// Generate Pareto frontier plot
    fn generate_pareto_plot(&self, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "scirs")]
        {
            use crate::scirs_stub::scirs2_plot::{Plot, Scatter};

            let mut plot = Plot::new();

            // All points
            let all_configs: Vec<String> = self
                .report
                .comparison
                .quality_comparison
                .keys()
                .cloned()
                .collect();
            let mut all_quality = Vec::new();
            let mut all_performance = Vec::new();

            for config in &all_configs {
                if let Some(&quality) = self.report.comparison.quality_comparison.get(config) {
                    all_quality.push(-quality); // Negative for visualization

                    // Find performance
                    let perf = self
                        .report
                        .comparison
                        .efficiency_ranking
                        .iter()
                        .find(|(c, _)| c == config)
                        .map_or(0.0, |(_, e)| *e);
                    all_performance.push(perf);
                }
            }

            plot.add_trace(
                Scatter::new(all_performance.clone(), all_quality.clone())
                    .mode("markers")
                    .name("All Configurations")
                    .text(all_configs),
            );

            // Pareto frontier
            let pareto_quality: Vec<f64> = self
                .report
                .comparison
                .pareto_frontier
                .iter()
                .map(|p| p.quality_score)
                .collect();
            let pareto_performance: Vec<f64> = self
                .report
                .comparison
                .pareto_frontier
                .iter()
                .map(|p| p.performance_score)
                .collect();
            let pareto_labels: Vec<String> = self
                .report
                .comparison
                .pareto_frontier
                .iter()
                .map(|p| p.configuration.clone())
                .collect();

            plot.add_trace(
                Scatter::new(pareto_performance, pareto_quality)
                    .mode("lines+markers")
                    .name("Pareto Frontier")
                    .text(pareto_labels)
                    .marker_size(10),
            );

            plot.set_title("Quality vs Performance Trade-off");
            plot.set_xlabel("Performance (samples/sec)");
            plot.set_ylabel("Solution Quality");
            plot.save(&format!("{output_dir}/pareto_plot.html"))?;
        }

        // Fallback: generate CSV
        self.generate_pareto_csv(output_dir)?;

        Ok(())
    }

    /// Generate CSV data for Pareto plot
    fn generate_pareto_csv(&self, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        let mut csv_content =
            String::from("configuration,quality_score,performance_score,is_pareto\n");

        // All configurations
        for (config, &quality) in &self.report.comparison.quality_comparison {
            let perf = self
                .report
                .comparison
                .efficiency_ranking
                .iter()
                .find(|(c, _)| c == config)
                .map_or(0.0, |(_, e)| *e);

            let is_pareto = self
                .report
                .comparison
                .pareto_frontier
                .iter()
                .any(|p| &p.configuration == config);

            csv_content.push_str(&format!(
                "{},{},{},{}\n",
                config,
                -quality, // Negative for visualization
                perf,
                is_pareto
            ));
        }

        let mut file = std::fs::File::create(format!("{output_dir}/pareto_data.csv"))?;
        file.write_all(csv_content.as_bytes())?;

        Ok(())
    }

    /// Generate comparison chart
    fn generate_comparison_chart(
        &self,
        output_dir: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "scirs")]
        {
            use crate::scirs_stub::scirs2_plot::{Bar, Plot};

            let mut plot = Plot::new();

            // Backend comparison
            let backend_names: Vec<String> = self.report.backend_analysis.keys().cloned().collect();
            let backend_times: Vec<f64> = backend_names
                .iter()
                .map(|name| {
                    self.report.backend_analysis[name]
                        .avg_time_per_sample
                        .as_secs_f64()
                })
                .collect();

            plot.add_trace(Bar::new(backend_names, backend_times).name("Avg Time per Sample"));

            plot.set_title("Backend Performance Comparison");
            plot.set_ylabel("Time per Sample (s)");
            plot.save(&format!("{output_dir}/comparison_chart.html"))?;
        }

        Ok(())
    }

    /// Generate HTML report
    fn generate_html_report(&self, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
        let html = format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #0066cc; }}
        .recommendation {{
            background-color: #f0f8ff;
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #0066cc;
        }}
        .high {{ border-left-color: #ff0000; }}
        .medium {{ border-left-color: #ff9900; }}
        .low {{ border-left-color: #00cc00; }}
    </style>
</head>
<body>
    <h1>Quantum Annealing Benchmark Report</h1>

    <h2>Summary</h2>
    <p>Generated: {:?}</p>
    <p>Total benchmarks: <span class="metric">{}</span></p>
    <p>Best time per sample: <span class="metric">{:?}</span></p>
    <p>Best energy found: <span class="metric">{:.4}</span></p>
    <p>Most efficient backend: <span class="metric">{}</span></p>
    <p>Most efficient sampler: <span class="metric">{}</span></p>

    <h2>Backend Analysis</h2>
    <table>
        <tr>
            <th>Backend</th>
            <th>Benchmarks</th>
            <th>Avg Time/Sample</th>
            <th>Avg Memory</th>
            <th>Best Size</th>
        </tr>
        {}
    </table>

    <h2>Scaling Analysis</h2>
    <p>Time complexity: <span class="metric">{}</span> (R² = {:.3})</p>
    <p>Memory complexity: <span class="metric">{}</span> (R² = {:.3})</p>
    <p>Optimal problem sizes: <span class="metric">{:?}</span></p>

    <h2>Recommendations</h2>
    {}

    <h2>Visualizations</h2>
    <p>See the following files in the output directory:</p>
    <ul>
        <li>scaling_plot.html / scaling_data.csv - Performance scaling analysis</li>
        <li>efficiency_heatmap.html / efficiency_ranking.csv - Backend-sampler efficiency</li>
        <li>pareto_plot.html / pareto_data.csv - Quality vs performance trade-off</li>
        <li>comparison_chart.html - Backend comparison</li>
    </ul>
</body>
</html>
"#,
            self.report.metadata.generated_at,
            self.report.metadata.total_benchmarks,
            self.report.summary.best_time_per_sample,
            self.report.summary.best_energy_found,
            self.report.summary.most_efficient_backend,
            self.report.summary.most_efficient_sampler,
            self.generate_backend_table_rows(),
            self.report.scaling_analysis.time_complexity.order,
            self.report.scaling_analysis.time_complexity.r_squared,
            self.report.scaling_analysis.memory_complexity.order,
            self.report.scaling_analysis.memory_complexity.r_squared,
            self.report.scaling_analysis.optimal_problem_sizes,
            self.generate_recommendations_html(),
        );

        std::fs::write(format!("{output_dir}/report.html"), html)?;

        Ok(())
    }

    /// Generate backend table rows for HTML
    fn generate_backend_table_rows(&self) -> String {
        let mut rows = String::new();

        for (name, analysis) in &self.report.backend_analysis {
            rows.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{:?}</td><td>{} MB</td><td>{}</td></tr>\n",
                name,
                analysis.num_benchmarks,
                analysis.avg_time_per_sample,
                analysis.avg_memory_usage / 1_048_576,
                analysis.best_problem_size,
            ));
        }

        rows
    }

    /// Generate recommendations HTML
    fn generate_recommendations_html(&self) -> String {
        let mut html = String::new();

        for rec in &self.report.recommendations {
            let impact_class = match rec.impact {
                crate::benchmark::analysis::ImpactLevel::High => "high",
                crate::benchmark::analysis::ImpactLevel::Medium => "medium",
                crate::benchmark::analysis::ImpactLevel::Low => "low",
            };

            html.push_str(&format!(
                r#"<div class="recommendation {}">{}</div>"#,
                impact_class, rec.message
            ));
        }

        html
    }
}
