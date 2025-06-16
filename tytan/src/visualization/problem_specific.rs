//! Problem-specific visualizations for quantum annealing
//!
//! This module provides specialized visualization routines for common
//! optimization problem types including TSP, graph coloring, scheduling, etc.

use crate::sampler::SampleResult;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "scirs")]
use crate::scirs_stub::{
    scirs2_graphs::{Graph, GraphLayout},
    scirs2_plot::{ColorMap, NetworkPlot, Plot2D},
};

/// Problem visualization types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VisualizationType {
    /// Traveling Salesman Problem
    TSP {
        coordinates: Vec<(f64, f64)>,
        city_names: Option<Vec<String>>,
    },
    /// Graph Coloring
    GraphColoring {
        adjacency_matrix: Array2<bool>,
        node_names: Option<Vec<String>>,
        max_colors: usize,
    },
    /// Maximum Cut
    MaxCut {
        adjacency_matrix: Array2<f64>,
        node_names: Option<Vec<String>>,
    },
    /// Job Shop Scheduling
    JobShop {
        n_jobs: usize,
        n_machines: usize,
        time_horizon: usize,
    },
    /// Number Partitioning
    NumberPartition { numbers: Vec<f64> },
    /// Knapsack Problem
    Knapsack {
        weights: Vec<f64>,
        values: Vec<f64>,
        capacity: f64,
    },
    /// Portfolio Optimization
    Portfolio {
        asset_names: Vec<String>,
        expected_returns: Vec<f64>,
        risk_matrix: Array2<f64>,
    },
    /// Custom visualization
    Custom {
        plot_function: String,
        metadata: HashMap<String, String>,
    },
}

/// Problem visualizer
pub struct ProblemVisualizer {
    problem_type: VisualizationType,
    samples: Vec<SampleResult>,
    config: VisualizationConfig,
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Show best solution only
    pub best_only: bool,
    /// Number of top solutions to show
    pub top_k: usize,
    /// Color scheme
    pub color_scheme: String,
    /// Node size for graph problems
    pub node_size: f64,
    /// Edge width for graph problems
    pub edge_width: f64,
    /// Animation settings
    pub animate: bool,
    /// Animation speed (fps)
    pub animation_speed: f64,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            best_only: false,
            top_k: 5,
            color_scheme: "viridis".to_string(),
            node_size: 50.0,
            edge_width: 2.0,
            animate: false,
            animation_speed: 2.0,
        }
    }
}

impl ProblemVisualizer {
    /// Create new problem visualizer
    pub fn new(problem_type: VisualizationType, config: VisualizationConfig) -> Self {
        Self {
            problem_type,
            samples: Vec::new(),
            config,
        }
    }

    /// Add sample results
    pub fn add_samples(&mut self, samples: Vec<SampleResult>) {
        self.samples.extend(samples);
    }

    /// Visualize the problem and solutions
    pub fn visualize(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.samples.is_empty() {
            return Err("No samples to visualize".into());
        }

        match &self.problem_type {
            VisualizationType::TSP {
                coordinates,
                city_names,
            } => self.visualize_tsp(coordinates, city_names)?,
            VisualizationType::GraphColoring {
                adjacency_matrix,
                node_names,
                max_colors,
            } => self.visualize_graph_coloring(adjacency_matrix, node_names, *max_colors)?,
            VisualizationType::MaxCut {
                adjacency_matrix,
                node_names,
            } => self.visualize_max_cut(adjacency_matrix, node_names)?,
            VisualizationType::JobShop {
                n_jobs,
                n_machines,
                time_horizon,
            } => self.visualize_job_shop(*n_jobs, *n_machines, *time_horizon)?,
            VisualizationType::NumberPartition { numbers } => {
                self.visualize_number_partition(numbers)?
            }
            VisualizationType::Knapsack {
                weights,
                values,
                capacity,
            } => self.visualize_knapsack(weights, values, *capacity)?,
            VisualizationType::Portfolio {
                asset_names,
                expected_returns,
                risk_matrix,
            } => self.visualize_portfolio(asset_names, expected_returns, risk_matrix)?,
            VisualizationType::Custom {
                plot_function,
                metadata,
            } => self.visualize_custom(plot_function, metadata)?,
        }

        Ok(())
    }

    /// Visualize TSP solution
    fn visualize_tsp(
        &self,
        coordinates: &[(f64, f64)],
        city_names: &Option<Vec<String>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n_cities = coordinates.len();

        // Get best solutions
        let best_samples = self.get_best_samples();

        #[cfg(feature = "scirs")]
        {
            use crate::scirs_stub::scirs2_plot::{Figure, Subplot};

            let fig = Figure::new();

            for (idx, sample) in best_samples.iter().enumerate() {
                if idx >= self.config.top_k {
                    break;
                }

                let subplot = fig.add_subplot(
                    (self.config.top_k as f64).sqrt().ceil() as usize,
                    (self.config.top_k as f64).sqrt().ceil() as usize,
                    idx + 1,
                )?;

                // Extract tour from binary variables
                let tour = self.extract_tsp_tour(sample, n_cities)?;

                // Plot cities
                let x: Vec<f64> = coordinates.iter().map(|c| c.0).collect();
                let y: Vec<f64> = coordinates.iter().map(|c| c.1).collect();

                subplot
                    .scatter(&x, &y)
                    .set_size(self.config.node_size)
                    .set_color("blue");

                // Plot tour edges
                for i in 0..tour.len() {
                    let from = tour[i];
                    let to = tour[(i + 1) % tour.len()];

                    subplot
                        .plot(
                            &[coordinates[from].0, coordinates[to].0],
                            &[coordinates[from].1, coordinates[to].1],
                        )
                        .set_color("red")
                        .set_linewidth(self.config.edge_width);
                }

                // Add city labels if provided
                if let Some(names) = city_names {
                    for (i, name) in names.iter().enumerate() {
                        subplot
                            .text(coordinates[i].0, coordinates[i].1, name)
                            .set_fontsize(8)
                            .set_ha("center");
                    }
                }

                subplot.set_title(&format!(
                    "Tour {}: Distance = {:.2}",
                    idx + 1,
                    sample.energy
                ));
                subplot.set_aspect("equal");
            }

            fig.suptitle("TSP Solutions");
            fig.show()?;
        }

        #[cfg(not(feature = "scirs"))]
        {
            // Export TSP data
            let export = TSPExport {
                coordinates: coordinates.to_vec(),
                city_names: city_names.clone(),
                best_tours: best_samples
                    .iter()
                    .take(self.config.top_k)
                    .map(|s| self.extract_tsp_tour(s, n_cities))
                    .collect::<Result<Vec<_>, _>>()?,
                tour_lengths: best_samples
                    .iter()
                    .take(self.config.top_k)
                    .map(|s| s.energy)
                    .collect(),
            };

            let json = serde_json::to_string_pretty(&export)?;
            std::fs::write("tsp_solution.json", json)?;
            println!("TSP solution exported to tsp_solution.json");
        }

        Ok(())
    }

    /// Extract TSP tour from binary variables
    fn extract_tsp_tour(
        &self,
        sample: &SampleResult,
        n_cities: usize,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        let tour = Vec::new();
        let mut visited = vec![false; n_cities];
        let current = 0;

        tour.push(current);
        visited[current] = true;

        // Follow edges to build tour
        for _ in 1..n_cities {
            let next_city = None;

            for j in 0..n_cities {
                if !visited[j] {
                    let edge_var = format!("x_{}_{}", current, j);
                    if sample.assignments.get(&edge_var).copied().unwrap_or(false) {
                        next_city = Some(j);
                        break;
                    }
                }
            }

            if let Some(next) = next_city {
                tour.push(next);
                visited[next] = true;
                current = next;
            } else {
                // Find first unvisited city as fallback
                for j in 0..n_cities {
                    if !visited[j] {
                        tour.push(j);
                        visited[j] = true;
                        current = j;
                        break;
                    }
                }
            }
        }

        Ok(tour)
    }

    /// Visualize graph coloring solution
    fn visualize_graph_coloring(
        &self,
        adjacency: &Array2<bool>,
        node_names: &Option<Vec<String>>,
        max_colors: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n_nodes = adjacency.nrows();
        let best_sample = self.get_best_sample()?;

        // Extract node colors
        let node_colors = self.extract_node_colors(best_sample, n_nodes, max_colors)?;

        #[cfg(feature = "scirs")]
        {
            use crate::scirs_stub::scirs2_graphs::spring_layout;
            use crate::scirs_stub::scirs2_plot::Figure;

            let fig = Figure::new();
            let ax = fig.add_subplot(1, 1, 1)?;

            // Create graph
            let edges = Vec::new();
            for i in 0..n_nodes {
                for j in i + 1..n_nodes {
                    if adjacency[[i, j]] {
                        edges.push((i, j));
                    }
                }
            }

            // Compute layout
            let positions = spring_layout(&edges, n_nodes)?;

            // Plot edges
            for (i, j) in &edges {
                ax.plot(
                    &[positions[*i].0, positions[*j].0],
                    &[positions[*i].1, positions[*j].1],
                )
                .set_color("gray")
                .set_alpha(0.5)
                .set_linewidth(1.0);
            }

            // Plot nodes with colors
            let color_palette = ["red", "blue", "green", "yellow", "purple", "orange"];

            for i in 0..n_nodes {
                let color = color_palette[node_colors[i] % color_palette.len()];

                ax.scatter(&[positions[i].0], &[positions[i].1])
                    .set_color(color)
                    .set_size(self.config.node_size)
                    .set_edgecolor("black");

                // Add labels
                let label = if let Some(names) = node_names {
                    &names[i]
                } else {
                    &i.to_string()
                };

                ax.text(positions[i].0, positions[i].1, label)
                    .set_ha("center")
                    .set_va("center")
                    .set_fontsize(8);
            }

            ax.set_title(&format!(
                "Graph Coloring: {} colors used",
                node_colors.iter().max().unwrap() + 1
            ));
            ax.set_aspect("equal");
            ax.axis("off");

            fig.show()?;
        }

        #[cfg(not(feature = "scirs"))]
        {
            // Export coloring data
            let export = GraphColoringExport {
                n_nodes,
                edges: self.extract_edges(adjacency),
                node_colors: node_colors.clone(),
                node_names: node_names.clone(),
                n_colors_used: node_colors.iter().max().copied().unwrap_or(0) + 1,
            };

            let json = serde_json::to_string_pretty(&export)?;
            std::fs::write("graph_coloring.json", json)?;
            println!("Graph coloring exported to graph_coloring.json");
        }

        Ok(())
    }

    /// Extract node colors from solution
    fn extract_node_colors(
        &self,
        sample: &SampleResult,
        n_nodes: usize,
        max_colors: usize,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        let mut colors = vec![0; n_nodes];

        for i in 0..n_nodes {
            for c in 0..max_colors {
                let var_name = format!("node_{}_color_{}", i, c);
                if sample.assignments.get(&var_name).copied().unwrap_or(false) {
                    colors[i] = c;
                    break;
                }
            }
        }

        Ok(colors)
    }

    /// Visualize max cut solution
    fn visualize_max_cut(
        &self,
        adjacency: &Array2<f64>,
        _node_names: &Option<Vec<String>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n_nodes = adjacency.nrows();
        let best_sample = self.get_best_sample()?;

        // Extract partition
        let partition = self.extract_partition(best_sample, n_nodes)?;

        #[cfg(feature = "scirs")]
        {
            use crate::scirs_stub::scirs2_plot::Figure;

            let fig = Figure::new();
            let ax = fig.add_subplot(1, 1, 1)?;

            // Compute layout with force-directed algorithm
            let positions = self.compute_graph_layout(adjacency)?;

            // Plot edges with cut edges highlighted
            let cut_weight = 0.0;
            for i in 0..n_nodes {
                for j in i + 1..n_nodes {
                    if adjacency[[i, j]] > 0.0 {
                        let is_cut = partition[i] != partition[j];
                        let color = if is_cut { "red" } else { "gray" };
                        let width = if is_cut { 3.0 } else { 1.0 };

                        if is_cut {
                            cut_weight += adjacency[[i, j]];
                        }

                        ax.plot(
                            &[positions[i].0, positions[j].0],
                            &[positions[i].1, positions[j].1],
                        )
                        .set_color(color)
                        .set_linewidth(width)
                        .set_alpha(if is_cut { 1.0 } else { 0.3 });
                    }
                }
            }

            // Plot nodes
            for i in 0..n_nodes {
                let color = if partition[i] { "blue" } else { "orange" };

                ax.scatter(&[positions[i].0], &[positions[i].1])
                    .set_color(color)
                    .set_size(self.config.node_size)
                    .set_edgecolor("black");

                // Add labels
                let label = if let Some(names) = node_names {
                    &names[i]
                } else {
                    &i.to_string()
                };

                ax.text(positions[i].0, positions[i].1, label)
                    .set_ha("center")
                    .set_va("center")
                    .set_fontsize(8);
            }

            ax.set_title(&format!("Max Cut: Weight = {:.2}", cut_weight));
            ax.set_aspect("equal");
            ax.axis("off");

            fig.show()?;
        }

        Ok(())
    }

    /// Extract partition from solution
    fn extract_partition(
        &self,
        sample: &SampleResult,
        n_nodes: usize,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        let mut partition = vec![false; n_nodes];

        for i in 0..n_nodes {
            let var_name = format!("x_{}", i);
            partition[i] = sample.assignments.get(&var_name).copied().unwrap_or(false);
        }

        Ok(partition)
    }

    /// Visualize job shop scheduling solution
    fn visualize_job_shop(
        &self,
        n_jobs: usize,
        n_machines: usize,
        time_horizon: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let best_sample = self.get_best_sample()?;

        // Extract schedule
        let schedule = self.extract_schedule(best_sample, n_jobs, n_machines, time_horizon)?;

        #[cfg(feature = "scirs")]
        {
            use crate::scirs_stub::scirs2_plot::Figure;

            let fig = Figure::new();
            let ax = fig.add_subplot(1, 1, 1)?;

            // Create Gantt chart
            let colors = ["red", "blue", "green", "yellow", "purple", "orange"];

            for (job, machine, start, duration) in &schedule {
                let y = *machine as f64;
                let color = colors[*job % colors.len()];

                ax.barh(&[y], &[*duration as f64], &[*start as f64], 0.8)
                    .set_color(color)
                    .set_edgecolor("black")
                    .set_label(&format!("Job {}", job));
            }

            ax.set_xlabel("Time");
            ax.set_ylabel("Machine");
            ax.set_title("Job Shop Schedule");
            ax.set_ylim(-0.5, n_machines as f64 - 0.5);
            ax.set_xlim(0.0, time_horizon as f64);

            // Set y-ticks
            ax.set_yticks(&(0..n_machines).map(|i| i as f64).collect::<Vec<_>>());
            ax.set_yticklabels(
                &(0..n_machines)
                    .map(|i| format!("M{}", i))
                    .collect::<Vec<_>>(),
            );

            // Remove duplicate labels in legend
            ax.legend_unique();

            fig.show()?;
        }

        Ok(())
    }

    /// Extract schedule from solution
    fn extract_schedule(
        &self,
        sample: &SampleResult,
        n_jobs: usize,
        n_machines: usize,
        time_horizon: usize,
    ) -> Result<Vec<(usize, usize, usize, usize)>, Box<dyn std::error::Error>> {
        let schedule = Vec::new();

        // This is problem-specific and would need the actual encoding scheme
        // For now, return a placeholder
        for j in 0..n_jobs {
            for m in 0..n_machines {
                for t in 0..time_horizon {
                    let var_name = format!("x_{}_{}_{}", j, m, t);
                    if sample.assignments.get(&var_name).copied().unwrap_or(false) {
                        // Find duration (would need problem-specific logic)
                        let duration = 5; // Placeholder
                        schedule.push((j, m, t, duration));
                        break;
                    }
                }
            }
        }

        Ok(schedule)
    }

    /// Visualize number partition solution
    fn visualize_number_partition(
        &self,
        numbers: &[f64],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let best_sample = self.get_best_sample()?;
        let partition = self.extract_partition(best_sample, numbers.len())?;

        #[cfg(feature = "scirs")]
        {
            use crate::scirs_stub::scirs2_plot::Figure;

            let fig = Figure::new();
            let ax = fig.add_subplot(1, 1, 1)?;

            // Separate numbers into two sets
            let set1 = Vec::new();
            let set2 = Vec::new();

            for (i, &num) in numbers.iter().enumerate() {
                if partition[i] {
                    set1.push(num);
                } else {
                    set2.push(num);
                }
            }

            let sum1: f64 = set1.iter().sum();
            let sum2: f64 = set2.iter().sum();

            // Create bar chart
            let x_pos = vec![1.0, 2.0];
            let heights = vec![sum1, sum2];
            let labels = vec!["Set 1", "Set 2"];

            ax.bar(&x_pos, &heights).set_color(&["blue", "orange"]);

            // Add value labels on bars
            for (x, h, nums) in [(1.0, sum1, &set1), (2.0, sum2, &set2)].iter() {
                ax.text(*x, *h + 0.5, &format!("{:.2}", h)).set_ha("center");

                // Show individual numbers
                let nums_str = nums
                    .iter()
                    .map(|n| format!("{:.1}", n))
                    .collect::<Vec<_>>()
                    .join(", ");
                ax.text(*x, -2.0, &nums_str)
                    .set_ha("center")
                    .set_fontsize(8);
            }

            ax.set_xticks(&x_pos);
            ax.set_xticklabels(&labels);
            ax.set_ylabel("Sum");
            ax.set_title(&format!(
                "Number Partition: |{:.2} - {:.2}| = {:.2}",
                sum1,
                sum2,
                (sum1 - sum2).abs()
            ));

            fig.show()?;
        }

        Ok(())
    }

    /// Visualize knapsack solution
    fn visualize_knapsack(
        &self,
        weights: &[f64],
        values: &[f64],
        _capacity: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let best_sample = self.get_best_sample()?;
        let n_items = weights.len();

        // Extract selected items
        let mut selected = vec![false; n_items];
        let total_weight = 0.0;
        let total_value = 0.0;

        for i in 0..n_items {
            let var_name = format!("x_{}", i);
            if best_sample
                .assignments
                .get(&var_name)
                .copied()
                .unwrap_or(false)
            {
                selected[i] = true;
                total_weight += weights[i];
                total_value += values[i];
            }
        }

        #[cfg(feature = "scirs")]
        {
            use crate::scirs_stub::scirs2_plot::Figure;

            let fig = Figure::new();

            // Item selection visualization
            let ax1 = fig.add_subplot(2, 1, 1)?;

            let x_pos: Vec<f64> = (0..n_items).map(|i| i as f64).collect();
            let colors: Vec<&str> = selected
                .iter()
                .map(|&s| if s { "green" } else { "red" })
                .collect();

            ax1.bar(&x_pos, values).set_color(&colors).set_alpha(0.7);

            // Add weight labels
            for (i, (&w, &v)) in weights.iter().zip(values.iter()).enumerate() {
                ax1.text(i as f64, v + 0.5, &format!("w={:.1}", w))
                    .set_ha("center")
                    .set_fontsize(8);
            }

            ax1.set_xlabel("Item");
            ax1.set_ylabel("Value");
            ax1.set_title(&format!(
                "Selected Items (Green): Value = {:.2}, Weight = {:.2}/{:.2}",
                total_value, total_weight, capacity
            ));

            // Capacity utilization
            let ax2 = fig.add_subplot(2, 1, 2)?;

            ax2.barh(&[1.0], &[total_weight], &[0.0], 0.5)
                .set_color("blue")
                .set_label("Used");

            ax2.barh(&[1.0], &[capacity - total_weight], &[total_weight], 0.5)
                .set_color("lightgray")
                .set_label("Remaining");

            ax2.axvline(capacity)
                .set_color("red")
                .set_linestyle("--")
                .set_label("Capacity");
            ax2.set_xlim(0.0, capacity * 1.1);
            ax2.set_ylim(0.5, 1.5);
            ax2.set_xlabel("Weight");
            ax2.set_yticks(&[]);
            ax2.legend();
            ax2.set_title("Capacity Utilization");

            fig.show()?;
        }

        Ok(())
    }

    /// Visualize portfolio optimization solution
    fn visualize_portfolio(
        &self,
        asset_names: &[String],
        expected_returns: &[f64],
        risk_matrix: &Array2<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let best_sample = self.get_best_sample()?;
        let n_assets = asset_names.len();

        // Extract portfolio weights
        let weights = self.extract_portfolio_weights(best_sample, n_assets)?;

        // Calculate portfolio metrics
        let _portfolio_return: f64 = weights
            .iter()
            .zip(expected_returns.iter())
            .map(|(w, r)| w * r)
            .sum();

        let portfolio_variance: f64 = weights
            .iter()
            .enumerate()
            .map(|(i, wi)| {
                weights
                    .iter()
                    .enumerate()
                    .map(|(j, wj)| wi * wj * risk_matrix[[i, j]])
                    .sum::<f64>()
            })
            .sum();

        let portfolio_risk = portfolio_variance.sqrt();

        #[cfg(feature = "scirs")]
        {
            use crate::scirs_stub::scirs2_plot::Figure;

            let fig = Figure::new();

            // Portfolio composition pie chart
            let ax1 = fig.add_subplot(2, 2, 1)?;

            let nonzero_weights: Vec<(String, f64)> = asset_names
                .iter()
                .zip(weights.iter())
                .filter(|(_, &w)| w > 0.01)
                .map(|(n, &w)| (n.clone(), w))
                .collect();

            if !nonzero_weights.is_empty() {
                let labels: Vec<String> = nonzero_weights.iter().map(|(n, _)| n.clone()).collect();
                let sizes: Vec<f64> = nonzero_weights.iter().map(|(_, w)| *w).collect();

                ax1.pie(&sizes, &labels).set_autopct("%1.1f%%");
                ax1.set_title("Portfolio Composition");
            }

            // Risk-return scatter
            let ax2 = fig.add_subplot(2, 2, 2)?;

            // Plot individual assets
            let risks: Vec<f64> = (0..n_assets).map(|i| risk_matrix[[i, i]].sqrt()).collect();

            ax2.scatter(&risks, expected_returns)
                .set_color("gray")
                .set_alpha(0.5)
                .set_label("Individual Assets");

            // Plot portfolio
            ax2.scatter(&[portfolio_risk], &[portfolio_return])
                .set_color("red")
                .set_size(100.0)
                .set_marker("*")
                .set_label("Portfolio");

            // Add asset labels
            for (i, name) in asset_names.iter().enumerate() {
                ax2.text(risks[i], expected_returns[i], name)
                    .set_fontsize(8)
                    .set_ha("right");
            }

            ax2.set_xlabel("Risk (Std Dev)");
            ax2.set_ylabel("Expected Return");
            ax2.set_title("Risk-Return Profile");
            ax2.legend();

            // Weight distribution
            let ax3 = fig.add_subplot(2, 2, 3)?;

            let x_pos: Vec<f64> = (0..n_assets).map(|i| i as f64).collect();
            ax3.bar(&x_pos, &weights);

            ax3.set_xticks(&x_pos);
            ax3.set_xticklabels(asset_names);
            ax3.set_xlabel("Asset");
            ax3.set_ylabel("Weight");
            ax3.set_title("Portfolio Weights");
            ax3.set_ylim(0.0, 1.0);

            for tick in ax3.get_xticklabels() {
                tick.set_rotation(45);
                tick.set_ha("right");
            }

            // Summary statistics
            let ax4 = fig.add_subplot(2, 2, 4)?;

            let summary_text = format!(
                "Portfolio Statistics\n\n\
                 Expected Return: {:.2}%\n\
                 Risk (Std Dev): {:.2}%\n\
                 Sharpe Ratio: {:.3}\n\
                 Number of Assets: {}",
                portfolio_return * 100.0,
                portfolio_risk * 100.0,
                portfolio_return / portfolio_risk,
                nonzero_weights.len()
            );

            ax4.text(0.1, 0.9, &summary_text)
                .set_fontsize(12)
                .set_verticalalignment("top")
                .set_transform(ax4.transAxes());
            ax4.axis("off");

            fig.suptitle("Portfolio Optimization Results");
            fig.tight_layout();
            fig.show()?;
        }

        Ok(())
    }

    /// Extract portfolio weights from solution
    fn extract_portfolio_weights(
        &self,
        sample: &SampleResult,
        n_assets: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let mut weights = vec![0.0; n_assets];

        // This depends on the encoding used
        // For discrete allocation, might be binary variables
        // For continuous, might need decoding from binary representation

        // Simple binary allocation example
        let total_selected = (0..n_assets)
            .filter(|&i| {
                let var_name = format!("x_{}", i);
                sample.assignments.get(&var_name).copied().unwrap_or(false)
            })
            .count();

        if total_selected > 0 {
            for i in 0..n_assets {
                let var_name = format!("x_{}", i);
                if sample.assignments.get(&var_name).copied().unwrap_or(false) {
                    weights[i] = 1.0 / total_selected as f64;
                }
            }
        }

        Ok(weights)
    }

    /// Visualize custom problem
    fn visualize_custom(
        &self,
        plot_function: &str,
        metadata: &HashMap<String, String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // This would call a user-provided plotting function
        println!(
            "Custom visualization: {} with metadata: {:?}",
            plot_function, metadata
        );
        Ok(())
    }

    /// Get best sample
    fn get_best_sample(&self) -> Result<&SampleResult, Box<dyn std::error::Error>> {
        self.samples
            .iter()
            .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
            .ok_or("No samples available".into())
    }

    /// Get best k samples
    fn get_best_samples(&self) -> Vec<&SampleResult> {
        let mut sorted_samples: Vec<_> = self.samples.iter().collect();
        sorted_samples.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());
        sorted_samples
    }

    /// Compute graph layout
    fn compute_graph_layout(
        &self,
        adjacency: &Array2<f64>,
    ) -> Result<Vec<(f64, f64)>, Box<dyn std::error::Error>> {
        let n = adjacency.nrows();

        // Simple circular layout as fallback
        let positions = Vec::new();
        for i in 0..n {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            positions.push((angle.cos(), angle.sin()));
        }

        Ok(positions)
    }

    /// Extract edges from adjacency matrix
    fn extract_edges(&self, adjacency: &Array2<bool>) -> Vec<(usize, usize)> {
        let edges = Vec::new();
        let n = adjacency.nrows();

        for i in 0..n {
            for j in i + 1..n {
                if adjacency[[i, j]] {
                    edges.push((i, j));
                }
            }
        }

        edges
    }
}

// Export structures for non-SciRS builds

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TSPExport {
    coordinates: Vec<(f64, f64)>,
    city_names: Option<Vec<String>>,
    best_tours: Vec<Vec<usize>>,
    tour_lengths: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphColoringExport {
    n_nodes: usize,
    edges: Vec<(usize, usize)>,
    node_colors: Vec<usize>,
    node_names: Option<Vec<String>>,
    n_colors_used: usize,
}
