//! Visualization functionality for the solution debugger.

use super::types::{ProblemInfo, Solution};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Solution visualizer
pub struct SolutionVisualizer {
    /// Visualization options
    options: VisualizationOptions,
    /// Color schemes
    color_schemes: HashMap<String, ColorScheme>,
}

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
pub struct ColorScheme {
    /// Variable colors
    pub variable_colors: HashMap<bool, String>,
    /// Constraint colors
    pub constraint_colors: HashMap<String, String>,
    /// Energy gradient
    pub energy_gradient: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Visualization {
    /// Visualization type
    pub viz_type: VisualizationType,
    /// Title
    pub title: String,
    /// Data for visualization
    pub data: VisualizationData,
    /// Rendering options
    pub options: VisualizationOptions,
}

#[derive(Debug, Clone, Serialize)]
pub enum VisualizationType {
    /// Solution matrix
    SolutionMatrix,
    /// Energy landscape
    EnergyLandscape,
    /// Constraint graph
    ConstraintGraph,
    /// Variable interaction graph
    InteractionGraph,
    /// Energy breakdown chart
    EnergyBreakdown,
}

#[derive(Debug, Clone, Serialize)]
pub struct VisualizationData {
    /// Nodes (variables, constraints, etc.)
    pub nodes: Vec<Node>,
    /// Edges (relationships, interactions)
    pub edges: Vec<Edge>,
    /// Additional data
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Node {
    /// Node ID
    pub id: String,
    /// Node label
    pub label: String,
    /// Node type
    pub node_type: String,
    /// Position (x, y)
    pub position: Option<(f64, f64)>,
    /// Size
    pub size: f64,
    /// Color
    pub color: String,
    /// Additional properties
    pub properties: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Edge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Edge type
    pub edge_type: String,
    /// Weight/strength
    pub weight: f64,
    /// Color
    pub color: String,
    /// Additional properties
    pub properties: HashMap<String, serde_json::Value>,
}

impl SolutionVisualizer {
    /// Create new solution visualizer
    pub fn new() -> Self {
        let mut color_schemes = HashMap::new();

        // Default color scheme
        let mut default_scheme = ColorScheme {
            variable_colors: HashMap::new(),
            constraint_colors: HashMap::new(),
            energy_gradient: vec![
                "#FF0000".to_string(), // High energy (red)
                "#FFFF00".to_string(), // Medium energy (yellow)
                "#00FF00".to_string(), // Low energy (green)
            ],
        };
        default_scheme
            .variable_colors
            .insert(true, "#0066CC".to_string());
        default_scheme
            .variable_colors
            .insert(false, "#CCCCCC".to_string());
        color_schemes.insert("default".to_string(), default_scheme);

        Self {
            options: VisualizationOptions::default(),
            color_schemes,
        }
    }

    /// Visualize solution matrix
    pub fn visualize_solution_matrix(
        &self,
        solution: &Solution,
        problem_info: &ProblemInfo,
    ) -> Visualization {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Create nodes for each variable
        for (var, &value) in &solution.assignments {
            let color = if value {
                self.color_schemes["default"].variable_colors[&true].clone()
            } else {
                self.color_schemes["default"].variable_colors[&false].clone()
            };

            nodes.push(Node {
                id: var.clone(),
                label: format!("{}: {}", var, if value { "1" } else { "0" }),
                node_type: "variable".to_string(),
                position: None,
                size: 1.0,
                color,
                properties: HashMap::new(),
            });
        }

        // Add edges based on QUBO interactions
        for i in 0..problem_info.qubo.nrows() {
            for j in i + 1..problem_info.qubo.ncols() {
                if problem_info.qubo[[i, j]].abs() > 1e-10 {
                    if let (Some(var1), Some(var2)) = (
                        problem_info.reverse_var_map.get(&i),
                        problem_info.reverse_var_map.get(&j),
                    ) {
                        edges.push(Edge {
                            source: var1.clone(),
                            target: var2.clone(),
                            edge_type: "interaction".to_string(),
                            weight: problem_info.qubo[[i, j]].abs(),
                            color: if problem_info.qubo[[i, j]] > 0.0 {
                                "#FF0000".to_string()
                            } else {
                                "#0000FF".to_string()
                            },
                            properties: HashMap::new(),
                        });
                    }
                }
            }
        }

        Visualization {
            viz_type: VisualizationType::SolutionMatrix,
            title: "Solution Variable Matrix".to_string(),
            data: VisualizationData {
                nodes,
                edges,
                metadata: HashMap::new(),
            },
            options: self.options.clone(),
        }
    }

    /// Visualize energy landscape
    pub fn visualize_energy_landscape(
        &self,
        solution: &Solution,
        problem_info: &ProblemInfo,
    ) -> Visualization {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Create node for current solution
        nodes.push(Node {
            id: "current".to_string(),
            label: format!("Current Solution (E={:.2})", solution.energy),
            node_type: "solution".to_string(),
            position: Some((0.0, 0.0)),
            size: 2.0,
            color: "#FF0000".to_string(),
            properties: HashMap::new(),
        });

        // Would add neighboring solutions and their energies
        // This is a placeholder implementation

        Visualization {
            viz_type: VisualizationType::EnergyLandscape,
            title: "Energy Landscape".to_string(),
            data: VisualizationData {
                nodes,
                edges,
                metadata: HashMap::new(),
            },
            options: self.options.clone(),
        }
    }

    /// Visualize constraint graph
    pub fn visualize_constraint_graph(
        &self,
        solution: &Solution,
        problem_info: &ProblemInfo,
    ) -> Visualization {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Create nodes for variables
        for (var, &value) in &solution.assignments {
            let color = if value {
                self.color_schemes["default"].variable_colors[&true].clone()
            } else {
                self.color_schemes["default"].variable_colors[&false].clone()
            };

            nodes.push(Node {
                id: var.clone(),
                label: var.clone(),
                node_type: "variable".to_string(),
                position: None,
                size: 1.0,
                color,
                properties: HashMap::new(),
            });
        }

        // Create nodes for constraints
        for (i, constraint) in problem_info.constraints.iter().enumerate() {
            let constraint_id = format!("constraint_{}", i);
            nodes.push(Node {
                id: constraint_id.clone(),
                label: constraint.name.as_ref().unwrap_or(&constraint_id).clone(),
                node_type: "constraint".to_string(),
                position: None,
                size: 1.5,
                color: "#FFAA00".to_string(),
                properties: HashMap::new(),
            });

            // Connect constraint to its variables
            for var in &constraint.variables {
                edges.push(Edge {
                    source: constraint_id.clone(),
                    target: var.clone(),
                    edge_type: "constrains".to_string(),
                    weight: 1.0,
                    color: "#888888".to_string(),
                    properties: HashMap::new(),
                });
            }
        }

        Visualization {
            viz_type: VisualizationType::ConstraintGraph,
            title: "Constraint Graph".to_string(),
            data: VisualizationData {
                nodes,
                edges,
                metadata: HashMap::new(),
            },
            options: self.options.clone(),
        }
    }
}

impl Default for VisualizationOptions {
    fn default() -> Self {
        Self {
            show_values: true,
            show_energy: true,
            show_violations: true,
            show_relationships: true,
            layout: LayoutAlgorithm::ForceDirected,
        }
    }
}

impl Default for SolutionVisualizer {
    fn default() -> Self {
        Self::new()
    }
}
