//! Advanced visualization for quantum annealing results
//!
//! This module provides comprehensive visualization capabilities for
//! QUBO/HOBO problems, optimization results, and solution analysis.

pub mod convergence;
pub mod energy_landscape;
pub mod export;
pub mod problem_specific;
pub mod solution_analysis;

pub use convergence::{plot_convergence, ConvergencePlot};
pub use energy_landscape::{plot_energy_landscape, EnergyLandscape};
pub use export::{export_visualization, ExportFormat};
pub use problem_specific::{ProblemVisualizer, VisualizationType};
pub use solution_analysis::{analyze_solution_distribution, SolutionDistribution};

/// Prelude for common visualization imports
pub mod prelude {
    pub use super::{
        analyze_solution_distribution, plot_convergence, plot_energy_landscape, ExportFormat,
    };
}
