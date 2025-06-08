//! Advanced visualization for quantum annealing results
//!
//! This module provides comprehensive visualization capabilities for
//! QUBO/HOBO problems, optimization results, and solution analysis.

pub mod energy_landscape;
pub mod solution_analysis;
pub mod convergence;
pub mod problem_specific;
pub mod export;

pub use energy_landscape::{EnergyLandscape, plot_energy_landscape};
pub use solution_analysis::{SolutionDistribution, analyze_solution_distribution};
pub use convergence::{ConvergencePlot, plot_convergence};
pub use problem_specific::{ProblemVisualizer, VisualizationType};
pub use export::{ExportFormat, export_visualization};

/// Prelude for common visualization imports
pub mod prelude {
    pub use super::{
        plot_energy_landscape,
        analyze_solution_distribution,
        plot_convergence,
        ExportFormat,
    };
}