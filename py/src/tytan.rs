//! Python bindings for Tytan quantum annealing and visualization

// Allow type_complexity for PyO3 return types with complex nested generics
#![allow(clippy::type_complexity)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use std::collections::HashMap;

#[cfg(feature = "tytan")]
use quantrs2_tytan::{
    analysis::visualization::{
        analyze_convergence, analyze_solution_distribution, export_solution_matrix, export_to_csv,
        extract_graph_coloring, extract_tsp_tour, prepare_energy_landscape, spring_layout,
        EnergyLandscapeConfig, SolutionDistributionConfig,
    },
    sampler::{SASampler, SampleResult, Sampler},
};

/// Python wrapper for sample result
#[pyclass]
#[derive(Clone)]
pub struct PySampleResult {
    #[pyo3(get)]
    pub assignments: HashMap<String, bool>,
    #[pyo3(get)]
    pub energy: f64,
    #[pyo3(get)]
    pub occurrences: usize,
}

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PySampleResult {
    #[new]
    pub fn new(assignments: HashMap<String, bool>, energy: f64, occurrences: usize) -> Self {
        Self {
            assignments,
            energy,
            occurrences,
        }
    }
}

#[cfg(feature = "tytan")]
impl From<PySampleResult> for SampleResult {
    fn from(py_result: PySampleResult) -> Self {
        Self {
            assignments: py_result.assignments,
            energy: py_result.energy,
            occurrences: py_result.occurrences,
        }
    }
}

/// Python wrapper for energy landscape visualization
#[pyclass]
pub struct PyEnergyLandscapeVisualizer;

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PyEnergyLandscapeVisualizer {
    /// Prepare energy landscape data
    #[staticmethod]
    #[pyo3(signature = (results, num_bins=None, compute_kde=None, kde_points=None))]
    fn prepare_landscape(
        py: Python,
        results: Vec<PySampleResult>,
        num_bins: Option<usize>,
        compute_kde: Option<bool>,
        kde_points: Option<usize>,
    ) -> PyResult<Py<PyDict>> {
        #[cfg(feature = "tytan")]
        {
            let config = EnergyLandscapeConfig {
                num_bins: num_bins.unwrap_or(50),
                compute_kde: compute_kde.unwrap_or(true),
                kde_points: kde_points.unwrap_or(200),
            };

            let rust_results: Vec<SampleResult> =
                results.into_iter().map(std::convert::Into::into).collect();

            let landscape_data = prepare_energy_landscape(&rust_results, Some(config))
                .map_err(|e| PyValueError::new_err(format!("Failed to prepare landscape: {e}")))?;

            let dict = PyDict::new(py);

            // Convert indices to numpy array
            let indices_array = landscape_data.indices.into_pyarray(py);
            dict.set_item("indices", indices_array)?;

            // Convert energies to numpy array
            let energies_array = landscape_data.energies.into_pyarray(py);
            dict.set_item("energies", energies_array)?;

            // Convert histogram data
            let bins_array = landscape_data.histogram_bins.into_pyarray(py);
            dict.set_item("histogram_bins", bins_array)?;

            let counts_array = landscape_data.histogram_counts.into_pyarray(py);
            dict.set_item("histogram_counts", counts_array)?;

            // Convert KDE data if present
            if let Some(kde_x) = landscape_data.kde_x {
                let kde_x_array = kde_x.into_pyarray(py);
                dict.set_item("kde_x", kde_x_array)?;
            }

            if let Some(kde_y) = landscape_data.kde_y {
                let kde_y_array = kde_y.into_pyarray(py);
                dict.set_item("kde_y", kde_y_array)?;
            }

            Ok(dict.into())
        }

        #[cfg(not(feature = "tytan"))]
        {
            Err(PyValueError::new_err(
                "Tytan features not enabled. Install with 'pip install quantrs2[tytan]'",
            ))
        }
    }

    /// Export landscape data to CSV
    #[staticmethod]
    fn export_csv(data: &Bound<'_, PyDict>, output_path: &str) -> PyResult<()> {
        #[cfg(feature = "tytan")]
        {
            // Note: This is a simplified version - in practice we'd reconstruct the data structure
            Err(PyValueError::new_err(
                "Use pandas DataFrame.to_csv() on the returned data",
            ))
        }

        #[cfg(not(feature = "tytan"))]
        {
            Err(PyValueError::new_err("Tytan features not enabled"))
        }
    }
}

/// Python wrapper for solution distribution analysis
#[pyclass]
pub struct PySolutionAnalyzer;

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PySolutionAnalyzer {
    /// Analyze solution distributions
    #[staticmethod]
    #[pyo3(signature = (results, compute_correlations=None, compute_pca=None, n_components=None))]
    fn analyze_distribution(
        py: Python,
        results: Vec<PySampleResult>,
        compute_correlations: Option<bool>,
        compute_pca: Option<bool>,
        n_components: Option<usize>,
    ) -> PyResult<Py<PyDict>> {
        #[cfg(feature = "tytan")]
        {
            let config = SolutionDistributionConfig {
                compute_correlations: compute_correlations.unwrap_or(true),
                compute_pca: compute_pca.unwrap_or(true),
                n_components: n_components.unwrap_or(2),
            };

            let rust_results: Vec<SampleResult> =
                results.into_iter().map(std::convert::Into::into).collect();

            let dist_data =
                analyze_solution_distribution(&rust_results, Some(config)).map_err(|e| {
                    PyValueError::new_err(format!("Failed to analyze distribution: {e}"))
                })?;

            let dict = PyDict::new(py);

            // Variable names
            dict.set_item("variable_names", dist_data.variable_names)?;

            // Variable frequencies
            dict.set_item("variable_frequencies", dist_data.variable_frequencies)?;

            // Correlations
            if let Some(correlations) = dist_data.correlations {
                let corr_dict = PyDict::new(py);
                for ((var1, var2), value) in correlations {
                    let key = format!("{var1}_{var2}");
                    corr_dict.set_item(key, value)?;
                }
                dict.set_item("correlations", corr_dict)?;
            }

            // Solution matrix
            let (rows, cols) = dist_data.solution_matrix.dim();
            let flat_data: Vec<f64> = dist_data.solution_matrix.into_raw_vec();
            let array = scirs2_core::ndarray::Array2::from_shape_vec((rows, cols), flat_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to create array: {e}")))?;
            let py_array = array.into_pyarray(py);
            dict.set_item("solution_matrix", py_array)?;

            // PCA components
            if let Some(components) = dist_data.pca_components {
                let (rows, cols) = components.dim();
                let flat_data: Vec<f64> = components.into_raw_vec();
                let array = scirs2_core::ndarray::Array2::from_shape_vec((rows, cols), flat_data)
                    .map_err(|e| {
                    PyValueError::new_err(format!("Failed to create array: {e}"))
                })?;
                let py_array = array.into_pyarray(py);
                dict.set_item("pca_components", py_array)?;
            }

            // PCA explained variance
            if let Some(variance) = dist_data.pca_explained_variance {
                let array = variance.into_pyarray(py);
                dict.set_item("pca_explained_variance", array)?;
            }

            Ok(dict.into())
        }

        #[cfg(not(feature = "tytan"))]
        {
            Err(PyValueError::new_err("Tytan features not enabled"))
        }
    }
}

/// Python wrapper for problem-specific visualizations
#[pyclass]
pub struct PyProblemVisualizer;

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PyProblemVisualizer {
    /// Extract TSP tour from solution
    #[staticmethod]
    fn extract_tsp_tour(result: &PySampleResult, n_cities: usize) -> PyResult<Vec<usize>> {
        #[cfg(feature = "tytan")]
        {
            let rust_result: SampleResult = result.clone().into();
            extract_tsp_tour(&rust_result, n_cities)
                .map_err(|e| PyValueError::new_err(format!("Failed to extract tour: {e}")))
        }

        #[cfg(not(feature = "tytan"))]
        {
            Err(PyValueError::new_err("Tytan features not enabled"))
        }
    }

    /// Extract graph coloring from solution
    #[staticmethod]
    #[allow(clippy::needless_pass_by_value)]
    fn extract_graph_coloring(
        result: &PySampleResult,
        n_nodes: usize,
        n_colors: usize,
        edges: Vec<(usize, usize)>,
    ) -> PyResult<(Vec<usize>, Vec<(usize, usize)>)> {
        #[cfg(feature = "tytan")]
        {
            let rust_result: SampleResult = result.clone().into();
            extract_graph_coloring(&rust_result, n_nodes, n_colors, &edges)
                .map_err(|e| PyValueError::new_err(format!("Failed to extract coloring: {e}")))
        }

        #[cfg(not(feature = "tytan"))]
        {
            Err(PyValueError::new_err("Tytan features not enabled"))
        }
    }

    /// Generate spring layout for graph visualization
    #[staticmethod]
    #[allow(clippy::needless_pass_by_value)]
    fn spring_layout(
        py: Python,
        n_nodes: usize,
        edges: Vec<(usize, usize)>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        #[cfg(feature = "tytan")]
        {
            let positions = spring_layout(n_nodes, &edges);

            // Convert to numpy array
            let flat_coords: Vec<f64> = positions
                .into_iter()
                .flat_map(|(x, y)| vec![x, y])
                .collect();

            let array = scirs2_core::ndarray::Array2::from_shape_vec((n_nodes, 2), flat_coords)
                .map_err(|e| PyValueError::new_err(format!("Failed to create array: {e}")))?;

            Ok(array.into_pyarray(py).into())
        }

        #[cfg(not(feature = "tytan"))]
        {
            Err(PyValueError::new_err("Tytan features not enabled"))
        }
    }
}

/// Python wrapper for convergence analysis
#[pyclass]
pub struct PyConvergenceAnalyzer;

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PyConvergenceAnalyzer {
    /// Analyze convergence behavior
    #[staticmethod]
    #[pyo3(signature = (iteration_results, ma_window=None))]
    fn analyze_convergence(
        py: Python,
        iteration_results: Vec<Vec<PySampleResult>>,
        ma_window: Option<usize>,
    ) -> PyResult<Py<PyDict>> {
        #[cfg(feature = "tytan")]
        {
            let rust_iterations: Vec<Vec<SampleResult>> = iteration_results
                .into_iter()
                .map(|iter| iter.into_iter().map(std::convert::Into::into).collect())
                .collect();

            let conv_data = analyze_convergence(&rust_iterations, ma_window).map_err(|e| {
                PyValueError::new_err(format!("Failed to analyze convergence: {e}"))
            })?;

            let dict = PyDict::new(py);

            // Iterations
            let iter_array = conv_data.iterations.into_pyarray(py);
            dict.set_item("iterations", iter_array)?;

            // Best energies
            let best_array = conv_data.best_energies.into_pyarray(py);
            dict.set_item("best_energies", best_array)?;

            // Average energies
            let avg_array = conv_data.avg_energies.into_pyarray(py);
            dict.set_item("avg_energies", avg_array)?;

            // Standard deviations
            let std_array = conv_data.std_devs.into_pyarray(py);
            dict.set_item("std_devs", std_array)?;

            // Moving averages
            if let Some(ma_best) = conv_data.ma_best {
                let ma_best_array = ma_best.into_pyarray(py);
                dict.set_item("ma_best", ma_best_array)?;
            }

            if let Some(ma_avg) = conv_data.ma_avg {
                let ma_avg_array = ma_avg.into_pyarray(py);
                dict.set_item("ma_avg", ma_avg_array)?;
            }

            Ok(dict.into())
        }

        #[cfg(not(feature = "tytan"))]
        {
            Err(PyValueError::new_err("Tytan features not enabled"))
        }
    }
}

/// Register the tytan module
pub fn register_tytan_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent_module.py(), "tytan")?;

    m.add_class::<PySampleResult>()?;
    m.add_class::<PyEnergyLandscapeVisualizer>()?;
    m.add_class::<PySolutionAnalyzer>()?;
    m.add_class::<PyProblemVisualizer>()?;
    m.add_class::<PyConvergenceAnalyzer>()?;

    parent_module.add_submodule(&m)?;
    Ok(())
}
