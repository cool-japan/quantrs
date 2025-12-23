//! Energy landscape visualization for QUBO/HOBO problems
//!
//! This module provides tools for visualizing the energy landscape
//! of optimization problems including 2D/3D projections and heatmaps.

use crate::sampler::SampleResult;
use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};

#[cfg(feature = "scirs")]
use crate::scirs_stub::{
    scirs2_linalg::svd::SVD,
    scirs2_plot::{ColorMap, Heatmap, Plot2D, Plot3D},
    scirs2_statistics::kde::KernelDensityEstimator,
};

/// Energy landscape configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyLandscapeConfig {
    /// Number of bins for histograms
    pub bins: usize,
    /// Projection method for high-dimensional data
    pub projection: ProjectionMethod,
    /// Color map for visualization
    pub colormap: String,
    /// Include density estimation
    pub include_density: bool,
    /// Resolution for heatmap
    pub resolution: usize,
    /// Energy range limits
    pub energy_limits: Option<(f64, f64)>,
}

/// Projection methods for dimensionality reduction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionMethod {
    /// Principal Component Analysis
    PCA,
    /// Random projection
    Random,
    /// Hamming distance from best solution
    HammingDistance,
    /// Custom projection matrix
    Custom,
}

impl Default for EnergyLandscapeConfig {
    fn default() -> Self {
        Self {
            bins: 50,
            projection: ProjectionMethod::PCA,
            colormap: "viridis".to_string(),
            include_density: true,
            resolution: 100,
            energy_limits: None,
        }
    }
}

/// Energy landscape analyzer
pub struct EnergyLandscape {
    config: EnergyLandscapeConfig,
    samples: Vec<SampleResult>,
    projection_matrix: Option<Array2<f64>>,
}

impl EnergyLandscape {
    /// Create new energy landscape analyzer
    pub const fn new(config: EnergyLandscapeConfig) -> Self {
        Self {
            config,
            samples: Vec::new(),
            projection_matrix: None,
        }
    }

    /// Add samples for analysis
    pub fn add_samples(&mut self, samples: Vec<SampleResult>) {
        self.samples.extend(samples);
    }

    /// Generate 1D energy histogram
    pub fn energy_histogram(&self) -> Result<HistogramData, Box<dyn std::error::Error>> {
        let energies: Vec<f64> = self.samples.iter().map(|s| s.energy).collect();

        let (min_energy, max_energy) = if let Some(limits) = self.config.energy_limits {
            limits
        } else {
            let min = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = energies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            (min, max)
        };

        let bin_width = (max_energy - min_energy) / self.config.bins as f64;
        let mut bins = vec![0; self.config.bins];
        let mut bin_centers = Vec::new();

        for i in 0..self.config.bins {
            bin_centers.push((i as f64 + 0.5).mul_add(bin_width, min_energy));
        }

        for &energy in &energies {
            if energy >= min_energy && energy <= max_energy {
                let bin_idx = ((energy - min_energy) / bin_width).floor() as usize;
                let bin_idx = bin_idx.min(self.config.bins - 1);
                bins[bin_idx] += 1;
            }
        }

        Ok(HistogramData {
            bin_centers,
            counts: bins,
            bin_width,
            total_samples: energies.len(),
        })
    }

    /// Generate 2D projected energy landscape
    pub fn project_2d(&mut self) -> Result<ProjectedLandscape, Box<dyn std::error::Error>> {
        // Convert samples to binary matrix
        let (binary_matrix, _var_names) = self.samples_to_matrix()?;

        // Project to 2D
        let projected = match self.config.projection {
            ProjectionMethod::PCA => self.project_pca(&binary_matrix, 2)?,
            ProjectionMethod::Random => self.project_random(&binary_matrix, 2)?,
            ProjectionMethod::HammingDistance => self.project_hamming(&binary_matrix)?,
            ProjectionMethod::Custom => {
                if let Some(ref proj_mat) = self.projection_matrix {
                    binary_matrix.dot(proj_mat)
                } else {
                    return Err("Custom projection matrix not set".into());
                }
            }
        };

        // Extract coordinates and energies
        let x_coords: Vec<f64> = (0..projected.nrows()).map(|i| projected[[i, 0]]).collect();
        let y_coords: Vec<f64> = (0..projected.nrows()).map(|i| projected[[i, 1]]).collect();
        let energies: Vec<f64> = self.samples.iter().map(|s| s.energy).collect();

        // Generate density estimation if requested
        let density_map = if self.config.include_density {
            #[cfg(feature = "scirs")]
            {
                Some(self.estimate_density_2d(&x_coords, &y_coords, &energies)?)
            }
            #[cfg(not(feature = "scirs"))]
            None
        } else {
            None
        };

        Ok(ProjectedLandscape {
            x_coords,
            y_coords,
            energies,
            density_map,
            projection_info: format!("{:?} projection", self.config.projection),
        })
    }

    /// Convert samples to binary matrix
    fn samples_to_matrix(&self) -> Result<(Array2<f64>, Vec<String>), Box<dyn std::error::Error>> {
        if self.samples.is_empty() {
            return Err("No samples to analyze".into());
        }

        // Get all variable names
        let mut all_vars = std::collections::HashSet::new();
        for sample in &self.samples {
            for var in sample.assignments.keys() {
                all_vars.insert(var.clone());
            }
        }

        let var_names: Vec<String> = all_vars.into_iter().collect();
        let n_vars = var_names.len();
        let n_samples = self.samples.len();

        // Create binary matrix
        let mut matrix = Array2::zeros((n_samples, n_vars));

        for (i, sample) in self.samples.iter().enumerate() {
            for (j, var_name) in var_names.iter().enumerate() {
                if let Some(&value) = sample.assignments.get(var_name) {
                    matrix[[i, j]] = if value { 1.0 } else { 0.0 };
                }
            }
        }

        Ok((matrix, var_names))
    }

    /// PCA projection
    fn project_pca(
        &self,
        data: &Array2<f64>,
        n_components: usize,
    ) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        #[cfg(feature = "scirs")]
        {
            use crate::scirs_stub::scirs2_linalg::pca::PCA;

            let pca = PCA::new(n_components);
            let projected = pca.fit_transform(data)?;
            Ok(projected)
        }

        #[cfg(not(feature = "scirs"))]
        {
            // Simple centering and truncation
            let n_samples = data.nrows();
            let n_features = n_components.min(data.ncols());
            let mut result = Array2::zeros((n_samples, n_components));

            // Center the data
            let means = data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
            for i in 0..n_samples {
                for j in 0..n_features {
                    result[[i, j]] = data[[i, j]] - means[j];
                }
            }

            Ok(result)
        }
    }

    /// Random projection
    fn project_random(
        &self,
        data: &Array2<f64>,
        n_components: usize,
    ) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        use scirs2_core::random::prelude::*;

        let n_features = data.ncols();
        let mut rng = StdRng::seed_from_u64(42);

        // Generate random projection matrix
        let mut proj_matrix = Array2::<f64>::zeros((n_features, n_components));
        for i in 0..n_features {
            for j in 0..n_components {
                proj_matrix[[i, j]] = rng.gen_range(-1.0..1.0);
            }
        }

        // Normalize columns
        for j in 0..n_components {
            let col_norm = (0..n_features)
                .map(|i| proj_matrix[[i, j]].powi(2))
                .sum::<f64>()
                .sqrt();

            if col_norm > 0.0 {
                for i in 0..n_features {
                    proj_matrix[[i, j]] /= col_norm;
                }
            }
        }

        Ok(data.dot(&proj_matrix))
    }

    /// Hamming distance projection
    fn project_hamming(
        &self,
        data: &Array2<f64>,
    ) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        let n_samples = data.nrows();

        // Find best solution
        let best_idx = self
            .samples
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.energy.partial_cmp(&b.energy).unwrap())
            .map_or(0, |(i, _)| i);

        let best_solution = data.row(best_idx);

        // Calculate Hamming distances
        let mut result = Array2::zeros((n_samples, 2));

        for i in 0..n_samples {
            let hamming_dist: f64 = data
                .row(i)
                .iter()
                .zip(best_solution.iter())
                .filter(|(&a, &b)| (a - b).abs() > 0.5)
                .count() as f64;

            result[[i, 0]] = hamming_dist;
            result[[i, 1]] = self.samples[i].energy;
        }

        Ok(result)
    }

    #[cfg(feature = "scirs")]
    /// Estimate density in 2D
    fn estimate_density_2d(
        &self,
        x: &[f64],
        y: &[f64],
        energies: &[f64],
    ) -> Result<DensityMap, Box<dyn std::error::Error>> {
        let kde = KernelDensityEstimator::new("gaussian")?;

        // Create grid
        let x_min = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        let mut grid_x = Vec::new();
        let mut grid_y = Vec::new();
        let mut grid_density = Vec::new();

        for i in 0..self.config.resolution {
            for j in 0..self.config.resolution {
                let gx = (i as f64 / self.config.resolution as f64).mul_add(x_range, x_min);
                let gy = (j as f64 / self.config.resolution as f64).mul_add(y_range, y_min);

                grid_x.push(gx);
                grid_y.push(gy);

                // Estimate density at this point
                let density = kde.estimate_2d(x, y, gx, gy)?;
                grid_density.push(density);
            }
        }

        Ok(DensityMap {
            x: grid_x,
            y: grid_y,
            density: grid_density,
            resolution: self.config.resolution,
        })
    }

    /// Set custom projection matrix
    pub fn set_projection_matrix(&mut self, matrix: Array2<f64>) {
        self.projection_matrix = Some(matrix);
    }
}

/// Histogram data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramData {
    pub bin_centers: Vec<f64>,
    pub counts: Vec<usize>,
    pub bin_width: f64,
    pub total_samples: usize,
}

/// Projected landscape data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectedLandscape {
    pub x_coords: Vec<f64>,
    pub y_coords: Vec<f64>,
    pub energies: Vec<f64>,
    pub density_map: Option<DensityMap>,
    pub projection_info: String,
}

/// Density map data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityMap {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub density: Vec<f64>,
    pub resolution: usize,
}

/// Plot energy landscape
pub fn plot_energy_landscape(
    samples: Vec<SampleResult>,
    config: Option<EnergyLandscapeConfig>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = config.unwrap_or_default();
    let mut landscape = EnergyLandscape::new(config);
    landscape.add_samples(samples);

    #[cfg(feature = "scirs")]
    {
        use crate::scirs_stub::scirs2_plot::{Figure, Subplot};

        let mut fig = Figure::new();

        // Add energy histogram
        let hist_data = landscape.energy_histogram()?;
        let counts_f64: Vec<f64> = hist_data.counts.iter().map(|&c| c as f64).collect();
        fig.add_subplot(2, 2, 1)?
            .bar(&hist_data.bin_centers, &counts_f64)
            .set_xlabel("Energy")
            .set_ylabel("Count")
            .set_title("Energy Distribution");

        // Add 2D projection
        let proj_data = landscape.project_2d()?;
        let subplot = fig.add_subplot(2, 2, 2)?;
        subplot
            .scatter(&proj_data.x_coords, &proj_data.y_coords)
            .set_color_data(&proj_data.energies)
            .set_colormap(&landscape.config.colormap)
            .set_xlabel("Component 1")
            .set_ylabel("Component 2")
            .set_title(&proj_data.projection_info);

        // Add density plot if available
        if let Some(density) = proj_data.density_map {
            fig.add_subplot(2, 2, 3)?
                .contourf(&density.x, &density.y, &density.density)
                .set_colormap("plasma")
                .set_xlabel("Component 1")
                .set_ylabel("Component 2")
                .set_title("Solution Density");
        }

        fig.show()?;
    }

    #[cfg(not(feature = "scirs"))]
    {
        // Export data for external plotting
        let hist_data = landscape.energy_histogram()?;
        let proj_data = landscape.project_2d()?;

        export_landscape_data(&hist_data, &proj_data, "energy_landscape.json")?;
        println!("Energy landscape data exported to energy_landscape.json");
    }

    Ok(())
}

/// Export landscape data
fn export_landscape_data(
    histogram: &HistogramData,
    projection: &ProjectedLandscape,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let export = LandscapeExport {
        histogram: histogram.clone(),
        projection: projection.clone(),
        timestamp: std::time::SystemTime::now(),
    };

    let json = serde_json::to_string_pretty(&export)?;
    std::fs::write(path, json)?;

    Ok(())
}

/// Landscape export format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandscapeExport {
    pub histogram: HistogramData,
    pub projection: ProjectedLandscape,
    pub timestamp: std::time::SystemTime,
}
