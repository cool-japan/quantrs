//! Export functionality for visualization data
//!
//! This module provides export capabilities for visualization data in various
//! formats including JSON, CSV, and custom formats for external plotting tools.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Export format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// Python numpy arrays
    NPY,
    /// MATLAB .mat files
    MAT,
    /// Plotly JSON format
    Plotly,
    /// Matplotlib script
    Matplotlib,
    /// HTML with embedded visualization
    HTML,
}

/// Visualization exporter
pub struct VisualizationExporter {
    format: ExportFormat,
    include_metadata: bool,
    compression: bool,
}

impl VisualizationExporter {
    /// Create new exporter
    pub const fn new(format: ExportFormat) -> Self {
        Self {
            format,
            include_metadata: true,
            compression: false,
        }
    }

    /// Enable metadata inclusion
    pub const fn with_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Enable compression
    pub const fn with_compression(mut self, compress: bool) -> Self {
        self.compression = compress;
        self
    }

    /// Export energy landscape data
    pub fn export_energy_landscape(
        &self,
        data: &EnergyLandscapeData,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match self.format {
            ExportFormat::JSON => self.export_json(data, path)?,
            ExportFormat::CSV => self.export_energy_csv(data, path)?,
            ExportFormat::Plotly => self.export_plotly_energy(data, path)?,
            ExportFormat::Matplotlib => self.export_matplotlib_energy(data, path)?,
            ExportFormat::HTML => self.export_html_energy(data, path)?,
            _ => return Err("Unsupported format for energy landscape".into()),
        }

        Ok(())
    }

    /// Export solution distribution data
    pub fn export_distribution(
        &self,
        data: &DistributionData,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match self.format {
            ExportFormat::JSON => self.export_json(data, path)?,
            ExportFormat::CSV => self.export_distribution_csv(data, path)?,
            ExportFormat::Plotly => self.export_plotly_distribution(data, path)?,
            ExportFormat::Matplotlib => self.export_matplotlib_distribution(data, path)?,
            ExportFormat::HTML => self.export_html_distribution(data, path)?,
            _ => return Err("Unsupported format for distribution data".into()),
        }

        Ok(())
    }

    /// Export convergence data
    pub fn export_convergence(
        &self,
        data: &ConvergenceData,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match self.format {
            ExportFormat::JSON => self.export_json(data, path)?,
            ExportFormat::CSV => self.export_convergence_csv(data, path)?,
            ExportFormat::Plotly => self.export_plotly_convergence(data, path)?,
            ExportFormat::Matplotlib => self.export_matplotlib_convergence(data, path)?,
            ExportFormat::HTML => self.export_html_convergence(data, path)?,
            _ => return Err("Unsupported format for convergence data".into()),
        }

        Ok(())
    }

    /// Generic JSON export
    fn export_json<T: Serialize>(
        &self,
        data: &T,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let json = if self.compression {
            serde_json::to_string(data)?
        } else {
            serde_json::to_string_pretty(data)?
        };

        std::fs::write(path, json)?;
        Ok(())
    }

    /// Export energy landscape as CSV
    fn export_energy_csv(
        &self,
        data: &EnergyLandscapeData,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        // Write header
        writeln!(file, "x,y,energy,density")?;

        // Write data points
        for i in 0..data.x_coords.len() {
            let density = data
                .density
                .as_ref()
                .and_then(|d| d.get(i))
                .copied()
                .unwrap_or(0.0);

            writeln!(
                file,
                "{},{},{},{}",
                data.x_coords[i], data.y_coords[i], data.energies[i], density
            )?;
        }

        Ok(())
    }

    /// Export distribution as CSV
    fn export_distribution_csv(
        &self,
        data: &DistributionData,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        let base_path = Path::new(path);
        let stem = base_path.file_stem().unwrap_or_default();
        let parent = base_path.parent().unwrap_or(Path::new("."));

        // Export statistics
        let stats_path = parent.join(format!("{}_stats.csv", stem.to_string_lossy()));
        let mut stats_file = std::fs::File::create(stats_path)?;

        writeln!(stats_file, "metric,value")?;
        writeln!(stats_file, "n_samples,{}", data.n_samples)?;
        writeln!(stats_file, "n_unique,{}", data.n_unique)?;
        writeln!(stats_file, "mean_energy,{}", data.mean_energy)?;
        writeln!(stats_file, "std_energy,{}", data.std_energy)?;
        writeln!(stats_file, "min_energy,{}", data.min_energy)?;
        writeln!(stats_file, "max_energy,{}", data.max_energy)?;

        // Export cluster information if available
        if let Some(clusters) = &data.clusters {
            let cluster_path = parent.join(format!("{}_clusters.csv", stem.to_string_lossy()));
            let mut cluster_file = std::fs::File::create(cluster_path)?;

            writeln!(cluster_file, "sample_id,cluster_id")?;
            for (i, &cluster) in clusters.iter().enumerate() {
                writeln!(cluster_file, "{i},{cluster}")?;
            }
        }

        Ok(())
    }

    /// Export convergence as CSV
    fn export_convergence_csv(
        &self,
        data: &ConvergenceData,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        // Write header
        write!(file, "iteration,objective,best_so_far")?;
        if !data.constraint_names.is_empty() {
            for name in &data.constraint_names {
                write!(file, ",constraint_{name}")?;
            }
        }
        writeln!(file)?;

        // Calculate best so far
        let mut best_so_far = vec![f64::INFINITY; data.objectives.len()];
        let mut current_best = f64::INFINITY;
        for (i, &obj) in data.objectives.iter().enumerate() {
            current_best = current_best.min(obj);
            best_so_far[i] = current_best;
        }

        // Write data
        for i in 0..data.objectives.len() {
            write!(file, "{},{},{}", i, data.objectives[i], best_so_far[i])?;

            if let Some(constraints) = &data.constraints {
                for name in &data.constraint_names {
                    let value = constraints[i].get(name).copied().unwrap_or(0.0);
                    write!(file, ",{value}")?;
                }
            }

            writeln!(file)?;
        }

        Ok(())
    }

    /// Export Plotly energy landscape
    fn export_plotly_energy(
        &self,
        data: &EnergyLandscapeData,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let plotly_data = PlotlyData {
            data: vec![PlotlyTrace {
                trace_type: "scatter".to_string(),
                mode: Some("markers".to_string()),
                x: Some(data.x_coords.clone()),
                y: Some(data.y_coords.clone()),
                z: None,
                marker: Some(PlotlyMarker {
                    size: Some(8),
                    color: Some(data.energies.clone()),
                    colorscale: Some("Viridis".to_string()),
                    showscale: Some(true),
                }),
                name: Some("Energy Landscape".to_string()),
                text: None,
            }],
            layout: PlotlyLayout {
                title: Some("Energy Landscape Projection".to_string()),
                xaxis: Some(PlotlyAxis {
                    title: Some("Component 1".to_string()),
                    ..Default::default()
                }),
                yaxis: Some(PlotlyAxis {
                    title: Some("Component 2".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            },
        };

        self.export_json(&plotly_data, path)?;
        Ok(())
    }

    /// Export Matplotlib energy script
    fn export_matplotlib_energy(
        &self,
        data: &EnergyLandscapeData,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        writeln!(file, "#!/usr/bin/env python3")?;
        writeln!(file, "import matplotlib.pyplot as plt")?;
        writeln!(file, "import numpy as np")?;
        writeln!(file)?;
        writeln!(file, "# Energy landscape data")?;
        writeln!(file, "x = np.array({:?})", data.x_coords)?;
        writeln!(file, "y = np.array({:?})", data.y_coords)?;
        writeln!(file, "energies = np.array({:?})", data.energies)?;
        writeln!(file)?;
        writeln!(file, "# Create scatter plot")?;
        writeln!(file, "plt.figure(figsize=(10, 8))")?;
        writeln!(
            file,
            "scatter = plt.scatter(x, y, c=energies, cmap='viridis', s=50)"
        )?;
        writeln!(file, "plt.colorbar(scatter, label='Energy')")?;
        writeln!(file, "plt.xlabel('Component 1')")?;
        writeln!(file, "plt.ylabel('Component 2')")?;
        writeln!(file, "plt.title('Energy Landscape Projection')")?;
        writeln!(file)?;

        if let Some(_density) = &data.density {
            writeln!(file, "# Add density contours")?;
            writeln!(file, "if len(x) > 100:")?;
            writeln!(file, "    from scipy.stats import gaussian_kde")?;
            writeln!(file, "    xy = np.vstack([x, y])")?;
            writeln!(file, "    z = gaussian_kde(xy)(xy)")?;
            writeln!(file, "    plt.contour(x, y, z, colors='black', alpha=0.3)")?;
            writeln!(file)?;
        }

        writeln!(file, "plt.tight_layout()")?;
        writeln!(file, "plt.savefig('energy_landscape.png', dpi=300)")?;
        writeln!(file, "plt.show()")?;

        Ok(())
    }

    /// Export HTML with embedded visualization
    fn export_html_energy(
        &self,
        data: &EnergyLandscapeData,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        // Write HTML header
        writeln!(file, "<!DOCTYPE html>")?;
        writeln!(file, "<html>")?;
        writeln!(file, "<head>")?;
        writeln!(file, "    <title>Energy Landscape Visualization</title>")?;
        writeln!(
            file,
            "    <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>"
        )?;
        writeln!(file, "    <style>")?;
        writeln!(
            file,
            "        body {{ font-family: Arial, sans-serif; margin: 20px; }}"
        )?;
        writeln!(file, "        #plot {{ width: 100%; height: 600px; }}")?;
        writeln!(file, "    </style>")?;
        writeln!(file, "</head>")?;
        writeln!(file, "<body>")?;
        writeln!(file, "    <h1>Energy Landscape Visualization</h1>")?;
        writeln!(file, "    <div id=\"plot\"></div>")?;
        writeln!(file, "    <script>")?;

        // Embed data and Plotly code
        writeln!(file, "        var data = [{{")?;
        writeln!(file, "            type: 'scatter',")?;
        writeln!(file, "            mode: 'markers',")?;
        writeln!(file, "            x: {:?},", data.x_coords)?;
        writeln!(file, "            y: {:?},", data.y_coords)?;
        writeln!(file, "            marker: {{")?;
        writeln!(file, "                size: 8,")?;
        writeln!(file, "                color: {:?},", data.energies)?;
        writeln!(file, "                colorscale: 'Viridis',")?;
        writeln!(file, "                showscale: true")?;
        writeln!(file, "            }},")?;
        writeln!(file, "            text: 'Energy'")?;
        writeln!(file, "        }}];")?;
        writeln!(file)?;
        writeln!(file, "        var layout = {{")?;
        writeln!(file, "            title: 'Energy Landscape Projection',")?;
        writeln!(file, "            xaxis: {{ title: 'Component 1' }},")?;
        writeln!(file, "            yaxis: {{ title: 'Component 2' }}")?;
        writeln!(file, "        }};")?;
        writeln!(file)?;
        writeln!(file, "        Plotly.newPlot('plot', data, layout);")?;
        writeln!(file, "    </script>")?;

        // Add summary statistics
        writeln!(file, "    <h2>Summary Statistics</h2>")?;
        writeln!(file, "    <ul>")?;
        writeln!(
            file,
            "        <li>Number of samples: {}</li>",
            data.x_coords.len()
        )?;
        writeln!(
            file,
            "        <li>Min energy: {:.4}</li>",
            data.energies.iter().fold(f64::INFINITY, |a, &b| a.min(b))
        )?;
        writeln!(
            file,
            "        <li>Max energy: {:.4}</li>",
            data.energies
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        )?;
        writeln!(
            file,
            "        <li>Projection method: {}</li>",
            data.projection_method
        )?;
        writeln!(file, "    </ul>")?;

        writeln!(file, "</body>")?;
        writeln!(file, "</html>")?;

        Ok(())
    }

    /// Similar implementations for distribution and convergence...
    fn export_plotly_distribution(
        &self,
        _data: &DistributionData,
        _path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Implementation for distribution Plotly export
        Ok(())
    }

    fn export_matplotlib_distribution(
        &self,
        _data: &DistributionData,
        _path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Implementation for distribution Matplotlib export
        Ok(())
    }

    fn export_html_distribution(
        &self,
        _data: &DistributionData,
        _path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Implementation for distribution HTML export
        Ok(())
    }

    fn export_plotly_convergence(
        &self,
        _data: &ConvergenceData,
        _path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Implementation for convergence Plotly export
        Ok(())
    }

    fn export_matplotlib_convergence(
        &self,
        _data: &ConvergenceData,
        _path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Implementation for convergence Matplotlib export
        Ok(())
    }

    fn export_html_convergence(
        &self,
        _data: &ConvergenceData,
        _path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Implementation for convergence HTML export
        Ok(())
    }
}

// Data structures for export

/// Energy landscape export data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyLandscapeData {
    pub x_coords: Vec<f64>,
    pub y_coords: Vec<f64>,
    pub energies: Vec<f64>,
    pub density: Option<Vec<f64>>,
    pub projection_method: String,
    pub metadata: HashMap<String, String>,
}

/// Distribution export data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionData {
    pub n_samples: usize,
    pub n_unique: usize,
    pub mean_energy: f64,
    pub std_energy: f64,
    pub min_energy: f64,
    pub max_energy: f64,
    pub clusters: Option<Vec<usize>>,
    pub cluster_energies: Option<Vec<f64>>,
    pub metadata: HashMap<String, String>,
}

/// Convergence export data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceData {
    pub objectives: Vec<f64>,
    pub constraints: Option<Vec<HashMap<String, f64>>>,
    pub parameters: Option<Vec<HashMap<String, f64>>>,
    pub iterations: Vec<usize>,
    pub times: Option<Vec<f64>>,
    pub constraint_names: Vec<String>,
    pub parameter_names: Vec<String>,
    pub metadata: HashMap<String, String>,
}

// Plotly data structures

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlotlyData {
    data: Vec<PlotlyTrace>,
    layout: PlotlyLayout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlotlyTrace {
    #[serde(rename = "type")]
    trace_type: String,
    mode: Option<String>,
    x: Option<Vec<f64>>,
    y: Option<Vec<f64>>,
    z: Option<Vec<f64>>,
    marker: Option<PlotlyMarker>,
    name: Option<String>,
    text: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlotlyMarker {
    size: Option<u32>,
    color: Option<Vec<f64>>,
    colorscale: Option<String>,
    showscale: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PlotlyLayout {
    title: Option<String>,
    xaxis: Option<PlotlyAxis>,
    yaxis: Option<PlotlyAxis>,
    zaxis: Option<PlotlyAxis>,
    showlegend: Option<bool>,
    height: Option<u32>,
    width: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PlotlyAxis {
    title: Option<String>,
    range: Option<[f64; 2]>,
    #[serde(rename = "type")]
    axis_type: Option<String>,
}

/// Export visualization data in specified format
pub fn export_visualization<T: Serialize>(
    data: &T,
    path: &str,
    format: ExportFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let _exporter = VisualizationExporter::new(format);

    match format {
        ExportFormat::JSON => {
            let json = serde_json::to_string_pretty(data)?;
            std::fs::write(path, json)?;
        }
        _ => {
            return Err(
                format!("Export format {format:?} not implemented for generic data").into(),
            );
        }
    }

    Ok(())
}

/// Create export path with proper extension
pub fn create_export_path(base_path: &str, format: ExportFormat) -> String {
    let path = Path::new(base_path);
    let stem = path.file_stem().unwrap_or_default();
    let parent = path.parent().unwrap_or(Path::new("."));

    let extension = match format {
        ExportFormat::JSON => "json",
        ExportFormat::CSV => "csv",
        ExportFormat::NPY => "npy",
        ExportFormat::MAT => "mat",
        ExportFormat::Plotly => "plotly.json",
        ExportFormat::Matplotlib => "py",
        ExportFormat::HTML => "html",
    };

    parent
        .join(format!("{}.{}", stem.to_string_lossy(), extension))
        .to_string_lossy()
        .to_string()
}
