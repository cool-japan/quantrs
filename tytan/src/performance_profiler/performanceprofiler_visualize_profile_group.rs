//! # PerformanceProfiler - visualize_profile_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;

use super::types::Profile;

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Visualize profile
    #[cfg(feature = "plotters")]
    pub fn visualize_profile(&self, profile: &Profile, output_path: &str) -> Result<(), String> {
        let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| format!("Drawing error: {e}"))?;
        let mut chart = ChartBuilder::on(&root)
            .caption("Performance Profile", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(
                0f64..profile.metrics.time_metrics.total_time.as_secs_f64(),
                0f64..100f64,
            )
            .map_err(|e| format!("Chart error: {e}"))?;
        chart
            .configure_mesh()
            .draw()
            .map_err(|e| format!("Mesh error: {e}"))?;
        if !profile.resource_usage.cpu_usage.is_empty() {
            let cpu_data: Vec<(f64, f64)> = profile
                .resource_usage
                .cpu_usage
                .iter()
                .map(|(t, usage)| (t.duration_since(profile.start_time).as_secs_f64(), *usage))
                .collect();
            chart
                .draw_series(LineSeries::new(cpu_data, &RED))
                .map_err(|e| format!("Series error: {e}"))?
                .label("CPU Usage")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED));
        }
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()
            .map_err(|e| format!("Legend error: {e}"))?;
        root.present().map_err(|e| format!("Present error: {e}"))?;
        Ok(())
    }
}
