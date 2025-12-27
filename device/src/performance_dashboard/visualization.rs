//! Visualization Configuration Types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Dashboard refresh rate in seconds
    pub refresh_rate: u64,
    /// Chart types to display
    pub chart_types: Vec<ChartType>,
    /// Layout configuration
    pub layout_config: LayoutConfig,
    /// Theme and styling
    pub theme_config: ThemeConfig,
    /// Interactive features
    pub interactive_config: InteractiveConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChartType {
    TimeSeries,
    Histogram,
    HeatMap,
    ScatterPlot,
    BarChart,
    BoxPlot,
    ViolinPlot,
    NetworkGraph,
    Sankey,
    Gauge,
    Table,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConfig {
    pub grid_layout: GridLayout,
    pub responsive_design: bool,
    pub panel_configuration: Vec<PanelConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeConfig {
    pub color_scheme: ColorScheme,
    pub dark_mode: bool,
    pub custom_styling: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveConfig {
    pub enable_drill_down: bool,
    pub enable_filtering: bool,
    pub enable_zooming: bool,
    pub enable_real_time_updates: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridLayout {
    pub rows: usize,
    pub columns: usize,
    pub gap_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelConfig {
    pub panel_id: String,
    pub chart_type: ChartType,
    pub data_source: String,
    pub position: (usize, usize),
    pub size: (usize, usize),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ColorScheme {
    Default,
    Light,
    Dark,
    HighContrast,
    Custom(Vec<String>),
}
