//! Report generation and visualization

use super::*;

/// Analysis report
#[derive(Debug, Clone)]
pub struct AnalysisReport {
    /// Report type
    pub report_type: ReportType,
    /// Report title
    pub title: String,
    /// Report content
    pub content: ReportContent,
    /// Generation timestamp
    pub timestamp: Instant,
    /// Report metadata
    pub metadata: ReportMetadata,
}

/// Report types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReportType {
    PerformanceSummary,
    DetailedAnalysis,
    TrendAnalysis,
    BenchmarkReport,
    BottleneckAnalysis,
    OptimizationReport,
    ComparisonReport,
    Custom { report_name: String },
}

/// Report content
#[derive(Debug, Clone)]
pub struct ReportContent {
    /// Executive summary
    pub executive_summary: String,
    /// Key findings
    pub key_findings: Vec<String>,
    /// Detailed sections
    pub sections: Vec<ReportSection>,
    /// Visualizations
    pub visualizations: Vec<Visualization>,
    /// Appendices
    pub appendices: Vec<Appendix>,
}

/// Report section
#[derive(Debug, Clone)]
pub struct ReportSection {
    /// Section title
    pub title: String,
    /// Section content
    pub content: String,
    /// Subsections
    pub subsections: Vec<Self>,
    /// Figures and tables
    pub figures: Vec<Figure>,
}

/// Visualization
#[derive(Debug, Clone)]
pub struct Visualization {
    /// Visualization type
    pub viz_type: VisualizationType,
    /// Title
    pub title: String,
    /// Data
    pub data: VisualizationData,
    /// Configuration
    pub config: RenderingConfig,
}

/// Visualization types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisualizationType {
    LineChart,
    BarChart,
    Histogram,
    ScatterPlot,
    BoxPlot,
    HeatMap,
    NetworkGraph,
    Timeline,
    Dashboard,
}

/// Visualization data
#[derive(Debug, Clone)]
pub enum VisualizationData {
    TimeSeries {
        x: Vec<f64>,
        y: Vec<f64>,
    },
    Scatter {
        x: Vec<f64>,
        y: Vec<f64>,
    },
    Histogram {
        values: Vec<f64>,
        bins: usize,
    },
    HeatMap {
        matrix: Array2<f64>,
    },
    Network {
        nodes: Vec<String>,
        edges: Vec<(usize, usize)>,
    },
}

/// Rendering configuration
#[derive(Debug, Clone)]
pub struct RenderingConfig {
    /// Width
    pub width: usize,
    /// Height
    pub height: usize,
    /// Color scheme
    pub color_scheme: String,
    /// Labels
    pub labels: HashMap<String, String>,
    /// Style options
    pub style_options: HashMap<String, String>,
}

/// Figure
#[derive(Debug, Clone)]
pub struct Figure {
    /// Figure caption
    pub caption: String,
    /// Figure data
    pub data: FigureData,
    /// Figure position
    pub position: FigurePosition,
}

/// Figure data
#[derive(Debug, Clone)]
pub enum FigureData {
    Table {
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
    },
    Image {
        path: String,
        alt_text: String,
    },
    Chart {
        visualization: Visualization,
    },
}

/// Figure position
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FigurePosition {
    Here,
    Top,
    Bottom,
    Page,
    Float,
}

/// Appendix
#[derive(Debug, Clone)]
pub struct Appendix {
    /// Appendix title
    pub title: String,
    /// Appendix content
    pub content: AppendixContent,
}

/// Appendix content
#[derive(Debug, Clone)]
pub enum AppendixContent {
    RawData { data: String },
    Code { language: String, code: String },
    Configuration { config: String },
    References { references: Vec<String> },
}

/// Report metadata
#[derive(Debug, Clone)]
pub struct ReportMetadata {
    /// Author
    pub author: String,
    /// Version
    pub version: String,
    /// Format
    pub format: ReportFormat,
    /// Tags
    pub tags: Vec<String>,
    /// Recipients
    pub recipients: Vec<String>,
}

/// Report formats
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReportFormat {
    PDF,
    HTML,
    Markdown,
    LaTeX,
    JSON,
    XML,
}
