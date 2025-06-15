//! Unified Quantum Hardware Benchmarking System
//!
//! This module provides a comprehensive, unified benchmarking system for quantum devices
//! that works across all quantum cloud providers (IBM, Azure, AWS) with advanced
//! statistical analysis, optimization, and reporting capabilities powered by SciRS2.

pub mod analysis;
pub mod config;
pub mod events;
pub mod optimization;
pub mod reporting;
pub mod results;
pub mod system;
pub mod types;

// Re-export commonly used types
pub use config::{
    AlgorithmBenchmarkConfig, BenchmarkExecutionParams, BenchmarkSuiteConfig,
    CircuitBenchmarkConfig, CircuitType, CustomBenchmarkDefinition, FidelityMeasurementMethod,
    GateBenchmarkConfig, HistoricalTrackingConfig, MLModelType, MultiQubitGate,
    OptimizationAlgorithm, OptimizationObjective, QuantumAlgorithm, ReportFormat, ReportingConfig,
    ResourceOptimizationConfig, SciRS2AnalysisConfig, SingleQubitGate, StatisticalTest,
    SystemBenchmarkConfig, TwoQubitGate, UnifiedBenchmarkConfig, VisualizationType,
};

pub use types::{BaselineMetric, BaselineMetricValue, PerformanceBaseline, QuantumPlatform};

pub use events::BenchmarkEvent;

pub use results::{
    AlgorithmLevelResults, CircuitLevelResults, CostAnalysisResult, CrossPlatformAnalysis,
    DeviceInfo, GateLevelResults, PlatformBenchmarkResult, ResourceAnalysisResult,
    SciRS2AnalysisResult, StatisticalSummary, SystemLevelResults, UnifiedBenchmarkResult,
};

pub use system::UnifiedQuantumBenchmarkSystem;

// Commonly used error and result types
use crate::{DeviceError, DeviceResult};
use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
