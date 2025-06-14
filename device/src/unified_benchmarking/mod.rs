//! Unified Quantum Hardware Benchmarking System
//!
//! This module provides a comprehensive, unified benchmarking system for quantum devices
//! that works across all quantum cloud providers (IBM, Azure, AWS) with advanced
//! statistical analysis, optimization, and reporting capabilities powered by SciRS2.

pub mod config;
pub mod types;
pub mod events;
pub mod results;
pub mod system;
pub mod optimization;
pub mod reporting;
pub mod analysis;

// Re-export commonly used types
pub use config::{
    UnifiedBenchmarkConfig, BenchmarkSuiteConfig, GateBenchmarkConfig, CircuitBenchmarkConfig,
    AlgorithmBenchmarkConfig, SystemBenchmarkConfig, SciRS2AnalysisConfig, ReportingConfig,
    ResourceOptimizationConfig, HistoricalTrackingConfig, SingleQubitGate, TwoQubitGate,
    MultiQubitGate, FidelityMeasurementMethod, CircuitType, QuantumAlgorithm, 
    BenchmarkExecutionParams, StatisticalTest, MLModelType, OptimizationObjective,
    OptimizationAlgorithm, ReportFormat, VisualizationType, CustomBenchmarkDefinition,
};

pub use types::{QuantumPlatform, BaselineMetric, BaselineMetricValue, PerformanceBaseline};

pub use events::BenchmarkEvent;

pub use results::{
    UnifiedBenchmarkResult, PlatformBenchmarkResult, DeviceInfo, GateLevelResults,
    CircuitLevelResults, AlgorithmLevelResults, SystemLevelResults, StatisticalSummary,
    CrossPlatformAnalysis, SciRS2AnalysisResult, ResourceAnalysisResult, CostAnalysisResult,
};

pub use system::UnifiedQuantumBenchmarkSystem;

// Commonly used error and result types
use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use crate::{DeviceError, DeviceResult};