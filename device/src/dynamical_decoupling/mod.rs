//! Dynamical Decoupling Sequences with SciRS2 optimization
//!
//! This module provides comprehensive dynamical decoupling (DD) sequence generation,
//! optimization, and analysis using SciRS2's advanced optimization and statistical capabilities
//! for robust coherence preservation on quantum hardware.

pub mod adaptive;
pub mod analysis;
pub mod config;
pub mod executor;
pub mod fallback_scirs2;
pub mod hardware;
pub mod noise;
pub mod optimization;
pub mod performance;
pub mod sequences;
pub mod validation;

#[cfg(test)]
pub mod test_suite;

pub use adaptive::*;
pub use analysis::*;
pub use config::*;
pub use hardware::*;
pub use noise::*;
pub use optimization::*;
pub use performance::*;
pub use sequences::*;

use std::collections::HashMap;
use std::time::{Duration, Instant};

use quantrs2_circuit::prelude::*;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

use crate::{
    backend_traits::{query_backend_capabilities, BackendCapabilities},
    calibration::{CalibrationManager, DeviceCalibration},
    noise_model::CalibrationNoiseModel,
    topology::HardwareTopology,
    translation::HardwareBackend,
    CircuitResult, DeviceError, DeviceResult,
};

use self::optimization::DDSequenceOptimizer;

/// Main result type for dynamical decoupling operations
#[derive(Debug, Clone)]
pub struct DynamicalDecouplingResult {
    /// Optimized DD sequence
    pub optimized_sequence: DDSequence,
    /// Execution timing
    pub execution_time: Duration,
    /// Success status
    pub success: bool,
    /// Quality metrics
    pub quality_score: f64,
    /// Performance analysis
    pub performance_analysis: Option<DDPerformanceAnalysis>,
    /// Noise analysis
    pub noise_analysis: Option<DDNoiseAnalysis>,
    /// Hardware analysis
    pub hardware_analysis: Option<DDHardwareAnalysis>,
    /// Adaptation statistics
    pub adaptation_stats: Option<AdaptationStatistics>,
}

/// Comprehensive DD system manager
pub struct DynamicalDecouplingManager {
    /// System configuration
    pub config: DynamicalDecouplingConfig,
    /// Adaptive DD system
    pub adaptive_system: Option<AdaptiveDDSystem>,
    /// Performance analyzer
    pub performance_analyzer: DDPerformanceAnalyzer,
    /// Noise analyzer
    pub noise_analyzer: DDNoiseAnalyzer,
    /// Hardware analyzer
    pub hardware_analyzer: DDHardwareAnalyzer,
    /// Sequence optimizer
    pub sequence_optimizer: DDSequenceOptimizer,
    /// Sequence cache
    pub sequence_cache: SequenceCache,
    /// Multi-qubit coordinator
    pub multi_qubit_coordinator: Option<MultiQubitDDCoordinator>,
}

impl DynamicalDecouplingManager {
    /// Create new DD manager
    pub fn new(
        config: DynamicalDecouplingConfig,
        device_id: String,
        topology: Option<crate::topology::HardwareTopology>,
        calibration_manager: Option<crate::calibration::CalibrationManager>,
    ) -> Self {
        let performance_analyzer = DDPerformanceAnalyzer::new(config.performance_config.clone());
        let noise_analyzer = DDNoiseAnalyzer::new(config.noise_characterization.clone());
        let hardware_analyzer = DDHardwareAnalyzer::new(
            config.hardware_adaptation.clone(),
            calibration_manager,
            topology,
        );
        let sequence_optimizer = DDSequenceOptimizer::new(config.optimization_config.clone());

        Self {
            config,
            adaptive_system: None,
            performance_analyzer,
            noise_analyzer,
            hardware_analyzer,
            sequence_optimizer,
            sequence_cache: SequenceCache::new(),
            multi_qubit_coordinator: None,
        }
    }

    /// Initialize adaptive DD system
    pub fn initialize_adaptive_system(
        &mut self,
        adaptive_config: AdaptiveDDConfig,
        initial_sequence: DDSequence,
        available_sequences: Vec<DDSequenceType>,
    ) -> DeviceResult<()> {
        let adaptive_system =
            AdaptiveDDSystem::new(adaptive_config, initial_sequence, available_sequences);

        self.adaptive_system = Some(adaptive_system);
        Ok(())
    }

    /// Initialize multi-qubit coordination
    pub fn initialize_multi_qubit_coordination(
        &mut self,
        crosstalk_mitigation: CrosstalkMitigationStrategy,
        synchronization: hardware::SynchronizationRequirements,
    ) {
        self.multi_qubit_coordinator = Some(MultiQubitDDCoordinator::new(
            crosstalk_mitigation,
            synchronization,
        ));
    }

    /// Generate optimized DD sequence
    pub async fn generate_optimized_sequence(
        &mut self,
        sequence_type: &DDSequenceType,
        target_qubits: &[quantrs2_core::qubit::QubitId],
        duration: f64,
        executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<DynamicalDecouplingResult> {
        let start_time = std::time::Instant::now();

        // Check cache first
        let cache_key = format!("{:?}_{}_{}", sequence_type, target_qubits.len(), duration);
        if let Some(cached_sequence) = self.sequence_cache.get_sequence(&cache_key) {
            println!("Using cached DD sequence: {}", cache_key);
            return Ok(DynamicalDecouplingResult {
                optimized_sequence: cached_sequence,
                execution_time: start_time.elapsed(),
                success: true,
                quality_score: 0.95,
                performance_analysis: None,
                noise_analysis: None,
                hardware_analysis: None,
                adaptation_stats: None,
            });
        }

        // Generate base sequence
        let base_sequence =
            DDSequenceGenerator::generate_base_sequence(sequence_type, target_qubits, duration)?;

        // Optimize sequence
        let optimization_result = self
            .sequence_optimizer
            .optimize_sequence(&base_sequence, executor)
            .await?;

        let optimized_sequence = optimization_result.optimized_sequence;

        // Analyze performance
        let performance_analysis = self
            .performance_analyzer
            .analyze_performance(&optimized_sequence, executor)
            .await?;

        // Analyze noise characteristics
        let noise_analysis = self
            .noise_analyzer
            .analyze_noise_characteristics(&optimized_sequence, &performance_analysis)?;

        // Analyze hardware compatibility
        let hardware_analysis = self.hardware_analyzer.analyze_hardware_implementation(
            &format!("device_{}", target_qubits.len()),
            &optimized_sequence,
        )?;

        // Cache the optimized sequence
        self.sequence_cache
            .store_sequence(cache_key, optimized_sequence.clone());

        // Calculate quality score
        let quality_score = self.calculate_quality_score(
            &performance_analysis,
            &noise_analysis,
            &hardware_analysis,
        );

        let result = DynamicalDecouplingResult {
            optimized_sequence,
            execution_time: start_time.elapsed(),
            success: optimization_result.optimization_metrics.success,
            quality_score,
            performance_analysis: Some(performance_analysis),
            noise_analysis: Some(noise_analysis),
            hardware_analysis: Some(hardware_analysis),
            adaptation_stats: self
                .adaptive_system
                .as_ref()
                .map(|sys| sys.get_adaptation_statistics()),
        };

        // Update adaptive system if available
        if let Some(ref mut adaptive_system) = self.adaptive_system {
            if let (Some(ref perf), Some(ref noise)) =
                (&result.performance_analysis, &result.noise_analysis)
            {
                adaptive_system.update_performance(perf, noise)?;
            }
        }

        Ok(result)
    }

    /// Generate coordinated multi-qubit sequence
    pub fn generate_multi_qubit_sequence(
        &mut self,
        qubit_groups: Vec<(Vec<quantrs2_core::qubit::QubitId>, DDSequenceType)>,
        duration: f64,
    ) -> DeviceResult<DDSequence> {
        if let Some(ref mut coordinator) = self.multi_qubit_coordinator {
            // Add sequences for each qubit group
            for (qubits, sequence_type) in qubit_groups {
                let sequence =
                    DDSequenceGenerator::generate_base_sequence(&sequence_type, &qubits, duration)?;
                coordinator.add_sequence(qubits, sequence);
            }

            // Generate coordinated sequence
            coordinator.generate_coordinated_sequence()
        } else {
            Err(crate::DeviceError::InvalidInput(
                "Multi-qubit coordinator not initialized".to_string(),
            ))
        }
    }

    /// Calculate overall quality score
    fn calculate_quality_score(
        &self,
        performance: &DDPerformanceAnalysis,
        noise: &DDNoiseAnalysis,
        hardware: &DDHardwareAnalysis,
    ) -> f64 {
        let performance_score =
            performance.metrics.values().sum::<f64>() / performance.metrics.len() as f64;
        let noise_score = noise.suppression_effectiveness.overall_suppression;
        let hardware_score = hardware.hardware_compatibility.compatibility_score;

        // Weighted average
        let weights = [0.4, 0.3, 0.3]; // Performance, noise, hardware
        let scores = [performance_score, noise_score, hardware_score];

        let score = weights
            .iter()
            .zip(scores.iter())
            .map(|(w, s)| w * s)
            .sum::<f64>();

        // Clamp to [0, 1] range
        score.max(0.0).min(1.0)
    }

    /// Get system status
    pub fn get_system_status(&self) -> DDSystemStatus {
        DDSystemStatus {
            adaptive_enabled: self.adaptive_system.is_some(),
            multi_qubit_enabled: self.multi_qubit_coordinator.is_some(),
            cache_statistics: self.sequence_cache.get_cache_statistics(),
            total_sequences_generated: self.sequence_cache.cached_sequences.len(),
            optimization_success_rate: self.sequence_optimizer.best_objective_value.abs(),
        }
    }
}

/// DD system status
#[derive(Debug, Clone)]
pub struct DDSystemStatus {
    /// Adaptive DD enabled
    pub adaptive_enabled: bool,
    /// Multi-qubit coordination enabled
    pub multi_qubit_enabled: bool,
    /// Cache statistics (hits, misses, hit rate)
    pub cache_statistics: (usize, usize, f64),
    /// Total sequences generated
    pub total_sequences_generated: usize,
    /// Optimization success rate
    pub optimization_success_rate: f64,
}

/// Circuit executor trait for DD operations  
pub trait DDCircuitExecutor: Send + Sync {
    /// Execute a circuit and return results
    fn execute_circuit(
        &self,
        circuit: &Circuit<16>,
    ) -> Result<CircuitExecutionResults, DeviceError>;

    /// Get backend capabilities
    fn get_capabilities(&self) -> BackendCapabilities;

    /// Estimate execution time
    fn estimate_execution_time(&self, circuit: &Circuit<16>) -> Duration;
}

/// Circuit execution results
#[derive(Debug, Clone)]
pub struct CircuitExecutionResults {
    /// Measurement results
    pub measurements: HashMap<String, Vec<i32>>,
    /// Execution fidelity
    pub fidelity: f64,
    /// Execution time
    pub execution_time: Duration,
    /// Error rates
    pub error_rates: HashMap<String, f64>,
    /// System metadata
    pub metadata: CircuitExecutionMetadata,
}

/// Circuit execution metadata
#[derive(Debug, Clone)]
pub struct CircuitExecutionMetadata {
    /// Backend used
    pub backend: String,
    /// Quantum volume
    pub quantum_volume: usize,
    /// Hardware topology
    pub topology_type: String,
    /// Calibration timestamp
    pub calibration_timestamp: std::time::SystemTime,
    /// Environmental conditions
    pub environmental_conditions: std::collections::HashMap<String, f64>,
}
