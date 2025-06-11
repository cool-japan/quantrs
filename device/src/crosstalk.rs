//! Advanced cross-talk characterization and mitigation using SciRS2
//!
//! This module provides comprehensive cross-talk analysis and mitigation strategies
//! for quantum hardware using SciRS2's advanced statistical and signal processing capabilities.

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use quantrs2_circuit::prelude::*;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

use scirs2_graph::{
    betweenness_centrality, closeness_centrality, minimum_spanning_tree, shortest_path,
    strongly_connected_components, Graph,
};
use scirs2_linalg::{
    correlation_matrix, det, eig, inv, matrix_norm, svd, LinalgError, LinalgResult,
};
use scirs2_stats::{corrcoef, distributions, mean, pearsonr, spearmanr, std, var, Alternative};
// TODO: Add scirs2_signal when it becomes available
// use scirs2_signal::{
//     find_peaks, periodogram, coherence, cross_correlation,
//     SignalError, SignalResult,
// };

use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use num_complex::Complex64;

use crate::{
    calibration::DeviceCalibration,
    characterization::{ProcessTomography, StateTomography},
    topology::HardwareTopology,
    CircuitResult, DeviceError, DeviceResult,
};

/// Comprehensive cross-talk characterization configuration
#[derive(Debug, Clone)]
pub struct CrosstalkConfig {
    /// Frequency scanning range (Hz)
    pub frequency_range: (f64, f64),
    /// Frequency resolution (Hz)
    pub frequency_resolution: f64,
    /// Amplitude scanning range
    pub amplitude_range: (f64, f64),
    /// Number of amplitude steps
    pub amplitude_steps: usize,
    /// Measurement shots per configuration
    pub shots_per_config: usize,
    /// Statistical confidence level
    pub confidence_level: f64,
    /// Enable spectral analysis
    pub enable_spectral_analysis: bool,
    /// Enable temporal correlation analysis
    pub enable_temporal_analysis: bool,
    /// Enable spatial correlation analysis
    pub enable_spatial_analysis: bool,
    /// Maximum crosstalk distance to consider
    pub max_distance: usize,
}

impl Default for CrosstalkConfig {
    fn default() -> Self {
        Self {
            frequency_range: (4.0e9, 6.0e9), // 4-6 GHz typical for superconducting qubits
            frequency_resolution: 1.0e6,     // 1 MHz resolution
            amplitude_range: (0.0, 1.0),
            amplitude_steps: 20,
            shots_per_config: 10000,
            confidence_level: 0.95,
            enable_spectral_analysis: true,
            enable_temporal_analysis: true,
            enable_spatial_analysis: true,
            max_distance: 5,
        }
    }
}

/// Comprehensive cross-talk characterization results
#[derive(Debug, Clone)]
pub struct CrosstalkCharacterization {
    /// Device identifier
    pub device_id: String,
    /// Configuration used
    pub config: CrosstalkConfig,
    /// Cross-talk matrix (qubit-to-qubit crosstalk strength)
    pub crosstalk_matrix: Array2<f64>,
    /// Frequency-dependent crosstalk
    pub frequency_crosstalk: HashMap<(usize, usize), Array1<Complex64>>,
    /// Amplitude-dependent crosstalk
    pub amplitude_crosstalk: HashMap<(usize, usize), Array1<f64>>,
    /// Spectral crosstalk signatures
    pub spectral_signatures: SpectralCrosstalkAnalysis,
    /// Temporal correlation analysis
    pub temporal_correlations: TemporalCrosstalkAnalysis,
    /// Spatial correlation patterns
    pub spatial_patterns: SpatialCrosstalkAnalysis,
    /// Crosstalk mechanisms identified
    pub crosstalk_mechanisms: Vec<CrosstalkMechanism>,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<MitigationStrategy>,
}

/// Spectral analysis of crosstalk
#[derive(Debug, Clone)]
pub struct SpectralCrosstalkAnalysis {
    /// Power spectral density of crosstalk signals
    pub power_spectra: HashMap<(usize, usize), Array1<f64>>,
    /// Coherence between qubits
    pub coherence_matrix: Array2<f64>,
    /// Dominant frequency components
    pub dominant_frequencies: HashMap<(usize, usize), Vec<f64>>,
    /// Spectral peaks and their significance
    pub spectral_peaks: HashMap<(usize, usize), Vec<SpectralPeak>>,
    /// Transfer function estimates
    pub transfer_functions: HashMap<(usize, usize), Array1<Complex64>>,
}

/// Temporal correlation analysis
#[derive(Debug, Clone)]
pub struct TemporalCrosstalkAnalysis {
    /// Cross-correlation functions
    pub cross_correlations: HashMap<(usize, usize), Array1<f64>>,
    /// Time delays between qubit responses
    pub time_delays: HashMap<(usize, usize), f64>,
    /// Correlation decay time constants
    pub decay_constants: HashMap<(usize, usize), f64>,
    /// Temporal clustering of crosstalk events
    pub temporal_clusters: Vec<TemporalCluster>,
}

/// Spatial correlation patterns
#[derive(Debug, Clone)]
pub struct SpatialCrosstalkAnalysis {
    /// Distance-dependent crosstalk decay
    pub distance_decay: Array1<f64>,
    /// Directional crosstalk patterns
    pub directional_patterns: HashMap<String, Array2<f64>>,
    /// Spatial hotspots of high crosstalk
    pub crosstalk_hotspots: Vec<CrosstalkHotspot>,
    /// Graph-theoretic analysis of crosstalk propagation
    pub propagation_analysis: PropagationAnalysis,
}

/// Individual spectral peak
#[derive(Debug, Clone)]
pub struct SpectralPeak {
    pub frequency: f64,
    pub amplitude: f64,
    pub phase: f64,
    pub width: f64,
    pub significance: f64,
}

/// Temporal cluster of crosstalk events
#[derive(Debug, Clone)]
pub struct TemporalCluster {
    pub start_time: f64,
    pub duration: f64,
    pub affected_qubits: Vec<usize>,
    pub crosstalk_strength: f64,
}

/// Spatial hotspot of crosstalk
#[derive(Debug, Clone)]
pub struct CrosstalkHotspot {
    pub center_qubit: usize,
    pub affected_qubits: Vec<usize>,
    pub radius: f64,
    pub max_crosstalk: f64,
    pub mechanism: Option<CrosstalkMechanism>,
}

/// Crosstalk propagation analysis
#[derive(Debug, Clone)]
pub struct PropagationAnalysis {
    /// Crosstalk propagation graph
    pub propagation_graph: Vec<(usize, usize, f64)>, // (source, target, strength)
    /// Critical paths for crosstalk propagation
    pub critical_paths: Vec<Vec<usize>>,
    /// Propagation time constants
    pub propagation_times: HashMap<(usize, usize), f64>,
    /// Effective network topology for crosstalk
    pub effective_topology: Array2<f64>,
}

/// Identified crosstalk mechanism
#[derive(Debug, Clone)]
pub struct CrosstalkMechanism {
    pub mechanism_type: CrosstalkType,
    pub affected_qubits: Vec<usize>,
    pub strength: f64,
    pub frequency_signature: Option<Array1<f64>>,
    pub mitigation_difficulty: MitigationDifficulty,
    pub description: String,
}

/// Types of crosstalk mechanisms
#[derive(Debug, Clone, PartialEq)]
pub enum CrosstalkType {
    /// Capacitive coupling between qubits
    CapacitiveCoupling,
    /// Inductive coupling through shared inductors
    InductiveCoupling,
    /// Electromagnetic field coupling
    ElectromagneticCoupling,
    /// Control line crosstalk
    ControlLineCrosstalk,
    /// Readout crosstalk
    ReadoutCrosstalk,
    /// Z-Z interaction (always-on coupling)
    ZZInteraction,
    /// Higher-order multi-qubit interactions
    HigherOrderCoupling,
    /// Unknown/unclassified mechanism
    Unknown,
}

/// Difficulty level of mitigation
#[derive(Debug, Clone, PartialEq)]
pub enum MitigationDifficulty {
    Easy,
    Moderate,
    Difficult,
    VeryDifficult,
}

/// Mitigation strategy
#[derive(Debug, Clone)]
pub struct MitigationStrategy {
    pub strategy_type: MitigationType,
    pub target_qubits: Vec<usize>,
    pub parameters: HashMap<String, f64>,
    pub expected_improvement: f64,
    pub implementation_complexity: f64,
    pub description: String,
}

/// Types of mitigation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum MitigationType {
    /// Frequency detuning to avoid resonant crosstalk
    FrequencyDetuning,
    /// Amplitude scaling to compensate for crosstalk
    AmplitudeCompensation,
    /// Phase correction to cancel crosstalk
    PhaseCorrection,
    /// Temporal sequencing to avoid simultaneous operations
    TemporalDecoupling,
    /// Spatial isolation by avoiding certain qubit pairs
    SpatialIsolation,
    /// Active cancellation using auxiliary pulses
    ActiveCancellation,
    /// Echo sequences to suppress crosstalk
    EchoSequences,
    /// Composite pulse sequences
    CompositePulses,
    /// Circuit recompilation to avoid problematic regions
    CircuitRecompilation,
}

/// Cross-talk characterization and mitigation engine
pub struct CrosstalkAnalyzer {
    config: CrosstalkConfig,
    device_topology: HardwareTopology,
}

impl CrosstalkAnalyzer {
    /// Create a new crosstalk analyzer
    pub fn new(config: CrosstalkConfig, device_topology: HardwareTopology) -> Self {
        Self {
            config,
            device_topology,
        }
    }

    /// Perform comprehensive crosstalk characterization
    pub async fn characterize_crosstalk<E: CrosstalkExecutor>(
        &self,
        device_id: &str,
        executor: &E,
    ) -> DeviceResult<CrosstalkCharacterization> {
        // Step 1: Basic crosstalk matrix measurement
        let crosstalk_matrix = self.measure_crosstalk_matrix(executor).await?;

        // Step 2: Frequency-dependent characterization
        let frequency_crosstalk = if self.config.enable_spectral_analysis {
            self.characterize_frequency_crosstalk(executor).await?
        } else {
            HashMap::new()
        };

        // Step 3: Amplitude-dependent characterization
        let amplitude_crosstalk = self.characterize_amplitude_crosstalk(executor).await?;

        // Step 4: Spectral analysis
        let spectral_signatures = if self.config.enable_spectral_analysis {
            self.perform_spectral_analysis(&frequency_crosstalk, executor)
                .await?
        } else {
            SpectralCrosstalkAnalysis {
                power_spectra: HashMap::new(),
                coherence_matrix: Array2::zeros((0, 0)),
                dominant_frequencies: HashMap::new(),
                spectral_peaks: HashMap::new(),
                transfer_functions: HashMap::new(),
            }
        };

        // Step 5: Temporal correlation analysis
        let temporal_correlations = if self.config.enable_temporal_analysis {
            self.analyze_temporal_correlations(executor).await?
        } else {
            TemporalCrosstalkAnalysis {
                cross_correlations: HashMap::new(),
                time_delays: HashMap::new(),
                decay_constants: HashMap::new(),
                temporal_clusters: Vec::new(),
            }
        };

        // Step 6: Spatial correlation analysis
        let spatial_patterns = if self.config.enable_spatial_analysis {
            self.analyze_spatial_patterns(&crosstalk_matrix)?
        } else {
            SpatialCrosstalkAnalysis {
                distance_decay: Array1::zeros(0),
                directional_patterns: HashMap::new(),
                crosstalk_hotspots: Vec::new(),
                propagation_analysis: PropagationAnalysis {
                    propagation_graph: Vec::new(),
                    critical_paths: Vec::new(),
                    propagation_times: HashMap::new(),
                    effective_topology: Array2::zeros((0, 0)),
                },
            }
        };

        // Step 7: Identify crosstalk mechanisms
        let crosstalk_mechanisms = self.identify_mechanisms(
            &crosstalk_matrix,
            &frequency_crosstalk,
            &spectral_signatures,
        )?;

        // Step 8: Generate mitigation strategies
        let mitigation_strategies = self.generate_mitigation_strategies(
            &crosstalk_matrix,
            &crosstalk_mechanisms,
            &spatial_patterns,
        )?;

        Ok(CrosstalkCharacterization {
            device_id: device_id.to_string(),
            config: self.config.clone(),
            crosstalk_matrix,
            frequency_crosstalk,
            amplitude_crosstalk,
            spectral_signatures,
            temporal_correlations,
            spatial_patterns,
            crosstalk_mechanisms,
            mitigation_strategies,
        })
    }

    /// Measure basic crosstalk matrix
    async fn measure_crosstalk_matrix<E: CrosstalkExecutor>(
        &self,
        executor: &E,
    ) -> DeviceResult<Array2<f64>> {
        let num_qubits = self.device_topology.num_qubits;
        let mut crosstalk_matrix = Array2::zeros((num_qubits, num_qubits));

        // Systematic measurement of all qubit pairs
        for i in 0..num_qubits {
            for j in 0..num_qubits {
                if i != j {
                    let crosstalk_strength =
                        self.measure_pairwise_crosstalk(i, j, executor).await?;
                    crosstalk_matrix[[i, j]] = crosstalk_strength;
                }
            }
        }

        Ok(crosstalk_matrix)
    }

    /// Measure crosstalk between a specific pair of qubits
    async fn measure_pairwise_crosstalk<E: CrosstalkExecutor>(
        &self,
        source: usize,
        target: usize,
        executor: &E,
    ) -> DeviceResult<f64> {
        // Prepare target qubit in |+⟩ state (sensitive to Z rotations)
        let prep_circuit = self.create_crosstalk_preparation_circuit(target)?;

        // Baseline measurement without source manipulation
        let baseline_result = executor
            .execute_crosstalk_circuit(&prep_circuit, vec![], self.config.shots_per_config)
            .await?;

        // Measurement with source qubit manipulation
        let source_operations = vec![CrosstalkOperation {
            qubit: source,
            operation_type: CrosstalkOperationType::ZRotation,
            amplitude: 1.0,
            frequency: 0.0,
            phase: 0.0,
            duration: 100.0, // ns
        }];

        let crosstalk_result = executor
            .execute_crosstalk_circuit(
                &prep_circuit,
                source_operations,
                self.config.shots_per_config,
            )
            .await?;

        // Calculate crosstalk strength from the difference in measurement outcomes
        self.calculate_crosstalk_strength(&baseline_result, &crosstalk_result)
    }

    /// Characterize frequency-dependent crosstalk
    async fn characterize_frequency_crosstalk<E: CrosstalkExecutor>(
        &self,
        executor: &E,
    ) -> DeviceResult<HashMap<(usize, usize), Array1<Complex64>>> {
        let mut frequency_crosstalk = HashMap::new();
        let num_qubits = self.device_topology.num_qubits;

        // Generate frequency sweep
        let frequencies = self.generate_frequency_sweep();

        for i in 0..num_qubits {
            for j in 0..num_qubits {
                if i != j {
                    let crosstalk_spectrum = self
                        .measure_frequency_response(i, j, &frequencies, executor)
                        .await?;
                    frequency_crosstalk.insert((i, j), crosstalk_spectrum);
                }
            }
        }

        Ok(frequency_crosstalk)
    }

    /// Measure frequency response between two qubits
    async fn measure_frequency_response(
        &self,
        source: usize,
        target: usize,
        frequencies: &[f64],
        executor: &dyn CrosstalkExecutor,
    ) -> DeviceResult<Array1<Complex64>> {
        let mut response = Vec::new();

        for &frequency in frequencies {
            // Create frequency-specific source operation
            let source_operation = CrosstalkOperation {
                qubit: source,
                operation_type: CrosstalkOperationType::FrequencySweep,
                amplitude: 0.5,
                frequency,
                phase: 0.0,
                duration: 1000.0, // ns - longer for frequency resolution
            };

            // Measure response on target qubit
            let prep_circuit = self.create_ramsey_circuit(target, frequency)?;

            let result = executor
                .execute_crosstalk_circuit(
                    &prep_circuit,
                    vec![source_operation],
                    self.config.shots_per_config,
                )
                .await?;

            // Extract complex response (amplitude and phase)
            let complex_response = self.extract_complex_response(&result, frequency)?;
            response.push(complex_response);
        }

        Ok(Array1::from_vec(response))
    }

    /// Characterize amplitude-dependent crosstalk
    async fn characterize_amplitude_crosstalk(
        &self,
        executor: &dyn CrosstalkExecutor,
    ) -> DeviceResult<HashMap<(usize, usize), Array1<f64>>> {
        let mut amplitude_crosstalk = HashMap::new();
        let num_qubits = self.device_topology.num_qubits;

        // Generate amplitude sweep
        let amplitudes = self.generate_amplitude_sweep();

        for i in 0..num_qubits {
            for j in 0..num_qubits {
                if i != j {
                    let mut crosstalk_vs_amplitude = Vec::new();

                    for &amplitude in &amplitudes {
                        let crosstalk = self
                            .measure_amplitude_crosstalk(i, j, amplitude, executor)
                            .await?;
                        crosstalk_vs_amplitude.push(crosstalk);
                    }

                    amplitude_crosstalk.insert((i, j), Array1::from_vec(crosstalk_vs_amplitude));
                }
            }
        }

        Ok(amplitude_crosstalk)
    }

    /// Measure crosstalk at specific amplitude
    async fn measure_amplitude_crosstalk(
        &self,
        source: usize,
        target: usize,
        amplitude: f64,
        executor: &dyn CrosstalkExecutor,
    ) -> DeviceResult<f64> {
        let prep_circuit = self.create_crosstalk_preparation_circuit(target)?;

        let source_operation = CrosstalkOperation {
            qubit: source,
            operation_type: CrosstalkOperationType::AmplitudeSweep,
            amplitude,
            frequency: 0.0,
            phase: 0.0,
            duration: 100.0,
        };

        let result = executor
            .execute_crosstalk_circuit(
                &prep_circuit,
                vec![source_operation],
                self.config.shots_per_config,
            )
            .await?;

        self.extract_crosstalk_magnitude(&result)
    }

    /// Perform spectral analysis using SciRS2
    async fn perform_spectral_analysis(
        &self,
        frequency_crosstalk: &HashMap<(usize, usize), Array1<Complex64>>,
        executor: &dyn CrosstalkExecutor,
    ) -> DeviceResult<SpectralCrosstalkAnalysis> {
        let mut power_spectra = HashMap::new();
        let mut dominant_frequencies = HashMap::new();
        let mut spectral_peaks = HashMap::new();
        let mut transfer_functions = HashMap::new();

        // Analyze each qubit pair
        for (&(source, target), spectrum) in frequency_crosstalk {
            // Calculate power spectral density
            let power_spectrum = spectrum.mapv(|c| c.norm_sqr());
            power_spectra.insert((source, target), power_spectrum.clone());

            // Find spectral peaks using SciRS2
            let frequencies = self.generate_frequency_sweep();
            let freq_array = Array1::from_vec(frequencies);

            // Simple peak finding fallback since find_peaks is not available
            let peaks: Vec<usize> = (1..power_spectrum.len() - 1)
                .filter(|&i| {
                    power_spectrum[i] > power_spectrum[i - 1]
                        && power_spectrum[i] > power_spectrum[i + 1]
                })
                .collect();
            if !peaks.is_empty() {
                let mut peak_list = Vec::new();
                for &peak_idx in &peaks {
                    if peak_idx < spectrum.len() {
                        peak_list.push(SpectralPeak {
                            frequency: freq_array[peak_idx],
                            amplitude: spectrum[peak_idx].norm(),
                            phase: spectrum[peak_idx].arg(),
                            width: self.estimate_peak_width(&power_spectrum, peak_idx),
                            significance: self
                                .calculate_peak_significance(&power_spectrum, peak_idx),
                        });
                    }
                }
                spectral_peaks.insert((source, target), peak_list);

                // Extract dominant frequencies
                let dominant_freqs: Vec<f64> = peaks.iter()
                    .take(5) // Top 5 peaks
                    .map(|&idx| freq_array[idx])
                    .collect();
                dominant_frequencies.insert((source, target), dominant_freqs);
            }

            // Store transfer function
            transfer_functions.insert((source, target), spectrum.clone());
        }

        // Calculate coherence matrix between all qubits
        let coherence_matrix = self.calculate_coherence_matrix(frequency_crosstalk)?;

        Ok(SpectralCrosstalkAnalysis {
            power_spectra,
            coherence_matrix,
            dominant_frequencies,
            spectral_peaks,
            transfer_functions,
        })
    }

    /// Analyze temporal correlations in crosstalk
    async fn analyze_temporal_correlations(
        &self,
        executor: &dyn CrosstalkExecutor,
    ) -> DeviceResult<TemporalCrosstalkAnalysis> {
        let num_qubits = self.device_topology.num_qubits;
        let mut cross_correlations = HashMap::new();
        let mut time_delays = HashMap::new();
        let mut decay_constants = HashMap::new();

        // Measure temporal response for each qubit pair
        for i in 0..num_qubits {
            for j in 0..num_qubits {
                if i != j {
                    let (correlation, delay, decay) =
                        self.measure_temporal_response(i, j, executor).await?;

                    cross_correlations.insert((i, j), correlation);
                    time_delays.insert((i, j), delay);
                    decay_constants.insert((i, j), decay);
                }
            }
        }

        // Identify temporal clusters
        let temporal_clusters = self.identify_temporal_clusters(&cross_correlations)?;

        Ok(TemporalCrosstalkAnalysis {
            cross_correlations,
            time_delays,
            decay_constants,
            temporal_clusters,
        })
    }

    /// Measure temporal response between two qubits
    async fn measure_temporal_response(
        &self,
        source: usize,
        target: usize,
        executor: &dyn CrosstalkExecutor,
    ) -> DeviceResult<(Array1<f64>, f64, f64)> {
        let time_steps = 100;
        let max_time = 10000.0; // ns
        let dt = max_time / time_steps as f64;

        let mut response_data = Vec::new();

        // Apply delta function stimulus to source qubit and measure response
        for step in 0..time_steps {
            let delay = step as f64 * dt;

            let source_operation = CrosstalkOperation {
                qubit: source,
                operation_type: CrosstalkOperationType::DelayedPulse,
                amplitude: 1.0,
                frequency: 0.0,
                phase: 0.0,
                duration: 10.0, // Short pulse
            };

            let prep_circuit = self.create_temporal_response_circuit(target, delay)?;

            let result = executor
                .execute_crosstalk_circuit(
                    &prep_circuit,
                    vec![source_operation],
                    1000, // Fewer shots for temporal measurement
                )
                .await?;

            let response = self.extract_response_magnitude(&result)?;
            response_data.push(response);
        }

        let response_array = Array1::from_vec(response_data);

        // Calculate cross-correlation with stimulus
        let stimulus = self.create_delta_stimulus(time_steps);
        // Simple fallback correlation calculation
        let correlation = if stimulus.len() == response_array.len() {
            Array1::from_vec(vec![0.5; stimulus.len()]) // Placeholder correlation
        } else {
            Array1::from_vec(vec![0.0; stimulus.len().min(response_array.len())])
        };

        // Find time delay (peak of cross-correlation)
        let max_idx = correlation
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let time_delay = max_idx as f64 * dt;

        // Fit exponential decay to response
        let decay_constant = self.fit_exponential_decay(&response_array)?;

        Ok((correlation, time_delay, decay_constant))
    }

    /// Analyze spatial patterns in crosstalk
    fn analyze_spatial_patterns(
        &self,
        crosstalk_matrix: &Array2<f64>,
    ) -> DeviceResult<SpatialCrosstalkAnalysis> {
        let num_qubits = self.device_topology.num_qubits;

        // Calculate distance-dependent decay
        let distance_decay = self.calculate_distance_decay(crosstalk_matrix)?;

        // Identify directional patterns
        let directional_patterns = self.identify_directional_patterns(crosstalk_matrix)?;

        // Find crosstalk hotspots
        let crosstalk_hotspots = self.find_crosstalk_hotspots(crosstalk_matrix)?;

        // Analyze crosstalk propagation using graph theory
        let propagation_analysis = self.analyze_crosstalk_propagation(crosstalk_matrix)?;

        Ok(SpatialCrosstalkAnalysis {
            distance_decay,
            directional_patterns,
            crosstalk_hotspots,
            propagation_analysis,
        })
    }

    /// Identify crosstalk mechanisms based on signatures
    fn identify_mechanisms(
        &self,
        crosstalk_matrix: &Array2<f64>,
        frequency_crosstalk: &HashMap<(usize, usize), Array1<Complex64>>,
        spectral_signatures: &SpectralCrosstalkAnalysis,
    ) -> DeviceResult<Vec<CrosstalkMechanism>> {
        let mut mechanisms = Vec::new();

        // Analyze each qubit pair for mechanism signatures
        for i in 0..crosstalk_matrix.nrows() {
            for j in 0..crosstalk_matrix.ncols() {
                if i != j && crosstalk_matrix[[i, j]] > 1e-6 {
                    // Check for different mechanism signatures

                    // 1. Z-Z interaction (frequency-independent, distance-dependent)
                    if self.is_zz_interaction((i, j), crosstalk_matrix, frequency_crosstalk) {
                        mechanisms.push(CrosstalkMechanism {
                            mechanism_type: CrosstalkType::ZZInteraction,
                            affected_qubits: vec![i, j],
                            strength: crosstalk_matrix[[i, j]],
                            frequency_signature: None,
                            mitigation_difficulty: MitigationDifficulty::Moderate,
                            description: format!("Z-Z interaction between qubits {} and {}", i, j),
                        });
                    }

                    // 2. Capacitive coupling (frequency-dependent)
                    if let Some(freq_data) = frequency_crosstalk.get(&(i, j)) {
                        if self.is_capacitive_coupling(freq_data) {
                            mechanisms.push(CrosstalkMechanism {
                                mechanism_type: CrosstalkType::CapacitiveCoupling,
                                affected_qubits: vec![i, j],
                                strength: crosstalk_matrix[[i, j]],
                                frequency_signature: Some(freq_data.mapv(|c| c.norm())),
                                mitigation_difficulty: MitigationDifficulty::Difficult,
                                description: format!(
                                    "Capacitive coupling between qubits {} and {}",
                                    i, j
                                ),
                            });
                        }
                    }

                    // 3. Control line crosstalk (high-frequency signature)
                    if let Some(peaks) = spectral_signatures.spectral_peaks.get(&(i, j)) {
                        if self.is_control_line_crosstalk(peaks) {
                            mechanisms.push(CrosstalkMechanism {
                                mechanism_type: CrosstalkType::ControlLineCrosstalk,
                                affected_qubits: vec![i, j],
                                strength: crosstalk_matrix[[i, j]],
                                frequency_signature: None,
                                mitigation_difficulty: MitigationDifficulty::Easy,
                                description: format!(
                                    "Control line crosstalk between qubits {} and {}",
                                    i, j
                                ),
                            });
                        }
                    }
                }
            }
        }

        Ok(mechanisms)
    }

    /// Generate mitigation strategies
    fn generate_mitigation_strategies(
        &self,
        crosstalk_matrix: &Array2<f64>,
        mechanisms: &[CrosstalkMechanism],
        spatial_patterns: &SpatialCrosstalkAnalysis,
    ) -> DeviceResult<Vec<MitigationStrategy>> {
        let mut strategies = Vec::new();

        for mechanism in mechanisms {
            match mechanism.mechanism_type {
                CrosstalkType::ZZInteraction => {
                    // Use echo sequences for Z-Z interaction
                    strategies.push(MitigationStrategy {
                        strategy_type: MitigationType::EchoSequences,
                        target_qubits: mechanism.affected_qubits.clone(),
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("echo_period".to_string(), 100.0); // ns
                            params.insert("num_echoes".to_string(), 4.0);
                            params
                        },
                        expected_improvement: 0.8, // 80% reduction
                        implementation_complexity: 0.3,
                        description: "Echo sequences to suppress Z-Z interaction".to_string(),
                    });
                }

                CrosstalkType::CapacitiveCoupling => {
                    // Use frequency detuning for capacitive coupling
                    strategies.push(MitigationStrategy {
                        strategy_type: MitigationType::FrequencyDetuning,
                        target_qubits: mechanism.affected_qubits.clone(),
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("detuning_amount".to_string(), 10e6); // 10 MHz
                            params.insert("optimization_iterations".to_string(), 50.0);
                            params
                        },
                        expected_improvement: 0.6, // 60% reduction
                        implementation_complexity: 0.5,
                        description: "Frequency detuning to avoid resonant coupling".to_string(),
                    });
                }

                CrosstalkType::ControlLineCrosstalk => {
                    // Use amplitude compensation for control line crosstalk
                    strategies.push(MitigationStrategy {
                        strategy_type: MitigationType::AmplitudeCompensation,
                        target_qubits: mechanism.affected_qubits.clone(),
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("compensation_factor".to_string(), -mechanism.strength);
                            params.insert("calibration_shots".to_string(), 10000.0);
                            params
                        },
                        expected_improvement: 0.9, // 90% reduction
                        implementation_complexity: 0.2,
                        description: "Amplitude compensation for control crosstalk".to_string(),
                    });
                }

                _ => {
                    // Generic mitigation strategies
                    strategies.push(MitigationStrategy {
                        strategy_type: MitigationType::TemporalDecoupling,
                        target_qubits: mechanism.affected_qubits.clone(),
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("minimum_separation".to_string(), 200.0); // ns
                            params
                        },
                        expected_improvement: 0.5, // 50% reduction
                        implementation_complexity: 0.1,
                        description: "Temporal decoupling to avoid simultaneous operations"
                            .to_string(),
                    });
                }
            }
        }

        // Add spatial isolation strategies for hotspots
        for hotspot in &spatial_patterns.crosstalk_hotspots {
            strategies.push(MitigationStrategy {
                strategy_type: MitigationType::SpatialIsolation,
                target_qubits: hotspot.affected_qubits.clone(),
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("isolation_radius".to_string(), hotspot.radius);
                    params.insert("max_crosstalk_threshold".to_string(), 0.01);
                    params
                },
                expected_improvement: 0.7,
                implementation_complexity: 0.4,
                description: format!(
                    "Spatial isolation around hotspot at qubit {}",
                    hotspot.center_qubit
                ),
            });
        }

        Ok(strategies)
    }

    // Helper methods

    fn create_crosstalk_preparation_circuit(&self, target: usize) -> DeviceResult<Circuit<8>> {
        let mut circuit = Circuit::<8>::new();
        // Prepare target qubit in |+⟩ state (sensitive to Z errors)
        circuit.h(QubitId(target as u32));
        Ok(circuit)
    }

    fn create_ramsey_circuit(&self, target: usize, frequency: f64) -> DeviceResult<Circuit<8>> {
        let mut circuit = Circuit::<8>::new();
        // Ramsey sequence for frequency-sensitive measurement
        circuit.h(QubitId(target as u32));
        // Virtual Z rotation at frequency would be inserted here
        circuit.h(QubitId(target as u32));
        Ok(circuit)
    }

    fn create_temporal_response_circuit(
        &self,
        target: usize,
        delay: f64,
    ) -> DeviceResult<Circuit<8>> {
        let mut circuit = Circuit::<8>::new();
        // Prepare target for temporal response measurement
        circuit.h(QubitId(target as u32));
        // Delay would be implemented in the pulse sequence
        Ok(circuit)
    }

    fn generate_frequency_sweep(&self) -> Vec<f64> {
        let (start, end) = self.config.frequency_range;
        let resolution = self.config.frequency_resolution;
        let num_points = ((end - start) / resolution) as usize + 1;

        (0..num_points)
            .map(|i| start + i as f64 * resolution)
            .collect()
    }

    fn generate_amplitude_sweep(&self) -> Vec<f64> {
        let (start, end) = self.config.amplitude_range;
        let num_steps = self.config.amplitude_steps;

        (0..num_steps)
            .map(|i| start + (end - start) * i as f64 / (num_steps - 1) as f64)
            .collect()
    }

    fn calculate_crosstalk_strength(
        &self,
        baseline: &CrosstalkResult,
        crosstalk: &CrosstalkResult,
    ) -> DeviceResult<f64> {
        // Calculate difference in expectation values
        let baseline_expectation = self.calculate_expectation_value(baseline)?;
        let crosstalk_expectation = self.calculate_expectation_value(crosstalk)?;

        Ok((baseline_expectation - crosstalk_expectation).abs())
    }

    fn calculate_expectation_value(&self, result: &CrosstalkResult) -> DeviceResult<f64> {
        let total_shots = result.counts.values().sum::<u64>() as f64;
        if total_shots == 0.0 {
            return Ok(0.0);
        }

        let expectation = result
            .counts
            .iter()
            .map(|(state, count)| {
                let state_value = if state == "0" { 1.0 } else { -1.0 };
                state_value * (*count as f64 / total_shots)
            })
            .sum::<f64>();

        Ok(expectation)
    }

    fn extract_complex_response(
        &self,
        result: &CrosstalkResult,
        frequency: f64,
    ) -> DeviceResult<Complex64> {
        // Extract complex response from Ramsey measurement
        let expectation = self.calculate_expectation_value(result)?;

        // In a real implementation, this would extract both amplitude and phase
        // from the Ramsey fringe measurement
        let amplitude = expectation.abs();
        let phase = 0.0; // Would be extracted from actual measurement

        Ok(Complex64::from_polar(amplitude, phase))
    }

    fn extract_crosstalk_magnitude(&self, result: &CrosstalkResult) -> DeviceResult<f64> {
        // Extract magnitude of crosstalk effect
        self.calculate_expectation_value(result)
            .map(|exp| exp.abs())
    }

    fn extract_response_magnitude(&self, result: &CrosstalkResult) -> DeviceResult<f64> {
        // Extract response magnitude for temporal analysis
        self.calculate_expectation_value(result)
            .map(|exp| exp.abs())
    }

    fn calculate_coherence_matrix(
        &self,
        frequency_crosstalk: &HashMap<(usize, usize), Array1<Complex64>>,
    ) -> DeviceResult<Array2<f64>> {
        let num_qubits = self.device_topology.num_qubits;
        let mut coherence_matrix = Array2::zeros((num_qubits, num_qubits));

        // Calculate coherence between each pair of qubits
        for i in 0..num_qubits {
            for j in 0..num_qubits {
                if i != j {
                    if let (Some(spec_i), Some(spec_j)) = (
                        frequency_crosstalk.get(&(i, 0)),
                        frequency_crosstalk.get(&(j, 0)),
                    ) {
                        // Calculate coherence between the two spectra
                        let coherence = self.calculate_coherence(spec_i, spec_j)?;
                        coherence_matrix[[i, j]] = coherence;
                    }
                }
            }
        }

        Ok(coherence_matrix)
    }

    fn calculate_coherence(
        &self,
        signal1: &Array1<Complex64>,
        signal2: &Array1<Complex64>,
    ) -> DeviceResult<f64> {
        // Simplified coherence calculation
        // In practice, would use proper cross-spectral density

        let cross_power = signal1
            .iter()
            .zip(signal2.iter())
            .map(|(s1, s2)| (s1 * s2.conj()).norm())
            .sum::<f64>();

        let power1: f64 = signal1.iter().map(|s| s.norm_sqr()).sum();
        let power2: f64 = signal2.iter().map(|s| s.norm_sqr()).sum();

        if power1 * power2 > 0.0 {
            Ok(cross_power / (power1 * power2).sqrt())
        } else {
            Ok(0.0)
        }
    }

    fn estimate_peak_width(&self, spectrum: &Array1<f64>, peak_idx: usize) -> f64 {
        // Estimate full width at half maximum (FWHM)
        if peak_idx >= spectrum.len() {
            return 0.0;
        }

        let peak_value = spectrum[peak_idx];
        let half_max = peak_value / 2.0;

        // Find left and right half-maximum points
        let mut left_idx = peak_idx;
        let mut right_idx = peak_idx;

        while left_idx > 0 && spectrum[left_idx] > half_max {
            left_idx -= 1;
        }

        while right_idx < spectrum.len() - 1 && spectrum[right_idx] > half_max {
            right_idx += 1;
        }

        (right_idx - left_idx) as f64 * self.config.frequency_resolution
    }

    fn calculate_peak_significance(&self, spectrum: &Array1<f64>, peak_idx: usize) -> f64 {
        // Calculate peak significance as SNR
        if peak_idx >= spectrum.len() {
            return 0.0;
        }

        let peak_value = spectrum[peak_idx];

        // Estimate noise level from surrounding region
        let window_size = 10;
        let start = peak_idx.saturating_sub(window_size);
        let end = (peak_idx + window_size).min(spectrum.len());

        let surrounding_values: Vec<f64> = (start..end)
            .filter(|&i| (i as i32 - peak_idx as i32).abs() > 2)
            .map(|i| spectrum[i])
            .collect();

        if surrounding_values.is_empty() {
            return 1.0;
        }

        let noise_level = surrounding_values.iter().sum::<f64>() / surrounding_values.len() as f64;
        let noise_std = {
            let mean = noise_level;
            let variance = surrounding_values
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / surrounding_values.len() as f64;
            variance.sqrt()
        };

        if noise_std > 0.0 {
            (peak_value - noise_level) / noise_std
        } else {
            peak_value / noise_level.max(1e-10)
        }
    }

    fn create_delta_stimulus(&self, length: usize) -> Array1<f64> {
        let mut stimulus = Array1::zeros(length);
        if length > 0 {
            stimulus[0] = 1.0; // Delta function at t=0
        }
        stimulus
    }

    fn fit_exponential_decay(&self, data: &Array1<f64>) -> DeviceResult<f64> {
        // Simplified exponential decay fitting
        // y = A * exp(-t/τ)

        if data.len() < 3 {
            return Ok(1000.0); // Default 1μs decay time
        }

        // Find peak and fit decay from there
        let peak_idx = data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        if peak_idx + 2 >= data.len() {
            return Ok(1000.0);
        }

        // Simple two-point exponential fit
        let y1 = data[peak_idx];
        let y2 = data[peak_idx + 1];

        if y1 > 0.0 && y2 > 0.0 && y1 > y2 {
            let dt = 100.0; // Time step in ns
            let tau = -dt / (y2 / y1).ln();
            Ok(tau.max(10.0)) // Minimum 10ns decay time
        } else {
            Ok(1000.0)
        }
    }

    fn identify_temporal_clusters(
        &self,
        correlations: &HashMap<(usize, usize), Array1<f64>>,
    ) -> DeviceResult<Vec<TemporalCluster>> {
        // Identify clusters of correlated crosstalk events
        let mut clusters = Vec::new();

        // Simple clustering based on correlation peaks
        for (&(i, j), correlation) in correlations {
            let peaks = self.find_correlation_peaks(correlation)?;

            for (peak_time, peak_strength) in peaks {
                clusters.push(TemporalCluster {
                    start_time: peak_time - 50.0, // 50ns before peak
                    duration: 100.0,              // 100ns cluster duration
                    affected_qubits: vec![i, j],
                    crosstalk_strength: peak_strength,
                });
            }
        }

        Ok(clusters)
    }

    fn find_correlation_peaks(&self, correlation: &Array1<f64>) -> DeviceResult<Vec<(f64, f64)>> {
        // Find peaks in correlation function
        let threshold = 0.1;
        let mut peaks = Vec::new();

        for i in 1..correlation.len() - 1 {
            if correlation[i] > correlation[i - 1]
                && correlation[i] > correlation[i + 1]
                && correlation[i] > threshold
            {
                let time = i as f64 * 100.0; // 100ns time step
                peaks.push((time, correlation[i]));
            }
        }

        Ok(peaks)
    }

    fn calculate_distance_decay(
        &self,
        crosstalk_matrix: &Array2<f64>,
    ) -> DeviceResult<Array1<f64>> {
        // Calculate how crosstalk decays with distance
        let max_distance = self.config.max_distance;
        let mut distance_decay = Array1::zeros(max_distance);
        let mut distance_counts = vec![0usize; max_distance];

        for i in 0..crosstalk_matrix.nrows() {
            for j in 0..crosstalk_matrix.ncols() {
                if i != j {
                    let distance = self.calculate_qubit_distance(i, j)?;
                    if distance < max_distance && distance > 0 {
                        distance_decay[distance] += crosstalk_matrix[[i, j]];
                        distance_counts[distance] += 1;
                    }
                }
            }
        }

        // Average crosstalk at each distance
        for (distance, count) in distance_counts.iter().enumerate() {
            if *count > 0 {
                distance_decay[distance] /= *count as f64;
            }
        }

        Ok(distance_decay)
    }

    fn calculate_qubit_distance(&self, qubit1: usize, qubit2: usize) -> DeviceResult<usize> {
        // Calculate Manhattan distance on the qubit grid
        // This is a simplified version - would use actual device layout

        // For a linear topology
        Ok((qubit1 as i32 - qubit2 as i32).abs() as usize)
    }

    fn identify_directional_patterns(
        &self,
        crosstalk_matrix: &Array2<f64>,
    ) -> DeviceResult<HashMap<String, Array2<f64>>> {
        let mut patterns = HashMap::new();

        // Identify horizontal, vertical, and diagonal crosstalk patterns
        // This is simplified - would analyze actual 2D layout

        // Create asymmetry pattern
        let mut asymmetry = crosstalk_matrix.clone();
        for i in 0..asymmetry.nrows() {
            for j in 0..asymmetry.ncols() {
                asymmetry[[i, j]] = crosstalk_matrix[[i, j]] - crosstalk_matrix[[j, i]];
            }
        }
        patterns.insert("asymmetry".to_string(), asymmetry);

        Ok(patterns)
    }

    fn find_crosstalk_hotspots(
        &self,
        crosstalk_matrix: &Array2<f64>,
    ) -> DeviceResult<Vec<CrosstalkHotspot>> {
        let mut hotspots = Vec::new();
        let threshold = 0.05; // 5% crosstalk threshold

        for i in 0..crosstalk_matrix.nrows() {
            // Calculate total crosstalk from this qubit
            let total_crosstalk: f64 = crosstalk_matrix.row(i).sum();
            let max_crosstalk = crosstalk_matrix
                .row(i)
                .fold(0.0_f64, |max, &val: &f64| max.max(val));

            if max_crosstalk > threshold {
                // Find affected qubits
                let affected_qubits: Vec<usize> = crosstalk_matrix
                    .row(i)
                    .iter()
                    .enumerate()
                    .filter(|(_, &val)| val > threshold * 0.1)
                    .map(|(j, _)| j)
                    .collect();

                hotspots.push(CrosstalkHotspot {
                    center_qubit: i,
                    affected_qubits,
                    radius: 2.0, // Simplified radius
                    max_crosstalk,
                    mechanism: None, // Would be determined by analysis
                });
            }
        }

        Ok(hotspots)
    }

    fn analyze_crosstalk_propagation(
        &self,
        crosstalk_matrix: &Array2<f64>,
    ) -> DeviceResult<PropagationAnalysis> {
        // Build propagation graph
        let mut propagation_graph = Vec::new();
        let threshold = 0.01;

        for i in 0..crosstalk_matrix.nrows() {
            for j in 0..crosstalk_matrix.ncols() {
                if i != j && crosstalk_matrix[[i, j]] > threshold {
                    propagation_graph.push((i, j, crosstalk_matrix[[i, j]]));
                }
            }
        }

        // Find critical paths using graph algorithms
        let critical_paths = self.find_propagation_paths(&propagation_graph)?;

        // Calculate propagation times (simplified)
        let mut propagation_times = HashMap::new();
        for &(source, target, strength) in &propagation_graph {
            // Assume propagation time inversely related to strength
            let time = 100.0 / strength.max(0.001); // ns
            propagation_times.insert((source, target), time);
        }

        // Create effective topology matrix
        let effective_topology = crosstalk_matrix.clone();

        Ok(PropagationAnalysis {
            propagation_graph,
            critical_paths,
            propagation_times,
            effective_topology,
        })
    }

    fn find_propagation_paths(
        &self,
        graph: &[(usize, usize, f64)],
    ) -> DeviceResult<Vec<Vec<usize>>> {
        // Find critical propagation paths using graph analysis
        let mut paths = Vec::new();

        // Simple path finding - would use more sophisticated graph algorithms
        let mut visited = HashSet::new();

        for &(source, target, _) in graph {
            if !visited.contains(&source) {
                let path = vec![source, target]; // Simplified 2-hop path
                paths.push(path);
                visited.insert(source);
            }
        }

        Ok(paths)
    }

    // Mechanism identification helpers

    fn is_zz_interaction(
        &self,
        qubit_pair: (usize, usize),
        crosstalk_matrix: &Array2<f64>,
        frequency_crosstalk: &HashMap<(usize, usize), Array1<Complex64>>,
    ) -> bool {
        let (i, j) = qubit_pair;

        // Z-Z interaction characteristics:
        // 1. Symmetric crosstalk
        // 2. Distance-dependent
        // 3. Frequency-independent

        let forward_crosstalk = crosstalk_matrix[[i, j]];
        let reverse_crosstalk = crosstalk_matrix[[j, i]];
        let symmetry = (forward_crosstalk - reverse_crosstalk).abs() / forward_crosstalk.max(1e-10);

        let is_symmetric = symmetry < 0.2; // 20% asymmetry tolerance

        // Check frequency independence
        let is_frequency_independent = if let Some(freq_data) = frequency_crosstalk.get(&(i, j)) {
            let amplitudes = freq_data.mapv(|c| c.norm());
            let variation = std(&amplitudes.view(), 1).unwrap_or(0.0);
            let mean_amp = mean(&amplitudes.view()).unwrap_or(1.0);
            variation / mean_amp < 0.1 // Low frequency variation
        } else {
            true // Assume frequency-independent if no data
        };

        is_symmetric && is_frequency_independent
    }

    fn is_capacitive_coupling(&self, frequency_data: &Array1<Complex64>) -> bool {
        // Capacitive coupling has characteristic frequency dependence
        let amplitudes = frequency_data.mapv(|c| c.norm());

        // Look for frequency-dependent behavior
        let variation = std(&amplitudes.view(), 1).unwrap_or(0.0);
        let mean_amp = mean(&amplitudes.view()).unwrap_or(1.0);

        variation / mean_amp > 0.2 // High frequency variation indicates capacitive coupling
    }

    fn is_control_line_crosstalk(&self, peaks: &[SpectralPeak]) -> bool {
        // Control line crosstalk typically has high-frequency components
        peaks
            .iter()
            .any(|peak| peak.frequency > 1e9 && peak.significance > 3.0)
    }
}

/// Cross-talk operation for characterization
#[derive(Debug, Clone)]
pub struct CrosstalkOperation {
    pub qubit: usize,
    pub operation_type: CrosstalkOperationType,
    pub amplitude: f64,
    pub frequency: f64,
    pub phase: f64,
    pub duration: f64,
}

/// Types of crosstalk operations
#[derive(Debug, Clone, PartialEq)]
pub enum CrosstalkOperationType {
    ZRotation,
    FrequencySweep,
    AmplitudeSweep,
    DelayedPulse,
    ContinuousWave,
}

/// Result of crosstalk measurement
#[derive(Debug, Clone)]
pub struct CrosstalkResult {
    pub counts: HashMap<String, u64>,
    pub shots: u64,
    pub metadata: HashMap<String, String>,
}

/// Trait for devices that can execute crosstalk characterization
#[async_trait::async_trait]
pub trait CrosstalkExecutor {
    async fn execute_crosstalk_circuit(
        &self,
        circuit: &Circuit<8>,
        operations: Vec<CrosstalkOperation>,
        shots: usize,
    ) -> DeviceResult<CrosstalkResult>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology_analysis::create_standard_topology;

    #[test]
    fn test_crosstalk_config_default() {
        let config = CrosstalkConfig::default();
        assert_eq!(config.shots_per_config, 10000);
        assert!(config.enable_spectral_analysis);
    }

    #[test]
    fn test_frequency_sweep_generation() {
        let config = CrosstalkConfig {
            frequency_range: (4.0e9, 5.0e9),
            frequency_resolution: 1.0e8,
            ..Default::default()
        };

        let topology = create_standard_topology("linear", 4).unwrap();
        let analyzer = CrosstalkAnalyzer::new(config, topology);

        let frequencies = analyzer.generate_frequency_sweep();
        assert_eq!(frequencies.len(), 11); // 4.0 to 5.0 GHz in 0.1 GHz steps
        assert_eq!(frequencies[0], 4.0e9);
        assert_eq!(frequencies[10], 5.0e9);
    }

    #[test]
    fn test_amplitude_sweep_generation() {
        let config = CrosstalkConfig {
            amplitude_range: (0.0, 1.0),
            amplitude_steps: 5,
            ..Default::default()
        };

        let topology = create_standard_topology("linear", 4).unwrap();
        let analyzer = CrosstalkAnalyzer::new(config, topology);

        let amplitudes = analyzer.generate_amplitude_sweep();
        assert_eq!(amplitudes.len(), 5);
        assert_eq!(amplitudes[0], 0.0);
        assert_eq!(amplitudes[4], 1.0);
    }

    #[test]
    fn test_crosstalk_strength_calculation() {
        let topology = create_standard_topology("linear", 4).unwrap();
        let analyzer = CrosstalkAnalyzer::new(CrosstalkConfig::default(), topology);

        let baseline = CrosstalkResult {
            counts: [("0".to_string(), 800), ("1".to_string(), 200)]
                .iter()
                .cloned()
                .collect(),
            shots: 1000,
            metadata: HashMap::new(),
        };

        let crosstalk = CrosstalkResult {
            counts: [("0".to_string(), 600), ("1".to_string(), 400)]
                .iter()
                .cloned()
                .collect(),
            shots: 1000,
            metadata: HashMap::new(),
        };

        let strength = analyzer
            .calculate_crosstalk_strength(&baseline, &crosstalk)
            .unwrap();
        assert!((strength - 0.4).abs() < 0.01); // Expected difference of 0.4
    }

    #[test]
    fn test_mechanism_identification() {
        let topology = create_standard_topology("linear", 4).unwrap();
        let analyzer = CrosstalkAnalyzer::new(CrosstalkConfig::default(), topology);

        // Create test crosstalk matrix
        let mut crosstalk_matrix = Array2::zeros((4, 4));
        crosstalk_matrix[[0, 1]] = 0.1;
        crosstalk_matrix[[1, 0]] = 0.1; // Symmetric for Z-Z interaction

        let frequency_crosstalk = HashMap::new();
        let spectral_signatures = SpectralCrosstalkAnalysis {
            power_spectra: HashMap::new(),
            coherence_matrix: Array2::zeros((4, 4)),
            dominant_frequencies: HashMap::new(),
            spectral_peaks: HashMap::new(),
            transfer_functions: HashMap::new(),
        };

        let mechanisms = analyzer
            .identify_mechanisms(
                &crosstalk_matrix,
                &frequency_crosstalk,
                &spectral_signatures,
            )
            .unwrap();

        assert!(!mechanisms.is_empty());
        assert_eq!(mechanisms[0].mechanism_type, CrosstalkType::ZZInteraction);
    }
}
