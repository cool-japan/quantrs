//! Comprehensive tests for advanced error mitigation and calibration.

use quantrs2_tytan::advanced_error_mitigation::*;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

#[test]
fn test_error_mitigation_manager_creation() {
    let config = ErrorMitigationConfig::default();
    let manager = AdvancedErrorMitigationManager::new(config);
    
    // Test that manager is created successfully
    assert!(true);
}

#[test]
fn test_default_error_mitigation_config() {
    let config = ErrorMitigationConfig::default();
    
    // Verify default configuration values
    assert!(config.real_time_monitoring);
    assert!(config.adaptive_protocols);
    assert!(config.device_calibration);
    assert!(config.syndrome_prediction);
    assert!(config.qec_integration);
    assert_eq!(config.monitoring_interval, Duration::from_millis(100));
    assert_eq!(config.calibration_interval, Duration::from_secs(3600));
    assert_eq!(config.noise_update_threshold, 0.05);
    assert_eq!(config.mitigation_threshold, 0.1);
    assert_eq!(config.history_retention, Duration::from_secs(24 * 3600));
}

#[test]
fn test_noise_characterization_config() {
    let config = NoiseCharacterizationConfig {
        sampling_frequency: 1000.0,
        benchmarking_sequences: 100,
        tomography_protocol: TomographyProtocol::MaximumLikelihood,
        spectroscopy_config: SpectroscopyConfig {
            frequency_range: (1e-6, 1e6),
            frequency_points: 1000,
            measurement_time: Duration::from_millis(1),
            processing_window: ProcessingWindow::Hanning,
        },
        history_length: 1000,
    };
    
    // Test configuration values
    assert_eq!(config.sampling_frequency, 1000.0);
    assert_eq!(config.benchmarking_sequences, 100);
    assert!(matches!(config.tomography_protocol, TomographyProtocol::MaximumLikelihood));
    assert_eq!(config.spectroscopy_config.frequency_points, 1000);
    assert_eq!(config.history_length, 1000);
}

#[test]
fn test_noise_model_creation() {
    let noise_model = NoiseModel::default();
    
    // Verify default noise model structure
    assert!(noise_model.single_qubit_errors.is_empty());
    assert!(noise_model.two_qubit_errors.is_empty());
    assert_eq!(noise_model.crosstalk_matrix.dim(), (1, 1));
    assert_eq!(noise_model.validation_score, 0.0);
}

#[test]
fn test_single_qubit_error_model() {
    let error_model = SingleQubitErrorModel {
        depolarizing_rate: 0.001,
        dephasing_rates: DephasingRates {
            t1: 100e-6,
            t2: 50e-6,
            t2_star: 30e-6,
            t2_echo: 70e-6,
        },
        amplitude_damping_rate: 0.0005,
        phase_damping_rate: 0.0003,
        thermal_population: 0.01,
        gate_errors: HashMap::new(),
    };
    
    // Verify error model structure
    assert_eq!(error_model.depolarizing_rate, 0.001);
    assert_eq!(error_model.dephasing_rates.t1, 100e-6);
    assert_eq!(error_model.dephasing_rates.t2, 50e-6);
    assert_eq!(error_model.amplitude_damping_rate, 0.0005);
    assert_eq!(error_model.phase_damping_rate, 0.0003);
    assert_eq!(error_model.thermal_population, 0.01);
    assert!(error_model.gate_errors.is_empty());
}

#[test]
fn test_two_qubit_error_model() {
    let error_model = TwoQubitErrorModel {
        entangling_error: 0.01,
        crosstalk_strength: 0.02,
        zz_coupling: 1e-6,
        conditional_phase_error: 0.005,
        gate_time_variation: 0.1,
    };
    
    // Verify two-qubit error model
    assert_eq!(error_model.entangling_error, 0.01);
    assert_eq!(error_model.crosstalk_strength, 0.02);
    assert_eq!(error_model.zz_coupling, 1e-6);
    assert_eq!(error_model.conditional_phase_error, 0.005);
    assert_eq!(error_model.gate_time_variation, 0.1);
}

#[test]
fn test_gate_error_model() {
    let gate_error = GateErrorModel {
        error_probability: 0.001,
        coherent_error_angle: 0.01,
        incoherent_components: Array1::from(vec![0.0005, 0.0003, 0.0002]),
        leakage_probability: 0.0001,
    };
    
    // Verify gate error model
    assert_eq!(gate_error.error_probability, 0.001);
    assert_eq!(gate_error.coherent_error_angle, 0.01);
    assert_eq!(gate_error.incoherent_components.len(), 3);
    assert_eq!(gate_error.leakage_probability, 0.0001);
}

#[test]
fn test_temporal_correlation_model() {
    let correlation_model = TemporalCorrelationModel {
        autocorrelation: Array1::from(vec![1.0, 0.8, 0.6, 0.4, 0.2]),
        power_spectrum: Array1::from(vec![1.0, 0.5, 0.25, 0.125, 0.0625]),
        timescales: vec![1e-6, 1e-5, 1e-4, 1e-3],
        one_over_f_params: OneOverFParameters {
            amplitude: 0.1,
            exponent: 1.0,
            cutoff_frequency: 1e3,
            high_freq_rolloff: 0.01,
        },
    };
    
    // Verify temporal correlation model
    assert_eq!(correlation_model.autocorrelation.len(), 5);
    assert_eq!(correlation_model.power_spectrum.len(), 5);
    assert_eq!(correlation_model.timescales.len(), 4);
    assert_eq!(correlation_model.one_over_f_params.amplitude, 0.1);
    assert_eq!(correlation_model.one_over_f_params.exponent, 1.0);
}

#[test]
fn test_environmental_noise_model() {
    let mut control_line_noise = HashMap::new();
    control_line_noise.insert("control_1".to_string(), 0.001);
    control_line_noise.insert("control_2".to_string(), 0.0005);
    
    let env_noise = EnvironmentalNoiseModel {
        temperature_noise: 0.01,
        magnetic_noise: 0.005,
        electric_noise: 0.003,
        vibration_sensitivity: 0.002,
        control_line_noise,
    };
    
    // Verify environmental noise model
    assert_eq!(env_noise.temperature_noise, 0.01);
    assert_eq!(env_noise.magnetic_noise, 0.005);
    assert_eq!(env_noise.electric_noise, 0.003);
    assert_eq!(env_noise.vibration_sensitivity, 0.002);
    assert_eq!(env_noise.control_line_noise.len(), 2);
    assert_eq!(env_noise.control_line_noise["control_1"], 0.001);
}

#[test]
fn test_mitigation_protocol_types() {
    // Test Zero Noise Extrapolation
    let zne = MitigationProtocol::ZeroNoiseExtrapolation {
        scaling_factors: vec![1.0, 3.0, 5.0, 7.0],
        extrapolation_method: ExtrapolationMethod::Linear,
    };
    
    match zne {
        MitigationProtocol::ZeroNoiseExtrapolation { scaling_factors, extrapolation_method } => {
            assert_eq!(scaling_factors.len(), 4);
            assert!(matches!(extrapolation_method, ExtrapolationMethod::Linear));
        }
        _ => panic!("Unexpected protocol type"),
    }
    
    // Test Probabilistic Error Cancellation
    let pec = MitigationProtocol::ProbabilisticErrorCancellation {
        inverse_map: HashMap::new(),
        sampling_overhead: 2.5,
    };
    
    match pec {
        MitigationProtocol::ProbabilisticErrorCancellation { inverse_map, sampling_overhead } => {
            assert!(inverse_map.is_empty());
            assert_eq!(sampling_overhead, 2.5);
        }
        _ => panic!("Unexpected protocol type"),
    }
    
    // Test Symmetry Verification
    let sv = MitigationProtocol::SymmetryVerification {
        symmetries: vec![],
        verification_threshold: 0.95,
    };
    
    match sv {
        MitigationProtocol::SymmetryVerification { symmetries, verification_threshold } => {
            assert!(symmetries.is_empty());
            assert_eq!(verification_threshold, 0.95);
        }
        _ => panic!("Unexpected protocol type"),
    }
}

#[test]
fn test_extrapolation_methods() {
    let methods = vec![
        ExtrapolationMethod::Linear,
        ExtrapolationMethod::Polynomial { degree: 3 },
        ExtrapolationMethod::Exponential,
        ExtrapolationMethod::Richardson,
        ExtrapolationMethod::AdaptivePolynomial,
    ];
    
    // Test each extrapolation method
    for method in methods {
        match method {
            ExtrapolationMethod::Linear => assert!(true),
            ExtrapolationMethod::Polynomial { degree } => assert_eq!(degree, 3),
            ExtrapolationMethod::Exponential => assert!(true),
            ExtrapolationMethod::Richardson => assert!(true),
            ExtrapolationMethod::AdaptivePolynomial => assert!(true),
        }
    }
}

#[test]
fn test_symmetry_operator() {
    let operator = SymmetryOperator {
        name: "Parity".to_string(),
        operator: Array2::eye(4),
        eigenvalues: Array1::from(vec![1.0, -1.0, 1.0, -1.0]),
        symmetry_type: SymmetryType::Parity,
    };
    
    // Verify symmetry operator
    assert_eq!(operator.name, "Parity");
    assert_eq!(operator.operator.dim(), (4, 4));
    assert_eq!(operator.eigenvalues.len(), 4);
    assert!(matches!(operator.symmetry_type, SymmetryType::Parity));
}

#[test]
fn test_calibration_config() {
    let config = CalibrationConfig {
        auto_calibration: true,
        calibration_frequency: Duration::from_secs(3600),
        drift_threshold: 0.01,
        precision_targets: {
            let mut targets = HashMap::new();
            targets.insert("gate_fidelity".to_string(), 0.999);
            targets.insert("readout_fidelity".to_string(), 0.95);
            targets
        },
        calibration_timeout: Duration::from_secs(300),
    };
    
    // Verify calibration configuration
    assert!(config.auto_calibration);
    assert_eq!(config.calibration_frequency, Duration::from_secs(3600));
    assert_eq!(config.drift_threshold, 0.01);
    assert_eq!(config.precision_targets.len(), 2);
    assert_eq!(config.precision_targets["gate_fidelity"], 0.999);
    assert_eq!(config.calibration_timeout, Duration::from_secs(300));
}

#[test]
fn test_calibration_result() {
    let mut updated_params = HashMap::new();
    updated_params.insert("frequency".to_string(), 5.0e9);
    updated_params.insert("amplitude".to_string(), 0.5);
    
    let mut achieved_precision = HashMap::new();
    achieved_precision.insert("gate_fidelity".to_string(), 0.998);
    achieved_precision.insert("readout_fidelity".to_string(), 0.94);
    
    let result = CalibrationResult {
        success: true,
        updated_parameters: updated_params,
        achieved_precision,
        duration: Duration::from_secs(120),
        quality_metrics: CalibrationQualityMetrics {
            fidelity_improvement: 0.01,
            parameter_stability: 0.95,
            convergence_rate: 0.1,
            reproducibility: 0.98,
        },
        recommendations: vec![],
    };
    
    // Verify calibration result
    assert!(result.success);
    assert_eq!(result.updated_parameters.len(), 2);
    assert_eq!(result.achieved_precision.len(), 2);
    assert_eq!(result.duration, Duration::from_secs(120));
    assert_eq!(result.quality_metrics.fidelity_improvement, 0.01);
    assert_eq!(result.quality_metrics.parameter_stability, 0.95);
}

#[test]
fn test_syndrome_prediction_config() {
    let config = SyndromePredictionConfig {
        prediction_horizon: Duration::from_secs(300),
        model_update_frequency: Duration::from_secs(1800),
        confidence_threshold: 0.8,
        pattern_history_length: 1000,
        learning_params: LearningParameters {
            learning_rate: 0.001,
            batch_size: 32,
            regularization: 0.01,
            dropout_rate: 0.1,
            hidden_layers: vec![64, 32],
        },
    };
    
    // Verify syndrome prediction configuration
    assert_eq!(config.prediction_horizon, Duration::from_secs(300));
    assert_eq!(config.model_update_frequency, Duration::from_secs(1800));
    assert_eq!(config.confidence_threshold, 0.8);
    assert_eq!(config.pattern_history_length, 1000);
    assert_eq!(config.learning_params.learning_rate, 0.001);
    assert_eq!(config.learning_params.batch_size, 32);
    assert_eq!(config.learning_params.hidden_layers.len(), 2);
}

#[test]
fn test_qec_integration_config() {
    let config = QECIntegrationConfig {
        real_time_decoding: true,
        adaptive_code_selection: true,
        performance_optimization: true,
        error_threshold: 0.01,
        code_switching_threshold: 0.05,
    };
    
    // Verify QEC integration configuration
    assert!(config.real_time_decoding);
    assert!(config.adaptive_code_selection);
    assert!(config.performance_optimization);
    assert_eq!(config.error_threshold, 0.01);
    assert_eq!(config.code_switching_threshold, 0.05);
}

#[test]
fn test_noise_characterizer_creation() {
    let config = ErrorMitigationConfig::default();
    let characterizer = NoiseCharacterizer::new(&config);
    
    // Test that characterizer is created successfully
    assert!(characterizer.history.is_empty());
    assert_eq!(characterizer.config.sampling_frequency, 1000.0);
    assert_eq!(characterizer.config.benchmarking_sequences, 100);
}

#[test]
fn test_randomized_benchmarking_result() {
    let rb_result = RandomizedBenchmarkingResult {
        error_rate_per_clifford: 0.001,
        confidence_interval: (0.0008, 0.0012),
        fitting_quality: 0.95,
        sequence_fidelities: {
            let mut fidelities = HashMap::new();
            fidelities.insert(1, 0.999);
            fidelities.insert(10, 0.990);
            fidelities.insert(100, 0.900);
            fidelities
        },
    };
    
    // Verify randomized benchmarking result
    assert_eq!(rb_result.error_rate_per_clifford, 0.001);
    assert_eq!(rb_result.confidence_interval.0, 0.0008);
    assert_eq!(rb_result.confidence_interval.1, 0.0012);
    assert_eq!(rb_result.fitting_quality, 0.95);
    assert_eq!(rb_result.sequence_fidelities.len(), 3);
    assert_eq!(rb_result.sequence_fidelities[&1], 0.999);
}

#[test]
fn test_spectroscopy_data() {
    let spectroscopy_data = SpectroscopyData {
        frequencies: Array1::from(vec![1e3, 1e4, 1e5, 1e6]),
        signals: Array1::from(vec![0.1, 0.2, 0.15, 0.05]),
        power_spectrum: Array1::from(vec![0.01, 0.04, 0.0225, 0.0025]),
        noise_sources: vec![
            NoiseSource {
                source_type: NoiseSourceType::OneOverFNoise,
                frequency: 1e4,
                amplitude: 0.1,
                phase: 0.0,
                bandwidth: 1e3,
            },
            NoiseSource {
                source_type: NoiseSourceType::ChargeNoise,
                frequency: 1e5,
                amplitude: 0.05,
                phase: PI / 4.0,
                bandwidth: 5e3,
            },
        ],
    };
    
    // Verify spectroscopy data
    assert_eq!(spectroscopy_data.frequencies.len(), 4);
    assert_eq!(spectroscopy_data.signals.len(), 4);
    assert_eq!(spectroscopy_data.power_spectrum.len(), 4);
    assert_eq!(spectroscopy_data.noise_sources.len(), 2);
    assert!(matches!(spectroscopy_data.noise_sources[0].source_type, NoiseSourceType::OneOverFNoise));
    assert!(matches!(spectroscopy_data.noise_sources[1].source_type, NoiseSourceType::ChargeNoise));
}

#[test]
fn test_confidence_intervals() {
    let mut gate_fidelities = HashMap::new();
    gate_fidelities.insert("X".to_string(), (0.998, 0.9995));
    gate_fidelities.insert("CNOT".to_string(), (0.985, 0.995));
    
    let mut error_rates = HashMap::new();
    error_rates.insert("X".to_string(), (0.0005, 0.002));
    error_rates.insert("CNOT".to_string(), (0.005, 0.015));
    
    let intervals = ConfidenceIntervals {
        process_fidelity: (0.95, 0.98),
        gate_fidelities,
        error_rates,
        coherence_times: HashMap::new(),
    };
    
    // Verify confidence intervals
    assert_eq!(intervals.process_fidelity.0, 0.95);
    assert_eq!(intervals.process_fidelity.1, 0.98);
    assert_eq!(intervals.gate_fidelities.len(), 2);
    assert_eq!(intervals.error_rates.len(), 2);
    assert_eq!(intervals.gate_fidelities["X"].0, 0.998);
    assert_eq!(intervals.error_rates["CNOT"].1, 0.015);
}

#[test]
fn test_characterization_quality() {
    let quality = CharacterizationQuality {
        overall_score: 0.9,
        statistical_significance: 0.95,
        model_validation_score: 0.85,
        cross_validation_results: vec![0.8, 0.85, 0.9, 0.88, 0.92],
        residual_analysis: ResidualAnalysis {
            mean_residual: 0.001,
            residual_variance: 0.0001,
            normality_test_p_value: 0.05,
            autocorrelation_coefficients: Array1::from(vec![1.0, 0.1, 0.05, 0.02]),
        },
    };
    
    // Verify characterization quality
    assert_eq!(quality.overall_score, 0.9);
    assert_eq!(quality.statistical_significance, 0.95);
    assert_eq!(quality.model_validation_score, 0.85);
    assert_eq!(quality.cross_validation_results.len(), 5);
    assert_eq!(quality.residual_analysis.mean_residual, 0.001);
    assert_eq!(quality.residual_analysis.autocorrelation_coefficients.len(), 4);
}

#[test]
fn test_adaptive_mitigation_config() {
    let config = AdaptiveMitigationConfig {
        update_frequency: Duration::from_secs(300),
        monitoring_window: Duration::from_secs(3600),
        adaptation_threshold: 0.05,
        max_active_protocols: 3,
        learning_rate: 0.01,
    };
    
    // Verify adaptive mitigation configuration
    assert_eq!(config.update_frequency, Duration::from_secs(300));
    assert_eq!(config.monitoring_window, Duration::from_secs(3600));
    assert_eq!(config.adaptation_threshold, 0.05);
    assert_eq!(config.max_active_protocols, 3);
    assert_eq!(config.learning_rate, 0.01);
}

#[test]
fn test_mitigated_result() {
    let result = MitigatedResult {
        original_result: Array1::from(vec![0.1, 0.3, 0.4, 0.2]),
        mitigated_result: Array1::from(vec![0.05, 0.25, 0.5, 0.2]),
        mitigation_overhead: 2.0,
        confidence: 0.9,
    };
    
    // Verify mitigated result
    assert_eq!(result.original_result.len(), 4);
    assert_eq!(result.mitigated_result.len(), 4);
    assert_eq!(result.mitigation_overhead, 2.0);
    assert_eq!(result.confidence, 0.9);
    assert_eq!(result.original_result[0], 0.1);
    assert_eq!(result.mitigated_result[2], 0.5);
}

#[test]
fn test_device_parameters() {
    let params = DeviceParameters {
        qubit_frequencies: Array1::from(vec![5.0e9, 5.1e9, 4.9e9, 5.05e9]),
        coupling_strengths: Array2::from_shape_vec((4, 4), vec![
            0.0, 1e6, 0.0, 0.0,
            1e6, 0.0, 1e6, 0.0,
            0.0, 1e6, 0.0, 1e6,
            0.0, 0.0, 1e6, 0.0,
        ]).unwrap(),
        gate_times: {
            let mut times = HashMap::new();
            times.insert("X".to_string(), 20e-9);
            times.insert("CNOT".to_string(), 40e-9);
            times
        },
        readout_fidelities: Array1::from(vec![0.95, 0.96, 0.94, 0.97]),
    };
    
    // Verify device parameters
    assert_eq!(params.qubit_frequencies.len(), 4);
    assert_eq!(params.coupling_strengths.dim(), (4, 4));
    assert_eq!(params.gate_times.len(), 2);
    assert_eq!(params.readout_fidelities.len(), 4);
    assert_eq!(params.qubit_frequencies[0], 5.0e9);
    assert_eq!(params.gate_times["X"], 20e-9);
    assert_eq!(params.readout_fidelities[3], 0.97);
}

#[test]
fn test_error_mitigation_manager_start_monitoring() {
    let config = ErrorMitigationConfig::default();
    let mut manager = AdvancedErrorMitigationManager::new(config);
    
    // Test starting monitoring
    let result = manager.start_monitoring();
    assert!(result.is_ok());
}

#[test]
fn test_error_mitigation_manager_with_disabled_features() {
    let config = ErrorMitigationConfig {
        real_time_monitoring: false,
        adaptive_protocols: false,
        device_calibration: false,
        syndrome_prediction: false,
        qec_integration: false,
        ..Default::default()
    };
    
    let mut manager = AdvancedErrorMitigationManager::new(config);
    
    // Starting monitoring should fail when disabled
    let result = manager.start_monitoring();
    assert!(result.is_err());
    
    match result {
        Err(MitigationError::InvalidParameters(msg)) => {
            assert!(msg.contains("Real-time monitoring is disabled"));
        }
        _ => panic!("Expected InvalidParameters error"),
    }
}

#[test]
fn test_error_mitigation_manager_apply_mitigation() {
    let config = ErrorMitigationConfig::default();
    let manager = AdvancedErrorMitigationManager::new(config);
    
    // Create a test quantum circuit
    let circuit = vec!["X 0".to_string(), "CNOT 0 1".to_string(), "H 1".to_string()];
    
    // Test applying mitigation
    let result = manager.apply_mitigation(&circuit);
    assert!(result.is_ok());
    
    let mitigated_result = result.unwrap();
    assert!(mitigated_result.mitigation_overhead >= 1.0);
    assert!(mitigated_result.confidence >= 0.0);
    assert!(mitigated_result.confidence <= 1.0);
}

#[test]
fn test_qec_integration_result() {
    let result = QECIntegrationResult {
        logical_circuit: vec!["Logical_X 0".to_string()],
        physical_circuit: vec!["X 0".to_string(), "X 1".to_string(), "X 2".to_string()],
        decoding_schedule: vec!["Decode syndrome".to_string()],
        resource_overhead: 5.0,
        expected_logical_error_rate: 1e-6,
    };
    
    // Verify QEC integration result
    assert_eq!(result.logical_circuit.len(), 1);
    assert_eq!(result.physical_circuit.len(), 3);
    assert_eq!(result.decoding_schedule.len(), 1);
    assert_eq!(result.resource_overhead, 5.0);
    assert_eq!(result.expected_logical_error_rate, 1e-6);
}

#[test]
fn test_syndrome_prediction() {
    use std::time::SystemTime;
    
    let prediction = SyndromePrediction {
        predicted_syndrome: Array1::from(vec![1, 0, 1, 0]),
        confidence: 0.85,
        time_to_occurrence: Duration::from_millis(100),
        mitigation_recommendation: "Apply X correction on qubits 0 and 2".to_string(),
    };
    
    // Verify syndrome prediction
    assert_eq!(prediction.predicted_syndrome.len(), 4);
    assert_eq!(prediction.confidence, 0.85);
    assert_eq!(prediction.time_to_occurrence, Duration::from_millis(100));
    assert!(!prediction.mitigation_recommendation.is_empty());
    assert_eq!(prediction.predicted_syndrome[0], 1);
    assert_eq!(prediction.predicted_syndrome[1], 0);
}

#[test]
fn test_noise_source_types() {
    let noise_sources = vec![
        NoiseSourceType::WhiteNoise,
        NoiseSourceType::OneOverFNoise,
        NoiseSourceType::RTS,
        NoiseSourceType::PeriodicDrift,
        NoiseSourceType::ThermalFluctuations,
        NoiseSourceType::ChargeNoise,
        NoiseSourceType::FluxNoise,
        NoiseSourceType::InstrumentNoise,
    ];
    
    // Test that all noise source types can be created
    assert_eq!(noise_sources.len(), 8);
    for noise_type in noise_sources {
        match noise_type {
            NoiseSourceType::WhiteNoise => assert!(true),
            NoiseSourceType::OneOverFNoise => assert!(true),
            NoiseSourceType::RTS => assert!(true),
            NoiseSourceType::PeriodicDrift => assert!(true),
            NoiseSourceType::ThermalFluctuations => assert!(true),
            NoiseSourceType::ChargeNoise => assert!(true),
            NoiseSourceType::FluxNoise => assert!(true),
            NoiseSourceType::InstrumentNoise => assert!(true),
        }
    }
}

#[test]
fn test_processing_window_types() {
    let windows = vec![
        ProcessingWindow::Hanning,
        ProcessingWindow::Blackman,
        ProcessingWindow::Kaiser { beta: 2.5 },
        ProcessingWindow::Gaussian { sigma: 1.0 },
        ProcessingWindow::Rectangular,
    ];
    
    // Test that all processing window types can be created
    assert_eq!(windows.len(), 5);
    for window in windows {
        match window {
            ProcessingWindow::Hanning => assert!(true),
            ProcessingWindow::Blackman => assert!(true),
            ProcessingWindow::Kaiser { beta } => assert_eq!(beta, 2.5),
            ProcessingWindow::Gaussian { sigma } => assert_eq!(sigma, 1.0),
            ProcessingWindow::Rectangular => assert!(true),
        }
    }
}

#[test]
fn test_tomography_protocols() {
    let protocols = vec![
        TomographyProtocol::StandardProcessTomography,
        TomographyProtocol::CompressedSensing,
        TomographyProtocol::BayesianInference,
        TomographyProtocol::MaximumLikelihood,
        TomographyProtocol::LinearInversion,
    ];
    
    // Test that all tomography protocols can be created
    assert_eq!(protocols.len(), 5);
    for protocol in protocols {
        match protocol {
            TomographyProtocol::StandardProcessTomography => assert!(true),
            TomographyProtocol::CompressedSensing => assert!(true),
            TomographyProtocol::BayesianInference => assert!(true),
            TomographyProtocol::MaximumLikelihood => assert!(true),
            TomographyProtocol::LinearInversion => assert!(true),
        }
    }
}

#[test]
fn test_create_advanced_error_mitigation_manager() {
    let manager = create_advanced_error_mitigation_manager();
    let status = manager.get_status();
    
    // Test manager creation via helper function
    assert!(matches!(status.calibration_status.overall_status, CalibrationOverallStatus::Good));
    assert_eq!(status.error_statistics.total_errors_detected, 0);
    assert!(status.current_noise_model.single_qubit_errors.is_empty());
}

#[test]
fn test_create_lightweight_error_mitigation_manager() {
    let manager = create_lightweight_error_mitigation_manager();
    let status = manager.get_status();
    
    // Test lightweight manager creation
    assert!(matches!(status.calibration_status.overall_status, CalibrationOverallStatus::Good));
    assert_eq!(status.error_statistics.total_errors_detected, 0);
}

#[test]
fn test_error_types() {
    // Test various error types
    let errors = vec![
        MitigationError::NoiseCharacterizationFailed("Test error".to_string()),
        MitigationError::ProtocolApplicationFailed("Test error".to_string()),
        MitigationError::CalibrationFailed("Test error".to_string()),
        MitigationError::PredictionFailed("Test error".to_string()),
        MitigationError::QECError("Test error".to_string()),
        MitigationError::InvalidParameters("Test error".to_string()),
        MitigationError::InsufficientData("Test error".to_string()),
        MitigationError::ComputationTimeout("Test error".to_string()),
    ];
    
    // Verify error types can be created and matched
    assert_eq!(errors.len(), 8);
    for error in errors {
        match error {
            MitigationError::NoiseCharacterizationFailed(msg) => assert_eq!(msg, "Test error"),
            MitigationError::ProtocolApplicationFailed(msg) => assert_eq!(msg, "Test error"),
            MitigationError::CalibrationFailed(msg) => assert_eq!(msg, "Test error"),
            MitigationError::PredictionFailed(msg) => assert_eq!(msg, "Test error"),
            MitigationError::QECError(msg) => assert_eq!(msg, "Test error"),
            MitigationError::InvalidParameters(msg) => assert_eq!(msg, "Test error"),
            MitigationError::InsufficientData(msg) => assert_eq!(msg, "Test error"),
            MitigationError::ComputationTimeout(msg) => assert_eq!(msg, "Test error"),
        }
    }
}

#[test]
fn test_calibration_status_types() {
    let statuses = vec![
        CalibrationOverallStatus::Excellent,
        CalibrationOverallStatus::Good,
        CalibrationOverallStatus::Degraded,
        CalibrationOverallStatus::Poor,
        CalibrationOverallStatus::CalibrationRequired,
    ];
    
    // Test calibration status types
    assert_eq!(statuses.len(), 5);
    for status in statuses {
        match status {
            CalibrationOverallStatus::Excellent => assert!(true),
            CalibrationOverallStatus::Good => assert!(true),
            CalibrationOverallStatus::Degraded => assert!(true),
            CalibrationOverallStatus::Poor => assert!(true),
            CalibrationOverallStatus::CalibrationRequired => assert!(true),
        }
    }
}

#[test]
fn test_parameter_status_types() {
    let statuses = vec![
        ParameterStatus::InTolerance,
        ParameterStatus::NearLimit,
        ParameterStatus::OutOfTolerance,
        ParameterStatus::Drifting,
        ParameterStatus::Unstable,
    ];
    
    // Test parameter status types
    assert_eq!(statuses.len(), 5);
    for status in statuses {
        match status {
            ParameterStatus::InTolerance => assert!(true),
            ParameterStatus::NearLimit => assert!(true),
            ParameterStatus::OutOfTolerance => assert!(true),
            ParameterStatus::Drifting => assert!(true),
            ParameterStatus::Unstable => assert!(true),
        }
    }
}

#[test]
fn test_comprehensive_noise_characterization_workflow() {
    let config = ErrorMitigationConfig::default();
    let mut manager = AdvancedErrorMitigationManager::new(config);
    
    // Create a test quantum device
    let mut device = HashMap::new();
    device.insert("qubit_count".to_string(), 4.0);
    device.insert("frequency_0".to_string(), 5.0e9);
    device.insert("frequency_1".to_string(), 5.1e9);
    
    // Perform error characterization
    let result = manager.characterize_errors(&device);
    assert!(result.is_ok());
    
    let characterization_result = result.unwrap();
    
    // Verify characterization result structure
    assert!(characterization_result.noise_model.validation_score >= 0.0);
    assert!(characterization_result.benchmarking_results.error_rate_per_clifford >= 0.0);
    assert!(characterization_result.confidence_intervals.process_fidelity.0 <= characterization_result.confidence_intervals.process_fidelity.1);
    assert!(characterization_result.quality_metrics.overall_score >= 0.0);
    assert!(characterization_result.quality_metrics.overall_score <= 1.0);
    
    println!("Noise characterization completed successfully");
    println!("Process fidelity: {:.3} - {:.3}", 
             characterization_result.confidence_intervals.process_fidelity.0,
             characterization_result.confidence_intervals.process_fidelity.1);
    println!("Error rate per Clifford: {:.6}", characterization_result.benchmarking_results.error_rate_per_clifford);
    println!("Overall quality score: {:.3}", characterization_result.quality_metrics.overall_score);
}

#[test]
fn test_integrated_error_mitigation_workflow() {
    let config = ErrorMitigationConfig::default();
    let mut manager = AdvancedErrorMitigationManager::new(config);
    
    // Create test circuit
    let circuit = vec![
        "H 0".to_string(),
        "CNOT 0 1".to_string(),
        "RZ 0.5 1".to_string(),
        "CNOT 0 1".to_string(),
        "H 0".to_string(),
    ];
    
    // Apply mitigation
    let mitigation_result = manager.apply_mitigation(&circuit);
    assert!(mitigation_result.is_ok());
    
    let result = mitigation_result.unwrap();
    assert!(result.mitigation_overhead >= 1.0);
    assert!(result.confidence >= 0.0);
    assert!(result.confidence <= 1.0);
    
    // Predict syndromes
    let syndrome_result = manager.predict_syndromes(&circuit, Duration::from_secs(60));
    assert!(syndrome_result.is_ok());
    
    // Integrate QEC
    let qec_result = manager.integrate_qec("surface_code", &circuit);
    assert!(qec_result.is_ok());
    
    let qec_integration = qec_result.unwrap();
    assert!(qec_integration.resource_overhead >= 1.0);
    assert!(qec_integration.expected_logical_error_rate >= 0.0);
    
    println!("Integrated error mitigation workflow completed successfully");
    println!("Mitigation overhead: {:.2}x", result.mitigation_overhead);
    println!("Mitigation confidence: {:.3}", result.confidence);
    println!("QEC resource overhead: {:.2}x", qec_integration.resource_overhead);
    println!("Expected logical error rate: {:.2e}", qec_integration.expected_logical_error_rate);
}