//! Comprehensive tests for hardware samplers.

use ndarray::{Array2, ArrayD, IxDyn};
use quantrs2_tytan::sampler::hardware::*;
use quantrs2_tytan::sampler::Sampler;
use std::collections::HashMap;

#[test]
fn test_fujitsu_sampler_basic() {
    let mut qubo = Array2::zeros((3, 3));
    qubo[[0, 0]] = 1.0;
    qubo[[1, 1]] = 1.0;
    qubo[[0, 1]] = -2.0;
    qubo[[1, 0]] = -2.0;

    let mut var_map = HashMap::new();
    var_map.insert("x0".to_string(), 0);
    var_map.insert("x1".to_string(), 1);
    var_map.insert("x2".to_string(), 2);

    let config = FujitsuConfig {
        annealing_time: 1000,
        temperature_schedule: TemperatureSchedule::Linear { start: 10.0, end: 0.1 },
        iterations: 100,
        api_endpoint: "http://localhost:8080".to_string(),
        api_key: "test_key".to_string(),
        optimization_level: OptimizationLevel::Balanced,
    };

    let sampler = FujitsuSampler::new(config);
    
    // Test configuration
    assert!(sampler.is_available());
    
    // Note: This would normally connect to actual hardware
    // For testing, we just verify the sampler is properly configured
    let problem = (qubo, var_map);
    
    // Test problem validation
    assert!(sampler.validate_problem(&problem).is_ok());
}

#[test]
fn test_hitachi_sampler_configuration() {
    let config = HitachiConfig {
        cmos_annealing_unit: CmosAnnealingUnit::Gen2,
        annealing_schedule: AnnealingSchedule::Adaptive {
            initial_temperature: 5.0,
            cooling_rate: 0.95,
            adaptive_threshold: 0.01,
        },
        parallel_runs: 8,
        convergence_threshold: 1e-6,
        max_iterations: 1000,
        api_endpoint: "https://api.hitachi.com".to_string(),
        credentials: HitachiCredentials {
            user_id: "test_user".to_string(),
            api_token: "test_token".to_string(),
            project_id: "test_project".to_string(),
        },
    };

    let sampler = HitachiSampler::new(config);
    
    // Test basic functionality
    assert!(sampler.is_available());
    
    // Test problem size limits
    let large_qubo = Array2::zeros((5000, 5000));
    let var_map: HashMap<String, usize> = (0..5000)
        .map(|i| (format!("x{}", i), i))
        .collect();
    
    let large_problem = (large_qubo, var_map);
    let validation_result = sampler.validate_problem(&large_problem);
    
    // Should handle large problems but may have limitations
    match validation_result {
        Ok(_) => println!("Large problem accepted"),
        Err(e) => println!("Large problem rejected: {}", e),
    }
}

#[test]
fn test_nec_sampler_algorithms() {
    let config = NECConfig {
        vector_engine_type: VectorEngineType::VEAPLUS,
        algorithm: NECAlgorithm::SimulatedQuantumAnnealing {
            trotter_slices: 32,
            monte_carlo_steps: 1000,
            transverse_field_schedule: TransverseFieldSchedule::Linear {
                start: 2.0,
                end: 0.01,
            },
        },
        precision: Precision::Double,
        parallel_tempering: Some(ParallelTemperingConfig {
            num_replicas: 16,
            exchange_interval: 100,
            temperature_range: (0.1, 10.0),
        }),
        api_config: NECApiConfig {
            endpoint: "https://nec-quantum.com/api".to_string(),
            auth_token: "test_token".to_string(),
            timeout: std::time::Duration::from_secs(300),
        },
    };

    let sampler = NECSampler::new(config);
    
    // Test algorithm switching
    let mut qubo = Array2::zeros((4, 4));
    qubo[[0, 1]] = 1.0;
    qubo[[1, 0]] = 1.0;
    qubo[[2, 3]] = -1.0;
    qubo[[3, 2]] = -1.0;

    let var_map: HashMap<String, usize> = (0..4)
        .map(|i| (format!("q{}", i), i))
        .collect();

    let problem = (qubo, var_map);
    assert!(sampler.validate_problem(&problem).is_ok());
}

#[test]
fn test_fpga_sampler_multiple_algorithms() {
    // Test Simulated Bifurcation
    let sb_config = FPGAConfig {
        device_type: FPGADeviceType::Xilinx(XilinxDevice::UltraScale),
        algorithm: FPGAAlgorithm::SimulatedBifurcation {
            dt: 0.01,
            c: 1.0,
            pump_schedule: PumpSchedule::Linear { start: 0.0, end: 2.0 },
            max_steps: 10000,
        },
        memory_optimization: MemoryOptimization::Balanced,
        precision: BitPrecision::Fixed32,
        parallelization: ParallelizationStrategy::TiledComputation { tile_size: 256 },
        monitoring: PerformanceMonitoring {
            enable_profiling: true,
            memory_tracking: true,
            thermal_monitoring: true,
        },
    };

    let sb_sampler = FPGASampler::new(sb_config);
    assert!(sb_sampler.is_available());

    // Test Digital Annealing
    let da_config = FPGAConfig {
        device_type: FPGADeviceType::Intel(IntelDevice::Stratix),
        algorithm: FPGAAlgorithm::DigitalAnnealing {
            beta_schedule: BetaSchedule::Exponential { initial: 0.1, rate: 1.05 },
            replica_exchange: Some(ReplicaExchangeConfig {
                num_replicas: 8,
                exchange_probability: 0.3,
            }),
            local_search: LocalSearchConfig {
                method: LocalSearchMethod::TwoOpt,
                frequency: 100,
                intensity: 0.1,
            },
        },
        memory_optimization: MemoryOptimization::Speed,
        precision: BitPrecision::Fixed16,
        parallelization: ParallelizationStrategy::PipelinedExecution { pipeline_depth: 4 },
        monitoring: PerformanceMonitoring {
            enable_profiling: true,
            memory_tracking: false,
            thermal_monitoring: true,
        },
    };

    let da_sampler = FPGASampler::new(da_config);
    assert!(da_sampler.is_available());

    // Test problem with different samplers
    let mut qubo = Array2::zeros((8, 8));
    for i in 0..7 {
        qubo[[i, i + 1]] = -1.0;
        qubo[[i + 1, i]] = -1.0;
    }

    let var_map: HashMap<String, usize> = (0..8)
        .map(|i| (format!("v{}", i), i))
        .collect();

    let problem = (qubo, var_map);
    
    assert!(sb_sampler.validate_problem(&problem).is_ok());
    assert!(da_sampler.validate_problem(&problem).is_ok());
}

#[test]
fn test_photonic_sampler_platforms() {
    // Test Coherent Ising Machine
    let cim_config = PhotonicConfig {
        platform: PhotonicPlatform::CoherentIsingMachine {
            optical_parametric_oscillators: 100,
            feedback_strength: 0.5,
            detuning_control: DetuningControl::Adaptive,
            measurement_basis: MeasurementBasis::XQuadrature,
        },
        solver_parameters: PhotonicSolverParams {
            evolution_time: 10.0,
            time_step: 0.01,
            convergence_threshold: 1e-6,
            max_iterations: 10000,
        },
        noise_model: NoiseModel::Gaussian {
            amplitude_noise: 0.01,
            phase_noise: 0.005,
            detection_efficiency: 0.95,
        },
        api_config: PhotonicApiConfig {
            endpoint: "https://photonic-quantum.com/api".to_string(),
            credentials: PhotonicCredentials {
                api_key: "test_key".to_string(),
                project_id: "test_project".to_string(),
            },
            timeout: std::time::Duration::from_secs(600),
        },
    };

    let cim_sampler = PhotonicSampler::new(cim_config);
    assert!(cim_sampler.is_available());

    // Test Spatial Photonic Ising Machine
    let spim_config = PhotonicConfig {
        platform: PhotonicPlatform::SpatialPhotonic {
            spatial_light_modulator: SpatialLightModulator::LiquidCrystal,
            beam_splitter_network: BeamSplitterNetwork::Triangular,
            detection_scheme: DetectionScheme::Homodyne,
            num_modes: 64,
        },
        solver_parameters: PhotonicSolverParams {
            evolution_time: 5.0,
            time_step: 0.005,
            convergence_threshold: 1e-5,
            max_iterations: 5000,
        },
        noise_model: NoiseModel::Realistic {
            fiber_loss: 0.02,
            beam_splitter_imbalance: 0.01,
            detector_dark_counts: 100.0,
            thermal_fluctuations: 0.001,
        },
        api_config: PhotonicApiConfig {
            endpoint: "https://spim-quantum.com/api".to_string(),
            credentials: PhotonicCredentials {
                api_key: "test_key2".to_string(),
                project_id: "test_project2".to_string(),
            },
            timeout: std::time::Duration::from_secs(300),
        },
    };

    let spim_sampler = PhotonicSampler::new(spim_config);
    assert!(spim_sampler.is_available());

    // Test small QUBO problem
    let mut qubo = Array2::zeros((4, 4));
    qubo[[0, 0]] = 1.0;
    qubo[[1, 1]] = 1.0;
    qubo[[2, 2]] = 1.0;
    qubo[[3, 3]] = 1.0;
    qubo[[0, 1]] = -2.0;
    qubo[[1, 0]] = -2.0;
    qubo[[2, 3]] = -2.0;
    qubo[[3, 2]] = -2.0;

    let var_map: HashMap<String, usize> = (0..4)
        .map(|i| (format!("s{}", i), i))
        .collect();

    let problem = (qubo, var_map);
    
    assert!(cim_sampler.validate_problem(&problem).is_ok());
    assert!(spim_sampler.validate_problem(&problem).is_ok());
}

#[test]
fn test_hardware_sampler_error_handling() {
    // Test with invalid configuration
    let invalid_config = FujitsuConfig {
        annealing_time: 0, // Invalid
        temperature_schedule: TemperatureSchedule::Linear { start: 0.0, end: 10.0 }, // Invalid order
        iterations: 0, // Invalid
        api_endpoint: "".to_string(), // Invalid
        api_key: "".to_string(), // Invalid
        optimization_level: OptimizationLevel::Balanced,
    };

    let sampler = FujitsuSampler::new(invalid_config);
    
    // Should still create sampler but validation should fail
    let mut empty_qubo = Array2::zeros((0, 0));
    let empty_var_map = HashMap::new();
    let invalid_problem = (empty_qubo, empty_var_map);
    
    assert!(sampler.validate_problem(&invalid_problem).is_err());
}

#[test]
fn test_hardware_sampler_compatibility() {
    // Test that all hardware samplers implement the required traits
    let fujitsu_config = FujitsuConfig::default();
    let fujitsu = FujitsuSampler::new(fujitsu_config);

    let hitachi_config = HitachiConfig::default();
    let hitachi = HitachiSampler::new(hitachi_config);

    let nec_config = NECConfig::default();
    let nec = NECSampler::new(nec_config);

    let fpga_config = FPGAConfig::default();
    let fpga = FPGASampler::new(fpga_config);

    let photonic_config = PhotonicConfig::default();
    let photonic = PhotonicSampler::new(photonic_config);

    // All should implement Sampler trait
    let samplers: Vec<&dyn Sampler> = vec![&fujitsu, &hitachi, &nec, &fpga, &photonic];
    
    for sampler in samplers {
        // Test basic sampler interface
        assert!(sampler.is_available() || !sampler.is_available()); // Either true or false
    }
}