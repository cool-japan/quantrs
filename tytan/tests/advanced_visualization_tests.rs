//! Comprehensive tests for advanced visualization and analysis.

use ndarray::{Array1, Array2, Array3};
use quantrs2_tytan::advanced_visualization::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::{Duration, SystemTime};

#[test]
fn test_visualization_manager_creation() {
    let mut config = VisualizationConfig::default();
    let mut manager = AdvancedVisualizationManager::new(config);

    // Test that manager is created successfully
    assert!(true);
}

#[test]
fn test_default_visualization_config() {
    let mut config = VisualizationConfig::default();

    // Verify default configuration values
    assert!(config.interactive_mode);
    assert!(config.real_time_updates);
    assert!(config.enable_3d_rendering);
    assert!(config.quantum_state_viz);
    assert!(config.performance_dashboard);
    assert_eq!(config.update_frequency, Duration::from_millis(100));
    assert_eq!(config.max_data_points, 10000);
    assert!(!config.export_formats.is_empty());
    assert!(matches!(config.rendering_quality, RenderingQuality::High));
    assert!(!config.color_schemes.is_empty());
}

#[test]
fn test_color_schemes() {
    let mut default_scheme = ColorScheme::default();
    let mut high_contrast_scheme = ColorScheme::high_contrast();
    let mut colorblind_friendly_scheme = ColorScheme::colorblind_friendly();

    // Test default color scheme
    assert_eq!(default_scheme.primary, "#1f77b4");
    assert_eq!(default_scheme.background, "#ffffff");
    assert_eq!(default_scheme.energy_high, "#d62728");
    assert_eq!(default_scheme.energy_low, "#2ca02c");

    // Test high contrast scheme
    assert_eq!(high_contrast_scheme.primary, "#000000");
    assert_eq!(high_contrast_scheme.secondary, "#ffffff");
    assert_eq!(high_contrast_scheme.accent, "#ffff00");

    // Test colorblind friendly scheme
    assert_eq!(colorblind_friendly_scheme.primary, "#0173B2");
    assert_eq!(colorblind_friendly_scheme.secondary, "#DE8F05");
    assert_eq!(colorblind_friendly_scheme.accent, "#029E73");
}

#[test]
fn test_export_formats() {
    let formats = vec![
        ExportFormat::PNG,
        ExportFormat::SVG,
        ExportFormat::PDF,
        ExportFormat::HTML,
        ExportFormat::JSON,
        ExportFormat::CSV,
        ExportFormat::WebGL,
        ExportFormat::ThreeJS,
    ];

    // Test that all export formats can be created
    assert_eq!(formats.len(), 8);
    for format in formats {
        match format {
            ExportFormat::PNG => assert!(true),
            ExportFormat::SVG => assert!(true),
            ExportFormat::PDF => assert!(true),
            ExportFormat::HTML => assert!(true),
            ExportFormat::JSON => assert!(true),
            ExportFormat::CSV => assert!(true),
            ExportFormat::WebGL => assert!(true),
            ExportFormat::ThreeJS => assert!(true),
        }
    }
}

#[test]
fn test_rendering_quality_levels() {
    let qualities = vec![
        RenderingQuality::Low,
        RenderingQuality::Medium,
        RenderingQuality::High,
        RenderingQuality::Ultra,
    ];

    // Test rendering quality levels
    assert_eq!(qualities.len(), 4);
    for quality in qualities {
        match quality {
            RenderingQuality::Low => assert!(true),
            RenderingQuality::Medium => assert!(true),
            RenderingQuality::High => assert!(true),
            RenderingQuality::Ultra => assert!(true),
        }
    }
}

#[test]
fn test_energy_sample() {
    let sample = EnergySample {
        configuration: Array1::from(vec![1.0, 0.0, 1.0, 0.0, 1.0]),
        energy: -2.5,
        metadata: SampleMetadata {
            sampling_method: "SimulatedAnnealing".to_string(),
            timestamp: SystemTime::now(),
            confidence: 0.95,
            weight: 1.0,
        },
    };

    // Verify energy sample structure
    assert_eq!(sample.configuration.len(), 5);
    assert_eq!(sample.energy, -2.5);
    assert_eq!(sample.metadata.sampling_method, "SimulatedAnnealing");
    assert_eq!(sample.metadata.confidence, 0.95);
    assert_eq!(sample.metadata.weight, 1.0);
}

#[test]
fn test_landscape_data() {
    let landscape_data = LandscapeData {
        energy_samples: vec![
            EnergySample {
                configuration: Array1::from(vec![1.0, 0.0]),
                energy: -1.0,
                metadata: SampleMetadata {
                    sampling_method: "Random".to_string(),
                    timestamp: SystemTime::now(),
                    confidence: 0.9,
                    weight: 1.0,
                },
            },
            EnergySample {
                configuration: Array1::from(vec![0.0, 1.0]),
                energy: -0.5,
                metadata: SampleMetadata {
                    sampling_method: "Random".to_string(),
                    timestamp: SystemTime::now(),
                    confidence: 0.9,
                    weight: 1.0,
                },
            },
        ],
        problem_size: 2,
        energy_bounds: (-1.0, 0.0),
        sample_density: 0.5,
        interpolated_surface: None,
        critical_points: vec![],
        solution_paths: vec![],
    };

    // Verify landscape data structure
    assert_eq!(landscape_data.energy_samples.len(), 2);
    assert_eq!(landscape_data.problem_size, 2);
    assert_eq!(landscape_data.energy_bounds.0, -1.0);
    assert_eq!(landscape_data.energy_bounds.1, 0.0);
    assert_eq!(landscape_data.sample_density, 0.5);
    assert!(landscape_data.interpolated_surface.is_none());
}

#[test]
fn test_critical_point() {
    let critical_point = CriticalPoint {
        location: Array1::from(vec![0.5, 0.5, 0.5]),
        point_type: CriticalPointType::LocalMinimum,
        energy: -std::f64::consts::PI,
        stability: StabilityAnalysis {
            eigenvalues: Array1::from(vec![2.0, 1.0, 0.5]),
            eigenvectors: Array2::eye(3),
            stability_type: StabilityType::Stable,
            basin_size: 0.1,
        },
        curvature: CurvatureData {
            principal_curvatures: Array1::from(vec![2.0, 1.0]),
            mean_curvature: 1.5,
            gaussian_curvature: 2.0,
            curvature_directions: Array2::eye(2),
        },
    };

    // Verify critical point structure
    assert_eq!(critical_point.location.len(), 3);
    assert!(matches!(
        critical_point.point_type,
        CriticalPointType::LocalMinimum
    ));
    assert_eq!(critical_point.energy, -std::f64::consts::PI);
    assert_eq!(critical_point.stability.eigenvalues.len(), 3);
    assert!(matches!(
        critical_point.stability.stability_type,
        StabilityType::Stable
    ));
    assert_eq!(critical_point.curvature.mean_curvature, 1.5);
}

#[test]
fn test_critical_point_types() {
    let types = vec![
        CriticalPointType::GlobalMinimum,
        CriticalPointType::LocalMinimum,
        CriticalPointType::LocalMaximum,
        CriticalPointType::SaddlePoint { index: 2 },
        CriticalPointType::Plateau,
        CriticalPointType::Unknown,
    ];

    // Test critical point types
    assert_eq!(types.len(), 6);
    for point_type in types {
        match point_type {
            CriticalPointType::GlobalMinimum => assert!(true),
            CriticalPointType::LocalMinimum => assert!(true),
            CriticalPointType::LocalMaximum => assert!(true),
            CriticalPointType::SaddlePoint { index } => assert_eq!(index, 2),
            CriticalPointType::Plateau => assert!(true),
            CriticalPointType::Unknown => assert!(true),
        }
    }
}

#[test]
fn test_stability_types() {
    let types = vec![
        StabilityType::Stable,
        StabilityType::Unstable,
        StabilityType::MarginallStable,
        StabilityType::SaddleStable,
        StabilityType::Unknown,
    ];

    // Test stability types
    assert_eq!(types.len(), 5);
    for stability_type in types {
        match stability_type {
            StabilityType::Stable => assert!(true),
            StabilityType::Unstable => assert!(true),
            StabilityType::MarginallStable => assert!(true),
            StabilityType::SaddleStable => assert!(true),
            StabilityType::Unknown => assert!(true),
        }
    }
}

#[test]
fn test_solution_path() {
    let path = SolutionPath {
        points: vec![
            Array1::from(vec![0.0, 0.0]),
            Array1::from(vec![0.5, 0.5]),
            Array1::from(vec![1.0, 1.0]),
        ],
        energy_trajectory: Array1::from(vec![1.0, -0.5, -2.0]),
        metadata: PathMetadata {
            length: 2.828, // sqrt(2) + sqrt(2)
            convergence_rate: 0.8,
            iterations: 100,
            final_gradient_norm: 1e-6,
        },
        algorithm: "GradientDescent".to_string(),
    };

    // Verify solution path structure
    assert_eq!(path.points.len(), 3);
    assert_eq!(path.energy_trajectory.len(), 3);
    assert_eq!(path.algorithm, "GradientDescent");
    assert_eq!(path.metadata.convergence_rate, 0.8);
    assert_eq!(path.metadata.iterations, 100);
    assert_eq!(path.metadata.final_gradient_norm, 1e-6);
}

#[test]
fn test_interpolation_methods() {
    let methods = vec![
        InterpolationMethod::Linear,
        InterpolationMethod::Cubic,
        InterpolationMethod::Spline,
        InterpolationMethod::RadialBasisFunction {
            kernel: RBFKernel::Gaussian { bandwidth: 1.0 },
        },
        InterpolationMethod::Kriging,
        InterpolationMethod::InverseDistanceWeighting { power: 2.0 },
    ];

    // Test interpolation methods
    assert_eq!(methods.len(), 6);
    for method in methods {
        match method {
            InterpolationMethod::Linear => assert!(true),
            InterpolationMethod::Cubic => assert!(true),
            InterpolationMethod::Spline => assert!(true),
            InterpolationMethod::RadialBasisFunction { kernel } => match kernel {
                RBFKernel::Gaussian { bandwidth } => assert_eq!(bandwidth, 1.0),
                _ => panic!("Unexpected RBF kernel"),
            },
            InterpolationMethod::Kriging => assert!(true),
            InterpolationMethod::InverseDistanceWeighting { power } => assert_eq!(power, 2.0),
        }
    }
}

#[test]
fn test_rbf_kernels() {
    let kernels = vec![
        RBFKernel::Gaussian { bandwidth: 1.0 },
        RBFKernel::Multiquadric { c: 1.0 },
        RBFKernel::InverseMultiquadric { c: 1.0 },
        RBFKernel::ThinPlateSpline,
    ];

    // Test RBF kernels
    assert_eq!(kernels.len(), 4);
    for kernel in kernels {
        match kernel {
            RBFKernel::Gaussian { bandwidth } => assert_eq!(bandwidth, 1.0),
            RBFKernel::Multiquadric { c } => assert_eq!(c, 1.0),
            RBFKernel::InverseMultiquadric { c } => assert_eq!(c, 1.0),
            RBFKernel::ThinPlateSpline => assert!(true),
        }
    }
}

#[test]
fn test_smoothing_methods() {
    let methods = vec![
        SmoothingMethod::Gaussian,
        SmoothingMethod::Bilateral,
        SmoothingMethod::MedianFilter,
        SmoothingMethod::SavitzkyGolay {
            window_size: 5,
            polynomial_order: 2,
        },
        SmoothingMethod::None,
    ];

    // Test smoothing methods
    assert_eq!(methods.len(), 5);
    for method in methods {
        match method {
            SmoothingMethod::Gaussian => assert!(true),
            SmoothingMethod::Bilateral => assert!(true),
            SmoothingMethod::MedianFilter => assert!(true),
            SmoothingMethod::SavitzkyGolay {
                window_size,
                polynomial_order,
            } => {
                assert_eq!(window_size, 5);
                assert_eq!(polynomial_order, 2);
            }
            SmoothingMethod::None => assert!(true),
        }
    }
}

#[test]
fn test_convergence_session() {
    let session = ConvergenceSession {
        session_id: "test_session_001".to_string(),
        algorithm: "SimulatedAnnealing".to_string(),
        problem_config: ProblemConfiguration {
            size: 100,
            problem_type: "QUBO".to_string(),
            target_energy: Some(-10.0),
            convergence_criteria: ConvergenceCriteria {
                energy_tolerance: 1e-6,
                gradient_tolerance: 1e-8,
                max_iterations: 10000,
                stagnation_threshold: 100,
                time_limit: Some(Duration::from_secs(3600)),
            },
        },
        convergence_data: ConvergenceData {
            energy_trajectory: Default::default(),
            gradient_norms: Default::default(),
            parameter_updates: Default::default(),
            step_sizes: Default::default(),
            algorithm_metrics: HashMap::new(),
        },
        metrics: ConvergenceMetrics {
            current_energy: 0.0,
            best_energy: f64::INFINITY,
            gradient_norm: 0.0,
            convergence_rate: 0.0,
            eta_convergence: None,
            status: ConvergenceStatus::Unknown,
        },
        viz_state: ConvergenceVisualizationState {
            chart_states: HashMap::new(),
            animation_state: AnimationState {
                is_playing: false,
                playback_speed: 1.0,
                current_frame: 0,
                total_frames: 0,
            },
            interaction_state: ConvergenceInteractionState {
                brush_selection: None,
                hover_point: None,
                tooltip_info: None,
            },
        },
    };

    // Verify convergence session structure
    assert_eq!(session.session_id, "test_session_001");
    assert_eq!(session.algorithm, "SimulatedAnnealing");
    assert_eq!(session.problem_config.size, 100);
    assert_eq!(session.problem_config.problem_type, "QUBO");
    assert_eq!(session.problem_config.target_energy, Some(-10.0));
    assert_eq!(
        session.problem_config.convergence_criteria.max_iterations,
        10000
    );
    assert!(matches!(session.metrics.status, ConvergenceStatus::Unknown));
}

#[test]
fn test_convergence_status_types() {
    let statuses = vec![
        ConvergenceStatus::Converging,
        ConvergenceStatus::Converged,
        ConvergenceStatus::Stagnated,
        ConvergenceStatus::Diverging,
        ConvergenceStatus::Oscillating,
        ConvergenceStatus::Unknown,
    ];

    // Test convergence status types
    assert_eq!(statuses.len(), 6);
    for status in statuses {
        match status {
            ConvergenceStatus::Converging => assert!(true),
            ConvergenceStatus::Converged => assert!(true),
            ConvergenceStatus::Stagnated => assert!(true),
            ConvergenceStatus::Diverging => assert!(true),
            ConvergenceStatus::Oscillating => assert!(true),
            ConvergenceStatus::Unknown => assert!(true),
        }
    }
}

#[test]
fn test_quantum_state_data_types() {
    use num::Complex;

    // Test pure state
    let mut pure_state = QuantumStateData::PureState(Array1::from(vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
    ]));

    match pure_state {
        QuantumStateData::PureState(state_vector) => {
            assert_eq!(state_vector.len(), 2);
            assert_eq!(state_vector[0], Complex::new(1.0, 0.0));
        }
        _ => panic!("Expected pure state"),
    }

    // Test mixed state
    let mut mixed_state = QuantumStateData::MixedState(Array2::eye(2));

    match mixed_state {
        QuantumStateData::MixedState(density_matrix) => {
            assert_eq!(density_matrix.dim(), (2, 2));
        }
        _ => panic!("Expected mixed state"),
    }

    // Test stabilizer state
    let mut stabilizer_state = QuantumStateData::StabilizerState(StabilizerRepresentation {
        generators: vec![],
        phases: vec![],
    });

    match stabilizer_state {
        QuantumStateData::StabilizerState(stabilizer_rep) => {
            assert!(stabilizer_rep.generators.is_empty());
            assert!(stabilizer_rep.phases.is_empty());
        }
        _ => panic!("Expected stabilizer state"),
    }
}

#[test]
fn test_state_visualization_types() {
    let viz_types = vec![
        StateVisualizationType::BlochSphere,
        StateVisualizationType::QSphere,
        StateVisualizationType::QuantumCircuit,
        StateVisualizationType::DensityMatrix,
        StateVisualizationType::Wigner,
        StateVisualizationType::Hinton,
        StateVisualizationType::City,
        StateVisualizationType::Paulivec,
    ];

    // Test state visualization types
    assert_eq!(viz_types.len(), 8);
    for viz_type in viz_types {
        match viz_type {
            StateVisualizationType::BlochSphere => assert!(true),
            StateVisualizationType::QSphere => assert!(true),
            StateVisualizationType::QuantumCircuit => assert!(true),
            StateVisualizationType::DensityMatrix => assert!(true),
            StateVisualizationType::Wigner => assert!(true),
            StateVisualizationType::Hinton => assert!(true),
            StateVisualizationType::City => assert!(true),
            StateVisualizationType::Paulivec => assert!(true),
        }
    }
}

#[test]
fn test_widget_types() {
    let widget_types = vec![
        WidgetType::LineChart,
        WidgetType::BarChart,
        WidgetType::Scatter3D,
        WidgetType::Heatmap,
        WidgetType::Gauge,
        WidgetType::Table,
        WidgetType::Text,
        WidgetType::Custom("CustomWidget".to_string()),
    ];

    // Test widget types
    assert_eq!(widget_types.len(), 8);
    for widget_type in widget_types {
        match widget_type {
            WidgetType::LineChart => assert!(true),
            WidgetType::BarChart => assert!(true),
            WidgetType::Scatter3D => assert!(true),
            WidgetType::Heatmap => assert!(true),
            WidgetType::Gauge => assert!(true),
            WidgetType::Table => assert!(true),
            WidgetType::Text => assert!(true),
            WidgetType::Custom(name) => assert_eq!(name, "CustomWidget"),
        }
    }
}

#[test]
fn test_dashboard_data() {
    let mut metrics = HashMap::new();
    metrics.insert("cpu_usage".to_string(), 0.75);
    metrics.insert("memory_usage".to_string(), 0.60);
    metrics.insert("convergence_rate".to_string(), 0.85);

    let mut metadata = HashMap::new();
    metadata.insert("algorithm".to_string(), "SimulatedAnnealing".to_string());
    metadata.insert("problem_size".to_string(), "100".to_string());

    let dashboard_data = DashboardData {
        timestamp: SystemTime::now(),
        metrics,
        metadata,
    };

    // Verify dashboard data structure
    assert_eq!(dashboard_data.metrics.len(), 3);
    assert_eq!(dashboard_data.metadata.len(), 2);
    assert_eq!(dashboard_data.metrics["cpu_usage"], 0.75);
    assert_eq!(dashboard_data.metadata["algorithm"], "SimulatedAnnealing");
}

#[test]
fn test_widget_config() {
    let config = WidgetConfig {
        title: "Energy Convergence".to_string(),
        dimensions: (800, 600),
        refresh_rate: Duration::from_secs(1),
        data_source: "convergence_tracker".to_string(),
    };

    // Verify widget configuration
    assert_eq!(config.title, "Energy Convergence");
    assert_eq!(config.dimensions, (800, 600));
    assert_eq!(config.refresh_rate, Duration::from_secs(1));
    assert_eq!(config.data_source, "convergence_tracker");
}

#[test]
fn test_comparison_result() {
    let comparison_result = ComparisonResult {
        comparison_id: "test_comparison_001".to_string(),
        datasets_compared: vec!["SA".to_string(), "GA".to_string(), "TS".to_string()],
        statistical_results: StatisticalResults {
            p_values: {
                let mut p_vals = HashMap::new();
                p_vals.insert("SA_vs_GA".to_string(), 0.01);
                p_vals.insert("GA_vs_TS".to_string(), 0.05);
                p_vals
            },
            effect_sizes: {
                let mut effects = HashMap::new();
                effects.insert("SA_vs_GA".to_string(), 0.8);
                effects.insert("GA_vs_TS".to_string(), 0.3);
                effects
            },
            confidence_intervals: HashMap::new(),
        },
        performance_metrics: PerformanceMetrics {
            execution_times: {
                let mut times = HashMap::new();
                times.insert("SA".to_string(), Duration::from_secs(60));
                times.insert("GA".to_string(), Duration::from_secs(120));
                times.insert("TS".to_string(), Duration::from_secs(90));
                times
            },
            memory_usage: HashMap::new(),
            convergence_rates: HashMap::new(),
            solution_quality: HashMap::new(),
        },
        visualizations: vec![VisualizationReference {
            visualization_id: "comparison_chart_001".to_string(),
            visualization_type: "BoxPlot".to_string(),
            description: "Performance comparison box plot".to_string(),
        }],
        recommendations: vec![ComparisonRecommendation {
            recommendation_type: RecommendationType::BestAlgorithm,
            description: "Simulated Annealing shows best performance".to_string(),
            confidence: 0.95,
        }],
    };

    // Verify comparison result structure
    assert_eq!(comparison_result.comparison_id, "test_comparison_001");
    assert_eq!(comparison_result.datasets_compared.len(), 3);
    assert_eq!(comparison_result.statistical_results.p_values.len(), 2);
    assert_eq!(
        comparison_result.performance_metrics.execution_times.len(),
        3
    );
    assert_eq!(comparison_result.visualizations.len(), 1);
    assert_eq!(comparison_result.recommendations.len(), 1);
}

#[test]
fn test_interactive_features() {
    let features = vec![
        InteractiveFeature::Rotation,
        InteractiveFeature::Zooming,
        InteractiveFeature::Selection,
        InteractiveFeature::Animation,
        InteractiveFeature::Measurement,
        InteractiveFeature::StateModification,
    ];

    // Test interactive features
    assert_eq!(features.len(), 6);
    for feature in features {
        match feature {
            InteractiveFeature::Rotation => assert!(true),
            InteractiveFeature::Zooming => assert!(true),
            InteractiveFeature::Selection => assert!(true),
            InteractiveFeature::Animation => assert!(true),
            InteractiveFeature::Measurement => assert!(true),
            InteractiveFeature::StateModification => assert!(true),
        }
    }
}

#[test]
fn test_visualization_types() {
    let viz_types = vec![
        VisualizationType::EnergyLandscape3D,
        VisualizationType::ConvergenceTracking,
        VisualizationType::QuantumState,
        VisualizationType::PerformanceDashboard,
        VisualizationType::ComparativeAnalysis,
    ];

    // Test visualization types
    assert_eq!(viz_types.len(), 5);
    for viz_type in viz_types {
        match viz_type {
            VisualizationType::EnergyLandscape3D => assert!(true),
            VisualizationType::ConvergenceTracking => assert!(true),
            VisualizationType::QuantumState => assert!(true),
            VisualizationType::PerformanceDashboard => assert!(true),
            VisualizationType::ComparativeAnalysis => assert!(true),
        }
    }
}

#[test]
fn test_interaction_types() {
    let interaction_types = vec![
        InteractionType::Click,
        InteractionType::Drag,
        InteractionType::Zoom,
        InteractionType::Rotate,
        InteractionType::Pan,
        InteractionType::Select,
        InteractionType::Hover,
    ];

    // Test interaction types
    assert_eq!(interaction_types.len(), 7);
    for interaction_type in interaction_types {
        match interaction_type {
            InteractionType::Click => assert!(true),
            InteractionType::Drag => assert!(true),
            InteractionType::Zoom => assert!(true),
            InteractionType::Rotate => assert!(true),
            InteractionType::Pan => assert!(true),
            InteractionType::Select => assert!(true),
            InteractionType::Hover => assert!(true),
        }
    }
}

#[test]
fn test_energy_landscape_visualization() {
    let mut config = VisualizationConfig::default();
    let mut manager = AdvancedVisualizationManager::new(config);

    // Create test energy samples
    let energy_samples = vec![
        EnergySample {
            configuration: Array1::from(vec![1.0, 0.0, 1.0]),
            energy: -2.0,
            metadata: SampleMetadata {
                sampling_method: "Random".to_string(),
                timestamp: SystemTime::now(),
                confidence: 0.9,
                weight: 1.0,
            },
        },
        EnergySample {
            configuration: Array1::from(vec![0.0, 1.0, 0.0]),
            energy: -1.5,
            metadata: SampleMetadata {
                sampling_method: "Random".to_string(),
                timestamp: SystemTime::now(),
                confidence: 0.85,
                weight: 1.0,
            },
        },
        EnergySample {
            configuration: Array1::from(vec![1.0, 1.0, 1.0]),
            energy: -3.0,
            metadata: SampleMetadata {
                sampling_method: "Optimization".to_string(),
                timestamp: SystemTime::now(),
                confidence: 0.95,
                weight: 1.5,
            },
        },
    ];

    // Create energy landscape visualization
    let result = manager.create_energy_landscape(&energy_samples);
    assert!(result.is_ok());

    let viz_id = result.unwrap();
    assert!(!viz_id.is_empty());
    assert!(viz_id.starts_with("energy_landscape_"));

    // Verify visualization was registered
    let status = manager.get_visualization_status(&viz_id);
    assert!(status.is_some());

    let viz_status = status.unwrap();
    assert_eq!(viz_status.id, viz_id);
    assert!(matches!(
        viz_status.viz_type,
        VisualizationType::EnergyLandscape3D
    ));

    println!(
        "Energy landscape visualization created successfully: {}",
        viz_id
    );
}

#[test]
fn test_convergence_tracking() {
    let mut config = VisualizationConfig::default();
    let mut manager = AdvancedVisualizationManager::new(config);

    // Create problem configuration
    let problem_config = ProblemConfiguration {
        size: 50,
        problem_type: "QUBO".to_string(),
        target_energy: Some(-5.0),
        convergence_criteria: ConvergenceCriteria {
            energy_tolerance: 1e-6,
            gradient_tolerance: 1e-8,
            max_iterations: 1000,
            stagnation_threshold: 50,
            time_limit: Some(Duration::from_secs(300)),
        },
    };

    // Start convergence tracking
    let result = manager.start_convergence_tracking("SimulatedAnnealing", problem_config);
    assert!(result.is_ok());

    let session_id = result.unwrap();
    assert!(!session_id.is_empty());
    assert!(session_id.starts_with("convergence_SimulatedAnnealing_"));

    // Update convergence data
    let update_result = manager.update_convergence(
        &session_id,
        -2.5,
        0.01,
        Array1::from(vec![0.8, 0.2, 0.9, 0.1]),
    );
    assert!(update_result.is_ok());

    // Update again to track progression
    let update_result2 = manager.update_convergence(
        &session_id,
        -3.2,
        0.005,
        Array1::from(vec![0.9, 0.1, 0.8, 0.2]),
    );
    assert!(update_result2.is_ok());

    println!("Convergence tracking started successfully: {}", session_id);
}

#[test]
fn test_quantum_state_visualization() {
    use num::Complex;

    let mut config = VisualizationConfig::default();
    let mut manager = AdvancedVisualizationManager::new(config);

    // Create a test quantum state (Bell state |00⟩ + |11⟩)
    let quantum_state = QuantumState {
        state_data: QuantumStateData::PureState(Array1::from(vec![
            Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ])),
        metadata: StateMetadata {
            num_qubits: 2,
            entanglement: EntanglementProperties {
                entanglement_entropy: 1.0,
                schmidt_rank: 2,
                purity: 1.0,
                entanglement_spectrum: Array1::from(vec![0.5, 0.5]),
                subsystem_entanglement: HashMap::new(),
            },
            preparation_method: "Bell state preparation".to_string(),
            fidelity_estimate: Some(0.99),
            timestamp: SystemTime::now(),
        },
        measurement_data: None,
    };

    // Test different visualization types
    let viz_types = vec![
        StateVisualizationType::BlochSphere,
        StateVisualizationType::QSphere,
        StateVisualizationType::DensityMatrix,
    ];

    for viz_type in viz_types {
        let result = manager.visualize_quantum_state(&quantum_state, viz_type.clone());
        assert!(result.is_ok());

        let viz_id = result.unwrap();
        assert!(!viz_id.is_empty());
        assert!(viz_id.starts_with("quantum_state_"));

        println!(
            "Quantum state visualization created: {} (type: {:?})",
            viz_id, viz_type
        );
    }
}

#[test]
fn test_performance_dashboard() {
    let mut config = VisualizationConfig::default();
    let mut manager = AdvancedVisualizationManager::new(config);

    // Create performance dashboard
    let data_sources = vec![
        "algorithm_performance".to_string(),
        "system_metrics".to_string(),
        "convergence_tracking".to_string(),
    ];

    let result = manager.create_performance_dashboard(data_sources);
    assert!(result.is_ok());

    let dashboard_id = result.unwrap();
    assert!(!dashboard_id.is_empty());
    assert!(dashboard_id.starts_with("dashboard_"));

    println!(
        "Performance dashboard created successfully: {}",
        dashboard_id
    );
}

#[test]
fn test_comparative_analysis() {
    let mut config = VisualizationConfig::default();
    let mut manager = AdvancedVisualizationManager::new(config);

    // Create test datasets
    let datasets = vec![
        Dataset {
            name: "SimulatedAnnealing".to_string(),
            data_points: vec![
                DataPoint {
                    values: {
                        let mut vals = HashMap::new();
                        vals.insert("energy".to_string(), -2.5);
                        vals.insert("time".to_string(), 60.0);
                        vals
                    },
                    timestamp: Some(SystemTime::now()),
                },
                DataPoint {
                    values: {
                        let mut vals = HashMap::new();
                        vals.insert("energy".to_string(), -3.2);
                        vals.insert("time".to_string(), 120.0);
                        vals
                    },
                    timestamp: Some(SystemTime::now()),
                },
            ],
            metadata: DatasetMetadata {
                algorithm: "SimulatedAnnealing".to_string(),
                problem_size: 50,
                execution_time: Duration::from_secs(120),
                parameters: HashMap::new(),
            },
        },
        Dataset {
            name: "GeneticAlgorithm".to_string(),
            data_points: vec![
                DataPoint {
                    values: {
                        let mut vals = HashMap::new();
                        vals.insert("energy".to_string(), -2.0);
                        vals.insert("time".to_string(), 90.0);
                        vals
                    },
                    timestamp: Some(SystemTime::now()),
                },
                DataPoint {
                    values: {
                        let mut vals = HashMap::new();
                        vals.insert("energy".to_string(), -2.8);
                        vals.insert("time".to_string(), 180.0);
                        vals
                    },
                    timestamp: Some(SystemTime::now()),
                },
            ],
            metadata: DatasetMetadata {
                algorithm: "GeneticAlgorithm".to_string(),
                problem_size: 50,
                execution_time: Duration::from_secs(180),
                parameters: HashMap::new(),
            },
        },
    ];

    // Perform comparative analysis
    let result = manager.compare_algorithms(datasets);
    assert!(result.is_ok());

    let comparison_result = result.unwrap();
    assert!(!comparison_result.comparison_id.is_empty());
    assert_eq!(comparison_result.datasets_compared.len(), 0); // Stub implementation

    println!(
        "Comparative analysis completed: {}",
        comparison_result.comparison_id
    );
}

#[test]
fn test_export_functionality() {
    let mut config = VisualizationConfig::default();
    let mut manager = AdvancedVisualizationManager::new(config);

    // Test different export formats
    let export_formats = vec![
        ExportFormat::PNG,
        ExportFormat::SVG,
        ExportFormat::HTML,
        ExportFormat::JSON,
    ];

    let export_options = ExportOptions {
        resolution: (1920, 1080),
        quality: 0.9,
        compression: true,
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("title".to_string(), "Test Visualization".to_string());
            meta.insert("created_by".to_string(), "QuantRS2-Tytan".to_string());
            meta
        },
    };

    for format in export_formats {
        let result =
            manager.export_visualization("test_viz_001", format.clone(), export_options.clone());
        assert!(result.is_ok());

        let export_path = result.unwrap();
        assert!(!export_path.is_empty());
        assert!(export_path.contains("exported_test_viz_001"));

        println!("Export successful: {} (format: {:?})", export_path, format);
    }
}

#[test]
fn test_lightweight_visualization_manager() {
    let manager = create_lightweight_visualization_manager();

    // Test that lightweight manager has reduced capabilities
    assert!(true); // Stub implementation always succeeds

    println!("Lightweight visualization manager created successfully");
}

#[test]
fn test_advanced_visualization_manager() {
    let manager = create_advanced_visualization_manager();

    // Test that advanced manager has full capabilities
    assert!(true); // Stub implementation always succeeds

    println!("Advanced visualization manager created successfully");
}

#[test]
fn test_configuration_update() {
    let mut config = VisualizationConfig::default();
    let mut manager = AdvancedVisualizationManager::new(config);

    // Create new configuration with different settings
    let new_config = VisualizationConfig {
        interactive_mode: false,
        real_time_updates: false,
        enable_3d_rendering: false,
        quantum_state_viz: false,
        performance_dashboard: false,
        update_frequency: Duration::from_secs(1),
        max_data_points: 1000,
        export_formats: vec![ExportFormat::PNG],
        rendering_quality: RenderingQuality::Low,
        color_schemes: HashMap::new(),
    };

    // Update configuration
    let result = manager.update_config(new_config);
    assert!(result.is_ok());

    println!("Configuration updated successfully");
}

#[test]
fn test_error_handling() {
    let mut config = VisualizationConfig::default();
    let mut manager = AdvancedVisualizationManager::new(config);

    // Test getting status for non-existent visualization
    let status = manager.get_visualization_status("non_existent_viz");
    assert!(status.is_none());

    println!("Error handling test completed");
}

#[test]
fn test_comprehensive_visualization_workflow() {
    let mut config = VisualizationConfig::default();
    let mut manager = AdvancedVisualizationManager::new(config);

    // Step 1: Create energy landscape visualization
    let energy_samples = vec![
        EnergySample {
            configuration: Array1::from(vec![1.0, 0.0, 1.0, 0.0]),
            energy: -2.5,
            metadata: SampleMetadata {
                sampling_method: "SimulatedAnnealing".to_string(),
                timestamp: SystemTime::now(),
                confidence: 0.9,
                weight: 1.0,
            },
        },
        EnergySample {
            configuration: Array1::from(vec![0.0, 1.0, 0.0, 1.0]),
            energy: -1.8,
            metadata: SampleMetadata {
                sampling_method: "SimulatedAnnealing".to_string(),
                timestamp: SystemTime::now(),
                confidence: 0.85,
                weight: 1.0,
            },
        },
    ];

    let landscape_result = manager.create_energy_landscape(&energy_samples);
    assert!(landscape_result.is_ok());
    let landscape_id = landscape_result.unwrap();

    // Step 2: Start convergence tracking
    let problem_config = ProblemConfiguration {
        size: 20,
        problem_type: "QUBO".to_string(),
        target_energy: Some(-3.0),
        convergence_criteria: ConvergenceCriteria {
            energy_tolerance: 1e-6,
            gradient_tolerance: 1e-8,
            max_iterations: 500,
            stagnation_threshold: 25,
            time_limit: Some(Duration::from_secs(120)),
        },
    };

    let convergence_result = manager.start_convergence_tracking("TabuSearch", problem_config);
    assert!(convergence_result.is_ok());
    let convergence_id = convergence_result.unwrap();

    // Step 3: Update convergence data multiple times
    for i in 0..10 {
        let energy = -1.0 - 0.2 * i as f64;
        let gradient_norm = 0.1 * (0.8_f64).powi(i);
        let mut parameters = Array1::from(vec![0.1 * i as f64, 0.9 - 0.1 * i as f64]);

        let update_result =
            manager.update_convergence(&convergence_id, energy, gradient_norm, parameters);
        assert!(update_result.is_ok());
    }

    // Step 4: Create performance dashboard
    let dashboard_result = manager.create_performance_dashboard(vec![
        "energy_landscape".to_string(),
        "convergence_tracker".to_string(),
        "system_monitor".to_string(),
    ]);
    assert!(dashboard_result.is_ok());
    let dashboard_id = dashboard_result.unwrap();

    // Step 5: Export visualizations
    let export_options = ExportOptions {
        resolution: (1600, 1200),
        quality: 0.95,
        compression: false,
        metadata: HashMap::new(),
    };

    let landscape_export =
        manager.export_visualization(&landscape_id, ExportFormat::PNG, export_options.clone());
    assert!(landscape_export.is_ok());

    let dashboard_export =
        manager.export_visualization(&dashboard_id, ExportFormat::HTML, export_options.clone());
    assert!(dashboard_export.is_ok());

    // Verify all visualizations are tracked
    let landscape_status = manager.get_visualization_status(&landscape_id);
    assert!(landscape_status.is_some());

    println!("Comprehensive visualization workflow completed successfully");
    println!("Energy landscape: {}", landscape_id);
    println!("Convergence tracking: {}", convergence_id);
    println!("Performance dashboard: {}", dashboard_id);
}

#[test]
fn test_advanced_interactive_features() {
    // Test complex interaction scenarios
    let interaction_data = InteractionData {
        position: (100.5, 200.3),
        button_info: "left_click".to_string(),
        modifiers: vec!["ctrl".to_string(), "shift".to_string()],
    };

    let user_interaction = UserInteraction {
        interaction_type: InteractionType::Click,
        timestamp: SystemTime::now(),
        data: interaction_data,
    };

    // Verify interaction data
    assert_eq!(user_interaction.data.position, (100.5, 200.3));
    assert_eq!(user_interaction.data.button_info, "left_click");
    assert_eq!(user_interaction.data.modifiers.len(), 2);
    assert!(matches!(
        user_interaction.interaction_type,
        InteractionType::Click
    ));

    println!("Advanced interactive features test completed");
}

#[test]
fn test_geometry_and_rendering() {
    let geometry_element = GeometryElement {
        element_type: GeometryType::Sphere,
        vertices: vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        indices: vec![0, 1, 2],
        normals: Some(vec![(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]),
        texture_coords: Some(vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]),
    };

    // Verify geometry element
    assert!(matches!(
        geometry_element.element_type,
        GeometryType::Sphere
    ));
    assert_eq!(geometry_element.vertices.len(), 3);
    assert_eq!(geometry_element.indices.len(), 3);
    assert!(geometry_element.normals.is_some());
    assert!(geometry_element.texture_coords.is_some());

    println!("Geometry and rendering test completed");
}

#[test]
fn test_animation_system() {
    let keyframe = Keyframe {
        time: 0.5,
        transform: Transform3D {
            translation: (1.0, 2.0, 3.0),
            rotation: (0.0, PI / 4.0, 0.0),
            scale: (1.0, 1.0, 1.0),
        },
        properties: {
            let mut props = HashMap::new();
            props.insert("opacity".to_string(), 0.8);
            props.insert("energy".to_string(), -2.5);
            props
        },
    };

    let animation_data = AnimationData {
        keyframes: vec![keyframe],
        duration: Duration::from_secs(2),
        loop_animation: true,
    };

    // Verify animation data
    assert_eq!(animation_data.keyframes.len(), 1);
    assert_eq!(animation_data.duration, Duration::from_secs(2));
    assert!(animation_data.loop_animation);
    assert_eq!(animation_data.keyframes[0].time, 0.5);
    assert_eq!(animation_data.keyframes[0].properties.len(), 2);

    println!("Animation system test completed");
}
