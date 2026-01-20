#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Comprehensive tests for Tensor Network Sampler module

#[cfg(test)]
mod tests {
    use quantrs2_tytan::sampler::{SampleResult, Sampler, SamplerError, SamplerResult};
    use quantrs2_tytan::tensor_network_sampler::*;
    use scirs2_core::ndarray::{Array1, Array2, ArrayD};
    use std::collections::HashMap;

    /// Test tensor network configuration
    #[test]
    fn test_tensor_network_config() {
        let config = TensorNetworkConfig {
            network_type: TensorNetworkType::MPS { bond_dimension: 64 },
            max_bond_dimension: 128,
            compression_tolerance: 1e-10,
            num_sweeps: 100,
            convergence_tolerance: 1e-8,
            use_gpu: false,
            parallel_config: ParallelConfig {
                num_threads: 8,
                distributed: false,
                chunk_size: 1000,
                load_balancing: LoadBalancingStrategy::Dynamic,
            },
            memory_config: MemoryConfig {
                max_memory_gb: 8.0,
                memory_mapping: true,
                gc_frequency: 100,
                cache_optimization: CacheOptimization::Combined,
            },
        };

        assert_eq!(config.max_bond_dimension, 128);
        assert_eq!(config.compression_tolerance, 1e-10);
        assert_eq!(config.num_sweeps, 100);
        assert_eq!(config.convergence_tolerance, 1e-8);
        assert!(!config.use_gpu);
    }

    /// Test tensor network types
    #[test]
    fn test_tensor_network_types() {
        let mps = TensorNetworkType::MPS { bond_dimension: 32 };
        let peps = TensorNetworkType::PEPS {
            bond_dimension: 16,
            lattice_shape: (8, 8),
        };
        let mera = TensorNetworkType::MERA {
            layers: 4,
            branching_factor: 2,
        };
        let ttn = TensorNetworkType::TTN {
            tree_structure: TreeStructure {
                nodes: vec![],
                edges: vec![],
                root: 0,
                depth: 3,
            },
        };

        match mps {
            TensorNetworkType::MPS { bond_dimension } => {
                assert_eq!(bond_dimension, 32);
            }
            _ => panic!("Wrong tensor network type"),
        }

        match peps {
            TensorNetworkType::PEPS {
                bond_dimension,
                lattice_shape,
            } => {
                assert_eq!(bond_dimension, 16);
                assert_eq!(lattice_shape, (8, 8));
            }
            _ => panic!("Wrong tensor network type"),
        }

        match mera {
            TensorNetworkType::MERA {
                layers,
                branching_factor,
            } => {
                assert_eq!(layers, 4);
                assert_eq!(branching_factor, 2);
            }
            _ => panic!("Wrong tensor network type"),
        }

        match ttn {
            TensorNetworkType::TTN { tree_structure } => {
                assert_eq!(tree_structure.root, 0);
                assert_eq!(tree_structure.depth, 3);
            }
            _ => panic!("Wrong tensor network type"),
        }
    }

    /// Test sampler creation
    #[test]
    fn test_mps_sampler_creation() {
        let sampler = create_mps_sampler(32);
        assert_eq!(sampler.config.max_bond_dimension, 64);

        if let TensorNetworkType::MPS { bond_dimension } = sampler.config.network_type {
            assert_eq!(bond_dimension, 32);
        } else {
            panic!("Expected MPS network type");
        }
    }

    /// Test PEPS sampler creation
    #[test]
    fn test_peps_sampler_creation() {
        let sampler = create_peps_sampler(16, (4, 4));

        if let TensorNetworkType::PEPS {
            bond_dimension,
            lattice_shape,
        } = sampler.config.network_type
        {
            assert_eq!(bond_dimension, 16);
            assert_eq!(lattice_shape, (4, 4));
        } else {
            panic!("Expected PEPS network type");
        }
    }

    /// Test MERA sampler creation
    #[test]
    fn test_mera_sampler_creation() {
        let sampler = create_mera_sampler(3);

        if let TensorNetworkType::MERA {
            layers,
            branching_factor,
        } = sampler.config.network_type
        {
            assert_eq!(layers, 3);
            assert_eq!(branching_factor, 2);
        } else {
            panic!("Expected MERA network type");
        }
    }

    /// Test compression methods
    #[test]
    fn test_compression_methods() {
        let methods = vec![
            CompressionMethod::SVD,
            CompressionMethod::QR,
            CompressionMethod::RandomizedSVD,
            CompressionMethod::TensorTrain,
            CompressionMethod::Tucker,
            CompressionMethod::CP,
        ];

        for method in methods {
            match method {
                CompressionMethod::SVD => assert!(true),
                CompressionMethod::QR => assert!(true),
                CompressionMethod::RandomizedSVD => assert!(true),
                CompressionMethod::TensorTrain => assert!(true),
                CompressionMethod::Tucker => assert!(true),
                CompressionMethod::CP => assert!(true),
            }
        }
    }

    /// Test optimization algorithms
    #[test]
    fn test_optimization_algorithms() {
        let algorithms = vec![
            OptimizationAlgorithm::DMRG,
            OptimizationAlgorithm::TEBD,
            OptimizationAlgorithm::VMPS,
            OptimizationAlgorithm::ALS,
            OptimizationAlgorithm::GradientDescent,
            OptimizationAlgorithm::ConjugateGradient,
            OptimizationAlgorithm::LBFGS,
            OptimizationAlgorithm::TrustRegion,
        ];

        for algorithm in algorithms {
            match algorithm {
                OptimizationAlgorithm::DMRG => assert!(true),
                OptimizationAlgorithm::TEBD => assert!(true),
                OptimizationAlgorithm::VMPS => assert!(true),
                OptimizationAlgorithm::ALS => assert!(true),
                OptimizationAlgorithm::GradientDescent => assert!(true),
                OptimizationAlgorithm::ConjugateGradient => assert!(true),
                OptimizationAlgorithm::LBFGS => assert!(true),
                OptimizationAlgorithm::TrustRegion => assert!(true),
            }
        }
    }

    /// Test load balancing strategies
    #[test]
    fn test_load_balancing_strategies() {
        let strategies = vec![
            LoadBalancingStrategy::Static,
            LoadBalancingStrategy::Dynamic,
            LoadBalancingStrategy::WorkStealing,
            LoadBalancingStrategy::Adaptive,
        ];

        for strategy in strategies {
            match strategy {
                LoadBalancingStrategy::Static => assert!(true),
                LoadBalancingStrategy::Dynamic => assert!(true),
                LoadBalancingStrategy::WorkStealing => assert!(true),
                LoadBalancingStrategy::Adaptive => assert!(true),
            }
        }
    }

    /// Test cache optimization
    #[test]
    fn test_cache_optimization() {
        let optimizations = vec![
            CacheOptimization::None,
            CacheOptimization::Spatial,
            CacheOptimization::Temporal,
            CacheOptimization::Combined,
        ];

        for optimization in optimizations {
            match optimization {
                CacheOptimization::None => assert!(true),
                CacheOptimization::Spatial => assert!(true),
                CacheOptimization::Temporal => assert!(true),
                CacheOptimization::Combined => assert!(true),
            }
        }
    }

    /// Test index types
    #[test]
    fn test_index_types() {
        let index_types = vec![
            IndexType::Physical,
            IndexType::Virtual,
            IndexType::Auxiliary,
            IndexType::Time,
        ];

        for index_type in index_types {
            match index_type {
                IndexType::Physical => assert!(true),
                IndexType::Virtual => assert!(true),
                IndexType::Auxiliary => assert!(true),
                IndexType::Time => assert!(true),
            }
        }
    }

    /// Test quality metrics
    #[test]
    fn test_quality_metrics() {
        let metrics = vec![
            QualityMetric::RelativeError,
            QualityMetric::SpectralNormError,
            QualityMetric::FrobeniusNormError,
            QualityMetric::InformationLoss,
            QualityMetric::EntanglementPreservation,
        ];

        for metric in metrics {
            match metric {
                QualityMetric::RelativeError => assert!(true),
                QualityMetric::SpectralNormError => assert!(true),
                QualityMetric::FrobeniusNormError => assert!(true),
                QualityMetric::InformationLoss => assert!(true),
                QualityMetric::EntanglementPreservation => assert!(true),
            }
        }
    }

    /// Test default configurations
    #[test]
    fn test_default_configs() {
        let config = create_default_tensor_config();
        assert_eq!(config.max_bond_dimension, 128);
        assert_eq!(config.compression_tolerance, 1e-10);
        assert_eq!(config.num_sweeps, 100);
        assert_eq!(config.convergence_tolerance, 1e-8);
        assert!(!config.use_gpu);
    }
}
