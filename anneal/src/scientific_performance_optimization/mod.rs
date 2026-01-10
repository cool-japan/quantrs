//! Scientific Performance Optimization for Large-Scale Applications
//!
//! This module implements comprehensive performance optimization strategies for large-scale
//! scientific computing applications including protein folding, materials science, and drug
//! discovery. It provides memory optimization, algorithmic improvements, parallel processing
//! enhancements, and distributed computing support for handling massive molecular systems,
//! crystal lattices, and pharmaceutical datasets.
//!
//! Key Features:
//! - Hierarchical memory management with intelligent caching
//! - Scalable algorithms with sub-quadratic complexity
//! - Multi-GPU acceleration and distributed computing
//! - Problem decomposition strategies for massive systems
//! - Performance profiling and bottleneck identification
//! - Adaptive optimization based on system characteristics
//! - Memory-mapped I/O for large datasets
//! - Streaming algorithms for continuous processing

mod algorithm;
mod config;
mod distributed;
mod memory;
mod optimizer;
mod parallel;
mod profiling;
mod results;

// Re-export all public types
pub use algorithm::*;
pub use config::*;
pub use distributed::*;
pub use memory::*;
pub use optimizer::*;
pub use parallel::*;
pub use profiling::*;
pub use results::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_optimizer_creation() {
        let config = PerformanceOptimizationConfig::default();
        let optimizer = ScientificPerformanceOptimizer::new(config);

        assert_eq!(optimizer.config.memory_config.cache_size_limit, 8192);
        assert_eq!(
            optimizer.config.parallel_config.num_threads,
            num_cpus::get()
        );
    }

    #[test]
    fn test_memory_optimization_config() {
        let config = MemoryOptimizationConfig::default();
        assert!(config.enable_hierarchical_memory);
        assert_eq!(config.cache_size_limit, 8192);
        assert!(config.enable_memory_mapping);
    }

    #[test]
    fn test_parallel_processing_config() {
        let config = ParallelProcessingConfig::default();
        assert_eq!(config.num_threads, num_cpus::get());
        assert_eq!(
            config.scheduling_strategy,
            TaskSchedulingStrategy::WorkStealing
        );
    }

    #[test]
    fn test_algorithm_optimization_config() {
        let config = AlgorithmOptimizationConfig::default();
        assert!(config.enable_algorithmic_improvements);
        assert!(
            config
                .decomposition_config
                .enable_hierarchical_decomposition
        );
        assert!(config.caching_config.enable_result_caching);
    }

    #[test]
    fn test_gpu_acceleration_config() {
        let config = GPUAccelerationConfig::default();
        assert!(!config.enable_gpu); // Disabled by default
        assert_eq!(config.device_selection, GPUDeviceSelection::Automatic);
    }

    #[test]
    fn test_optimization_recommendations() {
        let _optimizer = create_example_performance_optimizer()
            .expect("Failed to create example performance optimizer");
        let recommendations =
            ScientificPerformanceOptimizer::generate_optimization_recommendations()
                .expect("Failed to generate optimization recommendations");

        assert!(!recommendations.is_empty());
        assert!(recommendations
            .iter()
            .any(|r| r.category == OptimizationCategory::Memory));
    }

    #[test]
    fn test_performance_report_generation() {
        let optimizer = create_example_performance_optimizer()
            .expect("Failed to create example performance optimizer");
        let report = optimizer
            .get_performance_report()
            .expect("Failed to get performance report");

        assert!(report.system_metrics.overall_performance_score > 0.0);
        assert!(!report.optimization_recommendations.is_empty());
    }

    #[test]
    fn test_hierarchical_memory_manager() {
        let config = MemoryOptimizationConfig::default();
        let manager = HierarchicalMemoryManager::new(config);

        assert_eq!(manager.memory_stats.current_usage, 0);
        assert_eq!(manager.cache_hierarchy.cache_stats.hits, 0);
    }

    #[test]
    fn test_cache_hierarchy() {
        let cache_hierarchy = CacheHierarchy::new();

        assert_eq!(cache_hierarchy.l1_cache.capacity, 1024);
        assert_eq!(cache_hierarchy.l2_cache.capacity, 1024 * 1024);
        assert_eq!(cache_hierarchy.l3_cache.capacity, 10 * 1024 * 1024);
    }

    #[test]
    fn test_decomposition_strategies() {
        let strategies = vec![
            DecompositionStrategy::Uniform,
            DecompositionStrategy::Adaptive,
            DecompositionStrategy::GraphBased,
            DecompositionStrategy::Hierarchical,
        ];

        // Test that each strategy is a valid enum variant
        assert_eq!(strategies.len(), 4);

        // Test that different strategies are indeed different
        assert_ne!(
            DecompositionStrategy::Uniform,
            DecompositionStrategy::Adaptive
        );
        assert_ne!(
            DecompositionStrategy::Adaptive,
            DecompositionStrategy::GraphBased
        );
        assert_ne!(
            DecompositionStrategy::GraphBased,
            DecompositionStrategy::Hierarchical
        );

        // Test that strategies can be cloned and compared
        for strategy in &strategies {
            let cloned = strategy.clone();
            assert_eq!(strategy, &cloned);
        }
    }

    #[test]
    fn test_lru_cache() {
        let mut cache: LRUCache<String, Vec<u8>> = LRUCache::new(3);

        cache.insert("key1".to_string(), vec![1, 2, 3]);
        cache.insert("key2".to_string(), vec![4, 5, 6]);
        cache.insert("key3".to_string(), vec![7, 8, 9]);

        assert_eq!(cache.len(), 3);
        assert!(cache.contains(&"key1".to_string()));

        // Access key1 to make it recently used
        assert!(cache.get(&"key1".to_string()).is_some());

        // Add key4, should evict key2 (least recently used)
        cache.insert("key4".to_string(), vec![10, 11, 12]);
        assert!(!cache.contains(&"key2".to_string()));
        assert!(cache.contains(&"key1".to_string()));
    }

    #[test]
    fn test_task_scheduler() {
        let mut scheduler = TaskScheduler::new();
        assert!(scheduler.task_queue.is_empty());

        // Test that scheduler can be created with default values
        assert_eq!(scheduler.strategy, TaskSchedulingStrategy::WorkStealing);
    }

    #[test]
    fn test_load_balancer() {
        let mut balancer = LoadBalancer::new();
        assert!(balancer.worker_loads.is_empty());

        // Add worker load
        let load = WorkerLoad::new(0);
        balancer.update_load(0, load);

        assert_eq!(balancer.worker_loads.len(), 1);
    }

    #[test]
    fn test_problem_decomposer() {
        let mut decomposer = ProblemDecomposer::new();
        assert!(decomposer.subproblems.is_empty());

        // Decompose a problem
        let ids = decomposer.decompose("test_problem", ProblemData::Generic(vec![]));
        assert!(!ids.is_empty());
        assert!(!decomposer.subproblems.is_empty());
    }

    #[test]
    fn test_result_cache() {
        let mut cache = ResultCache::new();

        // Put a result
        cache.put("key1".to_string(), vec![1, 2, 3], 0.95);
        assert!(cache.contains("key1"));

        // Get the result
        let result = cache.get("key1");
        assert!(result.is_some());
        assert_eq!(result.map(|r| r.access_count).unwrap_or(0), 2); // 1 from put + 1 from get
    }

    #[test]
    fn test_streaming_processor() {
        let mut processor = StreamingProcessor::new();

        // Add elements
        processor.add_element(vec![1, 2, 3], std::collections::HashMap::new());
        processor.add_element(vec![4, 5, 6], std::collections::HashMap::new());

        assert_eq!(processor.statistics.windows_created, 1);
    }

    #[test]
    fn test_cluster_manager() {
        let mut manager = ClusterManager::new();
        assert!(manager.active_nodes.is_empty());

        // Add a node
        let resources = NodeResources {
            cpu_cores: 8,
            memory_mb: 16384,
            gpu_count: 1,
            network_bandwidth: 1000.0,
        };
        manager.add_node("localhost:8001".to_string(), resources);

        assert_eq!(manager.active_nodes.len(), 1);
        assert!(manager.get_node("localhost:8001").is_some());
    }

    #[test]
    fn test_communication_manager() {
        let mut comm = CommunicationManager::new();

        // Connect
        let result = comm.connect("node1");
        assert!(result.is_ok());
        assert_eq!(comm.statistics.connections_established, 1);

        // Send message
        let msg = Message::new(
            "master".to_string(),
            "node1".to_string(),
            MessageType::Heartbeat,
            vec![],
        );
        let send_result = comm.send("node1", msg);
        assert!(send_result.is_ok());
        assert_eq!(comm.statistics.messages_sent, 1);
    }

    #[test]
    fn test_gpu_accelerator() {
        let config = GPUAccelerationConfig::default();
        let accelerator = GPUAccelerator::new(config);

        // GPU is disabled by default
        assert!(!accelerator.is_available());
        assert_eq!(accelerator.device_count(), 0);
    }

    #[test]
    fn test_cpu_profiler() {
        let mut profiler = CPUProfiler::new();

        // Start profiling
        profiler.start();
        assert!(profiler.is_active);

        // Take samples
        profiler.sample();
        assert!(!profiler.cpu_samples.is_empty());

        // Record function call
        profiler.record_function_call("test_func", std::time::Duration::from_millis(10));
        assert!(profiler.function_stats.contains_key("test_func"));

        // Stop profiling
        profiler.stop();
        assert!(!profiler.is_active);
    }
}
