//! Default implementations for configuration types

use super::config::*;
use quantrs2_core::platform::PlatformCapabilities;
use std::time::Duration;

impl Default for ParallelizationConfig {
    fn default() -> Self {
        Self {
            strategy: ParallelizationStrategy::Hybrid,
            resource_allocation: ResourceAllocationConfig::default(),
            scheduling_config: ParallelSchedulingConfig::default(),
            hardware_awareness: HardwareAwarenessConfig::default(),
            performance_config: PerformanceOptimizationConfig::default(),
            load_balancing: LoadBalancingConfig::default(),
            monitoring_config: ResourceMonitoringConfig::default(),
        }
    }
}

impl Default for ResourceAllocationConfig {
    fn default() -> Self {
        Self {
            max_concurrent_circuits: PlatformCapabilities::detect().cpu.logical_cores,
            max_concurrent_gates: 16,
            cpu_allocation: CpuAllocationStrategy::PercentageCores(0.8),
            memory_limits: MemoryLimits::default(),
            qpu_allocation: QpuAllocationConfig::default(),
            network_allocation: NetworkAllocationConfig::default(),
        }
    }
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_total_memory_mb: 8192.0, // 8GB
            max_per_circuit_mb: 1024.0,  // 1GB
            allocation_strategy: MemoryAllocationStrategy::Dynamic,
            enable_pooling: true,
            gc_threshold: 0.8,
        }
    }
}

impl Default for QpuAllocationConfig {
    fn default() -> Self {
        Self {
            max_qpu_time_per_circuit: Duration::from_secs(300), // 5 minutes
            sharing_strategy: QpuSharingStrategy::HybridSlicing,
            queue_management: QueueManagementConfig::default(),
            fairness_config: FairnessConfig::default(),
        }
    }
}

impl Default for QueueManagementConfig {
    fn default() -> Self {
        Self {
            algorithm: QueueSchedulingAlgorithm::Priority,
            max_queue_size: 1000,
            priority_levels: 5,
            enable_preemption: true,
            timeout_config: TimeoutConfig::default(),
        }
    }
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            execution_timeout: Duration::from_secs(3600), // 1 hour
            queue_timeout: Duration::from_secs(1800),     // 30 minutes
            resource_timeout: Duration::from_secs(300),   // 5 minutes
            adaptive_timeouts: true,
        }
    }
}

impl Default for FairnessConfig {
    fn default() -> Self {
        Self {
            algorithm: FairnessAlgorithm::ProportionalFair,
            resource_quotas: ResourceQuotas::default(),
            aging_factor: 1.1,
            enable_burst_allowances: true,
        }
    }
}

impl Default for ResourceQuotas {
    fn default() -> Self {
        Self {
            cpu_quota: Some(Duration::from_secs(3600 * 24)), // 24 hours per day
            qpu_quota: Some(Duration::from_secs(3600)),      // 1 hour per day
            memory_quota: Some(16384.0),                     // 16GB
            circuit_quota: Some(1000),                       // 1000 circuits per day
        }
    }
}

impl Default for NetworkAllocationConfig {
    fn default() -> Self {
        Self {
            max_bandwidth_per_circuit: 100.0, // 100 Mbps
            qos_class: NetworkQoSClass::BestEffort,
            compression_config: CompressionConfig::default(),
            latency_optimization: true,
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::Zstd,
            level: 3,
            size_threshold: 1024, // 1KB
        }
    }
}

impl Default for ParallelSchedulingConfig {
    fn default() -> Self {
        Self {
            algorithm: ParallelSchedulingAlgorithm::WorkStealing,
            work_stealing: WorkStealingConfig::default(),
            load_balancing_params: LoadBalancingParams::default(),
            thread_pool_config: ThreadPoolConfig::default(),
        }
    }
}

impl Default for WorkStealingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: WorkStealingStrategy::LoadBased,
            queue_size: 1000,
            stealing_threshold: 0.5,
        }
    }
}

impl Default for LoadBalancingParams {
    fn default() -> Self {
        Self {
            rebalancing_frequency: Duration::from_secs(30),
            load_threshold: 0.8,
            migration_cost_factor: 0.1,
            adaptive_balancing: true,
        }
    }
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            core_threads: PlatformCapabilities::detect().cpu.logical_cores,
            max_threads: PlatformCapabilities::detect().cpu.logical_cores * 2,
            keep_alive_time: Duration::from_secs(60),
            thread_priority: ThreadPriority::Normal,
            affinity_config: ThreadAffinityConfig::default(),
        }
    }
}

impl Default for ThreadAffinityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            assignment_strategy: CoreAssignmentStrategy::Automatic,
            numa_preference: NumaPreference::None,
        }
    }
}

impl Default for HardwareAwarenessConfig {
    fn default() -> Self {
        Self {
            topology_awareness: TopologyAwarenessLevel::Connectivity,
            calibration_integration: CalibrationIntegrationConfig::default(),
            error_rate_config: ErrorRateConfig::default(),
            connectivity_config: ConnectivityConfig::default(),
            resource_tracking: ResourceTrackingConfig::default(),
        }
    }
}

impl Default for CalibrationIntegrationConfig {
    fn default() -> Self {
        Self {
            use_realtime_calibration: true,
            update_frequency: Duration::from_secs(300),
            quality_threshold: 0.95,
            enable_predictive: true,
        }
    }
}

impl Default for ErrorRateConfig {
    fn default() -> Self {
        Self {
            consider_error_rates: true,
            error_threshold: 0.01,
            mitigation_strategy: ErrorMitigationStrategy::Composite,
            prediction_model: ErrorPredictionModel::MachineLearning,
        }
    }
}

impl Default for ConnectivityConfig {
    fn default() -> Self {
        Self {
            enforce_constraints: true,
            swap_strategy: SwapInsertionStrategy::Lookahead,
            routing_preference: RoutingPreference::QualityAware,
            optimization_config: ConnectivityOptimizationConfig::default(),
        }
    }
}

impl Default for ConnectivityOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_parallel_routing: true,
            optimization_level: OptimizationLevel::Moderate,
            use_ml_routing: true,
            precompute_tables: true,
        }
    }
}

impl Default for ResourceTrackingConfig {
    fn default() -> Self {
        Self {
            track_cpu_usage: true,
            track_memory_usage: true,
            track_qpu_usage: true,
            track_network_usage: true,
            tracking_granularity: TrackingGranularity::Medium,
            reporting_frequency: Duration::from_secs(60),
        }
    }
}

impl Default for PerformanceOptimizationConfig {
    fn default() -> Self {
        Self {
            objectives: vec![OptimizationObjective::Balanced],
            caching_config: CachingConfig::default(),
            prefetching_config: PrefetchingConfig::default(),
            batch_config: BatchProcessingConfig::default(),
            adaptive_config: AdaptiveOptimizationConfig::default(),
        }
    }
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            enable_result_caching: true,
            enable_compilation_caching: true,
            size_limits: CacheSizeLimits::default(),
            eviction_policy: CacheEvictionPolicy::LRU,
            warming_strategies: vec![CacheWarmingStrategy::PreloadCommon],
        }
    }
}

impl Default for CacheSizeLimits {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            max_memory_mb: 1024.0,
            max_disk_mb: 5120.0,
            per_user_limits: None,
        }
    }
}

impl Default for PrefetchingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: PrefetchingStrategy::Adaptive,
            prefetch_distance: 3,
            confidence_threshold: 0.7,
        }
    }
}

impl Default for BatchProcessingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            size_limits: BatchSizeLimits::default(),
            strategy: BatchingStrategy::Adaptive,
            timeout: Duration::from_secs(30),
        }
    }
}

impl Default for BatchSizeLimits {
    fn default() -> Self {
        Self {
            min_size: 1,
            max_size: 100,
            optimal_size: 10,
            dynamic_sizing: true,
        }
    }
}

impl Default for AdaptiveOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            adaptation_frequency: Duration::from_secs(300),
            monitoring_window: Duration::from_secs(900),
            sensitivity: 0.1,
            ml_config: AdaptiveMLConfig::default(),
        }
    }
}

impl Default for AdaptiveMLConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_type: MLModelType::RandomForest,
            training_frequency: Duration::from_secs(3600),
            feature_config: FeatureEngineeringConfig::default(),
        }
    }
}

impl Default for FeatureEngineeringConfig {
    fn default() -> Self {
        Self {
            circuit_features: vec![
                CircuitFeature::QubitCount,
                CircuitFeature::Depth,
                CircuitFeature::GateCount,
            ],
            hardware_features: vec![
                HardwareFeature::AvailableQubits,
                HardwareFeature::ErrorRates,
                HardwareFeature::QueueStatus,
            ],
            performance_features: vec![
                PerformanceFeature::ExecutionTime,
                PerformanceFeature::Throughput,
                PerformanceFeature::ResourceEfficiency,
            ],
            normalization: FeatureNormalization::ZScore,
        }
    }
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            algorithm: LoadBalancingAlgorithm::ResourceBased,
            monitoring: LoadMonitoringConfig::default(),
            rebalancing_triggers: RebalancingTriggers::default(),
            migration_policies: MigrationPolicies::default(),
        }
    }
}

impl Default for LoadMonitoringConfig {
    fn default() -> Self {
        Self {
            frequency: Duration::from_secs(30),
            metrics: vec![
                LoadMetric::CpuUtilization,
                LoadMetric::MemoryUtilization,
                LoadMetric::QpuUtilization,
                LoadMetric::QueueLength,
            ],
            thresholds: LoadThresholds::default(),
            retention_period: Duration::from_secs(3600 * 24),
        }
    }
}

impl Default for LoadThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 0.8,
            memory_threshold: 0.85,
            qpu_threshold: 0.9,
            network_threshold: 0.8,
            queue_threshold: 100,
            response_time_threshold: Duration::from_secs(30),
        }
    }
}

impl Default for RebalancingTriggers {
    fn default() -> Self {
        Self {
            cpu_imbalance_threshold: 0.3,
            memory_imbalance_threshold: 0.3,
            queue_imbalance_threshold: 0.4,
            time_interval: Some(Duration::from_secs(300)),
            event_triggers: vec![
                RebalancingEvent::NodeFailure,
                RebalancingEvent::LoadSpike,
                RebalancingEvent::PerformanceDegradation,
            ],
        }
    }
}

impl Default for MigrationPolicies {
    fn default() -> Self {
        Self {
            cost_threshold: 0.1,
            max_migrations_per_period: 10,
            migration_period: Duration::from_secs(3600),
            circuit_migration_strategy: CircuitMigrationStrategy::CheckpointRestart,
            data_migration_strategy: DataMigrationStrategy::Copy,
        }
    }
}

impl Default for ResourceMonitoringConfig {
    fn default() -> Self {
        Self {
            real_time_monitoring: true,
            granularity: MonitoringGranularity::Circuit,
            metrics_collection: MetricsCollectionConfig::default(),
            alerting: AlertingConfig::default(),
            reporting: ReportingConfig::default(),
        }
    }
}

impl Default for MetricsCollectionConfig {
    fn default() -> Self {
        Self {
            frequency: Duration::from_secs(60),
            metrics: vec![
                MonitoringMetric::ResourceUtilization,
                MonitoringMetric::Performance,
                MonitoringMetric::Quality,
            ],
            retention_policy: RetentionPolicy::default(),
            storage_config: StorageConfig::default(),
        }
    }
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            raw_data_retention: Duration::from_secs(3600 * 24 * 7), // 1 week
            aggregated_data_retention: Duration::from_secs(3600 * 24 * 30), // 1 month
            archive_policy: ArchivePolicy::TimeBased(Duration::from_secs(3600 * 24 * 365)), // 1 year
            compression: CompressionConfig::default(),
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::LocalFilesystem,
            location: "/tmp/quantrs_metrics".to_string(),
            encryption: EncryptionConfig::default(),
            replication: ReplicationConfig::default(),
        }
    }
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: EncryptionAlgorithm::AES256,
            key_management: KeyManagementConfig::default(),
        }
    }
}

impl Default for KeyManagementConfig {
    fn default() -> Self {
        Self {
            rotation_frequency: Duration::from_secs(3600 * 24 * 30), // 1 month
            key_derivation: KeyDerivationFunction::Argon2,
            storage_backend: KeyStorageBackend::Local,
        }
    }
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            replication_factor: 1,
            strategy: ReplicationStrategy::Synchronous,
            consistency_level: ConsistencyLevel::Strong,
        }
    }
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rules: vec![],
            channels: vec![],
            aggregation: AlertAggregationConfig::default(),
        }
    }
}

impl Default for AlertAggregationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window: Duration::from_secs(300),
            strategy: AlertAggregationStrategy::SeverityBased,
            max_alerts_per_window: 10,
        }
    }
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            report_types: vec![ReportType::Performance, ReportType::ResourceUtilization],
            frequency: Duration::from_secs(3600 * 24), // Daily reports
            format: ReportFormat::JSON,
            distribution: ReportDistribution::default(),
        }
    }
}

impl Default for ReportDistribution {
    fn default() -> Self {
        Self {
            email_recipients: vec![],
            file_location: Some("/tmp/quantrs_reports".to_string()),
            cloud_location: None,
            api_endpoints: vec![],
        }
    }
}
