//! Universal Annealing Compiler for Any Quantum Platform
//!
//! This module implements the most advanced universal compiler for quantum annealing
//! that can target ANY quantum platform - from D-Wave to IBM, IonQ, Rigetti, and
//! future quantum devices. It provides hardware-agnostic optimization with automatic
//! translation, adaptive scheduling, and quantum advantage maximization.
//!
//! Revolutionary Features:
//! - Universal compilation to any quantum annealing platform
//! - Automatic hardware topology adaptation and optimization
//! - Cross-platform performance optimization and benchmarking
//! - Intelligent scheduling across heterogeneous quantum resources
//! - Real-time hardware capability discovery and utilization
//! - Quantum error correction integration for any platform
//! - Cost optimization across multiple cloud quantum services
//! - Performance prediction and guarantee systems
//!
//! # Module Structure
//!
//! - `config` - Compiler configuration and settings
//! - `platform` - Platform registry and quantum platform types
//! - `hardware` - Hardware specifications and requirements
//! - `compilation` - Compilation engine and results
//! - `execution` - Execution planning and results
//! - `scheduling` - Resource scheduling and allocation
//! - `compiler` - Main compiler implementation

pub mod compilation;
pub mod compiler;
pub mod config;
pub mod execution;
pub mod hardware;
pub mod platform;
pub mod scheduling;

// Re-export main types for convenience
pub use config::{
    CostConstraints, ErrorCorrectionRequirements, ErrorCorrectionStrategy, OptimizationLevel,
    PerformanceRequirements, ResourceAllocationStrategy, ResourcePreferences,
    SchedulingPreferences, SchedulingPriority, UniversalCompilerConfig,
};

pub use platform::{
    AvailabilityInfo, CostStructure, PlatformCapabilities, PlatformInfo, PlatformRegistry,
    PlatformStatus, ProblemType, QuantumPlatform,
};

pub use hardware::{
    CoherenceRequirements, ConnectivityGraph, ConnectivityRequirements, EmbeddingRequirements,
    EmbeddingStrategy, ErrorCharacteristics, ErrorRateRequirements, HardwareRequirements,
    HardwareSpecification, OperatingConditions, PlatformPerformanceCharacteristics, TopologyType,
};

pub use compilation::{
    CacheConfig, CacheEntry, CacheStatistics, ClassicalComputeRequirements, CompilationCache,
    CompilationEngine, CompilationEngineConfig, CompilationMetadata, CompilationResult,
    CompiledRepresentation, CompiledResourceRequirements, ConfidenceIntervals, EvictionPolicy,
    OptimizationPass, PerformancePredictions, PlatformCompiler, VerificationLevel,
};

pub use execution::{
    CostOptimizer, ExecutionMetadata, ExecutionParameters, ExecutionPlan, ExecutionQualityMetrics,
    ExecutionResourceUsage, OptimalPlatformSelection, PerformancePredictor,
    PlatformExecutionResult, PlatformPerformancePrediction, PlatformResourceAllocation,
    PredictedPerformance, PredictionMetadata, ResourceReservationInfo, SelectionMetadata,
    UniversalExecutionMetadata, UniversalExecutionResult,
};

pub use scheduling::{
    AffinityConstraint, AffinityStrength, AffinityType, AlertChannel, AlertChannelType,
    AlertingConfig, AllocatedResources, AllocationConstraints, AllocationObjective,
    AllocationObjectives, AllocationRecord, AllocationStrategy, AllocatorConfig,
    AvailableResources, CompletedJob, FairnessPolicy, JobResourceRequirements, JobStatus,
    LoadBalancingConfig, LoadBalancingStrategy, MetricValue, PerformanceSnapshot,
    PerformanceTracker, PlatformMetrics, QueueStatistics, ReservedResources, ResourceAllocator,
    ResourceCapacity, ResourceLoad, ResourceReservation, RunningJob, ScheduledJob, SchedulerConfig,
    SchedulingAlgorithm, SchedulingQueue, SystemState, TimeSlot, TrackerConfig,
    UniversalResourceScheduler,
};

pub use compiler::{create_example_universal_compiler, UniversalAnnealingCompiler};
