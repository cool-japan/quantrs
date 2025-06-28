//! Distributed Circuit Execution demonstration
//!
//! This example shows how to set up and use the distributed execution framework
//! for running quantum circuits across multiple backends with load balancing,
//! fault tolerance, and resource management.

use quantrs2_circuit::distributed::{self, *};
use quantrs2_circuit::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 Distributed Circuit Execution Demo");
    println!("====================================\n");

    // Example 1: Setting up a distributed executor
    println!("1. Creating Distributed Executor");
    println!("--------------------------------");

    let mut executor = DistributedExecutor::new();
    println!("Created distributed executor with default configuration");
    println!("  Load balancing: {:?}", executor.load_balancer.strategy);
    println!("  Scheduling policy: {:?}", executor.scheduler.policy);
    println!(
        "  Fault tolerance enabled: {}",
        executor.fault_tolerance.enable_failover
    );
    println!(
        "  Redundancy level: {}",
        executor.fault_tolerance.redundancy_level
    );

    // Example 2: Adding different types of backends
    println!("\n2. Adding Execution Backends");
    println!("----------------------------");

    // Add a quantum hardware backend
    let hardware_backend = create_hardware_backend();
    println!("Adding hardware backend: {}", hardware_backend.id);
    executor.add_backend(hardware_backend)?;

    // Add a simulator backend
    let simulator_backend = create_simulator_backend();
    println!("Adding simulator backend: {}", simulator_backend.id);
    executor.add_backend(simulator_backend)?;

    // Add a cloud service backend
    let cloud_backend = create_cloud_backend();
    println!("Adding cloud backend: {}", cloud_backend.id);
    executor.add_backend(cloud_backend)?;

    // Add a hybrid backend
    let hybrid_backend = create_hybrid_backend();
    println!("Adding hybrid backend: {}", hybrid_backend.id);
    executor.add_backend(hybrid_backend)?;

    println!("Total backends added: {}", executor.backends.len());

    // Example 3: System health status
    println!("\n3. System Health Status");
    println!("-----------------------");

    let health = executor.get_health_status();
    println!("System health:");
    println!("  Total backends: {}", health.total_backends);
    println!("  Available backends: {}", health.available_backends);
    println!("  Total qubits: {}", health.total_qubits);
    println!(
        "  Average queue time: {:.2} seconds",
        health.average_queue_time
    );
    println!("  System load: {:.1}%", health.system_load * 100.0);

    // Example 4: Creating and submitting jobs
    println!("\n4. Creating and Submitting Jobs");
    println!("-------------------------------");

    // Create different types of circuits for testing
    let jobs = create_test_jobs();

    for (i, job) in jobs.iter().enumerate() {
        println!(
            "Job {}: {} ({} qubits, {} gates, priority: {:?})",
            i + 1,
            job.id,
            job.circuit.num_qubits(),
            job.circuit.num_gates(),
            job.priority
        );

        match executor.submit_job(job.clone()) {
            Ok(job_id) => println!("  ✅ Submitted successfully: {}", job_id),
            Err(e) => println!("  ❌ Submission failed: {}", e),
        }
    }

    // Example 5: Different load balancing strategies
    println!("\n5. Load Balancing Strategies");
    println!("----------------------------");

    let strategies = vec![
        ("Round Robin", LoadBalancingStrategy::RoundRobin),
        ("Least Connections", LoadBalancingStrategy::LeastConnections),
        ("Least Queue Time", LoadBalancingStrategy::LeastQueueTime),
        ("Best Performance", LoadBalancingStrategy::BestPerformance),
    ];

    for (name, strategy) in strategies {
        executor.load_balancer.strategy = strategy.clone();
        println!("  {}: {:?}", name, strategy);
    }

    // Example 6: Backend types and capabilities
    println!("\n6. Backend Types and Capabilities");
    println!("---------------------------------");

    for backend in &executor.backends {
        println!("Backend: {} ({:?})", backend.id, backend.status);
        match &backend.backend_type {
            BackendType::Hardware {
                vendor,
                model,
                location,
            } => {
                println!("  Type: Hardware ({} {} in {})", vendor, model, location);
            }
            BackendType::Simulator {
                simulator_type,
                host,
            } => {
                println!("  Type: Simulator ({:?} on {})", simulator_type, host);
            }
            BackendType::CloudService {
                provider,
                service_name,
                region,
            } => {
                println!(
                    "  Type: Cloud ({} {} in {})",
                    provider, service_name, region
                );
            }
            BackendType::Hybrid {
                quantum_backend: _,
                classical_resources,
            } => {
                println!(
                    "  Type: Hybrid ({} CPU cores, {:.1} GB memory)",
                    classical_resources.cpu_cores, classical_resources.memory_gb
                );
            }
        }

        println!("  Capabilities:");
        println!("    Max qubits: {}", backend.performance.max_qubits);
        println!("    Max depth: {}", backend.performance.max_depth);
        println!(
            "    Supported gates: {:?}",
            backend.capabilities.supported_gates
        );
        println!(
            "    Mid-circuit measurements: {}",
            backend.capabilities.mid_circuit_measurements
        );
        println!(
            "    Queue length: {}/{}",
            backend.queue_info.queue_length, backend.queue_info.max_queue_size
        );
        println!(
            "    Estimated wait time: {:.1} seconds",
            backend.queue_info.estimated_wait_time
        );
        println!();
    }

    // Example 7: Execution parameters and error mitigation
    println!("7. Execution Parameters and Error Mitigation");
    println!("--------------------------------------------");

    let error_mitigation_techniques = vec![
        (
            "Readout Error Mitigation",
            ErrorMitigation::ReadoutErrorMitigation,
        ),
        (
            "Zero Noise Extrapolation",
            ErrorMitigation::ZeroNoiseExtrapolation,
        ),
        (
            "Clifford Data Regression",
            ErrorMitigation::CliffordDataRegression,
        ),
        (
            "Symmetry Verification",
            ErrorMitigation::SymmetryVerification,
        ),
    ];

    for (name, technique) in error_mitigation_techniques {
        println!("  {}: {:?}", name, technique);
    }

    let result_formats = vec![
        ("Counts", ResultFormat::Counts),
        ("Probabilities", ResultFormat::Probabilities),
        ("Statevector", ResultFormat::Statevector),
        ("Expectation Values", ResultFormat::ExpectationValues),
    ];

    println!("\nResult formats:");
    for (name, format) in result_formats {
        println!("  {}: {:?}", name, format);
    }

    // Example 8: Fault tolerance and redundancy
    println!("\n8. Fault Tolerance and Redundancy");
    println!("---------------------------------");

    println!("Current fault tolerance configuration:");
    println!(
        "  Failover enabled: {}",
        executor.fault_tolerance.enable_failover
    );
    println!(
        "  Redundancy level: {}",
        executor.fault_tolerance.redundancy_level
    );
    println!(
        "  Error correction: {:?}",
        executor.fault_tolerance.error_correction
    );

    // Demonstrate different error correction strategies
    let error_correction_strategies = vec![
        ("None", ErrorCorrectionStrategy::None),
        ("Majority Voting", ErrorCorrectionStrategy::MajorityVoting),
        (
            "Quantum Error Correction",
            ErrorCorrectionStrategy::QuantumErrorCorrection,
        ),
        (
            "Classical Post-processing",
            ErrorCorrectionStrategy::ClassicalPostProcessing,
        ),
    ];

    println!("\nAvailable error correction strategies:");
    for (name, strategy) in error_correction_strategies {
        println!("  {}: {:?}", name, strategy);
    }

    // Example 9: Resource management and allocation
    println!("\n9. Resource Management and Allocation");
    println!("------------------------------------");

    println!("Resource pool:");
    println!(
        "  Total qubits: {}",
        executor.resource_manager.resource_pool.total_qubits
    );
    println!(
        "  Available slots: {}",
        executor.resource_manager.resource_pool.available_slots
    );
    println!(
        "  Memory pool: {:.1} GB",
        executor.resource_manager.resource_pool.memory_pool
    );
    println!(
        "  Compute pool: {:.1} CPU hours",
        executor.resource_manager.resource_pool.compute_pool
    );

    println!("\nAllocation policies:");
    if let Some(max_qubits) = executor
        .resource_manager
        .allocation_policies
        .max_qubits_per_user
    {
        println!("  Max qubits per user: {}", max_qubits);
    }
    if let Some(max_time) = executor
        .resource_manager
        .allocation_policies
        .max_execution_time
    {
        println!("  Max execution time: {:.1} seconds", max_time);
    }
    println!(
        "  Fair share: {}",
        executor.resource_manager.allocation_policies.fair_share
    );
    println!(
        "  Reserved resources: {:.1}%",
        executor
            .resource_manager
            .allocation_policies
            .reserved_resources
            * 100.0
    );

    // Example 10: Network configuration and authentication
    println!("\n10. Network Configuration and Authentication");
    println!("-------------------------------------------");

    for backend in &executor.backends {
        println!(
            "Backend {}: {}",
            backend.id, backend.network_config.endpoint
        );
        println!(
            "  Auth type: {:?}",
            backend.network_config.credentials.auth_type
        );
        println!(
            "  Connection timeout: {:.1}s",
            backend.network_config.timeouts.connection_timeout
        );
        println!(
            "  Request timeout: {:.1}s",
            backend.network_config.timeouts.request_timeout
        );
        println!(
            "  Max retries: {}",
            backend.network_config.retry_policy.max_retries
        );
        println!(
            "  Backoff strategy: {:?}",
            backend.network_config.retry_policy.backoff_strategy
        );
    }

    // Example 11: Job status and results (mock)
    println!("\n11. Job Status and Results");
    println!("--------------------------");

    for job in jobs.iter().take(3) {
        let status = executor.get_job_status(&job.id)?;
        println!("Job {}: {:?}", job.id, status);

        // Mock getting results for completed jobs
        if status == ExecutionStatus::Queued {
            let result = executor.get_results(&job.id)?;
            println!("  Status: {:?}", result.status);
            println!("  Backends used: {:?}", result.metadata.backends_used);
            println!("  Total time: {:?}", result.metadata.total_time);
            println!("  Queue time: {:?}", result.metadata.queue_time);
            println!("  Resource usage:");
            println!(
                "    CPU hours: {:.3}",
                result.metadata.resource_usage.cpu_hours
            );
            println!(
                "    Memory hours: {:.3}",
                result.metadata.resource_usage.memory_hours
            );
            println!(
                "    Qubit hours: {:.3}",
                result.metadata.resource_usage.qubit_hours
            );
            println!(
                "    Network usage: {:.3} GB",
                result.metadata.resource_usage.network_usage
            );
        }
    }

    // Example 12: Connectivity topologies
    println!("\n12. Connectivity Topologies");
    println!("---------------------------");

    let topologies = vec![
        ("Linear", distributed::TopologyType::Linear),
        ("2D Grid (3x3)", distributed::TopologyType::Grid2D { rows: 3, cols: 3 }),
        ("All-to-all", distributed::TopologyType::AllToAll),
        (
            "Random (70% density)",
            distributed::TopologyType::Random { density: 0.7 },
        ),
    ];

    for (name, topology) in topologies {
        println!("  {}: {:?}", name, topology);
    }

    println!("\n✅ Distributed Circuit Execution Demo completed!");
    println!("\nNote: This demo shows the distributed execution framework structure.");
    println!("Real distributed execution requires actual quantum backends and networking.");

    Ok(())
}

fn create_hardware_backend() -> ExecutionBackend {
    ExecutionBackend {
        id: "ibm_quantum_jakarta".to_string(),
        backend_type: BackendType::Hardware {
            vendor: "IBM".to_string(),
            model: "ibm_jakarta".to_string(),
            location: "Yorktown Heights, NY".to_string(),
        },
        status: BackendStatus::Available,
        performance: BackendPerformance {
            max_qubits: 7,
            max_depth: 500,
            gate_fidelities: {
                let mut fidelities = HashMap::new();
                fidelities.insert("cx".to_string(), 0.95);
                fidelities.insert("u3".to_string(), 0.99);
                fidelities
            },
            coherence_times: {
                let mut times = HashMap::new();
                times.insert("T1".to_string(), 100.0);
                times.insert("T2".to_string(), 50.0);
                times
            },
            execution_time_model: ExecutionTimeModel {
                base_time: 0.5,
                time_per_gate: 0.01,
                time_per_qubit: 0.05,
                time_per_measurement: 0.1,
                network_latency: 0.2,
            },
            throughput: 2.0,
        },
        queue_info: QueueInfo {
            queue_length: 5,
            estimated_wait_time: 120.0,
            max_queue_size: 50,
            priority_levels: vec![Priority::Normal, Priority::High, Priority::Critical],
        },
        capabilities: BackendCapabilities {
            supported_gates: vec![
                "u1".to_string(),
                "u2".to_string(),
                "u3".to_string(),
                "cx".to_string(),
            ],
            mid_circuit_measurements: false,
            classical_control: false,
            reset_operations: true,
            connectivity: ConnectivityGraph {
                num_qubits: 7,
                edges: vec![(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (4, 6)],
                topology: distributed::TopologyType::Custom,
            },
            noise_model: Some(NoiseCharacteristics {
                single_qubit_errors: {
                    let mut errors = HashMap::new();
                    errors.insert("u3".to_string(), 0.001);
                    errors
                },
                two_qubit_errors: {
                    let mut errors = HashMap::new();
                    errors.insert("cx".to_string(), 0.01);
                    errors
                },
                measurement_errors: vec![0.02; 7],
                decoherence_times: vec![100.0; 7],
            }),
        },
        network_config: NetworkConfig {
            endpoint: "https://quantum-computing.ibm.com/api".to_string(),
            credentials: Credentials {
                auth_type: AuthenticationType::Token,
                api_key: None,
                token: Some("your_ibm_token".to_string()),
                username_password: None,
            },
            timeouts: TimeoutConfig {
                connection_timeout: 10.0,
                request_timeout: 300.0,
                total_timeout: 600.0,
            },
            retry_policy: RetryPolicy {
                max_retries: 5,
                base_delay: 2.0,
                backoff_strategy: BackoffStrategy::Exponential { multiplier: 2.0 },
                retryable_errors: vec![distributed::ErrorType::NetworkError, distributed::ErrorType::ServiceUnavailable],
            },
        },
    }
}

fn create_simulator_backend() -> ExecutionBackend {
    ExecutionBackend {
        id: "qiskit_aer_simulator".to_string(),
        backend_type: BackendType::Simulator {
            simulator_type: SimulatorType::StateVector,
            host: "localhost".to_string(),
        },
        status: BackendStatus::Available,
        performance: BackendPerformance {
            max_qubits: 32,
            max_depth: 10000,
            gate_fidelities: {
                let mut fidelities = HashMap::new();
                fidelities.insert("cx".to_string(), 1.0);
                fidelities.insert("u3".to_string(), 1.0);
                fidelities
            },
            coherence_times: HashMap::new(), // No decoherence in ideal simulator
            execution_time_model: ExecutionTimeModel {
                base_time: 0.1,
                time_per_gate: 0.0001,
                time_per_qubit: 0.001,
                time_per_measurement: 0.001,
                network_latency: 0.001,
            },
            throughput: 100.0,
        },
        queue_info: QueueInfo {
            queue_length: 0,
            estimated_wait_time: 0.0,
            max_queue_size: 1000,
            priority_levels: vec![Priority::Low, Priority::Normal, Priority::High],
        },
        capabilities: BackendCapabilities {
            supported_gates: vec![
                "u1".to_string(),
                "u2".to_string(),
                "u3".to_string(),
                "cx".to_string(),
                "ccx".to_string(),
                "cswap".to_string(),
            ],
            mid_circuit_measurements: true,
            classical_control: true,
            reset_operations: true,
            connectivity: ConnectivityGraph {
                num_qubits: 32,
                edges: (0..32)
                    .flat_map(|i| (0..32).filter(move |&j| i != j).map(move |j| (i, j)))
                    .collect(),
                topology: distributed::TopologyType::AllToAll,
            },
            noise_model: None, // Ideal simulator
        },
        network_config: NetworkConfig {
            endpoint: "http://localhost:8080".to_string(),
            credentials: Credentials {
                auth_type: AuthenticationType::None,
                api_key: None,
                token: None,
                username_password: None,
            },
            timeouts: TimeoutConfig {
                connection_timeout: 5.0,
                request_timeout: 60.0,
                total_timeout: 120.0,
            },
            retry_policy: RetryPolicy {
                max_retries: 3,
                base_delay: 1.0,
                backoff_strategy: BackoffStrategy::Fixed,
                retryable_errors: vec![distributed::ErrorType::NetworkError],
            },
        },
    }
}

fn create_cloud_backend() -> ExecutionBackend {
    ExecutionBackend {
        id: "aws_braket_sv1".to_string(),
        backend_type: BackendType::CloudService {
            provider: "AWS".to_string(),
            service_name: "Braket".to_string(),
            region: "us-east-1".to_string(),
        },
        status: BackendStatus::Available,
        performance: BackendPerformance {
            max_qubits: 34,
            max_depth: 5000,
            gate_fidelities: HashMap::new(),
            coherence_times: HashMap::new(),
            execution_time_model: ExecutionTimeModel {
                base_time: 1.0,
                time_per_gate: 0.001,
                time_per_qubit: 0.01,
                time_per_measurement: 0.01,
                network_latency: 0.5,
            },
            throughput: 50.0,
        },
        queue_info: QueueInfo {
            queue_length: 2,
            estimated_wait_time: 30.0,
            max_queue_size: 200,
            priority_levels: vec![Priority::Normal, Priority::High],
        },
        capabilities: BackendCapabilities {
            supported_gates: vec![
                "h".to_string(),
                "x".to_string(),
                "cnot".to_string(),
                "rx".to_string(),
                "ry".to_string(),
                "rz".to_string(),
            ],
            mid_circuit_measurements: false,
            classical_control: false,
            reset_operations: false,
            connectivity: ConnectivityGraph {
                num_qubits: 34,
                edges: (0..34)
                    .flat_map(|i| (0..34).filter(move |&j| i != j).map(move |j| (i, j)))
                    .collect(),
                topology: distributed::TopologyType::AllToAll,
            },
            noise_model: None,
        },
        network_config: NetworkConfig {
            endpoint: "https://braket.us-east-1.amazonaws.com".to_string(),
            credentials: Credentials {
                auth_type: AuthenticationType::ApiKey,
                api_key: Some("your_aws_access_key".to_string()),
                token: None,
                username_password: None,
            },
            timeouts: TimeoutConfig {
                connection_timeout: 10.0,
                request_timeout: 300.0,
                total_timeout: 900.0,
            },
            retry_policy: RetryPolicy {
                max_retries: 5,
                base_delay: 1.0,
                backoff_strategy: BackoffStrategy::Exponential { multiplier: 1.5 },
                retryable_errors: vec![
                    distributed::ErrorType::NetworkError,
                    distributed::ErrorType::RateLimited,
                    distributed::ErrorType::ServiceUnavailable,
                ],
            },
        },
    }
}

fn create_hybrid_backend() -> ExecutionBackend {
    ExecutionBackend {
        id: "hybrid_system_1".to_string(),
        backend_type: BackendType::Hybrid {
            quantum_backend: Box::new(BackendType::Hardware {
                vendor: "IonQ".to_string(),
                model: "IonQ Aria".to_string(),
                location: "College Park, MD".to_string(),
            }),
            classical_resources: ClassicalResources {
                cpu_cores: 64,
                memory_gb: 512.0,
                gpus: vec![
                    GPUInfo {
                        model: "NVIDIA A100".to_string(),
                        memory_gb: 40.0,
                        compute_capability: "8.0".to_string(),
                    },
                    GPUInfo {
                        model: "NVIDIA A100".to_string(),
                        memory_gb: 40.0,
                        compute_capability: "8.0".to_string(),
                    },
                ],
                storage_gb: 10000.0,
            },
        },
        status: BackendStatus::Available,
        performance: BackendPerformance {
            max_qubits: 25,
            max_depth: 1000,
            gate_fidelities: {
                let mut fidelities = HashMap::new();
                fidelities.insert("gpi".to_string(), 0.999);
                fidelities.insert("gpi2".to_string(), 0.999);
                fidelities.insert("ms".to_string(), 0.95);
                fidelities
            },
            coherence_times: {
                let mut times = HashMap::new();
                times.insert("T1".to_string(), 10000.0);
                times.insert("T2".to_string(), 1000.0);
                times
            },
            execution_time_model: ExecutionTimeModel {
                base_time: 2.0,
                time_per_gate: 0.1,
                time_per_qubit: 0.01,
                time_per_measurement: 0.5,
                network_latency: 1.0,
            },
            throughput: 5.0,
        },
        queue_info: QueueInfo {
            queue_length: 1,
            estimated_wait_time: 60.0,
            max_queue_size: 20,
            priority_levels: vec![Priority::Normal, Priority::High, Priority::Critical],
        },
        capabilities: BackendCapabilities {
            supported_gates: vec!["gpi".to_string(), "gpi2".to_string(), "ms".to_string()],
            mid_circuit_measurements: true,
            classical_control: true,
            reset_operations: true,
            connectivity: ConnectivityGraph {
                num_qubits: 25,
                edges: (0..25)
                    .flat_map(|i| ((i + 1)..25).map(move |j| (i, j)))
                    .collect(),
                topology: distributed::TopologyType::AllToAll,
            },
            noise_model: Some(NoiseCharacteristics {
                single_qubit_errors: {
                    let mut errors = HashMap::new();
                    errors.insert("gpi".to_string(), 0.0001);
                    errors.insert("gpi2".to_string(), 0.0001);
                    errors
                },
                two_qubit_errors: {
                    let mut errors = HashMap::new();
                    errors.insert("ms".to_string(), 0.005);
                    errors
                },
                measurement_errors: vec![0.001; 25],
                decoherence_times: vec![10000.0; 25],
            }),
        },
        network_config: NetworkConfig {
            endpoint: "https://api.ionq.co/v0.3".to_string(),
            credentials: Credentials {
                auth_type: AuthenticationType::Token,
                api_key: None,
                token: Some("your_ionq_token".to_string()),
                username_password: None,
            },
            timeouts: TimeoutConfig {
                connection_timeout: 15.0,
                request_timeout: 600.0,
                total_timeout: 1800.0,
            },
            retry_policy: RetryPolicy {
                max_retries: 3,
                base_delay: 5.0,
                backoff_strategy: BackoffStrategy::Exponential { multiplier: 2.0 },
                retryable_errors: vec![distributed::ErrorType::NetworkError, distributed::ErrorType::ServiceUnavailable],
            },
        },
    }
}

fn create_test_jobs() -> Vec<DistributedJob<4>> {
    let mut jobs = Vec::new();

    // Job 1: Simple Bell state circuit
    let mut bell_circuit = Circuit::<4>::new();
    bell_circuit.h(Qubit(0)).unwrap();
    bell_circuit.cnot(Qubit(0), Qubit(1)).unwrap();

    jobs.push(DistributedJob {
        id: "bell_state_job".to_string(),
        circuit: bell_circuit,
        parameters: ExecutionParameters {
            shots: 1000,
            optimization_level: 1,
            error_mitigation: vec![ErrorMitigation::ReadoutErrorMitigation],
            result_format: ResultFormat::Counts,
            memory_requirement: None,
        },
        priority: Priority::Normal,
        target_backends: None,
        submitted_at: Instant::now(),
        deadline: None,
    });

    // Job 2: GHZ state circuit
    let mut ghz_circuit = Circuit::<4>::new();
    ghz_circuit.h(Qubit(0)).unwrap();
    ghz_circuit.cnot(Qubit(0), Qubit(1)).unwrap();
    ghz_circuit.cnot(Qubit(1), Qubit(2)).unwrap();
    ghz_circuit.cnot(Qubit(2), Qubit(3)).unwrap();

    jobs.push(DistributedJob {
        id: "ghz_state_job".to_string(),
        circuit: ghz_circuit,
        parameters: ExecutionParameters {
            shots: 5000,
            optimization_level: 2,
            error_mitigation: vec![
                ErrorMitigation::ReadoutErrorMitigation,
                ErrorMitigation::ZeroNoiseExtrapolation,
            ],
            result_format: ResultFormat::Probabilities,
            memory_requirement: Some(1.0), // 1 GB
        },
        priority: Priority::High,
        target_backends: Some(vec![
            "qiskit_aer_simulator".to_string(),
            "aws_braket_sv1".to_string(),
        ]),
        submitted_at: Instant::now(),
        deadline: Some(Instant::now() + std::time::Duration::from_secs(300)),
    });

    // Job 3: Variational circuit
    let mut var_circuit = Circuit::<4>::new();
    for i in 0..4 {
        var_circuit.ry(Qubit(i), 0.5).unwrap();
    }
    for i in 0..3 {
        var_circuit.cnot(Qubit(i), Qubit(i + 1)).unwrap();
    }

    jobs.push(DistributedJob {
        id: "variational_job".to_string(),
        circuit: var_circuit,
        parameters: ExecutionParameters {
            shots: 10000,
            optimization_level: 3,
            error_mitigation: vec![
                ErrorMitigation::CliffordDataRegression,
                ErrorMitigation::SymmetryVerification,
            ],
            result_format: ResultFormat::ExpectationValues,
            memory_requirement: Some(2.0), // 2 GB
        },
        priority: Priority::Critical,
        target_backends: None,
        submitted_at: Instant::now(),
        deadline: None,
    });

    jobs
}
