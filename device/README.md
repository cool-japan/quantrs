# QuantRS2-Device: Universal Quantum Hardware Interface

[![Crates.io](https://img.shields.io/crates/v/quantrs2-device.svg)](https://crates.io/crates/quantrs2-device)
[![Documentation](https://docs.rs/quantrs2-device/badge.svg)](https://docs.rs/quantrs2-device)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-Device is the comprehensive quantum hardware abstraction layer of the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework, providing seamless connectivity to quantum computers from major cloud providers, with advanced transpilation, optimization, and characterization capabilities for production quantum computing applications.

## Version 0.1.0-beta.2

This release includes:
- Stable APIs for IBM Quantum, Azure Quantum, and AWS Braket
- Enhanced transpilation using SciRS2 v0.1.0-beta.3's graph algorithms for optimal qubit routing
- Improved error handling and asynchronous execution

## Core Features

### Multi-Provider Cloud Integration
- **IBM Quantum Network**: Full integration with IBM Quantum Experience, Qiskit Runtime, and premium backends
- **Azure Quantum Platform**: Access to Microsoft's quantum ecosystem including IonQ, Quantinuum, and Rigetti
- **AWS Braket Service**: Integration with Amazon's quantum computing platform and hardware partners
- **Google Quantum AI**: Planned integration with Google's quantum processors and Cirq ecosystem
- **Unified API**: Consistent interface across all providers with automatic provider detection

### Advanced Hardware Abstraction
- **Device Characterization**: Real-time calibration data integration and noise characterization
- **Hardware-Aware Compilation**: Automatic circuit transpilation with provider-specific optimizations
- **Pulse-Level Control**: Direct pulse sequence generation for supported hardware
- **Topology-Aware Routing**: Intelligent qubit mapping with SWAP insertion optimization
- **Cross-Talk Mitigation**: Advanced scheduling algorithms to minimize quantum interference

### Production-Ready Features
- **Asynchronous Execution**: Non-blocking job submission with real-time status monitoring
- **Batch Processing**: Efficient execution of multiple circuits with resource optimization
- **Error Mitigation**: Zero-noise extrapolation, virtual distillation, and readout correction
- **Cost Optimization**: Intelligent job scheduling and resource allocation across providers
- **Fault Tolerance**: Automatic retry mechanisms with exponential backoff strategies

### Performance & Reliability
- **Circuit Validation**: Pre-flight checks for hardware compatibility and constraints
- **Queue Management**: Real-time queue monitoring with estimated execution times
- **Resource Estimation**: Accurate cost and time predictions for circuit execution
- **Result Caching**: Intelligent caching of circuit results and calibration data
- **Monitoring & Telemetry**: Comprehensive logging and performance metrics

## Usage

### IBM Quantum

```rust
use quantrs2_circuit::builder::Circuit;
use quantrs2_device::{create_ibm_client, create_ibm_device, prelude::*};

#[cfg(feature = "ibm")]
async fn run_on_ibm() -> Result<(), Box<dyn std::error::Error>> {
    // Create a bell state circuit
    let mut circuit = Circuit::<2>::new();
    circuit.h(0)?
           .cnot(0, 1)?;
    
    // Get API token (from environment or config)
    let token = std::env::var("IBM_QUANTUM_TOKEN")?;
    
    // Connect to IBM Quantum
    let device = create_ibm_device(&token, "ibmq_qasm_simulator", None).await?;
    
    // Execute circuit with 1024 shots
    let result = device.execute_circuit(&circuit, 1024).await?;
    
    // Process results
    for (outcome, count) in result.counts {
        println!("{}: {}", outcome, count);
    }
    
    Ok(())
}
```

### Azure Quantum

```rust
use quantrs2_circuit::builder::Circuit;
use quantrs2_device::{create_azure_client, create_azure_device, prelude::*};

#[cfg(feature = "azure")]
async fn run_on_azure() -> Result<(), Box<dyn std::error::Error>> {
    // Create a circuit
    let mut circuit = Circuit::<2>::new();
    circuit.h(0)?
           .cnot(0, 1)?;
    
    // Azure credentials
    let token = std::env::var("AZURE_TOKEN")?;
    let subscription = std::env::var("AZURE_SUBSCRIPTION_ID")?;
    let resource_group = "my-resource-group";
    let workspace = "my-workspace";
    
    // Create Azure client
    let client = create_azure_client(
        &token, 
        &subscription, 
        resource_group, 
        workspace, 
        None
    )?;
    
    // Connect to a specific provider's device
    let device = create_azure_device(
        client, 
        "ionq.simulator", 
        Some("ionq"), 
        None
    ).await?;
    
    // Execute circuit
    let result = device.execute_circuit(&circuit, 500).await?;
    
    // Process results
    for (outcome, count) in result.counts {
        println!("{}: {}", outcome, count);
    }
    
    Ok(())
}
```

### AWS Braket

```rust
use quantrs2_circuit::builder::Circuit;
use quantrs2_device::{create_aws_client, create_aws_device, prelude::*};

#[cfg(feature = "aws")]
async fn run_on_aws() -> Result<(), Box<dyn std::error::Error>> {
    // Create a circuit
    let mut circuit = Circuit::<2>::new();
    circuit.h(0)?
           .cnot(0, 1)?;
    
    // AWS credentials
    let access_key = std::env::var("AWS_ACCESS_KEY_ID")?;
    let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY")?;
    let bucket = "my-quantum-results";
    
    // Create AWS client
    let client = create_aws_client(
        &access_key, 
        &secret_key, 
        Some("us-east-1"), 
        bucket, 
        None
    )?;
    
    // Connect to SV1 simulator
    let device = create_aws_device(
        client,
        "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
        None
    ).await?;
    
    // Execute circuit
    let result = device.execute_circuit(&circuit, 1000).await?;
    
    // Process results
    for (outcome, count) in result.counts {
        println!("{}: {}", outcome, count);
    }
    
    Ok(())
}
```

## Module Structure

- **ibm.rs / ibm_device.rs**: IBM Quantum client and device support
- **azure.rs / azure_device.rs**: Azure Quantum client and device support
- **aws.rs / aws_device.rs**: AWS Braket client and device support
- **transpiler.rs**: Circuit transformation for hardware constraints

## Core Types

- `QuantumDevice`: Trait representing quantum hardware capabilities
- `CircuitExecutor`: Trait for devices that can run quantum circuits
- `CircuitResult`: Standard result format for quantum executions

## Feature Flags

- **ibm**: Enables IBM Quantum connectivity
- **azure**: Enables Azure Quantum connectivity
- **aws**: Enables AWS Braket connectivity

Each feature flag can be enabled independently to minimize dependencies.

## Technical Details

- Async/await is used for non-blocking network operations
- Each provider has specific authentication and configuration requirements
- The circuit transpiler adapts circuits to provider-specific gate sets
- Error types are standardized across providers

## Future Plans

See [TODO.md](TODO.md) for planned features.

## Integration with Other QuantRS2 Modules

This module is designed to work seamlessly with:
- [quantrs2-core](../core/README.md): Uses core types for quantum operations
- [quantrs2-circuit](../circuit/README.md): Executes circuits on real hardware
- [quantrs2-sim](../sim/README.md): Local simulators match hardware behavior

## License

This project is licensed under either:

- [Apache License, Version 2.0](../LICENSE-APACHE)
- [MIT License](../LICENSE-MIT)

at your option.