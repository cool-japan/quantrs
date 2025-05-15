# QuantRS2-Device: Quantum Hardware Connectivity

[![Crates.io](https://img.shields.io/crates/v/quantrs2-device.svg)](https://crates.io/crates/quantrs2-device)
[![Documentation](https://docs.rs/quantrs2-device/badge.svg)](https://docs.rs/quantrs2-device)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-Device is part of the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework, providing connectivity to real quantum hardware from major providers including IBM Quantum, Azure Quantum, and AWS Braket.

## Features

- **Unified Hardware API**: Consistent interface across multiple quantum providers
- **IBM Quantum Integration**: Connect to IBM Quantum Experience and Qiskit Runtime
- **Azure Quantum Support**: Access Microsoft's quantum platform and partners
- **AWS Braket Connectivity**: Integration with Amazon's quantum computing service
- **Circuit Transpilation**: Adapt circuits to hardware constraints
- **Async Execution**: Non-blocking job submission and monitoring
- **Result Processing**: Standardized format for quantum computation results

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

- **ibm.rs / ibm_device.rs**: IBM Quantum client and device implementations
- **azure.rs / azure_device.rs**: Azure Quantum client and device implementations
- **aws.rs / aws_device.rs**: AWS Braket client and device implementations
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

## Implementation Notes

- Async/await is used for non-blocking network operations
- Each provider has specific authentication and configuration requirements
- The circuit transpiler adapts circuits to provider-specific gate sets
- Error types are standardized across providers

## Future Plans

See [TODO.md](TODO.md) for planned improvements and features.

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