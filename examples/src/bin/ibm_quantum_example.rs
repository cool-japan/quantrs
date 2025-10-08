use std::env;
use std::error::Error;

use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::prelude::{QubitId, Register};
use quantrs2_device::{
    create_ibm_client, create_ibm_device, ibm_device::IBMDeviceConfig, CircuitExecutor,
    QuantumDevice,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("IBM Quantum Integration Example");

    // Get IBM Quantum API token from environment variable
    let token =
        env::var("IBM_QUANTUM_TOKEN").expect("IBM_QUANTUM_TOKEN environment variable must be set");

    // Create IBM Quantum client
    let client = create_ibm_client(&token)?;

    // List available backends
    println!("\nListing available IBM Quantum backends...");
    let backends = client.list_backends().await?;

    println!("Found {} backends:", backends.len());
    for backend in &backends {
        println!(
            "  - {} ({} qubits, simulator: {})",
            backend.name, backend.n_qubits, backend.simulator
        );
    }

    // Select a simulator backend for this example
    // In a real application, you might want to use a real quantum device
    let simulator_backend = backends
        .iter()
        .find(|b| b.simulator && b.n_qubits >= 3)
        .expect("No suitable simulator backend found");

    println!(
        "\nUsing backend: {} ({} qubits)",
        simulator_backend.name, simulator_backend.n_qubits
    );

    // Configure the IBM Quantum device
    let config = IBMDeviceConfig {
        default_shots: 1024,
        optimization_level: 1,
        timeout_seconds: 300,
        optimize_routing: true,
        max_parallel_jobs: 5,
    };

    // Create IBM Quantum device
    let device = create_ibm_device(&token, &simulator_backend.name, Some(config)).await?;

    // Check if the device is available
    let available = device.is_available().await?;
    println!("Device available: {available}");

    if !available {
        println!("Selected backend is not available. Please try again later.");
        return Ok(());
    }

    // Create a simple circuit: Bell state
    println!("\nCreating Bell state circuit...");
    let mut circuit = Circuit::<2>::new();
    circuit.h(QubitId::new(0))?;
    circuit.cnot(QubitId::new(0), QubitId::new(1))?;

    // Check if we can execute this circuit on the selected backend
    let can_execute = device.can_execute_circuit(&circuit).await?;
    println!("Can execute circuit: {can_execute}");

    if !can_execute {
        println!("Circuit cannot be executed on the selected backend.");
        return Ok(());
    }

    // Get estimated queue time
    let queue_time = device.estimated_queue_time(&circuit).await?;
    println!("Estimated queue time: {} seconds", queue_time.as_secs());

    // Execute the circuit on the IBM Quantum device
    println!("\nExecuting circuit on IBM Quantum...");
    println!("This may take some time depending on the queue...");

    let shots = 1024;
    let result = device.execute_circuit(&circuit, shots).await?;

    // Display results
    println!("\nCircuit execution completed!");
    println!("Shots: {}", result.shots);
    println!("Measurement counts:");

    for (state, count) in &result.counts {
        let probability = *count as f64 / result.shots as f64;
        println!("  |{}⟩: {} ({:.2}%)", state, count, probability * 100.0);
    }

    // Results should show approximately 50% |00⟩ and 50% |11⟩ for a Bell state

    // Execute multiple circuits in parallel
    println!("\nCreating and executing multiple circuits in parallel...");

    // Create three simple circuits
    let mut circuit1 = Circuit::<1>::new();
    circuit1.h(QubitId::new(0)); // |+⟩ state

    let mut circuit2 = Circuit::<1>::new();
    circuit2.x(QubitId::new(0)); // |1⟩ state

    let mut circuit3 = Circuit::<1>::new();
    circuit3.h(QubitId::new(0))?;
    circuit3.x(QubitId::new(0))?; // |−⟩ state

    // Execute circuits in parallel
    let circuits = vec![&circuit1, &circuit2, &circuit3];
    let results = device.execute_circuits(circuits, shots).await?;

    // Display results for each circuit
    for (i, result) in results.iter().enumerate() {
        println!("\nResults for circuit {}:", i + 1);
        println!("Shots: {}", result.shots);
        println!("Measurement counts:");

        for (state, count) in &result.counts {
            let probability = *count as f64 / result.shots as f64;
            println!("  |{}⟩: {} ({:.2}%)", state, count, probability * 100.0);
        }
    }

    println!("\nIBM Quantum integration example completed successfully!");

    Ok(())
}
