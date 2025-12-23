//! Memory estimation and capacity planning example
//!
//! This example demonstrates how to estimate memory requirements for quantum
//! simulations and plan system capacity accordingly.
//!
//! Run with: cargo run --example memory_estimation

use quantrs2::utils;

fn main() {
    println!("=== QuantRS2 Memory Estimation Example ===\n");

    // 1. Estimate memory for various qubit counts
    println!("1. Memory Requirements for State Vector Simulation:");
    println!("   (Complex64 = 16 bytes per amplitude, 2^n amplitudes)\n");

    let qubit_counts = vec![5, 10, 15, 20, 25, 26, 27, 28, 29, 30];

    println!("   Qubits  │  Memory Required  │  Human Readable");
    println!("   ────────┼──────────────────┼─────────────────");

    for &qubits in &qubit_counts {
        let memory_bytes = utils::estimate_statevector_memory(qubits);
        let memory_str = utils::format_memory(memory_bytes);
        println!(
            "   {:6}  │  {:15} B │  {}",
            qubits, memory_bytes, memory_str
        );
    }
    println!();

    // 2. Determine maximum qubits for available memory
    println!("2. Maximum Qubits for Available Memory:\n");

    let memory_configs = vec![
        ("1 GB", 1 * 1024 * 1024 * 1024),
        ("4 GB", 4 * 1024 * 1024 * 1024),
        ("8 GB", 8 * 1024 * 1024 * 1024),
        ("16 GB", 16 * 1024 * 1024 * 1024),
        ("32 GB", 32 * 1024 * 1024 * 1024),
        ("64 GB", 64 * 1024 * 1024 * 1024),
        ("128 GB", 128 * 1024 * 1024 * 1024),
    ];

    println!("   Available Memory │ Max Qubits │ Memory Used");
    println!("   ─────────────────┼────────────┼─────────────");

    for (name, memory) in memory_configs {
        let max_qubits = utils::max_qubits_for_memory(memory);
        let used_memory = utils::estimate_statevector_memory(max_qubits);
        let used_str = utils::format_memory(used_memory);
        println!("   {:15}  │  {:8}   │  {}", name, max_qubits, used_str);
    }
    println!();

    // 3. Validate specific configurations
    println!("3. Validating Specific Configurations:\n");

    let test_cases = vec![
        (20, 32 * 1024 * 1024, "20 qubits in 32 MB"),
        (25, 1024 * 1024 * 1024, "25 qubits in 1 GB"),
        (30, 16 * 1024 * 1024 * 1024, "30 qubits in 16 GB"),
        (35, 16 * 1024 * 1024 * 1024, "35 qubits in 16 GB"),
    ];

    for (qubits, memory, description) in test_cases {
        let is_valid = utils::is_valid_qubit_count(qubits, memory);
        let status = if is_valid { "✓" } else { "✗" };
        let required = utils::estimate_statevector_memory(qubits);
        let required_str = utils::format_memory(required);

        println!("   {} {} (requires {})", status, description, required_str);
    }
    println!();

    // 4. Practical recommendations
    println!("4. Practical Recommendations:");
    println!();
    println!("   System Type         │ Recommended Max Qubits");
    println!("   ────────────────────┼───────────────────────");
    println!("   Laptop (8 GB)       │  26 qubits");
    println!("   Workstation (32 GB) │  28 qubits");
    println!("   Server (128 GB)     │  30 qubits");
    println!("   HPC Node (512 GB)   │  32 qubits");
    println!();

    // 5. Memory overhead considerations
    println!("5. Memory Overhead Considerations:");
    println!();
    println!("   State vector simulation memory includes:");
    println!("   - Primary state vector: 2^n × 16 bytes");
    println!("   - Temporary buffers: ~2-3× state vector size");
    println!("   - Gate matrices: negligible");
    println!("   - Runtime overhead: ~10-20 MB");
    println!();
    println!("   Recommended: Allocate 3-4× the theoretical requirement");
    println!();

    // 6. Alternative simulation methods
    println!("6. For Larger Systems, Consider:");
    println!();
    println!("   Method              │ Max Qubits │ Memory     │ Restrictions");
    println!("   ────────────────────┼────────────┼────────────┼──────────────────");
    println!("   State Vector        │  ~30       │ O(2^n)     │ None");
    println!("   Tensor Network      │  ~50+      │ O(poly)    │ Low entanglement");
    println!("   Stabilizer          │  ~100+     │ O(n²)      │ Clifford gates only");
    println!("   GPU (CUDA)          │  ~35       │ O(2^n)     │ Requires GPU");
    println!();

    // 7. Example calculation for a specific use case
    println!("7. Example: Planning for a 25-qubit VQE Simulation:");
    let vqe_qubits = 25;
    let state_memory = utils::estimate_statevector_memory(vqe_qubits);
    let total_memory = state_memory * 4; // 4× for overhead

    println!();
    println!("   Circuit size: {} qubits", vqe_qubits);
    println!("   State vector: {}", utils::format_memory(state_memory));
    println!(
        "   Total recommended: {}",
        utils::format_memory(total_memory)
    );
    println!(
        "   Minimum system: {} RAM",
        utils::format_memory(total_memory)
    );
    println!();

    println!("=== Example Complete ===");
}
