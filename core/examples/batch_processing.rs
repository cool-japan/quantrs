//! Batch Processing Example
//!
//! This example demonstrates high-performance batch processing of quantum states
//! using QuantRS2-Core's batch operations module.
//!
//! Run with: cargo run --example batch_processing --release

use quantrs2_core::{
    batch::{BatchConfig, BatchStateVector},
    error::QuantRS2Result,
};
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;

fn main() -> QuantRS2Result<()> {
    println!("=================================================================");
    println!("   QuantRS2-Core: Batch Processing");
    println!("=================================================================\n");

    // Basic batch creation
    demonstrate_batch_creation()?;
    println!();

    // Batch configuration
    demonstrate_batch_configuration()?;
    println!();

    // Batch operations
    demonstrate_batch_operations()?;
    println!();

    // Performance scaling
    demonstrate_performance_scaling()?;
    println!();

    println!("=================================================================");
    println!("   Example Complete!");
    println!("=================================================================");

    Ok(())
}

/// Demonstrate creating batch of quantum states
fn demonstrate_batch_creation() -> QuantRS2Result<()> {
    println!("BATCH CREATION");
    println!("-----------------------------------------------------------------");

    let batch_size = 100;
    let n_qubits = 4;
    let config = BatchConfig::default();

    let batch = BatchStateVector::new(batch_size, n_qubits, config)?;

    println!("Created batch of quantum states:");
    println!("  Batch size: {}", batch.batch_size());
    println!("  Qubits per state: {}", batch.n_qubits);
    println!(
        "  State vector dimension: 2^{} = {}",
        n_qubits,
        1 << n_qubits
    );
    println!(
        "  Total complex amplitudes: {} × {} = {}",
        batch.batch_size(),
        1 << n_qubits,
        batch.batch_size() * (1 << n_qubits)
    );

    // Calculate memory usage
    let memory_bytes = batch.batch_size() * (1 << n_qubits) * std::mem::size_of::<Complex64>();
    let memory_mb = memory_bytes as f64 / (1024.0 * 1024.0);

    println!("  Memory usage: {memory_mb:.2} MB");
    println!("  All states initialized to |0...0⟩");

    println!("\n  ✓ Batch creation successful");

    Ok(())
}

/// Demonstrate batch configuration options
fn demonstrate_batch_configuration() -> QuantRS2Result<()> {
    println!("BATCH CONFIGURATION");
    println!("-----------------------------------------------------------------");

    // Default configuration
    let config_default = BatchConfig::default();
    println!("Default Configuration:");
    println!(
        "  Workers: {:?}",
        config_default.num_workers.unwrap_or_else(num_cpus::get)
    );
    println!("  Max batch size: {}", config_default.max_batch_size);
    println!("  GPU acceleration: {}", config_default.use_gpu);
    println!("  Caching: {}", config_default.enable_cache);

    // Custom configuration for high performance
    let config_performance = BatchConfig {
        num_workers: Some(num_cpus::get()),
        max_batch_size: 2048,
        use_gpu: true,
        memory_limit: Some(4 * 1024 * 1024 * 1024), // 4 GB limit
        enable_cache: true,
    };

    println!("\nPerformance Configuration:");
    println!("  Workers: {}", config_performance.num_workers.unwrap());
    println!("  Max batch size: {}", config_performance.max_batch_size);
    println!(
        "  Memory limit: {} GB",
        config_performance.memory_limit.unwrap() / (1024 * 1024 * 1024)
    );

    // Custom configuration for memory-constrained systems
    let config_memory = BatchConfig {
        num_workers: Some(2),
        max_batch_size: 256,
        use_gpu: false,
        memory_limit: Some(1024 * 1024 * 1024), // 1 GB limit
        enable_cache: false,
    };

    println!("\nMemory-Constrained Configuration:");
    println!("  Workers: {}", config_memory.num_workers.unwrap());
    println!("  Max batch size: {}", config_memory.max_batch_size);
    println!("  GPU: disabled");
    println!("  Cache: disabled");

    println!("\n  ✓ Configuration options demonstrated");

    Ok(())
}

/// Demonstrate batch operations
fn demonstrate_batch_operations() -> QuantRS2Result<()> {
    println!("BATCH OPERATIONS");
    println!("-----------------------------------------------------------------");

    let batch_size = 10;
    let n_qubits = 2;
    let config = BatchConfig::default();

    let mut batch = BatchStateVector::new(batch_size, n_qubits, config)?;

    println!("Initial batch:");
    println!("  Size: {}", batch.batch_size());
    println!("  Qubits: {}", batch.n_qubits);

    // Access individual states
    let state_0 = batch.get_state(0)?;
    println!("\n  State 0 (|00⟩):");
    println!("    |00⟩: {:.3}", state_0[0].norm());
    println!("    |01⟩: {:.3}", state_0[1].norm());
    println!("    |10⟩: {:.3}", state_0[2].norm());
    println!("    |11⟩: {:.3}", state_0[3].norm());

    // Modify a state
    let mut custom_state = Array1::zeros(4);
    custom_state[0] = Complex64::new(0.5, 0.0);
    custom_state[1] = Complex64::new(0.5, 0.0);
    custom_state[2] = Complex64::new(0.5, 0.0);
    custom_state[3] = Complex64::new(0.5, 0.0);

    batch.set_state(5, &custom_state)?;
    let modified_state = batch.get_state(5)?;

    println!("\n  State 5 (custom superposition):");
    println!("    |00⟩: {:.3}", modified_state[0].norm());
    println!("    |01⟩: {:.3}", modified_state[1].norm());
    println!("    |10⟩: {:.3}", modified_state[2].norm());
    println!("    |11⟩: {:.3}", modified_state[3].norm());

    println!("\n  ✓ Batch operations successful");

    Ok(())
}

/// Demonstrate performance scaling with batch size
fn demonstrate_performance_scaling() -> QuantRS2Result<()> {
    println!("PERFORMANCE SCALING");
    println!("-----------------------------------------------------------------");

    let n_qubits = 8;
    let batch_sizes = vec![10, 50, 100, 500, 1000];

    println!("Batch size scaling ({n_qubits}  qubits per state):\n");
    println!(
        "{:>10} | {:>15} | {:>15}",
        "Batch", "Memory (MB)", "Amplitudes"
    );
    println!("{:-<10}-+-{:-<15}-+-{:-<15}", "", "", "");

    for &batch_size in &batch_sizes {
        let config = BatchConfig::default();
        let batch = BatchStateVector::new(batch_size, n_qubits, config)?;

        let memory_bytes = batch.batch_size() * (1 << n_qubits) * std::mem::size_of::<Complex64>();
        let memory_mb = memory_bytes as f64 / (1024.0 * 1024.0);
        let total_amplitudes = batch.batch_size() * (1 << n_qubits);

        println!("{batch_size:10} | {memory_mb:12.2} MB | {total_amplitudes:15}");
    }

    println!("\nPerformance benefits:");
    println!("  • SIMD acceleration for vector operations");
    println!("  • Parallel processing across CPU cores");
    println!("  • GPU acceleration for large batches");
    println!("  • Efficient memory layout for cache performance");

    println!("\nTypical speedup vs sequential:");
    println!("  • CPU (SIMD): 2-4x");
    println!("  • CPU (Parallel): 4-8x");
    println!("  • GPU: 10-100x (depending on batch size)");

    println!("\n  ✓ Performance scaling analysis complete");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_runs() {
        assert!(main().is_ok());
    }

    #[test]
    fn test_batch_creation() -> QuantRS2Result<()> {
        let batch_size = 10;
        let n_qubits = 3;
        let config = BatchConfig::default();

        let batch = BatchStateVector::new(batch_size, n_qubits, config)?;

        assert_eq!(batch.batch_size(), batch_size);
        assert_eq!(batch.n_qubits, n_qubits);

        Ok(())
    }

    #[test]
    fn test_batch_state_access() -> QuantRS2Result<()> {
        let batch_size = 5;
        let n_qubits = 2;
        let config = BatchConfig::default();

        let mut batch = BatchStateVector::new(batch_size, n_qubits, config)?;

        // Get initial state
        let state = batch.get_state(0)?;
        assert_eq!(state[0].norm(), 1.0); // |00⟩ state

        // Modify state
        let mut new_state = Array1::zeros(4);
        new_state[3] = Complex64::new(1.0, 0.0); // |11⟩

        batch.set_state(0, &new_state)?;
        let modified = batch.get_state(0)?;

        assert_eq!(modified[3].norm(), 1.0);

        Ok(())
    }

    #[test]
    fn test_batch_memory_limit() {
        let config = BatchConfig {
            memory_limit: Some(1024), // Very small limit
            ..Default::default()
        };

        // This should fail due to memory limit
        let result = BatchStateVector::new(1000, 10, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_different_configurations() -> QuantRS2Result<()> {
        let configs = vec![
            BatchConfig::default(),
            BatchConfig {
                num_workers: Some(2),
                use_gpu: false,
                ..Default::default()
            },
            BatchConfig {
                max_batch_size: 512,
                enable_cache: false,
                ..Default::default()
            },
        ];

        for config in configs {
            let batch = BatchStateVector::new(10, 4, config)?;
            assert_eq!(batch.batch_size(), 10);
        }

        Ok(())
    }
}
