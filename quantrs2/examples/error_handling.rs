//! Error handling and recovery example
//!
//! This example demonstrates comprehensive error handling patterns in QuantRS2.
//!
//! Run with: cargo run --example error_handling

use quantrs2::error::{
    with_context, ErrorCategory, QuantRS2Error, QuantRS2ErrorExt, QuantRS2Result,
};
use quantrs2::prelude::essentials::*;

fn main() {
    println!("=== QuantRS2 Error Handling Example ===\n");

    // 1. Error categorization
    println!("1. Error Categorization:");
    demonstrate_error_categories();
    println!();

    // 2. Error recovery patterns
    println!("2. Error Recovery Patterns:");
    demonstrate_recovery_patterns();
    println!();

    // 3. Error context
    println!("3. Adding Context to Errors:");
    demonstrate_error_context();
    println!();

    // 4. User-friendly messages
    println!("4. User-Friendly Error Messages:");
    demonstrate_user_messages();
    println!();

    // 5. Error type checking
    println!("5. Error Type Checking:");
    demonstrate_error_checking();
    println!();

    println!("=== Example Complete ===");
}

fn demonstrate_error_categories() {
    let errors = vec![
        QuantRS2Error::InvalidQubitId(5),
        QuantRS2Error::CircuitValidationFailed("invalid gate sequence".into()),
        QuantRS2Error::BackendExecutionFailed("simulation timeout".into()),
        QuantRS2Error::NetworkError("connection refused".into()),
        QuantRS2Error::OptimizationFailed("did not converge".into()),
    ];

    for err in errors {
        println!(
            "   Error: {}",
            format!("{err:?}").chars().take(40).collect::<String>()
        );
        println!("   Category: {:?}", err.category());
        println!("   Recoverable: {}", err.is_recoverable());
        println!();
    }
}

fn demonstrate_recovery_patterns() {
    // Example 1: Retry on recoverable errors
    let result = retry_on_network_error();
    match result {
        Ok(()) => println!("   ✓ Operation succeeded (or retry succeeded)"),
        Err(e) => println!("   ✗ Operation failed after retries: {e}"),
    }
    println!();

    // Example 2: Fallback on resource errors
    let result = fallback_on_resource_error();
    match result {
        Ok(msg) => println!("   ✓ {msg}"),
        Err(e) => println!("   ✗ Fallback failed: {e}"),
    }
}

fn retry_on_network_error() -> QuantRS2Result<()> {
    const MAX_RETRIES: u32 = 3;

    for attempt in 1..=MAX_RETRIES {
        // Simulate a network operation that might fail
        let result = simulate_network_operation(attempt);

        match result {
            Ok(()) => {
                println!("   Attempt {attempt}: Success");
                return Ok(());
            }
            Err(e) if e.is_recoverable() => {
                println!("   Attempt {attempt}: Recoverable error, retrying...");
            }
            Err(e) => {
                println!("   Attempt {attempt}: Non-recoverable error");
                return Err(e);
            }
        }
    }

    Err(QuantRS2Error::NetworkError("Max retries exceeded".into()))
}

fn simulate_network_operation(attempt: u32) -> QuantRS2Result<()> {
    // Simulate success on third attempt
    if attempt < 3 {
        Err(QuantRS2Error::NetworkError("timeout".into()))
    } else {
        Ok(())
    }
}

fn fallback_on_resource_error() -> QuantRS2Result<String> {
    // Try with 40 qubits first
    match try_large_simulation(40) {
        Ok(()) => Ok("Used 40-qubit simulation".into()),
        Err(e) if e.is_resource_error() => {
            println!("   Resource error with 40 qubits, falling back to 30...");
            // Fallback to smaller system
            try_large_simulation(30)?;
            Ok("Used 30-qubit simulation (fallback)".into())
        }
        Err(e) => Err(e),
    }
}

fn try_large_simulation(qubits: usize) -> QuantRS2Result<()> {
    if qubits > 35 {
        Err(QuantRS2Error::UnsupportedQubits(
            qubits,
            "Insufficient memory".into(),
        ))
    } else {
        Ok(())
    }
}

fn demonstrate_error_context() {
    // Original error
    let err = QuantRS2Error::InvalidInput("negative parameter".into());
    println!("   Original: {err}");

    // Add context
    let err_with_context = with_context(err, "while initializing VQE optimizer");
    println!("   With context: {err_with_context}");

    // Multiple context levels
    let err = QuantRS2Error::OptimizationFailed("gradient too small".into());
    let err = with_context(err, "in QAOA layer 3");
    let err = with_context(err, "while solving MaxCut problem");
    println!("   Multi-level: {err}");
}

fn demonstrate_user_messages() {
    let errors = vec![
        QuantRS2Error::InvalidQubitId(42),
        QuantRS2Error::NetworkError("DNS resolution failed".into()),
        QuantRS2Error::NoHardwareAvailable("All devices offline".into()),
        QuantRS2Error::OptimizationFailed("Reached max iterations".into()),
    ];

    for err in errors {
        println!("   Error: {err}");
        println!("   User message: {}", err.user_message());
        println!();
    }
}

fn demonstrate_error_checking() {
    let err = QuantRS2Error::InvalidInput("bad value".into());

    println!("   Error: {err}");
    println!("   Is invalid input: {}", err.is_invalid_input());
    println!("   Is recoverable: {}", err.is_recoverable());
    println!("   Is resource error: {}", err.is_resource_error());
    println!("   Category: {:?}", err.category());
    println!();

    // Pattern matching on category
    match err.category() {
        ErrorCategory::Core => println!("   → Handle as core error"),
        ErrorCategory::Hardware => println!("   → Handle as hardware error"),
        ErrorCategory::Runtime => println!("   → Handle as runtime error"),
        _ => println!("   → Handle with default strategy"),
    }
}
