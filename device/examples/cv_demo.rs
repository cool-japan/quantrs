//! Continuous Variable Quantum Computing Demo
//!
//! This example demonstrates the basic functionality of the CV quantum system.

use quantrs2_device::continuous_variable::{
    create_gaussian_cv_device, CVGateSequence, Complex, GaussianState,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Continuous Variable Quantum Computing Demo ===");

    // Create a Gaussian CV device
    let mut device = create_gaussian_cv_device(2, None)?;
    println!("Created CV device with {} modes", device.num_modes);

    // Connect to the device
    device.connect().await?;
    println!("Connected to CV device");

    // Apply some basic operations
    println!("\nApplying displacement operation...");
    device.displacement(0, 1.0, 0.5).await?;

    println!("Applying squeezing operation...");
    device.squeezing(1, 1.0, 0.0).await?;

    println!("Applying beamsplitter...");
    device.beamsplitter(0, 1, 0.5, 0.0).await?;

    // Perform measurements
    println!("\nPerforming homodyne measurement...");
    let homodyne_result = device.homodyne_measurement(0, 0.0).await?;
    println!("Homodyne result: {homodyne_result:.3}");

    println!("Performing heterodyne measurement...");
    let heterodyne_result = device.heterodyne_measurement(1).await?;
    println!(
        "Heterodyne result: {:.3} + {:.3}i",
        heterodyne_result.real, heterodyne_result.imag
    );

    // Get device diagnostics
    let diagnostics = device.get_diagnostics().await;
    println!("\nDevice diagnostics:");
    println!("  Connected: {}", diagnostics.is_connected);
    println!("  Modes: {}", diagnostics.num_modes);
    println!("  Total measurements: {}", diagnostics.total_measurements);
    println!(
        "  Average squeezing: {:.3} dB",
        diagnostics.average_squeezing
    );
    println!("  System purity: {:.3}", diagnostics.system_purity);

    // Demonstrate CV gate sequences
    println!("\n=== CV Gate Sequence Demo ===");
    let mut gate_sequence = CVGateSequence::new(2);

    gate_sequence.displacement(0, Complex::new(2.0, 0.0))?;
    gate_sequence.squeezing(1, 0.5, 0.0)?;
    gate_sequence.beamsplitter(0, 1, 0.5, 0.0)?;

    println!(
        "Created gate sequence with {} gates",
        gate_sequence.gate_count()
    );
    println!("Sequence depth: {}", gate_sequence.depth());
    println!("Is Gaussian: {}", gate_sequence.is_gaussian());

    // Execute gate sequence on a state
    let mut state = GaussianState::vacuum_state(2);
    gate_sequence.execute_on_state(&mut state)?;
    println!("Executed gate sequence on vacuum state");

    // Show entanglement measures
    let entanglement = state.calculate_entanglement_measures();
    println!("Entanglement measures:");
    println!(
        "  Logarithmic negativity: {:.3}",
        entanglement.logarithmic_negativity
    );
    println!("  EPR correlation: {:.3}", entanglement.epr_correlation);

    println!("\n=== CV Demo Complete ===");
    Ok(())
}
