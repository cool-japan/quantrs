use scirs2_core::Complex64;
use quantrs2_circuit::prelude::*;
use quantrs2_core::parametric::*;
use quantrs2_core::prelude::*;
use quantrs2_sim::prelude::*;

use std::f64::consts::PI;
use std::sync::Arc;

fn main() -> QuantRS2Result<()> {
    println!("Parametric Gates Demonstration");
    println!("==============================\n");

    // Example 1: Using symbolic parameters for a rotation gate
    let qubit = QubitId::new(0);
    let rx_symbolic = ParametricRotationX::new_symbolic(qubit, "theta");

    println!("Example 1: Symbolic Parameter Gate");
    println!("----------------------------------");
    println!("Created a parameterized RX gate with symbolic parameter 'theta'");
    println!("Parameter names: {:?}", rx_symbolic.parameter_names());
    println!("Gate is parameterized: {}", rx_symbolic.is_parameterized());

    // Bind the parameter to a value
    let values = [("theta".to_string(), PI / 2.0)];
    let rx_bound = rx_symbolic.bind(&values)?;

    println!("\nAfter binding 'theta' to π/2:");
    if let Some(matrix) = rx_bound.matrix().ok() {
        println!("Matrix representation:");
        print_matrix(&matrix);
    }

    // Example 2: Circuit with parametric gates
    println!("\nExample 2: Circuit with Parametric Gates");
    println!("---------------------------------------");

    let mut circuit = Circuit::<2>::new();

    // Add some standard gates
    circuit.h(0);
    circuit.x(1);

    // Create a parametric rotation gate
    let ry_param = ParametricRotationY::new_symbolic(QubitId::new(0), "angle");

    // We need to convert the parametric gate to a compatible type for the circuit
    // This would be easier with proper integration in the Circuit type
    println!("Created a circuit with H and X gates, plus a parameterized RY gate");
    println!("Parameter names: {:?}", ry_param.parameter_names());

    // Create multiple circuits with different parameter values
    println!("\nSimulating circuits with different parameter values:");

    for angle in [0.0, PI / 4.0, PI / 2.0, PI] {
        let bound_gate = ry_param.bind(&[("angle".to_string(), angle)])?;

        // For demonstration, we'll manually create new circuits with the bound parameter
        // In a real implementation, this would be handled more elegantly
        let mut param_circuit = circuit.clone();

        // Apply the bound gate directly
        // Convert Box<dyn ParametricGate> to Arc<dyn GateOp + Send + Sync>
        // Since ParametricGate extends GateOp, we need to convert the trait object
        let gate_arc: Arc<dyn GateOp + Send + Sync> =
            Arc::from(bound_gate as Box<dyn GateOp + Send + Sync>);
        param_circuit.add_gate_arc(gate_arc)?;

        // Simulate the circuit
        let simulator = StateVectorSimulator::new();
        let result = simulator.run(&param_circuit)?;

        println!("\nWith angle = {}:", angle);
        println!("Final state:");
        print_statevector(&result.amplitudes());
    }
    // Example 3: Multi-parameter gate (U gate)
    println!("\nExample 3: Multi-Parameter Gate (U Gate)");
    println!("---------------------------------------");

    let u_gate = ParametricU::new_symbolic(QubitId::new(0), "theta", "phi", "lambda");

    println!("Created a U gate with symbolic parameters");
    println!("Parameter names: {:?}", u_gate.parameter_names());

    // Bind some parameters but leave others symbolic
    let partially_bound = u_gate.bind(&[
        ("theta".to_string(), PI / 2.0),
        ("phi".to_string(), PI / 4.0),
    ])?;

    println!("\nAfter binding 'theta' to π/2 and 'phi' to π/4:");
    println!(
        "Remaining unbound parameters: {:?}",
        partially_bound
            .parameter_names()
            .iter()
            .filter(|&name| name == "lambda")
            .collect::<Vec<_>>()
    );

    // Fully bind all parameters
    let fully_bound = partially_bound.bind(&[("lambda".to_string(), PI / 3.0)])?;

    println!("\nAfter binding all parameters:");
    if let Ok(matrix) = fully_bound.matrix() {
        println!("Matrix representation:");
        print_matrix(&matrix);
    }

    Ok(())
}

// Helper function to print a matrix
fn print_matrix(matrix: &[Complex64]) {
    let size = (matrix.len() as f64).sqrt() as usize;

    for i in 0..size {
        print!("[ ");
        for j in 0..size {
            let idx = i * size + j;
            let element = matrix[idx];

            if element.im.abs() < 1e-10 {
                print!("{:.5} ", element.re);
            } else if element.re.abs() < 1e-10 {
                print!("{:.5}i ", element.im);
            } else {
                print!("{:.5}+{:.5}i ", element.re, element.im);
            }
        }
        println!("]");
    }
}

// Helper function to print a state vector
fn print_statevector(state: &[Complex64]) {
    println!("State vector: {}", state.len());
    for (i, &amplitude) in state.iter().enumerate() {
        if amplitude.norm() > 1e-10 {
            let probability = amplitude.norm_sqr();
            let binary = format!("{:b}", i).chars().collect::<Vec<_>>();
            let padding = state.len().ilog2() as usize - binary.len();
            let padded_binary: String = std::iter::repeat('0')
                .take(padding)
                .chain(binary.into_iter())
                .collect();

            println!(
                "|{}⟩: {} (probability: {:.5})",
                padded_binary,
                format_complex(amplitude),
                probability
            );
        }
    }
}

// Helper function to format a complex number nicely
fn format_complex(c: Complex64) -> String {
    if c.im.abs() < 1e-10 {
        format!("{:.5}", c.re)
    } else if c.re.abs() < 1e-10 {
        format!("{:.5}i", c.im)
    } else if c.im < 0.0 {
        format!("{:.5}{:.5}i", c.re, c.im)
    } else {
        format!("{:.5}+{:.5}i", c.re, c.im)
    }
}
