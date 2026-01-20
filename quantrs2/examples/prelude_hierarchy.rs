#![allow(clippy::pedantic, clippy::unreadable_literal)]
//! Prelude hierarchy demonstration
//!
//! This example shows how to use different prelude levels for various use cases.
//!
//! Run with: cargo run --example prelude_hierarchy

fn main() {
    println!("=== QuantRS2 Prelude Hierarchy Example ===\n");

    // Example 1: Essentials Prelude (minimal imports)
    println!("1. Essentials Prelude - Minimal Quantum Programming:");
    {
        use quantrs2::prelude::essentials::*;

        // Only core types are available
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);

        println!("   Created qubits: {}, {}", q0.id(), q1.id());
        println!("   QuantRS2 version: {VERSION}");
        println!("   Use case: Minimal quantum type definitions");
    }
    println!();

    // Example 2: Full Prelude (everything enabled with features)
    println!("2. Full Prelude - All Available Features:");
    {
        use quantrs2::prelude::full::*;

        // All enabled features are available
        let q0 = QubitId::new(0);
        println!("   QubitId from full prelude: {}", q0.id());
        println!("   Use case: Complete quantum computing applications");

        #[cfg(feature = "circuit")]
        {
            // Circuit building is available
            let circuit = Circuit::<2>::new();
            println!("   Circuit created with {} qubits", 2);
        }

        #[cfg(feature = "sim")]
        {
            // Simulation is available
            let _simulator = StateVectorSimulator::new();
            println!("   StateVectorSimulator created");
        }

        #[cfg(not(any(feature = "circuit", feature = "sim")))]
        {
            println!("   Note: Enable 'circuit' and 'sim' features for more capabilities");
        }
    }
    println!();

    // Example 3: Demonstrating feature-specific preludes
    println!("3. Feature-Specific Preludes:");

    #[cfg(feature = "circuit")]
    {
        use quantrs2::prelude::circuits::*;
        println!("   circuits prelude: Includes essentials + circuit building");
        let _circuit = Circuit::<3>::new();
    }

    #[cfg(feature = "sim")]
    {
        use quantrs2::prelude::simulation::*;
        println!("   simulation prelude: Includes circuits + simulators");
        let _sim = StateVectorSimulator::new();
    }

    #[cfg(feature = "ml")]
    {
        use quantrs2::prelude::algorithms::*;
        println!("   algorithms prelude: Includes simulation + ML algorithms");
    }

    #[cfg(feature = "device")]
    {
        use quantrs2::prelude::hardware::*;
        println!("   hardware prelude: Includes circuits + device integration");
    }

    #[cfg(feature = "anneal")]
    {
        use quantrs2::prelude::quantum_annealing::*;
        println!("   quantum_annealing prelude: Includes essentials + annealing");
    }

    #[cfg(feature = "tytan")]
    {
        use quantrs2::prelude::tytan::*;
        println!("   tytan prelude: Includes annealing + Tytan DSL");
    }

    println!();

    // Example 4: Choosing the right prelude
    println!("4. Choosing the Right Prelude:");
    println!("   - Use 'essentials' for: Type definitions only");
    println!("   - Use 'circuits' for: Circuit construction");
    println!("   - Use 'simulation' for: Circuit simulation");
    println!("   - Use 'algorithms' for: VQE, QAOA, quantum ML");
    println!("   - Use 'hardware' for: Real quantum device integration");
    println!("   - Use 'quantum_annealing' for: QUBO/Ising problems");
    println!("   - Use 'tytan' for: High-level annealing DSL");
    println!("   - Use 'full' for: Everything (slower compile times)");
    println!();

    // Example 5: Prelude best practices
    println!("5. Best Practices:");
    println!("   ✓ Start with 'essentials' and add features as needed");
    println!("   ✓ Use specific preludes for faster compilation");
    println!("   ✓ Use 'full' only when you need multiple features");
    println!("   ✓ Consider compile time vs convenience trade-offs");
    println!();

    println!("=== Example Complete ===");
}
