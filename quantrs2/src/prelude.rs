//! # QuantRS2 Unified Prelude Module
//!
//! This module provides hierarchical prelude modules for different use cases, allowing
//! developers to import exactly what they need for their specific quantum computing tasks.
//!
//! ## Prelude Hierarchy
//!
//! The prelude modules are organized in a hierarchy, where each level builds upon the previous:
//!
//! ```text
//! essentials     → Minimal imports for basic quantum programming
//!    ↓
//! circuits       → Circuit construction and manipulation
//!    ↓
//! simulation     → Quantum simulation capabilities
//!    ↓
//! algorithms     → Advanced quantum algorithms and ML
//!    ↓
//! hardware       → Real quantum hardware integration
//!    ↓
//! full           → All QuantRS2 features
//! ```
//!
//! ## Usage Examples
//!
//! ### Minimal Quantum Programming
//!
//! For basic quantum operations, use the `essentials` prelude:
//!
//! ```rust,ignore
//! use quantrs2::prelude::essentials::*;
//!
//! // QubitId, Register, GateOp, Complex64 are now available
//! let q0 = QubitId::new(0);
//! let mut register = Register::<2>::new();
//! ```
//!
//! ### Circuit Construction
//!
//! For building quantum circuits:
//!
//! ```rust,ignore
//! use quantrs2::prelude::circuits::*;
//!
//! // Includes essentials + circuit building
//! let mut circuit = Circuit::<2>::new();
//! circuit.h(QubitId::new(0))?;
//! circuit.cx(QubitId::new(0), QubitId::new(1))?;
//! ```
//!
//! ### Quantum Simulation
//!
//! For simulating quantum circuits:
//!
//! ```rust,ignore
//! use quantrs2::prelude::simulation::*;
//!
//! // Includes circuits + simulators
//! let mut simulator = StateVectorSimulator::new();
//! let result = simulator.run(&circuit)?;
//! ```
//!
//! ### Algorithm Development
//!
//! For implementing quantum algorithms with ML:
//!
//! ```rust,ignore
//! use quantrs2::prelude::algorithms::*;
//!
//! // Includes simulation + ML algorithms
//! // VQE, QAOA, quantum neural networks available
//! ```
//!
//! ### Hardware Programming
//!
//! For real quantum hardware:
//!
//! ```rust,ignore
//! use quantrs2::prelude::hardware::*;
//!
//! // Includes circuits + device integration
//! // IBM, Azure, AWS Braket connectors available
//! ```
//!
//! ### Full Feature Set
//!
//! For comprehensive quantum computing applications:
//!
//! ```rust,ignore
//! use quantrs2::prelude::full::*;
//!
//! // All QuantRS2 features available
//! ```

/// Minimal imports for basic quantum programming.
///
/// This prelude includes only the most essential types needed for quantum computing:
/// - `QubitId`: Type-safe qubit identifiers
/// - `QubitSet`: Collections of qubits
/// - `Register<N>`: Quantum register holding N qubits
/// - `GateOp`: Universal gate trait
/// - `Complex64`: Complex numbers from SciRS2
/// - `QuantRS2Error`, `QuantRS2Result<T>`: Error handling
///
/// # Example
///
/// ```rust,ignore
/// use quantrs2::prelude::essentials::*;
///
/// fn apply_hadamard(qubit: QubitId) -> QuantRS2Result<()> {
///     // Basic quantum operations
///     Ok(())
/// }
/// ```
pub mod essentials {
    // Core quantum types - always available
    pub use crate::core::api::prelude::essentials::*;

    // Error handling
    pub use crate::core::error::{QuantRS2Error, QuantRS2Result};

    // Version information
    pub use crate::{QUANTRS2_VERSION, VERSION};
}

/// Circuit construction and manipulation prelude.
///
/// Includes everything from `essentials` plus:
/// - `Circuit<N>`: Type-safe quantum circuit builder
/// - Circuit optimization and analysis tools
/// - Gate definitions and operations
/// - QASM import/export
/// - Circuit validation and verification
///
/// # Example
///
/// ```rust,ignore
/// use quantrs2::prelude::circuits::*;
///
/// fn create_bell_state() -> QuantRS2Result<Circuit<2>> {
///     let mut circuit = Circuit::<2>::new();
///     circuit.h(QubitId::new(0))?;
///     circuit.cx(QubitId::new(0), QubitId::new(1))?;
///     Ok(circuit)
/// }
/// ```
#[cfg(feature = "circuit")]
pub mod circuits {
    // Include all essentials
    pub use super::essentials::*;

    // Circuit building and manipulation
    pub use crate::circuit::builder::{Circuit, Simulator};

    // Gate operations
    pub use crate::core::gate::GateOp;

    // Circuit analysis and optimization
    // Note: These are available through circuit module's public API
    // Users should access via quantrs2::circuit::optimizer::CircuitOptimizer
}

/// Quantum simulation prelude.
///
/// Includes everything from `circuits` plus:
/// - `StateVectorSimulator`: CPU-based state vector simulation
/// - `StabilizerSimulator`: Efficient Clifford simulation
/// - `TensorNetworkSimulator`: Low-entanglement circuits
/// - Noise models and error simulation
/// - GPU-accelerated backends
///
/// # Example
///
/// ```rust,ignore
/// use quantrs2::prelude::simulation::*;
///
/// fn simulate_circuit(circuit: &Circuit<4>) -> QuantRS2Result<SimulatorResult<4>> {
///     let mut simulator = StateVectorSimulator::new();
///     simulator.run(circuit)
/// }
/// ```
#[cfg(feature = "sim")]
pub mod simulation {
    // Include all circuits functionality
    #[cfg(feature = "circuit")]
    pub use super::circuits::*;

    // Core simulation types
    pub use crate::sim::api::prelude::essentials::*;

    // Commonly used simulators
    pub use crate::sim::statevector::StateVectorSimulator;

    // Note: Additional simulators like StabilizerSimulator and TensorNetworkSimulator
    // are available through the sim module's public API
    // Access them via quantrs2::sim::stabilizer::StabilizerSimulator
}

/// Advanced quantum algorithms and machine learning prelude.
///
/// Includes everything from `simulation` plus:
/// - VQE (Variational Quantum Eigensolver)
/// - QAOA (Quantum Approximate Optimization Algorithm)
/// - Quantum neural networks
/// - Quantum GANs
/// - Gradient computation and optimization
///
/// # Example
///
/// ```rust,ignore
/// use quantrs2::prelude::algorithms::*;
///
/// fn optimize_molecule() -> QuantRS2Result<f64> {
///     let hamiltonian = /* ... */;
///     let vqe = VQE::new(hamiltonian);
///     vqe.optimize()
/// }
/// ```
#[cfg(feature = "ml")]
pub mod algorithms {
    // Include all simulation functionality
    #[cfg(feature = "sim")]
    pub use super::simulation::*;

    // ML-specific algorithms
    #[cfg(feature = "ml")]
    pub use crate::ml::*;

    // Note: VQE and QAOA are available from the ml module when ml feature is enabled
}

/// Real quantum hardware integration prelude.
///
/// Includes everything from `circuits` plus:
/// - IBM Quantum API client
/// - Azure Quantum integration
/// - AWS Braket connector
/// - Device topology and routing
/// - Calibration and characterization
///
/// # Example
///
/// ```rust,ignore
/// use quantrs2::prelude::hardware::*;
///
/// async fn run_on_ibm(circuit: &Circuit<5>) -> QuantRS2Result<MeasurementResult> {
///     let client = IBMClient::from_env()?;
///     client.execute(circuit).await
/// }
/// ```
#[cfg(feature = "device")]
pub mod hardware {
    // Include circuits functionality
    #[cfg(feature = "circuit")]
    pub use super::circuits::*;

    // Device integration
    #[cfg(feature = "device")]
    pub use crate::device::*;
}

/// Quantum annealing prelude.
///
/// Includes everything from `essentials` plus:
/// - QUBO problem formulation
/// - Ising model representation
/// - D-Wave quantum annealer integration
/// - Classical annealing simulators
///
/// # Example
///
/// ```rust,ignore
/// use quantrs2::prelude::quantum_annealing::*;
///
/// fn solve_maxcut(graph: &Graph) -> QuantRS2Result<Vec<bool>> {
///     let qubo = graph.to_qubo();
///     let solver = SimulatedAnnealer::new();
///     solver.solve(&qubo)
/// }
/// ```
#[cfg(feature = "anneal")]
pub mod quantum_annealing {
    // Include essentials
    pub use super::essentials::*;

    // Annealing functionality
    #[cfg(feature = "anneal")]
    pub use crate::anneal::*;
}

/// High-level quantum annealing with Tytan API.
///
/// Includes everything from `quantum_annealing` plus:
/// - Intuitive problem modeling DSL
/// - Automatic QUBO compilation
/// - Hybrid classical-quantum solvers
///
/// # Example
///
/// ```rust,ignore
/// use quantrs2::prelude::tytan::*;
///
/// fn solve_tsp(cities: usize) -> QuantRS2Result<Vec<usize>> {
///     let problem = TytanProblem::traveling_salesman(cities);
///     problem.solve()
/// }
/// ```
#[cfg(feature = "tytan")]
pub mod tytan {
    // Include quantum annealing
    #[cfg(feature = "anneal")]
    pub use super::quantum_annealing::*;

    // Tytan-specific functionality
    #[cfg(feature = "tytan")]
    pub use crate::tytan::*;
}

/// Research tools and advanced features prelude.
///
/// Includes everything from `simulation` and `algorithms` plus:
/// - Tensor network operations
/// - Topological quantum computing
/// - ZX-calculus optimization
/// - Symbolic computation
/// - Advanced debugging and profiling
///
/// # Example
///
/// ```rust,ignore
/// use quantrs2::prelude::research::*;
///
/// fn optimize_with_zx(circuit: &Circuit<10>) -> QuantRS2Result<Circuit<10>> {
///     let zx_diagram = circuit.to_zx();
///     let optimized = zx_diagram.optimize()?;
///     optimized.to_circuit()
/// }
/// ```
#[cfg(any(feature = "sim", feature = "ml"))]
pub mod research {
    // Include simulation if available
    #[cfg(feature = "sim")]
    pub use super::simulation::*;

    // Include algorithms if available
    #[cfg(feature = "ml")]
    pub use super::algorithms::*;

    // Advanced research features from core (if available)
    // Note: These modules may not be exported from core's public API
    // TODO: Expose these from core's public API in future versions

    // Symbolic computation
    #[cfg(feature = "symengine")]
    pub use crate::symengine::*;
}

/// Full QuantRS2 feature set prelude.
///
/// Includes all available features based on enabled feature flags.
/// This is the most comprehensive prelude but may increase compilation time.
///
/// # Example
///
/// ```rust,ignore
/// use quantrs2::prelude::full::*;
///
/// // All QuantRS2 functionality is available
/// ```
pub mod full {
    // Core essentials - always available
    pub use super::essentials::*;

    // Circuit features
    #[cfg(feature = "circuit")]
    pub use super::circuits::*;

    // Simulation features
    #[cfg(feature = "sim")]
    pub use super::simulation::*;

    // Algorithm and ML features
    #[cfg(feature = "ml")]
    pub use super::algorithms::*;

    // Hardware features
    #[cfg(feature = "device")]
    pub use super::hardware::*;

    // Annealing features
    #[cfg(feature = "anneal")]
    pub use super::quantum_annealing::*;

    // Tytan features
    #[cfg(feature = "tytan")]
    pub use super::tytan::*;

    // Research features
    #[cfg(any(feature = "sim", feature = "ml"))]
    pub use super::research::*;
}

// Provide a default prelude that matches common use cases
/// Default prelude - equivalent to `simulation` when sim feature is enabled,
/// otherwise equivalent to `circuits` when circuit feature is enabled,
/// otherwise equivalent to `essentials`.
#[cfg(feature = "sim")]
pub use simulation as default;

#[cfg(all(not(feature = "sim"), feature = "circuit"))]
pub use circuits as default;

#[cfg(all(not(feature = "sim"), not(feature = "circuit")))]
pub use essentials as default;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_essentials_imports() {
        // Test that essentials module is always available
        use essentials::*;

        // QubitId should be available
        let _q = QubitId::new(0);

        // Version constants should be available
        assert!(!essentials::VERSION.is_empty());
        assert_eq!(essentials::VERSION, essentials::QUANTRS2_VERSION);
    }

    #[cfg(feature = "circuit")]
    #[test]
    fn test_circuits_imports() {
        use circuits::*;

        // Circuit should be available
        let _circuit = Circuit::<2>::new();

        // QubitId from essentials should also be available
        let _q = QubitId::new(0);
    }

    #[cfg(feature = "sim")]
    #[test]
    fn test_simulation_imports() {
        use simulation::*;

        // StateVectorSimulator should be available
        let _simulator = StateVectorSimulator::new();

        // Circuit from circuits should be available
        let _circuit = Circuit::<2>::new();

        // QubitId from essentials should be available
        let _q = QubitId::new(0);
    }

    #[test]
    fn test_full_imports() {
        use full::*;

        // At minimum, essentials should be available
        let _q = QubitId::new(0);
        assert!(!crate::version::VERSION.is_empty());
    }
}
