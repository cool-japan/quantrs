extern crate proc_macro;

/// Quantum circuit representation and DSL for the quantrs framework.
///
/// This crate provides types for constructing and manipulating
/// quantum circuits with a fluent API.
pub mod builder;

// Re-exports of commonly used types and traits
pub mod prelude {
    pub use crate::builder::*;
}

// The following should be proc macros, but we'll implement them later
// for now they're just stubs

/// Creates a qubit set for quantum operations
///
/// # Example
///
/// ```ignore
/// let qs = qubits![0, 1, 2];
/// ```
#[macro_export]
macro_rules! qubits {
    ($($id:expr),* $(,)?) => {
        {
            use quantrs_core::qubit::QubitSet;

            let mut qs = QubitSet::new();
            $(qs.add($id);)*
            qs
        }
    };
}

/// Constructs a quantum circuit with a fixed number of qubits
///
/// # Example
///
/// ```ignore
/// let circuit = circuit![4; // 4 qubits
///     h(0),
///     cnot(0, 1),
///     h(2),
///     cnot(2, 3)
/// ];
/// ```
#[macro_export]
macro_rules! circuit {
    ($n:expr) => {
        quantrs_circuit::builder::Circuit::<$n>::new()
    };
}

/// Provides a DSL for constructing quantum circuits
///
/// # Example
///
/// ```ignore
/// use quantrs_circuit::quantum;
///
/// quantum! {
///     let qc = circuit(4);  // 4 qubits
///     qc.h(0);
///     qc.cnot(0, 1);
///     qc.measure_all();
/// }
/// ```
#[macro_export]
macro_rules! quantum {
    ($($tokens:tt)*) => {
        compile_error!("quantum! macro not fully implemented yet");
    };
}
