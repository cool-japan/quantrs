//! Quantum gate implementations (TorchQuantum-compatible)
//!
//! This module provides all quantum gate implementations including:
//! - Single-qubit gates (RX, RY, RZ, H, X, Y, Z, S, T, SX, U1, U2, U3, etc.)
//! - Two-qubit gates (CNOT, CZ, SWAP, iSWAP, ECR, CY, DCX, etc.)
//! - Parameterized two-qubit gates (RXX, RYY, RZZ, RZX, XXMinusYY, XXPlusYY, CPhase)
//! - Controlled rotation gates (CRX, CRY, CRZ, CH)
//! - Three-qubit gates (Toffoli/CCX, CSWAP/Fredkin, CCZ)

mod single_qubit;
mod three_qubit;
mod two_qubit;

// Re-export all gates
pub use single_qubit::*;
pub use three_qubit::*;
pub use two_qubit::*;
