//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::torchquantum::{
    gates::{
        TQHadamard, TQPauliX, TQPauliY, TQPauliZ, TQRx, TQRy, TQRz, TQCNOT, TQCRX, TQCRY, TQCRZ,
        TQCZ, TQRXX, TQRYY, TQRZX, TQRZZ, TQS, TQSWAP, TQSX, TQT,
    },
    CType, TQDevice, TQModule, TQOperator, TQParameter,
};

/// Helper to create single-qubit gates
pub(super) fn create_single_qubit_gate(
    name: &str,
    has_params: bool,
    trainable: bool,
) -> Box<dyn TQOperator> {
    match name.to_lowercase().as_str() {
        "rx" => Box::new(TQRx::new(has_params, trainable)),
        "ry" => Box::new(TQRy::new(has_params, trainable)),
        "rz" => Box::new(TQRz::new(has_params, trainable)),
        "h" | "hadamard" => Box::new(TQHadamard::new()),
        "x" | "paulix" => Box::new(TQPauliX::new()),
        "y" | "pauliy" => Box::new(TQPauliY::new()),
        "z" | "pauliz" => Box::new(TQPauliZ::new()),
        "s" => Box::new(TQS::new()),
        "t" => Box::new(TQT::new()),
        "sx" => Box::new(TQSX::new()),
        _ => Box::new(TQRy::new(has_params, trainable)),
    }
}
/// Helper to create two-qubit gates
pub(super) fn create_two_qubit_gate(
    name: &str,
    has_params: bool,
    trainable: bool,
) -> Box<dyn TQOperator> {
    match name.to_lowercase().as_str() {
        "cnot" | "cx" => Box::new(TQCNOT::new()),
        "cz" => Box::new(TQCZ::new()),
        "swap" => Box::new(TQSWAP::new()),
        "rxx" => Box::new(TQRXX::new(has_params, trainable)),
        "ryy" => Box::new(TQRYY::new(has_params, trainable)),
        "rzz" => Box::new(TQRZZ::new(has_params, trainable)),
        "rzx" => Box::new(TQRZX::new(has_params, trainable)),
        "crx" => Box::new(TQCRX::new(has_params, trainable)),
        "cry" => Box::new(TQCRY::new(has_params, trainable)),
        "crz" => Box::new(TQCRZ::new(has_params, trainable)),
        _ => Box::new(TQCNOT::new()),
    }
}
