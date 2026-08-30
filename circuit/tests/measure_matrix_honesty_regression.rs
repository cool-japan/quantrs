//! Regression test for a silent-fabrication finding in `builder::Measure`.
//!
//! `Measure::matrix()` used to return a hardcoded 2x2 identity matrix even
//! though its own comment stated measurement has no unitary representation.
//! Any generic code path that composes a circuit's unitary via
//! `gate.matrix()` would therefore silently treat a measurement as a no-op
//! identity gate instead of erroring. It must now return an honest
//! `QuantRS2Error::UnsupportedOperation` instead.

use quantrs2_circuit::builder::Measure;
use quantrs2_core::gate::GateOp;
use quantrs2_core::qubit::QubitId;

#[test]
fn test_measure_matrix_errors_honestly_instead_of_fabricating_identity() {
    let measure = Measure {
        target: QubitId::new(0),
    };

    let result = measure.matrix();
    assert!(
        result.is_err(),
        "Measure::matrix() must honestly error, not fabricate an identity matrix"
    );

    let err_message = result.unwrap_err().to_string();
    assert!(
        err_message.contains("no unitary matrix representation") || err_message.contains("Measure"),
        "unexpected error message for Measure::matrix(): {err_message}"
    );
}
