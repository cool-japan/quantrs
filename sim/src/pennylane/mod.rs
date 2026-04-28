//! PennyLane device backend for QuantRS2.
//!
//! This module provides a JSON-protocol device that PennyLane can call to
//! execute quantum circuits on QuantRS2's state-vector simulator.
//!
//! ## Usage (from Python via PyO3 or via the Rust API)
//!
//! ```no_run
//! use quantrs2_sim::pennylane::device::{PennyLaneCircuit, QuantRS2Device};
//!
//! let device = QuantRS2Device::new();
//! let json = r#"{"num_wires":2,"operations":[{"name":"Hadamard","wires":[0],"params":[]},{"name":"CNOT","wires":[0,1],"params":[]}],"observables":[]}"#;
//! let result_json = device.execute_json(json).unwrap();
//! ```

pub mod device;
pub mod wire;

pub use device::{
    DeviceError, PennyLaneCircuit, PennyLaneObservable, PennyLaneOperation, PennyLaneResult,
    QuantRS2Device,
};
pub use wire::WireMap;
