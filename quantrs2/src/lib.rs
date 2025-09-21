#![doc = r#"QuantRS2 facade crate.

This crate provides a single, convenient entry point that re-exports the
public APIs from the QuantRS2 subcrates. Enable features to opt into
additional modules. For example:

    quantrs2 = { version = "*", features = ["full"] }

or selectively:

    quantrs2 = { version = "*", features = ["circuit", "sim"] }
"#]

pub use quantrs2_core as core;

#[cfg(feature = "circuit")]
pub use quantrs2_circuit as circuit;

#[cfg(feature = "sim")]
pub use quantrs2_sim as sim;

#[cfg(feature = "anneal")]
pub use quantrs2_anneal as anneal;

#[cfg(feature = "device")]
pub use quantrs2_device as device;

#[cfg(feature = "ml")]
pub use quantrs2_ml as ml;

#[cfg(feature = "tytan")]
pub use quantrs2_tytan as tytan;

#[cfg(feature = "symengine")]
pub use quantrs2_symengine as symengine;
