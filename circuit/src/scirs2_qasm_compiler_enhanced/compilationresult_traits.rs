//! # CompilationResult - Trait Implementations
//!
//! This module contains trait implementations for `CompilationResult`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CompilationResult;
use std::fmt;

impl fmt::Display for CompilationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Compilation Result:")?;
        writeln!(f, "  Compilation time: {:?}", self.compilation_time)?;
        writeln!(f, "  Generated targets: {}", self.generated_code.len())?;
        writeln!(f, "  Gates: {}", self.statistics.gate_count)?;
        writeln!(f, "  Qubits: {}", self.statistics.qubit_count)?;
        writeln!(
            f,
            "  Optimizations applied: {}",
            self.optimizations_applied.len()
        )?;
        writeln!(f, "  Warnings: {}", self.warnings.len())?;
        Ok(())
    }
}
