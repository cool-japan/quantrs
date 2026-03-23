//! # EnhancedQASMCompiler - caching Methods
//!
//! This module contains method implementations for `EnhancedQASMCompiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
    register::Register,
};

use super::types::CompilationResult;

use super::enhancedqasmcompiler_type::EnhancedQASMCompiler;

impl EnhancedQASMCompiler {
    pub(super) fn cache_result(
        &self,
        source: &str,
        result: &CompilationResult,
    ) -> QuantRS2Result<()> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|e| QuantRS2Error::RuntimeError(format!("Cache lock poisoned: {e}")))?;
        cache.insert(source.to_string(), result.clone());
        Ok(())
    }
}
