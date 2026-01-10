//! # TQI - Trait Implementations
//!
//! This module contains trait implementations for `TQI`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `TQModule`
//! - `TQOperator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::super::{
    CType, NParamsEnum, OpHistoryEntry, TQDevice, TQModule, TQOperator, TQParameter, WiresEnum,
};
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayD, IxDyn};

use super::types::TQI;

impl Default for TQI {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQI {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Ok(())
    }
    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
    }
    fn n_wires(&self) -> Option<usize> {
        Some(1)
    }
    fn set_n_wires(&mut self, _n_wires: usize) {}
    fn is_static_mode(&self) -> bool {
        self.static_mode
    }
    fn static_on(&mut self) {
        self.static_mode = true;
    }
    fn static_off(&mut self) {
        self.static_mode = false;
    }
    fn name(&self) -> &str {
        "I"
    }
}

impl TQOperator for TQI {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(1)
    }
    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(0)
    }
    fn get_matrix(&self, _params: Option<&[f64]>) -> Array2<CType> {
        Array2::eye(2).mapv(|x| CType::new(x, 0.0))
    }
    fn apply(&mut self, qdev: &mut TQDevice, wires: &[usize]) -> Result<()> {
        let matrix = self.get_matrix(None);
        qdev.apply_single_qubit_gate(wires[0], &matrix)?;
        Ok(())
    }
    fn apply_with_params(
        &mut self,
        qdev: &mut TQDevice,
        wires: &[usize],
        _params: Option<&[f64]>,
    ) -> Result<()> {
        self.apply(qdev, wires)
    }
    fn has_params(&self) -> bool {
        false
    }
    fn trainable(&self) -> bool {
        false
    }
    fn inverse(&self) -> bool {
        self.inverse
    }
    fn set_inverse(&mut self, inverse: bool) {
        self.inverse = inverse;
    }
}
