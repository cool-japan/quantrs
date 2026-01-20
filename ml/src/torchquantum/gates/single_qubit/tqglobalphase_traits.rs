//! # TQGlobalPhase - Trait Implementations
//!
//! This module contains trait implementations for `TQGlobalPhase`.
//!
//! ## Implemented Traits
//!
//! - `TQModule`
//! - `TQOperator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::super::{
    CType, NParamsEnum, OpHistoryEntry, TQDevice, TQModule, TQOperator, TQParameter, WiresEnum,
};
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayD, IxDyn};

use super::types::TQGlobalPhase;

impl TQModule for TQGlobalPhase {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Ok(())
    }
    fn parameters(&self) -> Vec<TQParameter> {
        self.params.iter().cloned().collect()
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
        "GlobalPhase"
    }
}

impl TQOperator for TQGlobalPhase {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(1)
    }
    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(1)
    }
    fn get_matrix(&self, params: Option<&[f64]>) -> Array2<CType> {
        let phi = params.and_then(|p| p.first().copied()).unwrap_or(0.0);
        let sign = if self.inverse { -1.0 } else { 1.0 };
        let phase = CType::new(0.0, sign * phi).exp();
        Array2::from_shape_vec(
            (2, 2),
            vec![phase, CType::new(0.0, 0.0), CType::new(0.0, 0.0), phase],
        )
        .unwrap_or_else(|_| Array2::eye(2).mapv(|x| CType::new(x, 0.0)))
    }
    fn apply(&mut self, qdev: &mut TQDevice, wires: &[usize]) -> Result<()> {
        let params: Vec<f64> = self
            .params
            .as_ref()
            .map(|p| p.data.iter().copied().collect())
            .unwrap_or_default();
        self.apply_with_params(qdev, wires, Some(&params))
    }
    fn apply_with_params(
        &mut self,
        qdev: &mut TQDevice,
        wires: &[usize],
        params: Option<&[f64]>,
    ) -> Result<()> {
        let matrix = self.get_matrix(params);
        qdev.apply_single_qubit_gate(wires[0], &matrix)?;
        Ok(())
    }
    fn has_params(&self) -> bool {
        self.has_params
    }
    fn trainable(&self) -> bool {
        self.trainable
    }
    fn inverse(&self) -> bool {
        self.inverse
    }
    fn set_inverse(&mut self, inverse: bool) {
        self.inverse = inverse;
    }
}
