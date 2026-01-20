//! # TQT - Trait Implementations
//!
//! This module contains trait implementations for `TQT`.
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
use std::f64::consts::PI;

use super::types::TQT;

impl Default for TQT {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQT {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Err(MLError::InvalidConfiguration(
            "Use apply() instead of forward() for operators".to_string(),
        ))
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
        "T"
    }
}

impl TQOperator for TQT {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(1)
    }
    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(0)
    }
    fn get_matrix(&self, _params: Option<&[f64]>) -> Array2<CType> {
        let phase = if self.inverse {
            CType::from_polar(1.0, -PI / 4.0)
        } else {
            CType::from_polar(1.0, PI / 4.0)
        };
        Array2::from_shape_vec(
            (2, 2),
            vec![
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                phase,
            ],
        )
        .unwrap_or_else(|_| Array2::eye(2).mapv(|x| CType::new(x, 0.0)))
    }
    fn apply(&mut self, qdev: &mut TQDevice, wires: &[usize]) -> Result<()> {
        self.apply_with_params(qdev, wires, None)
    }
    fn apply_with_params(
        &mut self,
        qdev: &mut TQDevice,
        wires: &[usize],
        _params: Option<&[f64]>,
    ) -> Result<()> {
        if wires.is_empty() {
            return Err(MLError::InvalidConfiguration(
                "T gate requires exactly 1 wire".to_string(),
            ));
        }
        let matrix = self.get_matrix(None);
        qdev.apply_single_qubit_gate(wires[0], &matrix)?;
        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "t".to_string(),
                wires: wires.to_vec(),
                params: None,
                inverse: self.inverse,
                trainable: false,
            });
        }
        Ok(())
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
