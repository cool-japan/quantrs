//! # TQPhaseShift2 - Trait Implementations
//!
//! This module contains trait implementations for `TQPhaseShift2`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `TQModule`
//! - `TQOperator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use crate::torchquantum::{
    CType, NParamsEnum, OpHistoryEntry, TQDevice, TQModule, TQOperator, TQParameter, WiresEnum,
};
use scirs2_core::ndarray::{Array2, ArrayD, IxDyn};

use super::types::TQPhaseShift2;

impl Default for TQPhaseShift2 {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl TQModule for TQPhaseShift2 {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Err(MLError::InvalidConfiguration(
            "Use apply() instead of forward() for operators".to_string(),
        ))
    }
    fn parameters(&self) -> Vec<TQParameter> {
        self.params.iter().cloned().collect()
    }
    fn n_wires(&self) -> Option<usize> {
        Some(2)
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
        "PhaseShift2"
    }
    fn zero_grad(&mut self) {
        if let Some(ref mut p) = self.params {
            p.zero_grad();
        }
    }
}

impl TQOperator for TQPhaseShift2 {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }
    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(1)
    }
    fn get_matrix(&self, params: Option<&[f64]>) -> Array2<CType> {
        let phi = params
            .and_then(|p| p.first().copied())
            .or_else(|| self.params.as_ref().map(|p| p.data[[0, 0]]))
            .unwrap_or(0.0);
        let phi = if self.inverse { -phi } else { phi };
        let exp_i_phi = CType::from_polar(1.0, phi);
        Array2::from_shape_vec(
            (4, 4),
            vec![
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                exp_i_phi,
            ],
        )
        .unwrap_or_else(|_| Array2::eye(4).mapv(|x| CType::new(x, 0.0)))
    }
    fn apply(&mut self, qdev: &mut TQDevice, wires: &[usize]) -> Result<()> {
        self.apply_with_params(qdev, wires, None)
    }
    fn apply_with_params(
        &mut self,
        qdev: &mut TQDevice,
        wires: &[usize],
        params: Option<&[f64]>,
    ) -> Result<()> {
        if wires.len() < 2 {
            return Err(MLError::InvalidConfiguration(
                "PhaseShift2 gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(params);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;
        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "phaseshift2".to_string(),
                wires: wires.to_vec(),
                params: params.map(|p| p.to_vec()),
                inverse: self.inverse,
                trainable: self.trainable,
            });
        }
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
