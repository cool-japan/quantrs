//! Standard Two-Qubit Gates
//!
//! This module provides standard two-qubit gates: CNOT, CZ, SWAP

use crate::error::{MLError, Result};
use crate::torchquantum::{
    CType, NParamsEnum, OpHistoryEntry, TQDevice, TQModule, TQOperator, TQParameter, WiresEnum,
};
use scirs2_core::ndarray::Array2;

/// CNOT gate
#[derive(Debug, Clone)]
pub struct TQCNOT {
    inverse: bool,
    static_mode: bool,
}

impl TQCNOT {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}

impl Default for TQCNOT {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQCNOT {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Err(MLError::InvalidConfiguration(
            "Use apply() instead of forward() for operators".to_string(),
        ))
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
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
        "CNOT"
    }
}

impl TQOperator for TQCNOT {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(0)
    }

    fn get_matrix(&self, _params: Option<&[f64]>) -> Array2<CType> {
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
                CType::new(0.0, 0.0),
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
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
        _params: Option<&[f64]>,
    ) -> Result<()> {
        if wires.len() < 2 {
            return Err(MLError::InvalidConfiguration(
                "CNOT gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(None);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "cnot".to_string(),
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

/// CZ gate
#[derive(Debug, Clone)]
pub struct TQCZ {
    inverse: bool,
    static_mode: bool,
}

impl TQCZ {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}

impl Default for TQCZ {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQCZ {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Err(MLError::InvalidConfiguration(
            "Use apply() instead of forward() for operators".to_string(),
        ))
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
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
        "CZ"
    }
}

impl TQOperator for TQCZ {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(0)
    }

    fn get_matrix(&self, _params: Option<&[f64]>) -> Array2<CType> {
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
                CType::new(-1.0, 0.0),
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
        _params: Option<&[f64]>,
    ) -> Result<()> {
        if wires.len() < 2 {
            return Err(MLError::InvalidConfiguration(
                "CZ gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(None);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "cz".to_string(),
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

/// SWAP gate
#[derive(Debug, Clone)]
pub struct TQSWAP {
    inverse: bool,
    static_mode: bool,
}

impl TQSWAP {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}

impl Default for TQSWAP {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQSWAP {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Err(MLError::InvalidConfiguration(
            "Use apply() instead of forward() for operators".to_string(),
        ))
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
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
        "SWAP"
    }
}

impl TQOperator for TQSWAP {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(0)
    }

    fn get_matrix(&self, _params: Option<&[f64]>) -> Array2<CType> {
        Array2::from_shape_vec(
            (4, 4),
            vec![
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(1.0, 0.0),
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
        _params: Option<&[f64]>,
    ) -> Result<()> {
        if wires.len() < 2 {
            return Err(MLError::InvalidConfiguration(
                "SWAP gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(None);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "swap".to_string(),
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
