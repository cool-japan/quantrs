//! Three-qubit gate implementations
//!
//! This module provides all three-qubit gates including:
//! - Toffoli (CCX) gate
//! - CSWAP (Fredkin) gate
//! - CCZ gate

use super::super::{
    CType, NParamsEnum, OpHistoryEntry, TQDevice, TQModule, TQOperator, TQParameter, WiresEnum,
};
use crate::error::{MLError, Result};
use scirs2_core::ndarray::Array2;

/// CSWAP gate - Controlled SWAP (Fredkin gate)
#[derive(Debug, Clone)]
pub struct TQCSWAP {
    inverse: bool,
    static_mode: bool,
}

impl TQCSWAP {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}

impl Default for TQCSWAP {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQCSWAP {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Err(MLError::InvalidConfiguration(
            "Use apply() instead of forward() for operators".to_string(),
        ))
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(3)
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
        "CSWAP"
    }
}

impl TQOperator for TQCSWAP {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(3)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(0)
    }

    fn get_matrix(&self, _params: Option<&[f64]>) -> Array2<CType> {
        // CSWAP (Fredkin) 8x8 matrix
        // swaps qubits 1 and 2 when qubit 0 is |1>
        let mut matrix = Array2::eye(8).mapv(|x| CType::new(x, 0.0));
        // Swap |101> <-> |110> (indices 5 <-> 6)
        matrix[[5, 5]] = CType::new(0.0, 0.0);
        matrix[[5, 6]] = CType::new(1.0, 0.0);
        matrix[[6, 5]] = CType::new(1.0, 0.0);
        matrix[[6, 6]] = CType::new(0.0, 0.0);
        matrix
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
        if wires.len() < 3 {
            return Err(MLError::InvalidConfiguration(
                "CSWAP gate requires exactly 3 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(None);
        qdev.apply_multi_qubit_gate(wires, &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "cswap".to_string(),
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

/// Toffoli gate (CCX) - Controlled-Controlled-X gate
#[derive(Debug, Clone)]
pub struct TQToffoli {
    inverse: bool,
    static_mode: bool,
}

impl TQToffoli {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}

impl Default for TQToffoli {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQToffoli {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Err(MLError::InvalidConfiguration(
            "Use apply() instead of forward() for operators".to_string(),
        ))
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(3)
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
        "Toffoli"
    }
}

impl TQOperator for TQToffoli {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(3)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(0)
    }

    fn get_matrix(&self, _params: Option<&[f64]>) -> Array2<CType> {
        // Toffoli (CCX) 8x8 matrix
        // Flips qubit 2 when qubits 0 and 1 are both |1>
        let mut matrix = Array2::eye(8).mapv(|x| CType::new(x, 0.0));
        // Swap |110> <-> |111> (indices 6 <-> 7)
        matrix[[6, 6]] = CType::new(0.0, 0.0);
        matrix[[6, 7]] = CType::new(1.0, 0.0);
        matrix[[7, 6]] = CType::new(1.0, 0.0);
        matrix[[7, 7]] = CType::new(0.0, 0.0);
        matrix
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
        if wires.len() < 3 {
            return Err(MLError::InvalidConfiguration(
                "Toffoli gate requires exactly 3 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(None);
        qdev.apply_multi_qubit_gate(wires, &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "toffoli".to_string(),
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

/// CCZ gate - Controlled-Controlled-Z gate
#[derive(Debug, Clone)]
pub struct TQCCZ {
    inverse: bool,
    static_mode: bool,
}

impl TQCCZ {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}

impl Default for TQCCZ {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQCCZ {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Err(MLError::InvalidConfiguration(
            "Use apply() instead of forward() for operators".to_string(),
        ))
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(3)
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
        "CCZ"
    }
}

impl TQOperator for TQCCZ {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(3)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(0)
    }

    fn get_matrix(&self, _params: Option<&[f64]>) -> Array2<CType> {
        // CCZ 8x8 matrix - applies Z when both controls are |1>
        let mut matrix = Array2::eye(8).mapv(|x| CType::new(x, 0.0));
        // Apply -1 phase to |111> (index 7)
        matrix[[7, 7]] = CType::new(-1.0, 0.0);
        matrix
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
        if wires.len() < 3 {
            return Err(MLError::InvalidConfiguration(
                "CCZ gate requires exactly 3 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(None);
        qdev.apply_multi_qubit_gate(wires, &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "ccz".to_string(),
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

/// Type aliases for TorchQuantum compatibility
pub type TQCCX = TQToffoli;
pub type TQFredkin = TQCSWAP;
