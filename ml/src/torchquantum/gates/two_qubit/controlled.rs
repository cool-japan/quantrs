//! Controlled Rotation Gates
//!
//! This module provides controlled rotation gates: CRX, CRY, CRZ

use crate::error::{MLError, Result};
use crate::torchquantum::{
    CType, NParamsEnum, OpHistoryEntry, TQDevice, TQModule, TQOperator, TQParameter, WiresEnum,
};
use scirs2_core::ndarray::{Array2, ArrayD, IxDyn};

/// CRX gate - Controlled RX rotation
#[derive(Debug, Clone)]
pub struct TQCRX {
    params: Option<TQParameter>,
    has_params: bool,
    trainable: bool,
    inverse: bool,
    static_mode: bool,
}

impl TQCRX {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(ArrayD::zeros(IxDyn(&[1, 1])), "crx_theta"))
        } else {
            None
        };

        Self {
            params,
            has_params,
            trainable,
            inverse: false,
            static_mode: false,
        }
    }

    pub fn with_init_params(mut self, theta: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = theta;
        }
        self
    }
}

impl Default for TQCRX {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl TQModule for TQCRX {
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
        "CRX"
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut p) = self.params {
            p.zero_grad();
        }
    }
}

impl TQOperator for TQCRX {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(1)
    }

    fn get_matrix(&self, params: Option<&[f64]>) -> Array2<CType> {
        let theta = params
            .and_then(|p| p.first().copied())
            .or_else(|| self.params.as_ref().map(|p| p.data[[0, 0]]))
            .unwrap_or(0.0);

        let theta = if self.inverse { -theta } else { theta };
        let half_theta = theta / 2.0;
        let c = half_theta.cos();
        let s = half_theta.sin();

        // CRX(θ) - controlled rotation about X-axis
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
                CType::new(c, 0.0),
                CType::new(0.0, -s),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, -s),
                CType::new(c, 0.0),
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
                "CRX gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(params);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "crx".to_string(),
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

/// CRY gate - Controlled RY rotation
#[derive(Debug, Clone)]
pub struct TQCRY {
    params: Option<TQParameter>,
    has_params: bool,
    trainable: bool,
    inverse: bool,
    static_mode: bool,
}

impl TQCRY {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(ArrayD::zeros(IxDyn(&[1, 1])), "cry_theta"))
        } else {
            None
        };

        Self {
            params,
            has_params,
            trainable,
            inverse: false,
            static_mode: false,
        }
    }

    pub fn with_init_params(mut self, theta: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = theta;
        }
        self
    }
}

impl Default for TQCRY {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl TQModule for TQCRY {
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
        "CRY"
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut p) = self.params {
            p.zero_grad();
        }
    }
}

impl TQOperator for TQCRY {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(1)
    }

    fn get_matrix(&self, params: Option<&[f64]>) -> Array2<CType> {
        let theta = params
            .and_then(|p| p.first().copied())
            .or_else(|| self.params.as_ref().map(|p| p.data[[0, 0]]))
            .unwrap_or(0.0);

        let theta = if self.inverse { -theta } else { theta };
        let half_theta = theta / 2.0;
        let c = half_theta.cos();
        let s = half_theta.sin();

        // CRY(θ) - controlled rotation about Y-axis
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
                CType::new(c, 0.0),
                CType::new(-s, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(s, 0.0),
                CType::new(c, 0.0),
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
                "CRY gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(params);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "cry".to_string(),
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

/// CRZ gate - Controlled RZ rotation
#[derive(Debug, Clone)]
pub struct TQCRZ {
    params: Option<TQParameter>,
    has_params: bool,
    trainable: bool,
    inverse: bool,
    static_mode: bool,
}

impl TQCRZ {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(ArrayD::zeros(IxDyn(&[1, 1])), "crz_theta"))
        } else {
            None
        };

        Self {
            params,
            has_params,
            trainable,
            inverse: false,
            static_mode: false,
        }
    }

    pub fn with_init_params(mut self, theta: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = theta;
        }
        self
    }
}

impl Default for TQCRZ {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl TQModule for TQCRZ {
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
        "CRZ"
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut p) = self.params {
            p.zero_grad();
        }
    }
}

impl TQOperator for TQCRZ {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(1)
    }

    fn get_matrix(&self, params: Option<&[f64]>) -> Array2<CType> {
        let theta = params
            .and_then(|p| p.first().copied())
            .or_else(|| self.params.as_ref().map(|p| p.data[[0, 0]]))
            .unwrap_or(0.0);

        let theta = if self.inverse { -theta } else { theta };
        let half_theta = theta / 2.0;

        // CRZ(θ) - controlled rotation about Z-axis
        let exp_neg = CType::from_polar(1.0, -half_theta);
        let exp_pos = CType::from_polar(1.0, half_theta);

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
                exp_neg,
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                exp_pos,
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
                "CRZ gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(params);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "crz".to_string(),
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
