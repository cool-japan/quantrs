//! Parameterized Two-Qubit Rotation Gates
//!
//! This module provides Ising-type rotation gates: RXX, RYY, RZZ, RZX

use crate::error::{MLError, Result};
use crate::torchquantum::{
    CType, NParamsEnum, OpHistoryEntry, TQDevice, TQModule, TQOperator, TQParameter, WiresEnum,
};
use scirs2_core::ndarray::{Array2, ArrayD, IxDyn};

/// RXX gate - Ising XX coupling: exp(-i θ/2 XX)
#[derive(Debug, Clone)]
pub struct TQRXX {
    params: Option<TQParameter>,
    has_params: bool,
    trainable: bool,
    inverse: bool,
    static_mode: bool,
}

impl TQRXX {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(ArrayD::zeros(IxDyn(&[1, 1])), "rxx_theta"))
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

impl Default for TQRXX {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl TQModule for TQRXX {
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
        "RXX"
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut p) = self.params {
            p.zero_grad();
        }
    }
}

impl TQOperator for TQRXX {
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

        // RXX(θ) = [[cos(θ/2), 0, 0, -i sin(θ/2)],
        //           [0, cos(θ/2), -i sin(θ/2), 0],
        //           [0, -i sin(θ/2), cos(θ/2), 0],
        //           [-i sin(θ/2), 0, 0, cos(θ/2)]]
        Array2::from_shape_vec(
            (4, 4),
            vec![
                CType::new(c, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, -s),
                CType::new(0.0, 0.0),
                CType::new(c, 0.0),
                CType::new(0.0, -s),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, -s),
                CType::new(c, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, -s),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
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
                "RXX gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(params);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "rxx".to_string(),
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

/// RYY gate - Ising YY coupling: exp(-i θ/2 YY)
#[derive(Debug, Clone)]
pub struct TQRYY {
    params: Option<TQParameter>,
    has_params: bool,
    trainable: bool,
    inverse: bool,
    static_mode: bool,
}

impl TQRYY {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(ArrayD::zeros(IxDyn(&[1, 1])), "ryy_theta"))
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

impl Default for TQRYY {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl TQModule for TQRYY {
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
        "RYY"
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut p) = self.params {
            p.zero_grad();
        }
    }
}

impl TQOperator for TQRYY {
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

        // RYY(θ) = [[cos(θ/2), 0, 0, i sin(θ/2)],
        //           [0, cos(θ/2), -i sin(θ/2), 0],
        //           [0, -i sin(θ/2), cos(θ/2), 0],
        //           [i sin(θ/2), 0, 0, cos(θ/2)]]
        Array2::from_shape_vec(
            (4, 4),
            vec![
                CType::new(c, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, s),
                CType::new(0.0, 0.0),
                CType::new(c, 0.0),
                CType::new(0.0, -s),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, -s),
                CType::new(c, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, s),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
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
                "RYY gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(params);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "ryy".to_string(),
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

/// RZZ gate - Ising ZZ coupling: exp(-i θ/2 ZZ)
#[derive(Debug, Clone)]
pub struct TQRZZ {
    params: Option<TQParameter>,
    has_params: bool,
    trainable: bool,
    inverse: bool,
    static_mode: bool,
}

impl TQRZZ {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(ArrayD::zeros(IxDyn(&[1, 1])), "rzz_theta"))
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

impl Default for TQRZZ {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl TQModule for TQRZZ {
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
        "RZZ"
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut p) = self.params {
            p.zero_grad();
        }
    }
}

impl TQOperator for TQRZZ {
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

        // RZZ(θ) = diag(e^(-iθ/2), e^(iθ/2), e^(iθ/2), e^(-iθ/2))
        let exp_neg = CType::from_polar(1.0, -half_theta);
        let exp_pos = CType::from_polar(1.0, half_theta);

        Array2::from_shape_vec(
            (4, 4),
            vec![
                exp_neg,
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                exp_pos,
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                exp_pos,
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                exp_neg,
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
                "RZZ gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(params);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "rzz".to_string(),
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

/// RZX gate - Cross-resonance rotation: exp(-i θ/2 ZX)
#[derive(Debug, Clone)]
pub struct TQRZX {
    params: Option<TQParameter>,
    has_params: bool,
    trainable: bool,
    inverse: bool,
    static_mode: bool,
}

impl TQRZX {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(ArrayD::zeros(IxDyn(&[1, 1])), "rzx_theta"))
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

impl Default for TQRZX {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl TQModule for TQRZX {
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
        "RZX"
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut p) = self.params {
            p.zero_grad();
        }
    }
}

impl TQOperator for TQRZX {
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

        // RZX(θ) = [[cos(θ/2), -i sin(θ/2), 0, 0],
        //           [-i sin(θ/2), cos(θ/2), 0, 0],
        //           [0, 0, cos(θ/2), i sin(θ/2)],
        //           [0, 0, i sin(θ/2), cos(θ/2)]]
        Array2::from_shape_vec(
            (4, 4),
            vec![
                CType::new(c, 0.0),
                CType::new(0.0, -s),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, -s),
                CType::new(c, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(c, 0.0),
                CType::new(0.0, s),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, s),
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
                "RZX gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(params);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "rzx".to_string(),
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
