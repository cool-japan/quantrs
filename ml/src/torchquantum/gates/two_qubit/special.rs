//! Special Two-Qubit Gates
//!
//! This module provides special two-qubit gates including:
//! - iSWAP, SSWAP (sqrt-SWAP)
//! - ECR (Echoed Cross-Resonance)
//! - CY (Controlled-Y), CH (Controlled-H)
//! - DCX (Double CNOT)
//! - XXMinusYY, XXPlusYY (parameterized)
//! - CPhase, CU1

use crate::error::{MLError, Result};
use crate::torchquantum::{
    CType, NParamsEnum, OpHistoryEntry, TQDevice, TQModule, TQOperator, TQParameter, WiresEnum,
};
use scirs2_core::ndarray::{Array2, ArrayD, IxDyn};

/// iSWAP gate - swaps qubits and applies i phase
#[derive(Debug, Clone)]
pub struct TQiSWAP {
    inverse: bool,
    static_mode: bool,
}

impl TQiSWAP {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}

impl Default for TQiSWAP {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQiSWAP {
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
        "iSWAP"
    }
}

impl TQOperator for TQiSWAP {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(0)
    }

    fn get_matrix(&self, _params: Option<&[f64]>) -> Array2<CType> {
        let i_val = if self.inverse {
            CType::new(0.0, -1.0)
        } else {
            CType::new(0.0, 1.0)
        };

        Array2::from_shape_vec(
            (4, 4),
            vec![
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                i_val,
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                i_val,
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
                "iSWAP gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(None);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "iswap".to_string(),
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

/// ECR gate - Echoed Cross-Resonance gate
#[derive(Debug, Clone)]
pub struct TQECR {
    inverse: bool,
    static_mode: bool,
}

impl TQECR {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}

impl Default for TQECR {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQECR {
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
        "ECR"
    }
}

impl TQOperator for TQECR {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(0)
    }

    fn get_matrix(&self, _params: Option<&[f64]>) -> Array2<CType> {
        let scale = 1.0 / std::f64::consts::SQRT_2;
        let (sign_i, sign_neg_i) = if self.inverse {
            (CType::new(0.0, -scale), CType::new(0.0, scale))
        } else {
            (CType::new(0.0, scale), CType::new(0.0, -scale))
        };

        Array2::from_shape_vec(
            (4, 4),
            vec![
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(scale, 0.0),
                sign_i,
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                sign_i,
                CType::new(scale, 0.0),
                CType::new(scale, 0.0),
                sign_neg_i,
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                sign_neg_i,
                CType::new(scale, 0.0),
                CType::new(0.0, 0.0),
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
                "ECR gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(None);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "ecr".to_string(),
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

/// CY gate - Controlled Y gate
#[derive(Debug, Clone)]
pub struct TQCY {
    inverse: bool,
    static_mode: bool,
}

impl TQCY {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}

impl Default for TQCY {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQCY {
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
        "CY"
    }
}

impl TQOperator for TQCY {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(0)
    }

    fn get_matrix(&self, _params: Option<&[f64]>) -> Array2<CType> {
        let (i_val, neg_i_val) = if self.inverse {
            (CType::new(0.0, -1.0), CType::new(0.0, 1.0))
        } else {
            (CType::new(0.0, 1.0), CType::new(0.0, -1.0))
        };

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
                neg_i_val,
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                i_val,
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
                "CY gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(None);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "cy".to_string(),
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

/// SSWAP gate - Square root SWAP gate
#[derive(Debug, Clone)]
pub struct TQSSWAP {
    inverse: bool,
    static_mode: bool,
}

impl TQSSWAP {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}

impl Default for TQSSWAP {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQSSWAP {
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
        "SSWAP"
    }
}

impl TQOperator for TQSSWAP {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(0)
    }

    fn get_matrix(&self, _params: Option<&[f64]>) -> Array2<CType> {
        let (a, b) = if self.inverse {
            (CType::new(0.5, -0.5), CType::new(0.5, 0.5))
        } else {
            (CType::new(0.5, 0.5), CType::new(0.5, -0.5))
        };

        Array2::from_shape_vec(
            (4, 4),
            vec![
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                a,
                b,
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                b,
                a,
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
                "SSWAP gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(None);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "sswap".to_string(),
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

/// DCX gate - Double CNOT gate
#[derive(Debug, Clone)]
pub struct TQDCX {
    inverse: bool,
    static_mode: bool,
}

impl TQDCX {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}

impl Default for TQDCX {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQDCX {
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
        "DCX"
    }
}

impl TQOperator for TQDCX {
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
                CType::new(0.0, 0.0),
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
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
                "DCX gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(None);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "dcx".to_string(),
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

/// XXMinusYY gate - parameterized (XX - YY) interaction
#[derive(Debug, Clone)]
pub struct TQXXMinusYY {
    params: Option<TQParameter>,
    has_params: bool,
    trainable: bool,
    inverse: bool,
    static_mode: bool,
}

impl TQXXMinusYY {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(
                ArrayD::zeros(IxDyn(&[1, 2])),
                "xxmyy_params",
            ))
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

    pub fn with_init_params(mut self, theta: f64, beta: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = theta;
            p.data[[0, 1]] = beta;
        }
        self
    }
}

impl Default for TQXXMinusYY {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl TQModule for TQXXMinusYY {
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
        "XXMinusYY"
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut p) = self.params {
            p.zero_grad();
        }
    }
}

impl TQOperator for TQXXMinusYY {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(2)
    }

    fn get_matrix(&self, params: Option<&[f64]>) -> Array2<CType> {
        let theta = params
            .and_then(|p| p.first().copied())
            .or_else(|| self.params.as_ref().map(|p| p.data[[0, 0]]))
            .unwrap_or(0.0);
        let beta = params
            .and_then(|p| p.get(1).copied())
            .or_else(|| self.params.as_ref().map(|p| p.data[[0, 1]]))
            .unwrap_or(0.0);

        let theta = if self.inverse { -theta } else { theta };
        let half_theta = theta / 2.0;
        let c = half_theta.cos();
        let s = half_theta.sin();

        let exp_pos = CType::from_polar(1.0, beta);
        let exp_neg = CType::from_polar(1.0, -beta);

        Array2::from_shape_vec(
            (4, 4),
            vec![
                CType::new(c, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, -s) * exp_neg,
                CType::new(0.0, 0.0),
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, -s) * exp_pos,
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
                "XXMinusYY gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(params);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "xxmyy".to_string(),
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

/// XXPlusYY gate - parameterized (XX + YY) interaction
#[derive(Debug, Clone)]
pub struct TQXXPlusYY {
    params: Option<TQParameter>,
    has_params: bool,
    trainable: bool,
    inverse: bool,
    static_mode: bool,
}

impl TQXXPlusYY {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(
                ArrayD::zeros(IxDyn(&[1, 2])),
                "xxpyy_params",
            ))
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

    pub fn with_init_params(mut self, theta: f64, beta: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = theta;
            p.data[[0, 1]] = beta;
        }
        self
    }
}

impl Default for TQXXPlusYY {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl TQModule for TQXXPlusYY {
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
        "XXPlusYY"
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut p) = self.params {
            p.zero_grad();
        }
    }
}

impl TQOperator for TQXXPlusYY {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(2)
    }

    fn get_matrix(&self, params: Option<&[f64]>) -> Array2<CType> {
        let theta = params
            .and_then(|p| p.first().copied())
            .or_else(|| self.params.as_ref().map(|p| p.data[[0, 0]]))
            .unwrap_or(0.0);
        let beta = params
            .and_then(|p| p.get(1).copied())
            .or_else(|| self.params.as_ref().map(|p| p.data[[0, 1]]))
            .unwrap_or(0.0);

        let theta = if self.inverse { -theta } else { theta };
        let half_theta = theta / 2.0;
        let c = half_theta.cos();
        let s = half_theta.sin();

        let exp_pos = CType::from_polar(1.0, beta);
        let exp_neg = CType::from_polar(1.0, -beta);

        Array2::from_shape_vec(
            (4, 4),
            vec![
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(c, 0.0),
                CType::new(0.0, -s) * exp_neg,
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, -s) * exp_pos,
                CType::new(c, 0.0),
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
        params: Option<&[f64]>,
    ) -> Result<()> {
        if wires.len() < 2 {
            return Err(MLError::InvalidConfiguration(
                "XXPlusYY gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(params);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "xxpyy".to_string(),
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

/// CH gate - Controlled Hadamard gate
#[derive(Debug, Clone)]
pub struct TQCH {
    inverse: bool,
    static_mode: bool,
}

impl TQCH {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}

impl Default for TQCH {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQCH {
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
        "CH"
    }
}

impl TQOperator for TQCH {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(0)
    }

    fn get_matrix(&self, _params: Option<&[f64]>) -> Array2<CType> {
        let scale = 1.0 / std::f64::consts::SQRT_2;

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
                CType::new(scale, 0.0),
                CType::new(scale, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(scale, 0.0),
                CType::new(-scale, 0.0),
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
                "CH gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(None);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "ch".to_string(),
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

/// CPhase gate - Controlled phase gate (also known as CU1)
#[derive(Debug, Clone)]
pub struct TQCPhase {
    params: Option<TQParameter>,
    has_params: bool,
    trainable: bool,
    inverse: bool,
    static_mode: bool,
}

impl TQCPhase {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(
                ArrayD::zeros(IxDyn(&[1, 1])),
                "cphase_theta",
            ))
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

impl Default for TQCPhase {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl TQModule for TQCPhase {
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
        "CPhase"
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut p) = self.params {
            p.zero_grad();
        }
    }
}

impl TQOperator for TQCPhase {
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
        let exp_i = CType::from_polar(1.0, theta);

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
                exp_i,
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
                "CPhase gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(params);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "cphase".to_string(),
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

/// Type alias for CU1 gate (same as CPhase)
/// CU1 is the IBM/Qiskit name for controlled phase gate
pub type TQCU1 = TQCPhase;

/// fSim gate - Google's fermionic simulation gate
///
/// This gate is used in quantum chemistry simulations and is native
/// to Google's Sycamore processor. It combines an iSWAP-like interaction
/// with a controlled phase.
///
/// Matrix representation:
/// ```text
/// [[1,           0,                0,           0        ],
///  [0,           cos(θ),           -i·sin(θ),   0        ],
///  [0,           -i·sin(θ),        cos(θ),      0        ],
///  [0,           0,                0,           e^{-iφ}  ]]
/// ```
#[derive(Debug, Clone)]
pub struct TQFSimGate {
    params: Option<TQParameter>,
    has_params: bool,
    trainable: bool,
    inverse: bool,
    static_mode: bool,
}

impl TQFSimGate {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(
                ArrayD::zeros(IxDyn(&[1, 2])),
                "fsim_params",
            ))
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

    /// Create with initial parameters (theta, phi)
    pub fn with_init_params(mut self, theta: f64, phi: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = theta;
            p.data[[0, 1]] = phi;
        }
        self
    }

    /// Create a full iSWAP (theta=π/2, phi=0)
    pub fn iswap_like() -> Self {
        Self::new(true, false).with_init_params(std::f64::consts::FRAC_PI_2, 0.0)
    }

    /// Create a sqrt-iSWAP (theta=π/4, phi=0)
    pub fn sqrt_iswap() -> Self {
        Self::new(true, false).with_init_params(std::f64::consts::FRAC_PI_4, 0.0)
    }

    /// Create Sycamore gate (theta≈π/2, phi≈π/6) - Google's native gate
    pub fn sycamore() -> Self {
        Self::new(true, false)
            .with_init_params(std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_6)
    }
}

impl Default for TQFSimGate {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl TQModule for TQFSimGate {
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
        "fSim"
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut p) = self.params {
            p.zero_grad();
        }
    }
}

impl TQOperator for TQFSimGate {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(2)
    }

    fn get_matrix(&self, params: Option<&[f64]>) -> Array2<CType> {
        let theta = params
            .and_then(|p| p.first().copied())
            .or_else(|| self.params.as_ref().map(|p| p.data[[0, 0]]))
            .unwrap_or(0.0);
        let phi = params
            .and_then(|p| p.get(1).copied())
            .or_else(|| self.params.as_ref().map(|p| p.data[[0, 1]]))
            .unwrap_or(0.0);

        let theta = if self.inverse { -theta } else { theta };
        let phi = if self.inverse { -phi } else { phi };

        let c = theta.cos();
        let s = theta.sin();
        let exp_neg_i_phi = CType::from_polar(1.0, -phi);

        Array2::from_shape_vec(
            (4, 4),
            vec![
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
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                exp_neg_i_phi,
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
                "fSim gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(params);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "fsim".to_string(),
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

/// Givens rotation gate - fundamental for quantum chemistry
///
/// The Givens rotation performs a rotation in the (i,j) subspace of the Hilbert space.
/// It is widely used in molecular orbital transformations and VQE circuits for chemistry.
///
/// G(θ, φ) = exp(-i·θ/2·(e^{iφ}·|01⟩⟨10| + e^{-iφ}·|10⟩⟨01|))
///
/// Matrix representation:
/// ```text
/// [[1,   0,                   0,                   0],
///  [0,   cos(θ/2),            -e^{iφ}·sin(θ/2),   0],
///  [0,   e^{-iφ}·sin(θ/2),    cos(θ/2),           0],
///  [0,   0,                   0,                   1]]
/// ```
#[derive(Debug, Clone)]
pub struct TQGivensRotation {
    params: Option<TQParameter>,
    has_params: bool,
    trainable: bool,
    inverse: bool,
    static_mode: bool,
}

impl TQGivensRotation {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(
                ArrayD::zeros(IxDyn(&[1, 2])),
                "givens_params",
            ))
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

    /// Create with initial parameters (theta, phi)
    pub fn with_init_params(mut self, theta: f64, phi: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = theta;
            p.data[[0, 1]] = phi;
        }
        self
    }

    /// Create a real Givens rotation (phi=0)
    pub fn real(theta: f64) -> Self {
        Self::new(true, false).with_init_params(theta, 0.0)
    }

    /// Create a complex Givens rotation
    pub fn complex(theta: f64, phi: f64) -> Self {
        Self::new(true, false).with_init_params(theta, phi)
    }
}

impl Default for TQGivensRotation {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl TQModule for TQGivensRotation {
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
        "GivensRotation"
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut p) = self.params {
            p.zero_grad();
        }
    }
}

impl TQOperator for TQGivensRotation {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(2)
    }

    fn get_matrix(&self, params: Option<&[f64]>) -> Array2<CType> {
        let theta = params
            .and_then(|p| p.first().copied())
            .or_else(|| self.params.as_ref().map(|p| p.data[[0, 0]]))
            .unwrap_or(0.0);
        let phi = params
            .and_then(|p| p.get(1).copied())
            .or_else(|| self.params.as_ref().map(|p| p.data[[0, 1]]))
            .unwrap_or(0.0);

        let theta = if self.inverse { -theta } else { theta };
        let phi = if self.inverse { -phi } else { phi };

        let half_theta = theta / 2.0;
        let c = half_theta.cos();
        let s = half_theta.sin();

        let exp_i_phi = CType::from_polar(1.0, phi);
        let exp_neg_i_phi = CType::from_polar(1.0, -phi);

        Array2::from_shape_vec(
            (4, 4),
            vec![
                CType::new(1.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                CType::new(c, 0.0),
                -exp_i_phi * s,
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                exp_neg_i_phi * s,
                CType::new(c, 0.0),
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
        params: Option<&[f64]>,
    ) -> Result<()> {
        if wires.len() < 2 {
            return Err(MLError::InvalidConfiguration(
                "GivensRotation gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(params);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "givens".to_string(),
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

/// General controlled rotation gate
///
/// Applies a controlled rotation around an arbitrary axis.
/// CRot(theta, phi, omega) = CR_z(omega) @ CR_y(phi) @ CR_z(theta)
///
/// This is the controlled version of the general U3 rotation.
#[derive(Debug, Clone)]
pub struct TQControlledRot {
    params: Option<TQParameter>,
    has_params: bool,
    trainable: bool,
    inverse: bool,
    static_mode: bool,
}

impl TQControlledRot {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(
                ArrayD::zeros(IxDyn(&[1, 3])),
                "crot_params",
            ))
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

    /// Create with initial parameters (theta, phi, omega)
    pub fn with_init_params(mut self, theta: f64, phi: f64, omega: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = theta;
            p.data[[0, 1]] = phi;
            p.data[[0, 2]] = omega;
        }
        self
    }
}

impl Default for TQControlledRot {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl TQModule for TQControlledRot {
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
        "CRot"
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut p) = self.params {
            p.zero_grad();
        }
    }
}

impl TQOperator for TQControlledRot {
    fn num_wires(&self) -> WiresEnum {
        WiresEnum::Fixed(2)
    }

    fn num_params(&self) -> NParamsEnum {
        NParamsEnum::Fixed(3)
    }

    fn get_matrix(&self, params: Option<&[f64]>) -> Array2<CType> {
        let theta = params
            .and_then(|p| p.first().copied())
            .or_else(|| self.params.as_ref().map(|p| p.data[[0, 0]]))
            .unwrap_or(0.0);
        let phi = params
            .and_then(|p| p.get(1).copied())
            .or_else(|| self.params.as_ref().map(|p| p.data[[0, 1]]))
            .unwrap_or(0.0);
        let omega = params
            .and_then(|p| p.get(2).copied())
            .or_else(|| self.params.as_ref().map(|p| p.data[[0, 2]]))
            .unwrap_or(0.0);

        let (theta, phi, omega) = if self.inverse {
            (-omega, -phi, -theta)
        } else {
            (theta, phi, omega)
        };

        // CRot = CRz(omega) @ CRy(phi) @ CRz(theta)
        // For the target qubit: U = Rz(omega) @ Ry(phi) @ Rz(theta)
        let half_theta = theta / 2.0;
        let half_phi = phi / 2.0;
        let half_omega = omega / 2.0;

        // Compute U = Rz(omega) @ Ry(phi) @ Rz(theta)
        // U_00 = cos(phi/2) * e^{-i(theta+omega)/2}
        // U_01 = -sin(phi/2) * e^{-i(theta-omega)/2}
        // U_10 = sin(phi/2) * e^{i(theta-omega)/2}
        // U_11 = cos(phi/2) * e^{i(theta+omega)/2}
        let cos_phi = half_phi.cos();
        let sin_phi = half_phi.sin();

        let u00 = CType::from_polar(cos_phi, -(half_theta + half_omega));
        let u01 = CType::from_polar(-sin_phi, -(half_theta - half_omega));
        let u10 = CType::from_polar(sin_phi, half_theta - half_omega);
        let u11 = CType::from_polar(cos_phi, half_theta + half_omega);

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
                u00,
                u01,
                CType::new(0.0, 0.0),
                CType::new(0.0, 0.0),
                u10,
                u11,
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
                "CRot gate requires exactly 2 wires".to_string(),
            ));
        }
        let matrix = self.get_matrix(params);
        qdev.apply_two_qubit_gate(wires[0], wires[1], &matrix)?;

        if qdev.record_op {
            qdev.record_operation(OpHistoryEntry {
                name: "crot".to_string(),
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

/// Phase shift gate (also known as P gate)
///
/// This is a parameterized version of the phase gate that applies
/// a phase shift to the |1⟩ state.
///
/// Matrix representation:
/// ```text
/// [[1,   0,        0,           0],
///  [0,   1,        0,           0],
///  [0,   0,        1,           0],
///  [0,   0,        0,   e^{iφ}]]
/// ```
///
/// Note: This is the two-qubit controlled version. For single-qubit
/// phase shift, use the P gate in single_qubit module.
#[derive(Debug, Clone)]
pub struct TQPhaseShift2 {
    params: Option<TQParameter>,
    has_params: bool,
    trainable: bool,
    inverse: bool,
    static_mode: bool,
}

impl TQPhaseShift2 {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(
                ArrayD::zeros(IxDyn(&[1, 1])),
                "phaseshift2_phi",
            ))
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

    /// Create with initial phase parameter
    pub fn with_init_params(mut self, phi: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = phi;
        }
        self
    }
}

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
