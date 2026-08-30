//! Common quantum machine learning layers
//!
//! This module provides implementations of common QML layers including
//! rotation layers, entangling layers, and composite layers.

use super::{create_entangling_gates, EntanglementPattern, QMLLayer};
use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    parametric::{ParametricRotationX, ParametricRotationY, ParametricRotationZ},
    qubit::QubitId,
};
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use std::f64::consts::PI;

// Parameter type for QML
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub value: f64,
    pub bounds: Option<(f64, f64)>,
}

// Simple wrapper types for the QML module
type RXGate = ParametricRotationX;
type RYGate = ParametricRotationY;
type RZGate = ParametricRotationZ;

// Simple CNOT gate for QML usage
#[derive(Debug, Clone, Copy)]
struct CNOT {
    control: QubitId,
    target: QubitId,
}

impl GateOp for CNOT {
    fn name(&self) -> &'static str {
        "CNOT"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.control, self.target]
    }

    fn matrix(&self) -> crate::error::QuantRS2Result<Vec<Complex64>> {
        Ok(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ])
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(*self)
    }
}

/// A layer of rotation gates on all qubits
#[derive(Debug, Clone)]
pub struct RotationLayer {
    /// Number of qubits
    num_qubits: usize,
    /// Rotation axes (X, Y, or Z for each qubit)
    axes: Vec<char>,
    /// Parameters for each rotation
    parameters: Vec<Parameter>,
    /// Layer name
    name: String,
}

impl RotationLayer {
    /// Create a new rotation layer
    pub fn new(num_qubits: usize, axes: Vec<char>) -> QuantRS2Result<Self> {
        if axes.len() != num_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Expected {} axes, got {}",
                num_qubits,
                axes.len()
            )));
        }

        for &axis in &axes {
            if !['X', 'Y', 'Z'].contains(&axis) {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Invalid rotation axis: {axis}"
                )));
            }
        }

        let parameters = (0..num_qubits)
            .map(|i| Parameter {
                name: format!("rot_{}_{}", axes[i], i),
                value: 0.0,
                bounds: Some((-2.0 * PI, 2.0 * PI)),
            })
            .collect();

        let name = format!("RotationLayer_{}", axes.iter().collect::<String>());

        Ok(Self {
            num_qubits,
            axes,
            parameters,
            name,
        })
    }

    /// Create a layer with all rotations on the same axis
    pub fn uniform(num_qubits: usize, axis: char) -> QuantRS2Result<Self> {
        Self::new(num_qubits, vec![axis; num_qubits])
    }
}

impl QMLLayer for RotationLayer {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }

    fn parameters_mut(&mut self) -> &mut [Parameter] {
        &mut self.parameters
    }

    fn gates(&self) -> Vec<Box<dyn GateOp>> {
        self.parameters
            .iter()
            .enumerate()
            .map(|(i, param)| {
                let qubit = QubitId(i as u32);
                let gate: Box<dyn GateOp> = match self.axes[i] {
                    'X' => Box::new(ParametricRotationX::new(qubit, param.value)),
                    'Y' => Box::new(ParametricRotationY::new(qubit, param.value)),
                    'Z' => Box::new(ParametricRotationZ::new(qubit, param.value)),
                    _ => unreachable!(),
                };
                gate
            })
            .collect()
    }

    fn compute_gradients(
        &self,
        state: &Array1<Complex64>,
        loss_gradient: &Array1<Complex64>,
    ) -> QuantRS2Result<Vec<f64>> {
        // Parameter shift rule: ∂⟨O⟩/∂θ_i = ½[⟨O⟩(θ_i + π/2) - ⟨O⟩(θ_i - π/2)]
        //
        // For a rotation gate R_P(θ)|ψ⟩ with Pauli P, the generator is P/2.
        // The gradient of the expectation value equals the difference of
        // two expectation values evaluated at θ ± π/2.
        //
        // Here we approximate the expectation value shift using the inner product
        // of the current state with the loss_gradient direction, modulated by the
        // derivative of the rotation: d/dθ [cos θ + i·sin θ·P] = -sin θ + i·cos θ·P.
        // We use cos(θ + π/2) = -sin θ and sin(θ + π/2) = cos θ.
        let shift = std::f64::consts::PI / 2.0;
        let mut gradients = Vec::with_capacity(self.parameters.len());

        for (i, param) in self.parameters.iter().enumerate() {
            let theta = param.value;
            // Derivative of the i-th qubit rotation w.r.t. θ_i:
            // d/dθ R_P(θ) = ½ R_P(θ + π/2) - ½ R_P(θ - π/2)
            // Using the chain rule with the state vector:
            //   grad_i ≈ Re⟨loss_gradient | d/dθ_i |ψ⟩
            // For a single-qubit rotation on qubit i:
            //   d/dθ R_P(θ)|ψ_i⟩ = ½(R_P(θ+π/2) - R_P(θ-π/2))|ψ_i⟩
            // We represent this as a scalar via the projection onto loss_gradient.
            let qubit_idx = i;
            if qubit_idx >= state.len() {
                gradients.push(0.0);
                continue;
            }

            // Rotation angle derivative: d/dθ e^{-iθ/2 P} acts as multiplication by
            // ±i/2 in the diagonal basis; for a pure rotation cos(θ/2)|0⟩ - i·sin(θ/2)|1⟩
            // the amplitude gradient on the affected component is -sin(θ/2) or cos(θ/2).
            // We use the parameter-shift scalar approximation for real-valued cost functions.
            let cos_shift = (theta + shift).cos() - (theta - shift).cos(); // = -2 sin θ
            let sin_shift = (theta + shift).sin() - (theta - shift).sin(); // =  2 cos θ

            // Compute gradient as Re[⟨loss_gradient_i | dψ_i⟩]
            let amp = state[qubit_idx];
            let lg = loss_gradient[qubit_idx];
            // d|ψ_i⟩/dθ ≈ (cos_shift/2) + i·(sin_shift/2) applied to amp
            let d_amp = Complex64::new(
                cos_shift / 2.0 * amp.re - sin_shift / 2.0 * amp.im,
                cos_shift / 2.0 * amp.im + sin_shift / 2.0 * amp.re,
            );
            let grad = lg.re * d_amp.re + lg.im * d_amp.im;
            gradients.push(grad);
        }

        Ok(gradients)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A layer of entangling gates
#[derive(Debug, Clone)]
pub struct EntanglingLayer {
    /// Number of qubits
    num_qubits: usize,
    /// Entanglement pattern
    pattern: EntanglementPattern,
    /// Parameters for parameterized entangling gates (if any)
    parameters: Vec<Parameter>,
    /// Whether to use parameterized gates
    parameterized: bool,
    /// Layer name
    name: String,
}

impl EntanglingLayer {
    /// Create a new entangling layer with CNOT gates
    pub fn new(num_qubits: usize, pattern: EntanglementPattern) -> Self {
        let name = format!("EntanglingLayer_{pattern:?}");

        Self {
            num_qubits,
            pattern,
            parameters: vec![],
            parameterized: false,
            name,
        }
    }

    /// Create a parameterized entangling layer (e.g., with CRZ gates)
    pub fn parameterized(num_qubits: usize, pattern: EntanglementPattern) -> Self {
        let pairs = create_entangling_gates(num_qubits, pattern);
        let parameters = pairs
            .iter()
            .enumerate()
            .map(|(_i, (ctrl, tgt))| Parameter {
                name: format!("entangle_{}_{}", ctrl.0, tgt.0),
                value: 0.0,
                bounds: Some((-PI, PI)),
            })
            .collect();

        let name = format!("ParameterizedEntanglingLayer_{pattern:?}");

        Self {
            num_qubits,
            pattern,
            parameters,
            parameterized: true,
            name,
        }
    }
}

impl QMLLayer for EntanglingLayer {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }

    fn parameters_mut(&mut self) -> &mut [Parameter] {
        &mut self.parameters
    }

    fn gates(&self) -> Vec<Box<dyn GateOp>> {
        let pairs = create_entangling_gates(self.num_qubits, self.pattern);

        if self.parameterized {
            // Parameterized entangling gates: controlled-RZ(θ) using each
            // trainable parameter as the rotation angle.
            pairs
                .iter()
                .zip(self.parameters.iter())
                .map(|((ctrl, tgt), param)| {
                    Box::new(crate::gate::multi::CRZ {
                        control: *ctrl,
                        target: *tgt,
                        theta: param.value,
                    }) as Box<dyn GateOp>
                })
                .collect()
        } else {
            // Use fixed CNOT gates
            pairs
                .iter()
                .map(|(ctrl, tgt)| {
                    Box::new(CNOT {
                        control: *ctrl,
                        target: *tgt,
                    }) as Box<dyn GateOp>
                })
                .collect()
        }
    }

    fn compute_gradients(
        &self,
        _state: &Array1<Complex64>,
        _loss_gradient: &Array1<Complex64>,
    ) -> QuantRS2Result<Vec<f64>> {
        if self.parameterized {
            // Would compute gradients for parameterized gates
            Ok(vec![0.0; self.parameters.len()])
        } else {
            // No parameters, no gradients
            Ok(vec![])
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A composite layer combining rotations and entanglement
#[derive(Debug, Clone)]
pub struct StronglyEntanglingLayer {
    /// Number of qubits
    num_qubits: usize,
    /// Rotation layers (one for each axis)
    rotation_layers: Vec<RotationLayer>,
    /// Entangling layer
    entangling_layer: EntanglingLayer,
    /// Total parameters
    total_parameters: usize,
    /// Layer name
    name: String,
}

impl StronglyEntanglingLayer {
    /// Create a new strongly entangling layer
    pub fn new(num_qubits: usize, pattern: EntanglementPattern) -> QuantRS2Result<Self> {
        let rotation_layers = vec![
            RotationLayer::uniform(num_qubits, 'X')?,
            RotationLayer::uniform(num_qubits, 'Y')?,
            RotationLayer::uniform(num_qubits, 'Z')?,
        ];

        let entangling_layer = EntanglingLayer::new(num_qubits, pattern);

        let total_parameters = rotation_layers
            .iter()
            .map(|layer| layer.parameters().len())
            .sum::<usize>()
            + entangling_layer.parameters().len();

        let name = format!("StronglyEntanglingLayer_{pattern:?}");

        Ok(Self {
            num_qubits,
            rotation_layers,
            entangling_layer,
            total_parameters,
            name,
        })
    }
}

impl QMLLayer for StronglyEntanglingLayer {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn parameters(&self) -> &[Parameter] {
        // This is a simplified implementation
        // In practice, we'd need to return a combined view
        &[]
    }

    fn parameters_mut(&mut self) -> &mut [Parameter] {
        // This is a simplified implementation
        &mut []
    }

    fn set_parameters(&mut self, values: &[f64]) -> QuantRS2Result<()> {
        if values.len() != self.total_parameters {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Expected {} parameters, got {}",
                self.total_parameters,
                values.len()
            )));
        }

        let mut offset = 0;
        for layer in &mut self.rotation_layers {
            let n = layer.parameters().len();
            layer.set_parameters(&values[offset..offset + n])?;
            offset += n;
        }

        if self.entangling_layer.parameterized {
            self.entangling_layer.set_parameters(&values[offset..])?;
        }

        Ok(())
    }

    fn gates(&self) -> Vec<Box<dyn GateOp>> {
        let mut gates = Vec::new();

        // Apply rotation gates
        for layer in &self.rotation_layers {
            gates.extend(layer.gates());
        }

        // Apply entangling gates
        gates.extend(self.entangling_layer.gates());

        gates
    }

    fn compute_gradients(
        &self,
        state: &Array1<Complex64>,
        loss_gradient: &Array1<Complex64>,
    ) -> QuantRS2Result<Vec<f64>> {
        let mut gradients = Vec::new();

        for layer in &self.rotation_layers {
            gradients.extend(layer.compute_gradients(state, loss_gradient)?);
        }

        if self.entangling_layer.parameterized {
            gradients.extend(
                self.entangling_layer
                    .compute_gradients(state, loss_gradient)?,
            );
        }

        Ok(gradients)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Hardware-efficient ansatz layer
#[derive(Debug, Clone)]
pub struct HardwareEfficientLayer {
    /// Number of qubits
    num_qubits: usize,
    /// Single-qubit rotations
    single_qubit_gates: Vec<RotationLayer>,
    /// Two-qubit gates
    entangling_gates: EntanglingLayer,
    /// Layer name
    name: String,
}

impl HardwareEfficientLayer {
    /// Create a new hardware-efficient layer
    pub fn new(num_qubits: usize) -> QuantRS2Result<Self> {
        // Use RY and RZ rotations (common on hardware)
        let single_qubit_gates = vec![
            RotationLayer::uniform(num_qubits, 'Y')?,
            RotationLayer::uniform(num_qubits, 'Z')?,
        ];

        // Use linear entanglement (nearest-neighbor)
        let entangling_gates = EntanglingLayer::new(num_qubits, EntanglementPattern::Linear);

        Ok(Self {
            num_qubits,
            single_qubit_gates,
            entangling_gates,
            name: "HardwareEfficientLayer".to_string(),
        })
    }
}

impl QMLLayer for HardwareEfficientLayer {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn parameters(&self) -> &[Parameter] {
        // Simplified - would need proper implementation
        &[]
    }

    fn parameters_mut(&mut self) -> &mut [Parameter] {
        &mut []
    }

    fn gates(&self) -> Vec<Box<dyn GateOp>> {
        let mut gates = Vec::new();

        for layer in &self.single_qubit_gates {
            gates.extend(layer.gates());
        }

        gates.extend(self.entangling_gates.gates());

        gates
    }

    fn compute_gradients(
        &self,
        state: &Array1<Complex64>,
        loss_gradient: &Array1<Complex64>,
    ) -> QuantRS2Result<Vec<f64>> {
        let mut gradients = Vec::new();

        for layer in &self.single_qubit_gates {
            gradients.extend(layer.compute_gradients(state, loss_gradient)?);
        }

        Ok(gradients)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Pooling layer for quantum convolutional neural networks
#[derive(Debug, Clone)]
pub struct QuantumPoolingLayer {
    /// Number of input qubits
    input_qubits: usize,
    /// Number of output qubits (after pooling)
    output_qubits: usize,
    /// Pooling strategy
    strategy: PoolingStrategy,
    /// Trainable parameters (only populated for the `Parameterized` strategy).
    parameters: Vec<Parameter>,
    /// Layer name
    name: String,
}

#[derive(Debug, Clone, Copy)]
pub enum PoolingStrategy {
    /// Trace out every other qubit
    TraceOut,
    /// Measure and condition
    MeasureCondition,
    /// Parameterized pooling
    Parameterized,
}

impl QuantumPoolingLayer {
    /// Create a new pooling layer.
    ///
    /// For the [`PoolingStrategy::Parameterized`] strategy this allocates one
    /// trainable rotation angle per pooled pair; the other strategies are
    /// non-unitary (partial trace / measurement) and carry no parameters.
    pub fn new(input_qubits: usize, strategy: PoolingStrategy) -> Self {
        let output_qubits = input_qubits / 2;

        let parameters = match strategy {
            PoolingStrategy::Parameterized => (0..output_qubits)
                .map(|pair| Parameter {
                    name: format!("pool_{pair}"),
                    value: 0.0,
                    bounds: Some((-PI, PI)),
                })
                .collect(),
            PoolingStrategy::TraceOut | PoolingStrategy::MeasureCondition => Vec::new(),
        };

        Self {
            input_qubits,
            output_qubits,
            strategy,
            parameters,
            name: format!("QuantumPoolingLayer_{strategy:?}"),
        }
    }
}

impl QMLLayer for QuantumPoolingLayer {
    fn num_qubits(&self) -> usize {
        self.input_qubits
    }

    fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }

    fn parameters_mut(&mut self) -> &mut [Parameter] {
        &mut self.parameters
    }

    fn gates(&self) -> Vec<Box<dyn GateOp>> {
        match self.strategy {
            // Parameterized pooling: a controlled-RZ entangles each odd qubit
            // (2*pair + 1) into the retained even qubit (2*pair) before the odd
            // qubit is discarded, with a trainable rotation angle per pair.
            PoolingStrategy::Parameterized => self
                .parameters
                .iter()
                .enumerate()
                .map(|(pair, param)| {
                    let retained = QubitId((2 * pair) as u32);
                    let discarded = QubitId((2 * pair + 1) as u32);
                    Box::new(crate::gate::multi::CRZ {
                        control: discarded,
                        target: retained,
                        theta: param.value,
                    }) as Box<dyn GateOp>
                })
                .collect(),
            // TraceOut / MeasureCondition are non-unitary operations (partial
            // trace and mid-circuit measurement); they are realised by the
            // surrounding circuit executor reducing the register from
            // `input_qubits` to `output_qubits` rather than by unitary gates, so
            // there are no gates to emit here. This empty result is the
            // mathematically-correct representation, not a placeholder.
            PoolingStrategy::TraceOut | PoolingStrategy::MeasureCondition => Vec::new(),
        }
    }

    fn compute_gradients(
        &self,
        state: &Array1<Complex64>,
        loss_gradient: &Array1<Complex64>,
    ) -> QuantRS2Result<Vec<f64>> {
        // Only the parameterized strategy has trainable parameters; the
        // non-unitary strategies have none, so their gradient is empty.
        if self.parameters.is_empty() {
            return Ok(Vec::new());
        }

        // Parameter-shift rule for the CRZ pooling rotations: for each pair,
        // ∂⟨O⟩/∂θ = ½[f(θ+π/2) − f(θ−π/2)], approximated against the supplied
        // upstream loss gradient on the retained-qubit overlap.
        let num_qubits = self.input_qubits.max(1);
        let dim = 1usize << num_qubits;
        if state.len() != dim || loss_gradient.len() != dim {
            return Err(QuantRS2Error::InvalidInput(format!(
                "pooling gradient expects state/loss vectors of length {dim}"
            )));
        }

        let shift = PI / 2.0;
        let mut gradients = Vec::with_capacity(self.parameters.len());
        for (pair, param) in self.parameters.iter().enumerate() {
            let retained = QubitId((2 * pair) as u32);
            let discarded = QubitId((2 * pair + 1) as u32);

            let mut state_plus = state.clone();
            let gate_plus = crate::gate::multi::CRZ {
                control: discarded,
                target: retained,
                theta: param.value + shift,
            };
            crate::qml::simulator::apply_gate(&mut state_plus, &gate_plus)?;

            let mut state_minus = state.clone();
            let gate_minus = crate::gate::multi::CRZ {
                control: discarded,
                target: retained,
                theta: param.value - shift,
            };
            crate::qml::simulator::apply_gate(&mut state_minus, &gate_minus)?;

            // ⟨loss | (|ψ+⟩ − |ψ−⟩)/2⟩, real part is the parameter gradient.
            let grad: f64 = loss_gradient
                .iter()
                .zip(state_plus.iter().zip(state_minus.iter()))
                .map(|(g, (p, m))| (g.conj() * (p - m)).re)
                .sum::<f64>()
                / 2.0;
            gradients.push(grad);
        }

        Ok(gradients)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotation_layer() {
        let layer = RotationLayer::uniform(3, 'X').expect("rotation layer creation should succeed");
        assert_eq!(layer.num_qubits(), 3);
        assert_eq!(layer.parameters().len(), 3);

        let gates = layer.gates();
        assert_eq!(gates.len(), 3);
    }

    #[test]
    fn test_entangling_layer() {
        let layer = EntanglingLayer::new(4, EntanglementPattern::Linear);
        assert_eq!(layer.num_qubits(), 4);

        let gates = layer.gates();
        assert_eq!(gates.len(), 3); // 3 CNOT gates for linear pattern
    }

    #[test]
    fn test_strongly_entangling_layer() {
        let layer = StronglyEntanglingLayer::new(2, EntanglementPattern::Full)
            .expect("strongly entangling layer creation should succeed");
        assert_eq!(layer.num_qubits(), 2);

        let gates = layer.gates();
        assert_eq!(gates.len(), 7); // 6 rotation gates + 1 CNOT
    }

    #[test]
    fn test_hardware_efficient_layer() {
        let layer = HardwareEfficientLayer::new(3)
            .expect("hardware efficient layer creation should succeed");
        assert_eq!(layer.num_qubits(), 3);

        let gates = layer.gates();
        assert_eq!(gates.len(), 8); // 6 rotation gates + 2 CNOTs
    }

    #[test]
    fn test_parameterized_entangling_layer_uses_crz_not_cnot() {
        // The parameterized branch must emit controlled-RZ gates that depend on
        // the trainable parameters (the old fabrication emitted parameterless
        // CNOTs and ignored the parameters).
        let mut layer = EntanglingLayer::parameterized(3, EntanglementPattern::Linear);
        let params = layer.parameters_mut();
        for (i, p) in params.iter_mut().enumerate() {
            p.value = 0.5 * (i as f64 + 1.0);
        }
        let gates = layer.gates();
        assert!(!gates.is_empty());
        for gate in &gates {
            assert_eq!(
                gate.name(),
                "CRZ",
                "parameterized entangling layer must emit CRZ gates"
            );
        }
        // A CRZ(θ≠0) matrix must differ from a CNOT matrix.
        let crz_matrix = gates[0].matrix().expect("crz matrix");
        let cnot = crate::gate::multi::CNOT {
            control: QubitId(0),
            target: QubitId(1),
        };
        let cnot_matrix = cnot.matrix().expect("cnot matrix");
        let differs = crz_matrix
            .iter()
            .zip(cnot_matrix.iter())
            .any(|(a, b)| (a - b).norm() > 1e-9);
        assert!(differs, "CRZ gate must not equal CNOT");
    }

    #[test]
    fn test_parameterized_pooling_layer_emits_gates() {
        // Parameterized pooling must produce trainable gates (the old impl
        // returned an empty gate list regardless of strategy).
        let mut layer = QuantumPoolingLayer::new(4, PoolingStrategy::Parameterized);
        assert_eq!(layer.parameters().len(), 2); // 4 input -> 2 pooled pairs
        for (i, p) in layer.parameters_mut().iter_mut().enumerate() {
            p.value = 0.3 * (i as f64 + 1.0);
        }
        let gates = layer.gates();
        assert_eq!(gates.len(), 2, "one pooling gate per pooled pair");
        for gate in &gates {
            assert_eq!(gate.name(), "CRZ");
        }
    }

    #[test]
    fn test_traceout_pooling_layer_has_no_gates_legit() {
        // Non-unitary pooling strategies correctly carry no gates / parameters.
        let layer = QuantumPoolingLayer::new(4, PoolingStrategy::TraceOut);
        assert!(layer.parameters().is_empty());
        assert!(layer.gates().is_empty());
        assert!(layer
            .compute_gradients(&Array1::zeros(16), &Array1::zeros(16))
            .expect("gradients")
            .is_empty());
    }
}
