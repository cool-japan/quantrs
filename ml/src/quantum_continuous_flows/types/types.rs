//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
use scirs2_core::{Complex32, Complex64};

#[derive(Debug, Clone)]
pub struct FlowOptimizationState {
    pub learning_rate: f64,
    pub momentum: f64,
    pub gradient_clipping_norm: f64,
    pub quantum_parameter_learning_rate: f64,
    pub entanglement_preservation_weight: f64,
    pub invertibility_penalty_weight: f64,
}
#[derive(Debug, Clone)]
pub struct ClassicalDynamics {
    pub(super) dynamics_network: Vec<ClassicalFlowLayer>,
    pub(super) nonlinearity: FlowActivation,
}
#[derive(Debug, Clone)]
pub struct QuantumTransformation {
    pub(super) transformation_type: QuantumTransformationType,
    pub(super) unitary_matrix: Array2<Complex64>,
    pub(super) parameters: Array1<f64>,
    pub(super) invertibility_guaranteed: bool,
}
#[derive(Debug, Clone)]
pub enum QuantumODESolver {
    QuantumEuler,
    QuantumRungeKutta4,
    QuantumDormandPrince,
    AdaptiveQuantumSolver,
    QuantumMidpoint,
}
#[derive(Debug, Clone)]
pub enum EntanglementPatternType {
    Linear,
    Circular,
    AllToAll,
    Hierarchical { levels: usize },
    Random { probability: f64 },
    LongRange { decay_rate: f64 },
}
#[derive(Debug, Clone)]
pub struct QuantumODEFunction {
    pub(super) quantum_dynamics: QuantumDynamics,
    pub(super) classical_dynamics: ClassicalDynamics,
    pub(super) hybrid_coupling: HybridCoupling,
}
#[derive(Debug, Clone, Default)]
pub struct QuantumLayerState {
    pub quantum_fidelity: f64,
    pub entanglement_measure: f64,
    pub coherence_time: f64,
    pub quantum_volume: f64,
}
#[derive(Debug, Clone)]
pub enum ClassicalFlowLayerType {
    Dense { input_dim: usize, output_dim: usize },
    Convolutional { channels: usize, kernel_size: usize },
    Residual { skip_connection: bool },
}
#[derive(Debug, Clone)]
pub enum FlowArchitecture {
    /// Quantum Real NVP with entanglement coupling
    QuantumRealNVP {
        hidden_dims: Vec<usize>,
        num_coupling_layers: usize,
        quantum_coupling_type: QuantumCouplingType,
    },
    /// Quantum Glow with invertible 1x1 convolutions
    QuantumGlow {
        num_levels: usize,
        num_steps_per_level: usize,
        quantum_invertible_conv: bool,
    },
    /// Quantum Neural Spline Flows
    QuantumNeuralSplineFlow {
        num_bins: usize,
        spline_range: f64,
        quantum_spline_parameters: bool,
    },
    /// Quantum Continuous Normalizing Flows with Neural ODEs
    QuantumContinuousNormalizing {
        ode_net_dims: Vec<usize>,
        quantum_ode_solver: QuantumODESolver,
        trace_estimation_method: TraceEstimationMethod,
    },
    /// Quantum Autoregressive Flows
    QuantumAutoregressiveFlow {
        num_layers: usize,
        hidden_dim: usize,
        quantum_masking_type: QuantumMaskingType,
    },
}
#[derive(Debug, Clone)]
pub struct ConnectivityGraph {
    pub(super) adjacency_matrix: Array2<bool>,
    pub(super) edge_weights: Array2<f64>,
    pub(super) num_nodes: usize,
}
#[derive(Debug, Clone)]
pub struct SplineParameters {
    pub(super) knot_positions: Array2<f64>,
    pub(super) knot_derivatives: Array2<f64>,
    pub(super) num_bins: usize,
    pub(super) spline_range: f64,
}
#[derive(Debug, Clone)]
pub struct QuantumNetworkOutput {
    pub output: Array1<f64>,
    pub quantum_state: QuantumLayerState,
    pub entanglement_measure: f64,
}
#[derive(Debug, Clone)]
pub struct HybridCoupling {
    pub(super) quantum_to_classical: Array2<f64>,
    pub(super) classical_to_quantum: Array2<f64>,
    pub(super) coupling_strength: f64,
}
#[derive(Debug, Clone)]
pub struct QuantumFlowMetrics {
    pub average_entanglement: f64,
    pub coherence_preservation: f64,
    pub invertibility_accuracy: f64,
    pub quantum_volume_utilization: f64,
    pub flow_conditioning: f64,
    pub quantum_speedup_factor: f64,
    pub density_estimation_accuracy: f64,
}
#[derive(Debug, Clone)]
pub enum QuantumFlowLayerType {
    QuantumLinear {
        input_features: usize,
        output_features: usize,
    },
    QuantumConvolutional {
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    },
    QuantumAttention {
        num_heads: usize,
        head_dim: usize,
        attention_type: QuantumAttentionType,
    },
    QuantumResidual {
        inner_layers: Vec<Box<QuantumFlowNetworkLayer>>,
    },
    QuantumNormalization {
        normalization_type: QuantumNormalizationType,
    },
}
#[derive(Debug, Clone)]
pub enum JacobianComputation {
    ExactJacobian,
    ApproximateJacobian {
        epsilon: f64,
    },
    QuantumJacobian {
        trace_estimator: TraceEstimationMethod,
    },
    HutchinsonEstimator {
        num_samples: usize,
    },
}
#[derive(Debug, Clone)]
pub enum EntanglementCouplingType {
    QuantumCNOT,
    QuantumIsingCoupling,
    QuantumExchangeCoupling,
    QuantumDipolarCoupling,
    CustomCoupling { hamiltonian: Array2<Complex64> },
}
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    pub(super) method: QuantumODESolver,
    pub(super) tolerance: f64,
    pub(super) max_steps: usize,
    pub(super) adaptive_step_size: bool,
}
#[derive(Debug, Clone)]
pub struct EntanglementCoupling {
    pub(super) coupling_qubits: Vec<usize>,
    pub(super) coupling_strength: f64,
    pub(super) coupling_type: EntanglementCouplingType,
    pub(super) time_evolution: TimeEvolution,
}
#[derive(Debug, Clone)]
pub struct QuantumDistributionState {
    pub(super) quantum_state_vector: Array1<Complex64>,
    pub(super) density_matrix: Array2<Complex64>,
    pub(super) entanglement_structure: EntanglementStructure,
}
#[derive(Debug, Clone)]
pub struct QuantumClassicalConnection {
    pub(super) quantum_layer_idx: usize,
    pub(super) classical_layer_idx: usize,
    pub(super) connection_type: ConnectionType,
    pub(super) transformation_matrix: Array2<f64>,
}
#[derive(Debug, Clone)]
pub struct QuantumBaseDistribution {
    pub(super) distribution_type: QuantumDistributionType,
    pub(super) parameters: DistributionParameters,
    pub(super) quantum_state: QuantumDistributionState,
}
#[derive(Debug, Clone)]
pub struct SampleQuantumMetrics {
    pub sample_idx: usize,
    pub entanglement_measure: f64,
    pub quantum_fidelity: f64,
    pub coherence_time: f64,
}
#[derive(Debug, Clone)]
pub struct InvertibleComponent {
    pub(super) forward_transform: InvertibleTransform,
    pub(super) inverse_transform: InvertibleTransform,
    pub(super) jacobian_computation: JacobianComputation,
    pub(super) invertibility_check: InvertibilityCheck,
}
#[derive(Debug, Clone)]
pub enum ODEIntegrationMethod {
    Euler,
    RungeKutta4,
    DormandPrince,
    QuantumAdaptive,
}
#[derive(Debug, Clone)]
pub struct QuantumDistributionComponent {
    pub(super) distribution: Box<QuantumDistributionType>,
    pub(super) weight: f64,
    pub(super) quantum_phase: Complex64,
}
#[derive(Debug, Clone)]
pub struct QuantumFlowLayer {
    pub(super) layer_id: usize,
    pub(super) layer_type: FlowLayerType,
    pub(super) quantum_parameters: Array1<f64>,
    pub(super) classical_parameters: Array2<f64>,
    pub(super) coupling_network: QuantumCouplingNetwork,
    pub(super) invertible_component: InvertibleComponent,
    pub(super) entanglement_pattern: EntanglementPattern,
}
#[derive(Debug, Clone)]
pub struct InvertibilityTracker {
    pub inversion_errors: Vec<f64>,
    pub jacobian_conditioning: Vec<f64>,
    pub quantum_unitarity_violations: Vec<f64>,
    pub average_inversion_time: f64,
}
#[derive(Debug, Clone)]
pub enum QuantumDivergenceType {
    KLDivergence,
    WassersteinDistance,
    QuantumRelativeEntropy,
    EntanglementDivergence,
    QuantumFisherInformation,
}
#[derive(Debug, Clone)]
pub enum QuantumAttentionType {
    StandardQuantumAttention,
    QuantumMultiHeadAttention,
    EntanglementBasedAttention,
    QuantumSelfAttention,
    QuantumCrossAttention,
}
#[derive(Debug, Clone)]
pub struct FlowTrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub learning_rate_decay: f64,
    pub log_interval: usize,
    pub gradient_clipping_norm: f64,
    pub regularization_weight: f64,
}
#[derive(Debug, Clone)]
pub struct DecoherenceModel {
    pub t1_time: f64,
    pub t2_time: f64,
    pub gate_error_rate: f64,
    pub measurement_error_rate: f64,
}
#[derive(Debug, Clone)]
pub enum EntanglementType {
    CNOT,
    CZ,
    QuantumSwap,
    CustomEntangling { matrix: Array2<Complex64> },
}
#[derive(Debug, Clone)]
pub enum MeasurementStrategy {
    ExpectationValue { observables: Vec<Observable> },
    ProbabilityDistribution,
    QuantumStateVector,
    EntanglementMeasure,
    CoherenceMeasure,
}
#[derive(Debug, Clone)]
pub struct QuantumNetwork {
    pub(super) layers: Vec<QuantumFlowNetworkLayer>,
    pub(super) output_dim: usize,
    pub(super) quantum_enhancement: f64,
}
#[derive(Debug, Clone, Default)]
pub struct FlowConvergenceAnalysis {
    pub convergence_rate: f64,
    pub final_loss: f64,
    pub loss_variance: f64,
    pub is_converged: bool,
    pub invertibility_maintained: bool,
}
/// Configuration for Quantum Continuous Normalization Flows
#[derive(Debug, Clone)]
pub struct QuantumContinuousFlowConfig {
    pub input_dim: usize,
    pub latent_dim: usize,
    pub num_qubits: usize,
    pub num_flow_layers: usize,
    pub flow_architecture: FlowArchitecture,
    pub quantum_enhancement_level: f64,
    pub integration_method: ODEIntegrationMethod,
    pub invertibility_tolerance: f64,
    pub entanglement_coupling_strength: f64,
    pub quantum_divergence_type: QuantumDivergenceType,
    pub use_quantum_attention_flows: bool,
    pub adaptive_step_size: bool,
    pub regularization_config: FlowRegularizationConfig,
}
#[derive(Debug, Clone)]
pub struct EntanglementPattern {
    pub(super) pattern_type: EntanglementPatternType,
    pub(super) connectivity: ConnectivityGraph,
    pub(super) entanglement_strength: Array1<f64>,
}
#[derive(Debug, Clone)]
pub struct MeasurementOutput {
    pub expectation_values: Array1<f64>,
    pub variance_measures: Array1<f64>,
    pub entanglement_measure: f64,
    pub average_phase: Complex64,
}
#[derive(Debug, Clone)]
pub enum FlowLayerType {
    QuantumCouplingLayer {
        coupling_type: QuantumCouplingType,
        split_dimension: usize,
    },
    QuantumAffineCoupling {
        scale_network: QuantumNetwork,
        translation_network: QuantumNetwork,
    },
    QuantumInvertibleConv {
        kernel_size: usize,
        quantum_weights: bool,
    },
    QuantumActNorm {
        data_dependent_init: bool,
    },
    QuantumSplineTransform {
        num_bins: usize,
        spline_range: f64,
    },
    QuantumNeuralODE {
        ode_func: QuantumODEFunction,
        integration_time: f64,
    },
}
#[derive(Debug, Clone)]
pub enum QuantumCouplingType {
    AffineCoupling,
    AdditiveCouplering,
    QuantumEntangledCoupling,
    PhaseRotationCoupling,
    SplineCoupling,
}
#[derive(Debug, Clone)]
pub struct LayerOutput {
    pub transformed_data: Array1<f64>,
    pub log_jacobian_det: f64,
    pub quantum_state: QuantumLayerState,
    pub entanglement_measure: f64,
}
#[derive(Debug, Clone)]
pub struct FlowInverseOutput {
    pub data_sample: Array1<f64>,
    pub log_probability: f64,
    pub log_jacobian_determinant: f64,
    pub quantum_states: Vec<QuantumLayerState>,
}
#[derive(Debug, Clone)]
pub struct Observable {
    pub(super) name: String,
    pub(super) matrix: Array2<Complex64>,
    pub(super) qubits: Vec<usize>,
}
#[derive(Debug, Clone)]
pub enum QuantumTransformationType {
    UnitaryTransformation,
    QuantumFourierTransform,
    QuantumHadamardTransform,
    ParameterizedQuantumCircuit,
    QuantumWaveletTransform,
}
#[derive(Debug, Clone)]
pub enum QuantumMaskingType {
    Sequential,
    Random,
    QuantumSuperposition,
    EntanglementBased,
}
#[derive(Debug, Clone)]
pub enum TraceEstimationMethod {
    HutchinsonTrace,
    SkewedHutchinson,
    QuantumStateTrace,
    EntanglementBasedTrace,
}
#[derive(Debug, Clone)]
pub struct QuantumEnhancement {
    pub log_enhancement: f64,
    pub entanglement_contribution: f64,
    pub fidelity_contribution: f64,
    pub coherence_contribution: f64,
    pub quantum_advantage_ratio: f64,
}
#[derive(Debug, Clone)]
pub enum FlowActivation {
    ReLU,
    Swish,
    GELU,
    Tanh,
    LeakyReLU,
    ELU,
}
#[derive(Debug, Clone)]
pub enum InvertibilityCheck {
    DeterminantCheck { tolerance: f64 },
    SingularValueCheck { min_singular_value: f64 },
    QuantumUnitarityCheck { fidelity_threshold: f64 },
    NumericalInversion { max_iterations: usize },
}
#[derive(Debug, Clone)]
pub struct ClassicalFlowLayer {
    pub(super) layer_type: ClassicalFlowLayerType,
    pub(super) parameters: Array2<f64>,
    pub(super) activation: FlowActivation,
    pub(super) normalization: Option<FlowNormalization>,
}
#[derive(Debug, Clone)]
pub struct FlowTrainingMetrics {
    pub epoch: usize,
    pub negative_log_likelihood: f64,
    pub bits_per_dimension: f64,
    pub quantum_likelihood: f64,
    pub entanglement_measure: f64,
    pub invertibility_score: f64,
    pub jacobian_determinant_mean: f64,
    pub jacobian_determinant_std: f64,
    pub quantum_fidelity: f64,
    pub coherence_time: f64,
    pub quantum_advantage_ratio: f64,
}
#[derive(Debug, Clone)]
pub enum InvertibleTransform {
    QuantumUnitaryTransform {
        unitary_matrix: Array2<Complex64>,
        parameters: Array1<f64>,
    },
    QuantumCouplingTransform {
        coupling_function: CouplingFunction,
        mask: Array1<bool>,
    },
    QuantumSplineTransform {
        spline_parameters: SplineParameters,
    },
    QuantumNeuralODETransform {
        ode_function: QuantumODEFunction,
        integration_config: IntegrationConfig,
    },
}
#[derive(Debug, Clone)]
pub enum RotationAxis {
    X,
    Y,
    Z,
    Custom { direction: Array1<f64> },
}
#[derive(Debug, Clone)]
pub enum FlowNormalization {
    BatchNorm,
    LayerNorm,
    InstanceNorm,
    GroupNorm,
}
#[derive(Debug, Clone)]
pub struct FlowTrainingOutput {
    pub training_losses: Vec<f64>,
    pub validation_losses: Vec<f64>,
    pub quantum_metrics_history: Vec<QuantumFlowMetrics>,
    pub final_invertibility_score: f64,
    pub convergence_analysis: FlowConvergenceAnalysis,
}
#[derive(Debug, Clone)]
pub enum ConnectionType {
    MeasurementToClassical,
    ClassicalToQuantum,
    ParameterSharing,
    GradientCoupling,
}
#[derive(Debug, Clone)]
pub enum QuantumFlowGateType {
    ParameterizedRotation { axis: RotationAxis },
    ControlledRotation { axis: RotationAxis },
    QuantumCoupling { coupling_strength: f64 },
    EntanglementGate { entanglement_type: EntanglementType },
    InvertibleQuantumGate { inverse_parameters: Array1<f64> },
}
#[derive(Debug, Clone)]
pub struct QuantumFlowState {
    pub amplitudes: Array1<Complex64>,
    pub phases: Array1<Complex64>,
    pub entanglement_measure: f64,
    pub coherence_time: f64,
    pub fidelity: f64,
}
#[derive(Debug, Clone)]
pub struct TimeEvolution {
    pub(super) time_steps: Array1<f64>,
    pub(super) evolution_operators: Vec<Array2<Complex64>>,
    pub(super) adaptive_time_stepping: bool,
}
#[derive(Debug, Clone)]
pub struct QuantumDynamics {
    pub(super) hamiltonian: Array2<Complex64>,
    pub(super) time_evolution_operator: Array2<Complex64>,
    pub(super) decoherence_model: DecoherenceModel,
}
#[derive(Debug, Clone)]
pub struct QuantumCouplingNetwork {
    pub(super) network_type: CouplingNetworkType,
    pub(super) quantum_layers: Vec<QuantumFlowNetworkLayer>,
    pub(super) classical_layers: Vec<ClassicalFlowLayer>,
    pub(super) hybrid_connections: Vec<QuantumClassicalConnection>,
}
#[derive(Debug, Clone)]
pub struct DistributionParameters {
    pub(super) location: Array1<f64>,
    pub(super) scale: Array1<f64>,
    pub(super) shape: Array1<f64>,
    pub(super) quantum_parameters: Array1<Complex64>,
}
#[derive(Debug, Clone)]
pub enum QuantumNormalizationType {
    QuantumBatchNorm,
    QuantumLayerNorm,
    QuantumInstanceNorm,
    EntanglementNormalization,
}
#[derive(Debug, Clone)]
pub struct EntanglementStructure {
    pub(super) entanglement_measure: f64,
    pub(super) schmidt_decomposition: SchmidtDecomposition,
    pub(super) quantum_correlations: Array2<f64>,
}
#[derive(Debug, Clone)]
pub struct FlowForwardOutput {
    pub latent_sample: Array1<f64>,
    pub log_probability: f64,
    pub quantum_log_probability: f64,
    pub log_jacobian_determinant: f64,
    pub quantum_states: Vec<QuantumLayerState>,
    pub entanglement_history: Vec<f64>,
    pub quantum_enhancement: QuantumEnhancement,
}
#[derive(Debug, Clone)]
pub struct SchmidtDecomposition {
    pub(super) schmidt_coefficients: Array1<f64>,
    pub(super) left_basis: Array2<Complex64>,
    pub(super) right_basis: Array2<Complex64>,
}
#[derive(Debug, Clone)]
pub struct FlowRegularizationConfig {
    pub weight_decay: f64,
    pub spectral_normalization: bool,
    pub kinetic_energy_regularization: f64,
    pub entanglement_regularization: f64,
    pub jacobian_regularization: f64,
    pub quantum_volume_preservation: f64,
}
#[derive(Debug, Clone)]
pub struct QuantumFlowGate {
    pub(super) gate_type: QuantumFlowGateType,
    pub(super) target_qubits: Vec<usize>,
    pub(super) control_qubits: Vec<usize>,
    pub(super) parameters: Array1<f64>,
    pub(super) is_invertible: bool,
}
#[derive(Debug, Clone)]
pub enum CouplingNetworkType {
    QuantumMLP,
    QuantumConvolutional,
    QuantumTransformer,
    QuantumResNet,
    HybridQuantumClassical,
}
#[derive(Debug, Clone)]
pub struct FlowSamplingOutput {
    pub samples: Array2<f64>,
    pub log_probabilities: Array1<f64>,
    pub quantum_metrics: Vec<SampleQuantumMetrics>,
    pub overall_quantum_performance: QuantumFlowMetrics,
}
#[derive(Debug, Clone)]
pub enum QuantumDistributionType {
    QuantumGaussian {
        mean: Array1<f64>,
        covariance: Array2<f64>,
        quantum_enhancement: f64,
    },
    QuantumUniform {
        bounds: Array2<f64>,
        quantum_superposition: bool,
    },
    QuantumMixture {
        components: Vec<QuantumDistributionComponent>,
        mixing_weights: Array1<f64>,
    },
    QuantumThermalState {
        temperature: f64,
        hamiltonian: Array2<Complex64>,
    },
    QuantumCoherentState {
        coherence_parameters: Array1<Complex64>,
    },
}
#[derive(Debug, Clone, Default)]
pub struct QuantumFlowBatchMetrics {
    pub entanglement_measure: f64,
    pub invertibility_score: f64,
    pub jacobian_determinant_mean: f64,
    pub jacobian_determinant_std: f64,
    pub quantum_fidelity: f64,
    pub coherence_time: f64,
    pub quantum_advantage_ratio: f64,
}
impl QuantumFlowBatchMetrics {
    pub fn accumulate(&mut self, forward_output: &FlowForwardOutput) -> Result<()> {
        self.entanglement_measure += forward_output.quantum_enhancement.entanglement_contribution;
        self.invertibility_score += 1.0;
        self.jacobian_determinant_mean += forward_output.log_jacobian_determinant;
        self.jacobian_determinant_std += forward_output.log_jacobian_determinant.powi(2);
        self.quantum_fidelity += forward_output.quantum_enhancement.fidelity_contribution;
        self.coherence_time += forward_output.quantum_enhancement.coherence_contribution;
        self.quantum_advantage_ratio += forward_output.quantum_enhancement.quantum_advantage_ratio;
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub struct CouplingFunction {
    pub(super) scale_function: QuantumNetwork,
    pub(super) translation_function: QuantumNetwork,
    pub(super) coupling_type: QuantumCouplingType,
}
#[derive(Debug, Clone)]
pub struct QuantumFlowNetworkLayer {
    pub(super) layer_type: QuantumFlowLayerType,
    pub(super) num_qubits: usize,
    pub(super) parameters: Array1<f64>,
    pub(super) quantum_gates: Vec<QuantumFlowGate>,
    pub(super) measurement_strategy: MeasurementStrategy,
}
#[derive(Debug, Clone)]
pub struct CouplingNetworkOutput {
    pub scale_params: Array1<f64>,
    pub translation_params: Array1<f64>,
    pub entanglement_factor: f64,
    pub quantum_phase: Complex64,
    pub quantum_state: QuantumLayerState,
}
