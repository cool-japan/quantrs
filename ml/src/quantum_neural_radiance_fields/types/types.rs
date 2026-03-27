//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
use scirs2_core::{Complex32, Complex64};

#[derive(Debug, Clone)]
pub enum QuantumColorSpace {
    RGB,
    HSV,
    LAB,
    QuantumColorSpace { basis_vectors: Array2<f64> },
    EntangledColorChannels,
}
#[derive(Debug, Clone)]
pub enum QuantumActivationType {
    QuantumReLU,
    QuantumSigmoid,
    QuantumSoftplus,
    QuantumTanh,
    QuantumEntanglementActivation,
    QuantumPhaseActivation,
}
#[derive(Debug, Clone)]
pub struct QuantumPositionalEncoder {
    pub(super) encoding_type: QuantumPositionalEncodingType,
    pub(super) num_frequencies: usize,
    pub(super) quantum_frequencies: Array1<f64>,
    pub(super) entanglement_encoding: bool,
    pub(super) phase_encoding: bool,
    pub(super) max_frequency: f64,
}
#[derive(Debug, Clone)]
pub enum QuantumRenderingEquation {
    StandardVolumeRendering,
    QuantumVolumeRendering {
        quantum_transmittance: bool,
        entangled_scattering: bool,
    },
    QuantumPathTracing {
        max_bounces: usize,
        quantum_importance_sampling: bool,
    },
    QuantumPhotonMapping {
        num_photons: usize,
        quantum_photon_transport: bool,
    },
}
#[derive(Debug, Clone)]
pub struct SceneBounds {
    pub min_bound: Array1<f64>,
    pub max_bound: Array1<f64>,
    pub voxel_resolution: Array1<usize>,
}
#[derive(Debug, Clone)]
pub struct QuantumSpatialAttention {
    pub(super) num_heads: usize,
    pub(super) head_dim: usize,
    pub(super) quantum_query_projection: Array2<Complex64>,
    pub(super) quantum_key_projection: Array2<Complex64>,
    pub(super) quantum_value_projection: Array2<Complex64>,
    pub(super) entanglement_weights: Array1<f64>,
}
#[derive(Debug, Clone)]
pub enum QuantumBasisType {
    QuantumRadialBasis { sigma: f64 },
    QuantumWavelet { wavelet_type: String },
    QuantumFourier { frequency: f64 },
    QuantumSpline { order: usize },
}
#[derive(Debug, Clone)]
pub enum RotationAxis {
    X,
    Y,
    Z,
    Custom { direction: Array1<f64> },
}
#[derive(Debug, Clone)]
pub struct QuantumLightSource {
    pub(super) position: Array1<f64>,
    pub(super) intensity: Array1<f64>,
    pub(super) light_type: QuantumLightType,
    pub(super) quantum_coherence: f64,
}
#[derive(Debug, Clone)]
pub struct QuantumEnvironmentEncoding {
    pub(super) encoding_type: QuantumEnvironmentEncodingType,
    pub(super) quantum_coefficients: Array1<Complex64>,
    pub(super) spatial_frequency_components: Array1<f64>,
}
#[derive(Debug, Clone)]
pub enum QuantumAttentionType {
    StandardQuantumAttention,
    QuantumMultiHeadAttention,
    QuantumSpatialAttention,
    QuantumViewAttention,
    EntanglementBasedAttention,
    QuantumCrossAttention,
}
#[derive(Debug, Clone)]
pub struct QuantumViewAttention {
    pub(super) view_embedding_dim: usize,
    pub(super) quantum_view_weights: Array2<Complex64>,
    pub(super) view_dependent_parameters: Array1<f64>,
    pub(super) quantum_view_interpolation: bool,
}
#[derive(Debug, Clone)]
pub enum QuantumBRDFType {
    LambertianBRDF,
    PhongBRDF,
    CookTorranceBRDF,
    QuantumBRDF {
        quantum_surface_model: Array2<Complex64>,
    },
}
#[derive(Debug, Clone)]
pub struct MLPOutput {
    pub color: Array1<f64>,
    pub density: f64,
    pub quantum_state: QuantumMLPState,
}
#[derive(Debug, Clone)]
pub enum QuantumMLPLayerType {
    QuantumLinear,
    QuantumConvolutional3D {
        kernel_size: usize,
        stride: usize,
        padding: usize,
    },
    QuantumResidual {
        inner_layers: Vec<Box<QuantumMLPLayer>>,
    },
    QuantumAttentionLayer {
        attention_config: QuantumAttentionConfig,
    },
}
#[derive(Debug, Clone)]
pub struct QuantumAttentionConfig {
    pub use_spatial_attention: bool,
    pub use_view_attention: bool,
    pub use_scale_attention: bool,
    pub num_attention_heads: usize,
    pub attention_type: QuantumAttentionType,
    pub entanglement_in_attention: bool,
    pub quantum_query_key_value: bool,
}
#[derive(Debug, Clone)]
pub struct QuantumRenderOutput {
    pub rendered_image: Array3<f64>,
    pub quantum_depth_map: Array2<f64>,
    pub quantum_uncertainty_map: Array2<f64>,
    pub pixel_quantum_states: Vec<QuantumMLPState>,
    pub rendering_metrics: RenderingMetrics,
}
#[derive(Debug, Clone)]
pub struct QuantumMLPLayer {
    pub(super) layer_type: QuantumMLPLayerType,
    pub(super) input_dim: usize,
    pub(super) output_dim: usize,
    pub(super) quantum_gates: Vec<QuantumMLPGate>,
    pub(super) activation: QuantumActivationType,
    pub(super) normalization: Option<QuantumNormalizationType>,
}
#[derive(Debug, Clone)]
pub struct QuantumMaterialParameters {
    pub(super) albedo: Array1<f64>,
    pub(super) roughness: f64,
    pub(super) metallic: f64,
    pub(super) quantum_properties: QuantumMaterialProperties,
}
#[derive(Debug, Clone)]
pub struct QuantumSDF {
    pub(super) quantum_parameters: Array1<f64>,
    pub(super) quantum_basis_functions: Vec<QuantumBasisFunction>,
    pub(super) multi_resolution_levels: usize,
}
#[derive(Debug, Clone)]
pub struct MLPLayerOutput {
    pub features: Array1<f64>,
    pub quantum_state: QuantumMLPState,
}
#[derive(Debug, Clone)]
pub enum QuantumSubdivisionCriterion {
    DensityThreshold { threshold: f64 },
    QuantumUncertainty { uncertainty_threshold: f64 },
    EntanglementComplexity { complexity_threshold: f64 },
    AdaptiveQuantum { adaptive_parameters: Array1<f64> },
}
#[derive(Debug, Clone)]
pub struct QuantumViewEncoder {
    pub(super) encoding_dimension: usize,
    pub(super) quantum_view_embedding: Array2<Complex64>,
    pub(super) spherical_harmonics_order: usize,
    pub(super) quantum_spherical_harmonics: bool,
}
#[derive(Debug, Clone)]
pub struct QuantumMLPGate {
    pub(super) gate_type: QuantumMLPGateType,
    pub(super) target_qubits: Vec<usize>,
    pub(super) control_qubits: Vec<usize>,
    pub(super) parameters: Array1<f64>,
    pub(super) is_trainable: bool,
}
#[derive(Debug, Clone)]
pub enum QuantumPositionalEncodingType {
    StandardQuantumEncoding,
    QuantumFourierEncoding,
    QuantumWaveletEncoding,
    EntanglementBasedEncoding,
    QuantumHashEncoding { hash_table_size: usize },
    QuantumMultiresolutionEncoding { num_levels: usize },
}
#[derive(Debug, Clone)]
pub struct QuantumGradientFunction {
    pub(super) gradient_quantum_mlp: QuantumMLP,
    pub(super) analytical_gradients: bool,
    pub(super) quantum_finite_differences: bool,
}
#[derive(Debug, Clone)]
pub struct QuantumIllumination {
    pub(super) light_sources: Vec<QuantumLightSource>,
    pub(super) ambient_lighting: QuantumAmbientLight,
    pub(super) quantum_shadows: bool,
    pub(super) quantum_global_illumination: bool,
}
#[derive(Debug, Clone)]
pub struct QuantumRenderingMetrics {
    pub average_rendering_time: f64,
    pub quantum_acceleration_factor: f64,
    pub entanglement_utilization: f64,
    pub coherence_preservation: f64,
    pub quantum_memory_efficiency: f64,
    pub view_synthesis_quality: f64,
    pub volumetric_accuracy: f64,
}
#[derive(Debug, Clone)]
pub struct QuantumSceneRepresentation {
    pub(super) voxel_grid: QuantumVoxelGrid,
    pub(super) implicit_surface: QuantumImplicitSurface,
    pub(super) quantum_octree: QuantumOctree,
    pub(super) multi_scale_features: Vec<QuantumFeatureLevel>,
}
#[derive(Debug, Clone)]
pub struct Ray {
    pub origin: Array1<f64>,
    pub direction: Array1<f64>,
    pub near: f64,
    pub far: f64,
}
#[derive(Debug, Clone)]
pub enum QuantumUpsampling {
    QuantumBilinearInterpolation,
    QuantumTransposedConvolution,
    QuantumAttentionUpsampling,
    EntanglementBasedUpsampling,
}
#[derive(Debug, Clone)]
pub enum QuantumMaterialType {
    Lambertian,
    Phong,
    PBR,
    QuantumMaterial {
        quantum_reflectance: Array2<Complex64>,
        quantum_transmittance: Array2<Complex64>,
    },
}
#[derive(Debug, Clone)]
pub struct QuantumBasisFunction {
    pub(super) basis_type: QuantumBasisType,
    pub(super) parameters: Array1<Complex64>,
    pub(super) support_region: Array1<f64>,
}
#[derive(Debug, Clone)]
pub struct PixelRenderOutput {
    pub color: Array1<f64>,
    pub depth: f64,
    pub quantum_uncertainty: f64,
    pub quantum_state: QuantumMLPState,
}
#[derive(Debug, Clone)]
pub enum QuantumEnvironmentEncodingType {
    SphericalHarmonics,
    QuantumSphericalHarmonics,
    QuantumWavelets,
    QuantumFourierSeries,
}
#[derive(Debug, Clone)]
pub struct NeRFOptimizationState {
    pub learning_rate: f64,
    pub momentum: f64,
    pub quantum_parameter_learning_rate: f64,
    pub adaptive_sampling_rate: f64,
    pub entanglement_preservation_weight: f64,
    pub rendering_loss_weight: f64,
}
#[derive(Debug, Clone)]
pub struct RenderingMetrics {
    pub average_pixel_entanglement: f64,
    pub average_quantum_fidelity: f64,
    pub rendering_quantum_advantage: f64,
    pub coherence_preservation: f64,
}
#[derive(Debug, Clone)]
pub struct VolumeRenderOutput {
    pub final_color: Array1<f64>,
    pub depth: f64,
    pub quantum_uncertainty: f64,
    pub accumulated_quantum_state: QuantumMLPState,
}
#[derive(Debug, Clone)]
pub struct NeRFTrainingOutput {
    pub training_losses: Vec<f64>,
    pub quantum_metrics_history: Vec<QuantumRenderingMetrics>,
    pub final_rendering_quality: f64,
    pub convergence_analysis: NeRFConvergenceAnalysis,
}
#[derive(Debug, Clone)]
pub struct QuantumSurfaceProperties {
    pub(super) surface_normal: Array1<f64>,
    pub(super) curvature: f64,
    pub(super) quantum_surface_features: Array1<Complex64>,
}
#[derive(Debug, Clone)]
pub struct SamplingPoint {
    pub position: Array1<f64>,
    pub quantum_weight: f64,
    pub entanglement_correlation: f64,
}
#[derive(Debug, Clone)]
pub struct NeRFTrainingMetrics {
    pub epoch: usize,
    pub loss: f64,
    pub psnr: f64,
    pub ssim: f64,
    pub lpips: f64,
    pub quantum_fidelity: f64,
    pub entanglement_measure: f64,
    pub rendering_time: f64,
    pub quantum_advantage_ratio: f64,
    pub memory_usage: f64,
}
#[derive(Debug, Clone)]
pub struct VolumetricRenderingConfig {
    pub use_quantum_alpha_compositing: bool,
    pub quantum_density_activation: QuantumActivationType,
    pub quantum_color_space: QuantumColorSpace,
    pub quantum_illumination_model: QuantumIlluminationModel,
    pub quantum_material_properties: bool,
    pub quantum_light_transport: bool,
}
#[derive(Debug, Clone)]
pub struct QuantumMaterialModel {
    pub(super) material_type: QuantumMaterialType,
    pub(super) quantum_brdf: QuantumBRDF,
    pub(super) material_parameters: QuantumMaterialParameters,
}
#[derive(Debug, Clone)]
pub struct NeRFTrainingConfig {
    pub epochs: usize,
    pub rays_per_batch: usize,
    pub learning_rate: f64,
    pub learning_rate_decay: f64,
    pub quantum_loss_weight: f64,
    pub log_interval: usize,
}
#[derive(Debug, Clone)]
pub struct QuantumVoxelGrid {
    pub(super) density_grid: Array3<f64>,
    pub(super) color_grid: Array4<f64>,
    pub(super) quantum_features: Array4<Complex64>,
    pub(super) entanglement_structure: VoxelEntanglementStructure,
}
#[derive(Debug, Clone)]
pub struct QuantumLightField {
    pub(super) light_directions: Array2<f64>,
    pub(super) light_intensities: Array2<f64>,
    pub(super) quantum_light_coherence: Array2<Complex64>,
    pub(super) spherical_harmonics_coefficients: Array2<f64>,
    pub(super) quantum_environment_encoding: QuantumEnvironmentEncoding,
}
#[derive(Debug, Clone)]
pub struct QuantumOctree {
    pub(super) root: QuantumOctreeNode,
    pub(super) max_depth: usize,
    pub(super) quantum_subdivision_criterion: QuantumSubdivisionCriterion,
}
#[derive(Debug, Clone)]
pub struct QuantumMLPState {
    pub quantum_amplitudes: Array1<Complex64>,
    pub entanglement_measure: f64,
    pub quantum_fidelity: f64,
}
#[derive(Debug, Clone)]
pub struct VoxelEntanglementStructure {
    pub(super) entanglement_matrix: Array2<f64>,
    pub(super) correlation_radius: f64,
    pub(super) entanglement_strength: f64,
}
/// Configuration for Quantum Neural Radiance Fields
#[derive(Debug, Clone)]
pub struct QuantumNeRFConfig {
    pub scene_bounds: SceneBounds,
    pub num_qubits: usize,
    pub quantum_encoding_levels: usize,
    pub max_ray_samples: usize,
    pub quantum_sampling_strategy: QuantumSamplingStrategy,
    pub quantum_enhancement_level: f64,
    pub use_quantum_positional_encoding: bool,
    pub quantum_attention_config: QuantumAttentionConfig,
    pub volumetric_rendering_config: VolumetricRenderingConfig,
    pub quantum_multiscale_features: bool,
    pub entanglement_based_interpolation: bool,
    pub quantum_view_synthesis: bool,
    pub decoherence_mitigation: DecoherenceMitigationConfig,
}
#[derive(Debug, Clone)]
pub struct QuantumAmbientLight {
    pub(super) ambient_color: Array1<f64>,
    pub(super) quantum_ambient_occlusion: bool,
    pub(super) quantum_environment_probe: Option<Array3<f64>>,
}
#[derive(Debug, Clone)]
pub struct QuantumRayMarcher {
    pub(super) marching_strategy: QuantumMarchingStrategy,
    pub(super) quantum_sampling_points: Array2<f64>,
    pub(super) entanglement_based_sampling: bool,
    pub(super) adaptive_step_size: bool,
}
#[derive(Debug, Clone)]
pub struct CameraMatrix {
    pub position: Array1<f64>,
    pub forward: Array1<f64>,
    pub right: Array1<f64>,
    pub up: Array1<f64>,
    pub fov: f64,
}
#[derive(Debug, Clone)]
pub enum QuantumMarchingStrategy {
    UniformMarching {
        step_size: f64,
    },
    AdaptiveMarching {
        initial_step_size: f64,
        min_step_size: f64,
        max_step_size: f64,
    },
    QuantumImportanceMarching {
        importance_threshold: f64,
        quantum_importance_estimation: bool,
    },
    EntanglementGuidedMarching {
        entanglement_threshold: f64,
        correlation_distance: f64,
    },
}
#[derive(Debug, Clone)]
pub struct QuantumMaterialProperties {
    pub(super) quantum_reflectivity: Complex64,
    pub(super) quantum_absorption: Complex64,
    pub(super) quantum_scattering: Complex64,
    pub(super) entanglement_factor: f64,
}
#[derive(Debug, Clone)]
pub enum QuantumDownsampling {
    QuantumAveragePooling,
    QuantumMaxPooling,
    QuantumAttentionPooling,
    EntanglementBasedPooling,
}
#[derive(Debug, Clone)]
pub struct QuantumMLP {
    pub(super) layers: Vec<QuantumMLPLayer>,
    pub(super) skip_connections: Vec<usize>,
    pub(super) quantum_parameters: Array1<f64>,
    pub(super) classical_parameters: Array2<f64>,
    pub(super) quantum_enhancement_factor: f64,
}
#[derive(Debug, Clone)]
pub enum QuantumBlendingMode {
    StandardAlphaBlending,
    QuantumSuperpositionBlending,
    EntanglementBasedBlending,
    QuantumInterferenceBlending,
}
#[derive(Debug, Clone)]
pub struct QuantumEncodingOutput {
    pub features: Array1<f64>,
    pub quantum_amplitudes: Array1<Complex64>,
    pub entanglement_measure: f64,
}
#[derive(Debug, Clone)]
pub struct DecoherenceMitigationConfig {
    pub enable_error_correction: bool,
    pub coherence_preservation_weight: f64,
    pub decoherence_compensation_factor: f64,
    pub quantum_error_rate_threshold: f64,
}
#[derive(Debug, Clone)]
pub struct QuantumFeatureLevel {
    pub(super) level: usize,
    pub(super) resolution: Array1<usize>,
    pub(super) quantum_features: Array4<Complex64>,
    pub(super) downsampling_operator: QuantumDownsampling,
    pub(super) upsampling_operator: QuantumUpsampling,
}
#[derive(Debug, Clone)]
pub enum QuantumProposalType {
    QuantumGaussian { sigma: f64 },
    QuantumLevyFlight { alpha: f64 },
    QuantumMetropolis { temperature: f64 },
}
#[derive(Debug, Clone)]
pub enum QuantumIlluminationModel {
    Lambertian,
    Phong,
    PBR,
    QuantumPhotonMapping,
    QuantumLightTransport,
    EntanglementBasedLighting,
}
#[derive(Debug, Clone, Default)]
pub struct NeRFConvergenceAnalysis {
    pub convergence_rate: f64,
    pub final_loss: f64,
    pub rendering_quality_score: f64,
    pub quantum_advantage_achieved: bool,
}
#[derive(Debug, Clone)]
pub enum QuantumSamplingStrategy {
    /// Uniform sampling with quantum noise
    QuantumUniform {
        min_samples: usize,
        max_samples: usize,
        quantum_jitter: f64,
    },
    /// Hierarchical sampling with quantum importance
    QuantumHierarchical {
        coarse_samples: usize,
        fine_samples: usize,
        quantum_importance_threshold: f64,
    },
    /// Quantum adaptive sampling based on uncertainty
    QuantumAdaptive {
        initial_samples: usize,
        max_refinements: usize,
        uncertainty_threshold: f64,
        quantum_uncertainty_estimation: bool,
    },
    /// Entanglement-based correlated sampling
    EntanglementCorrelated {
        base_samples: usize,
        correlation_strength: f64,
        entanglement_radius: f64,
    },
    /// Quantum Monte Carlo sampling
    QuantumMonteCarlo {
        num_chains: usize,
        chain_length: usize,
        quantum_proposal_distribution: QuantumProposalType,
    },
}
#[derive(Debug, Clone)]
pub struct QuantumImplicitSurface {
    pub(super) sdf_function: QuantumSDF,
    pub(super) gradient_function: QuantumGradientFunction,
    pub(super) quantum_surface_properties: QuantumSurfaceProperties,
}
#[derive(Debug, Clone)]
pub struct QuantumSamplingOutput {
    pub points: Vec<SamplingPoint>,
    pub distances: Vec<f64>,
    pub is_hierarchical: bool,
}
#[derive(Debug, Clone)]
pub struct QuantumScaleAttention {
    pub(super) num_scales: usize,
    pub(super) scale_weights: Array1<f64>,
    pub(super) quantum_scale_mixing: Array2<Complex64>,
    pub(super) adaptive_scale_selection: bool,
}
#[derive(Debug, Clone)]
pub struct RaySample {
    pub ray: Ray,
    pub target_color: Array1<f64>,
    pub pixel_coords: [usize; 2],
}
#[derive(Debug, Clone)]
pub struct TrainingImage {
    pub image: Array3<f64>,
    pub camera_matrix: CameraMatrix,
    pub fov: f64,
}
#[derive(Debug, Clone)]
pub enum QuantumNormalizationType {
    QuantumBatchNorm,
    QuantumLayerNorm,
    QuantumInstanceNorm,
    QuantumGroupNorm { num_groups: usize },
    EntanglementNorm,
}
#[derive(Debug, Clone)]
pub struct QuantumVolumeRenderer {
    pub(super) rendering_equation: QuantumRenderingEquation,
    pub(super) quantum_alpha_blending: QuantumAlphaBlending,
    pub(super) quantum_illumination: QuantumIllumination,
    pub(super) quantum_material_model: QuantumMaterialModel,
}
#[derive(Debug, Clone)]
pub enum QuantumLightType {
    QuantumPointLight,
    QuantumDirectionalLight,
    QuantumAreaLight { area_size: Array1<f64> },
    QuantumEnvironmentLight { environment_map: Array3<f64> },
    QuantumCoherentLight { coherence_length: f64 },
}
#[derive(Debug, Clone)]
pub enum QuantumMLPGateType {
    ParameterizedRotation { axis: RotationAxis },
    ControlledRotation { axis: RotationAxis },
    EntanglementGate { gate_name: String },
    QuantumFourierGate,
    CustomQuantumGate { matrix: Array2<Complex64> },
}
#[derive(Debug, Clone)]
pub struct QuantumOctreeNode {
    pub(super) bounds: SceneBounds,
    pub(super) children: Option<Box<[QuantumOctreeNode; 8]>>,
    pub(super) quantum_features: Array1<Complex64>,
    pub(super) occupancy_probability: f64,
    pub(super) entanglement_with_neighbors: Array1<f64>,
}
#[derive(Debug, Clone)]
pub struct QuantumAlphaBlending {
    pub(super) blending_mode: QuantumBlendingMode,
    pub(super) quantum_compositing: bool,
    pub(super) entanglement_based_blending: bool,
}
#[derive(Debug, Clone)]
pub struct QuantumBRDF {
    pub(super) brdf_type: QuantumBRDFType,
    pub(super) quantum_parameters: Array1<Complex64>,
    pub(super) view_dependent: bool,
}
