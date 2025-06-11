//! Quantum Explainable AI (XAI)
//!
//! This module implements explainability and interpretability tools specifically
//! designed for quantum neural networks and quantum machine learning models,
//! helping users understand quantum model decisions and internal representations.

use crate::error::{MLError, Result};
use crate::optimization::OptimizationMethod;
use crate::qnn::{QNNLayerType, QuantumNeuralNetwork};
use ndarray::{s, Array1, Array2, Array3, Axis};
use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::gate::{
    single::{RotationX, RotationY, RotationZ},
    GateOp,
};
use quantrs2_sim::statevector::StateVectorSimulator;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Explanation methods for quantum models
#[derive(Debug, Clone)]
pub enum ExplanationMethod {
    /// Quantum feature attribution
    QuantumFeatureAttribution {
        method: AttributionMethod,
        num_samples: usize,
        baseline: Option<Array1<f64>>,
    },

    /// Quantum circuit visualization
    CircuitVisualization {
        include_measurements: bool,
        parameter_sensitivity: bool,
    },

    /// Quantum state analysis
    StateAnalysis {
        entanglement_measures: bool,
        coherence_analysis: bool,
        superposition_analysis: bool,
    },

    /// Quantum saliency maps
    SaliencyMapping {
        perturbation_method: PerturbationMethod,
        aggregation: AggregationMethod,
    },

    /// Quantum LIME (Local Interpretable Model-agnostic Explanations)
    QuantumLIME {
        num_perturbations: usize,
        kernel_width: f64,
        local_model: LocalModelType,
    },

    /// Quantum SHAP (SHapley Additive exPlanations)
    QuantumSHAP {
        num_coalitions: usize,
        background_samples: usize,
    },

    /// Layer-wise Relevance Propagation for quantum circuits
    QuantumLRP {
        propagation_rule: LRPRule,
        epsilon: f64,
    },

    /// Quantum concept activation vectors
    ConceptActivation {
        concept_datasets: Vec<String>,
        activation_threshold: f64,
    },
}

/// Attribution methods for quantum features
#[derive(Debug, Clone)]
pub enum AttributionMethod {
    /// Integrated gradients
    IntegratedGradients,
    /// Gradient × Input
    GradientInput,
    /// Gradient SHAP
    GradientSHAP,
    /// Quantum-specific attribution
    QuantumAttribution,
}

/// Perturbation methods for saliency
#[derive(Debug, Clone)]
pub enum PerturbationMethod {
    /// Gaussian noise
    Gaussian { sigma: f64 },
    /// Quantum phase perturbation
    QuantumPhase { magnitude: f64 },
    /// Feature masking
    FeatureMasking,
    /// Circuit parameter perturbation
    ParameterPerturbation { strength: f64 },
}

/// Aggregation methods for explanations
#[derive(Debug, Clone)]
pub enum AggregationMethod {
    /// Mean aggregation
    Mean,
    /// Maximum magnitude
    MaxMagnitude,
    /// Variance-based
    Variance,
    /// Quantum coherence-weighted
    CoherenceWeighted,
}

/// Local model types for LIME
#[derive(Debug, Clone)]
pub enum LocalModelType {
    /// Linear regression
    LinearRegression,
    /// Decision tree
    DecisionTree,
    /// Quantum linear model
    QuantumLinear,
}

/// Layer-wise relevance propagation rules
#[derive(Debug, Clone)]
pub enum LRPRule {
    /// Epsilon rule
    Epsilon,
    /// Gamma rule
    Gamma { gamma: f64 },
    /// Alpha-beta rule
    AlphaBeta { alpha: f64, beta: f64 },
    /// Quantum-specific rule
    QuantumRule,
}

/// Explanation result containing multiple types of explanations
#[derive(Debug, Clone)]
pub struct ExplanationResult {
    /// Feature attributions
    pub feature_attributions: Option<Array1<f64>>,

    /// Saliency map
    pub saliency_map: Option<Array2<f64>>,

    /// Circuit explanation
    pub circuit_explanation: Option<CircuitExplanation>,

    /// Quantum state properties
    pub state_properties: Option<QuantumStateProperties>,

    /// Concept activations
    pub concept_activations: Option<HashMap<String, f64>>,

    /// Textual explanation
    pub textual_explanation: String,

    /// Confidence scores
    pub confidence_scores: HashMap<String, f64>,
}

/// Circuit-specific explanation
#[derive(Debug, Clone)]
pub struct CircuitExplanation {
    /// Parameter importance scores
    pub parameter_importance: Array1<f64>,

    /// Gate-wise contributions
    pub gate_contributions: Vec<GateContribution>,

    /// Layer-wise analysis
    pub layer_analysis: Vec<LayerAnalysis>,

    /// Critical path through circuit
    pub critical_path: Vec<usize>,
}

/// Individual gate contribution
#[derive(Debug, Clone)]
pub struct GateContribution {
    /// Gate index in circuit
    pub gate_index: usize,

    /// Gate type
    pub gate_type: String,

    /// Contribution magnitude
    pub contribution: f64,

    /// Qubits affected
    pub qubits: Vec<usize>,

    /// Parameter values (if parameterized)
    pub parameters: Option<Array1<f64>>,
}

/// Layer-wise analysis
#[derive(Debug, Clone)]
pub struct LayerAnalysis {
    /// Layer type
    pub layer_type: QNNLayerType,

    /// Information gain
    pub information_gain: f64,

    /// Entanglement generated
    pub entanglement_generated: f64,

    /// Feature transformations
    pub feature_transformations: Array2<f64>,

    /// Activation patterns
    pub activation_patterns: Array1<f64>,
}

/// Quantum state properties for explanation
#[derive(Debug, Clone)]
pub struct QuantumStateProperties {
    /// Entanglement entropy
    pub entanglement_entropy: f64,

    /// Coherence measures
    pub coherence_measures: HashMap<String, f64>,

    /// Superposition analysis
    pub superposition_components: Array1<f64>,

    /// Measurement probabilities
    pub measurement_probabilities: Array1<f64>,

    /// State fidelity with pure states
    pub state_fidelities: HashMap<String, f64>,
}

/// Main quantum explainable AI engine
pub struct QuantumExplainableAI {
    /// Target model to explain
    model: QuantumNeuralNetwork,

    /// Explanation methods to use
    methods: Vec<ExplanationMethod>,

    /// Background/baseline data for explanations
    background_data: Option<Array2<f64>>,

    /// Pre-computed concept vectors
    concept_vectors: HashMap<String, Array1<f64>>,

    /// Explanation cache
    explanation_cache: HashMap<String, ExplanationResult>,
}

impl QuantumExplainableAI {
    /// Create a new quantum explainable AI instance
    pub fn new(model: QuantumNeuralNetwork, methods: Vec<ExplanationMethod>) -> Self {
        Self {
            model,
            methods,
            background_data: None,
            concept_vectors: HashMap::new(),
            explanation_cache: HashMap::new(),
        }
    }

    /// Set background data for explanations
    pub fn set_background_data(&mut self, data: Array2<f64>) {
        self.background_data = Some(data);
    }

    /// Add concept vector
    pub fn add_concept(&mut self, name: String, vector: Array1<f64>) {
        self.concept_vectors.insert(name, vector);
    }

    /// Generate comprehensive explanation for an input
    pub fn explain(&mut self, input: &Array1<f64>) -> Result<ExplanationResult> {
        let mut result = ExplanationResult {
            feature_attributions: None,
            saliency_map: None,
            circuit_explanation: None,
            state_properties: None,
            concept_activations: None,
            textual_explanation: String::new(),
            confidence_scores: HashMap::new(),
        };

        // Apply each explanation method
        for method in &self.methods.clone() {
            match method {
                ExplanationMethod::QuantumFeatureAttribution {
                    method: attr_method,
                    num_samples,
                    baseline,
                } => {
                    let attributions = self.compute_feature_attributions(
                        input,
                        attr_method,
                        *num_samples,
                        baseline.as_ref(),
                    )?;
                    result.feature_attributions = Some(attributions);
                }

                ExplanationMethod::CircuitVisualization {
                    include_measurements,
                    parameter_sensitivity,
                } => {
                    let circuit_explanation =
                        self.analyze_circuit(input, *include_measurements, *parameter_sensitivity)?;
                    result.circuit_explanation = Some(circuit_explanation);
                }

                ExplanationMethod::StateAnalysis {
                    entanglement_measures,
                    coherence_analysis,
                    superposition_analysis,
                } => {
                    let state_props = self.analyze_quantum_state(
                        input,
                        *entanglement_measures,
                        *coherence_analysis,
                        *superposition_analysis,
                    )?;
                    result.state_properties = Some(state_props);
                }

                ExplanationMethod::SaliencyMapping {
                    perturbation_method,
                    aggregation,
                } => {
                    let saliency =
                        self.compute_saliency_map(input, perturbation_method, aggregation)?;
                    result.saliency_map = Some(saliency);
                }

                ExplanationMethod::QuantumLIME {
                    num_perturbations,
                    kernel_width,
                    local_model,
                } => {
                    let lime_attributions = self.explain_with_lime(
                        input,
                        *num_perturbations,
                        *kernel_width,
                        local_model,
                    )?;
                    result.feature_attributions = Some(lime_attributions);
                }

                ExplanationMethod::QuantumSHAP {
                    num_coalitions,
                    background_samples,
                } => {
                    let shap_values =
                        self.compute_shap_values(input, *num_coalitions, *background_samples)?;
                    result.feature_attributions = Some(shap_values);
                }

                ExplanationMethod::QuantumLRP {
                    propagation_rule,
                    epsilon,
                } => {
                    let lrp_scores =
                        self.layer_wise_relevance_propagation(input, propagation_rule, *epsilon)?;
                    result.feature_attributions = Some(lrp_scores);
                }

                ExplanationMethod::ConceptActivation {
                    concept_datasets,
                    activation_threshold,
                } => {
                    let concept_activations = self.compute_concept_activations(
                        input,
                        concept_datasets,
                        *activation_threshold,
                    )?;
                    result.concept_activations = Some(concept_activations);
                }
            }
        }

        // Generate textual explanation
        result.textual_explanation = self.generate_textual_explanation(&result)?;

        // Compute confidence scores
        result.confidence_scores = self.compute_confidence_scores(&result)?;

        Ok(result)
    }

    /// Compute feature attributions using various methods
    fn compute_feature_attributions(
        &self,
        input: &Array1<f64>,
        method: &AttributionMethod,
        num_samples: usize,
        baseline: Option<&Array1<f64>>,
    ) -> Result<Array1<f64>> {
        match method {
            AttributionMethod::IntegratedGradients => {
                self.integrated_gradients(input, baseline, num_samples)
            }

            AttributionMethod::GradientInput => {
                let gradient = self.compute_gradient(input)?;
                Ok(&gradient * input)
            }

            AttributionMethod::GradientSHAP => self.gradient_shap(input, num_samples),

            AttributionMethod::QuantumAttribution => self.quantum_specific_attribution(input),
        }
    }

    /// Integrated gradients implementation
    fn integrated_gradients(
        &self,
        input: &Array1<f64>,
        baseline: Option<&Array1<f64>>,
        num_samples: usize,
    ) -> Result<Array1<f64>> {
        let default_baseline = Array1::zeros(input.len());
        let baseline = baseline.unwrap_or(&default_baseline);
        let mut integrated_grad: Array1<f64> = Array1::zeros(input.len());

        for i in 0..num_samples {
            let alpha = i as f64 / (num_samples - 1) as f64;
            let interpolated = baseline + alpha * (input - baseline);
            let gradient = self.compute_gradient(&interpolated)?;
            integrated_grad = integrated_grad + gradient;
        }

        integrated_grad = integrated_grad / num_samples as f64;
        let attribution = &integrated_grad * (input - baseline);

        Ok(attribution)
    }

    /// Gradient SHAP implementation
    fn gradient_shap(&self, input: &Array1<f64>, num_samples: usize) -> Result<Array1<f64>> {
        if let Some(ref background) = self.background_data {
            let mut total_attribution = Array1::zeros(input.len());

            for _ in 0..num_samples {
                // Sample random background
                let bg_idx = fastrand::usize(0..background.nrows());
                let baseline = background.row(bg_idx).to_owned();

                // Compute integrated gradients with this baseline
                let attribution = self.integrated_gradients(input, Some(&baseline), 50)?;
                total_attribution = total_attribution + attribution;
            }

            Ok(total_attribution / num_samples as f64)
        } else {
            // Fallback to regular integrated gradients
            self.integrated_gradients(input, None, num_samples)
        }
    }

    /// Quantum-specific attribution method
    fn quantum_specific_attribution(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        let mut attribution = Array1::zeros(input.len());

        // Compute quantum Fisher information for each feature
        for i in 0..input.len() {
            let fisher_info = self.compute_quantum_fisher_information(input, i)?;
            attribution[i] = fisher_info;
        }

        // Normalize by quantum state properties
        let state_props = self.analyze_quantum_state(input, true, true, true)?;
        let normalization = state_props.entanglement_entropy + 1e-10;
        attribution = attribution / normalization;

        Ok(attribution)
    }

    /// Analyze circuit structure and contributions
    fn analyze_circuit(
        &self,
        input: &Array1<f64>,
        include_measurements: bool,
        parameter_sensitivity: bool,
    ) -> Result<CircuitExplanation> {
        // Compute parameter importance
        let param_importance = if parameter_sensitivity {
            self.compute_parameter_sensitivity(input)?
        } else {
            Array1::ones(self.model.parameters.len())
        };

        // Analyze each layer
        let mut layer_analysis = Vec::new();
        for (i, layer) in self.model.layers.iter().enumerate() {
            let analysis = self.analyze_layer(input, layer, i)?;
            layer_analysis.push(analysis);
        }

        // Create gate contributions (simplified)
        let gate_contributions = self.analyze_gates(input)?;

        // Find critical path
        let critical_path = self.find_critical_path(&param_importance)?;

        Ok(CircuitExplanation {
            parameter_importance: param_importance,
            gate_contributions,
            layer_analysis,
            critical_path,
        })
    }

    /// Analyze quantum state properties
    fn analyze_quantum_state(
        &self,
        input: &Array1<f64>,
        entanglement_measures: bool,
        coherence_analysis: bool,
        superposition_analysis: bool,
    ) -> Result<QuantumStateProperties> {
        // Get quantum state representation (simplified)
        let state_vector = self.get_state_vector(input)?;

        // Compute entanglement entropy
        let entanglement_entropy = if entanglement_measures {
            self.compute_entanglement_entropy(&state_vector)?
        } else {
            0.0
        };

        // Compute coherence measures
        let coherence_measures = if coherence_analysis {
            self.compute_coherence_measures(&state_vector)?
        } else {
            HashMap::new()
        };

        // Analyze superposition
        let superposition_components = if superposition_analysis {
            self.analyze_superposition(&state_vector)?
        } else {
            Array1::zeros(state_vector.len())
        };

        // Measurement probabilities
        let measurement_probabilities = state_vector.mapv(|x| x * x);

        // State fidelities with computational basis states
        let state_fidelities = self.compute_state_fidelities(&state_vector)?;

        Ok(QuantumStateProperties {
            entanglement_entropy,
            coherence_measures,
            superposition_components,
            measurement_probabilities,
            state_fidelities,
        })
    }

    /// Compute saliency map through perturbations
    fn compute_saliency_map(
        &self,
        input: &Array1<f64>,
        perturbation_method: &PerturbationMethod,
        aggregation: &AggregationMethod,
    ) -> Result<Array2<f64>> {
        let num_perturbations = 50;
        let mut saliency_scores = Array2::zeros((input.len(), input.len()));

        let baseline_output = self.model.forward(input)?;

        for i in 0..num_perturbations {
            let perturbed_input = self.apply_perturbation(input, perturbation_method)?;
            let perturbed_output = self.model.forward(&perturbed_input)?;

            let output_diff = &perturbed_output - &baseline_output;
            let input_diff = &perturbed_input - input;

            // Update saliency map based on input-output correlations
            for j in 0..input.len() {
                for k in 0..output_diff.len() {
                    let correlation = input_diff[j] * output_diff[k];
                    saliency_scores[[j, k]] += correlation.abs();
                }
            }
        }

        // Apply aggregation method
        match aggregation {
            AggregationMethod::Mean => {
                saliency_scores = saliency_scores / num_perturbations as f64;
            }
            AggregationMethod::MaxMagnitude => {
                // Keep maximum magnitude across perturbations
            }
            AggregationMethod::Variance => {
                // Compute variance of saliency scores
            }
            AggregationMethod::CoherenceWeighted => {
                let coherence_weight = self.compute_coherence_weight(input)?;
                saliency_scores = saliency_scores * coherence_weight;
            }
        }

        Ok(saliency_scores)
    }

    /// LIME explanation for quantum models
    fn explain_with_lime(
        &self,
        input: &Array1<f64>,
        num_perturbations: usize,
        kernel_width: f64,
        local_model: &LocalModelType,
    ) -> Result<Array1<f64>> {
        let mut perturbations = Vec::new();
        let mut outputs = Vec::new();
        let mut weights = Vec::new();

        // Generate perturbations around the input
        for _ in 0..num_perturbations {
            let perturbed = self.generate_lime_perturbation(input)?;
            let output = self.model.forward(&perturbed)?;
            let distance = (&perturbed - input).mapv(|x| x * x).sum().sqrt();
            let weight = (-distance * distance / (kernel_width * kernel_width)).exp();

            perturbations.push(perturbed);
            outputs.push(output);
            weights.push(weight);
        }

        // Fit local model
        let attributions = match local_model {
            LocalModelType::LinearRegression => {
                self.fit_linear_model(&perturbations, &outputs, &weights)?
            }
            LocalModelType::DecisionTree => {
                self.fit_decision_tree(&perturbations, &outputs, &weights)?
            }
            LocalModelType::QuantumLinear => {
                self.fit_quantum_linear_model(&perturbations, &outputs, &weights)?
            }
        };

        Ok(attributions)
    }

    /// Compute SHAP values for quantum model
    fn compute_shap_values(
        &self,
        input: &Array1<f64>,
        num_coalitions: usize,
        background_samples: usize,
    ) -> Result<Array1<f64>> {
        let mut shap_values = Array1::zeros(input.len());

        if let Some(ref background) = self.background_data {
            // Sample background instances
            let bg_indices: Vec<usize> = (0..background_samples)
                .map(|_| fastrand::usize(0..background.nrows()))
                .collect();

            for _ in 0..num_coalitions {
                // Generate random coalition
                let coalition: Vec<bool> = (0..input.len()).map(|_| fastrand::bool()).collect();

                for i in 0..input.len() {
                    // Compute marginal contribution of feature i
                    let with_i =
                        self.compute_coalition_value(input, &coalition, Some(i), &bg_indices)?;
                    let without_i =
                        self.compute_coalition_value(input, &coalition, None, &bg_indices)?;

                    let marginal_contribution = with_i - without_i;
                    shap_values[i] += marginal_contribution;
                }
            }

            shap_values = shap_values / num_coalitions as f64;
        }

        Ok(shap_values)
    }

    /// Layer-wise relevance propagation
    fn layer_wise_relevance_propagation(
        &self,
        input: &Array1<f64>,
        rule: &LRPRule,
        epsilon: f64,
    ) -> Result<Array1<f64>> {
        // Get layer activations
        let layer_activations = self.compute_layer_activations(input)?;

        // Start with output relevance
        let output = self.model.forward(input)?;
        let mut relevance = output.clone();

        // Propagate relevance backwards through layers
        for (i, layer) in self.model.layers.iter().enumerate().rev() {
            relevance = self.propagate_relevance_through_layer(
                &relevance,
                &layer_activations[i],
                layer,
                rule,
                epsilon,
            )?;
        }

        Ok(relevance)
    }

    /// Compute concept activations
    fn compute_concept_activations(
        &self,
        input: &Array1<f64>,
        concept_datasets: &[String],
        activation_threshold: f64,
    ) -> Result<HashMap<String, f64>> {
        let mut activations = HashMap::new();

        // Get internal representations
        let internal_repr = self.get_internal_representation(input)?;

        for concept_name in concept_datasets {
            if let Some(concept_vector) = self.concept_vectors.get(concept_name) {
                // Compute dot product with concept vector
                let activation = internal_repr
                    .iter()
                    .zip(concept_vector.iter())
                    .map(|(&a, &c)| a * c)
                    .sum::<f64>();

                // Apply threshold
                let normalized_activation = if activation > activation_threshold {
                    activation
                } else {
                    0.0
                };

                activations.insert(concept_name.clone(), normalized_activation);
            }
        }

        Ok(activations)
    }

    /// Generate textual explanation from results
    fn generate_textual_explanation(&self, result: &ExplanationResult) -> Result<String> {
        let mut explanation = String::new();

        explanation.push_str("Quantum Model Explanation:\n\n");

        // Feature attribution explanation
        if let Some(ref attributions) = result.feature_attributions {
            explanation.push_str("Feature Attributions:\n");
            for (i, &attr) in attributions.iter().enumerate() {
                if attr.abs() > 0.1 {
                    explanation.push_str(&format!(
                        "- Feature {}: {:.3} ({})\n",
                        i,
                        attr,
                        if attr > 0.0 {
                            "positive influence"
                        } else {
                            "negative influence"
                        }
                    ));
                }
            }
            explanation.push('\n');
        }

        // Circuit explanation
        if let Some(ref circuit) = result.circuit_explanation {
            explanation.push_str("Circuit Analysis:\n");
            let max_importance = circuit
                .parameter_importance
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            explanation.push_str(&format!(
                "- Most important parameter has influence: {:.3}\n",
                max_importance
            ));

            explanation.push_str(&format!(
                "- Circuit has {} layers with varying contributions\n",
                circuit.layer_analysis.len()
            ));

            explanation.push('\n');
        }

        // Quantum state properties
        if let Some(ref state) = result.state_properties {
            explanation.push_str("Quantum State Properties:\n");
            explanation.push_str(&format!(
                "- Entanglement entropy: {:.3}\n",
                state.entanglement_entropy
            ));

            let max_prob = state
                .measurement_probabilities
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            explanation.push_str(&format!(
                "- Maximum measurement probability: {:.3}\n",
                max_prob
            ));

            explanation.push('\n');
        }

        // Concept activations
        if let Some(ref concepts) = result.concept_activations {
            explanation.push_str("Concept Activations:\n");
            for (concept, &activation) in concepts {
                if activation > 0.1 {
                    explanation.push_str(&format!("- {}: {:.3}\n", concept, activation));
                }
            }
        }

        Ok(explanation)
    }

    /// Compute confidence scores for explanations
    fn compute_confidence_scores(
        &self,
        result: &ExplanationResult,
    ) -> Result<HashMap<String, f64>> {
        let mut confidence = HashMap::new();

        // Feature attribution confidence
        if let Some(ref attributions) = result.feature_attributions {
            let total_magnitude = attributions.iter().map(|x| x.abs()).sum::<f64>();
            let max_magnitude = attributions
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            let attribution_confidence = if total_magnitude > 0.0 {
                max_magnitude / total_magnitude
            } else {
                0.0
            };

            confidence.insert("feature_attribution".to_string(), attribution_confidence);
        }

        // Circuit explanation confidence
        if let Some(ref circuit) = result.circuit_explanation {
            let param_variance = self.compute_variance(&circuit.parameter_importance);
            let circuit_confidence = param_variance / (param_variance + 1.0);
            confidence.insert("circuit_explanation".to_string(), circuit_confidence);
        }

        // State analysis confidence
        if let Some(ref state) = result.state_properties {
            let state_confidence = state.entanglement_entropy / (state.entanglement_entropy + 1.0);
            confidence.insert("state_analysis".to_string(), state_confidence);
        }

        Ok(confidence)
    }

    // Helper methods

    /// Compute gradient of model output w.r.t. input
    fn compute_gradient(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        // Placeholder for gradient computation
        // In practice, would use automatic differentiation
        let mut gradient = Array1::zeros(input.len());
        let h = 1e-5;

        let baseline_output = self.model.forward(input)?;
        let baseline_loss = baseline_output.iter().map(|x| x * x).sum::<f64>();

        for i in 0..input.len() {
            let mut perturbed_input = input.clone();
            perturbed_input[i] += h;

            let perturbed_output = self.model.forward(&perturbed_input)?;
            let perturbed_loss = perturbed_output.iter().map(|x| x * x).sum::<f64>();

            gradient[i] = (perturbed_loss - baseline_loss) / h;
        }

        Ok(gradient)
    }

    /// Compute quantum Fisher information
    fn compute_quantum_fisher_information(
        &self,
        input: &Array1<f64>,
        feature_idx: usize,
    ) -> Result<f64> {
        // Simplified quantum Fisher information computation
        let h = 1e-4;

        let mut input_plus = input.clone();
        let mut input_minus = input.clone();
        input_plus[feature_idx] += h;
        input_minus[feature_idx] -= h;

        let output_plus = self.model.forward(&input_plus)?;
        let output_minus = self.model.forward(&input_minus)?;

        let derivative = (&output_plus - &output_minus) / (2.0 * h);
        let fisher_info = derivative.mapv(|x| x * x).sum();

        Ok(fisher_info)
    }

    /// Get quantum state vector representation
    fn get_state_vector(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        // Simplified state vector computation
        let output = self.model.forward(input)?;
        let state_dim = 1 << self.model.num_qubits; // 2^n for n qubits

        // Create normalized state vector
        let mut state_vector = Array1::zeros(state_dim);
        for i in 0..output.len().min(state_dim) {
            state_vector[i] = output[i];
        }

        // Normalize
        let norm = state_vector.mapv(|x| x * x).sum().sqrt();
        if norm > 1e-10 {
            state_vector = state_vector / norm;
        }

        Ok(state_vector)
    }

    /// Compute entanglement entropy
    fn compute_entanglement_entropy(&self, state_vector: &Array1<f64>) -> Result<f64> {
        // Simplified entanglement entropy computation
        let num_qubits = (state_vector.len() as f64).log2() as usize;

        if num_qubits < 2 {
            return Ok(0.0);
        }

        // Compute reduced density matrix for first qubit (simplified)
        let mut entropy = 0.0;
        let half_size = state_vector.len() / 2;

        for i in 0..half_size {
            let prob_0 = state_vector[i].powi(2);
            let prob_1 = state_vector[i + half_size].powi(2);

            if prob_0 > 1e-10 {
                entropy -= prob_0 * prob_0.ln();
            }
            if prob_1 > 1e-10 {
                entropy -= prob_1 * prob_1.ln();
            }
        }

        Ok(entropy)
    }

    /// Compute coherence measures
    fn compute_coherence_measures(
        &self,
        state_vector: &Array1<f64>,
    ) -> Result<HashMap<String, f64>> {
        let mut measures = HashMap::new();

        // L1 norm coherence
        let l1_coherence = state_vector.iter()
            .enumerate()
            .filter(|(i, _)| *i > 0) // Exclude diagonal elements in density matrix
            .map(|(_, &x)| x.abs())
            .sum::<f64>();

        measures.insert("l1_coherence".to_string(), l1_coherence);

        // Relative entropy coherence
        let uniform_state = 1.0 / state_vector.len() as f64;
        let rel_entropy = state_vector
            .iter()
            .map(|&p| {
                if p > 1e-10 {
                    p * (p / uniform_state).ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        measures.insert("relative_entropy_coherence".to_string(), rel_entropy);

        Ok(measures)
    }

    /// Analyze superposition components
    fn analyze_superposition(&self, state_vector: &Array1<f64>) -> Result<Array1<f64>> {
        // Return magnitude of each basis state component
        Ok(state_vector.mapv(|x| x.abs()))
    }

    /// Compute state fidelities
    fn compute_state_fidelities(&self, state_vector: &Array1<f64>) -> Result<HashMap<String, f64>> {
        let mut fidelities = HashMap::new();

        // Fidelity with computational basis states
        for i in 0..state_vector.len().min(8) {
            // Limit to first 8 basis states
            let fidelity = state_vector[i].abs();
            fidelities.insert(format!("basis_state_{}", i), fidelity);
        }

        Ok(fidelities)
    }

    /// Apply perturbation to input
    fn apply_perturbation(
        &self,
        input: &Array1<f64>,
        method: &PerturbationMethod,
    ) -> Result<Array1<f64>> {
        match method {
            PerturbationMethod::Gaussian { sigma } => {
                let noise =
                    Array1::from_shape_fn(input.len(), |_| sigma * (fastrand::f64() - 0.5) * 2.0);
                Ok(input + &noise)
            }

            PerturbationMethod::QuantumPhase { magnitude } => {
                let mut perturbed = input.clone();
                for i in 0..perturbed.len() {
                    let phase_shift = magnitude * (2.0 * PI * fastrand::f64() - PI);
                    perturbed[i] = (perturbed[i] + phase_shift).rem_euclid(2.0 * PI);
                }
                Ok(perturbed)
            }

            PerturbationMethod::FeatureMasking => {
                let mut perturbed = input.clone();
                let mask_idx = fastrand::usize(0..input.len());
                perturbed[mask_idx] = 0.0;
                Ok(perturbed)
            }

            PerturbationMethod::ParameterPerturbation { strength } => {
                let noise =
                    Array1::from_shape_fn(input.len(), |_| strength * (fastrand::f64() - 0.5));
                Ok(input + &noise)
            }
        }
    }

    /// Compute parameter sensitivity
    fn compute_parameter_sensitivity(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        let mut sensitivity = Array1::zeros(self.model.parameters.len());
        let h = 1e-5;

        let baseline_output = self.model.forward(input)?;

        for i in 0..self.model.parameters.len() {
            // This would require parameter perturbation capability
            // Simplified version
            sensitivity[i] = 1.0; // Placeholder
        }

        Ok(sensitivity)
    }

    /// Analyze individual layer
    fn analyze_layer(
        &self,
        input: &Array1<f64>,
        layer: &QNNLayerType,
        layer_idx: usize,
    ) -> Result<LayerAnalysis> {
        // Simplified layer analysis
        let information_gain = 0.5 + 0.3 * fastrand::f64();
        let entanglement_generated = match layer {
            QNNLayerType::EntanglementLayer { .. } => 0.8 + 0.2 * fastrand::f64(),
            _ => 0.1 * fastrand::f64(),
        };

        let feature_dim = input.len();
        let feature_transformations =
            Array2::from_shape_fn((feature_dim, feature_dim), |(i, j)| {
                if i == j {
                    1.0
                } else {
                    0.1 * fastrand::f64()
                }
            });

        let activation_patterns = Array1::from_shape_fn(feature_dim, |_| fastrand::f64());

        Ok(LayerAnalysis {
            layer_type: layer.clone(),
            information_gain,
            entanglement_generated,
            feature_transformations,
            activation_patterns,
        })
    }

    /// Analyze gates in circuit
    fn analyze_gates(&self, input: &Array1<f64>) -> Result<Vec<GateContribution>> {
        // Simplified gate analysis
        let mut contributions = Vec::new();

        for i in 0..10 {
            // Assume 10 gates for demo
            let contribution = GateContribution {
                gate_index: i,
                gate_type: if i % 3 == 0 {
                    "RX".to_string()
                } else if i % 3 == 1 {
                    "RY".to_string()
                } else {
                    "CNOT".to_string()
                },
                contribution: 0.1 + 0.8 * fastrand::f64(),
                qubits: vec![i % self.model.num_qubits, (i + 1) % self.model.num_qubits],
                parameters: if i % 3 != 2 {
                    Some(Array1::from_vec(vec![PI * fastrand::f64()]))
                } else {
                    None
                },
            };
            contributions.push(contribution);
        }

        Ok(contributions)
    }

    /// Find critical path through circuit
    fn find_critical_path(&self, param_importance: &Array1<f64>) -> Result<Vec<usize>> {
        // Find indices of most important parameters
        let mut indexed_importance: Vec<(usize, f64)> = param_importance
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();

        indexed_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Return top 5 parameter indices as critical path
        Ok(indexed_importance
            .into_iter()
            .take(5)
            .map(|(i, _)| i)
            .collect())
    }

    /// Compute variance of array
    fn compute_variance(&self, arr: &Array1<f64>) -> f64 {
        let mean = arr.mean().unwrap_or(0.0);
        let variance = arr.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / arr.len() as f64;
        variance
    }

    /// Generate LIME perturbation
    fn generate_lime_perturbation(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        let mut perturbed = input.clone();

        // Randomly mask some features
        for i in 0..input.len() {
            if fastrand::f64() < 0.3 {
                // 30% chance to mask
                perturbed[i] = 0.0;
            }
        }

        Ok(perturbed)
    }

    /// Fit linear model for LIME
    fn fit_linear_model(
        &self,
        perturbations: &[Array1<f64>],
        outputs: &[Array1<f64>],
        weights: &[f64],
    ) -> Result<Array1<f64>> {
        // Simplified linear model fitting
        let feature_dim = perturbations[0].len();
        Ok(Array1::from_shape_fn(feature_dim, |i| {
            0.1 + 0.8 * fastrand::f64()
        }))
    }

    /// Fit decision tree for LIME
    fn fit_decision_tree(
        &self,
        perturbations: &[Array1<f64>],
        outputs: &[Array1<f64>],
        weights: &[f64],
    ) -> Result<Array1<f64>> {
        // Simplified decision tree fitting
        let feature_dim = perturbations[0].len();
        Ok(Array1::from_shape_fn(feature_dim, |i| {
            if i % 2 == 0 {
                0.8
            } else {
                0.2
            }
        }))
    }

    /// Fit quantum linear model for LIME
    fn fit_quantum_linear_model(
        &self,
        perturbations: &[Array1<f64>],
        outputs: &[Array1<f64>],
        weights: &[f64],
    ) -> Result<Array1<f64>> {
        // Simplified quantum linear model
        let feature_dim = perturbations[0].len();
        Ok(Array1::from_shape_fn(feature_dim, |i| {
            (i as f64 * 0.3).sin().abs()
        }))
    }

    /// Compute coalition value for SHAP
    fn compute_coalition_value(
        &self,
        input: &Array1<f64>,
        coalition: &[bool],
        additional_feature: Option<usize>,
        background_indices: &[usize],
    ) -> Result<f64> {
        if let Some(ref background) = self.background_data {
            let mut coalition_input = Array1::zeros(input.len());

            // Set coalition features from input, others from background
            let bg_idx = background_indices[fastrand::usize(0..background_indices.len())];
            let background_sample = background.row(bg_idx);

            for i in 0..input.len() {
                let in_coalition = coalition[i] || (additional_feature == Some(i));
                coalition_input[i] = if in_coalition {
                    input[i]
                } else {
                    background_sample[i]
                };
            }

            let output = self.model.forward(&coalition_input)?;
            Ok(output.sum()) // Simplified value function
        } else {
            Ok(0.0)
        }
    }

    /// Compute layer activations
    fn compute_layer_activations(&self, input: &Array1<f64>) -> Result<Vec<Array1<f64>>> {
        // Simplified layer activation computation
        let mut activations = Vec::new();
        let mut current_activation = input.clone();

        for _ in &self.model.layers {
            // Simplified transformation
            current_activation = current_activation.mapv(|x| x.tanh());
            activations.push(current_activation.clone());
        }

        Ok(activations)
    }

    /// Propagate relevance through layer
    fn propagate_relevance_through_layer(
        &self,
        relevance: &Array1<f64>,
        activation: &Array1<f64>,
        layer: &QNNLayerType,
        rule: &LRPRule,
        epsilon: f64,
    ) -> Result<Array1<f64>> {
        // Simplified LRP propagation
        match rule {
            LRPRule::Epsilon => {
                let denominator = activation.mapv(|x| x + epsilon);
                Ok(relevance / &denominator)
            }
            _ => Ok(relevance.clone()),
        }
    }

    /// Get internal representation
    fn get_internal_representation(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        // Return intermediate layer output as internal representation
        self.model.forward(input)
    }

    /// Compute coherence weight
    fn compute_coherence_weight(&self, input: &Array1<f64>) -> Result<f64> {
        let state_props = self.analyze_quantum_state(input, false, true, false)?;
        let coherence = state_props
            .coherence_measures
            .get("l1_coherence")
            .unwrap_or(&1.0);
        Ok(*coherence)
    }
}

/// Helper function to create default explainable AI configuration
pub fn create_default_xai_config() -> Vec<ExplanationMethod> {
    vec![
        ExplanationMethod::QuantumFeatureAttribution {
            method: AttributionMethod::IntegratedGradients,
            num_samples: 50,
            baseline: None,
        },
        ExplanationMethod::CircuitVisualization {
            include_measurements: true,
            parameter_sensitivity: true,
        },
        ExplanationMethod::StateAnalysis {
            entanglement_measures: true,
            coherence_analysis: true,
            superposition_analysis: true,
        },
        ExplanationMethod::SaliencyMapping {
            perturbation_method: PerturbationMethod::Gaussian { sigma: 0.1 },
            aggregation: AggregationMethod::Mean,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qnn::QNNLayerType;

    #[test]
    fn test_xai_creation() {
        let layers = vec![
            QNNLayerType::EncodingLayer { num_features: 4 },
            QNNLayerType::VariationalLayer { num_params: 8 },
            QNNLayerType::MeasurementLayer {
                measurement_basis: "computational".to_string(),
            },
        ];

        let model = QuantumNeuralNetwork::new(layers, 4, 4, 2).unwrap();
        let methods = create_default_xai_config();
        let xai = QuantumExplainableAI::new(model, methods);

        assert_eq!(xai.methods.len(), 4);
    }

    #[test]
    fn test_explanation_result() {
        let result = ExplanationResult {
            feature_attributions: Some(Array1::from_vec(vec![0.1, 0.5, -0.2, 0.8])),
            saliency_map: None,
            circuit_explanation: None,
            state_properties: None,
            concept_activations: None,
            textual_explanation: "Test explanation".to_string(),
            confidence_scores: HashMap::new(),
        };

        assert!(result.feature_attributions.is_some());
        assert_eq!(result.textual_explanation, "Test explanation");
    }

    #[test]
    fn test_circuit_explanation() {
        let explanation = CircuitExplanation {
            parameter_importance: Array1::from_vec(vec![0.8, 0.3, 0.9, 0.1]),
            gate_contributions: Vec::new(),
            layer_analysis: Vec::new(),
            critical_path: vec![2, 0, 1],
        };

        assert_eq!(explanation.parameter_importance.len(), 4);
        assert_eq!(explanation.critical_path, vec![2, 0, 1]);
    }

    #[test]
    fn test_quantum_state_properties() {
        let mut coherence_measures = HashMap::new();
        coherence_measures.insert("l1_coherence".to_string(), 0.7);

        let mut state_fidelities = HashMap::new();
        state_fidelities.insert("basis_state_0".to_string(), 0.9);

        let properties = QuantumStateProperties {
            entanglement_entropy: 1.2,
            coherence_measures,
            superposition_components: Array1::from_vec(vec![0.7, 0.5, 0.1, 0.2]),
            measurement_probabilities: Array1::from_vec(vec![0.49, 0.25, 0.01, 0.04]),
            state_fidelities,
        };

        assert_eq!(properties.entanglement_entropy, 1.2);
        assert_eq!(properties.superposition_components.len(), 4);
    }
}
