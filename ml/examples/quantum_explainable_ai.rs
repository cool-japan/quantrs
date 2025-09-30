//! Quantum Explainable AI Example
//!
//! This example demonstrates various explainability and interpretability methods
//! for quantum neural networks, including feature attribution, circuit analysis,
//! quantum state analysis, and concept activation vectors.

use scirs2_core::ndarray::{Array1, Array2};
use quantrs2_ml::prelude::*;
use quantrs2_ml::qnn::QNNLayerType;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("=== Quantum Explainable AI Demo ===\n");

    // Step 1: Feature attribution methods
    println!("1. Quantum Feature Attribution...");
    feature_attribution_demo()?;

    // Step 2: Circuit visualization and analysis
    println!("\n2. Circuit Visualization and Analysis...");
    circuit_analysis_demo()?;

    // Step 3: Quantum state analysis
    println!("\n3. Quantum State Analysis...");
    quantum_state_demo()?;

    // Step 4: Saliency mapping
    println!("\n4. Quantum Saliency Mapping...");
    saliency_mapping_demo()?;

    // Step 5: Quantum LIME explanations
    println!("\n5. Quantum LIME (Local Interpretable Model-agnostic Explanations)...");
    quantum_lime_demo()?;

    // Step 6: Quantum SHAP values
    println!("\n6. Quantum SHAP (SHapley Additive exPlanations)...");
    quantum_shap_demo()?;

    // Step 7: Layer-wise relevance propagation
    println!("\n7. Layer-wise Relevance Propagation...");
    quantum_lrp_demo()?;

    // Step 8: Comprehensive explanation
    println!("\n8. Comprehensive Explanation Analysis...");
    comprehensive_explanation_demo()?;

    println!("\n=== Quantum Explainable AI Demo Complete ===");

    Ok(())
}

/// Demonstrate quantum feature attribution methods
fn feature_attribution_demo() -> Result<()> {
    // Create quantum model
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 12 },
        QNNLayerType::EntanglementLayer {
            connectivity: "circular".to_string(),
        },
        QNNLayerType::VariationalLayer { num_params: 8 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;

    println!(
        "   Created quantum model with {} parameters",
        model.parameters.len()
    );

    // Test different attribution methods
    let attribution_methods = vec![
        (
            "Integrated Gradients",
            ExplanationMethod::QuantumFeatureAttribution {
                method: AttributionMethod::IntegratedGradients,
                num_samples: 50,
                baseline: Some(Array1::zeros(4)),
            },
        ),
        (
            "Gradient × Input",
            ExplanationMethod::QuantumFeatureAttribution {
                method: AttributionMethod::GradientInput,
                num_samples: 1,
                baseline: None,
            },
        ),
        (
            "Gradient SHAP",
            ExplanationMethod::QuantumFeatureAttribution {
                method: AttributionMethod::GradientSHAP,
                num_samples: 30,
                baseline: None,
            },
        ),
        (
            "Quantum Attribution",
            ExplanationMethod::QuantumFeatureAttribution {
                method: AttributionMethod::QuantumAttribution,
                num_samples: 25,
                baseline: None,
            },
        ),
    ];

    // Test input
    let test_input = Array1::from_vec(vec![0.8, 0.3, 0.9, 0.1]);

    println!(
        "\n   Feature attribution analysis for input: [{:.1}, {:.1}, {:.1}, {:.1}]",
        test_input[0], test_input[1], test_input[2], test_input[3]
    );

    for (method_name, method) in attribution_methods {
        let mut xai = QuantumExplainableAI::new(model.clone(), vec![method]);

        // Set background data for gradient SHAP
        let background_data =
            Array2::from_shape_fn((20, 4), |(_, j)| 0.5 + 0.3 * (j as f64 * 0.2).sin());
        xai.set_background_data(background_data);

        let explanation = xai.explain(&test_input)?;

        if let Some(ref attributions) = explanation.feature_attributions {
            println!("\n   {} Attribution:", method_name);
            for (i, &attr) in attributions.iter().enumerate() {
                println!(
                    "     Feature {}: {:+.4} {}",
                    i,
                    attr,
                    if attr.abs() > 0.1 {
                        if attr > 0.0 {
                            "(strong positive)"
                        } else {
                            "(strong negative)"
                        }
                    } else {
                        "(weak influence)"
                    }
                );
            }

            // Find most important feature
            let max_idx = attributions
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            println!(
                "     → Most important feature: Feature {} ({:.4})",
                max_idx, attributions[max_idx]
            );
        }
    }

    Ok(())
}

/// Demonstrate circuit analysis and visualization
fn circuit_analysis_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 6 },
        QNNLayerType::EntanglementLayer {
            connectivity: "full".to_string(),
        },
        QNNLayerType::VariationalLayer { num_params: 6 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "Pauli-Z".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;

    let method = ExplanationMethod::CircuitVisualization {
        include_measurements: true,
        parameter_sensitivity: true,
    };

    let mut xai = QuantumExplainableAI::new(model, vec![method]);

    println!("   Analyzing quantum circuit structure and parameter importance...");

    let test_input = Array1::from_vec(vec![0.6, 0.4, 0.7, 0.3]);
    let explanation = xai.explain(&test_input)?;

    if let Some(ref circuit) = explanation.circuit_explanation {
        println!("\n   Circuit Analysis Results:");

        // Parameter importance
        println!("   Parameter Importance Scores:");
        for (i, &importance) in circuit.parameter_importance.iter().enumerate() {
            if importance > 0.5 {
                println!("     Parameter {}: {:.3} (high importance)", i, importance);
            } else if importance > 0.2 {
                println!(
                    "     Parameter {}: {:.3} (medium importance)",
                    i, importance
                );
            }
        }

        // Layer analysis
        println!("\n   Layer-wise Analysis:");
        for (i, layer_analysis) in circuit.layer_analysis.iter().enumerate() {
            println!(
                "     Layer {}: {}",
                i,
                format_layer_type(&layer_analysis.layer_type)
            );
            println!(
                "       Information gain: {:.3}",
                layer_analysis.information_gain
            );
            println!(
                "       Entanglement generated: {:.3}",
                layer_analysis.entanglement_generated
            );

            if layer_analysis.entanglement_generated > 0.5 {
                println!("       → Significant entanglement layer");
            }
        }

        // Gate contributions
        println!("\n   Gate Contribution Analysis:");
        for (i, gate) in circuit.gate_contributions.iter().enumerate().take(5) {
            println!(
                "     Gate {}: {} on qubits {:?}",
                gate.gate_index, gate.gate_type, gate.qubits
            );
            println!("       Contribution: {:.3}", gate.contribution);

            if let Some(ref params) = gate.parameters {
                println!("       Parameters: {:.3}", params[0]);
            }
        }

        // Critical path
        println!("\n   Critical Path (most important parameters):");
        print!("     ");
        for (i, &param_idx) in circuit.critical_path.iter().enumerate() {
            if i > 0 {
                print!(" → ");
            }
            print!("P{}", param_idx);
        }
        println!();

        println!("   → This path represents the most influential quantum operations");
    }

    Ok(())
}

/// Demonstrate quantum state analysis
fn quantum_state_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 3 },
        QNNLayerType::VariationalLayer { num_params: 9 },
        QNNLayerType::EntanglementLayer {
            connectivity: "circular".to_string(),
        },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 3, 3, 2)?;

    let method = ExplanationMethod::StateAnalysis {
        entanglement_measures: true,
        coherence_analysis: true,
        superposition_analysis: true,
    };

    let mut xai = QuantumExplainableAI::new(model, vec![method]);

    println!("   Analyzing quantum state properties...");

    // Test different inputs to see state evolution
    let test_inputs = vec![
        Array1::from_vec(vec![0.0, 0.0, 0.0]),
        Array1::from_vec(vec![1.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.5, 0.5, 0.5]),
        Array1::from_vec(vec![1.0, 1.0, 1.0]),
    ];

    for (i, input) in test_inputs.iter().enumerate() {
        println!(
            "\n   Input {}: [{:.1}, {:.1}, {:.1}]",
            i + 1,
            input[0],
            input[1],
            input[2]
        );

        let explanation = xai.explain(input)?;

        if let Some(ref state) = explanation.state_properties {
            println!("     Quantum State Properties:");
            println!(
                "     - Entanglement entropy: {:.3}",
                state.entanglement_entropy
            );

            // Coherence measures
            for (measure_name, &value) in &state.coherence_measures {
                println!("     - {}: {:.3}", measure_name, value);
            }

            // Superposition analysis
            let max_component = state
                .superposition_components
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            println!("     - Max superposition component: {:.3}", max_component);

            // Measurement probabilities
            let total_prob = state.measurement_probabilities.sum();
            println!("     - Total measurement probability: {:.3}", total_prob);

            // Most likely measurement outcome
            let most_likely = state
                .measurement_probabilities
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, &prob)| (idx, prob))
                .unwrap_or((0, 0.0));

            println!(
                "     - Most likely outcome: state {} with prob {:.3}",
                most_likely.0, most_likely.1
            );

            // State fidelities
            if let Some(highest_fidelity) = state
                .state_fidelities
                .values()
                .cloned()
                .fold(None, |acc, x| Some(acc.map_or(x, |y| f64::max(x, y))))
            {
                println!(
                    "     - Highest basis state fidelity: {:.3}",
                    highest_fidelity
                );
            }

            // Interpretation
            if state.entanglement_entropy > 0.5 {
                println!("     → Highly entangled state");
            } else if state.entanglement_entropy > 0.1 {
                println!("     → Moderately entangled state");
            } else {
                println!("     → Separable or weakly entangled state");
            }
        }
    }

    Ok(())
}

/// Demonstrate saliency mapping
fn saliency_mapping_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 8 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;

    // Test different perturbation methods
    let perturbation_methods = vec![
        (
            "Gaussian Noise",
            PerturbationMethod::Gaussian { sigma: 0.1 },
        ),
        (
            "Quantum Phase",
            PerturbationMethod::QuantumPhase { magnitude: 0.2 },
        ),
        ("Feature Masking", PerturbationMethod::FeatureMasking),
        (
            "Parameter Perturbation",
            PerturbationMethod::ParameterPerturbation { strength: 0.1 },
        ),
    ];

    let test_input = Array1::from_vec(vec![0.7, 0.2, 0.8, 0.4]);

    println!("   Computing saliency maps with different perturbation methods...");
    println!(
        "   Input: [{:.1}, {:.1}, {:.1}, {:.1}]",
        test_input[0], test_input[1], test_input[2], test_input[3]
    );

    for (method_name, perturbation_method) in perturbation_methods {
        let method = ExplanationMethod::SaliencyMapping {
            perturbation_method,
            aggregation: AggregationMethod::Mean,
        };

        let mut xai = QuantumExplainableAI::new(model.clone(), vec![method]);
        let explanation = xai.explain(&test_input)?;

        if let Some(ref saliency) = explanation.saliency_map {
            println!("\n   {} Saliency Map:", method_name);

            // Analyze saliency for each output
            for output_idx in 0..saliency.ncols() {
                println!("     Output {}:", output_idx);
                for input_idx in 0..saliency.nrows() {
                    let saliency_score = saliency[[input_idx, output_idx]];
                    if saliency_score > 0.1 {
                        println!(
                            "       Feature {} → Output {}: {:.3} (important)",
                            input_idx, output_idx, saliency_score
                        );
                    } else if saliency_score > 0.05 {
                        println!(
                            "       Feature {} → Output {}: {:.3} (moderate)",
                            input_idx, output_idx, saliency_score
                        );
                    }
                }
            }

            // Find most salient feature-output pair
            let mut max_saliency = 0.0;
            let mut max_pair = (0, 0);

            for i in 0..saliency.nrows() {
                for j in 0..saliency.ncols() {
                    if saliency[[i, j]] > max_saliency {
                        max_saliency = saliency[[i, j]];
                        max_pair = (i, j);
                    }
                }
            }

            println!(
                "     → Most salient: Feature {} → Output {} ({:.3})",
                max_pair.0, max_pair.1, max_saliency
            );
        }
    }

    Ok(())
}

/// Demonstrate Quantum LIME
fn quantum_lime_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 10 },
        QNNLayerType::EntanglementLayer {
            connectivity: "circular".to_string(),
        },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;

    // Test different local models
    let local_models = vec![
        ("Linear Regression", LocalModelType::LinearRegression),
        ("Decision Tree", LocalModelType::DecisionTree),
        ("Quantum Linear", LocalModelType::QuantumLinear),
    ];

    let test_input = Array1::from_vec(vec![0.6, 0.8, 0.2, 0.9]);

    println!("   Quantum LIME: Local Interpretable Model-agnostic Explanations");
    println!(
        "   Input: [{:.1}, {:.1}, {:.1}, {:.1}]",
        test_input[0], test_input[1], test_input[2], test_input[3]
    );

    for (model_name, local_model) in local_models {
        let method = ExplanationMethod::QuantumLIME {
            num_perturbations: 100,
            kernel_width: 0.5,
            local_model,
        };

        let mut xai = QuantumExplainableAI::new(model.clone(), vec![method]);
        let explanation = xai.explain(&test_input)?;

        if let Some(ref attributions) = explanation.feature_attributions {
            println!("\n   LIME with {}:", model_name);

            for (i, &attr) in attributions.iter().enumerate() {
                let impact = if attr.abs() > 0.3 {
                    "high"
                } else if attr.abs() > 0.1 {
                    "medium"
                } else {
                    "low"
                };

                println!("     Feature {}: {:+.3} ({} impact)", i, attr, impact);
            }

            // Local model interpretation
            match model_name {
                "Linear Regression" => {
                    println!("     → Linear relationship approximation in local region");
                }
                "Decision Tree" => {
                    println!("     → Rule-based approximation with thresholds");
                }
                "Quantum Linear" => {
                    println!("     → Quantum-aware linear approximation");
                }
                _ => {}
            }

            // Compute local fidelity (simplified)
            let local_complexity = attributions.iter().map(|x| x.abs()).sum::<f64>();
            println!(
                "     → Local explanation complexity: {:.3}",
                local_complexity
            );
        }
    }

    Ok(())
}

/// Demonstrate Quantum SHAP
fn quantum_shap_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 3 },
        QNNLayerType::VariationalLayer { num_params: 6 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "Pauli-Z".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 3, 3, 2)?;

    let method = ExplanationMethod::QuantumSHAP {
        num_coalitions: 100,
        background_samples: 20,
    };

    let mut xai = QuantumExplainableAI::new(model, vec![method]);

    // Set background data for SHAP
    let background_data =
        Array2::from_shape_fn((50, 3), |(i, j)| 0.5 + 0.3 * ((i + j) as f64 * 0.1).sin());
    xai.set_background_data(background_data);

    println!("   Quantum SHAP: SHapley Additive exPlanations");

    // Test multiple inputs
    let test_inputs = vec![
        Array1::from_vec(vec![0.1, 0.5, 0.9]),
        Array1::from_vec(vec![0.8, 0.3, 0.6]),
        Array1::from_vec(vec![0.4, 0.7, 0.2]),
    ];

    for (i, input) in test_inputs.iter().enumerate() {
        println!(
            "\n   Input {}: [{:.1}, {:.1}, {:.1}]",
            i + 1,
            input[0],
            input[1],
            input[2]
        );

        let explanation = xai.explain(input)?;

        if let Some(ref shap_values) = explanation.feature_attributions {
            println!("     SHAP Values:");

            let mut total_shap = 0.0;
            for (j, &value) in shap_values.iter().enumerate() {
                total_shap += value;
                println!("     - Feature {}: {:+.4}", j, value);
            }

            println!("     - Sum of SHAP values: {:.4}", total_shap);

            // Feature ranking
            let mut indexed_shap: Vec<(usize, f64)> = shap_values
                .iter()
                .enumerate()
                .map(|(idx, &val)| (idx, val.abs()))
                .collect();
            indexed_shap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("     Feature importance ranking:");
            for (rank, (feature_idx, abs_value)) in indexed_shap.iter().enumerate() {
                let original_value = shap_values[*feature_idx];
                println!(
                    "     {}. Feature {}: {:.4} (|{:.4}|)",
                    rank + 1,
                    feature_idx,
                    original_value,
                    abs_value
                );
            }

            // SHAP properties
            println!(
                "     → SHAP values satisfy efficiency property (sum to prediction difference)"
            );
            println!("     → Each value represents feature's average marginal contribution");
        }
    }

    Ok(())
}

/// Demonstrate Layer-wise Relevance Propagation
fn quantum_lrp_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 8 },
        QNNLayerType::VariationalLayer { num_params: 6 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;

    // Test different LRP rules
    let lrp_rules = vec![
        ("Epsilon Rule", LRPRule::Epsilon),
        ("Gamma Rule", LRPRule::Gamma { gamma: 0.25 }),
        (
            "Alpha-Beta Rule",
            LRPRule::AlphaBeta {
                alpha: 2.0,
                beta: 1.0,
            },
        ),
        ("Quantum Rule", LRPRule::QuantumRule),
    ];

    let test_input = Array1::from_vec(vec![0.7, 0.1, 0.8, 0.4]);

    println!("   Layer-wise Relevance Propagation for Quantum Circuits");
    println!(
        "   Input: [{:.1}, {:.1}, {:.1}, {:.1}]",
        test_input[0], test_input[1], test_input[2], test_input[3]
    );

    for (rule_name, lrp_rule) in lrp_rules {
        let method = ExplanationMethod::QuantumLRP {
            propagation_rule: lrp_rule,
            epsilon: 1e-6,
        };

        let mut xai = QuantumExplainableAI::new(model.clone(), vec![method]);
        let explanation = xai.explain(&test_input)?;

        if let Some(ref relevance) = explanation.feature_attributions {
            println!("\n   LRP with {}:", rule_name);

            let total_relevance = relevance.sum();

            for (i, &rel) in relevance.iter().enumerate() {
                let percentage = if total_relevance.abs() > 1e-10 {
                    rel / total_relevance * 100.0
                } else {
                    0.0
                };

                println!(
                    "     Feature {}: {:.4} ({:.1}% of total relevance)",
                    i, rel, percentage
                );
            }

            println!("     Total relevance: {:.4}", total_relevance);

            // Rule-specific interpretation
            match rule_name {
                "Epsilon Rule" => {
                    println!("     → Distributes relevance proportionally to activations");
                }
                "Gamma Rule" => {
                    println!("     → Emphasizes positive contributions");
                }
                "Alpha-Beta Rule" => {
                    println!("     → Separates positive and negative contributions");
                }
                "Quantum Rule" => {
                    println!("     → Accounts for quantum superposition and entanglement");
                }
                _ => {}
            }
        }
    }

    Ok(())
}

/// Comprehensive explanation demonstration
fn comprehensive_explanation_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 12 },
        QNNLayerType::EntanglementLayer {
            connectivity: "full".to_string(),
        },
        QNNLayerType::VariationalLayer { num_params: 8 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 3)?;

    // Use comprehensive explanation methods
    let methods = vec![
        ExplanationMethod::QuantumFeatureAttribution {
            method: AttributionMethod::IntegratedGradients,
            num_samples: 30,
            baseline: Some(Array1::zeros(4)),
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
        ExplanationMethod::ConceptActivation {
            concept_datasets: vec!["pattern_A".to_string(), "pattern_B".to_string()],
            activation_threshold: 0.3,
        },
    ];

    let mut xai = QuantumExplainableAI::new(model, methods);

    // Add concept vectors
    xai.add_concept(
        "pattern_A".to_string(),
        Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]),
    );
    xai.add_concept(
        "pattern_B".to_string(),
        Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]),
    );

    // Set background data
    let background_data =
        Array2::from_shape_fn((30, 4), |(i, j)| 0.3 + 0.4 * ((i * j) as f64 * 0.15).sin());
    xai.set_background_data(background_data);

    println!("   Comprehensive Quantum Model Explanation");

    // Test input representing a specific pattern
    let test_input = Array1::from_vec(vec![0.9, 0.1, 0.8, 0.2]); // Similar to pattern_A

    println!(
        "\n   Analyzing input: [{:.1}, {:.1}, {:.1}, {:.1}]",
        test_input[0], test_input[1], test_input[2], test_input[3]
    );

    let explanation = xai.explain(&test_input)?;

    // Display comprehensive results
    println!("\n   === COMPREHENSIVE EXPLANATION RESULTS ===");

    // Feature attributions
    if let Some(ref attributions) = explanation.feature_attributions {
        println!("\n   Feature Attributions:");
        for (i, &attr) in attributions.iter().enumerate() {
            println!("   - Feature {}: {:+.3}", i, attr);
        }
    }

    // Circuit analysis summary
    if let Some(ref circuit) = explanation.circuit_explanation {
        println!("\n   Circuit Analysis Summary:");
        let avg_importance = circuit.parameter_importance.mean().unwrap_or(0.0);
        println!("   - Average parameter importance: {:.3}", avg_importance);
        println!(
            "   - Number of analyzed layers: {}",
            circuit.layer_analysis.len()
        );
        println!("   - Critical path length: {}", circuit.critical_path.len());
    }

    // Quantum state properties
    if let Some(ref state) = explanation.state_properties {
        println!("\n   Quantum State Properties:");
        println!(
            "   - Entanglement entropy: {:.3}",
            state.entanglement_entropy
        );
        println!(
            "   - Coherence measures: {} types",
            state.coherence_measures.len()
        );

        let max_measurement_prob = state
            .measurement_probabilities
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "   - Max measurement probability: {:.3}",
            max_measurement_prob
        );
    }

    // Concept activations
    if let Some(ref concepts) = explanation.concept_activations {
        println!("\n   Concept Activations:");
        for (concept, &activation) in concepts {
            let similarity = if activation > 0.7 {
                "high"
            } else if activation > 0.3 {
                "medium"
            } else {
                "low"
            };
            println!(
                "   - {}: {:.3} ({} similarity)",
                concept, activation, similarity
            );
        }
    }

    // Confidence scores
    println!("\n   Explanation Confidence Scores:");
    for (component, &confidence) in &explanation.confidence_scores {
        println!("   - {}: {:.3}", component, confidence);
    }

    // Textual explanation
    println!("\n   Generated Explanation:");
    println!("{}", explanation.textual_explanation);

    // Summary insights
    println!("\n   === KEY INSIGHTS ===");

    if let Some(ref attributions) = explanation.feature_attributions {
        let max_attr_idx = attributions
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        println!(
            "   • Most influential feature: Feature {} ({:.3})",
            max_attr_idx, attributions[max_attr_idx]
        );
    }

    if let Some(ref state) = explanation.state_properties {
        if state.entanglement_entropy > 0.5 {
            println!("   • Model creates significant quantum entanglement");
        }

        let coherence_level = state
            .coherence_measures
            .values()
            .cloned()
            .fold(0.0, f64::max);
        if coherence_level > 0.5 {
            println!("   • High quantum coherence detected");
        }
    }

    if let Some(ref concepts) = explanation.concept_activations {
        if let Some((best_concept, &max_activation)) =
            concepts.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        {
            if max_activation > 0.5 {
                println!("   • Input strongly matches concept: {}", best_concept);
            }
        }
    }

    println!("   • Explanation provides multi-faceted interpretation of quantum model behavior");

    Ok(())
}

/// Helper function to format layer type for display
fn format_layer_type(layer_type: &QNNLayerType) -> String {
    match layer_type {
        QNNLayerType::EncodingLayer { num_features } => {
            format!("Encoding Layer ({} features)", num_features)
        }
        QNNLayerType::VariationalLayer { num_params } => {
            format!("Variational Layer ({} parameters)", num_params)
        }
        QNNLayerType::EntanglementLayer { connectivity } => {
            format!("Entanglement Layer ({})", connectivity)
        }
        QNNLayerType::MeasurementLayer { measurement_basis } => {
            format!("Measurement Layer ({})", measurement_basis)
        }
    }
}
