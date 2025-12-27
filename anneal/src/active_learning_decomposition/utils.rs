//! Utility functions for active learning decomposition

use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ising::IsingModel;

    #[test]
    fn test_active_learning_decomposer_creation() {
        let config = ActiveLearningConfig::default();
        let decomposer = ActiveLearningDecomposer::new(config);
        assert!(decomposer.is_ok());
    }

    #[test]
    fn test_problem_analysis() {
        let config = ActiveLearningConfig::default();
        let mut decomposer = ActiveLearningDecomposer::new(config)
            .expect("Failed to create ActiveLearningDecomposer");

        let mut problem = IsingModel::new(4);
        problem
            .set_bias(0, 1.0)
            .expect("Failed to set bias for qubit 0");
        problem
            .set_coupling(0, 1, -0.5)
            .expect("Failed to set coupling between qubits 0 and 1");

        let analysis = decomposer.analyze_problem(&problem);
        assert!(analysis.is_ok());

        let analysis = analysis.expect("Failed to analyze problem");
        assert_eq!(analysis.graph_metrics.num_vertices, 4);
        assert_eq!(analysis.problem_features.len(), 20);
    }

    #[test]
    fn test_feature_extraction() {
        let config = ActiveLearningConfig::default();
        let decomposer = ActiveLearningDecomposer::new(config)
            .expect("Failed to create ActiveLearningDecomposer");

        let mut problem = IsingModel::new(3);
        problem
            .set_bias(0, 1.0)
            .expect("Failed to set bias for qubit 0");
        problem
            .set_coupling(0, 1, -0.5)
            .expect("Failed to set coupling between qubits 0 and 1");
        problem
            .set_coupling(1, 2, 0.3)
            .expect("Failed to set coupling between qubits 1 and 2");

        let features = decomposer
            .extract_problem_features(&problem)
            .expect("Failed to extract problem features");
        assert_eq!(features.len(), 20);
        assert_eq!(features[0], 3.0); // num_qubits
        assert_eq!(features[1], 2.0); // num_couplings
    }

    #[test]
    fn test_problem_decomposition() {
        let config = ActiveLearningConfig::default();
        let mut decomposer = ActiveLearningDecomposer::new(config)
            .expect("Failed to create ActiveLearningDecomposer");

        let mut problem = IsingModel::new(6);
        problem
            .set_bias(0, 1.0)
            .expect("Failed to set bias for qubit 0");
        problem
            .set_coupling(0, 1, -0.5)
            .expect("Failed to set coupling between qubits 0 and 1");
        problem
            .set_coupling(2, 3, 0.3)
            .expect("Failed to set coupling between qubits 2 and 3");

        let result = decomposer.decompose_problem(&problem);
        assert!(result.is_ok());

        let decomposition = result.expect("Failed to decompose problem");
        assert!(!decomposition.subproblems.is_empty());
        assert!(decomposition.quality_score >= 0.0);
        assert!(decomposition.quality_score <= 1.0);
    }

    #[test]
    fn test_subproblem_creation() {
        let config = ActiveLearningConfig::default();
        let decomposer = ActiveLearningDecomposer::new(config)
            .expect("Failed to create ActiveLearningDecomposer");

        let mut problem = IsingModel::new(4);
        problem
            .set_bias(0, 1.0)
            .expect("Failed to set bias for qubit 0");
        problem
            .set_coupling(0, 1, -0.5)
            .expect("Failed to set coupling between qubits 0 and 1");
        problem
            .set_coupling(1, 2, 0.3)
            .expect("Failed to set coupling between qubits 1 and 2");

        let vertices = vec![0, 1];
        let subproblem = decomposer
            .create_subproblem(&problem, &vertices, 0)
            .expect("Failed to create subproblem");

        assert_eq!(subproblem.id, 0);
        assert_eq!(subproblem.vertices, vec![0, 1]);
        assert_eq!(subproblem.model.num_qubits, 2);
        assert!(!subproblem.boundary_edges.is_empty()); // Should have boundary to vertex 2
    }

    #[test]
    fn test_decomposition_quality_validation() {
        let config = ActiveLearningConfig::default();
        let decomposer = ActiveLearningDecomposer::new(config)
            .expect("Failed to create ActiveLearningDecomposer");

        let problem = IsingModel::new(6);

        // Create test subproblems
        let subproblem1 = Subproblem {
            id: 0,
            model: IsingModel::new(3),
            vertices: vec![0, 1, 2],
            boundary_edges: Vec::new(),
            metadata: SubproblemMetadata::new(),
        };

        let subproblem2 = Subproblem {
            id: 1,
            model: IsingModel::new(3),
            vertices: vec![3, 4, 5],
            boundary_edges: Vec::new(),
            metadata: SubproblemMetadata::new(),
        };

        let subproblems = vec![subproblem1, subproblem2];
        let quality = decomposer
            .validate_decomposition_quality(&subproblems, &problem)
            .expect("Failed to validate decomposition quality");

        assert!(quality >= 0.0);
        assert!(quality <= 1.0);
    }

    #[test]
    fn test_strategy_selection() {
        let mut learner = DecompositionStrategyLearner::new()
            .expect("Failed to create DecompositionStrategyLearner");
        let problem = IsingModel::new(10);
        let analysis = ProblemAnalysis {
            graph_metrics: GraphMetrics {
                num_vertices: 10,
                num_edges: 15,
                density: 0.3,
                clustering_coefficient: 0.5,
                avg_path_length: 2.5,
                modularity: 0.4,
                spectral_gap: 0.2,
                treewidth_estimate: 3,
            },
            communities: Vec::new(),
            structures: Vec::new(),
            complexity: ComplexityEstimate {
                complexity_class: ComplexityClass::NP,
                numeric_estimate: 100.0,
                confidence_interval: (50.0, 200.0),
                estimation_method: "test".to_string(),
            },
            decomposability: DecomposabilityScore {
                overall_score: 0.8,
                component_scores: std::collections::HashMap::new(),
                recommendation: DecompositionRecommendation {
                    strategy: DecompositionStrategy::GraphPartitioning,
                    cut_points: Vec::new(),
                    expected_benefit: 0.8,
                    risk_assessment: RiskAssessment {
                        risk_level: RiskLevel::Low,
                        risk_factors: Vec::new(),
                        mitigation_strategies: Vec::new(),
                    },
                },
                confidence: 0.9,
            },
            problem_features: scirs2_core::ndarray::Array1::ones(20),
        };

        let strategy = learner
            .recommend_strategy(&problem, &analysis)
            .expect("Failed to recommend strategy");
        // For a 10-qubit problem, should recommend GraphPartitioning
        assert_eq!(strategy, DecompositionStrategy::GraphPartitioning);
    }
}
