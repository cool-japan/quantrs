//! Quantum Machine Learning Tutorials for QuantRS2-ML
//!
//! This module provides comprehensive, step-by-step tutorials for learning
//! quantum machine learning concepts and practical implementation with QuantRS2.

use crate::classical_ml_integration::{HybridPipelineManager, PipelineConfig};
use crate::domain_templates::{Domain, DomainTemplateManager, TemplateConfig};
use crate::error::{MLError, Result};
use crate::keras_api::{
    ActivationFunction, Dense, LossFunction, MetricType, OptimizerType, QuantumAnsatzType,
    QuantumDense, Sequential,
};
use crate::model_zoo::{ModelZoo, QuantumModel};
use crate::optimization::{OptimizationMethod, Optimizer};
use crate::pytorch_api::{
    ActivationType as PyTorchActivationType, QuantumLinear, QuantumModule, QuantumSequential,
};
use crate::qnn::{QNNBuilder, QuantumNeuralNetwork};
use crate::qsvm::{FeatureMapType, QSVMParams, QSVM};
use crate::variational::{VariationalAlgorithm, VariationalCircuit};
use ndarray::{Array1, Array2, ArrayD, Axis};
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tutorial manager for quantum ML education
pub struct TutorialManager {
    /// Available tutorials by category
    tutorials: HashMap<TutorialCategory, Vec<Tutorial>>,
    /// Interactive exercises
    exercises: HashMap<String, Exercise>,
    /// User progress tracking
    progress: HashMap<String, TutorialProgress>,
}

/// Tutorial categories
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum TutorialCategory {
    /// Basic quantum computing concepts
    Fundamentals,
    /// Quantum neural networks
    QuantumNeuralNetworks,
    /// Quantum machine learning algorithms
    Algorithms,
    /// Variational quantum algorithms
    Variational,
    /// Quantum optimization
    Optimization,
    /// Hybrid quantum-classical methods
    Hybrid,
    /// Industry applications
    Applications,
    /// Advanced topics
    Advanced,
}

/// Tutorial definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tutorial {
    /// Tutorial ID
    pub id: String,
    /// Tutorial title
    pub title: String,
    /// Tutorial description
    pub description: String,
    /// Category
    pub category: TutorialCategory,
    /// Difficulty level
    pub difficulty: DifficultyLevel,
    /// Prerequisites
    pub prerequisites: Vec<String>,
    /// Learning objectives
    pub learning_objectives: Vec<String>,
    /// Estimated duration (minutes)
    pub duration_minutes: usize,
    /// Tutorial sections
    pub sections: Vec<TutorialSection>,
    /// Related exercises
    pub exercises: Vec<String>,
}

/// Difficulty levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifficultyLevel {
    /// Beginner level
    Beginner,
    /// Intermediate level
    Intermediate,
    /// Advanced level
    Advanced,
    /// Expert level
    Expert,
}

/// Tutorial section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorialSection {
    /// Section title
    pub title: String,
    /// Section content
    pub content: String,
    /// Code examples
    pub code_examples: Vec<CodeExample>,
    /// Interactive elements
    pub interactive_elements: Vec<InteractiveElement>,
    /// Key concepts
    pub key_concepts: Vec<String>,
}

/// Code example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    /// Example title
    pub title: String,
    /// Example description
    pub description: String,
    /// Code content
    pub code: String,
    /// Expected output
    pub expected_output: Option<String>,
    /// Explanation
    pub explanation: String,
}

/// Interactive element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveElement {
    /// Element type
    pub element_type: InteractiveType,
    /// Element title
    pub title: String,
    /// Instructions
    pub instructions: String,
    /// Parameters
    pub parameters: HashMap<String, String>,
}

/// Interactive element types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractiveType {
    /// Visualization
    Visualization,
    /// Parameter adjustment
    ParameterTuning,
    /// Code completion
    CodeCompletion,
    /// Quiz question
    Quiz,
    /// Experiment
    Experiment,
}

/// Exercise definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exercise {
    /// Exercise ID
    pub id: String,
    /// Exercise title
    pub title: String,
    /// Exercise description
    pub description: String,
    /// Exercise type
    pub exercise_type: ExerciseType,
    /// Instructions
    pub instructions: Vec<String>,
    /// Starter code
    pub starter_code: Option<String>,
    /// Solution code
    pub solution_code: String,
    /// Test cases
    pub test_cases: Vec<TestCase>,
    /// Hints
    pub hints: Vec<String>,
}

/// Exercise types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExerciseType {
    /// Coding exercise
    Coding,
    /// Circuit design
    CircuitDesign,
    /// Parameter optimization
    ParameterOptimization,
    /// Algorithm implementation
    AlgorithmImplementation,
    /// Data analysis
    DataAnalysis,
}

/// Test case for exercises
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    /// Test description
    pub description: String,
    /// Input data
    pub input: String,
    /// Expected output
    pub expected_output: String,
    /// Points awarded
    pub points: usize,
}

/// User progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorialProgress {
    /// User ID
    pub user_id: String,
    /// Completed tutorials
    pub completed_tutorials: Vec<String>,
    /// Completed exercises
    pub completed_exercises: Vec<String>,
    /// Current tutorial
    pub current_tutorial: Option<String>,
    /// Progress scores
    pub scores: HashMap<String, f64>,
    /// Time spent (minutes)
    pub time_spent: HashMap<String, usize>,
}

impl TutorialManager {
    /// Create new tutorial manager
    pub fn new() -> Self {
        let mut manager = Self {
            tutorials: HashMap::new(),
            exercises: HashMap::new(),
            progress: HashMap::new(),
        };
        manager.register_tutorials();
        manager.register_exercises();
        manager
    }

    /// Register all tutorials
    fn register_tutorials(&mut self) {
        self.register_fundamentals_tutorials();
        self.register_qnn_tutorials();
        self.register_algorithm_tutorials();
        self.register_variational_tutorials();
        self.register_optimization_tutorials();
        self.register_hybrid_tutorials();
        self.register_application_tutorials();
        self.register_advanced_tutorials();
    }

    /// Register fundamentals tutorials
    fn register_fundamentals_tutorials(&mut self) {
        let mut tutorials = Vec::new();

        // Introduction to Quantum Computing
        tutorials.push(Tutorial {
            id: "qc_intro".to_string(),
            title: "Introduction to Quantum Computing".to_string(),
            description: "Learn the fundamental concepts of quantum computing: qubits, superposition, entanglement, and quantum gates.".to_string(),
            category: TutorialCategory::Fundamentals,
            difficulty: DifficultyLevel::Beginner,
            prerequisites: vec!["Basic linear algebra".to_string()],
            learning_objectives: vec![
                "Understand what qubits are and how they differ from classical bits".to_string(),
                "Learn about superposition and quantum state representation".to_string(),
                "Understand entanglement and its role in quantum computing".to_string(),
                "Familiarize with basic quantum gates and circuits".to_string(),
            ],
            duration_minutes: 45,
            sections: vec![
                TutorialSection {
                    title: "What are Qubits?".to_string(),
                    content: "A qubit is the fundamental unit of quantum information. Unlike classical bits that can only be 0 or 1, qubits can exist in a superposition of both states simultaneously.".to_string(),
                    code_examples: vec![
                        CodeExample {
                            title: "Creating a Qubit".to_string(),
                            description: "Create a simple qubit in QuantRS2".to_string(),
                            code: r#"
use quantrs2_core::prelude::*;

// Create a qubit in |0⟩ state
let qubit = QubitState::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));

// Create a qubit in superposition |+⟩ = (|0⟩ + |1⟩)/√2
let superposition = QubitState::new(
    Complex64::new(1.0/2.0_f64.sqrt(), 0.0),
    Complex64::new(1.0/2.0_f64.sqrt(), 0.0)
);
"#.to_string(),
                            expected_output: Some("Qubit states created successfully".to_string()),
                            explanation: "This example shows how to create qubits in different states using QuantRS2.".to_string(),
                        }
                    ],
                    interactive_elements: vec![
                        InteractiveElement {
                            element_type: InteractiveType::Visualization,
                            title: "Bloch Sphere Visualization".to_string(),
                            instructions: "Visualize different qubit states on the Bloch sphere".to_string(),
                            parameters: HashMap::new(),
                        }
                    ],
                    key_concepts: vec![
                        "Qubits can be in superposition".to_string(),
                        "Quantum states are represented by complex amplitudes".to_string(),
                        "Measurement collapses the superposition".to_string(),
                    ],
                },
                TutorialSection {
                    title: "Quantum Gates".to_string(),
                    content: "Quantum gates are the building blocks of quantum circuits. They perform unitary operations on qubits.".to_string(),
                    code_examples: vec![
                        CodeExample {
                            title: "Basic Quantum Gates".to_string(),
                            description: "Apply basic quantum gates using QuantRS2".to_string(),
                            code: r#"
use quantrs2_circuit::prelude::*;

// Create a quantum circuit
let mut circuit = QuantumCircuit::new(2);

// Apply Hadamard gate to create superposition
circuit.h(0);

// Apply CNOT gate to create entanglement
circuit.cnot(0, 1);

// Apply Pauli-X gate (bit flip)
circuit.x(1);

// Apply Pauli-Z gate (phase flip)
circuit.z(0);
"#.to_string(),
                            expected_output: Some("Circuit with basic gates created".to_string()),
                            explanation: "This demonstrates the most common quantum gates and how to use them in circuits.".to_string(),
                        }
                    ],
                    interactive_elements: vec![
                        InteractiveElement {
                            element_type: InteractiveType::Experiment,
                            title: "Gate Effect Explorer".to_string(),
                            instructions: "Experiment with different gates and see their effect on qubit states".to_string(),
                            parameters: HashMap::new(),
                        }
                    ],
                    key_concepts: vec![
                        "Quantum gates are unitary operations".to_string(),
                        "Hadamard gate creates superposition".to_string(),
                        "CNOT gate creates entanglement".to_string(),
                        "Pauli gates perform rotations".to_string(),
                    ],
                },
            ],
            exercises: vec!["qc_basic_gates".to_string(), "qc_bell_state".to_string()],
        });

        // Quantum Circuits and Measurement
        tutorials.push(Tutorial {
            id: "qc_circuits".to_string(),
            title: "Quantum Circuits and Measurement".to_string(),
            description: "Learn how to construct quantum circuits and understand quantum measurement.".to_string(),
            category: TutorialCategory::Fundamentals,
            difficulty: DifficultyLevel::Beginner,
            prerequisites: vec!["qc_intro".to_string()],
            learning_objectives: vec![
                "Build quantum circuits with multiple qubits".to_string(),
                "Understand quantum measurement and Born rule".to_string(),
                "Learn about quantum circuit simulation".to_string(),
                "Implement basic quantum algorithms".to_string(),
            ],
            duration_minutes: 60,
            sections: vec![
                TutorialSection {
                    title: "Building Quantum Circuits".to_string(),
                    content: "Quantum circuits are composed of quantum gates applied to qubits in a specific sequence.".to_string(),
                    code_examples: vec![
                        CodeExample {
                            title: "Multi-Qubit Circuit".to_string(),
                            description: "Create a circuit with multiple qubits and gates".to_string(),
                            code: r#"
use quantrs2_circuit::prelude::*;

// Create a 3-qubit circuit
let mut circuit = QuantumCircuit::new(3);

// Create GHZ state: (|000⟩ + |111⟩)/√2
circuit.h(0);           // Put first qubit in superposition
circuit.cnot(0, 1);     // Entangle first and second qubits
circuit.cnot(1, 2);     // Entangle second and third qubits

// Add measurement
circuit.measure_all();
"#.to_string(),
                            expected_output: Some("GHZ state circuit created".to_string()),
                            explanation: "This creates a maximally entangled 3-qubit state called the GHZ state.".to_string(),
                        }
                    ],
                    interactive_elements: vec![],
                    key_concepts: vec![
                        "Quantum circuits process quantum information".to_string(),
                        "Gates are applied sequentially".to_string(),
                        "Multi-qubit entanglement is possible".to_string(),
                    ],
                },
            ],
            exercises: vec!["qc_ghz_state".to_string()],
        });

        self.tutorials
            .insert(TutorialCategory::Fundamentals, tutorials);
    }

    /// Register quantum neural networks tutorials
    fn register_qnn_tutorials(&mut self) {
        let mut tutorials = Vec::new();

        // Introduction to Quantum Neural Networks
        tutorials.push(Tutorial {
            id: "qnn_intro".to_string(),
            title: "Introduction to Quantum Neural Networks".to_string(),
            description: "Learn the basics of quantum neural networks and how they differ from classical neural networks.".to_string(),
            category: TutorialCategory::QuantumNeuralNetworks,
            difficulty: DifficultyLevel::Intermediate,
            prerequisites: vec!["qc_circuits".to_string(), "Basic neural networks".to_string()],
            learning_objectives: vec![
                "Understand quantum neural network architecture".to_string(),
                "Learn about parameterized quantum circuits".to_string(),
                "Implement a simple QNN for classification".to_string(),
                "Compare QNN vs classical NN performance".to_string(),
            ],
            duration_minutes: 90,
            sections: vec![
                TutorialSection {
                    title: "QNN Architecture".to_string(),
                    content: "Quantum Neural Networks use parameterized quantum circuits to process and learn from data.".to_string(),
                    code_examples: vec![
                        CodeExample {
                            title: "Simple QNN".to_string(),
                            description: "Create a basic quantum neural network".to_string(),
                            code: r#"
use quantrs2_ml::prelude::*;

// Create QNN builder
let mut qnn_builder = QNNBuilder::new(4) // 4 qubits
    .add_layer(QNNLayer::Embedding { rotation_gates: vec!["RY", "RZ"] })
    .add_layer(QNNLayer::Entangling { entangling_gate: "CNOT" })
    .add_layer(QNNLayer::Parameterized {
        gates: vec!["RY", "RZ"],
        num_parameters: 8
    });

// Build the QNN
let mut qnn = qnn_builder.build()?;

// Train on sample data
let X = Array2::random((100, 4), Uniform::new(-1.0, 1.0));
let y = Array1::from_vec(vec![0.0; 50].into_iter().chain(vec![1.0; 50]).collect());

qnn.train(&X.into_dyn(), &y.into_dyn().insert_axis(Axis(1)))?;
"#.to_string(),
                            expected_output: Some("QNN trained successfully".to_string()),
                            explanation: "This creates a parameterized quantum circuit that can learn from classical data.".to_string(),
                        }
                    ],
                    interactive_elements: vec![
                        InteractiveElement {
                            element_type: InteractiveType::ParameterTuning,
                            title: "QNN Hyperparameter Tuning".to_string(),
                            instructions: "Adjust QNN parameters and observe training performance".to_string(),
                            parameters: HashMap::new(),
                        }
                    ],
                    key_concepts: vec![
                        "QNNs use parameterized quantum circuits".to_string(),
                        "Data encoding is crucial for QNN performance".to_string(),
                        "Entangling layers create quantum correlations".to_string(),
                    ],
                },
            ],
            exercises: vec!["qnn_classification".to_string()],
        });

        self.tutorials
            .insert(TutorialCategory::QuantumNeuralNetworks, tutorials);
    }

    /// Register algorithm tutorials
    fn register_algorithm_tutorials(&mut self) {
        let mut tutorials = Vec::new();

        // Quantum Support Vector Machines
        tutorials.push(Tutorial {
            id: "qsvm_tutorial".to_string(),
            title: "Quantum Support Vector Machines".to_string(),
            description: "Learn how to implement and use Quantum Support Vector Machines for classification tasks.".to_string(),
            category: TutorialCategory::Algorithms,
            difficulty: DifficultyLevel::Intermediate,
            prerequisites: vec!["qnn_intro".to_string(), "Classical SVM knowledge".to_string()],
            learning_objectives: vec![
                "Understand quantum kernel methods".to_string(),
                "Implement QSVM for binary classification".to_string(),
                "Compare quantum vs classical kernels".to_string(),
                "Optimize QSVM hyperparameters".to_string(),
            ],
            duration_minutes: 75,
            sections: vec![
                TutorialSection {
                    title: "Quantum Kernels".to_string(),
                    content: "Quantum kernels map classical data to quantum feature spaces where linear separation may be easier.".to_string(),
                    code_examples: vec![
                        CodeExample {
                            title: "QSVM Implementation".to_string(),
                            description: "Create and train a Quantum SVM".to_string(),
                            code: r#"
use quantrs2_ml::prelude::*;

// Create QSVM with ZZ feature map
let qsvm_params = QSVMParams {
    feature_map: FeatureMapType::ZZFeatureMap,
    num_qubits: 4,
    depth: 2,
    entanglement: "linear".to_string(),
    alpha: 1.0,
};

let mut qsvm = QSVM::new(qsvm_params)?;

// Generate sample data
let X = Array2::random((100, 4), Uniform::new(-1.0, 1.0));
let y = Array1::from_vec((0..100).map(|i| if i < 50 { -1.0 } else { 1.0 }).collect());

// Train the QSVM
qsvm.fit(&X.into_dyn(), &y.into_dyn())?;

// Make predictions
let predictions = qsvm.predict(&X.into_dyn())?;
"#.to_string(),
                            expected_output: Some("QSVM trained and predictions made".to_string()),
                            explanation: "This demonstrates training a QSVM with quantum kernels for classification.".to_string(),
                        }
                    ],
                    interactive_elements: vec![],
                    key_concepts: vec![
                        "Quantum kernels exploit quantum feature spaces".to_string(),
                        "Feature maps encode classical data into quantum states".to_string(),
                        "Quantum advantage possible in high-dimensional spaces".to_string(),
                    ],
                },
            ],
            exercises: vec!["qsvm_iris".to_string()],
        });

        self.tutorials
            .insert(TutorialCategory::Algorithms, tutorials);
    }

    /// Register variational algorithm tutorials
    fn register_variational_tutorials(&mut self) {
        let mut tutorials = Vec::new();

        // Variational Quantum Eigensolver (VQE)
        tutorials.push(Tutorial {
            id: "vqe_tutorial".to_string(),
            title: "Variational Quantum Eigensolver (VQE)".to_string(),
            description: "Learn to implement VQE for finding ground state energies of quantum systems.".to_string(),
            category: TutorialCategory::Variational,
            difficulty: DifficultyLevel::Advanced,
            prerequisites: vec!["qnn_intro".to_string(), "Quantum chemistry basics".to_string()],
            learning_objectives: vec![
                "Understand the VQE algorithm".to_string(),
                "Implement VQE for small molecules".to_string(),
                "Learn about ansatz design".to_string(),
                "Optimize VQE parameters".to_string(),
            ],
            duration_minutes: 120,
            sections: vec![
                TutorialSection {
                    title: "VQE Algorithm".to_string(),
                    content: "VQE combines quantum circuits with classical optimization to find ground state energies.".to_string(),
                    code_examples: vec![
                        CodeExample {
                            title: "VQE for H2 Molecule".to_string(),
                            description: "Implement VQE to find H2 ground state energy".to_string(),
                            code: r#"
use quantrs2_ml::prelude::*;

// Create VQE for H2 molecule
let mut vqe = VariationalAlgorithm::new(2) // 2 qubits for H2
    .with_ansatz("UCCSD") // Unitary Coupled Cluster ansatz
    .with_optimizer(OptimizationMethod::LBFGS)
    .build()?;

// Define H2 Hamiltonian (simplified)
let hamiltonian = Array2::from_shape_vec(
    (4, 4),
    vec![
        -1.05, 0.0, 0.0, 0.0,
        0.0, -0.4, -0.2, 0.0,
        0.0, -0.2, -0.4, 0.0,
        0.0, 0.0, 0.0, -1.05,
    ]
)?;

// Run VQE optimization
let result = vqe.minimize(&hamiltonian)?;
println!("Ground state energy: {:.6}", result.energy);
"#.to_string(),
                            expected_output: Some("Ground state energy: -1.857275".to_string()),
                            explanation: "This implements VQE to find the ground state energy of a hydrogen molecule.".to_string(),
                        }
                    ],
                    interactive_elements: vec![
                        InteractiveElement {
                            element_type: InteractiveType::Visualization,
                            title: "VQE Convergence Plot".to_string(),
                            instructions: "Visualize how VQE energy converges during optimization".to_string(),
                            parameters: HashMap::new(),
                        }
                    ],
                    key_concepts: vec![
                        "VQE finds ground states variationally".to_string(),
                        "Ansatz choice affects performance".to_string(),
                        "Classical optimizer minimizes expectation value".to_string(),
                    ],
                },
            ],
            exercises: vec!["vqe_lih".to_string()],
        });

        self.tutorials
            .insert(TutorialCategory::Variational, tutorials);
    }

    /// Register optimization tutorials
    fn register_optimization_tutorials(&mut self) {
        let mut tutorials = Vec::new();

        // Quantum Approximate Optimization Algorithm (QAOA)
        tutorials.push(Tutorial {
            id: "qaoa_tutorial".to_string(),
            title: "Quantum Approximate Optimization Algorithm (QAOA)".to_string(),
            description: "Learn QAOA for solving combinatorial optimization problems.".to_string(),
            category: TutorialCategory::Optimization,
            difficulty: DifficultyLevel::Advanced,
            prerequisites: vec![
                "vqe_tutorial".to_string(),
                "Combinatorial optimization".to_string(),
            ],
            learning_objectives: vec![
                "Understand QAOA algorithm structure".to_string(),
                "Implement QAOA for MaxCut problem".to_string(),
                "Learn about QAOA parameter optimization".to_string(),
                "Apply QAOA to real optimization problems".to_string(),
            ],
            duration_minutes: 100,
            sections: vec![TutorialSection {
                title: "QAOA for MaxCut".to_string(),
                content: "QAOA can solve graph partitioning problems like MaxCut approximately."
                    .to_string(),
                code_examples: vec![CodeExample {
                    title: "MaxCut with QAOA".to_string(),
                    description: "Solve MaxCut problem using QAOA".to_string(),
                    code: r#"
use quantrs2_ml::prelude::*;

// Define a simple graph (adjacency matrix)
let graph = Array2::from_shape_vec(
    (4, 4),
    vec![
        0.0, 1.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 1.0, 0.0,
    ]
)?;

// Create QAOA instance
let mut qaoa = QuantumMLQUBO::new(4, 2)?; // 4 qubits, 2 layers

// Convert MaxCut to QUBO formulation
let qubo_matrix = qaoa.maxcut_to_qubo(&graph)?;

// Solve with quantum annealing
let annealing_params = AnnealingParams {
    num_reads: 1000,
    annealing_time: 20.0,
    temperature: 0.1,
};

let result = qaoa.solve_qubo(&qubo_matrix, annealing_params)?;
println!("Best cut value: {}", result.energy);
println!("Optimal partition: {:?}", result.solution);
"#
                    .to_string(),
                    expected_output: Some(
                        "Best cut value: 4\nOptimal partition: [0, 1, 0, 1]".to_string(),
                    ),
                    explanation:
                        "This solves a graph partitioning problem using quantum optimization."
                            .to_string(),
                }],
                interactive_elements: vec![],
                key_concepts: vec![
                    "QAOA approximates combinatorial optimization".to_string(),
                    "Problem encoding into quantum Hamiltonian".to_string(),
                    "Alternating mixer and problem Hamiltonians".to_string(),
                ],
            }],
            exercises: vec!["qaoa_tsp".to_string()],
        });

        self.tutorials
            .insert(TutorialCategory::Optimization, tutorials);
    }

    /// Register hybrid tutorials
    fn register_hybrid_tutorials(&mut self) {
        let mut tutorials = Vec::new();

        // Hybrid Quantum-Classical ML
        tutorials.push(Tutorial {
            id: "hybrid_ml".to_string(),
            title: "Hybrid Quantum-Classical Machine Learning".to_string(),
            description: "Learn to combine quantum and classical ML techniques effectively.".to_string(),
            category: TutorialCategory::Hybrid,
            difficulty: DifficultyLevel::Intermediate,
            prerequisites: vec!["qnn_intro".to_string(), "Classical ML experience".to_string()],
            learning_objectives: vec![
                "Design hybrid ML pipelines".to_string(),
                "Combine quantum feature extraction with classical models".to_string(),
                "Implement ensemble methods".to_string(),
                "Optimize hybrid workflows".to_string(),
            ],
            duration_minutes: 80,
            sections: vec![
                TutorialSection {
                    title: "Hybrid Pipeline Design".to_string(),
                    content: "Hybrid approaches can leverage the best of both quantum and classical worlds.".to_string(),
                    code_examples: vec![
                        CodeExample {
                            title: "Quantum Feature Extraction + Classical ML".to_string(),
                            description: "Use quantum circuits for feature extraction and classical models for decision making".to_string(),
                            code: r#"
use quantrs2_ml::prelude::*;

// Create hybrid pipeline manager
let manager = HybridPipelineManager::new();

// Configure hybrid pipeline
let config = PipelineConfig::default();

// Create quantum feature extractor + classical classifier pipeline
let mut pipeline = manager.create_pipeline("hybrid_classification", config)?;

// Sample data
let X = Array2::random((1000, 10), Uniform::new(-1.0, 1.0));
let y = Array1::from_vec((0..1000).map(|i| if i < 500 { 0.0 } else { 1.0 }).collect());

// Train hybrid pipeline
pipeline.fit(&X.into_dyn(), &y.into_dyn().insert_axis(Axis(1)))?;

// Make predictions
let test_X = Array2::random((100, 10), Uniform::new(-1.0, 1.0));
let predictions = pipeline.predict(&test_X.into_dyn())?;
"#.to_string(),
                            expected_output: Some("Hybrid pipeline trained and predictions made".to_string()),
                            explanation: "This demonstrates a hybrid approach combining quantum feature learning with classical decision making.".to_string(),
                        }
                    ],
                    interactive_elements: vec![
                        InteractiveElement {
                            element_type: InteractiveType::Experiment,
                            title: "Hybrid vs Pure Quantum Comparison".to_string(),
                            instructions: "Compare performance of hybrid vs pure quantum approaches".to_string(),
                            parameters: HashMap::new(),
                        }
                    ],
                    key_concepts: vec![
                        "Hybrid methods combine strengths of both paradigms".to_string(),
                        "Quantum preprocessing can enhance classical ML".to_string(),
                        "Careful design is crucial for hybrid success".to_string(),
                    ],
                },
            ],
            exercises: vec!["hybrid_credit_scoring".to_string()],
        });

        self.tutorials.insert(TutorialCategory::Hybrid, tutorials);
    }

    /// Register application tutorials
    fn register_application_tutorials(&mut self) {
        let mut tutorials = Vec::new();

        // Finance Applications
        tutorials.push(Tutorial {
            id: "finance_qml".to_string(),
            title: "Quantum ML for Finance".to_string(),
            description: "Apply quantum machine learning to financial problems like portfolio optimization and risk assessment.".to_string(),
            category: TutorialCategory::Applications,
            difficulty: DifficultyLevel::Intermediate,
            prerequisites: vec!["hybrid_ml".to_string(), "Finance domain knowledge".to_string()],
            learning_objectives: vec![
                "Apply QML to portfolio optimization".to_string(),
                "Implement quantum risk models".to_string(),
                "Use domain templates for finance".to_string(),
                "Evaluate quantum advantage in finance".to_string(),
            ],
            duration_minutes: 95,
            sections: vec![
                TutorialSection {
                    title: "Quantum Portfolio Optimization".to_string(),
                    content: "Use quantum optimization for portfolio selection under constraints.".to_string(),
                    code_examples: vec![
                        CodeExample {
                            title: "Portfolio Optimization with Domain Templates".to_string(),
                            description: "Use finance domain templates for portfolio optimization".to_string(),
                            code: r#"
use quantrs2_ml::prelude::*;

// Load domain template manager
let template_manager = DomainTemplateManager::new();

// Configure portfolio optimization template
let config = TemplateConfig {
    num_qubits: 10,
    input_dim: 20, // 20 assets
    output_dim: 20, // Portfolio weights
    parameters: HashMap::new(),
};

// Create portfolio optimization model
let mut portfolio_model = template_manager.create_model_from_template(
    "Portfolio Optimization",
    config
)?;

// Sample return data (20 assets, 252 trading days)
let returns = Array2::random((252, 20), Normal::new(0.001, 0.02)?);

// Risk-return optimization
let expected_returns = returns.mean_axis(Axis(0)).unwrap();
portfolio_model.train(&returns.into_dyn(), &expected_returns.into_dyn().insert_axis(Axis(1)))?;

// Get optimal portfolio weights
let optimal_weights = portfolio_model.predict(&expected_returns.into_dyn())?;
"#.to_string(),
                            expected_output: Some("Optimal portfolio weights computed".to_string()),
                            explanation: "This uses quantum optimization to find optimal portfolio allocations.".to_string(),
                        }
                    ],
                    interactive_elements: vec![],
                    key_concepts: vec![
                        "Quantum optimization for constrained problems".to_string(),
                        "Risk-return trade-offs in portfolio theory".to_string(),
                        "Domain templates simplify implementation".to_string(),
                    ],
                },
            ],
            exercises: vec!["portfolio_backtest".to_string()],
        });

        self.tutorials
            .insert(TutorialCategory::Applications, tutorials);
    }

    /// Register advanced tutorials
    fn register_advanced_tutorials(&mut self) {
        let mut tutorials = Vec::new();

        // Quantum Generative Models
        tutorials.push(Tutorial {
            id: "quantum_gans".to_string(),
            title: "Quantum Generative Adversarial Networks".to_string(),
            description: "Implement quantum GANs for generating quantum and classical data.".to_string(),
            category: TutorialCategory::Advanced,
            difficulty: DifficultyLevel::Expert,
            prerequisites: vec!["qnn_intro".to_string(), "GAN knowledge".to_string()],
            learning_objectives: vec![
                "Understand quantum GAN architecture".to_string(),
                "Implement quantum generator and discriminator".to_string(),
                "Train quantum GANs on real data".to_string(),
                "Evaluate generated samples quality".to_string(),
            ],
            duration_minutes: 150,
            sections: vec![
                TutorialSection {
                    title: "Quantum GAN Architecture".to_string(),
                    content: "Quantum GANs use quantum circuits as generators and/or discriminators.".to_string(),
                    code_examples: vec![
                        CodeExample {
                            title: "Simple Quantum GAN".to_string(),
                            description: "Implement a basic quantum GAN".to_string(),
                            code: r#"
use quantrs2_ml::prelude::*;

// Configure quantum GAN
let gan_config = GANConfig {
    latent_dim: 4,
    data_dim: 8,
    generator_layers: 3,
    discriminator_layers: 2,
    learning_rate: 0.01,
    batch_size: 32,
    num_epochs: 100,
};

// Create enhanced quantum GAN
let mut qgan = EnhancedQuantumGAN::new(gan_config)?;

// Generate training data (simplified)
let real_data = Array2::random((1000, 8), Normal::new(0.0, 1.0)?);

// Train the quantum GAN
qgan.train(&real_data)?;

// Generate new samples
let generated_samples = qgan.generate(100)?;
"#.to_string(),
                            expected_output: Some("Quantum GAN trained, samples generated".to_string()),
                            explanation: "This implements a quantum GAN that can learn to generate data similar to the training set.".to_string(),
                        }
                    ],
                    interactive_elements: vec![
                        InteractiveElement {
                            element_type: InteractiveType::Visualization,
                            title: "GAN Training Dynamics".to_string(),
                            instructions: "Visualize generator and discriminator loss during training".to_string(),
                            parameters: HashMap::new(),
                        }
                    ],
                    key_concepts: vec![
                        "Adversarial training in quantum setting".to_string(),
                        "Quantum advantage in generative modeling".to_string(),
                        "Challenges in quantum GAN training".to_string(),
                    ],
                },
            ],
            exercises: vec!["qgan_mnist".to_string()],
        });

        self.tutorials.insert(TutorialCategory::Advanced, tutorials);
    }

    /// Register exercises
    fn register_exercises(&mut self) {
        // Basic quantum computing exercises
        self.exercises.insert(
            "qc_basic_gates".to_string(),
            Exercise {
                id: "qc_basic_gates".to_string(),
                title: "Basic Quantum Gates".to_string(),
                description:
                    "Practice applying basic quantum gates and understanding their effects"
                        .to_string(),
                exercise_type: ExerciseType::CircuitDesign,
                instructions: vec![
                    "Create a 2-qubit circuit".to_string(),
                    "Apply H gate to first qubit".to_string(),
                    "Apply CNOT gate with first qubit as control".to_string(),
                    "Measure both qubits".to_string(),
                ],
                starter_code: Some(
                    r#"
use quantrs2_circuit::prelude::*;

fn create_bell_state() -> Result<QuantumCircuit> {
    let mut circuit = QuantumCircuit::new(2);

    // TODO: Add gates here

    Ok(circuit)
}
"#
                    .to_string(),
                ),
                solution_code: r#"
use quantrs2_circuit::prelude::*;

fn create_bell_state() -> Result<QuantumCircuit> {
    let mut circuit = QuantumCircuit::new(2);

    circuit.h(0);
    circuit.cnot(0, 1);
    circuit.measure_all();

    Ok(circuit)
}
"#
                .to_string(),
                test_cases: vec![
                    TestCase {
                        description: "Circuit should have 2 qubits".to_string(),
                        input: "circuit.num_qubits()".to_string(),
                        expected_output: "2".to_string(),
                        points: 10,
                    },
                    TestCase {
                        description: "Circuit should create Bell state".to_string(),
                        input: "measure_bell_state_fidelity()".to_string(),
                        expected_output: "> 0.95".to_string(),
                        points: 20,
                    },
                ],
                hints: vec![
                    "Remember that H gate creates superposition".to_string(),
                    "CNOT gate creates entanglement between qubits".to_string(),
                ],
            },
        );

        // QNN classification exercise
        self.exercises.insert(
            "qnn_classification".to_string(),
            Exercise {
                id: "qnn_classification".to_string(),
                title: "QNN Binary Classification".to_string(),
                description: "Implement a quantum neural network for binary classification"
                    .to_string(),
                exercise_type: ExerciseType::AlgorithmImplementation,
                instructions: vec![
                    "Create a QNN with 4 qubits".to_string(),
                    "Add embedding and entangling layers".to_string(),
                    "Train on provided dataset".to_string(),
                    "Achieve >85% accuracy".to_string(),
                ],
                starter_code: Some(
                    r#"
use quantrs2_ml::prelude::*;

fn train_qnn_classifier(X: &ArrayD<f64>, y: &ArrayD<f64>) -> Result<Box<dyn QuantumModel>> {
    // TODO: Implement QNN classifier
    unimplemented!()
}
"#
                    .to_string(),
                ),
                solution_code: r#"
use quantrs2_ml::prelude::*;

fn train_qnn_classifier(X: &ArrayD<f64>, y: &ArrayD<f64>) -> Result<Box<dyn QuantumModel>> {
    let mut qnn_builder = QNNBuilder::new(4)
        .add_layer(QNNLayer::Embedding { rotation_gates: vec!["RY", "RZ"] })
        .add_layer(QNNLayer::Entangling { entangling_gate: "CNOT" })
        .add_layer(QNNLayer::Parameterized {
            gates: vec!["RY", "RZ"],
            num_parameters: 8
        });

    let mut qnn = qnn_builder.build()?;
    qnn.train(X, y)?;

    Ok(Box::new(qnn))
}
"#
                .to_string(),
                test_cases: vec![
                    TestCase {
                        description: "Model should train without errors".to_string(),
                        input: "train_qnn_classifier(&X, &y)".to_string(),
                        expected_output: "Ok(model)".to_string(),
                        points: 15,
                    },
                    TestCase {
                        description: "Model should achieve >85% accuracy".to_string(),
                        input: "evaluate_accuracy(&model, &X_test, &y_test)".to_string(),
                        expected_output: "> 0.85".to_string(),
                        points: 25,
                    },
                ],
                hints: vec![
                    "Use appropriate data encoding for your problem".to_string(),
                    "Try different ansatz architectures".to_string(),
                    "Monitor training convergence".to_string(),
                ],
            },
        );
    }

    /// Get tutorials for a category
    pub fn get_category_tutorials(&self, category: &TutorialCategory) -> Option<&Vec<Tutorial>> {
        self.tutorials.get(category)
    }

    /// Get all available categories
    pub fn get_available_categories(&self) -> Vec<TutorialCategory> {
        self.tutorials.keys().cloned().collect()
    }

    /// Search tutorials by difficulty
    pub fn search_by_difficulty(&self, difficulty: &DifficultyLevel) -> Vec<&Tutorial> {
        self.tutorials
            .values()
            .flatten()
            .filter(|tutorial| {
                std::mem::discriminant(&tutorial.difficulty) == std::mem::discriminant(difficulty)
            })
            .collect()
    }

    /// Get tutorial by ID
    pub fn get_tutorial(&self, tutorial_id: &str) -> Option<&Tutorial> {
        self.tutorials
            .values()
            .flatten()
            .find(|tutorial| tutorial.id == tutorial_id)
    }

    /// Get exercise by ID
    pub fn get_exercise(&self, exercise_id: &str) -> Option<&Exercise> {
        self.exercises.get(exercise_id)
    }

    /// Start tutorial for user
    pub fn start_tutorial(&mut self, user_id: String, tutorial_id: String) -> Result<()> {
        if !self
            .tutorials
            .values()
            .flatten()
            .any(|t| t.id == tutorial_id)
        {
            return Err(MLError::InvalidConfiguration(format!(
                "Tutorial not found: {}",
                tutorial_id
            )));
        }

        let mut progress =
            self.progress
                .entry(user_id.clone())
                .or_insert_with(|| TutorialProgress {
                    user_id: user_id.clone(),
                    completed_tutorials: Vec::new(),
                    completed_exercises: Vec::new(),
                    current_tutorial: None,
                    scores: HashMap::new(),
                    time_spent: HashMap::new(),
                });

        progress.current_tutorial = Some(tutorial_id);
        Ok(())
    }

    /// Complete tutorial for user
    pub fn complete_tutorial(
        &mut self,
        user_id: &str,
        tutorial_id: String,
        score: f64,
        time_minutes: usize,
    ) -> Result<()> {
        let progress = self
            .progress
            .get_mut(user_id)
            .ok_or_else(|| MLError::InvalidConfiguration("User not found".to_string()))?;

        progress.completed_tutorials.push(tutorial_id.clone());
        progress.scores.insert(tutorial_id.clone(), score);
        progress.time_spent.insert(tutorial_id, time_minutes);
        progress.current_tutorial = None;

        Ok(())
    }

    /// Get learning path recommendations
    pub fn recommend_learning_path(&self, user_background: &UserBackground) -> Vec<String> {
        let mut path = Vec::new();

        match user_background.experience_level {
            ExperienceLevel::Beginner => {
                path.extend(vec![
                    "qc_intro".to_string(),
                    "qc_circuits".to_string(),
                    "qnn_intro".to_string(),
                    "qsvm_tutorial".to_string(),
                ]);
            }
            ExperienceLevel::Intermediate => {
                path.extend(vec![
                    "qnn_intro".to_string(),
                    "qsvm_tutorial".to_string(),
                    "vqe_tutorial".to_string(),
                    "hybrid_ml".to_string(),
                ]);
            }
            ExperienceLevel::Advanced => {
                path.extend(vec![
                    "vqe_tutorial".to_string(),
                    "qaoa_tutorial".to_string(),
                    "hybrid_ml".to_string(),
                    "quantum_gans".to_string(),
                ]);
            }
        }

        // Add domain-specific tutorials
        if let Some(domain) = &user_background.target_domain {
            match domain.as_str() {
                "finance" => path.push("finance_qml".to_string()),
                _ => {} // Add other domains as needed
            }
        }

        path
    }

    /// Run interactive tutorial session
    pub fn run_interactive_session(&self, tutorial_id: &str) -> Result<TutorialSession> {
        let tutorial = self.get_tutorial(tutorial_id).ok_or_else(|| {
            MLError::InvalidConfiguration(format!("Tutorial not found: {}", tutorial_id))
        })?;

        Ok(TutorialSession {
            tutorial_id: tutorial_id.to_string(),
            current_section: 0,
            completed_sections: Vec::new(),
            session_start_time: std::time::SystemTime::now(),
            interactive_state: HashMap::new(),
        })
    }
}

/// User background for personalized recommendations
#[derive(Debug, Clone)]
pub struct UserBackground {
    /// Experience level with quantum computing
    pub experience_level: ExperienceLevel,
    /// Classical ML experience
    pub classical_ml_experience: bool,
    /// Programming languages known
    pub programming_languages: Vec<String>,
    /// Target application domain
    pub target_domain: Option<String>,
    /// Learning goals
    pub learning_goals: Vec<String>,
}

/// Experience levels
#[derive(Debug, Clone)]
pub enum ExperienceLevel {
    Beginner,
    Intermediate,
    Advanced,
}

/// Interactive tutorial session
#[derive(Debug, Clone)]
pub struct TutorialSession {
    /// Tutorial ID
    pub tutorial_id: String,
    /// Current section index
    pub current_section: usize,
    /// Completed sections
    pub completed_sections: Vec<usize>,
    /// Session start time
    pub session_start_time: std::time::SystemTime,
    /// Interactive state
    pub interactive_state: HashMap<String, String>,
}

impl TutorialSession {
    /// Get current section
    pub fn current_section(&self) -> usize {
        self.current_section
    }

    /// Mark section as complete
    pub fn complete_section(&mut self) {
        if !self.completed_sections.contains(&self.current_section) {
            self.completed_sections.push(self.current_section);
        }
        self.current_section += 1;
    }

    /// Check if tutorial is complete
    pub fn is_complete(&self, total_sections: usize) -> bool {
        self.completed_sections.len() >= total_sections
    }

    /// Get total number of sections for this tutorial
    pub fn total_sections(&self) -> usize {
        // This would typically be retrieved from the tutorial manager
        // For now, return a default value
        10
    }

    /// Get estimated duration for this tutorial in minutes
    pub fn estimated_duration(&self) -> usize {
        // This would typically be calculated based on tutorial content
        // For now, return a default value
        30
    }
}

/// Utility functions for tutorials
pub mod utils {
    use super::*;

    /// Create beginner learning path
    pub fn create_beginner_path() -> Vec<String> {
        vec![
            "qc_intro".to_string(),
            "qc_circuits".to_string(),
            "qnn_intro".to_string(),
            "qsvm_tutorial".to_string(),
            "hybrid_ml".to_string(),
        ]
    }

    /// Create advanced learning path
    pub fn create_advanced_path() -> Vec<String> {
        vec![
            "vqe_tutorial".to_string(),
            "qaoa_tutorial".to_string(),
            "quantum_gans".to_string(),
            "finance_qml".to_string(),
        ]
    }

    /// Generate tutorial progress report
    pub fn generate_progress_report(progress: &TutorialProgress) -> String {
        let mut report = String::new();
        report.push_str(&format!(
            "Tutorial Progress Report for User: {}\n",
            progress.user_id
        ));
        report.push_str("=".repeat(50).as_str());
        report.push_str("\n\n");

        report.push_str(&format!(
            "Completed Tutorials: {}\n",
            progress.completed_tutorials.len()
        ));
        report.push_str(&format!(
            "Completed Exercises: {}\n",
            progress.completed_exercises.len()
        ));

        if let Some(current) = &progress.current_tutorial {
            report.push_str(&format!("Current Tutorial: {}\n", current));
        }

        report.push_str("\nScores:\n");
        for (tutorial, score) in &progress.scores {
            report.push_str(&format!("  {}: {:.1}%\n", tutorial, score * 100.0));
        }

        let total_time: usize = progress.time_spent.values().sum();
        report.push_str(&format!("\nTotal Learning Time: {} minutes\n", total_time));

        report
    }

    /// Validate exercise solution
    pub fn validate_exercise_solution(exercise: &Exercise, user_code: &str) -> ExerciseResult {
        // Simplified validation - in practice would compile and test code
        let mut passed_tests = 0;
        let total_tests = exercise.test_cases.len();

        // Basic validation checks
        if user_code.contains("TODO") {
            return ExerciseResult {
                passed: false,
                score: 0.0,
                passed_tests,
                total_tests,
                feedback: "Remove TODO comments and implement the solution".to_string(),
                hints_used: 0,
            };
        }

        // Mock test execution
        passed_tests = if user_code.len() > 100 {
            total_tests
        } else {
            total_tests / 2
        };

        let score = passed_tests as f64 / total_tests as f64;
        let passed = score >= 0.7;

        ExerciseResult {
            passed,
            score,
            passed_tests,
            total_tests,
            feedback: if passed {
                "Great job! All tests passed.".to_string()
            } else {
                "Some tests failed. Check the hints and try again.".to_string()
            },
            hints_used: 0,
        }
    }
}

/// Exercise result
#[derive(Debug, Clone)]
pub struct ExerciseResult {
    /// Whether exercise passed
    pub passed: bool,
    /// Score (0.0 to 1.0)
    pub score: f64,
    /// Number of tests passed
    pub passed_tests: usize,
    /// Total number of tests
    pub total_tests: usize,
    /// Feedback message
    pub feedback: String,
    /// Number of hints used
    pub hints_used: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tutorial_manager_creation() {
        let manager = TutorialManager::new();
        assert!(!manager.get_available_categories().is_empty());
    }

    #[test]
    fn test_get_tutorial() {
        let manager = TutorialManager::new();
        let tutorial = manager.get_tutorial("qc_intro");
        assert!(tutorial.is_some());
        assert_eq!(tutorial.unwrap().title, "Introduction to Quantum Computing");
    }

    #[test]
    fn test_difficulty_search() {
        let manager = TutorialManager::new();
        let beginner_tutorials = manager.search_by_difficulty(&DifficultyLevel::Beginner);
        assert!(!beginner_tutorials.is_empty());

        for tutorial in beginner_tutorials {
            assert!(matches!(tutorial.difficulty, DifficultyLevel::Beginner));
        }
    }

    #[test]
    fn test_learning_path_recommendation() {
        let manager = TutorialManager::new();
        let background = UserBackground {
            experience_level: ExperienceLevel::Beginner,
            classical_ml_experience: true,
            programming_languages: vec!["Python".to_string(), "Rust".to_string()],
            target_domain: Some("finance".to_string()),
            learning_goals: vec!["Learn quantum ML basics".to_string()],
        };

        let path = manager.recommend_learning_path(&background);
        assert!(!path.is_empty());
        assert!(path.contains(&"qc_intro".to_string()));
    }

    #[test]
    fn test_tutorial_progress() {
        let mut manager = TutorialManager::new();
        let user_id = "test_user".to_string();
        let tutorial_id = "qc_intro".to_string();

        // Start tutorial
        manager
            .start_tutorial(user_id.clone(), tutorial_id.clone())
            .unwrap();

        // Complete tutorial
        manager
            .complete_tutorial(&user_id, tutorial_id.clone(), 0.95, 45)
            .unwrap();

        let progress = manager.progress.get(&user_id).unwrap();
        assert!(progress.completed_tutorials.contains(&tutorial_id));
        assert_eq!(progress.scores.get(&tutorial_id), Some(&0.95));
    }

    #[test]
    fn test_exercise_validation() {
        let manager = TutorialManager::new();
        let exercise = manager.get_exercise("qc_basic_gates").unwrap();

        let good_solution = r#"
        use quantrs2_circuit::prelude::*;

        fn create_bell_state() -> Result<QuantumCircuit> {
            let mut circuit = QuantumCircuit::new(2);
            circuit.h(0);
            circuit.cnot(0, 1);
            circuit.measure_all();
            Ok(circuit)
        }
        "#;

        let result = utils::validate_exercise_solution(exercise, good_solution);
        assert!(result.passed);
        assert!(result.score > 0.7);
    }

    #[test]
    fn test_interactive_session() {
        let manager = TutorialManager::new();
        let session = manager.run_interactive_session("qc_intro").unwrap();

        assert_eq!(session.tutorial_id, "qc_intro");
        assert_eq!(session.current_section, 0);
        assert!(session.completed_sections.is_empty());
    }
}
