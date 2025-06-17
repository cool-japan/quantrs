//! Test scenario engine for complex problem generation

use super::*;

/// Test scenario engine for complex problem generation
#[derive(Debug)]
pub struct TestScenarioEngine {
    /// Available test scenarios
    pub scenarios: HashMap<String, TestScenario>,
    /// Scenario execution history
    pub execution_history: VecDeque<ScenarioExecution>,
    /// Problem generators
    pub generators: Vec<ProblemGenerator>,
    /// Validation rules
    pub validation_rules: Vec<ValidationRule>,
}

/// Individual test scenario
#[derive(Debug, Clone)]
pub struct TestScenario {
    /// Scenario identifier
    pub id: String,
    /// Scenario description
    pub description: String,
    /// Problem specification
    pub problem_specs: ProblemSpecification,
    /// Expected performance metrics
    pub expected_metrics: ExpectedMetrics,
    /// Validation criteria
    pub validation_criteria: Vec<ValidationCriterion>,
    /// Timeout for scenario execution
    pub timeout: Duration,
    /// Maximum number of retries
    pub max_retries: usize,
}

/// Scenario execution record
#[derive(Debug, Clone)]
pub struct ScenarioExecution {
    /// Scenario identifier
    pub scenario_id: String,
    /// Execution timestamp
    pub timestamp: Instant,
    /// Execution duration
    pub duration: Duration,
    /// Success status
    pub success: bool,
    /// Performance metrics achieved
    pub metrics: HashMap<String, f64>,
    /// Error information (if any)
    pub error: Option<String>,
}

/// Problem generator for test scenarios
#[derive(Debug)]
pub struct ProblemGenerator {
    /// Generator identifier
    pub id: String,
    /// Generator type
    pub generator_type: GeneratorType,
    /// Generation parameters
    pub parameters: HashMap<String, f64>,
    /// Problem constraints
    pub constraints: Vec<GeneratorConstraint>,
}

/// Types of problem generators
#[derive(Debug, Clone, PartialEq)]
pub enum GeneratorType {
    /// Random problem generator
    Random,
    /// Structured problem generator
    Structured,
    /// Real-world instance generator
    RealWorld,
    /// Adversarial generator
    Adversarial,
    /// Benchmark generator
    Benchmark,
}

/// Constraints for problem generation
#[derive(Debug, Clone)]
pub struct GeneratorConstraint {
    /// Constraint type
    pub constraint_type: String,
    /// Constraint parameters
    pub parameters: HashMap<String, f64>,
    /// Constraint priority
    pub priority: f64,
}

/// Validation rule for test scenarios
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule identifier
    pub id: String,
    /// Rule description
    pub description: String,
    /// Rule condition
    pub condition: RuleCondition,
    /// Expected outcome
    pub expected_outcome: RuleOutcome,
    /// Rule severity
    pub severity: RuleSeverity,
}

/// Condition for validation rule
#[derive(Debug, Clone)]
pub struct RuleCondition {
    /// Condition expression
    pub expression: String,
    /// Condition parameters
    pub parameters: HashMap<String, PropertyValue>,
    /// Evaluation method
    pub evaluation_method: EvaluationMethod,
}

/// Expected outcome for validation rule
#[derive(Debug, Clone)]
pub struct RuleOutcome {
    /// Expected result
    pub expected_result: PropertyValue,
    /// Tolerance for comparison
    pub tolerance: f64,
    /// Comparison operator
    pub comparison_op: ConditionOperator,
}

/// Severity levels for validation rules
#[derive(Debug, Clone, PartialEq)]
pub enum RuleSeverity {
    /// Critical rule (must pass)
    Critical,
    /// Warning rule (should pass)
    Warning,
    /// Informational rule
    Info,
}

/// Methods for rule evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationMethod {
    /// Direct comparison
    Direct,
    /// Statistical test
    Statistical,
    /// Machine learning model
    MachineLearning,
    /// Custom evaluation
    Custom(String),
}

impl TestScenarioEngine {
    pub fn new() -> Self {
        let mut scenarios = HashMap::new();

        // Create default test scenarios
        scenarios.insert(
            "basic_optimization".to_string(),
            TestScenario {
                id: "basic_optimization".to_string(),
                description: "Basic optimization scenario".to_string(),
                problem_specs: ProblemSpecification {
                    problem_type: ProblemType::RandomIsing,
                    size_range: (10, 100),
                    density: DensitySpec {
                        edge_density: (0.1, 0.3),
                        constraint_density: None,
                        bias_sparsity: Some(0.5),
                    },
                    constraints: ConstraintSpec {
                        num_constraints: None,
                        constraint_types: Vec::new(),
                        strength_range: (0.1, 1.0),
                    },
                    seed: Some(42),
                },
                expected_metrics: ExpectedMetrics {
                    solution_quality: (0.7, 1.0),
                    runtime: (Duration::from_millis(100), Duration::from_secs(10)),
                    success_rate: 0.9,
                    convergence: ConvergenceExpectation {
                        convergence_time: Duration::from_secs(5),
                        final_energy: None,
                        energy_gap: None,
                    },
                },
                validation_criteria: vec![ValidationCriterion {
                    criterion_type: CriterionType::Performance,
                    expected_value: CriterionValue::Range(0.7, 1.0),
                    tolerance: 0.1,
                    mandatory: true,
                }],
                timeout: Duration::from_secs(30),
                max_retries: 3,
            },
        );

        scenarios.insert(
            "large_scale_test".to_string(),
            TestScenario {
                id: "large_scale_test".to_string(),
                description: "Large scale problem test".to_string(),
                problem_specs: ProblemSpecification {
                    problem_type: ProblemType::RandomIsing,
                    size_range: (1000, 5000),
                    density: DensitySpec {
                        edge_density: (0.05, 0.15),
                        constraint_density: None,
                        bias_sparsity: Some(0.3),
                    },
                    constraints: ConstraintSpec {
                        num_constraints: None,
                        constraint_types: Vec::new(),
                        strength_range: (0.1, 1.0),
                    },
                    seed: Some(123),
                },
                expected_metrics: ExpectedMetrics {
                    solution_quality: (0.6, 0.9),
                    runtime: (Duration::from_secs(10), Duration::from_secs(300)),
                    success_rate: 0.8,
                    convergence: ConvergenceExpectation {
                        convergence_time: Duration::from_secs(60),
                        final_energy: None,
                        energy_gap: None,
                    },
                },
                validation_criteria: vec![
                    ValidationCriterion {
                        criterion_type: CriterionType::Performance,
                        expected_value: CriterionValue::Range(0.6, 0.9),
                        tolerance: 0.1,
                        mandatory: true,
                    },
                    ValidationCriterion {
                        criterion_type: CriterionType::Runtime,
                        expected_value: CriterionValue::Maximum(300.0),
                        tolerance: 0.0,
                        mandatory: true,
                    },
                ],
                timeout: Duration::from_secs(600),
                max_retries: 2,
            },
        );

        Self {
            scenarios,
            execution_history: VecDeque::new(),
            generators: Self::create_default_generators(),
            validation_rules: Self::create_default_validation_rules(),
        }
    }

    /// Create default problem generators
    fn create_default_generators() -> Vec<ProblemGenerator> {
        vec![
            ProblemGenerator {
                id: "random_ising".to_string(),
                generator_type: GeneratorType::Random,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("density".to_string(), 0.2);
                    params.insert("bias_range".to_string(), 1.0);
                    params.insert("coupling_range".to_string(), 1.0);
                    params
                },
                constraints: Vec::new(),
            },
            ProblemGenerator {
                id: "structured_ising".to_string(),
                generator_type: GeneratorType::Structured,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("regularity".to_string(), 0.8);
                    params.insert("locality".to_string(), 0.9);
                    params
                },
                constraints: Vec::new(),
            },
        ]
    }

    /// Create default validation rules
    fn create_default_validation_rules() -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                id: "solution_feasibility".to_string(),
                description: "Solution must be feasible".to_string(),
                condition: RuleCondition {
                    expression: "solution_valid == true".to_string(),
                    parameters: HashMap::new(),
                    evaluation_method: EvaluationMethod::Direct,
                },
                expected_outcome: RuleOutcome {
                    expected_result: PropertyValue::Boolean(true),
                    tolerance: 0.0,
                    comparison_op: ConditionOperator::Equal,
                },
                severity: RuleSeverity::Critical,
            },
            ValidationRule {
                id: "performance_threshold".to_string(),
                description: "Performance must exceed minimum threshold".to_string(),
                condition: RuleCondition {
                    expression: "solution_quality >= threshold".to_string(),
                    parameters: {
                        let mut params = HashMap::new();
                        params.insert("threshold".to_string(), PropertyValue::Numeric(0.5));
                        params
                    },
                    evaluation_method: EvaluationMethod::Direct,
                },
                expected_outcome: RuleOutcome {
                    expected_result: PropertyValue::Boolean(true),
                    tolerance: 0.0,
                    comparison_op: ConditionOperator::Equal,
                },
                severity: RuleSeverity::Warning,
            },
        ]
    }

    /// Add new test scenario
    pub fn add_scenario(&mut self, scenario: TestScenario) {
        self.scenarios.insert(scenario.id.clone(), scenario);
    }

    /// Remove test scenario
    pub fn remove_scenario(&mut self, scenario_id: &str) -> Option<TestScenario> {
        self.scenarios.remove(scenario_id)
    }

    /// Get scenario by ID
    pub fn get_scenario(&self, scenario_id: &str) -> Option<&TestScenario> {
        self.scenarios.get(scenario_id)
    }

    /// Record scenario execution
    pub fn record_execution(&mut self, execution: ScenarioExecution) {
        self.execution_history.push_back(execution);

        // Keep only recent executions
        while self.execution_history.len() > 1000 {
            self.execution_history.pop_front();
        }
    }

    /// Get execution history for scenario
    pub fn get_execution_history(&self, scenario_id: &str) -> Vec<&ScenarioExecution> {
        self.execution_history
            .iter()
            .filter(|exec| exec.scenario_id == scenario_id)
            .collect()
    }

    /// Generate problem from specification
    pub fn generate_problem(&self, spec: &ProblemSpecification) -> ApplicationResult<IsingModel> {
        // Find appropriate generator
        let generator = self
            .generators
            .iter()
            .find(|g| self.can_generate_problem_type(g, &spec.problem_type))
            .ok_or_else(|| {
                ApplicationError::ConfigurationError(format!(
                    "No generator available for problem type: {:?}",
                    spec.problem_type
                ))
            })?;

        self.generate_with_generator(generator, spec)
    }

    /// Check if generator can handle problem type
    fn can_generate_problem_type(
        &self,
        generator: &ProblemGenerator,
        problem_type: &ProblemType,
    ) -> bool {
        match (generator.generator_type.clone(), problem_type) {
            (GeneratorType::Random, ProblemType::RandomIsing) => true,
            (GeneratorType::Structured, _) => true,
            (GeneratorType::Benchmark, _) => true,
            _ => false,
        }
    }

    /// Generate problem using specific generator
    fn generate_with_generator(
        &self,
        generator: &ProblemGenerator,
        spec: &ProblemSpecification,
    ) -> ApplicationResult<IsingModel> {
        let size = (spec.size_range.0 + spec.size_range.1) / 2;
        let mut problem = IsingModel::new(size);

        match generator.generator_type {
            GeneratorType::Random => self.generate_random_problem(&mut problem, spec, generator)?,
            GeneratorType::Structured => {
                self.generate_structured_problem(&mut problem, spec, generator)?
            }
            _ => {
                return Err(ApplicationError::ConfigurationError(format!(
                    "Generator type {:?} not implemented",
                    generator.generator_type
                )));
            }
        }

        Ok(problem)
    }

    /// Generate random problem
    fn generate_random_problem(
        &self,
        problem: &mut IsingModel,
        spec: &ProblemSpecification,
        generator: &ProblemGenerator,
    ) -> ApplicationResult<()> {
        let size = problem.num_qubits;
        let bias_range = generator.parameters.get("bias_range").unwrap_or(&1.0);
        let coupling_range = generator.parameters.get("coupling_range").unwrap_or(&1.0);

        // Set random biases
        for i in 0..size {
            let bias = (i as f64 % 10.0) / 10.0 * bias_range - bias_range / 2.0;
            problem.set_bias(i, bias)?;
        }

        // Set random couplings based on density
        let target_density = (spec.density.edge_density.0 + spec.density.edge_density.1) / 2.0;
        let max_edges = size * (size - 1) / 2;
        let target_edges = (max_edges as f64 * target_density) as usize;

        let mut edges_added = 0;
        for i in 0..size {
            for j in (i + 1)..size {
                if edges_added >= target_edges {
                    break;
                }

                if (i + j) % 3 == 0 {
                    let coupling =
                        ((i + j) as f64 % 20.0) / 20.0 * coupling_range - coupling_range / 2.0;
                    problem.set_coupling(i, j, coupling)?;
                    edges_added += 1;
                }
            }
            if edges_added >= target_edges {
                break;
            }
        }

        Ok(())
    }

    /// Generate structured problem
    fn generate_structured_problem(
        &self,
        problem: &mut IsingModel,
        spec: &ProblemSpecification,
        generator: &ProblemGenerator,
    ) -> ApplicationResult<()> {
        let size = problem.num_qubits;
        let regularity = generator.parameters.get("regularity").unwrap_or(&0.8);
        let locality = generator.parameters.get("locality").unwrap_or(&0.9);

        // Create structured biases
        for i in 0..size {
            let bias = if (i as f64) < size as f64 * regularity {
                // Regular pattern
                ((i % 4) as f64 - 1.5) / 2.0
            } else {
                // Random component
                (i as f64 % 7.0) / 7.0 - 0.5
            };
            problem.set_bias(i, bias)?;
        }

        // Create local connections
        let local_range = (size as f64 * locality) as usize;
        for i in 0..size {
            let max_j = (i + local_range).min(size);
            for j in (i + 1)..max_j {
                if (i + j) % 2 == 0 {
                    let coupling = ((i as f64 - j as f64).abs() / local_range as f64) * 0.5;
                    problem.set_coupling(i, j, coupling)?;
                }
            }
        }

        Ok(())
    }
}
