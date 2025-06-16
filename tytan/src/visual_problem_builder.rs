//! Visual problem builder for interactive QUBO construction.
//!
//! This module provides a visual interface for building quantum optimization
//! problems without requiring direct code writing. It includes drag-and-drop
//! variable creation, constraint specification, and real-time validation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Visual problem builder for interactive QUBO construction
pub struct VisualProblemBuilder {
    /// Current problem being built
    problem: VisualProblem,
    /// Builder configuration
    config: BuilderConfig,
    /// Validation engine
    validator: ProblemValidator,
    /// Code generator
    generator: CodeGenerator,
    /// Undo/redo stack
    history: ProblemHistory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuilderConfig {
    /// Enable real-time validation
    pub real_time_validation: bool,
    /// Auto-save interval
    pub auto_save_interval: Option<std::time::Duration>,
    /// Maximum problem size
    pub max_problem_size: usize,
    /// Default variable type
    pub default_variable_type: VariableType,
    /// Export formats supported
    pub supported_formats: Vec<ExportFormat>,
    /// Theme settings
    pub theme: Theme,
}

impl Default for BuilderConfig {
    fn default() -> Self {
        Self {
            real_time_validation: true,
            auto_save_interval: Some(std::time::Duration::from_secs(30)),
            max_problem_size: 10000,
            default_variable_type: VariableType::Binary,
            supported_formats: vec![
                ExportFormat::Python,
                ExportFormat::Rust,
                ExportFormat::JSON,
                ExportFormat::QUBO,
            ],
            theme: Theme::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theme {
    /// Primary color
    pub primary_color: String,
    /// Secondary color
    pub secondary_color: String,
    /// Background color
    pub background_color: String,
    /// Text color
    pub text_color: String,
    /// Grid settings
    pub grid: GridSettings,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            primary_color: "#007acc".to_string(),
            secondary_color: "#ffa500".to_string(),
            background_color: "#ffffff".to_string(),
            text_color: "#000000".to_string(),
            grid: GridSettings::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSettings {
    /// Show grid
    pub enabled: bool,
    /// Grid size
    pub size: usize,
    /// Grid color
    pub color: String,
    /// Snap to grid
    pub snap: bool,
}

impl Default for GridSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            size: 20,
            color: "#e0e0e0".to_string(),
            snap: true,
        }
    }
}

/// Visual representation of a problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualProblem {
    /// Problem metadata
    pub metadata: ProblemMetadata,
    /// Variables in the problem
    pub variables: Vec<VisualVariable>,
    /// Constraints
    pub constraints: Vec<VisualConstraint>,
    /// Objective function
    pub objective: Option<VisualObjective>,
    /// Visual layout
    pub layout: ProblemLayout,
    /// Problem state
    pub state: ProblemState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemMetadata {
    /// Problem name
    pub name: String,
    /// Description
    pub description: String,
    /// Author
    pub author: String,
    /// Creation timestamp
    pub created: std::time::SystemTime,
    /// Last modified
    pub modified: std::time::SystemTime,
    /// Problem category
    pub category: ProblemCategory,
    /// Tags
    pub tags: Vec<String>,
    /// Version
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProblemCategory {
    /// Optimization problem
    Optimization,
    /// Decision problem
    Decision,
    /// Constraint satisfaction
    ConstraintSatisfaction,
    /// Scheduling
    Scheduling,
    /// Routing
    Routing,
    /// Finance
    Finance,
    /// Machine learning
    MachineLearning,
    /// Custom category
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualVariable {
    /// Variable ID
    pub id: String,
    /// Variable name
    pub name: String,
    /// Variable type
    pub var_type: VariableType,
    /// Domain
    pub domain: VariableDomain,
    /// Position in visual layout
    pub position: Position,
    /// Visual properties
    pub visual_properties: VariableVisualProperties,
    /// Description
    pub description: String,
    /// Groups/categories
    pub groups: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableType {
    /// Binary variable (0 or 1)
    Binary,
    /// Integer variable
    Integer { min: i64, max: i64 },
    /// Real variable
    Real { min: f64, max: f64 },
    /// Categorical variable
    Categorical { options: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableDomain {
    /// Binary domain
    Binary,
    /// Integer range
    IntegerRange { min: i64, max: i64 },
    /// Real range
    RealRange { min: f64, max: f64 },
    /// Discrete set
    Discrete { values: Vec<String> },
    /// Custom domain
    Custom { specification: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
    /// Z coordinate (for 3D layouts)
    pub z: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableVisualProperties {
    /// Color
    pub color: String,
    /// Size
    pub size: f64,
    /// Shape
    pub shape: VariableShape,
    /// Visibility
    pub visible: bool,
    /// Label settings
    pub label: LabelSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableShape {
    Circle,
    Square,
    Triangle,
    Diamond,
    Hexagon,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelSettings {
    /// Show label
    pub show: bool,
    /// Label text
    pub text: Option<String>,
    /// Font size
    pub font_size: f64,
    /// Label position
    pub position: LabelPosition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LabelPosition {
    Top,
    Bottom,
    Left,
    Right,
    Center,
    Custom(Position),
}

/// Visual constraint representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualConstraint {
    /// Constraint ID
    pub id: String,
    /// Constraint name
    pub name: String,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Variables involved
    pub variables: Vec<String>,
    /// Parameters
    pub parameters: HashMap<String, ConstraintParameter>,
    /// Visual representation
    pub visual_properties: ConstraintVisualProperties,
    /// Validation status
    pub validation_status: ValidationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Linear constraint: sum(a_i * x_i) ≤ b
    Linear {
        coefficients: Vec<f64>,
        operator: ComparisonOperator,
        rhs: f64,
    },
    /// Quadratic constraint
    Quadratic {
        matrix: Vec<Vec<f64>>,
        operator: ComparisonOperator,
        rhs: f64,
    },
    /// Logical constraint
    Logical { expression: LogicalExpression },
    /// Cardinality constraint
    Cardinality {
        min: Option<usize>,
        max: Option<usize>,
    },
    /// All different constraint
    AllDifferent,
    /// Custom constraint
    Custom { name: String, expression: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    LessEqual,
    GreaterEqual,
    Equal,
    LessThan,
    GreaterThan,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalExpression {
    And(Vec<String>),
    Or(Vec<String>),
    Not(String),
    Implies(String, String),
    Equivalent(String, String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintParameter {
    Integer(i64),
    Real(f64),
    String(String),
    Boolean(bool),
    Vector(Vec<f64>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintVisualProperties {
    /// Color
    pub color: String,
    /// Line style
    pub line_style: LineStyle,
    /// Thickness
    pub thickness: f64,
    /// Connection points
    pub connections: Vec<Connection>,
    /// Show equation
    pub show_equation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
    DashDot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// From variable
    pub from: String,
    /// To variable
    pub to: String,
    /// Connection type
    pub connection_type: ConnectionType,
    /// Path points
    pub path: Vec<Position>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionType {
    Direct,
    Curved,
    Orthogonal,
    Custom,
}

/// Visual objective function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualObjective {
    /// Objective ID
    pub id: String,
    /// Objective name
    pub name: String,
    /// Objective type
    pub objective_type: ObjectiveType,
    /// Expression
    pub expression: ObjectiveExpression,
    /// Optimization direction
    pub direction: OptimizationDirection,
    /// Visual properties
    pub visual_properties: ObjectiveVisualProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObjectiveType {
    Linear,
    Quadratic,
    Polynomial,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObjectiveExpression {
    Linear {
        coefficients: HashMap<String, f64>,
        constant: f64,
    },
    Quadratic {
        linear_terms: HashMap<String, f64>,
        quadratic_terms: HashMap<(String, String), f64>,
        constant: f64,
    },
    Custom {
        expression: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationDirection {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveVisualProperties {
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Show as heatmap
    pub show_heatmap: bool,
    /// Show contour lines
    pub show_contours: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorScheme {
    Viridis,
    Plasma,
    Inferno,
    Magma,
    Blues,
    Reds,
    Greens,
    Custom(Vec<String>),
}

/// Problem layout information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemLayout {
    /// Layout type
    pub layout_type: LayoutType,
    /// Dimensions
    pub dimensions: Dimensions,
    /// Auto-layout settings
    pub auto_layout: AutoLayoutSettings,
    /// Zoom level
    pub zoom: f64,
    /// View center
    pub center: Position,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutType {
    /// Free-form layout
    FreeForm,
    /// Grid layout
    Grid { rows: usize, cols: usize },
    /// Force-directed layout
    ForceDirected,
    /// Hierarchical layout
    Hierarchical,
    /// Circular layout
    Circular,
    /// Custom layout
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dimensions {
    /// Width
    pub width: f64,
    /// Height
    pub height: f64,
    /// Depth (for 3D)
    pub depth: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoLayoutSettings {
    /// Enable auto-layout
    pub enabled: bool,
    /// Layout algorithm
    pub algorithm: LayoutAlgorithm,
    /// Parameters
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutAlgorithm {
    SpringEmbedder,
    ForceAtlas2,
    Fruchterman,
    Kamada,
    Custom(String),
}

/// Problem state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProblemState {
    /// Problem is being edited
    Editing,
    /// Problem is being validated
    Validating,
    /// Problem is valid
    Valid,
    /// Problem has errors
    Invalid { errors: Vec<ValidationError> },
    /// Problem is being solved
    Solving,
    /// Problem has solution
    Solved { solution: ProblemSolution },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemSolution {
    /// Variable assignments
    pub assignments: HashMap<String, f64>,
    /// Objective value
    pub objective_value: f64,
    /// Solution quality
    pub quality: f64,
    /// Solving time
    pub solve_time: std::time::Duration,
}

/// Validation system
pub struct ProblemValidator {
    /// Validation rules
    rules: Vec<ValidationRule>,
    /// Error collector
    errors: Vec<ValidationError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule type
    pub rule_type: ValidationRuleType,
    /// Severity
    pub severity: ValidationSeverity,
    /// Error message template
    pub message_template: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    /// Check variable count
    VariableCount { min: usize, max: usize },
    /// Check constraint consistency
    ConstraintConsistency,
    /// Check objective function
    ObjectiveFunction,
    /// Check variable dependencies
    VariableDependencies,
    /// Check domain validity
    DomainValidity,
    /// Custom validation
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error ID
    pub id: String,
    /// Error type
    pub error_type: ValidationRuleType,
    /// Severity
    pub severity: ValidationSeverity,
    /// Message
    pub message: String,
    /// Location
    pub location: Option<ErrorLocation>,
    /// Suggestions
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorLocation {
    Variable(String),
    Constraint(String),
    Objective,
    Global,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    Valid,
    Warning,
    Error,
    Unknown,
}

/// Code generation system
pub struct CodeGenerator {
    /// Templates
    templates: HashMap<ExportFormat, CodeTemplate>,
    /// Generation settings
    settings: GenerationSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ExportFormat {
    /// Python code
    Python,
    /// Rust code
    Rust,
    /// JSON representation
    JSON,
    /// QUBO matrix
    QUBO,
    /// MPS format
    MPS,
    /// LP format
    LP,
    /// Custom format
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeTemplate {
    /// Template content
    pub template: String,
    /// Variable placeholders
    pub placeholders: Vec<String>,
    /// Template metadata
    pub metadata: TemplateMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMetadata {
    /// Template name
    pub name: String,
    /// Description
    pub description: String,
    /// Version
    pub version: String,
    /// Author
    pub author: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationSettings {
    /// Include comments
    pub include_comments: bool,
    /// Code style
    pub code_style: CodeStyle,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodeStyle {
    Compact,
    Readable,
    Verbose,
    Custom(HashMap<String, String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Advanced,
    Aggressive,
}

/// Undo/redo system
pub struct ProblemHistory {
    /// History stack
    history: Vec<HistoryEntry>,
    /// Current position
    current: usize,
    /// Maximum history size
    max_size: usize,
}

#[derive(Debug, Clone)]
pub struct HistoryEntry {
    /// Action type
    pub action_type: ActionType,
    /// Timestamp
    pub timestamp: std::time::Instant,
    /// Problem state before action
    pub before_state: VisualProblem,
    /// Problem state after action
    pub after_state: VisualProblem,
    /// Action description
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum ActionType {
    /// Add variable
    AddVariable(String),
    /// Remove variable
    RemoveVariable(String),
    /// Modify variable
    ModifyVariable(String),
    /// Add constraint
    AddConstraint(String),
    /// Remove constraint
    RemoveConstraint(String),
    /// Modify constraint
    ModifyConstraint(String),
    /// Set objective
    SetObjective,
    /// Modify objective
    ModifyObjective,
    /// Layout change
    LayoutChange,
    /// Bulk operation
    BulkOperation(Vec<ActionType>),
}

impl VisualProblemBuilder {
    /// Create new visual problem builder
    pub fn new(config: BuilderConfig) -> Self {
        Self {
            problem: VisualProblem::new(),
            config,
            validator: ProblemValidator::new(),
            generator: CodeGenerator::new(),
            history: ProblemHistory::new(100),
        }
    }

    /// Create new empty problem
    pub fn new_problem(&mut self, name: &str) -> Result<(), String> {
        let problem = VisualProblem {
            metadata: ProblemMetadata {
                name: name.to_string(),
                description: String::new(),
                author: String::new(),
                created: std::time::SystemTime::now(),
                modified: std::time::SystemTime::now(),
                category: ProblemCategory::Optimization,
                tags: Vec::new(),
                version: "1.0.0".to_string(),
            },
            variables: Vec::new(),
            constraints: Vec::new(),
            objective: None,
            layout: ProblemLayout {
                layout_type: LayoutType::FreeForm,
                dimensions: Dimensions {
                    width: 1000.0,
                    height: 800.0,
                    depth: None,
                },
                auto_layout: AutoLayoutSettings {
                    enabled: false,
                    algorithm: LayoutAlgorithm::SpringEmbedder,
                    parameters: HashMap::new(),
                },
                zoom: 1.0,
                center: Position {
                    x: 500.0,
                    y: 400.0,
                    z: None,
                },
            },
            state: ProblemState::Editing,
        };

        let old_problem = self.problem.clone();
        self.problem = problem;
        let current_problem = self.problem.clone();
        self.record_action(
            ActionType::BulkOperation(vec![]),
            &old_problem,
            &current_problem,
            "New problem created",
        );

        Ok(())
    }

    /// Add variable to problem
    pub fn add_variable(
        &mut self,
        name: &str,
        var_type: VariableType,
        position: Position,
    ) -> Result<String, String> {
        // Check if variable already exists
        if self.problem.variables.iter().any(|v| v.name == name) {
            return Err(format!("Variable '{}' already exists", name));
        }

        let id = format!("var_{}", self.problem.variables.len());

        let variable = VisualVariable {
            id: id.clone(),
            name: name.to_string(),
            var_type: var_type.clone(),
            domain: match var_type {
                VariableType::Binary => VariableDomain::Binary,
                VariableType::Integer { min, max } => VariableDomain::IntegerRange { min, max },
                VariableType::Real { min, max } => VariableDomain::RealRange { min, max },
                VariableType::Categorical { options } => {
                    VariableDomain::Discrete { values: options }
                }
            },
            position,
            visual_properties: VariableVisualProperties {
                color: self.config.theme.primary_color.clone(),
                size: 20.0,
                shape: VariableShape::Circle,
                visible: true,
                label: LabelSettings {
                    show: true,
                    text: Some(name.to_string()),
                    font_size: 12.0,
                    position: LabelPosition::Bottom,
                },
            },
            description: String::new(),
            groups: Vec::new(),
        };

        let before_state = self.problem.clone();
        self.problem.variables.push(variable);
        self.problem.metadata.modified = std::time::SystemTime::now();

        let current_problem = self.problem.clone();
        self.record_action(
            ActionType::AddVariable(id.clone()),
            &before_state,
            &current_problem,
            &format!("Added variable '{}'", name),
        );

        if self.config.real_time_validation {
            self.validate_problem()?;
        }

        Ok(id)
    }

    /// Remove variable
    pub fn remove_variable(&mut self, variable_id: &str) -> Result<(), String> {
        let pos = self
            .problem
            .variables
            .iter()
            .position(|v| v.id == variable_id)
            .ok_or_else(|| format!("Variable '{}' not found", variable_id))?;

        let before_state = self.problem.clone();
        let variable = self.problem.variables.remove(pos);

        // Remove variable from constraints
        for constraint in &mut self.problem.constraints {
            constraint.variables.retain(|v| v != &variable.id);
        }

        // Remove empty constraints
        self.problem.constraints.retain(|c| !c.variables.is_empty());

        self.problem.metadata.modified = std::time::SystemTime::now();

        let current_problem = self.problem.clone();
        self.record_action(
            ActionType::RemoveVariable(variable_id.to_string()),
            &before_state,
            &current_problem,
            &format!("Removed variable '{}'", variable.name),
        );

        if self.config.real_time_validation {
            self.validate_problem()?;
        }

        Ok(())
    }

    /// Add constraint
    pub fn add_constraint(
        &mut self,
        name: &str,
        constraint_type: ConstraintType,
        variables: Vec<String>,
    ) -> Result<String, String> {
        // Validate that all variables exist
        for var_id in &variables {
            if !self.problem.variables.iter().any(|v| &v.id == var_id) {
                return Err(format!("Variable '{}' not found", var_id));
            }
        }

        let id = format!("constraint_{}", self.problem.constraints.len());

        let constraint = VisualConstraint {
            id: id.clone(),
            name: name.to_string(),
            constraint_type,
            variables,
            parameters: HashMap::new(),
            visual_properties: ConstraintVisualProperties {
                color: self.config.theme.secondary_color.clone(),
                line_style: LineStyle::Solid,
                thickness: 2.0,
                connections: Vec::new(),
                show_equation: true,
            },
            validation_status: ValidationStatus::Unknown,
        };

        let before_state = self.problem.clone();
        self.problem.constraints.push(constraint);
        self.problem.metadata.modified = std::time::SystemTime::now();

        let current_problem = self.problem.clone();
        self.record_action(
            ActionType::AddConstraint(id.clone()),
            &before_state,
            &current_problem,
            &format!("Added constraint '{}'", name),
        );

        if self.config.real_time_validation {
            self.validate_problem()?;
        }

        Ok(id)
    }

    /// Set objective function
    pub fn set_objective(
        &mut self,
        name: &str,
        expression: ObjectiveExpression,
        direction: OptimizationDirection,
    ) -> Result<(), String> {
        let mut objective = VisualObjective {
            id: "objective_0".to_string(),
            name: name.to_string(),
            objective_type: match &expression {
                ObjectiveExpression::Linear { .. } => ObjectiveType::Linear,
                ObjectiveExpression::Quadratic { .. } => ObjectiveType::Quadratic,
                ObjectiveExpression::Custom { .. } => ObjectiveType::Custom,
            },
            expression,
            direction,
            visual_properties: ObjectiveVisualProperties {
                color_scheme: ColorScheme::Viridis,
                show_heatmap: false,
                show_contours: false,
            },
        };

        let before_state = self.problem.clone();
        self.problem.objective = Some(objective);
        self.problem.metadata.modified = std::time::SystemTime::now();

        let current_problem = self.problem.clone();
        self.record_action(
            ActionType::SetObjective,
            &before_state,
            &current_problem,
            &format!("Set objective function '{}'", name),
        );

        if self.config.real_time_validation {
            self.validate_problem()?;
        }

        Ok(())
    }

    /// Auto-layout variables
    pub fn auto_layout(&mut self, algorithm: LayoutAlgorithm) -> Result<(), String> {
        let before_state = self.problem.clone();

        match algorithm {
            LayoutAlgorithm::SpringEmbedder => {
                self.apply_spring_layout()?;
            }
            LayoutAlgorithm::ForceAtlas2 => {
                self.apply_force_atlas_layout()?;
            }
            LayoutAlgorithm::Fruchterman => {
                self.apply_fruchterman_layout()?;
            }
            _ => return Err("Layout algorithm not implemented".to_string()),
        }

        self.problem.metadata.modified = std::time::SystemTime::now();

        let current_problem = self.problem.clone();
        self.record_action(
            ActionType::LayoutChange,
            &before_state,
            &current_problem,
            &format!("Applied {:?} layout", algorithm),
        );

        Ok(())
    }

    /// Validate current problem
    pub fn validate_problem(&mut self) -> Result<(), String> {
        self.problem.state = ProblemState::Validating;

        let errors = self.validator.validate(&self.problem)?;

        if errors.is_empty() {
            self.problem.state = ProblemState::Valid;
        } else {
            self.problem.state = ProblemState::Invalid { errors };
        }

        Ok(())
    }

    /// Generate code in specified format
    pub fn generate_code(&self, format: ExportFormat) -> Result<String, String> {
        self.generator.generate(&self.problem, format)
    }

    /// Export problem as QUBO matrix
    // TODO: Fix QUBO export - requires proper access to compiler internals
    // pub fn export_qubo(&self) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
    //     // Convert visual problem to DSL AST
    //     let ast = self.build_ast()?;
    //
    //     // Use existing compiler to generate QUBO
    //     let mut compiler = Compiler::new(ast);
    //     compiler.generate_qubo()
    // }
    /// Undo last action
    pub fn undo(&mut self) -> Result<(), String> {
        if let Some(entry) = self.history.undo() {
            self.problem = entry.before_state.clone();
            Ok(())
        } else {
            Err("Nothing to undo".to_string())
        }
    }

    /// Redo last undone action
    pub fn redo(&mut self) -> Result<(), String> {
        if let Some(entry) = self.history.redo() {
            self.problem = entry.after_state.clone();
            Ok(())
        } else {
            Err("Nothing to redo".to_string())
        }
    }

    /// Get current problem
    pub fn get_problem(&self) -> &VisualProblem {
        &self.problem
    }

    /// Load problem from JSON
    pub fn load_problem(&mut self, json: &str) -> Result<(), String> {
        let problem: VisualProblem =
            serde_json::from_str(json).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let before_state = self.problem.clone();
        let old_problem = self.problem.clone();
        self.problem = problem;

        let current_problem = self.problem.clone();
        self.record_action(
            ActionType::BulkOperation(vec![]),
            &before_state,
            &current_problem,
            "Loaded problem from JSON",
        );

        Ok(())
    }

    /// Save problem to JSON
    pub fn save_problem(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&self.problem)
            .map_err(|e| format!("Failed to serialize to JSON: {}", e))
    }

    // TODO: Fix AST building - requires proper access to compiler internals
    // /// Convert visual problem to DSL AST
    // fn build_ast(&self) -> Result<AST, String> {
    //     let mut variables = Vec::new();
    //     let mut constraints = Vec::new();
    //
    //     // Convert variables
    //     for var in &self.problem.variables {
    //         let domain = match &var.domain {
    //             VariableDomain::Binary => crate::problem_dsl::compiler::VariableDomain::Binary,
    //             VariableDomain::IntegerRange { min, max } => {
    //                 crate::problem_dsl::compiler::VariableDomain::Integer { min: *min, max: *max }
    //             }
    //             VariableDomain::RealRange { min, max } => {
    //                 crate::problem_dsl::compiler::VariableDomain::Real { min: *min, max: *max }
    //             }
    //             _ => return Err("Unsupported variable domain".to_string()),
    //         };
    //
    //         // TODO: Fix Variable type construction
    //         // variables.push(Variable {
    //         //     name: var.name.clone(),
    //         //     domain,
    //         //     indexed: false,
    //         //     indices: Vec::new(),
    //         // });
    //     }
    //
    //     // Convert constraints
    //     for constraint in &self.problem.constraints {
    //         // Simplified constraint conversion
    //         constraints.push(DslConstraint {
    //             name: constraint.name.clone(),
    //             expression: Expression::Literal(0.0), // Placeholder
    //             penalty: 1.0,
    //         });
    //     }
    //
    //     // Build objective
    //     let mut objective = if let Some(obj) = &self.problem.objective {
    //         match &obj.expression {
    //             ObjectiveExpression::Linear { coefficients, constant } => {
    //                 // Build linear expression
    //                 Expression::Literal(*constant)
    //             }
    //             _ => Expression::Literal(0.0),
    //         }
    //     } else {
    //         Expression::Literal(0.0)
    //     };
    //
    //     Ok(AST {
    //         variables,
    //         constraints,
    //         objective,
    //     })
    // }

    /// Record action in history
    fn record_action(
        &mut self,
        action_type: ActionType,
        before: &VisualProblem,
        after: &VisualProblem,
        description: &str,
    ) {
        let entry = HistoryEntry {
            action_type,
            timestamp: std::time::Instant::now(),
            before_state: before.clone(),
            after_state: after.clone(),
            description: description.to_string(),
        };

        self.history.push(entry);
    }

    /// Apply spring-embedder layout
    fn apply_spring_layout(&mut self) -> Result<(), String> {
        let n = self.problem.variables.len();
        if n == 0 {
            return Ok(());
        }

        // Simple spring-embedder algorithm
        let mut positions: Vec<(f64, f64)> = self
            .problem
            .variables
            .iter()
            .map(|v| (v.position.x, v.position.y))
            .collect();

        for _ in 0..100 {
            let mut forces = vec![(0.0, 0.0); n];

            // Repulsive forces
            for i in 0..n {
                for j in i + 1..n {
                    let dx = positions[i].0 - positions[j].0;
                    let dy = positions[i].1 - positions[j].1;
                    let dist = (dx * dx + dy * dy).sqrt().max(1.0);

                    let force = 1000.0 / (dist * dist);
                    let fx = force * dx / dist;
                    let fy = force * dy / dist;

                    forces[i].0 += fx;
                    forces[i].1 += fy;
                    forces[j].0 -= fx;
                    forces[j].1 -= fy;
                }
            }

            // Attractive forces (simplified)
            for constraint in &self.problem.constraints {
                for i in 0..constraint.variables.len() {
                    for j in i + 1..constraint.variables.len() {
                        if let (Some(idx1), Some(idx2)) = (
                            self.problem
                                .variables
                                .iter()
                                .position(|v| v.id == constraint.variables[i]),
                            self.problem
                                .variables
                                .iter()
                                .position(|v| v.id == constraint.variables[j]),
                        ) {
                            let dx = positions[idx1].0 - positions[idx2].0;
                            let dy = positions[idx1].1 - positions[idx2].1;
                            let dist = (dx * dx + dy * dy).sqrt().max(1.0);

                            let mut force = 0.1 * dist;
                            let fx = force * dx / dist;
                            let fy = force * dy / dist;

                            forces[idx1].0 -= fx;
                            forces[idx1].1 -= fy;
                            forces[idx2].0 += fx;
                            forces[idx2].1 += fy;
                        }
                    }
                }
            }

            // Update positions
            for i in 0..n {
                positions[i].0 += forces[i].0 * 0.01;
                positions[i].1 += forces[i].1 * 0.01;
            }
        }

        // Update variable positions
        for (i, var) in self.problem.variables.iter_mut().enumerate() {
            var.position.x = positions[i].0;
            var.position.y = positions[i].1;
        }

        Ok(())
    }

    /// Apply Force Atlas 2 layout
    fn apply_force_atlas_layout(&mut self) -> Result<(), String> {
        // Simplified Force Atlas 2
        self.apply_spring_layout()
    }

    /// Apply Fruchterman-Reingold layout
    fn apply_fruchterman_layout(&mut self) -> Result<(), String> {
        // Simplified Fruchterman-Reingold
        self.apply_spring_layout()
    }
}

impl VisualProblem {
    /// Create new empty problem
    pub fn new() -> Self {
        Self {
            metadata: ProblemMetadata {
                name: "Untitled".to_string(),
                description: String::new(),
                author: String::new(),
                created: std::time::SystemTime::now(),
                modified: std::time::SystemTime::now(),
                category: ProblemCategory::Optimization,
                tags: Vec::new(),
                version: "1.0.0".to_string(),
            },
            variables: Vec::new(),
            constraints: Vec::new(),
            objective: None,
            layout: ProblemLayout {
                layout_type: LayoutType::FreeForm,
                dimensions: Dimensions {
                    width: 1000.0,
                    height: 800.0,
                    depth: None,
                },
                auto_layout: AutoLayoutSettings {
                    enabled: false,
                    algorithm: LayoutAlgorithm::SpringEmbedder,
                    parameters: HashMap::new(),
                },
                zoom: 1.0,
                center: Position {
                    x: 500.0,
                    y: 400.0,
                    z: None,
                },
            },
            state: ProblemState::Editing,
        }
    }
}

impl ProblemValidator {
    /// Create new validator
    pub fn new() -> Self {
        Self {
            rules: Self::default_rules(),
            errors: Vec::new(),
        }
    }

    /// Get default validation rules
    fn default_rules() -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                name: "Variable count".to_string(),
                rule_type: ValidationRuleType::VariableCount { min: 1, max: 10000 },
                severity: ValidationSeverity::Error,
                message_template: "Problem must have between {min} and {max} variables".to_string(),
            },
            ValidationRule {
                name: "Objective function".to_string(),
                rule_type: ValidationRuleType::ObjectiveFunction,
                severity: ValidationSeverity::Warning,
                message_template: "Problem should have an objective function".to_string(),
            },
        ]
    }

    /// Validate problem
    pub fn validate(&mut self, problem: &VisualProblem) -> Result<Vec<ValidationError>, String> {
        self.errors.clear();

        for rule in &self.rules {
            match &rule.rule_type {
                ValidationRuleType::VariableCount { min, max } => {
                    let count = problem.variables.len();
                    if count < *min || count > *max {
                        self.errors.push(ValidationError {
                            id: format!("var_count_{}", count),
                            error_type: rule.rule_type.clone(),
                            severity: rule.severity.clone(),
                            message: rule
                                .message_template
                                .replace("{min}", &min.to_string())
                                .replace("{max}", &max.to_string()),
                            location: Some(ErrorLocation::Global),
                            suggestions: vec![
                                "Add more variables".to_string(),
                                "Remove unnecessary variables".to_string(),
                            ],
                        });
                    }
                }
                ValidationRuleType::ObjectiveFunction => {
                    if problem.objective.is_none() {
                        self.errors.push(ValidationError {
                            id: "missing_objective".to_string(),
                            error_type: rule.rule_type.clone(),
                            severity: rule.severity.clone(),
                            message: rule.message_template.clone(),
                            location: Some(ErrorLocation::Objective),
                            suggestions: vec!["Add an objective function".to_string()],
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(self.errors.clone())
    }
}

impl CodeGenerator {
    /// Create new code generator
    pub fn new() -> Self {
        Self {
            templates: Self::default_templates(),
            settings: GenerationSettings {
                include_comments: true,
                code_style: CodeStyle::Readable,
                optimization_level: OptimizationLevel::Basic,
            },
        }
    }

    /// Get default templates
    fn default_templates() -> HashMap<ExportFormat, CodeTemplate> {
        let mut templates = HashMap::new();

        templates.insert(
            ExportFormat::Python,
            CodeTemplate {
                template: r#"
# Generated quantum optimization problem
import numpy as np
from quantrs2_tytan import *

# Variables
{variables}

# Objective function
{objective}

# Constraints
{constraints}

# Build and solve
{solve_code}
"#
                .to_string(),
                placeholders: vec![
                    "variables".to_string(),
                    "objective".to_string(),
                    "constraints".to_string(),
                    "solve_code".to_string(),
                ],
                metadata: TemplateMetadata {
                    name: "Python Template".to_string(),
                    description: "Generate Python code using quantrs2-tytan".to_string(),
                    version: "1.0.0".to_string(),
                    author: "QuantRS2".to_string(),
                },
            },
        );

        templates.insert(
            ExportFormat::Rust,
            CodeTemplate {
                template: r#"
// Generated quantum optimization problem
use quantrs2_tytan::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {{
    // Variables
    {variables}
    
    // Objective function
    {objective}
    
    // Constraints
    {constraints}
    
    // Build and solve
    {solve_code}
    
    Ok(())
}}
"#
                .to_string(),
                placeholders: vec![
                    "variables".to_string(),
                    "objective".to_string(),
                    "constraints".to_string(),
                    "solve_code".to_string(),
                ],
                metadata: TemplateMetadata {
                    name: "Rust Template".to_string(),
                    description: "Generate Rust code using quantrs2-tytan".to_string(),
                    version: "1.0.0".to_string(),
                    author: "QuantRS2".to_string(),
                },
            },
        );

        templates
    }

    /// Generate code
    pub fn generate(
        &self,
        problem: &VisualProblem,
        format: ExportFormat,
    ) -> Result<String, String> {
        match format {
            ExportFormat::JSON => serde_json::to_string_pretty(problem)
                .map_err(|e| format!("JSON generation error: {}", e)),
            ExportFormat::Python => self.generate_python_code(problem),
            ExportFormat::Rust => self.generate_rust_code(problem),
            _ => Err("Format not supported yet".to_string()),
        }
    }

    /// Generate Python code
    fn generate_python_code(&self, problem: &VisualProblem) -> Result<String, String> {
        let template = self
            .templates
            .get(&ExportFormat::Python)
            .ok_or("Python template not found")?;

        let mut code = template.template.clone();

        // Generate variables
        let variables_code = problem
            .variables
            .iter()
            .map(|v| format!("{} = symbols(\"{}\")", v.name, v.name))
            .collect::<Vec<_>>()
            .join("\n");

        // Generate objective
        let objective_code = if let Some(obj) = &problem.objective {
            match &obj.expression {
                ObjectiveExpression::Linear {
                    coefficients,
                    constant,
                } => {
                    let terms: Vec<String> = coefficients
                        .iter()
                        .map(|(var, coef)| {
                            if *coef == 1.0 {
                                var.clone()
                            } else {
                                format!("{} * {}", coef, var)
                            }
                        })
                        .collect();

                    if *constant != 0.0 {
                        format!("h = {} + {}", terms.join(" + "), constant)
                    } else {
                        format!("h = {}", terms.join(" + "))
                    }
                }
                _ => "h = 0  # Complex objective not implemented".to_string(),
            }
        } else {
            "h = 0  # No objective function".to_string()
        };

        // Generate constraints
        let constraints_code = problem
            .constraints
            .iter()
            .map(|c| format!("# Constraint: {}", c.name))
            .collect::<Vec<_>>()
            .join("\n");

        // Generate solve code
        let solve_code = r#"
# Compile to QUBO
qubo, offset = Compile(h).get_qubo()

# Choose solver
solver = SASampler()

# Solve
result = solver.run_qubo(qubo, 100)

# Display results
for r in result:
    print(r)
"#
        .to_string();

        code = code.replace("{variables}", &variables_code);
        code = code.replace("{objective}", &objective_code);
        code = code.replace("{constraints}", &constraints_code);
        code = code.replace("{solve_code}", &solve_code);

        Ok(code)
    }

    /// Generate Rust code
    fn generate_rust_code(&self, problem: &VisualProblem) -> Result<String, String> {
        let template = self
            .templates
            .get(&ExportFormat::Rust)
            .ok_or("Rust template not found")?;

        let mut code = template.template.clone();

        // Generate variables
        let variables_code = problem
            .variables
            .iter()
            .map(|v| format!("    let {} = symbols(\"{}\");", v.name, v.name))
            .collect::<Vec<_>>()
            .join("\n");

        // Generate objective (simplified)
        let objective_code = "    let h = x; // Simplified objective".to_string();

        // Generate constraints
        let constraints_code = "    // Constraints not implemented in template".to_string();

        // Generate solve code
        let solve_code = r#"    
    // Compile to QUBO
    let (qubo, offset) = Compile::new(&h).get_qubo()?;
    
    // Choose solver
    let solver = SASampler::new(None);
    
    // Solve
    let mut result = solver.run_qubo(&qubo, 100)?;
    
    // Display results
    for r in &result {
        println!("{:?}", r);
    }"#
        .to_string();

        code = code.replace("{variables}", &variables_code);
        code = code.replace("{objective}", &objective_code);
        code = code.replace("{constraints}", &constraints_code);
        code = code.replace("{solve_code}", &solve_code);

        Ok(code)
    }
}

impl ProblemHistory {
    /// Create new history
    pub fn new(max_size: usize) -> Self {
        Self {
            history: Vec::new(),
            current: 0,
            max_size,
        }
    }

    /// Push new entry
    pub fn push(&mut self, entry: HistoryEntry) {
        // Remove entries after current position
        self.history.truncate(self.current);

        // Add new entry
        self.history.push(entry);
        self.current = self.history.len();

        // Trim if too large
        if self.history.len() > self.max_size {
            self.history.remove(0);
            self.current -= 1;
        }
    }

    /// Undo operation
    pub fn undo(&mut self) -> Option<&HistoryEntry> {
        if self.current > 0 {
            self.current -= 1;
            self.history.get(self.current)
        } else {
            None
        }
    }

    /// Redo operation
    pub fn redo(&mut self) -> Option<&HistoryEntry> {
        if self.current < self.history.len() {
            let entry = self.history.get(self.current);
            self.current += 1;
            entry
        } else {
            None
        }
    }
}

impl fmt::Display for VisualProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Problem '{}': {} variables, {} constraints",
            self.metadata.name,
            self.variables.len(),
            self.constraints.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visual_problem_builder() {
        let mut config = BuilderConfig::default();
        let mut builder = VisualProblemBuilder::new(config);

        // Create new problem
        builder.new_problem("Test Problem").unwrap();

        // Add variables
        let var1_id = builder
            .add_variable(
                "x1",
                VariableType::Binary,
                Position {
                    x: 100.0,
                    y: 100.0,
                    z: None,
                },
            )
            .unwrap();

        let var2_id = builder
            .add_variable(
                "x2",
                VariableType::Binary,
                Position {
                    x: 200.0,
                    y: 100.0,
                    z: None,
                },
            )
            .unwrap();

        assert_eq!(builder.problem.variables.len(), 2);

        // Add constraint
        let constraint_id = builder
            .add_constraint(
                "Sum constraint",
                ConstraintType::Linear {
                    coefficients: vec![1.0, 1.0],
                    operator: ComparisonOperator::LessEqual,
                    rhs: 1.0,
                },
                vec![var1_id.clone(), var2_id.clone()],
            )
            .unwrap();

        assert_eq!(builder.problem.constraints.len(), 1);

        // Set objective
        let mut coefficients = HashMap::new();
        coefficients.insert(var1_id.clone(), 1.0);
        coefficients.insert(var2_id.clone(), 2.0);

        builder
            .set_objective(
                "Linear objective",
                ObjectiveExpression::Linear {
                    coefficients,
                    constant: 0.0,
                },
                OptimizationDirection::Maximize,
            )
            .unwrap();

        assert!(builder.problem.objective.is_some());

        // Test undo/redo
        builder.undo().unwrap();
        assert!(builder.problem.objective.is_none());

        builder.redo().unwrap();
        assert!(builder.problem.objective.is_some());

        // Test code generation
        let python_code = builder.generate_code(ExportFormat::Python).unwrap();
        assert!(python_code.contains("symbols"));
        assert!(python_code.contains("SASampler"));

        // Test JSON export
        let mut json = builder.save_problem().unwrap();
        assert!(json.contains("Test Problem"));
    }

    #[test]
    fn test_validation() {
        let mut validator = ProblemValidator::new();
        let mut problem = VisualProblem::new();

        // Empty problem should have errors
        let errors = validator.validate(&problem).unwrap();
        assert!(!errors.is_empty());

        // Add variables
        problem.variables.push(VisualVariable {
            id: "var1".to_string(),
            name: "x1".to_string(),
            var_type: VariableType::Binary,
            domain: VariableDomain::Binary,
            position: Position {
                x: 0.0,
                y: 0.0,
                z: None,
            },
            visual_properties: VariableVisualProperties {
                color: "#000000".to_string(),
                size: 10.0,
                shape: VariableShape::Circle,
                visible: true,
                label: LabelSettings {
                    show: true,
                    text: None,
                    font_size: 12.0,
                    position: LabelPosition::Bottom,
                },
            },
            description: String::new(),
            groups: Vec::new(),
        });

        // Problem with variables but no objective should have warnings
        let errors = validator.validate(&problem).unwrap();
        assert!(errors
            .iter()
            .any(|e| matches!(e.severity, ValidationSeverity::Warning)));
    }
}
