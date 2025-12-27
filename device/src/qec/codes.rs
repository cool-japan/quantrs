//! Quantum Error Correction Code Types and Configurations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Types of quantum error correction codes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QECCodeType {
    /// Surface codes
    SurfaceCode {
        distance: usize,
        layout: SurfaceCodeLayout,
    },
    /// Color codes
    ColorCode {
        distance: usize,
        color_type: ColorCodeType,
    },
    /// Topological codes
    TopologicalCode { code_type: TopologicalCodeType },
    /// CSS codes
    CSSCode { stabilizers: Vec<String> },
    /// Steane code
    SteaneCode,
    /// Shor code
    ShorCode,
    /// Repetition code
    RepetitionCode { length: usize },
    /// Custom code
    CustomCode {
        name: String,
        parameters: HashMap<String, f64>,
    },
}

/// Surface code layouts
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurfaceCodeLayout {
    Square,
    Triangular,
    Hexagonal,
    Rotated,
    Custom(String),
}

/// Color code types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorCodeType {
    TriangularLattice,
    HexagonalLattice,
    Custom(String),
}

/// Topological code types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologicalCodeType {
    ToricCode,
    PlanarCode,
    TwistedToricCode,
    Custom(String),
}

/// Surface code configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceCodeConfig {
    /// Code distance
    pub distance: usize,
    /// Layout type
    pub layout: SurfaceCodeLayout,
    /// Physical qubit allocation
    pub qubit_allocation: QubitAllocation,
    /// Stabilizer configuration
    pub stabilizer_config: StabilizerConfig,
    /// Boundary conditions
    pub boundary_conditions: BoundaryConditions,
}

/// Qubit allocation for surface codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QubitAllocation {
    /// Data qubits
    pub data_qubits: Vec<usize>,
    /// Syndrome qubits
    pub syndrome_qubits: Vec<usize>,
    /// Auxiliary qubits
    pub auxiliary_qubits: Vec<usize>,
    /// Spare qubits for replacement
    pub spare_qubits: Vec<usize>,
}

/// Stabilizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilizerConfig {
    /// X stabilizers
    pub x_stabilizers: Vec<Stabilizer>,
    /// Z stabilizers
    pub z_stabilizers: Vec<Stabilizer>,
    /// Measurement schedule
    pub measurement_schedule: MeasurementSchedule,
    /// Error detection threshold
    pub error_threshold: f64,
}

/// Individual stabilizer definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stabilizer {
    /// Qubits involved in the stabilizer
    pub qubits: Vec<usize>,
    /// Pauli operators for each qubit
    pub operators: Vec<PauliOperator>,
    /// Weight of the stabilizer
    pub weight: usize,
    /// Expected eigenvalue
    pub expected_eigenvalue: i8,
}

/// Pauli operators
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PauliOperator {
    I, // Identity
    X, // Pauli-X
    Y, // Pauli-Y
    Z, // Pauli-Z
}

/// Measurement schedule for stabilizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementSchedule {
    /// Stabilizer measurement order
    pub measurement_order: Vec<usize>,
    /// Time between measurements
    pub measurement_interval: std::time::Duration,
    /// Parallel measurement groups
    pub parallel_groups: Vec<Vec<usize>>,
    /// Adaptive scheduling
    pub adaptive_scheduling: bool,
}

/// Boundary conditions for codes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryConditions {
    Open,
    Periodic,
    Twisted,
    Rough,
    Smooth,
}

/// Color code configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorCodeConfig {
    /// Code distance
    pub distance: usize,
    /// Color code type
    pub color_type: ColorCodeType,
    /// Face coloring
    pub face_coloring: FaceColoring,
    /// Vertex operators
    pub vertex_operators: Vec<VertexOperator>,
    /// Face operators
    pub face_operators: Vec<FaceOperator>,
}

/// Face coloring for color codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceColoring {
    /// Color assignments
    pub color_assignments: HashMap<usize, Color>,
    /// Color constraints
    pub constraints: Vec<ColorConstraint>,
}

/// Color types for color codes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Color {
    Red,
    Green,
    Blue,
}

/// Color constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorConstraint {
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Affected faces
    pub faces: Vec<usize>,
    /// Required colors
    pub required_colors: Vec<Color>,
}

/// Types of color constraints
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    AdjacentFaces,
    VertexNeighbors,
    EdgeConnected,
}

/// Vertex operators for color codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexOperator {
    /// Vertex index
    pub vertex: usize,
    /// Connected qubits
    pub qubits: Vec<usize>,
    /// Operator type
    pub operator_type: VertexOperatorType,
}

/// Types of vertex operators
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VertexOperatorType {
    XType,
    ZType,
    YType,
}

/// Face operators for color codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceOperator {
    /// Face index
    pub face: usize,
    /// Face color
    pub color: Color,
    /// Boundary qubits
    pub boundary_qubits: Vec<usize>,
    /// Operator weight
    pub weight: usize,
}

/// Topological code configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalCodeConfig {
    /// Code type
    pub code_type: TopologicalCodeType,
    /// Lattice configuration
    pub lattice: LatticeConfig,
    /// Logical operators
    pub logical_operators: LogicalOperators,
    /// Syndrome extraction
    pub syndrome_extraction: SyndromeExtractionConfig,
}

/// Lattice configuration for topological codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeConfig {
    /// Lattice dimensions
    pub dimensions: (usize, usize),
    /// Periodic boundaries
    pub periodic: bool,
    /// Twisted boundaries
    pub twisted: bool,
    /// Defect locations
    pub defects: Vec<DefectLocation>,
}

/// Defect location in lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefectLocation {
    /// Position
    pub position: (usize, usize),
    /// Defect type
    pub defect_type: DefectType,
}

/// Types of defects in topological codes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DefectType {
    Hole,
    Twist,
    Boundary,
    Custom(String),
}

/// Logical operators for topological codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalOperators {
    /// Logical X operators
    pub logical_x: Vec<LogicalOperator>,
    /// Logical Z operators
    pub logical_z: Vec<LogicalOperator>,
    /// Commutation relations
    pub commutation_relations: CommutationTable,
}

/// Individual logical operator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalOperator {
    /// Operator path
    pub path: Vec<usize>,
    /// Operator type
    pub operator_type: LogicalOperatorType,
    /// Homology class
    pub homology_class: Vec<i32>,
}

/// Types of logical operators
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogicalOperatorType {
    XOperator,
    ZOperator,
    YOperator,
}

/// Commutation table for logical operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommutationTable {
    /// Commutation matrix
    pub matrix: Vec<Vec<i8>>,
    /// Operator labels
    pub labels: Vec<String>,
}

/// Syndrome extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyndromeExtractionConfig {
    /// Extraction method
    pub method: ExtractionMethod,
    /// Measurement circuits
    pub measurement_circuits: Vec<MeasurementCircuit>,
    /// Error correction cycles
    pub correction_cycles: usize,
}

/// Syndrome extraction methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtractionMethod {
    Standard,
    Fast,
    Adaptive,
    FaultTolerant,
}

/// Measurement circuit for syndrome extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCircuit {
    /// Circuit gates
    pub gates: Vec<Gate>,
    /// Measurement qubits
    pub measurement_qubits: Vec<usize>,
    /// Classical registers
    pub classical_registers: Vec<usize>,
}

/// Gate definition for measurement circuits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gate {
    /// Gate type
    pub gate_type: GateType,
    /// Target qubits
    pub targets: Vec<usize>,
    /// Control qubits
    pub controls: Vec<usize>,
    /// Gate parameters
    pub parameters: Vec<f64>,
}

/// Types of gates
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateType {
    X,
    Y,
    Z,
    H,
    S,
    T,
    CNOT,
    CZ,
    Measurement,
    Reset,
}

/// CSS code configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CSSCodeConfig {
    /// Classical codes
    pub classical_codes: ClassicalCodes,
    /// Parity check matrices
    pub parity_matrices: ParityMatrices,
    /// Generator matrices
    pub generator_matrices: GeneratorMatrices,
    /// Code parameters
    pub code_parameters: CSSParameters,
}

/// Classical codes for CSS construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalCodes {
    /// X code
    pub x_code: ClassicalCode,
    /// Z code
    pub z_code: ClassicalCode,
    /// Dual relationship
    pub dual_relationship: DualRelationship,
}

/// Individual classical code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalCode {
    /// Code length
    pub length: usize,
    /// Code dimension
    pub dimension: usize,
    /// Minimum distance
    pub min_distance: usize,
    /// Generator matrix
    pub generator: Vec<Vec<u8>>,
    /// Parity check matrix
    pub parity_check: Vec<Vec<u8>>,
}

/// Dual relationship between classical codes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DualRelationship {
    SelfDual,
    MutuallyDual,
    NonDual,
}

/// Parity check matrices for CSS codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityMatrices {
    /// X parity check matrix
    pub hx_matrix: Vec<Vec<u8>>,
    /// Z parity check matrix
    pub hz_matrix: Vec<Vec<u8>>,
    /// Orthogonality check
    pub orthogonality_verified: bool,
}

/// Generator matrices for CSS codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorMatrices {
    /// X generator matrix
    pub gx_matrix: Vec<Vec<u8>>,
    /// Z generator matrix
    pub gz_matrix: Vec<Vec<u8>>,
    /// Logical operator generators
    pub logical_generators: LogicalGenerators,
}

/// Logical operator generators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalGenerators {
    /// Logical X generators
    pub logical_x_generators: Vec<Vec<u8>>,
    /// Logical Z generators
    pub logical_z_generators: Vec<Vec<u8>>,
    /// Stabilizer generators
    pub stabilizer_generators: Vec<Vec<u8>>,
}

/// CSS code parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CSSParameters {
    /// Number of physical qubits
    pub n: usize,
    /// Number of logical qubits
    pub k: usize,
    /// Code distance
    pub d: usize,
    /// Error correction capability
    pub t: usize,
}

/// Steane code configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteaneCodeConfig {
    /// Standard Steane code parameters
    pub parameters: SteaneParameters,
    /// Encoding circuits
    pub encoding_circuits: Vec<EncodingCircuit>,
    /// Decoding tables
    pub decoding_tables: DecodingTables,
}

/// Steane code parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteaneParameters {
    /// Code is \[7,1,3\]
    pub n: usize, // 7
    pub k: usize, // 1
    pub d: usize, // 3
}

/// Encoding circuit definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingCircuit {
    /// Circuit name
    pub name: String,
    /// Circuit gates
    pub gates: Vec<Gate>,
    /// Input qubits
    pub input_qubits: Vec<usize>,
    /// Output qubits
    pub output_qubits: Vec<usize>,
}

/// Decoding lookup tables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodingTables {
    /// Syndrome to error mapping
    pub syndrome_error_map: HashMap<Vec<u8>, ErrorPattern>,
    /// Recovery operations
    pub recovery_operations: HashMap<Vec<u8>, Vec<Gate>>,
}

/// Error pattern definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    /// Error locations
    pub locations: Vec<usize>,
    /// Error types at each location
    pub error_types: Vec<ErrorType>,
    /// Pattern weight
    pub weight: usize,
}

/// Types of errors
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorType {
    X,
    Y,
    Z,
    Identity,
}

/// Shor code configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShorCodeConfig {
    /// Shor code parameters \[9,1,3\]
    pub parameters: ShorParameters,
    /// Bit flip code structure
    pub bit_flip_structure: BitFlipStructure,
    /// Phase flip code structure
    pub phase_flip_structure: PhaseFlipStructure,
}

/// Shor code parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShorParameters {
    /// Total qubits
    pub n: usize, // 9
    /// Logical qubits
    pub k: usize, // 1
    /// Distance
    pub d: usize, // 3
}

/// Bit flip code structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitFlipStructure {
    /// Code blocks
    pub blocks: Vec<CodeBlock>,
    /// Parity qubits
    pub parity_qubits: Vec<usize>,
}

/// Phase flip code structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseFlipStructure {
    /// Superposition states
    pub superposition_states: Vec<SuperpositionState>,
    /// Phase parity qubits
    pub phase_parity_qubits: Vec<usize>,
}

/// Code block for bit flip protection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeBlock {
    /// Block qubits
    pub qubits: Vec<usize>,
    /// Majority vote qubits
    pub majority_vote_qubits: Vec<usize>,
}

/// Superposition state definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperpositionState {
    /// State coefficients
    pub coefficients: Vec<f64>,
    /// Basis states
    pub basis_states: Vec<String>,
}

/// Repetition code configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepetitionCodeConfig {
    /// Code length
    pub length: usize,
    /// Majority voting
    pub majority_voting: MajorityVoting,
    /// Error threshold
    pub error_threshold: f64,
}

/// Majority voting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MajorityVoting {
    /// Voting strategy
    pub strategy: VotingStrategy,
    /// Confidence threshold
    pub confidence_threshold: f64,
    /// Tie-breaking rule
    pub tie_breaking: TieBreaking,
}

/// Voting strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VotingStrategy {
    Simple,
    Weighted,
    Adaptive,
}

/// Tie-breaking rules
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TieBreaking {
    Random,
    Conservative,
    Aggressive,
    Historical,
}
