//! Topological quantum circuit support
//!
//! This module provides support for topological quantum computation,
//! including anyonic braiding operations, fusion rules, and
//! topologically protected quantum gates.

use crate::builder::Circuit;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;

/// Types of anyons supported in topological quantum computation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnyonType {
    /// Vacuum (trivial anyon)
    Vacuum,
    /// Ising anyons (σ)
    Ising,
    /// Fibonacci anyons (τ)
    Fibonacci,
    /// Majorana fermions
    Majorana,
    /// Non-Abelian SU(2) anyons
    SU2 { level: usize },
    /// Parafermions
    Parafermion { n: usize },
    /// Custom anyon type
    Custom {
        name: String,
        quantum_dimension: f64,
        fusion_rules: Vec<FusionRule>,
    },
}

impl AnyonType {
    /// Get the quantum dimension of the anyon
    pub fn quantum_dimension(&self) -> f64 {
        match self {
            AnyonType::Vacuum => 1.0,
            AnyonType::Ising => (2.0_f64).sqrt(),
            AnyonType::Fibonacci => (1.0 + 5.0_f64.sqrt()) / 2.0, // Golden ratio
            AnyonType::Majorana => (2.0_f64).sqrt(),
            AnyonType::SU2 { level } => (*level as f64 + 1.0).sqrt(),
            AnyonType::Parafermion { n } => (2.0 * (*n as f64)).sqrt(),
            AnyonType::Custom {
                quantum_dimension, ..
            } => *quantum_dimension,
        }
    }

    /// Check if this anyon type is Abelian
    pub fn is_abelian(&self) -> bool {
        matches!(self, AnyonType::Vacuum)
    }

    /// Check if this anyon type supports universal quantum computation
    pub fn is_universal(&self) -> bool {
        match self {
            AnyonType::Fibonacci => true,
            AnyonType::SU2 { level } => *level >= 3,
            AnyonType::Parafermion { n } => *n >= 3,
            _ => false,
        }
    }
}

/// Fusion rules for anyons
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionRule {
    /// Input anyon types
    pub inputs: Vec<AnyonType>,
    /// Possible output anyon types with multiplicities
    pub outputs: Vec<(AnyonType, usize)>,
}

/// Anyon representation in the topological circuit
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Anyon {
    /// Unique identifier
    pub id: usize,
    /// Anyon type
    pub anyon_type: AnyonType,
    /// Position in 2D space (for braiding)
    pub position: (f64, f64),
    /// Current charge state
    pub charge: Option<usize>,
}

impl Anyon {
    /// Create a new anyon
    pub fn new(id: usize, anyon_type: AnyonType, position: (f64, f64)) -> Self {
        Self {
            id,
            anyon_type,
            position,
            charge: None,
        }
    }

    /// Check if this anyon can fuse with another
    pub fn can_fuse_with(&self, other: &Anyon) -> bool {
        // For now, allow fusion between same types or with vacuum
        self.anyon_type == other.anyon_type
            || self.anyon_type == AnyonType::Vacuum
            || other.anyon_type == AnyonType::Vacuum
    }
}

/// Braiding operation between anyons
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BraidingOperation {
    /// Anyons involved in the braiding
    pub anyons: Vec<usize>,
    /// Braiding type (clockwise or counterclockwise)
    pub braiding_type: BraidingType,
    /// Number of braiding operations
    pub braiding_count: usize,
    /// Resulting phase or unitary matrix
    pub phase: Option<f64>,
}

/// Types of braiding operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BraidingType {
    /// Clockwise braiding
    Clockwise,
    /// Counterclockwise braiding
    Counterclockwise,
    /// Exchange (swap positions)
    Exchange,
    /// Yang-Baxter move
    YangBaxter,
}

/// Topological gate operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TopologicalGate {
    /// Braiding operation
    Braiding {
        anyon1: usize,
        anyon2: usize,
        braiding_type: BraidingType,
    },
    /// Fusion operation
    Fusion {
        input_anyons: Vec<usize>,
        output_anyon: usize,
        fusion_channel: usize,
    },
    /// Splitting operation (inverse of fusion)
    Splitting {
        input_anyon: usize,
        output_anyons: Vec<usize>,
        splitting_channel: usize,
    },
    /// Measurement in fusion basis
    FusionMeasurement {
        anyons: Vec<usize>,
        measurement_basis: FusionBasis,
    },
    /// Anyon creation
    Creation {
        anyon_type: AnyonType,
        position: (f64, f64),
    },
    /// Anyon annihilation
    Annihilation { anyon: usize },
}

/// Fusion measurement basis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FusionBasis {
    /// Vacuum channel
    Vacuum,
    /// Non-trivial channel for non-Abelian anyons
    NonTrivial { channel: usize },
}

/// Topological quantum circuit
#[derive(Debug, Clone)]
pub struct TopologicalCircuit {
    /// Anyons in the circuit
    pub anyons: HashMap<usize, Anyon>,
    /// Topological gates
    pub gates: Vec<TopologicalGate>,
    /// Braiding sequence
    pub braiding_sequence: Vec<BraidingOperation>,
    /// Fusion rules for the anyon model
    pub fusion_rules: Vec<(AnyonType, Vec<FusionRule>)>,
    /// 2D layout for anyon positions
    pub layout: TopologicalLayout,
}

/// 2D layout for topological computation
#[derive(Debug, Clone)]
pub struct TopologicalLayout {
    /// Width of the 2D region
    pub width: f64,
    /// Height of the 2D region
    pub height: f64,
    /// Forbidden regions (holes or obstacles)
    pub forbidden_regions: Vec<Region>,
    /// Braiding paths
    pub braiding_paths: Vec<Path>,
}

/// 2D region
#[derive(Debug, Clone)]
pub struct Region {
    /// Center position
    pub center: (f64, f64),
    /// Radius (for circular regions)
    pub radius: f64,
    /// Shape type
    pub shape: RegionShape,
}

/// Shape types for regions
#[derive(Debug, Clone, PartialEq)]
pub enum RegionShape {
    Circle,
    Rectangle { width: f64, height: f64 },
    Polygon { vertices: Vec<(f64, f64)> },
}

/// Path for braiding operations
#[derive(Debug, Clone)]
pub struct Path {
    /// Waypoints along the path
    pub waypoints: Vec<(f64, f64)>,
    /// Path type
    pub path_type: PathType,
}

/// Types of braiding paths
#[derive(Debug, Clone, PartialEq)]
pub enum PathType {
    /// Straight line
    Straight,
    /// Curved path
    Curved,
    /// Optimized path avoiding obstacles
    Optimized,
}

/// Topological circuit compiler
pub struct TopologicalCompiler {
    /// Default anyon model
    default_anyon_model: AnyonModel,
    /// Braiding optimizer
    braiding_optimizer: BraidingOptimizer,
    /// Layout optimizer
    layout_optimizer: LayoutOptimizer,
}

/// Anyon model configuration
#[derive(Debug, Clone)]
pub struct AnyonModel {
    /// Primary anyon type
    pub primary_anyon: AnyonType,
    /// Supported anyon types
    pub supported_anyons: Vec<AnyonType>,
    /// Fusion rules
    pub fusion_rules: Vec<(AnyonType, Vec<FusionRule>)>,
    /// Braiding phases
    pub braiding_phases: Vec<((AnyonType, AnyonType), f64)>,
}

impl AnyonModel {
    /// Create Fibonacci anyon model
    pub fn fibonacci() -> Self {
        let fusion_rules = vec![(
            AnyonType::Fibonacci,
            vec![FusionRule {
                inputs: vec![AnyonType::Fibonacci, AnyonType::Fibonacci],
                outputs: vec![(AnyonType::Vacuum, 1), (AnyonType::Fibonacci, 1)],
            }],
        )];

        let braiding_phases = vec![(
            (AnyonType::Fibonacci, AnyonType::Fibonacci),
            2.0 * PI / 5.0, // e^(2πi/5)
        )];

        Self {
            primary_anyon: AnyonType::Fibonacci,
            supported_anyons: vec![AnyonType::Vacuum, AnyonType::Fibonacci],
            fusion_rules,
            braiding_phases,
        }
    }

    /// Create Ising anyon model
    pub fn ising() -> Self {
        let fusion_rules = vec![(
            AnyonType::Ising,
            vec![FusionRule {
                inputs: vec![AnyonType::Ising, AnyonType::Ising],
                outputs: vec![
                    (AnyonType::Vacuum, 1),
                    (AnyonType::Majorana, 1), // Using Majorana as fermion
                ],
            }],
        )];

        let braiding_phases = vec![(
            (AnyonType::Ising, AnyonType::Ising),
            PI / 4.0, // e^(πi/4)
        )];

        Self {
            primary_anyon: AnyonType::Ising,
            supported_anyons: vec![AnyonType::Vacuum, AnyonType::Ising, AnyonType::Majorana],
            fusion_rules,
            braiding_phases,
        }
    }
}

/// Braiding optimizer for efficient gate sequences
#[derive(Debug, Clone)]
pub struct BraidingOptimizer {
    /// Optimization strategy
    pub strategy: OptimizationStrategy,
    /// Maximum braiding distance
    pub max_braiding_distance: f64,
    /// Enable path optimization
    pub optimize_paths: bool,
}

/// Optimization strategies for braiding
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationStrategy {
    /// Minimize total braiding length
    MinimizeLength,
    /// Minimize number of crossings
    MinimizeCrossings,
    /// Minimize execution time
    MinimizeTime,
    /// Balanced optimization
    Balanced,
}

/// Layout optimizer for anyon placement
#[derive(Debug, Clone)]
pub struct LayoutOptimizer {
    /// Target layout density
    pub target_density: f64,
    /// Enable automatic spacing
    pub auto_spacing: bool,
    /// Minimum distance between anyons
    pub min_distance: f64,
}

impl TopologicalCompiler {
    /// Create a new topological compiler
    pub fn new(anyon_model: AnyonModel) -> Self {
        Self {
            default_anyon_model: anyon_model,
            braiding_optimizer: BraidingOptimizer {
                strategy: OptimizationStrategy::Balanced,
                max_braiding_distance: 10.0,
                optimize_paths: true,
            },
            layout_optimizer: LayoutOptimizer {
                target_density: 0.5,
                auto_spacing: true,
                min_distance: 1.0,
            },
        }
    }

    /// Compile a standard quantum circuit to topological representation
    pub fn compile_quantum_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<TopologicalCircuit> {
        // Create initial anyons for each logical qubit
        let mut anyons = HashMap::new();
        let mut next_anyon_id = 0;

        // Create anyon pairs for each qubit (encoding in anyon pairs)
        for qubit_id in 0..N {
            let pos_x = qubit_id as f64 * 2.0;

            // Create anyon pair
            let anyon1 = Anyon::new(
                next_anyon_id,
                self.default_anyon_model.primary_anyon.clone(),
                (pos_x, 0.0),
            );
            let anyon2 = Anyon::new(
                next_anyon_id + 1,
                self.default_anyon_model.primary_anyon.clone(),
                (pos_x + 1.0, 0.0),
            );

            anyons.insert(next_anyon_id, anyon1);
            anyons.insert(next_anyon_id + 1, anyon2);
            next_anyon_id += 2;
        }

        // Compile gates to topological operations
        let mut topological_gates = Vec::new();
        let mut braiding_sequence = Vec::new();

        for gate in circuit.gates() {
            let (topo_gates, braidings) = self.compile_gate(gate.as_ref(), &anyons)?;
            topological_gates.extend(topo_gates);
            braiding_sequence.extend(braidings);
        }

        // Create layout
        let layout = self.create_layout(N)?;

        Ok(TopologicalCircuit {
            anyons,
            gates: topological_gates,
            braiding_sequence,
            fusion_rules: self.default_anyon_model.fusion_rules.clone(),
            layout,
        })
    }

    /// Compile a single quantum gate to topological operations
    fn compile_gate(
        &self,
        gate: &dyn GateOp,
        anyons: &HashMap<usize, Anyon>,
    ) -> QuantRS2Result<(Vec<TopologicalGate>, Vec<BraidingOperation>)> {
        let gate_name = gate.name();
        let qubits = gate.qubits();

        match gate_name {
            "H" => self.compile_hadamard_gate(&qubits, anyons),
            "X" => self.compile_pauli_x_gate(&qubits, anyons),
            "Z" => self.compile_pauli_z_gate(&qubits, anyons),
            "CNOT" => self.compile_cnot_gate(&qubits, anyons),
            "T" => self.compile_t_gate(&qubits, anyons),
            _ => Err(QuantRS2Error::InvalidInput(format!(
                "Gate {} not supported in topological compilation",
                gate_name
            ))),
        }
    }

    /// Compile Hadamard gate using braiding
    fn compile_hadamard_gate(
        &self,
        qubits: &[QubitId],
        anyons: &HashMap<usize, Anyon>,
    ) -> QuantRS2Result<(Vec<TopologicalGate>, Vec<BraidingOperation>)> {
        if qubits.len() != 1 {
            return Err(QuantRS2Error::InvalidInput(
                "Hadamard requires one qubit".to_string(),
            ));
        }

        let qubit_id = qubits[0].id() as usize;
        let anyon1_id = qubit_id * 2;
        let anyon2_id = qubit_id * 2 + 1;

        // Hadamard implemented as specific braiding sequence
        let braiding = BraidingOperation {
            anyons: vec![anyon1_id, anyon2_id],
            braiding_type: BraidingType::Clockwise,
            braiding_count: 1,
            phase: Some(PI / 2.0),
        };

        let topo_gate = TopologicalGate::Braiding {
            anyon1: anyon1_id,
            anyon2: anyon2_id,
            braiding_type: BraidingType::Clockwise,
        };

        Ok((vec![topo_gate], vec![braiding]))
    }

    /// Compile Pauli-X gate
    fn compile_pauli_x_gate(
        &self,
        qubits: &[QubitId],
        anyons: &HashMap<usize, Anyon>,
    ) -> QuantRS2Result<(Vec<TopologicalGate>, Vec<BraidingOperation>)> {
        if qubits.len() != 1 {
            return Err(QuantRS2Error::InvalidInput(
                "Pauli-X requires one qubit".to_string(),
            ));
        }

        let qubit_id = qubits[0].id() as usize;
        let anyon1_id = qubit_id * 2;
        let anyon2_id = qubit_id * 2 + 1;

        // Pauli-X as double braiding (full exchange)
        let braiding = BraidingOperation {
            anyons: vec![anyon1_id, anyon2_id],
            braiding_type: BraidingType::Exchange,
            braiding_count: 2,
            phase: Some(PI),
        };

        let topo_gate = TopologicalGate::Braiding {
            anyon1: anyon1_id,
            anyon2: anyon2_id,
            braiding_type: BraidingType::Exchange,
        };

        Ok((vec![topo_gate], vec![braiding]))
    }

    /// Compile Pauli-Z gate
    fn compile_pauli_z_gate(
        &self,
        qubits: &[QubitId],
        anyons: &HashMap<usize, Anyon>,
    ) -> QuantRS2Result<(Vec<TopologicalGate>, Vec<BraidingOperation>)> {
        // For many anyon models, Z gates are trivial or require measurement
        let qubit_id = qubits[0].id() as usize;
        let anyon1_id = qubit_id * 2;
        let anyon2_id = qubit_id * 2 + 1;

        // Z gate might be implemented via measurement and feedback
        let topo_gate = TopologicalGate::FusionMeasurement {
            anyons: vec![anyon1_id, anyon2_id],
            measurement_basis: FusionBasis::Vacuum,
        };

        Ok((vec![topo_gate], vec![]))
    }

    /// Compile CNOT gate using braiding
    fn compile_cnot_gate(
        &self,
        qubits: &[QubitId],
        anyons: &HashMap<usize, Anyon>,
    ) -> QuantRS2Result<(Vec<TopologicalGate>, Vec<BraidingOperation>)> {
        if qubits.len() != 2 {
            return Err(QuantRS2Error::InvalidInput(
                "CNOT requires two qubits".to_string(),
            ));
        }

        let control_id = qubits[0].id() as usize;
        let target_id = qubits[1].id() as usize;

        let control_anyon1 = control_id * 2;
        let control_anyon2 = control_id * 2 + 1;
        let target_anyon1 = target_id * 2;
        let target_anyon2 = target_id * 2 + 1;

        // CNOT requires complex braiding between control and target
        let mut braidings = Vec::new();
        let mut topo_gates = Vec::new();

        // Braiding sequence for CNOT (simplified)
        braidings.push(BraidingOperation {
            anyons: vec![control_anyon1, target_anyon1],
            braiding_type: BraidingType::Clockwise,
            braiding_count: 1,
            phase: Some(PI / 4.0),
        });

        braidings.push(BraidingOperation {
            anyons: vec![control_anyon2, target_anyon2],
            braiding_type: BraidingType::Counterclockwise,
            braiding_count: 1,
            phase: Some(-PI / 4.0),
        });

        // Corresponding topological gates
        topo_gates.push(TopologicalGate::Braiding {
            anyon1: control_anyon1,
            anyon2: target_anyon1,
            braiding_type: BraidingType::Clockwise,
        });

        topo_gates.push(TopologicalGate::Braiding {
            anyon1: control_anyon2,
            anyon2: target_anyon2,
            braiding_type: BraidingType::Counterclockwise,
        });

        Ok((topo_gates, braidings))
    }

    /// Compile T gate (non-Clifford gate requiring special treatment)
    fn compile_t_gate(
        &self,
        qubits: &[QubitId],
        anyons: &HashMap<usize, Anyon>,
    ) -> QuantRS2Result<(Vec<TopologicalGate>, Vec<BraidingOperation>)> {
        if !self.default_anyon_model.primary_anyon.is_universal() {
            return Err(QuantRS2Error::InvalidInput(
                "T gate requires universal anyon model".to_string(),
            ));
        }

        let qubit_id = qubits[0].id() as usize;
        let anyon1_id = qubit_id * 2;
        let anyon2_id = qubit_id * 2 + 1;

        // T gate requires specific braiding sequence for universal models
        let braiding = BraidingOperation {
            anyons: vec![anyon1_id, anyon2_id],
            braiding_type: BraidingType::YangBaxter,
            braiding_count: 1,
            phase: Some(PI / 4.0),
        };

        let topo_gate = TopologicalGate::Braiding {
            anyon1: anyon1_id,
            anyon2: anyon2_id,
            braiding_type: BraidingType::YangBaxter,
        };

        Ok((vec![topo_gate], vec![braiding]))
    }

    /// Create layout for the topological circuit
    fn create_layout(&self, num_qubits: usize) -> QuantRS2Result<TopologicalLayout> {
        let width = (num_qubits as f64 * 2.0 + 2.0).max(10.0);
        let height = 10.0;

        Ok(TopologicalLayout {
            width,
            height,
            forbidden_regions: Vec::new(),
            braiding_paths: Vec::new(),
        })
    }

    /// Optimize braiding sequence
    pub fn optimize_braiding_sequence(
        &self,
        circuit: &mut TopologicalCircuit,
    ) -> QuantRS2Result<OptimizationStats> {
        let initial_length = self.calculate_total_braiding_length(&circuit.braiding_sequence);
        let initial_crossings = self.count_braiding_crossings(&circuit.braiding_sequence);

        // Apply optimization based on strategy
        match self.braiding_optimizer.strategy {
            OptimizationStrategy::MinimizeLength => {
                self.optimize_for_length(&mut circuit.braiding_sequence)?;
            }
            OptimizationStrategy::MinimizeCrossings => {
                self.optimize_for_crossings(&mut circuit.braiding_sequence)?;
            }
            OptimizationStrategy::MinimizeTime => {
                self.optimize_for_time(&mut circuit.braiding_sequence)?;
            }
            OptimizationStrategy::Balanced => {
                self.optimize_balanced(&mut circuit.braiding_sequence)?;
            }
        }

        let final_length = self.calculate_total_braiding_length(&circuit.braiding_sequence);
        let final_crossings = self.count_braiding_crossings(&circuit.braiding_sequence);

        Ok(OptimizationStats {
            initial_length,
            final_length,
            initial_crossings,
            final_crossings,
            length_improvement: (initial_length - final_length) / initial_length,
            crossings_improvement: (initial_crossings as f64 - final_crossings as f64)
                / initial_crossings as f64,
        })
    }

    /// Calculate total braiding length
    fn calculate_total_braiding_length(&self, braidings: &[BraidingOperation]) -> f64 {
        braidings.iter().map(|b| b.braiding_count as f64).sum()
    }

    /// Count braiding crossings
    fn count_braiding_crossings(&self, braidings: &[BraidingOperation]) -> usize {
        // Simplified crossing count
        braidings.len()
    }

    /// Optimize for minimum length
    fn optimize_for_length(&self, braidings: &mut Vec<BraidingOperation>) -> QuantRS2Result<()> {
        // Implement length optimization
        Ok(())
    }

    /// Optimize for minimum crossings
    fn optimize_for_crossings(&self, braidings: &mut Vec<BraidingOperation>) -> QuantRS2Result<()> {
        // Implement crossing optimization
        Ok(())
    }

    /// Optimize for minimum time
    fn optimize_for_time(&self, braidings: &mut Vec<BraidingOperation>) -> QuantRS2Result<()> {
        // Implement time optimization
        Ok(())
    }

    /// Balanced optimization
    fn optimize_balanced(&self, braidings: &mut Vec<BraidingOperation>) -> QuantRS2Result<()> {
        // Implement balanced optimization
        Ok(())
    }
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    pub initial_length: f64,
    pub final_length: f64,
    pub initial_crossings: usize,
    pub final_crossings: usize,
    pub length_improvement: f64,
    pub crossings_improvement: f64,
}

impl TopologicalCircuit {
    /// Calculate the total number of braiding operations
    pub fn total_braiding_operations(&self) -> usize {
        self.braiding_sequence
            .iter()
            .map(|b| b.braiding_count)
            .sum()
    }

    /// Get the number of anyons
    pub fn anyon_count(&self) -> usize {
        self.anyons.len()
    }

    /// Check if the circuit uses universal anyons
    pub fn is_universal(&self) -> bool {
        self.anyons
            .values()
            .any(|anyon| anyon.anyon_type.is_universal())
    }

    /// Calculate circuit depth in terms of braiding layers
    pub fn braiding_depth(&self) -> usize {
        // Simplified depth calculation
        self.braiding_sequence.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::Hadamard;

    #[test]
    fn test_anyon_types() {
        let fibonacci = AnyonType::Fibonacci;
        assert!(fibonacci.is_universal());
        assert!(!fibonacci.is_abelian());
        assert!((fibonacci.quantum_dimension() - 1.618).abs() < 0.01);

        let ising = AnyonType::Ising;
        assert!(!ising.is_universal());
        assert!(!ising.is_abelian());
        assert!((ising.quantum_dimension() - 1.414).abs() < 0.01);
    }

    #[test]
    fn test_anyon_creation() {
        let anyon = Anyon::new(0, AnyonType::Fibonacci, (1.0, 2.0));
        assert_eq!(anyon.id, 0);
        assert_eq!(anyon.position, (1.0, 2.0));
        assert_eq!(anyon.anyon_type, AnyonType::Fibonacci);
    }

    #[test]
    fn test_anyon_model_creation() {
        let fibonacci_model = AnyonModel::fibonacci();
        assert_eq!(fibonacci_model.primary_anyon, AnyonType::Fibonacci);
        assert!(fibonacci_model
            .fusion_rules
            .iter()
            .any(|(anyon_type, _)| *anyon_type == AnyonType::Fibonacci));

        let ising_model = AnyonModel::ising();
        assert_eq!(ising_model.primary_anyon, AnyonType::Ising);
        assert!(ising_model
            .braiding_phases
            .iter()
            .any(|((a1, a2), _)| *a1 == AnyonType::Ising && *a2 == AnyonType::Ising));
    }

    #[test]
    fn test_topological_compiler() {
        let model = AnyonModel::fibonacci();
        let compiler = TopologicalCompiler::new(model);

        assert_eq!(
            compiler.default_anyon_model.primary_anyon,
            AnyonType::Fibonacci
        );
        assert_eq!(
            compiler.braiding_optimizer.strategy,
            OptimizationStrategy::Balanced
        );
    }

    #[test]
    fn test_braiding_operation() {
        let braiding = BraidingOperation {
            anyons: vec![0, 1],
            braiding_type: BraidingType::Clockwise,
            braiding_count: 2,
            phase: Some(PI / 4.0),
        };

        assert_eq!(braiding.anyons.len(), 2);
        assert_eq!(braiding.braiding_count, 2);
        assert_eq!(braiding.braiding_type, BraidingType::Clockwise);
    }

    #[test]
    fn test_topological_gates() {
        let fusion_gate = TopologicalGate::Fusion {
            input_anyons: vec![0, 1],
            output_anyon: 2,
            fusion_channel: 0,
        };

        let braiding_gate = TopologicalGate::Braiding {
            anyon1: 0,
            anyon2: 1,
            braiding_type: BraidingType::Exchange,
        };

        assert!(matches!(fusion_gate, TopologicalGate::Fusion { .. }));
        assert!(matches!(braiding_gate, TopologicalGate::Braiding { .. }));
    }

    #[test]
    fn test_layout_creation() {
        let layout = TopologicalLayout {
            width: 10.0,
            height: 8.0,
            forbidden_regions: vec![],
            braiding_paths: vec![],
        };

        assert_eq!(layout.width, 10.0);
        assert_eq!(layout.height, 8.0);
        assert!(layout.forbidden_regions.is_empty());
    }

    #[test]
    fn test_circuit_properties() {
        let mut anyons = HashMap::new();
        anyons.insert(0, Anyon::new(0, AnyonType::Fibonacci, (0.0, 0.0)));
        anyons.insert(1, Anyon::new(1, AnyonType::Fibonacci, (1.0, 0.0)));

        let circuit = TopologicalCircuit {
            anyons,
            gates: vec![],
            braiding_sequence: vec![BraidingOperation {
                anyons: vec![0, 1],
                braiding_type: BraidingType::Clockwise,
                braiding_count: 3,
                phase: Some(PI / 4.0),
            }],
            fusion_rules: Vec::new(),
            layout: TopologicalLayout {
                width: 5.0,
                height: 5.0,
                forbidden_regions: vec![],
                braiding_paths: vec![],
            },
        };

        assert_eq!(circuit.anyon_count(), 2);
        assert_eq!(circuit.total_braiding_operations(), 3);
        assert!(circuit.is_universal());
        assert_eq!(circuit.braiding_depth(), 1);
    }
}
