//! Materials science applications: Crystal structure prediction and optimization.
//!
//! This module provides quantum optimization tools for materials science
//! including crystal structure prediction, phase transitions, and property optimization.

use crate::sampler::{Sampler, SampleResult, SamplerError, SamplerResult};
use ndarray::{Array, Array1, Array2, Array3, IxDyn};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use rand::prelude::*;

/// Crystal structure predictor
pub struct CrystalStructurePredictor {
    /// Chemical composition
    composition: ChemicalComposition,
    /// Prediction method
    method: PredictionMethod,
    /// Constraints
    constraints: StructureConstraints,
    /// Energy model
    energy_model: EnergyModel,
    /// Search strategy
    search_strategy: SearchStrategy,
}

#[derive(Debug, Clone)]
pub struct ChemicalComposition {
    /// Elements and their counts
    pub elements: HashMap<String, usize>,
    /// Total atoms in unit cell
    pub total_atoms: usize,
    /// Stoichiometry constraints
    pub stoichiometry: Option<Vec<f64>>,
    /// Oxidation states
    pub oxidation_states: Option<HashMap<String, i32>>,
}

#[derive(Debug, Clone)]
pub enum PredictionMethod {
    /// Global optimization
    GlobalOptimization {
        algorithm: GlobalOptAlgorithm,
        max_iterations: usize,
    },
    /// Data mining
    DataMining {
        database: StructureDatabase,
        similarity_threshold: f64,
    },
    /// Evolutionary algorithm
    Evolutionary {
        population_size: usize,
        generations: usize,
        mutation_rate: f64,
    },
    /// Machine learning
    MachineLearning {
        model: MLModel,
        confidence_threshold: f64,
    },
    /// Ab initio random structure searching
    AIRSS {
        num_searches: usize,
        symmetry_constraints: bool,
    },
}

#[derive(Debug, Clone)]
pub enum GlobalOptAlgorithm {
    SimulatedAnnealing,
    BasinHopping,
    ParticleSwarm,
    GeneticAlgorithm,
    MinimumHopping,
}

#[derive(Debug, Clone)]
pub struct StructureDatabase {
    /// Database source
    pub source: DatabaseSource,
    /// Number of structures
    pub size: usize,
    /// Filters applied
    pub filters: Vec<DatabaseFilter>,
}

#[derive(Debug, Clone)]
pub enum DatabaseSource {
    /// Materials Project
    MaterialsProject,
    /// ICSD
    ICSD,
    /// COD
    COD,
    /// Custom database
    Custom { path: String },
}

#[derive(Debug, Clone)]
pub enum DatabaseFilter {
    /// Element filter
    Elements { required: Vec<String>, forbidden: Vec<String> },
    /// Space group filter
    SpaceGroup { allowed: Vec<u32> },
    /// Property filter
    Property { name: String, min: f64, max: f64 },
    /// Stability filter
    Stability { max_above_hull: f64 },
}

#[derive(Debug, Clone)]
pub struct MLModel {
    /// Model type
    pub model_type: MLModelType,
    /// Feature representation
    pub features: FeatureRepresentation,
    /// Training data size
    pub training_size: usize,
    /// Validation accuracy
    pub accuracy: f64,
}

#[derive(Debug, Clone)]
pub enum MLModelType {
    /// Graph neural network
    GraphNN,
    /// Crystal graph CNN
    CGCNN,
    /// SchNet
    SchNet,
    /// MEGNet
    MEGNet,
    /// Gaussian approximation potential
    GAP,
}

#[derive(Debug, Clone)]
pub enum FeatureRepresentation {
    /// Coulomb matrix
    CoulombMatrix,
    /// Sine matrix
    SineMatrix,
    /// Many-body tensor
    ManyBodyTensor { order: usize },
    /// Smooth overlap of atomic positions
    SOAP { cutoff: f64, n_max: usize, l_max: usize },
    /// Crystal graph
    CrystalGraph,
}

#[derive(Debug, Clone)]
pub struct StructureConstraints {
    /// Lattice constraints
    pub lattice: LatticeConstraints,
    /// Symmetry constraints
    pub symmetry: SymmetryConstraints,
    /// Distance constraints
    pub distances: DistanceConstraints,
    /// Coordination constraints
    pub coordination: CoordinationConstraints,
    /// Pressure constraint
    pub pressure: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct LatticeConstraints {
    /// Minimum lattice parameters
    pub min_lengths: Option<Vec3D>,
    /// Maximum lattice parameters
    pub max_lengths: Option<Vec3D>,
    /// Angle constraints
    pub angle_ranges: Option<Vec<(f64, f64)>>,
    /// Volume constraint
    pub volume_range: Option<(f64, f64)>,
    /// Fixed lattice type
    pub lattice_type: Option<LatticeType>,
}

#[derive(Debug, Clone)]
pub enum LatticeType {
    Cubic,
    Tetragonal,
    Orthorhombic,
    Hexagonal,
    Rhombohedral,
    Monoclinic,
    Triclinic,
}

#[derive(Debug, Clone)]
pub struct Vec3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone)]
pub struct SymmetryConstraints {
    /// Space group constraints
    pub space_groups: Option<Vec<u32>>,
    /// Point group constraints
    pub point_groups: Option<Vec<String>>,
    /// Minimum symmetry operations
    pub min_symmetry: Option<usize>,
    /// Wyckoff position constraints
    pub wyckoff_positions: Option<Vec<WyckoffPosition>>,
}

#[derive(Debug, Clone)]
pub struct WyckoffPosition {
    /// Wyckoff letter
    pub letter: char,
    /// Multiplicity
    pub multiplicity: usize,
    /// Site symmetry
    pub site_symmetry: String,
    /// Coordinates
    pub coordinates: Vec<Vec3D>,
}

#[derive(Debug, Clone)]
pub struct DistanceConstraints {
    /// Minimum distances between elements
    pub min_distances: HashMap<(String, String), f64>,
    /// Maximum distances
    pub max_distances: HashMap<(String, String), f64>,
    /// Bond length constraints
    pub bond_lengths: Vec<BondConstraint>,
}

#[derive(Debug, Clone)]
pub struct BondConstraint {
    /// Atom types
    pub atoms: (String, String),
    /// Target length
    pub target_length: f64,
    /// Tolerance
    pub tolerance: f64,
    /// Bond order
    pub bond_order: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct CoordinationConstraints {
    /// Coordination numbers
    pub coordination_numbers: HashMap<String, (usize, usize)>,
    /// Coordination geometry
    pub geometries: HashMap<String, CoordinationGeometry>,
    /// Allowed ligands
    pub allowed_ligands: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
pub enum CoordinationGeometry {
    Linear,
    Trigonal,
    Tetrahedral,
    SquarePlanar,
    TrigonalBipyramidal,
    Octahedral,
    PentagonalBipyramidal,
    CubicCoordination,
    Custom { angles: Vec<f64> },
}

#[derive(Debug, Clone)]
pub enum EnergyModel {
    /// Empirical potentials
    Empirical {
        potential: EmpiricalPotential,
        parameters: HashMap<String, f64>,
    },
    /// Density functional theory
    DFT {
        functional: String,
        basis_set: String,
        k_points: Vec<usize>,
    },
    /// Machine learning potential
    MLPotential {
        model: MLPotentialModel,
        uncertainty_quantification: bool,
    },
    /// Tight binding
    TightBinding {
        parameterization: String,
        k_points: Vec<usize>,
    },
}

#[derive(Debug, Clone)]
pub enum EmpiricalPotential {
    /// Lennard-Jones
    LennardJones,
    /// Buckingham
    Buckingham,
    /// Morse
    Morse,
    /// Embedded atom method
    EAM,
    /// Tersoff
    Tersoff,
    /// Stillinger-Weber
    StillingerWeber,
}

#[derive(Debug, Clone)]
pub struct MLPotentialModel {
    /// Model architecture
    pub architecture: String,
    /// Training error
    pub training_rmse: f64,
    /// Validation error
    pub validation_rmse: f64,
    /// Elements covered
    pub elements: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// Random search
    Random { num_trials: usize },
    /// Grid search
    Grid { resolution: Vec<usize> },
    /// Bayesian optimization
    Bayesian {
        acquisition_function: AcquisitionFunction,
        num_initial: usize,
    },
    /// Metadynamics
    Metadynamics {
        collective_variables: Vec<CollectiveVariable>,
        bias_factor: f64,
    },
}

#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    /// Expected improvement
    ExpectedImprovement,
    /// Upper confidence bound
    UCB { kappa: f64 },
    /// Probability of improvement
    ProbabilityOfImprovement,
    /// Thompson sampling
    ThompsonSampling,
}

#[derive(Debug, Clone)]
pub enum CollectiveVariable {
    /// Lattice parameter
    LatticeParameter { index: usize },
    /// Coordination number
    CoordinationNumber { element: String },
    /// Order parameter
    OrderParameter { definition: String },
    /// Density
    Density,
}

impl CrystalStructurePredictor {
    /// Create new crystal structure predictor
    pub fn new(
        composition: ChemicalComposition,
        energy_model: EnergyModel,
    ) -> Self {
        Self {
            composition,
            method: PredictionMethod::GlobalOptimization {
                algorithm: GlobalOptAlgorithm::SimulatedAnnealing,
                max_iterations: 1000,
            },
            constraints: StructureConstraints::default(),
            energy_model,
            search_strategy: SearchStrategy::Random { num_trials: 100 },
        }
    }
    
    /// Set prediction method
    pub fn with_method(mut self, method: PredictionMethod) -> Self {
        self.method = method;
        self
    }
    
    /// Set constraints
    pub fn with_constraints(mut self, constraints: StructureConstraints) -> Self {
        self.constraints = constraints;
        self
    }
    
    /// Build QUBO for structure prediction
    pub fn build_qubo(&self) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        match &self.method {
            PredictionMethod::GlobalOptimization { .. } => {
                self.build_global_optimization_qubo()
            }
            PredictionMethod::Evolutionary { .. } => {
                self.build_evolutionary_qubo()
            }
            _ => Err("QUBO formulation not available for this method".to_string()),
        }
    }
    
    /// Build QUBO for global optimization
    fn build_global_optimization_qubo(&self) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        // Discretize unit cell parameters and atomic positions
        let lattice_resolution = 10; // Number of discrete values per parameter
        let position_resolution = 20; // Grid points per dimension
        
        // Variables:
        // - Lattice parameters (a, b, c, α, β, γ)
        // - Atomic positions for each atom
        
        let n_lattice_vars = 6 * lattice_resolution;
        let n_atoms = self.composition.total_atoms;
        let n_position_vars = n_atoms * 3 * position_resolution;
        let n_vars = n_lattice_vars + n_position_vars;
        
        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();
        
        // Create variable mapping
        self.create_lattice_variables(&mut var_map, lattice_resolution)?;
        self.create_position_variables(&mut var_map, n_atoms, position_resolution, n_lattice_vars)?;
        
        // Add energy terms
        self.add_energy_objective(&mut qubo, &var_map)?;
        
        // Add constraints
        self.add_lattice_constraints(&mut qubo, &var_map, lattice_resolution)?;
        self.add_distance_constraints(&mut qubo, &var_map)?;
        self.add_symmetry_constraints(&mut qubo, &var_map)?;
        
        Ok((qubo, var_map))
    }
    
    /// Create lattice parameter variables
    fn create_lattice_variables(
        &self,
        var_map: &mut HashMap<String, usize>,
        resolution: usize,
    ) -> Result<(), String> {
        let params = ["a", "b", "c", "alpha", "beta", "gamma"];
        let mut var_idx = 0;
        
        for param in &params {
            for i in 0..resolution {
                let var_name = format!("lattice_{}_{}", param, i);
                var_map.insert(var_name, var_idx);
                var_idx += 1;
            }
        }
        
        Ok(())
    }
    
    /// Create atomic position variables
    fn create_position_variables(
        &self,
        var_map: &mut HashMap<String, usize>,
        n_atoms: usize,
        resolution: usize,
        offset: usize,
    ) -> Result<(), String> {
        let mut var_idx = offset;
        
        for atom in 0..n_atoms {
            for coord in ["x", "y", "z"] {
                for i in 0..resolution {
                    let var_name = format!("pos_{}_{}_{}", atom, coord, i);
                    var_map.insert(var_name, var_idx);
                    var_idx += 1;
                }
            }
        }
        
        Ok(())
    }
    
    /// Add energy objective
    fn add_energy_objective(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        // Simplified: use pairwise interactions
        match &self.energy_model {
            EnergyModel::Empirical { potential, parameters } => {
                self.add_empirical_energy(qubo, var_map, potential, parameters)
            }
            _ => {
                // For other models, use surrogate approximation
                self.add_surrogate_energy(qubo, var_map)
            }
        }
    }
    
    /// Add empirical potential energy
    fn add_empirical_energy(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        potential: &EmpiricalPotential,
        parameters: &HashMap<String, f64>,
    ) -> Result<(), String> {
        // Lennard-Jones example
        match potential {
            EmpiricalPotential::LennardJones => {
                let epsilon = parameters.get("epsilon").unwrap_or(&1.0);
                let sigma = parameters.get("sigma").unwrap_or(&3.4);
                
                // Add pairwise interactions
                for i in 0..self.composition.total_atoms {
                    for j in i+1..self.composition.total_atoms {
                        // This would compute LJ potential based on distance
                        // Simplified: add distance-dependent terms
                        self.add_pairwise_energy(qubo, var_map, i, j, *epsilon, *sigma)?;
                    }
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Add pairwise energy term
    fn add_pairwise_energy(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        atom1: usize,
        atom2: usize,
        epsilon: f64,
        sigma: f64,
    ) -> Result<(), String> {
        // Discretized distance calculation
        // This is a simplification - actual implementation would be more complex
        
        for coord in ["x", "y", "z"] {
            for i in 0..20 { // position resolution
                let var1 = format!("pos_{}_{}_{}", atom1, coord, i);
                let var2 = format!("pos_{}_{}_{}", atom2, coord, i);
                
                if let (Some(&idx1), Some(&idx2)) = (var_map.get(&var1), var_map.get(&var2)) {
                    // Same position = zero distance contribution
                    qubo[[idx1, idx2]] -= epsilon;
                }
            }
        }
        
        Ok(())
    }
    
    /// Add surrogate energy model
    fn add_surrogate_energy(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        // Use a simplified energy model based on:
        // - Coordination preferences
        // - Ideal bond lengths
        // - Electrostatic interactions
        
        // Add coordination energy
        self.add_coordination_energy(qubo, var_map)?;
        
        // Add electrostatic energy
        self.add_electrostatic_energy(qubo, var_map)?;
        
        Ok(())
    }
    
    /// Add coordination energy
    fn add_coordination_energy(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        // Penalize deviations from ideal coordination
        if !self.constraints.coordination.coordination_numbers.is_empty() {
            let coord_numbers = &self.constraints.coordination.coordination_numbers;
            // Simplified: just favor certain distance ranges
            let penalty = 10.0;
            
            for i in 0..self.composition.total_atoms {
                // Add terms that favor having neighbors at ideal distances
                // This is highly simplified
                for coord in ["x", "y", "z"] {
                    for pos in 0..20 {
                        let var_name = format!("pos_{}_{}_{}", i, coord, pos);
                        if let Some(&idx) = var_map.get(&var_name) {
                            // Favor middle positions (simplified)
                            let deviation = (pos as f64 - 10.0).abs();
                            qubo[[idx, idx]] += penalty * deviation / 10.0;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Add electrostatic energy
    fn add_electrostatic_energy(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        if let Some(oxidation_states) = &self.composition.oxidation_states {
            // Add Coulomb repulsion/attraction
            // Simplified: just use oxidation states
            
            let elements: Vec<_> = self.composition.elements.keys().collect();
            
            for i in 0..self.composition.total_atoms {
                for j in i+1..self.composition.total_atoms {
                    // Get charges (simplified assignment)
                    let charge1 = 1.0; // Would map from oxidation states
                    let charge2 = -1.0;
                    
                    // Electrostatic interaction
                    let interaction = charge1 * charge2;
                    
                    // Add to QUBO (simplified)
                    for coord in ["x", "y", "z"] {
                        for pos in 0..20 {
                            let var1 = format!("pos_{}_{}_{}", i, coord, pos);
                            let var2 = format!("pos_{}_{}_{}", j, coord, pos);
                            
                            if let (Some(&idx1), Some(&idx2)) = 
                                (var_map.get(&var1), var_map.get(&var2)) {
                                if idx1 != idx2 {
                                    qubo[[idx1, idx2]] += interaction * 0.1;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Add lattice constraints
    fn add_lattice_constraints(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        resolution: usize,
    ) -> Result<(), String> {
        let penalty = 100.0;
        
        // Enforce one-hot encoding for each lattice parameter
        for param in ["a", "b", "c", "alpha", "beta", "gamma"] {
            for i in 0..resolution {
                for j in i+1..resolution {
                    let var1 = format!("lattice_{}_{}", param, i);
                    let var2 = format!("lattice_{}_{}", param, j);
                    
                    if let (Some(&idx1), Some(&idx2)) = 
                        (var_map.get(&var1), var_map.get(&var2)) {
                        qubo[[idx1, idx2]] += penalty;
                    }
                }
            }
        }
        
        // Add lattice type constraints
        if let Some(lattice_type) = &self.constraints.lattice.lattice_type {
            self.add_lattice_type_constraints(qubo, var_map, lattice_type, resolution)?;
        }
        
        Ok(())
    }
    
    /// Add lattice type constraints
    fn add_lattice_type_constraints(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        lattice_type: &LatticeType,
        resolution: usize,
    ) -> Result<(), String> {
        let penalty = 200.0;
        
        match lattice_type {
            LatticeType::Cubic => {
                // a = b = c, α = β = γ = 90°
                for i in 0..resolution {
                    let var_a = format!("lattice_a_{}", i);
                    let var_b = format!("lattice_b_{}", i);
                    let var_c = format!("lattice_c_{}", i);
                    
                    // Encourage a = b = c
                    if let (Some(&idx_a), Some(&idx_b), Some(&idx_c)) = 
                        (var_map.get(&var_a), var_map.get(&var_b), var_map.get(&var_c)) {
                        
                        // Reward if all three are selected together
                        qubo[[idx_a, idx_b]] -= penalty;
                        qubo[[idx_b, idx_c]] -= penalty;
                        qubo[[idx_a, idx_c]] -= penalty;
                    }
                }
                
                // Fix angles at 90°
                let angle_90_idx = resolution / 2; // Assuming middle index represents 90°
                for angle in ["alpha", "beta", "gamma"] {
                    let var_name = format!("lattice_{}_{}", angle, angle_90_idx);
                    if let Some(&idx) = var_map.get(&var_name) {
                        qubo[[idx, idx]] -= penalty * 2.0;
                    }
                }
            }
            LatticeType::Hexagonal => {
                // a = b ≠ c, α = β = 90°, γ = 120°
                // Similar constraints...
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Add distance constraints
    fn add_distance_constraints(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        if !self.constraints.distances.min_distances.is_empty() {
            let min_distances = &self.constraints.distances.min_distances;
            let penalty = 50.0;
            
            // Penalize configurations where atoms are too close
            for ((elem1, elem2), &min_dist) in min_distances {
                // This would need proper element-to-atom mapping
                // Simplified: penalize same positions
                for i in 0..self.composition.total_atoms {
                    for j in i+1..self.composition.total_atoms {
                        for coord in ["x", "y", "z"] {
                            for pos in 0..20 {
                                let var1 = format!("pos_{}_{}_{}", i, coord, pos);
                                let var2 = format!("pos_{}_{}_{}", j, coord, pos);
                                
                                if let (Some(&idx1), Some(&idx2)) = 
                                    (var_map.get(&var1), var_map.get(&var2)) {
                                    if idx1 == idx2 {
                                        // Same position - too close
                                        qubo[[idx1, idx2]] += penalty;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Add symmetry constraints
    fn add_symmetry_constraints(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        if let Some(space_groups) = &self.constraints.symmetry.space_groups {
            // Simplified: encourage symmetric positions
            let symmetry_bonus = -10.0;
            
            // For high symmetry, atoms should be at special positions
            // This is highly simplified
            for i in 0..self.composition.total_atoms {
                // Favor positions at 0, 0.5, etc.
                for coord in ["x", "y", "z"] {
                    for special_pos in [0, 10, 19] { // 0, 0.5, 1 in fractional
                        let var_name = format!("pos_{}_{}_{}", i, coord, special_pos);
                        if let Some(&idx) = var_map.get(&var_name) {
                            qubo[[idx, idx]] += symmetry_bonus;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Build QUBO for evolutionary algorithm
    fn build_evolutionary_qubo(&self) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        // Encode genetic representation
        let genome_length = 100; // Simplified genome
        let mut qubo = Array2::zeros((genome_length, genome_length));
        let mut var_map = HashMap::new();
        
        for i in 0..genome_length {
            var_map.insert(format!("gene_{}", i), i);
        }
        
        // Add fitness function
        self.add_fitness_function(&mut qubo, &var_map)?;
        
        Ok((qubo, var_map))
    }
    
    /// Add fitness function for evolutionary algorithm
    fn add_fitness_function(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        // Simplified fitness based on:
        // - Energy (lower is better)
        // - Constraint satisfaction
        // - Diversity
        
        // This would be problem-specific
        for i in 0..qubo.shape()[0] {
            qubo[[i, i]] = -1.0; // Favor diversity
        }
        
        Ok(())
    }
    
    /// Decode solution to crystal structure
    pub fn decode_solution(
        &self,
        solution: &HashMap<String, bool>,
    ) -> Result<CrystalStructure, String> {
        // Extract lattice parameters
        let lattice = self.decode_lattice(solution)?;
        
        // Extract atomic positions
        let positions = self.decode_positions(solution)?;
        
        // Determine space group
        let space_group = self.determine_space_group(&lattice, &positions)?;
        
        Ok(CrystalStructure {
            composition: self.composition.clone(),
            lattice,
            positions,
            space_group,
            energy: None,
            properties: HashMap::new(),
        })
    }
    
    /// Decode lattice parameters
    fn decode_lattice(&self, solution: &HashMap<String, bool>) -> Result<Lattice, String> {
        let mut params = HashMap::new();
        
        for param in ["a", "b", "c", "alpha", "beta", "gamma"] {
            for i in 0..10 { // resolution
                let var_name = format!("lattice_{}_{}", param, i);
                if *solution.get(&var_name).unwrap_or(&false) {
                    // Map index to value
                    let value = match param {
                        "a" | "b" | "c" => 3.0 + i as f64 * 0.5, // 3-8 Å
                        "alpha" | "beta" | "gamma" => 60.0 + i as f64 * 6.0, // 60-120°
                        _ => 0.0,
                    };
                    params.insert(param.to_string(), value);
                    break;
                }
            }
        }
        
        Ok(Lattice {
            a: params.get("a").copied().unwrap_or(5.0),
            b: params.get("b").copied().unwrap_or(5.0),
            c: params.get("c").copied().unwrap_or(5.0),
            alpha: params.get("alpha").copied().unwrap_or(90.0),
            beta: params.get("beta").copied().unwrap_or(90.0),
            gamma: params.get("gamma").copied().unwrap_or(90.0),
        })
    }
    
    /// Decode atomic positions
    fn decode_positions(&self, solution: &HashMap<String, bool>) -> Result<Vec<AtomicPosition>, String> {
        let mut positions = Vec::new();
        
        // Simplified: assign elements round-robin
        let elements: Vec<_> = self.composition.elements.keys().cloned().collect();
        
        for atom in 0..self.composition.total_atoms {
            let mut coords = [0.0, 0.0, 0.0];
            
            for (i, coord) in ["x", "y", "z"].iter().enumerate() {
                for pos in 0..20 {
                    let var_name = format!("pos_{}_{}_{}", atom, coord, pos);
                    if *solution.get(&var_name).unwrap_or(&false) {
                        coords[i] = pos as f64 / 19.0; // Fractional coordinates
                        break;
                    }
                }
            }
            
            positions.push(AtomicPosition {
                element: elements[atom % elements.len()].clone(),
                x: coords[0],
                y: coords[1],
                z: coords[2],
                occupancy: 1.0,
            });
        }
        
        Ok(positions)
    }
    
    /// Determine space group
    fn determine_space_group(
        &self,
        lattice: &Lattice,
        positions: &[AtomicPosition],
    ) -> Result<SpaceGroup, String> {
        // Simplified: determine based on lattice type
        let lattice_type = lattice.determine_type();
        
        Ok(SpaceGroup {
            number: 1, // P1 by default
            symbol: "P1".to_string(),
            lattice_type,
            point_group: "1".to_string(),
        })
    }
}

impl Default for StructureConstraints {
    fn default() -> Self {
        Self {
            lattice: LatticeConstraints {
                min_lengths: None,
                max_lengths: None,
                angle_ranges: None,
                volume_range: None,
                lattice_type: None,
            },
            symmetry: SymmetryConstraints {
                space_groups: None,
                point_groups: None,
                min_symmetry: None,
                wyckoff_positions: None,
            },
            distances: DistanceConstraints {
                min_distances: HashMap::new(),
                max_distances: HashMap::new(),
                bond_lengths: Vec::new(),
            },
            coordination: CoordinationConstraints {
                coordination_numbers: HashMap::new(),
                geometries: HashMap::new(),
                allowed_ligands: HashMap::new(),
            },
            pressure: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Lattice {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl Lattice {
    /// Calculate unit cell volume
    pub fn volume(&self) -> f64 {
        let alpha_rad = self.alpha * PI / 180.0;
        let beta_rad = self.beta * PI / 180.0;
        let gamma_rad = self.gamma * PI / 180.0;
        
        self.a * self.b * self.c * (1.0 - alpha_rad.cos().powi(2) 
            - beta_rad.cos().powi(2) - gamma_rad.cos().powi(2) 
            + 2.0 * alpha_rad.cos() * beta_rad.cos() * gamma_rad.cos()).sqrt()
    }
    
    /// Determine lattice type
    pub fn determine_type(&self) -> LatticeType {
        let tol = 0.01;
        
        if (self.a - self.b).abs() < tol && (self.b - self.c).abs() < tol {
            if (self.alpha - 90.0).abs() < tol && 
               (self.beta - 90.0).abs() < tol && 
               (self.gamma - 90.0).abs() < tol {
                LatticeType::Cubic
            } else if (self.alpha - self.beta).abs() < tol && 
                      (self.beta - self.gamma).abs() < tol {
                LatticeType::Rhombohedral
            } else {
                LatticeType::Triclinic
            }
        } else if (self.a - self.b).abs() < tol {
            if (self.alpha - 90.0).abs() < tol && 
               (self.beta - 90.0).abs() < tol && 
               (self.gamma - 120.0).abs() < tol {
                LatticeType::Hexagonal
            } else if (self.alpha - 90.0).abs() < tol && 
                      (self.beta - 90.0).abs() < tol && 
                      (self.gamma - 90.0).abs() < tol {
                LatticeType::Tetragonal
            } else {
                LatticeType::Monoclinic
            }
        } else if (self.alpha - 90.0).abs() < tol && 
                  (self.beta - 90.0).abs() < tol && 
                  (self.gamma - 90.0).abs() < tol {
            LatticeType::Orthorhombic
        } else if (self.alpha - 90.0).abs() < tol && 
                  (self.gamma - 90.0).abs() < tol {
            LatticeType::Monoclinic
        } else {
            LatticeType::Triclinic
        }
    }
    
    /// Get transformation matrix
    pub fn transformation_matrix(&self) -> Array2<f64> {
        let alpha_rad = self.alpha * PI / 180.0;
        let beta_rad = self.beta * PI / 180.0;
        let gamma_rad = self.gamma * PI / 180.0;
        
        let mut matrix = Array2::zeros((3, 3));
        
        matrix[[0, 0]] = self.a;
        matrix[[0, 1]] = self.b * gamma_rad.cos();
        matrix[[0, 2]] = self.c * beta_rad.cos();
        
        matrix[[1, 0]] = 0.0;
        matrix[[1, 1]] = self.b * gamma_rad.sin();
        matrix[[1, 2]] = self.c * (alpha_rad.cos() - beta_rad.cos() * gamma_rad.cos()) / gamma_rad.sin();
        
        matrix[[2, 0]] = 0.0;
        matrix[[2, 1]] = 0.0;
        matrix[[2, 2]] = self.c * (1.0 - alpha_rad.cos().powi(2) - beta_rad.cos().powi(2) 
            - gamma_rad.cos().powi(2) + 2.0 * alpha_rad.cos() * beta_rad.cos() * gamma_rad.cos()).sqrt() / gamma_rad.sin();
        
        matrix
    }
}

#[derive(Debug, Clone)]
pub struct AtomicPosition {
    pub element: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub occupancy: f64,
}

#[derive(Debug, Clone)]
pub struct SpaceGroup {
    pub number: u32,
    pub symbol: String,
    pub lattice_type: LatticeType,
    pub point_group: String,
}

#[derive(Debug, Clone)]
pub struct CrystalStructure {
    pub composition: ChemicalComposition,
    pub lattice: Lattice,
    pub positions: Vec<AtomicPosition>,
    pub space_group: SpaceGroup,
    pub energy: Option<f64>,
    pub properties: HashMap<String, f64>,
}

impl CrystalStructure {
    /// Calculate density
    pub fn density(&self) -> f64 {
        let volume = self.lattice.volume();
        let mass = self.calculate_mass();
        
        // Convert to g/cm³
        mass / volume * 1.66054
    }
    
    /// Calculate formula unit mass
    fn calculate_mass(&self) -> f64 {
        // Atomic masses (simplified)
        let masses: HashMap<&str, f64> = [
            ("H", 1.008),
            ("C", 12.011),
            ("N", 14.007),
            ("O", 15.999),
            ("Na", 22.990),
            ("Mg", 24.305),
            ("Al", 26.982),
            ("Si", 28.085),
            ("Fe", 55.845),
        ].iter().cloned().collect();
        
        self.composition.elements.iter()
            .map(|(elem, count)| masses.get(elem.as_str()).unwrap_or(&1.0) * *count as f64)
            .sum()
    }
    
    /// Generate supercell
    pub fn supercell(&self, nx: usize, ny: usize, nz: usize) -> CrystalStructure {
        let mut new_positions = Vec::new();
        
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    for pos in &self.positions {
                        new_positions.push(AtomicPosition {
                            element: pos.element.clone(),
                            x: (pos.x + i as f64) / nx as f64,
                            y: (pos.y + j as f64) / ny as f64,
                            z: (pos.z + k as f64) / nz as f64,
                            occupancy: pos.occupancy,
                        });
                    }
                }
            }
        }
        
        let mut new_composition = self.composition.clone();
        for count in new_composition.elements.values_mut() {
            *count *= nx * ny * nz;
        }
        new_composition.total_atoms *= nx * ny * nz;
        
        CrystalStructure {
            composition: new_composition,
            lattice: Lattice {
                a: self.lattice.a * nx as f64,
                b: self.lattice.b * ny as f64,
                c: self.lattice.c * nz as f64,
                ..self.lattice.clone()
            },
            positions: new_positions,
            space_group: self.space_group.clone(),
            energy: None,
            properties: HashMap::new(),
        }
    }
}

/// Phase transition analyzer
pub struct PhaseTransitionAnalyzer {
    /// Structures to analyze
    structures: Vec<CrystalStructure>,
    /// Analysis method
    method: TransitionMethod,
    /// Order parameters
    order_parameters: Vec<OrderParameter>,
}

#[derive(Debug, Clone)]
pub enum TransitionMethod {
    /// Nudged elastic band
    NEB { images: usize, spring_constant: f64 },
    /// Metadynamics
    Metadynamics { bias_factor: f64, gaussian_width: f64 },
    /// Transition path sampling
    TPS { shooting_moves: usize },
    /// Machine learning
    ML { model: String },
}

#[derive(Debug, Clone)]
pub struct OrderParameter {
    /// Parameter name
    pub name: String,
    /// Definition
    pub definition: OrderParameterDef,
    /// Range
    pub range: (f64, f64),
}

#[derive(Debug, Clone)]
pub enum OrderParameterDef {
    /// Structural parameter
    Structural { description: String },
    /// Electronic parameter
    Electronic { property: String },
    /// Magnetic parameter
    Magnetic { moment_type: String },
    /// Custom function
    Custom,
}

/// Defect modeler
pub struct DefectModeler {
    /// Host structure
    host: CrystalStructure,
    /// Defect types to consider
    defect_types: Vec<DefectType>,
    /// Defect interactions
    interactions: DefectInteractions,
}

#[derive(Debug, Clone)]
pub enum DefectType {
    /// Vacancy
    Vacancy { site: usize },
    /// Interstitial
    Interstitial { element: String, position: Vec3D },
    /// Substitution
    Substitution { site: usize, new_element: String },
    /// Frenkel pair
    Frenkel { vacancy_site: usize, interstitial_pos: Vec3D },
    /// Schottky defect
    Schottky { sites: Vec<usize> },
    /// Grain boundary
    GrainBoundary { plane: (i32, i32, i32), angle: f64 },
}

#[derive(Debug, Clone)]
pub struct DefectInteractions {
    /// Interaction range
    pub cutoff: f64,
    /// Interaction model
    pub model: InteractionModel,
    /// Clustering tendency
    pub clustering: bool,
}

#[derive(Debug, Clone)]
pub enum InteractionModel {
    /// Coulombic
    Coulombic,
    /// Elastic
    Elastic { elastic_constants: Array2<f64> },
    /// Combined
    Combined,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_crystal_structure_predictor() {
        let composition = ChemicalComposition {
            elements: {
                let mut elements = HashMap::new();
                elements.insert("Na".to_string(), 1);
                elements.insert("Cl".to_string(), 1);
                elements
            },
            total_atoms: 2,
            stoichiometry: Some(vec![1.0, 1.0]),
            oxidation_states: Some({
                let mut states = HashMap::new();
                states.insert("Na".to_string(), 1);
                states.insert("Cl".to_string(), -1);
                states
            }),
        };
        
        let energy_model = EnergyModel::Empirical {
            potential: EmpiricalPotential::LennardJones,
            parameters: {
                let mut params = HashMap::new();
                params.insert("epsilon".to_string(), 1.0);
                params.insert("sigma".to_string(), 3.4);
                params
            },
        };
        
        let predictor = CrystalStructurePredictor::new(composition, energy_model);
        let result = predictor.build_qubo();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_lattice() {
        let lattice = Lattice {
            a: 5.0,
            b: 5.0,
            c: 5.0,
            alpha: 90.0,
            beta: 90.0,
            gamma: 90.0,
        };
        
        assert_eq!(lattice.determine_type() as u8, LatticeType::Cubic as u8);
        assert!((lattice.volume() - 125.0).abs() < 0.01);
    }
    
    #[test]
    fn test_supercell() {
        let structure = CrystalStructure {
            composition: ChemicalComposition {
                elements: {
                    let mut elements = HashMap::new();
                    elements.insert("Si".to_string(), 1);
                    elements
                },
                total_atoms: 1,
                stoichiometry: None,
                oxidation_states: None,
            },
            lattice: Lattice {
                a: 5.0,
                b: 5.0,
                c: 5.0,
                alpha: 90.0,
                beta: 90.0,
                gamma: 90.0,
            },
            positions: vec![
                AtomicPosition {
                    element: "Si".to_string(),
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    occupancy: 1.0,
                }
            ],
            space_group: SpaceGroup {
                number: 225,
                symbol: "Fm-3m".to_string(),
                lattice_type: LatticeType::Cubic,
                point_group: "m-3m".to_string(),
            },
            energy: None,
            properties: HashMap::new(),
        };
        
        let supercell = structure.supercell(2, 2, 2);
        assert_eq!(supercell.positions.len(), 8);
        assert_eq!(supercell.composition.total_atoms, 8);
    }
}