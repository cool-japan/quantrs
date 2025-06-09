//! Drug discovery applications: Molecular design and optimization.
//!
//! This module provides quantum optimization tools for drug discovery
//! including molecular design, lead optimization, and virtual screening.

use crate::sampler::{Sampler, SampleResult, SamplerError, SamplerResult};
use ndarray::{Array, Array1, Array2, Array3, IxDyn};
use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;
use rand::prelude::*;

/// Molecular design optimizer
pub struct MolecularDesignOptimizer {
    /// Target properties
    target_properties: TargetProperties,
    /// Fragment library
    fragment_library: FragmentLibrary,
    /// Scoring function
    scoring_function: ScoringFunction,
    /// Design constraints
    constraints: DesignConstraints,
    /// Optimization strategy
    strategy: OptimizationStrategy,
}

#[derive(Debug, Clone)]
pub struct TargetProperties {
    /// Target molecular weight
    pub molecular_weight: Option<(f64, f64)>, // (min, max)
    /// LogP range
    pub logp: Option<(f64, f64)>,
    /// LogS range
    pub logs: Option<(f64, f64)>,
    /// H-bond donors
    pub hbd: Option<(usize, usize)>,
    /// H-bond acceptors
    pub hba: Option<(usize, usize)>,
    /// Rotatable bonds
    pub rotatable_bonds: Option<(usize, usize)>,
    /// TPSA range
    pub tpsa: Option<(f64, f64)>,
    /// Custom descriptors
    pub custom_descriptors: HashMap<String, (f64, f64)>,
}

#[derive(Debug, Clone)]
pub struct FragmentLibrary {
    /// Available fragments
    pub fragments: Vec<MolecularFragment>,
    /// Connection rules
    pub connection_rules: ConnectionRules,
    /// Fragment frequencies in known drugs
    pub fragment_scores: HashMap<usize, f64>,
    /// Privileged scaffolds
    pub privileged_scaffolds: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct MolecularFragment {
    /// Fragment ID
    pub id: usize,
    /// SMILES representation
    pub smiles: String,
    /// Attachment points
    pub attachment_points: Vec<AttachmentPoint>,
    /// Fragment properties
    pub properties: FragmentProperties,
    /// Pharmacophore features
    pub pharmacophores: Vec<PharmacophoreFeature>,
}

#[derive(Debug, Clone)]
pub struct AttachmentPoint {
    /// Atom index
    pub atom_idx: usize,
    /// Bond type allowed
    pub bond_types: Vec<BondType>,
    /// Directionality
    pub direction: Vec3D,
}

#[derive(Debug, Clone)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
}

#[derive(Debug, Clone)]
pub struct Vec3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone)]
pub struct FragmentProperties {
    /// Molecular weight contribution
    pub mw_contribution: f64,
    /// LogP contribution
    pub logp_contribution: f64,
    /// H-bond donors
    pub hbd_count: usize,
    /// H-bond acceptors
    pub hba_count: usize,
    /// Rotatable bonds
    pub rotatable_count: usize,
    /// TPSA contribution
    pub tpsa_contribution: f64,
}

#[derive(Debug, Clone)]
pub struct PharmacophoreFeature {
    /// Feature type
    pub feature_type: PharmacophoreType,
    /// Position relative to fragment
    pub position: Vec3D,
    /// Tolerance radius
    pub tolerance: f64,
}

#[derive(Debug, Clone)]
pub enum PharmacophoreType {
    HBondDonor,
    HBondAcceptor,
    Hydrophobic,
    Aromatic,
    PositiveCharge,
    NegativeCharge,
    MetalCoordination,
}

#[derive(Debug, Clone)]
pub struct ConnectionRules {
    /// Compatible fragment pairs
    pub compatible_pairs: HashMap<(usize, usize), f64>,
    /// Forbidden connections
    pub forbidden_connections: HashSet<(usize, usize)>,
    /// Reaction templates
    pub reaction_templates: Vec<ReactionTemplate>,
}

#[derive(Debug, Clone)]
pub struct ReactionTemplate {
    /// Template name
    pub name: String,
    /// Required functional groups
    pub reactants: Vec<FunctionalGroup>,
    /// Product pattern
    pub product_pattern: String,
    /// Reaction feasibility score
    pub feasibility: f64,
}

#[derive(Debug, Clone)]
pub struct FunctionalGroup {
    /// SMARTS pattern
    pub smarts: String,
    /// Required count
    pub count: usize,
}

#[derive(Debug, Clone)]
pub enum ScoringFunction {
    /// Simple additive scoring
    Additive { weights: HashMap<String, f64> },
    /// Machine learning based
    MLBased { model_path: String },
    /// Docking score
    DockingBased { receptor: ProteinStructure },
    /// Multi-objective
    MultiObjective { objectives: Vec<ObjectiveFunction> },
    /// Pharmacophore matching
    PharmacophoreMatching { reference: PharmacophoreModel },
}

#[derive(Debug, Clone)]
pub struct ProteinStructure {
    /// PDB ID or path
    pub pdb_id: String,
    /// Active site residues
    pub active_site: Vec<usize>,
    /// Grid box for docking
    pub grid_box: GridBox,
}

#[derive(Debug, Clone)]
pub struct GridBox {
    pub center: Vec3D,
    pub dimensions: Vec3D,
    pub spacing: f64,
}

#[derive(Debug, Clone)]
pub struct PharmacophoreModel {
    /// Required features
    pub features: Vec<PharmacophoreFeature>,
    /// Distance constraints
    pub distance_constraints: Vec<DistanceConstraint>,
    /// Angle constraints
    pub angle_constraints: Vec<AngleConstraint>,
}

#[derive(Debug, Clone)]
pub struct DistanceConstraint {
    pub feature1: usize,
    pub feature2: usize,
    pub min_distance: f64,
    pub max_distance: f64,
}

#[derive(Debug, Clone)]
pub struct AngleConstraint {
    pub feature1: usize,
    pub feature2: usize,
    pub feature3: usize,
    pub min_angle: f64,
    pub max_angle: f64,
}

#[derive(Debug, Clone)]
pub enum ObjectiveFunction {
    /// Binding affinity
    BindingAffinity { weight: f64 },
    /// Synthetic accessibility
    SyntheticAccessibility { weight: f64 },
    /// ADMET properties
    ADMET { property: ADMETProperty, weight: f64 },
    /// Novelty
    Novelty { reference_set: Vec<String>, weight: f64 },
    /// Diversity
    Diversity { weight: f64 },
}

#[derive(Debug, Clone)]
pub enum ADMETProperty {
    Absorption,
    Distribution,
    Metabolism,
    Excretion,
    Toxicity,
    Solubility,
    Permeability,
    Stability,
}

#[derive(Debug, Clone)]
pub struct DesignConstraints {
    /// Maximum molecular weight
    pub max_mw: f64,
    /// Lipinski's rule of five
    pub lipinski: bool,
    /// Veber's rules
    pub veber: bool,
    /// PAINS filters
    pub pains_filter: bool,
    /// Synthetic accessibility threshold
    pub max_sa_score: f64,
    /// Minimum QED score
    pub min_qed: f64,
    /// Custom SMARTS filters
    pub smarts_filters: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    /// Fragment growing
    FragmentGrowing { core: MolecularFragment },
    /// Fragment linking
    FragmentLinking { fragments: Vec<MolecularFragment> },
    /// Fragment hopping
    FragmentHopping { scaffold: MolecularFragment },
    /// De novo design
    DeNovo,
    /// Lead optimization
    LeadOptimization { lead: String },
}

impl MolecularDesignOptimizer {
    /// Create new molecular design optimizer
    pub fn new(
        target_properties: TargetProperties,
        fragment_library: FragmentLibrary,
    ) -> Self {
        Self {
            target_properties,
            fragment_library,
            scoring_function: ScoringFunction::Additive {
                weights: Self::default_weights(),
            },
            constraints: DesignConstraints::default(),
            strategy: OptimizationStrategy::DeNovo,
        }
    }
    
    /// Default scoring weights
    fn default_weights() -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        weights.insert("mw_penalty".to_string(), -0.1);
        weights.insert("logp_penalty".to_string(), -0.2);
        weights.insert("hbd_penalty".to_string(), -0.1);
        weights.insert("hba_penalty".to_string(), -0.1);
        weights.insert("rotatable_penalty".to_string(), -0.05);
        weights.insert("tpsa_penalty".to_string(), -0.1);
        weights.insert("fragment_score".to_string(), 1.0);
        weights
    }
    
    /// Set scoring function
    pub fn with_scoring(mut self, scoring: ScoringFunction) -> Self {
        self.scoring_function = scoring;
        self
    }
    
    /// Set constraints
    pub fn with_constraints(mut self, constraints: DesignConstraints) -> Self {
        self.constraints = constraints;
        self
    }
    
    /// Set optimization strategy
    pub fn with_strategy(mut self, strategy: OptimizationStrategy) -> Self {
        self.strategy = strategy;
        self
    }
    
    /// Build QUBO for molecular design
    pub fn build_qubo(&self) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        match &self.strategy {
            OptimizationStrategy::FragmentGrowing { core } => {
                self.build_fragment_growing_qubo(core)
            }
            OptimizationStrategy::DeNovo => {
                self.build_de_novo_qubo()
            }
            _ => Err("Strategy not yet implemented".to_string()),
        }
    }
    
    /// Build QUBO for fragment growing
    fn build_fragment_growing_qubo(
        &self,
        core: &MolecularFragment,
    ) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        // Variables: x_{f,p} = 1 if fragment f is attached at position p
        let positions = core.attachment_points.len();
        let fragments = self.fragment_library.fragments.len();
        let n_vars = positions * fragments;
        
        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();
        
        // Create variable mapping
        for p in 0..positions {
            for f in 0..fragments {
                let var_name = format!("x_{}_{}", f, p);
                var_map.insert(var_name, p * fragments + f);
            }
        }
        
        // Add scoring terms
        self.add_fragment_scores(&mut qubo, &var_map, core)?;
        
        // Add property constraints
        self.add_property_constraints(&mut qubo, &var_map, core)?;
        
        // Add connection compatibility
        self.add_connection_compatibility(&mut qubo, &var_map, core)?;
        
        // At most one fragment per position
        self.add_uniqueness_constraints(&mut qubo, &var_map, positions, fragments)?;
        
        Ok((qubo, var_map))
    }
    
    /// Build QUBO for de novo design
    fn build_de_novo_qubo(&self) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        // Variables: x_{f,i} = 1 if fragment f is at position i in molecule
        let max_positions = 10; // Maximum molecule size
        let fragments = self.fragment_library.fragments.len();
        
        // Additional variables for connections
        // y_{i,j} = 1 if position i connects to position j
        
        let position_vars = max_positions * fragments;
        let connection_vars = max_positions * (max_positions - 1) / 2;
        let n_vars = position_vars + connection_vars;
        
        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();
        
        // Position variables
        for i in 0..max_positions {
            for f in 0..fragments {
                let var_name = format!("x_{}_{}", f, i);
                var_map.insert(var_name, i * fragments + f);
            }
        }
        
        // Connection variables
        let mut var_idx = position_vars;
        for i in 0..max_positions {
            for j in i+1..max_positions {
                let var_name = format!("y_{}_{}", i, j);
                var_map.insert(var_name, var_idx);
                var_idx += 1;
            }
        }
        
        // Add de novo specific terms
        self.add_de_novo_objective(&mut qubo, &var_map, max_positions)?;
        
        // Connectivity constraints
        self.add_connectivity_constraints(&mut qubo, &var_map, max_positions)?;
        
        // Property constraints
        self.add_global_property_constraints(&mut qubo, &var_map, max_positions)?;
        
        Ok((qubo, var_map))
    }
    
    /// Add fragment scoring terms
    fn add_fragment_scores(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        core: &MolecularFragment,
    ) -> Result<(), String> {
        let positions = core.attachment_points.len();
        
        for p in 0..positions {
            for f in 0..self.fragment_library.fragments.len() {
                let var_name = format!("x_{}_{}", f, p);
                if let Some(&var_idx) = var_map.get(&var_name) {
                    // Fragment score
                    let score = self.fragment_library.fragment_scores
                        .get(&f)
                        .unwrap_or(&0.0);
                    
                    // Compatibility with attachment point
                    let compatibility = self.compute_attachment_compatibility(
                        &core.attachment_points[p],
                        &self.fragment_library.fragments[f],
                    );
                    
                    qubo[[var_idx, var_idx]] -= score * compatibility;
                }
            }
        }
        
        Ok(())
    }
    
    /// Compute attachment compatibility
    fn compute_attachment_compatibility(
        &self,
        attachment: &AttachmentPoint,
        fragment: &MolecularFragment,
    ) -> f64 {
        // Check if fragment has compatible attachment points
        let compatible = fragment.attachment_points.iter()
            .any(|frag_attach| {
                attachment.bond_types.iter()
                    .any(|bt| frag_attach.bond_types.contains(bt))
            });
        
        if compatible { 1.0 } else { 0.0 }
    }
    
    /// Add property constraints
    fn add_property_constraints(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        core: &MolecularFragment,
    ) -> Result<(), String> {
        let penalty = 100.0;
        
        // Molecular weight constraint
        if let Some((min_mw, max_mw)) = self.target_properties.molecular_weight {
            let core_mw = core.properties.mw_contribution;
            
            for f in 0..self.fragment_library.fragments.len() {
                let frag_mw = self.fragment_library.fragments[f].properties.mw_contribution;
                let total_mw = core_mw + frag_mw;
                
                if total_mw < min_mw || total_mw > max_mw {
                    // Penalize out-of-range combinations
                    for p in 0..core.attachment_points.len() {
                        let var_name = format!("x_{}_{}", f, p);
                        if let Some(&var_idx) = var_map.get(&var_name) {
                            qubo[[var_idx, var_idx]] += penalty;
                        }
                    }
                }
            }
        }
        
        // Similar constraints for other properties
        self.add_logp_constraints(qubo, var_map, core, penalty)?;
        self.add_hbond_constraints(qubo, var_map, core, penalty)?;
        
        Ok(())
    }
    
    /// Add LogP constraints
    fn add_logp_constraints(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        core: &MolecularFragment,
        penalty: f64,
    ) -> Result<(), String> {
        if let Some((min_logp, max_logp)) = self.target_properties.logp {
            let core_logp = core.properties.logp_contribution;
            
            for f in 0..self.fragment_library.fragments.len() {
                let frag_logp = self.fragment_library.fragments[f].properties.logp_contribution;
                let total_logp = core_logp + frag_logp;
                
                if total_logp < min_logp || total_logp > max_logp {
                    for p in 0..core.attachment_points.len() {
                        let var_name = format!("x_{}_{}", f, p);
                        if let Some(&var_idx) = var_map.get(&var_name) {
                            qubo[[var_idx, var_idx]] += penalty * 0.5;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Add H-bond constraints
    fn add_hbond_constraints(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        core: &MolecularFragment,
        penalty: f64,
    ) -> Result<(), String> {
        // H-bond donor constraints
        if let Some((min_hbd, max_hbd)) = self.target_properties.hbd {
            let core_hbd = core.properties.hbd_count;
            
            for f in 0..self.fragment_library.fragments.len() {
                let frag_hbd = self.fragment_library.fragments[f].properties.hbd_count;
                let total_hbd = core_hbd + frag_hbd;
                
                if total_hbd < min_hbd || total_hbd > max_hbd {
                    for p in 0..core.attachment_points.len() {
                        let var_name = format!("x_{}_{}", f, p);
                        if let Some(&var_idx) = var_map.get(&var_name) {
                            qubo[[var_idx, var_idx]] += penalty * 0.3;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Add connection compatibility
    fn add_connection_compatibility(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        core: &MolecularFragment,
    ) -> Result<(), String> {
        let positions = core.attachment_points.len();
        
        // Penalize incompatible fragment pairs at different positions
        for p1 in 0..positions {
            for p2 in p1+1..positions {
                for f1 in 0..self.fragment_library.fragments.len() {
                    for f2 in 0..self.fragment_library.fragments.len() {
                        let var1 = format!("x_{}_{}", f1, p1);
                        let var2 = format!("x_{}_{}", f2, p2);
                        
                        if let (Some(&idx1), Some(&idx2)) = (var_map.get(&var1), var_map.get(&var2)) {
                            // Check compatibility
                            if self.fragment_library.connection_rules
                                .forbidden_connections.contains(&(f1, f2)) {
                                qubo[[idx1, idx2]] += 1000.0;
                            } else if let Some(&score) = self.fragment_library.connection_rules
                                .compatible_pairs.get(&(f1, f2)) {
                                qubo[[idx1, idx2]] -= score;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Add uniqueness constraints
    fn add_uniqueness_constraints(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        positions: usize,
        fragments: usize,
    ) -> Result<(), String> {
        let penalty = 100.0;
        
        // At most one fragment per position
        for p in 0..positions {
            // (sum_f x_{f,p} - 1)^2 if we want exactly one
            // or just penalize multiple selections
            for f1 in 0..fragments {
                for f2 in f1+1..fragments {
                    let var1 = format!("x_{}_{}", f1, p);
                    let var2 = format!("x_{}_{}", f2, p);
                    
                    if let (Some(&idx1), Some(&idx2)) = (var_map.get(&var1), var_map.get(&var2)) {
                        qubo[[idx1, idx2]] += penalty;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Add de novo objective
    fn add_de_novo_objective(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        max_positions: usize,
    ) -> Result<(), String> {
        // Favor molecules with good fragment scores
        for i in 0..max_positions {
            for f in 0..self.fragment_library.fragments.len() {
                let var_name = format!("x_{}_{}", f, i);
                if let Some(&var_idx) = var_map.get(&var_name) {
                    let score = self.fragment_library.fragment_scores
                        .get(&f)
                        .unwrap_or(&0.0);
                    qubo[[var_idx, var_idx]] -= score;
                    
                    // Privileged scaffolds get bonus
                    if self.fragment_library.privileged_scaffolds.contains(&f) {
                        qubo[[var_idx, var_idx]] -= 2.0;
                    }
                }
            }
        }
        
        // Favor connected molecules
        for i in 0..max_positions {
            for j in i+1..max_positions {
                let conn_var = format!("y_{}_{}", i, j);
                if let Some(&conn_idx) = var_map.get(&conn_var) {
                    // Small penalty for connections (want some but not too many)
                    qubo[[conn_idx, conn_idx]] += 0.1;
                }
            }
        }
        
        Ok(())
    }
    
    /// Add connectivity constraints
    fn add_connectivity_constraints(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        max_positions: usize,
    ) -> Result<(), String> {
        let penalty = 100.0;
        
        // If position i has a fragment, it should connect to at least one other
        for i in 0..max_positions {
            // Connection indicator for position i
            for f in 0..self.fragment_library.fragments.len() {
                let frag_var = format!("x_{}_{}", f, i);
                if let Some(&frag_idx) = var_map.get(&frag_var) {
                    // Must have at least one connection if fragment present
                    let mut has_connection = false;
                    for j in 0..max_positions {
                        if i != j {
                            let conn_var = if i < j {
                                format!("y_{}_{}", i, j)
                            } else {
                                format!("y_{}_{}", j, i)
                            };
                            
                            if let Some(&conn_idx) = var_map.get(&conn_var) {
                                // Fragment at i but no connections is penalized
                                qubo[[frag_idx, frag_idx]] += penalty;
                                qubo[[frag_idx, conn_idx]] -= penalty;
                                has_connection = true;
                            }
                        }
                    }
                }
            }
        }
        
        // Connection compatibility
        self.add_connection_compatibility_de_novo(qubo, var_map, max_positions)?;
        
        Ok(())
    }
    
    /// Add connection compatibility for de novo
    fn add_connection_compatibility_de_novo(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        max_positions: usize,
    ) -> Result<(), String> {
        // If positions i and j are connected, fragments must be compatible
        for i in 0..max_positions {
            for j in i+1..max_positions {
                let conn_var = format!("y_{}_{}", i, j);
                if let Some(&conn_idx) = var_map.get(&conn_var) {
                    for f1 in 0..self.fragment_library.fragments.len() {
                        for f2 in 0..self.fragment_library.fragments.len() {
                            let var1 = format!("x_{}_{}", f1, i);
                            let var2 = format!("x_{}_{}", f2, j);
                            
                            if let (Some(&idx1), Some(&idx2)) = 
                                (var_map.get(&var1), var_map.get(&var2)) {
                                
                                if self.fragment_library.connection_rules
                                    .forbidden_connections.contains(&(f1, f2)) {
                                    // Penalize: connection + incompatible fragments
                                    // This is a 3-way interaction, approximate with 2-way
                                    qubo[[conn_idx, idx1]] += 50.0;
                                    qubo[[conn_idx, idx2]] += 50.0;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Add global property constraints
    fn add_global_property_constraints(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        max_positions: usize,
    ) -> Result<(), String> {
        // This is challenging as properties are additive over all selected fragments
        // Use penalty approximation
        
        let penalty = 10.0;
        
        // Approximate molecular weight constraint
        if let Some((min_mw, max_mw)) = self.target_properties.molecular_weight {
            for i in 0..max_positions {
                for f in 0..self.fragment_library.fragments.len() {
                    let var_name = format!("x_{}_{}", f, i);
                    if let Some(&var_idx) = var_map.get(&var_name) {
                        let mw = self.fragment_library.fragments[f].properties.mw_contribution;
                        
                        // Penalize if single fragment already exceeds limits
                        if mw > max_mw {
                            qubo[[var_idx, var_idx]] += penalty * 10.0;
                        }
                        
                        // Soft penalty based on contribution
                        let mw_penalty = if mw > max_mw / max_positions as f64 {
                            (mw - max_mw / max_positions as f64) * penalty
                        } else {
                            0.0
                        };
                        
                        qubo[[var_idx, var_idx]] += mw_penalty;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Decode solution to molecule
    pub fn decode_solution(
        &self,
        solution: &HashMap<String, bool>,
    ) -> Result<DesignedMolecule, String> {
        match &self.strategy {
            OptimizationStrategy::FragmentGrowing { core } => {
                self.decode_fragment_growing(solution, core)
            }
            OptimizationStrategy::DeNovo => {
                self.decode_de_novo(solution)
            }
            _ => Err("Decoding not implemented for this strategy".to_string()),
        }
    }
    
    /// Decode fragment growing solution
    fn decode_fragment_growing(
        &self,
        solution: &HashMap<String, bool>,
        core: &MolecularFragment,
    ) -> Result<DesignedMolecule, String> {
        let mut fragments = vec![core.clone()];
        let mut connections = Vec::new();
        
        // Find attached fragments
        for (var_name, &value) in solution {
            if value && var_name.starts_with("x_") {
                let parts: Vec<&str> = var_name[2..].split('_').collect();
                if parts.len() == 2 {
                    let frag_idx: usize = parts[0].parse().unwrap_or(0);
                    let pos_idx: usize = parts[1].parse().unwrap_or(0);
                    
                    if frag_idx < self.fragment_library.fragments.len() {
                        fragments.push(self.fragment_library.fragments[frag_idx].clone());
                        connections.push(Connection {
                            from_fragment: 0,
                            from_attachment: pos_idx,
                            to_fragment: fragments.len() - 1,
                            to_attachment: 0,
                            bond_type: BondType::Single,
                        });
                    }
                }
            }
        }
        
        let properties = self.calculate_properties(&fragments);
        let score = self.calculate_score(&fragments, &connections);
        
        Ok(DesignedMolecule {
            fragments,
            connections,
            properties,
            score,
            smiles: None, // Would need to construct SMILES
        })
    }
    
    /// Decode de novo solution
    fn decode_de_novo(
        &self,
        solution: &HashMap<String, bool>,
    ) -> Result<DesignedMolecule, String> {
        let mut fragment_positions: HashMap<usize, usize> = HashMap::new();
        let mut connections = Vec::new();
        
        // Find fragment positions
        for (var_name, &value) in solution {
            if value && var_name.starts_with("x_") {
                let parts: Vec<&str> = var_name[2..].split('_').collect();
                if parts.len() == 2 {
                    let frag_idx: usize = parts[0].parse().unwrap_or(0);
                    let pos_idx: usize = parts[1].parse().unwrap_or(0);
                    fragment_positions.insert(pos_idx, frag_idx);
                }
            }
        }
        
        // Find connections
        for (var_name, &value) in solution {
            if value && var_name.starts_with("y_") {
                let parts: Vec<&str> = var_name[2..].split('_').collect();
                if parts.len() == 2 {
                    let pos1: usize = parts[0].parse().unwrap_or(0);
                    let pos2: usize = parts[1].parse().unwrap_or(0);
                    
                    if fragment_positions.contains_key(&pos1) && 
                       fragment_positions.contains_key(&pos2) {
                        connections.push(Connection {
                            from_fragment: pos1,
                            from_attachment: 0,
                            to_fragment: pos2,
                            to_attachment: 0,
                            bond_type: BondType::Single,
                        });
                    }
                }
            }
        }
        
        // Build fragment list
        let fragments: Vec<_> = fragment_positions.iter()
            .map(|(_, &frag_idx)| self.fragment_library.fragments[frag_idx].clone())
            .collect();
        
        Ok(DesignedMolecule {
            fragments,
            connections,
            properties: MolecularProperties::default(),
            score: 0.0,
            smiles: None,
        })
    }
    
    /// Calculate molecular properties
    fn calculate_properties(&self, fragments: &[MolecularFragment]) -> MolecularProperties {
        let mut props = MolecularProperties::default();
        
        for fragment in fragments {
            props.molecular_weight += fragment.properties.mw_contribution;
            props.logp += fragment.properties.logp_contribution;
            props.hbd += fragment.properties.hbd_count;
            props.hba += fragment.properties.hba_count;
            props.rotatable_bonds += fragment.properties.rotatable_count;
            props.tpsa += fragment.properties.tpsa_contribution;
        }
        
        props
    }
    
    /// Calculate molecule score
    fn calculate_score(
        &self,
        fragments: &[MolecularFragment],
        connections: &[Connection],
    ) -> f64 {
        match &self.scoring_function {
            ScoringFunction::Additive { weights } => {
                let mut score = 0.0;
                let props = self.calculate_properties(fragments);
                
                // Property penalties
                if let Some((min, max)) = self.target_properties.molecular_weight {
                    if props.molecular_weight < min || props.molecular_weight > max {
                        score += weights.get("mw_penalty").unwrap_or(&0.0) * 
                                (props.molecular_weight - (min + max) / 2.0).abs();
                    }
                }
                
                // Fragment scores
                for fragment in fragments {
                    if let Some(&frag_score) = self.fragment_library.fragment_scores.get(&fragment.id) {
                        score += weights.get("fragment_score").unwrap_or(&1.0) * frag_score;
                    }
                }
                
                score
            }
            _ => 0.0,
        }
    }
}

impl Default for DesignConstraints {
    fn default() -> Self {
        Self {
            max_mw: 500.0,
            lipinski: true,
            veber: true,
            pains_filter: true,
            max_sa_score: 6.0,
            min_qed: 0.3,
            smarts_filters: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Connection {
    pub from_fragment: usize,
    pub from_attachment: usize,
    pub to_fragment: usize,
    pub to_attachment: usize,
    pub bond_type: BondType,
}

#[derive(Debug, Clone)]
pub struct DesignedMolecule {
    pub fragments: Vec<MolecularFragment>,
    pub connections: Vec<Connection>,
    pub properties: MolecularProperties,
    pub score: f64,
    pub smiles: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct MolecularProperties {
    pub molecular_weight: f64,
    pub logp: f64,
    pub logs: f64,
    pub hbd: usize,
    pub hba: usize,
    pub rotatable_bonds: usize,
    pub tpsa: f64,
    pub sa_score: f64,
    pub qed_score: f64,
}

/// Lead optimization
pub struct LeadOptimizer {
    /// Starting lead compound
    lead_compound: String,
    /// Optimization objectives
    objectives: Vec<OptimizationObjective>,
    /// Allowed modifications
    modifications: AllowedModifications,
    /// ADMET predictor
    admet_predictor: ADMETPredictor,
}

#[derive(Debug, Clone)]
pub enum OptimizationObjective {
    /// Improve potency
    Potency { target_ic50: f64 },
    /// Improve selectivity
    Selectivity { off_targets: Vec<String> },
    /// Improve ADMET
    ADMET { properties: Vec<ADMETProperty> },
    /// Reduce molecular weight
    ReduceMW { target_mw: f64 },
    /// Improve solubility
    Solubility { target_logs: f64 },
}

#[derive(Debug, Clone)]
pub struct AllowedModifications {
    /// Bioisosteric replacements
    pub bioisosteres: HashMap<String, Vec<String>>,
    /// Allowed R-group modifications
    pub r_groups: Vec<RGroupModification>,
    /// Scaffold hopping allowed
    pub scaffold_hopping: bool,
    /// Maximum similarity to lead
    pub max_similarity: f64,
}

#[derive(Debug, Clone)]
pub struct RGroupModification {
    /// Position in molecule
    pub position: String,
    /// Allowed substituents
    pub substituents: Vec<String>,
    /// Preferred properties
    pub preferred_properties: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct ADMETPredictor {
    /// Prediction models
    pub models: HashMap<ADMETProperty, PredictionModel>,
    /// Experimental data
    pub experimental_data: HashMap<String, ADMETProfile>,
}

#[derive(Debug, Clone)]
pub struct PredictionModel {
    /// Model type
    pub model_type: ModelType,
    /// Model parameters
    pub parameters: Vec<f64>,
    /// Accuracy metrics
    pub accuracy: f64,
}

#[derive(Debug, Clone)]
pub enum ModelType {
    /// Random forest
    RandomForest,
    /// Neural network
    NeuralNetwork,
    /// Support vector machine
    SVM,
    /// Physics-based
    PhysicsBased,
}

#[derive(Debug, Clone)]
pub struct ADMETProfile {
    /// Absorption properties
    pub absorption: AbsorptionProfile,
    /// Distribution properties
    pub distribution: DistributionProfile,
    /// Metabolism properties
    pub metabolism: MetabolismProfile,
    /// Excretion properties
    pub excretion: ExcretionProfile,
    /// Toxicity properties
    pub toxicity: ToxicityProfile,
}

#[derive(Debug, Clone)]
pub struct AbsorptionProfile {
    pub caco2_permeability: f64,
    pub pgp_substrate: bool,
    pub pgp_inhibitor: bool,
    pub oral_bioavailability: f64,
}

#[derive(Debug, Clone)]
pub struct DistributionProfile {
    pub plasma_protein_binding: f64,
    pub vd: f64, // Volume of distribution
    pub bbb_penetration: bool,
    pub tissue_distribution: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct MetabolismProfile {
    pub cyp_substrate: HashMap<String, bool>,
    pub cyp_inhibitor: HashMap<String, bool>,
    pub metabolic_stability: f64,
    pub major_metabolites: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ExcretionProfile {
    pub renal_clearance: f64,
    pub hepatic_clearance: f64,
    pub half_life: f64,
}

#[derive(Debug, Clone)]
pub struct ToxicityProfile {
    pub ld50: f64,
    pub mutagenicity: bool,
    pub hepatotoxicity: bool,
    pub cardiotoxicity: bool,
    pub herg_inhibition: f64,
}

/// Virtual screening engine
pub struct VirtualScreeningEngine {
    /// Compound library
    library: CompoundLibrary,
    /// Screening protocol
    protocol: ScreeningProtocol,
    /// Hit selection criteria
    hit_criteria: HitSelectionCriteria,
}

#[derive(Debug, Clone)]
pub struct CompoundLibrary {
    /// Library source
    pub source: LibrarySource,
    /// Number of compounds
    pub size: usize,
    /// Diversity metrics
    pub diversity: DiversityMetrics,
    /// Filters applied
    pub filters: Vec<LibraryFilter>,
}

#[derive(Debug, Clone)]
pub enum LibrarySource {
    /// Commercial vendor
    Commercial { vendor: String },
    /// FDA approved drugs
    FDAApproved,
    /// Natural products
    NaturalProducts,
    /// Fragment library
    Fragments,
    /// Virtual enumeration
    Virtual { rules: Vec<EnumerationRule> },
}

#[derive(Debug, Clone)]
pub struct DiversityMetrics {
    pub scaffold_diversity: f64,
    pub property_coverage: f64,
    pub pharmacophore_coverage: f64,
}

#[derive(Debug, Clone)]
pub enum LibraryFilter {
    /// Molecular weight range
    MolecularWeight { min: f64, max: f64 },
    /// Lipinski compliance
    Lipinski,
    /// PAINS removal
    PAINS,
    /// Custom SMARTS
    SMARTS { pattern: String, exclude: bool },
}

#[derive(Debug, Clone)]
pub struct EnumerationRule {
    /// Core scaffold
    pub scaffold: String,
    /// Variation points
    pub variation_points: Vec<VariationPoint>,
    /// Enumeration strategy
    pub strategy: EnumerationStrategy,
}

#[derive(Debug, Clone)]
pub struct VariationPoint {
    /// Position in scaffold
    pub position: String,
    /// Available building blocks
    pub building_blocks: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum EnumerationStrategy {
    /// Exhaustive enumeration
    Exhaustive,
    /// Random sampling
    RandomSampling { size: usize },
    /// Focused enumeration
    Focused { criteria: Vec<String> },
}

#[derive(Debug, Clone)]
pub enum ScreeningProtocol {
    /// Structure-based
    StructureBased {
        receptor: ProteinStructure,
        docking_program: DockingProgram,
        scoring_function: String,
    },
    /// Ligand-based
    LigandBased {
        reference_ligands: Vec<String>,
        similarity_metric: SimilarityMetric,
        threshold: f64,
    },
    /// Pharmacophore-based
    PharmacaophoreBased {
        pharmacophore: PharmacophoreModel,
        tolerance: f64,
    },
    /// Machine learning
    MachineLearning {
        model: String,
        features: Vec<String>,
    },
}

#[derive(Debug, Clone)]
pub enum DockingProgram {
    AutoDock,
    Glide,
    FlexX,
    GOLD,
    Vina,
}

#[derive(Debug, Clone)]
pub enum SimilarityMetric {
    Tanimoto,
    Dice,
    Cosine,
    Euclidean,
}

#[derive(Debug, Clone)]
pub struct HitSelectionCriteria {
    /// Score threshold
    pub score_threshold: f64,
    /// Top N compounds
    pub top_n: Option<usize>,
    /// Diversity selection
    pub diversity_selection: bool,
    /// Visual inspection required
    pub manual_inspection: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_molecular_design() {
        let target = TargetProperties {
            molecular_weight: Some((300.0, 500.0)),
            logp: Some((2.0, 5.0)),
            logs: None,
            hbd: Some((0, 5)),
            hba: Some((0, 10)),
            rotatable_bonds: Some((0, 10)),
            tpsa: Some((40.0, 140.0)),
            custom_descriptors: HashMap::new(),
        };
        
        let mut fragments = Vec::new();
        for i in 0..5 {
            fragments.push(MolecularFragment {
                id: i,
                smiles: format!("C{}O", "C".repeat(i)),
                attachment_points: vec![
                    AttachmentPoint {
                        atom_idx: 0,
                        bond_types: vec![BondType::Single],
                        direction: Vec3D { x: 1.0, y: 0.0, z: 0.0 },
                    }
                ],
                properties: FragmentProperties {
                    mw_contribution: 50.0 + i as f64 * 14.0,
                    logp_contribution: 0.5 + i as f64 * 0.5,
                    hbd_count: 1,
                    hba_count: 1,
                    rotatable_count: i,
                    tpsa_contribution: 20.0,
                },
                pharmacophores: vec![],
            });
        }
        
        let library = FragmentLibrary {
            fragments,
            connection_rules: ConnectionRules {
                compatible_pairs: HashMap::new(),
                forbidden_connections: HashSet::new(),
                reaction_templates: vec![],
            },
            fragment_scores: HashMap::new(),
            privileged_scaffolds: vec![],
        };
        
        let optimizer = MolecularDesignOptimizer::new(target, library);
        let result = optimizer.build_qubo();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_fragment_growing() {
        let core = MolecularFragment {
            id: 999,
            smiles: "c1ccccc1".to_string(),
            attachment_points: vec![
                AttachmentPoint {
                    atom_idx: 0,
                    bond_types: vec![BondType::Single, BondType::Aromatic],
                    direction: Vec3D { x: 1.0, y: 0.0, z: 0.0 },
                },
                AttachmentPoint {
                    atom_idx: 3,
                    bond_types: vec![BondType::Single, BondType::Aromatic],
                    direction: Vec3D { x: -1.0, y: 0.0, z: 0.0 },
                },
            ],
            properties: FragmentProperties {
                mw_contribution: 78.0,
                logp_contribution: 2.0,
                hbd_count: 0,
                hba_count: 0,
                rotatable_count: 0,
                tpsa_contribution: 0.0,
            },
            pharmacophores: vec![
                PharmacophoreFeature {
                    feature_type: PharmacophoreType::Aromatic,
                    position: Vec3D { x: 0.0, y: 0.0, z: 0.0 },
                    tolerance: 1.0,
                }
            ],
        };
        
        let target = TargetProperties {
            molecular_weight: Some((200.0, 400.0)),
            logp: Some((1.0, 4.0)),
            logs: None,
            hbd: Some((0, 3)),
            hba: Some((0, 6)),
            rotatable_bonds: None,
            tpsa: None,
            custom_descriptors: HashMap::new(),
        };
        
        let library = FragmentLibrary {
            fragments: vec![
                MolecularFragment {
                    id: 0,
                    smiles: "CCO".to_string(),
                    attachment_points: vec![
                        AttachmentPoint {
                            atom_idx: 0,
                            bond_types: vec![BondType::Single],
                            direction: Vec3D { x: 1.0, y: 0.0, z: 0.0 },
                        }
                    ],
                    properties: FragmentProperties {
                        mw_contribution: 45.0,
                        logp_contribution: 0.2,
                        hbd_count: 1,
                        hba_count: 1,
                        rotatable_count: 1,
                        tpsa_contribution: 20.0,
                    },
                    pharmacophores: vec![],
                },
            ],
            connection_rules: ConnectionRules {
                compatible_pairs: HashMap::new(),
                forbidden_connections: HashSet::new(),
                reaction_templates: vec![],
            },
            fragment_scores: {
                let mut scores = HashMap::new();
                scores.insert(0, 1.0);
                scores
            },
            privileged_scaffolds: vec![],
        };
        
        let optimizer = MolecularDesignOptimizer::new(target, library)
            .with_strategy(OptimizationStrategy::FragmentGrowing { core });
        
        let result = optimizer.build_qubo();
        assert!(result.is_ok());
        
        let (qubo, var_map) = result.unwrap();
        assert!(var_map.len() > 0);
    }
}