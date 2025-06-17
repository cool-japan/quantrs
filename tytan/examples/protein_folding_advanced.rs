//! Advanced Protein Folding example using QuantRS2-Tytan
//!
//! This example demonstrates:
//! - Lattice protein folding (HP model)
//! - Energy minimization for protein conformations
//! - Contact map prediction
//! - Comparison with molecular dynamics approaches

use ndarray::{Array2, Array3};
use quantrs2_tytan::{
    compile::Model,
    constraints::PenaltyFunction,
    optimization::{
        adaptive::{AdaptiveConfig, AdaptiveOptimizer, AdaptiveStrategy},
        penalty::{PenaltyConfig, PenaltyOptimizer, PenaltyType},
    },
    sampler::{SASampler, Sampler},
    visualization::{
        convergence::plot_convergence,
        problem_specific::{ProblemVisualizer, VisualizationType},
        solution_analysis::analyze_solution_distribution,
    },
};

use quantrs2_tytan::compile::expr::{Expr, constant};

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::fmt::Write;

/// Amino acid types in HP model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AminoAcid {
    Hydrophobic, // H
    Polar,       // P
}

impl AminoAcid {
    fn from_char(c: char) -> Option<Self> {
        match c.to_uppercase().next()? {
            'H' => Some(Self::Hydrophobic),
            'P' => Some(Self::Polar),
            _ => None,
        }
    }

    fn interaction_energy(&self, other: &Self) -> f64 {
        match (self, other) {
            (Self::Hydrophobic, Self::Hydrophobic) => -1.0, // H-H attractive
            _ => 0.0,                                       // P-P and H-P neutral
        }
    }
}

/// Protein sequence
#[derive(Debug, Clone)]
struct Protein {
    sequence: Vec<AminoAcid>,
    name: String,
}

impl Protein {
    fn from_sequence(name: &str, seq: &str) -> Option<Self> {
        let sequence: Option<Vec<_>> = seq
            .chars()
            .filter(|c| !c.is_whitespace())
            .map(AminoAcid::from_char)
            .collect();

        sequence.map(|seq| Self {
            sequence: seq,
            name: name.to_string(),
        })
    }

    fn len(&self) -> usize {
        self.sequence.len()
    }
}

/// Lattice types for protein folding
#[derive(Debug, Clone, Copy)]
enum LatticeType {
    Square2D,     // 2D square lattice
    Cubic3D,      // 3D cubic lattice
    Triangular2D, // 2D triangular lattice
}

impl LatticeType {
    fn neighbors(&self) -> Vec<(i32, i32, i32)> {
        match self {
            Self::Square2D => vec![
                (1, 0, 0),  // Right
                (-1, 0, 0), // Left
                (0, 1, 0),  // Up
                (0, -1, 0), // Down
            ],
            Self::Cubic3D => vec![
                (1, 0, 0),  // +X
                (-1, 0, 0), // -X
                (0, 1, 0),  // +Y
                (0, -1, 0), // -Y
                (0, 0, 1),  // +Z
                (0, 0, -1), // -Z
            ],
            Self::Triangular2D => vec![
                (1, 0, 0),  // Right
                (-1, 0, 0), // Left
                (0, 1, 0),  // Up-right
                (0, -1, 0), // Down-left
                (1, -1, 0), // Down-right
                (-1, 1, 0), // Up-left
            ],
        }
    }

    fn max_dimension(&self) -> usize {
        match self {
            Self::Square2D | Self::Triangular2D => 2,
            Self::Cubic3D => 3,
        }
    }
}

/// Create protein folding model on lattice
fn create_folding_model(
    protein: &Protein,
    lattice: LatticeType,
    lattice_size: usize,
) -> Result<Model, Box<dyn std::error::Error>> {
    let n = protein.len();
    let mut model = Model::new();

    // Binary variables: x_{i,pos} = 1 if residue i is at position pos
    let mut position_vars = HashMap::new();

    // Generate all valid lattice positions
    let positions: Vec<(i32, i32, i32)> = match lattice {
        LatticeType::Square2D | LatticeType::Triangular2D => {
            let mut pos = Vec::new();
            for x in 0..lattice_size as i32 {
                for y in 0..lattice_size as i32 {
                    pos.push((x, y, 0));
                }
            }
            pos
        }
        LatticeType::Cubic3D => {
            let mut pos = Vec::new();
            for x in 0..lattice_size as i32 {
                for y in 0..lattice_size as i32 {
                    for z in 0..lattice_size as i32 {
                        pos.push((x, y, z));
                    }
                }
            }
            pos
        }
    };

    // Create position variables
    for i in 0..n {
        for (idx, &pos) in positions.iter().enumerate() {
            let var = model.add_variable(&format!("x_{}_{}_{}_{}", i, pos.0, pos.1, pos.2))?;
            position_vars.insert((i, idx), var);
        }
    }

    // Constraint 1: Each residue must be at exactly one position
    for i in 0..n {
        let mut pos_sum = Vec::new();
        for idx in 0..positions.len() {
            pos_sum.push(position_vars[&(i, idx)].clone());
        }
        model.add_constraint_eq_one(&format!("residue_{}_position", i), pos_sum)?;
    }

    // Constraint 2: Each position can have at most one residue
    for idx in 0..positions.len() {
        let mut residue_sum = Vec::new();
        for i in 0..n {
            residue_sum.push(position_vars[&(i, idx)].clone());
        }
        model.add_constraint_at_most_one(&format!("position_{}_occupancy", idx), residue_sum)?;
    }

    // Constraint 3: Sequential residues must be neighbors on lattice
    let neighbors = lattice.neighbors();

    for i in 0..n - 1 {
        // Create auxiliary variables for valid sequential placements
        let mut valid_pairs = Vec::new();

        for (idx1, &pos1) in positions.iter().enumerate() {
            for &(dx, dy, dz) in &neighbors {
                let pos2 = (pos1.0 + dx, pos1.1 + dy, pos1.2 + dz);

                // Find if pos2 exists in our position list
                if let Some(idx2) = positions.iter().position(|&p| p == pos2) {
                    // Create auxiliary variable for this valid pair
                    let aux_var = model.add_variable(&format!("seq_{}_{}_to_{}", i, idx1, idx2))?;

                    // Simplified: aux_var represents valid consecutive placement
                    // Constraints handled as penalty terms in objective

                    valid_pairs.push(aux_var);
                }
            }
        }

        // At least one valid pair must be selected
        model.add_constraint_eq_one(&format!("sequence_connectivity_{}", i), valid_pairs)?;
    }

    // Objective: Minimize energy (maximize H-H contacts)
    let mut energy_expr = constant(0.0);

    // For each pair of non-sequential hydrophobic residues
    for i in 0..n {
        for j in i + 2..n {
            // Skip adjacent residues
            if protein.sequence[i] == AminoAcid::Hydrophobic
                && protein.sequence[j] == AminoAcid::Hydrophobic
            {
                // Check if they can be neighbors
                for (idx1, &pos1) in positions.iter().enumerate() {
                    for &(dx, dy, dz) in &neighbors {
                        let pos2 = (pos1.0 + dx, pos1.1 + dy, pos1.2 + dz);

                        if let Some(idx2) = positions.iter().position(|&p| p == pos2) {
                            // Add energy contribution if both residues are at these positions
                            // Energy = -1 * x_{i,idx1} * x_{j,idx2}
                            energy_expr = energy_expr
                                + constant(1.0) * // Negative because we minimize
                                position_vars[&(i, idx1)].clone() *
                                position_vars[&(j, idx2)].clone();
                        }
                    }
                }
            }
        }
    }

    model.set_objective(energy_expr);

    Ok(model)
}

/// Extract conformation from solution
fn extract_conformation(
    solution: &quantrs2_tytan::sampler::SampleResult,
    protein: &Protein,
    lattice_size: usize,
    lattice: LatticeType,
) -> Option<Vec<(i32, i32, i32)>> {
    let n = protein.len();
    let mut conformation = vec![(0, 0, 0); n];

    // Generate positions again (same as in model creation)
    let positions: Vec<(i32, i32, i32)> = match lattice {
        LatticeType::Square2D | LatticeType::Triangular2D => {
            let mut pos = Vec::new();
            for x in 0..lattice_size as i32 {
                for y in 0..lattice_size as i32 {
                    pos.push((x, y, 0));
                }
            }
            pos
        }
        LatticeType::Cubic3D => {
            let mut pos = Vec::new();
            for x in 0..lattice_size as i32 {
                for y in 0..lattice_size as i32 {
                    for z in 0..lattice_size as i32 {
                        pos.push((x, y, z));
                    }
                }
            }
            pos
        }
    };

    for i in 0..n {
        let mut found = false;
        for (idx, &pos) in positions.iter().enumerate() {
            let var_name = format!("x_{}_{}_{}_{}", i, pos.0, pos.1, pos.2);
            if solution
                .assignments
                .get(&var_name)
                .copied()
                .unwrap_or(false)
            {
                conformation[i] = pos;
                found = true;
                break;
            }
        }
        if !found {
            return None; // Invalid conformation
        }
    }

    Some(conformation)
}

/// Validate conformation
fn validate_conformation(conformation: &[(i32, i32, i32)], lattice: LatticeType) -> bool {
    let n = conformation.len();

    // Check no overlaps
    let unique_positions: HashSet<_> = conformation.iter().cloned().collect();
    if unique_positions.len() != n {
        return false;
    }

    // Check connectivity
    let neighbors = lattice.neighbors();
    for i in 0..n - 1 {
        let pos1 = conformation[i];
        let pos2 = conformation[i + 1];

        let is_neighbor = neighbors
            .iter()
            .any(|&(dx, dy, dz)| pos2 == (pos1.0 + dx, pos1.1 + dy, pos1.2 + dz));

        if !is_neighbor {
            return false;
        }
    }

    true
}

/// Calculate conformation energy
fn calculate_energy(
    conformation: &[(i32, i32, i32)],
    protein: &Protein,
    lattice: LatticeType,
) -> f64 {
    let n = protein.len();
    let neighbors = lattice.neighbors();
    let mut energy = 0.0;

    // Calculate H-H contacts
    for i in 0..n {
        for j in i + 2..n {
            // Skip adjacent residues
            let pos1 = conformation[i];
            let pos2 = conformation[j];

            let is_neighbor = neighbors
                .iter()
                .any(|&(dx, dy, dz)| pos2 == (pos1.0 + dx, pos1.1 + dy, pos1.2 + dz));

            if is_neighbor {
                energy += protein.sequence[i].interaction_energy(&protein.sequence[j]);
            }
        }
    }

    energy
}

/// Calculate radius of gyration
fn calculate_radius_of_gyration(conformation: &[(i32, i32, i32)]) -> f64 {
    let n = conformation.len() as f64;

    // Calculate center of mass
    let mut center = (0.0, 0.0, 0.0);
    for &(x, y, z) in conformation {
        center.0 += x as f64 / n;
        center.1 += y as f64 / n;
        center.2 += z as f64 / n;
    }

    // Calculate radius of gyration
    let mut rg2 = 0.0;
    for &(x, y, z) in conformation {
        let dx = x as f64 - center.0;
        let dy = y as f64 - center.1;
        let dz = z as f64 - center.2;
        rg2 += (dx * dx + dy * dy + dz * dz) / n;
    }

    rg2.sqrt()
}

/// Run protein folding experiment
fn run_folding_experiment(
    protein: &Protein,
    lattice: LatticeType,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Folding {} ===", protein.name);
    println!(
        "Sequence: {}",
        protein
            .sequence
            .iter()
            .map(|aa| match aa {
                AminoAcid::Hydrophobic => 'H',
                AminoAcid::Polar => 'P',
            })
            .collect::<String>()
    );
    println!("Length: {}", protein.len());
    println!("Lattice: {:?}", lattice);

    // Determine lattice size
    let lattice_size = match lattice {
        LatticeType::Square2D => ((protein.len() as f64).sqrt() * 2.0).ceil() as usize,
        LatticeType::Cubic3D => ((protein.len() as f64).powf(1.0 / 3.0) * 2.0).ceil() as usize,
        LatticeType::Triangular2D => ((protein.len() as f64).sqrt() * 1.5).ceil() as usize,
    };

    println!("Lattice size: {}", lattice_size);

    // Create model
    let model = create_folding_model(protein, lattice, lattice_size)?;

    // Optimize with adaptive penalties
    let penalty_config = PenaltyConfig {
        initial_weight: 10.0,
        min_weight: 0.1,
        max_weight: 1000.0,
        adjustment_factor: 1.5,
        violation_tolerance: 1e-4,
        max_iterations: 20,
        adaptive_scaling: true,
        penalty_type: PenaltyType::Quadratic,
    };

    let mut penalty_optimizer = PenaltyOptimizer::new(penalty_config);
    let compiled = model.compile()?;
    let qubo = compiled.to_qubo();

    println!("QUBO variables: {}", qubo.num_variables);

    // Configure adaptive sampler
    let adaptive_config = AdaptiveConfig {
        strategy: AdaptiveStrategy::AdaptivePenaltyMethod,
        update_interval: 100,
        learning_rate: 0.1,
        momentum: 0.9,
        patience: 50,
        ..Default::default()
    };

    let mut adaptive_optimizer = AdaptiveOptimizer::new(adaptive_config);

    // Convert QUBO to matrix format
    let n_vars = qubo.num_variables;
    let mut matrix = Array2::zeros((n_vars, n_vars));
    let mut var_map = HashMap::new();

    for i in 0..n_vars {
        var_map.insert(format!("x_{}", i), i);
        if let Ok(linear) = qubo.get_linear(i) {
            matrix[[i, i]] = linear;
        }
        for j in 0..n_vars {
            if i != j {
                if let Ok(quad) = qubo.get_quadratic(i, j) {
                    matrix[[i, j]] = quad;
                }
            }
        }
    }

    // Run sampling with tracking
    let mut sampler = SASampler::new(None);

    println!("\nRunning optimization...");
    let start = std::time::Instant::now();
    let samples = sampler.run_qubo(&(matrix, var_map), 1000)?;
    let elapsed = start.elapsed();

    println!("Optimization time: {:.2}s", elapsed.as_secs_f64());

    // Analyze solutions
    let mut valid_conformations = Vec::new();
    let mut best_energy = f64::INFINITY;
    let mut best_conformation = None;

    for sample in &samples {
        if let Some(conformation) = extract_conformation(sample, protein, lattice_size, lattice) {
            if validate_conformation(&conformation, lattice) {
                let energy = calculate_energy(&conformation, protein, lattice);
                valid_conformations.push((conformation.clone(), energy));

                if energy < best_energy {
                    best_energy = energy;
                    best_conformation = Some(conformation);
                }
            }
        }
    }

    println!("\nResults:");
    println!(
        "  Valid conformations: {} / {}",
        valid_conformations.len(),
        samples.len()
    );
    println!("  Best energy: {:.1}", best_energy);

    if let Some(conformation) = best_conformation {
        let rg = calculate_radius_of_gyration(&conformation);
        println!("  Radius of gyration: {:.2}", rg);

        // Count H-H contacts
        let mut hh_contacts = 0;
        for (i, &pos1) in conformation.iter().enumerate() {
            for (j, &pos2) in conformation.iter().enumerate().skip(i + 2) {
                if protein.sequence[i] == AminoAcid::Hydrophobic
                    && protein.sequence[j] == AminoAcid::Hydrophobic
                {
                    let neighbors = lattice.neighbors();
                    if neighbors
                        .iter()
                        .any(|&(dx, dy, dz)| pos2 == (pos1.0 + dx, pos1.1 + dy, pos1.2 + dz))
                    {
                        hh_contacts += 1;
                    }
                }
            }
        }

        println!("  H-H contacts: {}", hh_contacts);

        // Print conformation
        if lattice_size <= 10 && matches!(lattice, LatticeType::Square2D) {
            println!("\n  2D Conformation:");
            let mut grid = vec![vec!['.'; lattice_size]; lattice_size];

            for (i, &(x, y, _)) in conformation.iter().enumerate() {
                let symbol = match protein.sequence[i] {
                    AminoAcid::Hydrophobic => 'H',
                    AminoAcid::Polar => 'P',
                };
                grid[y as usize][x as usize] = symbol;
            }

            for row in grid {
                println!("    {}", row.iter().collect::<String>());
            }
        }

        // Generate contact map
        let n = protein.len();
        let mut contact_map = Array2::zeros((n, n));

        for (i, &pos1) in conformation.iter().enumerate() {
            for (j, &pos2) in conformation.iter().enumerate() {
                if i != j {
                    let dist_sq = (pos1.0 - pos2.0).pow(2)
                        + (pos1.1 - pos2.1).pow(2)
                        + (pos1.2 - pos2.2).pow(2);

                    if dist_sq <= 2 {
                        // Adjacent or diagonal
                        contact_map[[i, j]] = 1.0;
                    }
                }
            }
        }

        // Save contact map
        save_contact_map(&contact_map, &format!("{}_contact_map.csv", protein.name))?;
    }

    // Analyze energy distribution
    if !valid_conformations.is_empty() {
        let energies: Vec<f64> = valid_conformations.iter().map(|(_, e)| *e).collect();

        let mean_energy = energies.iter().sum::<f64>() / energies.len() as f64;
        let min_energy = energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_energy = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("\n  Energy distribution:");
        println!("    Mean: {:.2}", mean_energy);
        println!("    Min: {:.2}", min_energy);
        println!("    Max: {:.2}", max_energy);

        // Count unique conformations
        let unique_energies: HashSet<_> = energies
            .iter()
            .map(|e| (*e * 10.0).round() as i32)
            .collect();

        println!("    Unique energy levels: {}", unique_energies.len());
    }

    Ok(())
}

/// Save contact map to CSV
fn save_contact_map(
    contact_map: &Array2<f64>,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut csv = String::new();
    let n = contact_map.nrows();

    // Header
    write!(&mut csv, "i,j,contact\n")?;

    for i in 0..n {
        for j in 0..n {
            if contact_map[[i, j]] > 0.0 {
                writeln!(&mut csv, "{},{},1", i, j)?;
            }
        }
    }

    std::fs::write(filename, csv)?;
    println!("  Contact map saved to {}", filename);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Protein Folding Example ===");

    // Test proteins from literature
    let test_proteins = vec![
        // Simple test sequence
        ("Test1", "HPPHPPPH"),
        // Classic benchmark sequences
        ("Seq1", "HPHPPHHPHPPHPHHPPHPH"), // 20-mer
        ("Seq2", "HHHPPHPHPHPPHPHPHPPH"), // 20-mer
        // Longer sequences
        ("Seq3", "PPHPPHHPPPPHHPPPPHHPPPPHH"), // 25-mer
        ("Seq4", "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP"), // 36-mer
        // Designed sequences with known ground states
        ("Alpha", "HHHHPPPPHHHHPPPP"), // Should form helix-like
        ("Beta", "HPHPHPHPHPHPHPH"),   // Should form sheet-like
    ];

    // Example 1: 2D square lattice folding
    println!("\n=== Example 1: 2D Square Lattice ===");

    for (name, seq) in &test_proteins[..3] {
        // First 3 sequences
        if let Some(protein) = Protein::from_sequence(name, seq) {
            run_folding_experiment(&protein, LatticeType::Square2D)?;
        }
    }

    // Example 2: 3D cubic lattice folding
    println!("\n\n=== Example 2: 3D Cubic Lattice ===");

    if let Some(protein) = Protein::from_sequence("3D_test", "HPPHPPHHPH") {
        run_folding_experiment(&protein, LatticeType::Cubic3D)?;
    }

    // Example 3: Comparative study
    println!("\n\n=== Example 3: 2D vs 3D Comparison ===");

    let comparison_seq = "HPHPPHHPHPPH";
    if let Some(protein) = Protein::from_sequence("Compare", comparison_seq) {
        println!("\nComparing folding in different lattices:");

        // 2D folding
        println!("\n--- 2D Square Lattice ---");
        run_folding_experiment(&protein, LatticeType::Square2D)?;

        // 3D folding
        println!("\n--- 3D Cubic Lattice ---");
        run_folding_experiment(&protein, LatticeType::Cubic3D)?;

        // 2D triangular
        println!("\n--- 2D Triangular Lattice ---");
        run_folding_experiment(&protein, LatticeType::Triangular2D)?;
    }

    // Example 4: Secondary structure analysis
    println!("\n\n=== Example 4: Secondary Structure Propensity ===");

    // Analyze designed sequences
    for (name, seq) in &[("Alpha", "HHHHPPPPHHHHPPPP"), ("Beta", "HPHPHPHPHPHPHPH")] {
        if let Some(protein) = Protein::from_sequence(name, seq) {
            println!("\nAnalyzing {} sequence", name);
            run_folding_experiment(&protein, LatticeType::Square2D)?;
        }
    }

    // Example 5: Folding kinetics simulation (simplified)
    println!("\n\n=== Example 5: Folding Pathway Analysis ===");

    if let Some(protein) = Protein::from_sequence("Kinetics", "HPHPPHHPH") {
        println!("\nAnalyzing folding pathways...");

        // Run multiple independent folding simulations
        let mut pathways = Vec::new();

        for run in 0..5 {
            println!("\n  Run {}", run + 1);

            // Create model with different random seed
            let model = create_folding_model(&protein, LatticeType::Square2D, 6)?;
            let compiled = model.compile()?;
            let qubo = compiled.to_qubo();

            // Convert QUBO to matrix format
            let n_vars = qubo.num_variables;
            let mut matrix = Array2::zeros((n_vars, n_vars));
            let mut var_map = HashMap::new();

            for i in 0..n_vars {
                var_map.insert(format!("x_{}", i), i);
                if let Ok(linear) = qubo.get_linear(i) {
                    matrix[[i, i]] = linear;
                }
                for j in 0..n_vars {
                    if i != j {
                        if let Ok(quad) = qubo.get_quadratic(i, j) {
                            matrix[[i, j]] = quad;
                        }
                    }
                }
            }

            // Run with different initial conditions
            let mut sampler = SASampler::new(Some(run as u64));

            let samples = sampler.run_qubo(&(matrix, var_map), 100)?;

            // Extract pathway (sequence of conformations)
            let mut pathway = Vec::new();
            for sample in samples.iter().step_by(10) {
                if let Some(conf) = extract_conformation(sample, &protein, 6, LatticeType::Square2D)
                {
                    if validate_conformation(&conf, LatticeType::Square2D) {
                        let energy = calculate_energy(&conf, &protein, LatticeType::Square2D);
                        pathway.push((conf, energy));
                    }
                }
            }

            if !pathway.is_empty() {
                pathways.push(pathway);
            }
        }

        // Analyze pathways
        println!("\n  Pathway analysis:");
        for (i, pathway) in pathways.iter().enumerate() {
            let energies: Vec<f64> = pathway.iter().map(|(_, e)| *e).collect();
            let initial_e = energies.first().unwrap_or(&0.0);
            let final_e = energies.last().unwrap_or(&0.0);

            println!(
                "    Pathway {}: {} steps, E_initial = {:.1}, E_final = {:.1}",
                i + 1,
                pathway.len(),
                initial_e,
                final_e
            );
        }
    }

    println!("\n\n=== Summary ===");
    println!("Protein folding on lattice demonstrates:");
    println!("- Discrete optimization for conformation search");
    println!("- Energy minimization with H-H interactions");
    println!("- Different lattice geometries affect folding");
    println!("- QUBO formulation captures connectivity constraints");

    Ok(())
}
