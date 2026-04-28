//! Ising Ground State via Simulated Annealing
//!
//! Demonstrates finding the ground state of a frustrated 1-D Ising chain
//! with random couplings using the ClassicalAnnealingSimulator.
//!
//! Ising Hamiltonian:
//!   H = Σ_i h_i * s_i  +  Σ_{i<j} J_{ij} * s_i * s_j
//! where s_i ∈ {-1, +1}.
//!
//! For a 1-D antiferromagnetic chain (J < 0 for nearest-neighbours), the
//! exact ground state alternates spins: +1,-1,+1,-1,... with energy
//! = -(N-1)*|J|.
//!
//! Run with:
//!   cargo run --example ising_ground_state -p quantrs2-anneal --all-features

use quantrs2_anneal::{
    ising::IsingModel,
    simulator::{AnnealingParams, ClassicalAnnealingSimulator},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 20-Spin Ising Ground State via Simulated Annealing ===\n");

    let n_spins = 20usize;

    // ---- Problem 1: Antiferromagnetic chain ----
    // Nearest-neighbour coupling J = -1 → ground state alternates ±1
    println!("Problem 1: Antiferromagnetic 1-D chain (N={n_spins})");
    println!("  Expected ground energy: -{} (alternating spins)", n_spins - 1);

    let mut af_model = IsingModel::new(n_spins);
    for i in 0..(n_spins - 1) {
        // Negative coupling → energy lowered when neighbours are antiparallel
        af_model.set_coupling(i, i + 1, -1.0)?;
    }

    let mut params = AnnealingParams::new();
    params.num_sweeps = 2000;
    params.num_repetitions = 20;
    params.initial_temperature = 4.0;
    params.final_temperature = 0.01;
    params.seed = Some(42);

    let solver = ClassicalAnnealingSimulator::new(params)?;
    let solution = solver.solve(&af_model)?;

    println!("  Best energy found : {:.4}", solution.best_energy);
    println!(
        "  Best spins (first 10): {:?}",
        &solution.best_spins[..10]
    );
    println!(
        "  Runtime           : {:.2?}",
        solution.runtime
    );

    // Verify near-optimal solution
    let expected = -((n_spins - 1) as f64);
    let gap = (solution.best_energy - expected).abs();
    println!("  Gap from optimum  : {gap:.4}");

    // ---- Problem 2: Ferromagnetic ring + random longitudinal fields ----
    // Ground state is all-up or all-down (broken by fields)
    println!("\nProblem 2: Ferromagnetic ring with random longitudinal fields");

    let mut fm_model = IsingModel::new(n_spins);
    // Ring: all couplings +1 → energy lowered when neighbours are parallel
    for i in 0..n_spins {
        fm_model.set_coupling(i, (i + 1) % n_spins, 1.0)?;
    }
    // Small longitudinal fields (break Z2 symmetry, guide optimiser)
    for i in 0..n_spins {
        let h = if i % 2 == 0 { 0.05 } else { -0.05 };
        fm_model.set_bias(i, h)?;
    }

    let mut params2 = AnnealingParams::new();
    params2.num_sweeps = 3000;
    params2.num_repetitions = 30;
    params2.initial_temperature = 6.0;
    params2.final_temperature = 0.005;
    params2.seed = Some(99);

    let solver2 = ClassicalAnnealingSimulator::new(params2)?;
    let solution2 = solver2.solve(&fm_model)?;

    println!("  Best energy found : {:.4}", solution2.best_energy);
    println!(
        "  Best spins (first 10): {:?}",
        &solution2.best_spins[..10]
    );

    // Count domain walls (adjacent antiparallel spins) — ferromagnetic ground state has 0
    let domain_walls: usize = (0..n_spins)
        .filter(|&i| solution2.best_spins[i] != solution2.best_spins[(i + 1) % n_spins])
        .count();
    println!("  Domain walls      : {domain_walls}");

    // ---- Summary ----
    println!("\n=== Summary ===");
    println!(
        "  Antiferromagnetic: energy={:.4}  (target={})",
        solution.best_energy,
        -((n_spins - 1) as f64)
    );
    println!(
        "  Ferromagnetic ring: energy={:.4}, domain_walls={domain_walls}",
        solution2.best_energy
    );

    assert!(
        gap < 1.0,
        "Antiferromagnetic chain: expected near-optimal solution, gap={gap}"
    );
    println!("\nOK");

    Ok(())
}
