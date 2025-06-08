//! Advanced CIM demonstration with quantum noise and adaptive pumping.

use quantrs2_tytan::coherent_ising_machine::*;
use quantrs2_tytan::sampler::Sampler;
use ndarray::{Array2, Array1};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced CIM Demo ===\n");

    // Create a frustrated Ising problem
    let n = 6;
    let mut coupling = Array2::zeros((n, n));
    
    // Create frustration: triangular lattice with antiferromagnetic coupling
    let edges = vec![
        (0, 1), (1, 2), (2, 0),  // Triangle 1
        (3, 4), (4, 5), (5, 3),  // Triangle 2
        (0, 3), (1, 4), (2, 5),  // Inter-triangle connections
    ];
    
    for (i, j) in edges {
        coupling[[i, j]] = -1.0;  // Antiferromagnetic
        coupling[[j, i]] = -1.0;
    }
    
    let biases = Array1::zeros(n);
    
    println!("Problem: Frustrated triangular lattice");
    println!("  Nodes: {}", n);
    println!("  Antiferromagnetic couplings: {}\n", edges.len());

    // 1. Basic CIM
    println!("1. Basic CIM:");
    run_basic_cim(&coupling, &biases)?;
    
    // 2. Advanced CIM with pulse shaping
    println!("\n2. Advanced CIM with Pulse Shaping:");
    run_pulse_shaped_cim(&coupling, &biases)?;
    
    // 3. CIM with error correction
    println!("\n3. CIM with Error Correction:");
    run_error_corrected_cim(&coupling, &biases)?;
    
    // 4. Networked CIM for larger problems
    println!("\n4. Networked CIM:");
    run_networked_cim()?;
    
    // 5. CIM with bifurcation control
    println!("\n5. CIM with Adaptive Bifurcation:");
    run_bifurcation_controlled_cim(&coupling, &biases)?;

    Ok(())
}

fn run_basic_cim(coupling: &Array2<f64>, biases: &Array1<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let mut cim = CIMSimulator::new(coupling.shape()[0])
        .with_pump_parameter(1.5)
        .with_evolution_time(20.0)
        .with_noise_strength(0.1)
        .with_seed(42);
    
    let start = std::time::Instant::now();
    let results = cim.solve_ising(coupling, biases, 10)?;
    let elapsed = start.elapsed();
    
    println!("  Time: {:.2}ms", elapsed.as_millis());
    println!("  Solutions found: {}", results.len());
    
    if let Some(best) = results.first() {
        println!("  Best energy: {:.4}", best.energy);
        println!("  Ground state: {:?}", best.sample);
    }
    
    Ok(())
}

fn run_pulse_shaped_cim(coupling: &Array2<f64>, biases: &Array1<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let cim = AdvancedCIM::new(coupling.shape()[0])
        .with_pulse_shape(PulseShape::Gaussian { 
            width: 2.0, 
            amplitude: 2.0 
        })
        .with_num_rounds(3);
    
    let qubo = ising_to_qubo(coupling, biases);
    
    let start = std::time::Instant::now();
    let results = cim.run_qubo(&qubo, 10)?;
    let elapsed = start.elapsed();
    
    println!("  Pulse shape: Gaussian (width=2.0, amplitude=2.0)");
    println!("  Rounds: 3");
    println!("  Time: {:.2}ms", elapsed.as_millis());
    println!("  Solutions found: {}", results.len());
    
    if let Some(best) = results.first() {
        println!("  Best energy: {:.4}", best.energy);
    }
    
    Ok(())
}

fn run_error_corrected_cim(coupling: &Array2<f64>, biases: &Array1<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let n = coupling.shape()[0];
    
    // Create parity check matrix for simple repetition code
    let mut check_matrix = Array2::from_elem((n/2, n), false);
    for i in 0..n/2 {
        check_matrix[[i, 2*i]] = true;
        check_matrix[[i, 2*i + 1]] = true;
    }
    
    let cim = AdvancedCIM::new(n)
        .with_error_correction(ErrorCorrectionScheme::ParityCheck { 
            check_matrix 
        })
        .with_pulse_shape(PulseShape::Sech { 
            width: 1.5, 
            amplitude: 1.8 
        });
    
    let qubo = ising_to_qubo(coupling, biases);
    
    let start = std::time::Instant::now();
    let results = cim.run_qubo(&qubo, 10)?;
    let elapsed = start.elapsed();
    
    println!("  Error correction: Parity check");
    println!("  Pulse shape: Hyperbolic secant");
    println!("  Time: {:.2}ms", elapsed.as_millis());
    println!("  Solutions found: {}", results.len());
    
    if let Some(best) = results.first() {
        println!("  Best energy: {:.4}", best.energy);
    }
    
    Ok(())
}

fn run_networked_cim() -> Result<(), Box<dyn std::error::Error>> {
    // Create a larger problem that will be distributed
    let total_spins = 12;
    let modules = 4;
    let spins_per_module = total_spins / modules;
    
    let net_cim = NetworkedCIM::new(
        modules,
        spins_per_module,
        NetworkTopology::Ring,
    )
    .with_sync_scheme(SynchronizationScheme::BlockSynchronous { 
        block_size: 2 
    })
    .with_comm_delay(0.1);
    
    println!("  Network topology: Ring");
    println!("  Modules: {}", modules);
    println!("  Spins per module: {}", spins_per_module);
    println!("  Synchronization: Block synchronous");
    
    // Would solve a problem here in full implementation
    println!("  ✓ Networked CIM configured successfully");
    
    Ok(())
}

fn run_bifurcation_controlled_cim(coupling: &Array2<f64>, biases: &Array1<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let cim = AdvancedCIM::new(coupling.shape()[0])
        .with_bifurcation_control(BifurcationControl {
            initial_param: 0.0,
            final_param: 2.5,
            ramp_time: 15.0,
            ramp_type: RampType::Sigmoid,
        })
        .with_error_correction(ErrorCorrectionScheme::MajorityVoting { 
            window_size: 3 
        });
    
    let qubo = ising_to_qubo(coupling, biases);
    
    let start = std::time::Instant::now();
    let results = cim.run_qubo(&qubo, 10)?;
    let elapsed = start.elapsed();
    
    println!("  Bifurcation: Sigmoid ramp (0.0 → 2.5)");
    println!("  Error correction: Majority voting (window=3)");
    println!("  Time: {:.2}ms", elapsed.as_millis());
    println!("  Solutions found: {}", results.len());
    
    if let Some(best) = results.first() {
        println!("  Best energy: {:.4}", best.energy);
        analyze_solution(&best, coupling.shape()[0]);
    }
    
    Ok(())
}

fn ising_to_qubo(j_matrix: &Array2<f64>, h_vector: &Array1<f64>) -> (Array2<f64>, HashMap<String, usize>) {
    let n = j_matrix.shape()[0];
    let mut qubo = Array2::zeros((n, n));
    let mut var_map = HashMap::new();
    
    // Convert Ising to QUBO: x_i = (s_i + 1) / 2
    for i in 0..n {
        var_map.insert(format!("s{}", i), i);
        
        // Linear terms
        let mut linear = h_vector[i];
        for j in 0..n {
            if i != j {
                linear += j_matrix[[i, j]];
            }
        }
        qubo[[i, i]] = -2.0 * linear;
        
        // Quadratic terms
        for j in i+1..n {
            qubo[[i, j]] = 4.0 * j_matrix[[i, j]];
            qubo[[j, i]] = 4.0 * j_matrix[[i, j]];
        }
    }
    
    (qubo, var_map)
}

fn analyze_solution(result: &quantrs2_tytan::sampler::SampleResult, n: usize) {
    // Analyze frustration in the solution
    let spins: Vec<i32> = (0..n)
        .map(|i| {
            let var = format!("s{}", i);
            if result.assignments.get(&var).copied().unwrap_or(false) {
                1
            } else {
                -1
            }
        })
        .collect();
    
    // Check triangles for frustration
    let triangles = vec![
        (0, 1, 2),
        (3, 4, 5),
    ];
    
    println!("\n  Solution analysis:");
    for (i, (a, b, c)) in triangles.iter().enumerate() {
        let product = spins[*a] * spins[*b] * spins[*c];
        let frustrated = product > 0;  // Positive product means frustrated
        println!("    Triangle {}: {} (spins: {}, {}, {})", 
                 i + 1,
                 if frustrated { "frustrated" } else { "satisfied" },
                 spins[*a], spins[*b], spins[*c]);
    }
}