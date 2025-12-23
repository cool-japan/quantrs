//! 3-Rooks Problem Solver
//!
//! This example shows how to use quantrs-tytan to solve the 3-rooks problem,
//! which involves placing 3 rooks on a 3x3 chess board such that none can
//! attack each other (i.e., no two rooks share a row or column).

use quantrs2_tytan::sampler::Sampler;
use quantrs2_tytan::symbol::Expression;
use quantrs2_tytan::{symbols_list, AutoArray, Compile, SASampler};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("3-Rooks Problem Solver");
    println!("======================");

    // Define a 3x3 grid of variables
    let q = symbols_list([3, 3], "q{}_{}").expect("Failed to create symbols");
    println!("Created symbols:\n{q:?}");

    // Constraint 1: Each row must have exactly one rook
    let mut h: Expression = 0.into();
    let two: Expression = 2.into();
    for i in 0..3 {
        let row_sum = q[[i, 0]].clone() + q[[i, 1]].clone() + q[[i, 2]].clone() - 1;
        h = h + row_sum.pow(&two);
    }

    // Constraint 2: Each column must have exactly one rook
    for j in 0..3 {
        let col_sum = q[[0, j]].clone() + q[[1, j]].clone() + q[[2, j]].clone() - 1;
        h = h + col_sum.pow(&two);
    }

    println!("Created QUBO expression");

    // Compile to QUBO
    let (qubo, offset) = Compile::new(h).get_qubo()?;
    println!("Compiled to QUBO with offset: {offset}");

    // Choose a sampler
    let solver = SASampler::new(Some(42)); // Fixed seed for reproducibility

    // Sample
    println!("Running solver with 100 shots...");
    let result = solver.run_qubo(&qubo, 100)?;

    // Display all results
    println!("\nAll solutions:");
    println!("{:=<40}", "");
    for (i, r) in result.iter().enumerate() {
        println!(
            "Solution {}: Energy = {}, Occurrences = {}",
            i + 1,
            r.energy,
            r.occurrences
        );
        println!("Assignments: {:?}", r.assignments);
    }

    // Get the best solution and convert to an array
    if !result.is_empty() {
        let best = &result[0];
        println!("\nBest solution visualization:");
        println!("{:=<40}", "");

        let (arr, subs) = AutoArray::new(best)
            .get_ndarray("q{}_{}")
            .expect("Failed to convert to array");

        // Print the array in a nice grid format
        for i in 0..3 {
            for j in 0..3 {
                print!("{} ", if arr[[i, j]] == 1 { "R" } else { "." });
            }
            println!();
        }
    }

    Ok(())
}
