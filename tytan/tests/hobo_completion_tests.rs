//! Tests verifying that previously-stubbed HOBO/QUBO sampler entry-points now
//! return real results rather than `SamplerError::NotImplemented`.

#![allow(clippy::pedantic, clippy::unnecessary_wraps)]

use scirs2_core::ndarray::{Array2, ArrayD, IxDyn};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// 1. TensorNetworkSampler::run_qubo should delegate to run_hobo and succeed.
// ---------------------------------------------------------------------------
#[test]
fn test_tensor_network_sampler_run_qubo() {
    use quantrs2_tytan::sampler::Sampler;
    use quantrs2_tytan::tensor_network_sampler::create_mps_sampler;

    // 2-variable QUBO: minimise -x0 - x1 (optimal: both 1).
    let mut q = Array2::<f64>::zeros((2, 2));
    q[[0, 0]] = -1.0;
    q[[1, 1]] = -1.0;

    let mut var_map = HashMap::<String, usize>::new();
    var_map.insert("x0".to_string(), 0);
    var_map.insert("x1".to_string(), 1);

    let sampler = create_mps_sampler(4);
    let result = sampler.run_qubo(&(q, var_map), 5);

    assert!(
        result.is_ok(),
        "run_qubo should succeed (not NotImplemented): {result:?}"
    );
    assert!(
        !result.unwrap().is_empty(),
        "run_qubo should return at least one sample"
    );
}

// ---------------------------------------------------------------------------
// 2. CIMSimulator::run_hobo should quadratize and return results.
// ---------------------------------------------------------------------------
#[test]
fn test_cim_run_hobo() {
    use quantrs2_tytan::coherent_ising_machine::CIMSimulator;
    use quantrs2_tytan::sampler::Sampler;

    // 3-variable, 3-body HOBO tensor.
    let mut tensor = ArrayD::<f64>::zeros(IxDyn(&[3, 3, 3]));
    tensor[[0, 1, 2]] = -1.0; // x0*x1*x2 coupling

    let mut var_map = HashMap::<String, usize>::new();
    var_map.insert("x0".to_string(), 0);
    var_map.insert("x1".to_string(), 1);
    var_map.insert("x2".to_string(), 2);

    // n_spins must match the *original* variable count before quadratization;
    // run_hobo builds a temporary CIM with the enlarged spin count internally.
    let cim = CIMSimulator::new(3).with_seed(42).with_evolution_time(2.0);
    let result = cim.run_hobo(&(tensor, var_map), 5);

    assert!(result.is_ok(), "CIM run_hobo should succeed: {result:?}");
    let samples = result.unwrap();
    assert!(!samples.is_empty(), "CIM run_hobo should return samples");

    // Auxiliary variables must not leak into the results.
    for sample in &samples {
        for key in sample.assignments.keys() {
            assert!(
                !key.starts_with("_aux_"),
                "auxiliary variable '{key}' leaked into CIM run_hobo results"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. PhotonicIsingMachineSampler::run_hobo should quadratize and succeed.
// ---------------------------------------------------------------------------
#[test]
fn test_photonic_run_hobo() {
    use quantrs2_tytan::sampler::hardware::photonic::{
        PhotonicConfig, PhotonicIsingMachineSampler,
    };
    use quantrs2_tytan::sampler::Sampler;

    // 3-variable 3-body HOBO.
    let mut tensor = ArrayD::<f64>::zeros(IxDyn(&[3, 3, 3]));
    tensor[[0, 1, 2]] = -1.0;

    let mut var_map = HashMap::<String, usize>::new();
    var_map.insert("a".to_string(), 0);
    var_map.insert("b".to_string(), 1);
    var_map.insert("c".to_string(), 2);

    let sampler = PhotonicIsingMachineSampler::new(PhotonicConfig::default());
    let result = sampler.run_hobo(&(tensor, var_map), 3);

    assert!(
        result.is_ok(),
        "Photonic run_hobo should succeed: {result:?}"
    );
    let samples = result.unwrap();
    assert!(!samples.is_empty());

    // No auxiliary variables should appear.
    for sample in &samples {
        for key in sample.assignments.keys() {
            assert!(
                !key.starts_with("_aux_"),
                "auxiliary variable '{key}' leaked into photonic run_hobo results"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. optimize_hobo_basic should actually minimise rather than return zeros.
// ---------------------------------------------------------------------------
#[test]
fn test_optimize_hobo_basic_finds_minimum() {
    use quantrs2_tytan::optimize::optimize_hobo;

    // 2-variable problem: E = -x0 - x1 + 4·x0·x1
    // Possible values:
    //   (0,0) → 0,  (1,0) → -1,  (0,1) → -1,  (1,1) → -1-1+4 = 2
    // Optimal is -1 at either (1,0) or (0,1).
    let mut tensor = ArrayD::<f64>::zeros(IxDyn(&[2, 2]));
    tensor[[0, 0]] = -1.0; // linear x0
    tensor[[1, 1]] = -1.0; // linear x1
    tensor[[0, 1]] = 4.0; // quadratic x0·x1

    let mut var_map = HashMap::<String, usize>::new();
    var_map.insert("x0".to_string(), 0);
    var_map.insert("x1".to_string(), 1);

    // 100 sweeps is enough for a 2-variable problem.
    let results = optimize_hobo(&tensor, &var_map, None, 100);

    assert!(!results.is_empty(), "optimize_hobo should return a result");
    let best_energy = results[0].energy;
    assert!(
        best_energy <= -0.9,
        "expected energy ≤ -1.0, got {best_energy}"
    );
}

// ---------------------------------------------------------------------------
// 5. optimize_hobo_basic with an empty var_map returns an empty Vec.
// ---------------------------------------------------------------------------
#[test]
fn test_optimize_hobo_empty_var_map() {
    use quantrs2_tytan::optimize::optimize_hobo;

    let tensor = ArrayD::<f64>::zeros(IxDyn(&[0]));
    let var_map = HashMap::<String, usize>::new();

    let results = optimize_hobo(&tensor, &var_map, None, 10);
    assert!(
        results.is_empty(),
        "optimize_hobo with empty var_map should return empty Vec"
    );
}
