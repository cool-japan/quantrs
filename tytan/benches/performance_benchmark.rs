//! Performance benchmarks for critical operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use quantrs2_tytan::performance_optimization::*;
use quantrs2_tytan::sampler::{SASampler, Sampler};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn benchmark_qubo_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("QUBO Evaluation");
    
    for size in [10, 50, 100, 200].iter() {
        // Create random QUBO
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let qubo = Array2::random_using(
            (*size, *size),
            ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0),
            &mut rng,
        );
        
        let evaluator = OptimizedQUBOEvaluator::new(qubo.clone());
        let x = Array1::random_using(
            *size,
            ndarray_rand::rand_distr::Bernoulli::new(0.5).unwrap(),
            &mut rng,
        ).mapv(|b| if b { 1 } else { 0 });
        
        group.bench_with_input(
            BenchmarkId::new("Optimized", size),
            size,
            |b, _| {
                b.iter(|| {
                    evaluator.evaluate(&x.view())
                });
            },
        );
        
        // Compare with naive implementation
        group.bench_with_input(
            BenchmarkId::new("Naive", size),
            size,
            |b, _| {
                b.iter(|| {
                    naive_qubo_eval(&qubo, &x)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_delta_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("Delta Energy");
    
    let size = 100;
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let qubo = Array2::random_using(
        (size, size),
        ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0),
        &mut rng,
    );
    
    let evaluator = OptimizedQUBOEvaluator::new(qubo);
    let x = Array1::random_using(
        size,
        ndarray_rand::rand_distr::Bernoulli::new(0.5).unwrap(),
        &mut rng,
    ).mapv(|b| if b { 1 } else { 0 });
    
    group.bench_function("delta_energy", |b| {
        let mut bit = 0;
        b.iter(|| {
            let delta = evaluator.delta_energy(&x.view(), bit);
            bit = (bit + 1) % size;
            black_box(delta)
        });
    });
    
    group.finish();
}

fn benchmark_sa_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("SA Optimization");
    
    for size in [20, 50, 100].iter() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let qubo = Array2::random_using(
            (*size, *size),
            ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0),
            &mut rng,
        );
        
        let opt_sa = OptimizedSA::new(qubo.clone())
            .with_schedule(AnnealingSchedule::Geometric { 
                t0: 1.0, 
                alpha: 0.99 
            });
        
        let initial = Array1::zeros(*size);
        
        group.bench_with_input(
            BenchmarkId::new("Optimized SA", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut rng = ChaCha20Rng::seed_from_u64(42);
                    opt_sa.anneal(initial.clone(), 100, &mut rng)
                });
            },
        );
        
        // Compare with standard SA
        let sampler = SASampler::new(Some(42));
        let mut var_map = std::collections::HashMap::new();
        for i in 0..*size {
            var_map.insert(format!("x{}", i), i);
        }
        
        group.bench_with_input(
            BenchmarkId::new("Standard SA", size),
            size,
            |b, _| {
                b.iter(|| {
                    sampler.run_qubo(&(qubo.clone(), var_map.clone()), 1).ok()
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_matrix_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Operations");
    
    let size = 200;
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let qubo = Array2::random_using(
        (size, size),
        ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0),
        &mut rng,
    );
    
    // Test with different sparsity levels
    for sparsity in [0.1, 0.3, 0.5, 0.8].iter() {
        let x = Array1::random_using(
            size,
            ndarray_rand::rand_distr::Bernoulli::new(*sparsity).unwrap(),
            &mut rng,
        ).mapv(|b| if b { 1 } else { 0 });
        
        group.bench_with_input(
            BenchmarkId::new("Sparse Multiply", format!("{}% active", (sparsity * 100.0) as u32)),
            sparsity,
            |b, _| {
                b.iter(|| {
                    matrix_ops::sparse_qubo_multiply(&qubo, &x.view(), 1e-6)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("Block Eval", format!("{}% active", (sparsity * 100.0) as u32)),
            sparsity,
            |b, _| {
                b.iter(|| {
                    matrix_ops::block_qubo_eval(&qubo, &x.view(), 16)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_parallel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel Operations");
    
    let size = 500;
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let qubo = Array2::random_using(
        (size, size),
        ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0),
        &mut rng,
    );
    
    let evaluator = OptimizedQUBOEvaluator::new(qubo.clone());
    let x = Array1::random_using(
        size,
        ndarray_rand::rand_distr::Bernoulli::new(0.5).unwrap(),
        &mut rng,
    ).mapv(|b| if b { 1 } else { 0 });
    
    // Sequential vs parallel evaluation
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            evaluator.evaluate_scalar(&x.view())
        });
    });
    
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            evaluator.evaluate_parallel(&x.view())
        });
    });
    
    // Test parallel SA
    let opt_sa_parallel = OptimizedSA::new(qubo.clone())
        .with_parallel_moves(true);
    
    let opt_sa_sequential = OptimizedSA::new(qubo)
        .with_parallel_moves(false);
    
    let initial = Array1::zeros(size);
    
    group.bench_function("SA Sequential", |b| {
        b.iter(|| {
            let mut rng = ChaCha20Rng::seed_from_u64(42);
            opt_sa_sequential.anneal(initial.clone(), 50, &mut rng)
        });
    });
    
    group.bench_function("SA Parallel", |b| {
        b.iter(|| {
            let mut rng = ChaCha20Rng::seed_from_u64(42);
            opt_sa_parallel.anneal(initial.clone(), 50, &mut rng)
        });
    });
    
    group.finish();
}

fn benchmark_memory_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Pool");
    
    let size = 1000;
    let mut pool: MemoryPool<f64> = MemoryPool::new(size, 10);
    
    group.bench_function("Pool Allocation", |b| {
        b.iter(|| {
            if let Some(mut buffer) = pool.get() {
                // Use buffer
                for i in 0..size {
                    buffer[i] = i as f64;
                }
                pool.put(buffer);
            }
        });
    });
    
    group.bench_function("Direct Allocation", |b| {
        b.iter(|| {
            let mut buffer = vec![0.0; size];
            // Use buffer
            for i in 0..size {
                buffer[i] = i as f64;
            }
            // Drop buffer
        });
    });
    
    group.finish();
}

// Helper functions
fn naive_qubo_eval(qubo: &Array2<f64>, x: &Array1<u8>) -> f64 {
    let n = x.len();
    let mut energy = 0.0;
    
    for i in 0..n {
        for j in 0..n {
            energy += qubo[[i, j]] * (x[i] as f64) * (x[j] as f64);
        }
    }
    
    energy
}

criterion_group!(
    benches,
    benchmark_qubo_evaluation,
    benchmark_delta_energy,
    benchmark_sa_optimization,
    benchmark_matrix_ops,
    benchmark_parallel_operations,
    benchmark_memory_pool
);
criterion_main!(benches);