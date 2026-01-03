//! Performance benchmarks for symbolic operations.
//!
//! Run with: cargo bench --package quantrs2-symengine-pure

#![allow(clippy::redundant_clone)]
#![allow(clippy::semicolon_if_nothing_returned)]
#![allow(clippy::approx_constant)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;

use quantrs2_symengine_pure::matrix;
use quantrs2_symengine_pure::simplify;
use quantrs2_symengine_pure::{Expression, SymbolicMatrix};

/// Benchmark basic expression construction
fn bench_expression_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("expression_construction");

    group.bench_function("symbol", |b| b.iter(|| black_box(Expression::symbol("x"))));

    group.bench_function("integer", |b| b.iter(|| black_box(Expression::int(42))));

    group.bench_function("float", |b| {
        b.iter(|| black_box(Expression::float_unchecked(3.14159)))
    });

    group.bench_function("complex", |b| {
        use quantrs2_symengine_pure::Complex64;
        let c = Complex64::new(1.0, 2.0);
        b.iter(|| black_box(Expression::from_complex64(c)))
    });

    group.finish();
}

/// Benchmark arithmetic operations
fn bench_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("arithmetic");

    let x = Expression::symbol("x");
    let y = Expression::symbol("y");

    group.bench_function("add", |b| b.iter(|| black_box(x.clone() + y.clone())));

    group.bench_function("mul", |b| b.iter(|| black_box(x.clone() * y.clone())));

    group.bench_function("pow", |b| {
        let two = Expression::int(2);
        b.iter(|| black_box(x.pow(&two)))
    });

    // Build increasingly complex expressions
    group.bench_function("complex_expr_10_ops", |b| {
        b.iter(|| {
            let mut expr = Expression::symbol("x");
            for i in 0..10 {
                let y = Expression::int(i);
                expr = expr * y + Expression::symbol("x");
            }
            black_box(expr)
        })
    });

    group.finish();
}

/// Benchmark differentiation
fn bench_differentiation(c: &mut Criterion) {
    let mut group = c.benchmark_group("differentiation");

    let x = Expression::symbol("x");

    // Simple derivative
    group.bench_function("d/dx(x)", |b| b.iter(|| black_box(x.diff(&x))));

    // Polynomial derivative
    let poly =
        x.clone() * x.clone() * x.clone() + Expression::int(2) * x.clone() * x.clone() + x.clone();
    group.bench_function("d/dx(x³+2x²+x)", |b| b.iter(|| black_box(poly.diff(&x))));

    // Trigonometric derivative
    let sin_x = quantrs2_symengine_pure::ops::trig::sin(&x);
    group.bench_function("d/dx(sin(x))", |b| b.iter(|| black_box(sin_x.diff(&x))));

    // Gradient computation
    let theta = Expression::symbol("theta");
    let phi = Expression::symbol("phi");
    let expr = quantrs2_symengine_pure::ops::trig::sin(&theta)
        * quantrs2_symengine_pure::ops::trig::cos(&phi);
    let vars = vec![theta.clone(), phi.clone()];
    group.bench_function("gradient_2_vars", |b| {
        b.iter(|| black_box(expr.gradient(&vars)))
    });

    group.finish();
}

/// Benchmark simplification
fn bench_simplification(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplification");

    // Simple simplification
    let x = Expression::symbol("x");
    let zero = Expression::zero();
    let expr_add_zero = x.clone() + zero;

    group.bench_function("x+0", |b| {
        b.iter(|| black_box(simplify::simplify(&expr_add_zero)))
    });

    // More complex expression
    let y = Expression::symbol("y");
    let z = Expression::symbol("z");
    let complex_expr = (x.clone() + y.clone()) * (x.clone() + z.clone());

    group.bench_function("expand_(x+y)*(x+z)", |b| {
        b.iter(|| black_box(simplify::expand(&complex_expr)))
    });

    // Factor test
    let factor_expr = x.clone() * y.clone() + x.clone() * z.clone();
    group.bench_function("factor_xy+xz", |b| {
        b.iter(|| black_box(simplify::factor(&factor_expr)))
    });

    group.finish();
}

/// Benchmark evaluation
fn bench_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluation");

    let x = Expression::symbol("x");
    let mut values = HashMap::new();
    values.insert("x".to_string(), 2.0);

    // Simple evaluation
    group.bench_function("eval_x", |b| b.iter(|| black_box(x.eval(&values))));

    // Polynomial evaluation
    let poly = x.clone() * x.clone() * x.clone()
        + Expression::int(2) * x.clone() * x.clone()
        + x.clone()
        + Expression::int(1);

    group.bench_function("eval_x³+2x²+x+1", |b| {
        b.iter(|| black_box(poly.eval(&values)))
    });

    // Multi-variable evaluation
    let y = Expression::symbol("y");
    let mut values2 = HashMap::new();
    values2.insert("x".to_string(), 2.0);
    values2.insert("y".to_string(), 3.0);

    let multi_expr = x.clone() * y.clone() + x.clone() + y;
    group.bench_function("eval_xy+x+y", |b| {
        b.iter(|| black_box(multi_expr.eval(&values2)))
    });

    group.finish();
}

/// Benchmark matrix operations
fn bench_matrix_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix");

    // Matrix creation
    group.bench_function("identity_2x2", |b| {
        b.iter(|| black_box(SymbolicMatrix::identity(2)))
    });

    group.bench_function("identity_4x4", |b| {
        b.iter(|| black_box(SymbolicMatrix::identity(4)))
    });

    // Pauli matrix creation
    group.bench_function("pauli_x", |b| b.iter(|| black_box(matrix::pauli_x())));

    // Matrix multiplication
    let mat_a = SymbolicMatrix::identity(2);
    let mat_b = matrix::pauli_x();

    group.bench_function("matmul_2x2", |bencher| {
        bencher.iter(|| black_box(mat_a.matmul(&mat_b)))
    });

    // Kronecker product
    group.bench_function("kron_2x2", |bencher| {
        bencher.iter(|| black_box(mat_a.kron(&mat_b)))
    });

    // Parametric gate
    let theta = Expression::symbol("theta");
    group.bench_function("rx_gate", |b| b.iter(|| black_box(matrix::rx(&theta))));

    group.finish();
}

/// Benchmark substitution
fn bench_substitution(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution");

    let x = Expression::symbol("x");
    let y = Expression::symbol("y");
    let val = Expression::int(5);

    // Simple substitution
    group.bench_function("x->5", |b| b.iter(|| black_box(x.substitute(&x, &val))));

    // Substitution in expression
    let expr = x.clone() * x.clone() + y.clone();
    group.bench_function("x²+y_x->5", |b| {
        b.iter(|| black_box(expr.substitute(&x, &val)))
    });

    // Multiple substitutions
    let complex = x.clone() * y.clone() + x.clone() + y.clone();
    group.bench_function("xy+x+y_multi_sub", |b| {
        let mut subs = HashMap::new();
        subs.insert(x.clone(), Expression::int(3));
        subs.insert(y.clone(), Expression::int(4));
        b.iter(|| black_box(complex.substitute_many(&subs)))
    });

    group.finish();
}

/// Benchmark VQE optimization operations
fn bench_vqe_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("vqe");

    use quantrs2_symengine_pure::optimization::*;

    // Gradient computation
    let theta = Expression::symbol("theta");
    let energy = theta.clone() * theta.clone() + quantrs2_symengine_pure::ops::trig::sin(&theta);
    let params = vec![theta.clone()];
    let mut values = HashMap::new();
    values.insert("theta".to_string(), 1.0);

    group.bench_function("gradient_at", |b| {
        b.iter(|| black_box(gradient_at(&energy, &params, &values)))
    });

    // Parameter-shift rule
    let psr = ParameterShiftRule::new();
    group.bench_function("parameter_shift_2_params", |b| {
        b.iter(|| {
            black_box(psr.compute_gradient(|params| params[0].sin() + params[1].cos(), &[0.5, 0.5]))
        })
    });

    group.finish();
}

/// Benchmark scaling behavior
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");

    // Expression depth scaling
    for depth in [5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("nested_expr", depth),
            &depth,
            |b, &depth| {
                b.iter(|| {
                    let mut expr = Expression::symbol("x");
                    for _ in 0..depth {
                        expr = expr.clone() * expr;
                    }
                    black_box(expr)
                })
            },
        );
    }

    // Gradient scaling with number of variables
    for n_vars in [2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("gradient_n_vars", n_vars),
            &n_vars,
            |b, &n| {
                let vars: Vec<Expression> = (0..n)
                    .map(|i| Expression::symbol(&format!("x{i}")))
                    .collect();

                // Build an expression that uses all variables
                let mut expr = Expression::zero();
                for v in &vars {
                    expr = expr + v.clone() * v.clone();
                }

                b.iter(|| black_box(expr.gradient(&vars)))
            },
        );
    }

    // Matrix size scaling
    for size in [2, 4, 8] {
        group.bench_with_input(BenchmarkId::new("identity_nxn", size), &size, |b, &n| {
            b.iter(|| black_box(SymbolicMatrix::identity(n)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_expression_construction,
    bench_arithmetic,
    bench_differentiation,
    bench_simplification,
    bench_evaluation,
    bench_matrix_ops,
    bench_substitution,
    bench_vqe_ops,
    bench_scaling,
);

criterion_main!(benches);
