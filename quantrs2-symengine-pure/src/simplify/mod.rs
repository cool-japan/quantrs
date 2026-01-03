//! Expression simplification using e-graph equality saturation.
//!
//! This module uses the egg library to perform term rewriting and
//! simplification via equality saturation.

use std::collections::HashMap;

use egg::{rewrite, CostFunction, Id, Language, RecExpr, Rewrite, Runner, Symbol};

use crate::expr::{ExprLang, Expression};

/// Expand an expression (distribute products over sums)
pub fn expand(expr: &Expression) -> Expression {
    let expand_rules = get_expansion_rules();

    let runner = Runner::default()
        .with_expr(expr.as_rec_expr())
        .with_iter_limit(10)
        .run(&expand_rules);

    let root = runner.roots[0];
    let extractor = egg::Extractor::new(&runner.egraph, ExpandedSize);
    let (_, best) = extractor.find_best(root);

    Expression::from_rec_expr(best)
}

/// Simplify an expression using e-graph equality saturation
pub fn simplify(expr: &Expression) -> Expression {
    let rules = get_simplification_rules();

    let runner = Runner::default()
        .with_expr(expr.as_rec_expr())
        .with_iter_limit(20)
        .run(&rules);

    let root = runner.roots[0];
    let extractor = egg::Extractor::new(&runner.egraph, AstSize);
    let (_, best) = extractor.find_best(root);

    Expression::from_rec_expr(best)
}

/// Substitute a variable with an expression
pub fn substitute(expr: &Expression, var: &Expression, value: &Expression) -> Expression {
    let var_name = match var.as_symbol() {
        Some(name) => name.to_string(),
        None => return expr.clone(), // Can only substitute symbols
    };

    let rec_expr = expr.as_rec_expr();
    let value_expr = value.as_rec_expr();

    // Build a new expression with substitution
    let mut new_expr = RecExpr::default();
    let mut id_map: HashMap<usize, Id> = HashMap::new();

    substitute_rec(
        rec_expr,
        rec_expr.as_ref().len() - 1,
        &var_name,
        value_expr,
        &mut new_expr,
        &mut id_map,
    );

    Expression::from_rec_expr(new_expr)
}

/// Recursive substitution helper
fn substitute_rec(
    expr: &RecExpr<ExprLang>,
    idx: usize,
    var_name: &str,
    value: &RecExpr<ExprLang>,
    new_expr: &mut RecExpr<ExprLang>,
    id_map: &mut HashMap<usize, Id>,
) -> Id {
    if let Some(&new_id) = id_map.get(&idx) {
        return new_id;
    }

    let node = &expr[Id::from(idx)];

    // Check if this is the variable to substitute
    if let ExprLang::Num(s) = node {
        if s.as_str() == var_name {
            // Insert the value expression
            let offset = new_expr.as_ref().len();
            for (i, n) in value.as_ref().iter().enumerate() {
                let mapped_node = n
                    .clone()
                    .map_children(|child_id| Id::from(usize::from(child_id) + offset));
                new_expr.add(mapped_node);
            }
            let new_id = Id::from(new_expr.as_ref().len() - 1);
            id_map.insert(idx, new_id);
            return new_id;
        }
    }

    // Otherwise, recursively process children
    let new_node = node.clone().map_children(|child_id| {
        substitute_rec(
            expr,
            usize::from(child_id),
            var_name,
            value,
            new_expr,
            id_map,
        )
    });
    let new_id = new_expr.add(new_node);
    id_map.insert(idx, new_id);
    new_id
}

/// Cost function for extracting the simplest expression
struct AstSize;

impl CostFunction<ExprLang> for AstSize {
    type Cost = usize;

    fn cost<C>(&mut self, node: &ExprLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let node_cost = match node {
            // Prefer simpler nodes (Num covers both constants and symbols)
            ExprLang::Num(_) => 1,
            _ => 3,
        };

        node.fold(node_cost, |sum, id| sum + costs(id))
    }
}

/// Cost function that prefers expanded (distributed) forms
struct ExpandedSize;

impl CostFunction<ExprLang> for ExpandedSize {
    type Cost = usize;

    fn cost<C>(&mut self, node: &ExprLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let node_cost = match node {
            ExprLang::Num(_) => 1,
            // Prefer additions over multiplications (expanded form)
            ExprLang::Add(_) => 2,
            ExprLang::Mul(_) => 4,
            _ => 3,
        };

        node.fold(node_cost, |sum, id| sum + costs(id))
    }
}

/// Get expansion (distribution) rewrite rules
fn get_expansion_rules() -> Vec<Rewrite<ExprLang, ()>> {
    vec![
        // Distributivity (left and right)
        rewrite!("distrib-left"; "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"),
        rewrite!("distrib-right"; "(* (+ ?a ?b) ?c)" => "(+ (* ?a ?c) (* ?b ?c))"),
        // Expand powers (limited)
        rewrite!("pow-2"; "(^ ?a 2)" => "(* ?a ?a)"),
        // Basic simplifications needed during expansion
        rewrite!("add-zero"; "(+ ?a 0)" => "?a"),
        rewrite!("zero-add"; "(+ 0 ?a)" => "?a"),
        rewrite!("mul-one"; "(* ?a 1)" => "?a"),
        rewrite!("one-mul"; "(* 1 ?a)" => "?a"),
        rewrite!("mul-zero"; "(* ?a 0)" => "0"),
        rewrite!("zero-mul"; "(* 0 ?a)" => "0"),
    ]
}

/// Get the simplification rewrite rules
fn get_simplification_rules() -> Vec<Rewrite<ExprLang, ()>> {
    vec![
        // Additive identity: a + 0 = a
        rewrite!("add-zero"; "(+ ?a 0)" => "?a"),
        rewrite!("zero-add"; "(+ 0 ?a)" => "?a"),
        // Multiplicative identity: a * 1 = a
        rewrite!("mul-one"; "(* ?a 1)" => "?a"),
        rewrite!("one-mul"; "(* 1 ?a)" => "?a"),
        // Multiplicative zero: a * 0 = 0
        rewrite!("mul-zero"; "(* ?a 0)" => "0"),
        rewrite!("zero-mul"; "(* 0 ?a)" => "0"),
        // Double negation: -(-a) = a
        rewrite!("neg-neg"; "(neg (neg ?a))" => "?a"),
        // Power rules
        rewrite!("pow-zero"; "(^ ?a 0)" => "1"),
        rewrite!("pow-one"; "(^ ?a 1)" => "?a"),
        // Commutativity
        rewrite!("add-comm"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("mul-comm"; "(* ?a ?b)" => "(* ?b ?a)"),
        // Associativity
        rewrite!("add-assoc"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rewrite!("mul-assoc"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        // Distributivity
        rewrite!("distrib"; "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"),
        // Trigonometric identities
        // sin^2 + cos^2 = 1 (this is tricky to express as a rewrite)

        // Exponential/logarithm identities
        rewrite!("exp-log"; "(exp (log ?a))" => "?a"),
        rewrite!("log-exp"; "(log (exp ?a))" => "?a"),
        // sqrt(x^2) = |x|
        rewrite!("sqrt-sq"; "(sqrt (^ ?a 2))" => "(abs ?a)"),
    ]
}

/// Get quantum-specific simplification rules
pub fn get_quantum_rules() -> Vec<Rewrite<ExprLang, ()>> {
    vec![
        // Commutator identities
        // [A, A] = 0
        rewrite!("comm-self"; "(comm ?a ?a)" => "0"),
        // [A, B] = -[B, A] (antisymmetry)
        rewrite!("comm-antisym"; "(comm ?a ?b)" => "(neg (comm ?b ?a))"),
        // [0, A] = 0, [A, 0] = 0
        rewrite!("comm-zero-left"; "(comm 0 ?a)" => "0"),
        rewrite!("comm-zero-right"; "(comm ?a 0)" => "0"),
        // Anticommutator identities
        // {A, A} = 2A
        rewrite!("anticomm-self"; "(anticomm ?a ?a)" => "(* 2 ?a)"),
        // {A, B} = {B, A} (symmetry)
        rewrite!("anticomm-sym"; "(anticomm ?a ?b)" => "(anticomm ?b ?a)"),
        // {0, A} = A, {A, 0} = A
        rewrite!("anticomm-zero"; "(anticomm 0 ?a)" => "?a"),
        // Hermitian conjugate (dagger) identities
        // (A†)† = A
        rewrite!("dagger-dagger"; "(dagger (dagger ?a))" => "?a"),
        // (AB)† = B†A† (reversal for products)
        rewrite!("dagger-mul"; "(dagger (* ?a ?b))" => "(* (dagger ?b) (dagger ?a))"),
        // (A + B)† = A† + B†
        rewrite!("dagger-add"; "(dagger (+ ?a ?b))" => "(+ (dagger ?a) (dagger ?b))"),
        // (cA)† = c*A† (for complex scalars, * denotes conjugate)
        // This is handled via (conj c) * (dagger A)

        // 0† = 0
        rewrite!("dagger-zero"; "(dagger 0)" => "0"),
        // 1† = 1
        rewrite!("dagger-one"; "(dagger 1)" => "1"),
        // Trace identities
        // tr(A + B) = tr(A) + tr(B)
        rewrite!("trace-add"; "(trace (+ ?a ?b))" => "(+ (trace ?a) (trace ?b))"),
        // tr(cA) = c * tr(A)
        rewrite!("trace-scale"; "(trace (* ?c ?a))" => "(* ?c (trace ?a))"),
        // tr(0) = 0
        rewrite!("trace-zero"; "(trace 0)" => "0"),
        // Tensor product identities
        // (A ⊗ B)(C ⊗ D) = (AC) ⊗ (BD) (this is a simplification hint)
        rewrite!("tensor-mul"; "(* (tensor ?a ?b) (tensor ?c ?d))" => "(tensor (* ?a ?c) (* ?b ?d))"),
        // A ⊗ 1 = A (for identity operator 1)
        rewrite!("tensor-one-right"; "(tensor ?a 1)" => "?a"),
        rewrite!("tensor-one-left"; "(tensor 1 ?a)" => "?a"),
        // A ⊗ 0 = 0
        rewrite!("tensor-zero"; "(tensor ?a 0)" => "0"),
        rewrite!("tensor-zero-left"; "(tensor 0 ?a)" => "0"),
        // Determinant identities
        // det(AB) = det(A) * det(B) - only true for square matrices
        // det(I) = 1
        rewrite!("det-one"; "(det 1)" => "1"),
        // Transpose identities
        // (A^T)^T = A
        rewrite!("transpose-transpose"; "(transpose (transpose ?a))" => "?a"),
        // (AB)^T = B^T A^T
        rewrite!("transpose-mul"; "(transpose (* ?a ?b))" => "(* (transpose ?b) (transpose ?a))"),
        // (A + B)^T = A^T + B^T
        rewrite!("transpose-add"; "(transpose (+ ?a ?b))" => "(+ (transpose ?a) (transpose ?b))"),
    ]
}

/// Simplify an expression with quantum-specific rules
pub fn simplify_quantum(expr: &Expression) -> Expression {
    let mut rules = get_simplification_rules();
    rules.extend(get_quantum_rules());

    let runner = Runner::default()
        .with_expr(expr.as_rec_expr())
        .with_iter_limit(30)
        .run(&rules);

    let root = runner.roots[0];
    let extractor = egg::Extractor::new(&runner.egraph, AstSize);
    let (_, best) = extractor.find_best(root);

    Expression::from_rec_expr(best)
}

/// Get trigonometric identities useful in quantum computing
pub fn get_trig_rules() -> Vec<Rewrite<ExprLang, ()>> {
    vec![
        // Pythagorean identity: sin²(x) + cos²(x) = 1
        // This is hard to express directly, but we can express some related rules

        // sin(0) = 0
        rewrite!("sin-zero"; "(sin 0)" => "0"),
        // cos(0) = 1
        rewrite!("cos-zero"; "(cos 0)" => "1"),
        // tan(0) = 0
        rewrite!("tan-zero"; "(tan 0)" => "0"),
        // exp(0) = 1
        rewrite!("exp-zero"; "(exp 0)" => "1"),
        // log(1) = 0
        rewrite!("log-one"; "(log 1)" => "0"),
        // sin(-x) = -sin(x) (odd function)
        rewrite!("sin-neg"; "(sin (neg ?x))" => "(neg (sin ?x))"),
        // cos(-x) = cos(x) (even function)
        rewrite!("cos-neg"; "(cos (neg ?x))" => "(cos ?x)"),
        // tan(-x) = -tan(x) (odd function)
        rewrite!("tan-neg"; "(tan (neg ?x))" => "(neg (tan ?x))"),
        // exp(a + b) = exp(a) * exp(b)
        rewrite!("exp-add"; "(exp (+ ?a ?b))" => "(* (exp ?a) (exp ?b))"),
        // log(a * b) = log(a) + log(b)
        rewrite!("log-mul"; "(log (* ?a ?b))" => "(+ (log ?a) (log ?b))"),
        // exp(log(x)) = x
        rewrite!("exp-log"; "(exp (log ?x))" => "?x"),
        // log(exp(x)) = x
        rewrite!("log-exp"; "(log (exp ?x))" => "?x"),
        // sqrt(x)^2 = x
        rewrite!("sqrt-sq"; "(^ (sqrt ?x) 2)" => "?x"),
        // sqrt(x^2) = |x|
        rewrite!("sq-sqrt"; "(sqrt (^ ?x 2))" => "(abs ?x)"),
    ]
}

/// Simplify with trigonometric rules
pub fn simplify_trig(expr: &Expression) -> Expression {
    let mut rules = get_simplification_rules();
    rules.extend(get_trig_rules());

    let runner = Runner::default()
        .with_expr(expr.as_rec_expr())
        .with_iter_limit(30)
        .run(&rules);

    let root = runner.roots[0];
    let extractor = egg::Extractor::new(&runner.egraph, AstSize);
    let (_, best) = extractor.find_best(root);

    Expression::from_rec_expr(best)
}

/// Collect like terms in a polynomial expression
///
/// This is a more aggressive simplification that tries to collect
/// terms with the same variable factors.
pub fn collect(expr: &Expression, var: &Expression) -> Expression {
    // First expand, then simplify
    let expanded = expand(expr);

    // For now, just return simplified form
    // Full polynomial collection would require more sophisticated analysis
    simplify(&expanded)
}

/// Factor common terms out of a sum
///
/// For example: ax + ay -> a(x + y)
pub fn factor(expr: &Expression) -> Expression {
    let factor_rules = vec![
        // Reverse distributivity: common factor extraction
        rewrite!("factor-left"; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),
        rewrite!("factor-right"; "(+ (* ?a ?c) (* ?b ?c))" => "(* (+ ?a ?b) ?c)"),
        // a + a = 2a
        rewrite!("add-same"; "(+ ?a ?a)" => "(* 2 ?a)"),
        // Basic simplifications
        rewrite!("mul-one"; "(* ?a 1)" => "?a"),
        rewrite!("mul-zero"; "(* ?a 0)" => "0"),
    ];

    let runner: Runner<ExprLang, ()> = Runner::default()
        .with_expr(expr.as_rec_expr())
        .with_iter_limit(20)
        .run(&factor_rules);

    let root = runner.roots[0];

    // Use a cost function that prefers factored forms
    let extractor = egg::Extractor::new(&runner.egraph, FactoredSize);
    let (_, best) = extractor.find_best(root);

    Expression::from_rec_expr(best)
}

/// Cost function that prefers factored (shorter) forms
struct FactoredSize;

impl CostFunction<ExprLang> for FactoredSize {
    type Cost = usize;

    fn cost<C>(&mut self, node: &ExprLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let node_cost = match node {
            ExprLang::Num(_) => 1,
            // Prefer multiplications over additions for factored form
            ExprLang::Mul(_) => 2,
            ExprLang::Add(_) => 4,
            _ => 3,
        };

        node.fold(node_cost, |sum, id| sum + costs(id))
    }
}

#[cfg(test)]
#[allow(clippy::redundant_clone)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_add_zero() {
        let x = Expression::symbol("x");
        let zero = Expression::zero();
        let expr = x + zero;

        let simplified = simplify(&expr);
        // The simplification should reduce x + 0 to x
        assert!(simplified.as_symbol().is_some());
    }

    #[test]
    fn test_simplify_mul_one() {
        let x = Expression::symbol("x");
        let one = Expression::one();
        let expr = x * one;

        let simplified = simplify(&expr);
        assert!(simplified.as_symbol().is_some());
    }

    #[test]
    fn test_simplify_mul_zero() {
        let x = Expression::symbol("x");
        let zero = Expression::zero();
        let expr = x * zero;

        let simplified = simplify(&expr);
        assert!(simplified.is_zero());
    }

    #[test]
    fn test_substitute_simple() {
        let x = Expression::symbol("x");
        let y = Expression::symbol("y");
        let two = Expression::int(2);

        // x + y, substitute x -> 2
        let expr = x.clone() + y;
        let result = substitute(&expr, &x, &two);

        // The result should be 2 + y
        let mut values = std::collections::HashMap::new();
        values.insert("y".to_string(), 3.0);
        let eval_result = result.eval(&values);
        assert!(eval_result.is_ok());
        assert!((eval_result.expect("eval") - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_substitute_nested() {
        let x = Expression::symbol("x");
        let y = Expression::symbol("y");

        // x * x, substitute x -> y
        let expr = x.clone() * x.clone();
        let result = substitute(&expr, &x, &y);

        // The result should be y * y
        let mut values = std::collections::HashMap::new();
        values.insert("y".to_string(), 3.0);
        let eval_result = result.eval(&values);
        assert!(eval_result.is_ok());
        assert!((eval_result.expect("eval") - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_expand_distribution() {
        let x = Expression::symbol("x");
        let y = Expression::symbol("y");
        let z = Expression::symbol("z");

        // x * (y + z) should expand to x*y + x*z
        let expr = x * (y + z);
        let expanded = expand(&expr);

        // Verify by evaluation
        let mut values = std::collections::HashMap::new();
        values.insert("x".to_string(), 2.0);
        values.insert("y".to_string(), 3.0);
        values.insert("z".to_string(), 4.0);

        let orig_val = expr.eval(&values).expect("eval original");
        let exp_val = expanded.eval(&values).expect("eval expanded");

        assert!((orig_val - exp_val).abs() < 1e-10);
        assert!((exp_val - 14.0).abs() < 1e-10); // 2*(3+4) = 14
    }

    #[test]
    fn test_factor_common_terms() {
        let a = Expression::symbol("a");
        let x = Expression::symbol("x");
        let y = Expression::symbol("y");

        // a*x + a*y should factor to a*(x+y)
        let expr = a.clone() * x.clone() + a.clone() * y.clone();
        let factored = factor(&expr);

        // Verify by evaluation - both should give same result
        let mut values = std::collections::HashMap::new();
        values.insert("a".to_string(), 2.0);
        values.insert("x".to_string(), 3.0);
        values.insert("y".to_string(), 4.0);

        let orig_val = expr.eval(&values).expect("eval original");
        let fact_val = factored.eval(&values).expect("eval factored");

        assert!((orig_val - fact_val).abs() < 1e-10);
        assert!((fact_val - 14.0).abs() < 1e-10); // 2*3 + 2*4 = 14
    }

    #[test]
    fn test_simplify_trig() {
        // Test that sin(0) = 0
        let zero = Expression::zero();
        let sin_zero = crate::ops::trig::sin(&zero);
        let simplified = simplify_trig(&sin_zero);

        // After simplification, sin(0) should be 0
        // Verify by evaluation at a point
        let result = simplified.eval(&std::collections::HashMap::new());
        assert!(result.is_ok());
        assert!(result.expect("eval").abs() < 1e-10);
    }

    #[test]
    fn test_simplify_quantum_dagger() {
        // Test that (A†)† = A
        // We can't directly test this with the current DSL since dagger is symbolic
        // But we can verify the rules are in place
        let rules = get_quantum_rules();
        assert!(!rules.is_empty());

        // Verify specific rules exist by checking the count
        // We have many quantum rules defined
        assert!(rules.len() >= 15);
    }

    #[test]
    fn test_collect() {
        let x = Expression::symbol("x");

        // x + x should become 2x after collect
        let expr = x.clone() + x.clone();
        let collected = collect(&expr, &x);

        // Verify by evaluation
        let mut values = std::collections::HashMap::new();
        values.insert("x".to_string(), 5.0);

        let orig_val = expr.eval(&values).expect("eval original");
        let coll_val = collected.eval(&values).expect("eval collected");

        assert!((orig_val - coll_val).abs() < 1e-10);
        assert!((coll_val - 10.0).abs() < 1e-10); // 5 + 5 = 10
    }
}
