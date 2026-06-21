//! Expression simplification using e-graph equality saturation.
//!
//! This module uses the egg library to perform term rewriting and
//! simplification via equality saturation.

use std::collections::HashMap;

use egg::{rewrite, CostFunction, Id, Language, RecExpr, Rewrite, Runner, Symbol};

use crate::expr::{ExprLang, Expression};

/// Expand an expression (distribute products over sums)
pub fn expand(expr: &Expression) -> Expression {
    // First, manually expand any power-2 expressions
    let expanded_pow = expand_powers(expr);

    // Then fully distribute all multiplications over additions
    distribute_fully(&expanded_pow)
}

/// Fully distribute multiplications over additions
/// This implements FOIL-like expansion for all product-of-sums
fn distribute_fully(expr: &Expression) -> Expression {
    // Recursively process the expression
    if expr.is_mul() {
        // SAFETY: is_mul() check guarantees as_mul() will succeed
        let operands = expr.as_mul().expect("is_mul() was true");
        let left = distribute_fully(&operands[0]);
        let right = distribute_fully(&operands[1]);

        // Distribute multiplication over additions
        distribute_product(&left, &right)
    } else if expr.is_add() {
        // SAFETY: is_add() check guarantees as_add() will succeed
        let operands = expr.as_add().expect("is_add() was true");
        let left = distribute_fully(&operands[0]);
        let right = distribute_fully(&operands[1]);
        left + right
    } else if expr.is_neg() {
        // SAFETY: is_neg() check guarantees as_neg() will succeed
        let inner = expr.as_neg().expect("is_neg() was true");
        -distribute_fully(&inner)
    } else if expr.is_pow() {
        // SAFETY: is_pow() check guarantees as_pow() will succeed
        let (base, exp) = expr.as_pow().expect("is_pow() was true");
        let expanded_base = distribute_fully(&base);
        expanded_base.pow(&exp)
    } else {
        // Symbols, numbers, etc. - return as-is
        expr.clone()
    }
}

/// Distribute a product: (a + b) * (c + d) = a*c + a*d + b*c + b*d
fn distribute_product(left: &Expression, right: &Expression) -> Expression {
    // Get all addends from left
    let left_terms = collect_addends(left);
    // Get all addends from right
    let right_terms = collect_addends(right);

    // Multiply each pair
    let mut result_terms: Vec<Expression> = Vec::new();
    for l in &left_terms {
        for r in &right_terms {
            let product = multiply_terms(l, r);
            result_terms.push(product);
        }
    }

    // Build sum
    if result_terms.is_empty() {
        Expression::zero()
    } else {
        let mut result = result_terms.remove(0);
        for term in result_terms {
            result = result + term;
        }
        result
    }
}

/// Collect all addends from an expression (handles nested additions)
fn collect_addends(expr: &Expression) -> Vec<Expression> {
    if expr.is_add() {
        // SAFETY: is_add() check guarantees as_add() will succeed
        let operands = expr.as_add().expect("is_add() was true");
        let mut terms = collect_addends(&operands[0]);
        terms.extend(collect_addends(&operands[1]));
        terms
    } else {
        vec![expr.clone()]
    }
}

/// Multiply two terms, handling negations
fn multiply_terms(a: &Expression, b: &Expression) -> Expression {
    // Handle negations to keep things clean
    let (a_neg, a_inner) = unwrap_neg(a);
    let (b_neg, b_inner) = unwrap_neg(b);

    let product = a_inner * b_inner;

    // XOR the negations
    if a_neg ^ b_neg {
        -product
    } else {
        product
    }
}

/// Unwrap negation: returns (is_negated, inner_expression)
fn unwrap_neg(expr: &Expression) -> (bool, Expression) {
    if expr.is_neg() {
        // SAFETY: is_neg() check guarantees as_neg() will succeed
        let inner = expr.as_neg().expect("is_neg() was true");
        let (inner_neg, inner_expr) = unwrap_neg(&inner);
        // Double negation cancels out
        (!inner_neg, inner_expr)
    } else {
        (false, expr.clone())
    }
}

/// Recursively expand power expressions with exponent 2
fn expand_powers(expr: &Expression) -> Expression {
    // Check if this is a power expression
    if expr.is_pow() {
        // SAFETY: is_pow() check guarantees as_pow() will succeed
        let (base, exp) = expr.as_pow().expect("is_pow() was true");

        // First recursively expand powers in the base
        let expanded_base = expand_powers(&base);

        // Check if exponent is 2
        if exp.is_number() {
            if let Some(exp_val) = exp.to_f64() {
                if (exp_val - 2.0).abs() < 1e-10 {
                    // a^2 => a * a
                    return expanded_base.clone() * expanded_base;
                }
            }
        }

        // For other exponents, return base^exp with expanded base
        return expanded_base.pow(&exp);
    }

    // Check if this is an addition - recursively expand
    if expr.is_add() {
        // SAFETY: is_add() check guarantees as_add() will succeed
        let operands = expr.as_add().expect("is_add() was true");
        let left = expand_powers(&operands[0]);
        let right = expand_powers(&operands[1]);
        return left + right;
    }

    // Check if this is a multiplication - recursively expand
    if expr.is_mul() {
        // SAFETY: is_mul() check guarantees as_mul() will succeed
        let operands = expr.as_mul().expect("is_mul() was true");
        let left = expand_powers(&operands[0]);
        let right = expand_powers(&operands[1]);
        return left * right;
    }

    // Check if this is a negation - recursively expand
    if expr.is_neg() {
        // SAFETY: is_neg() check guarantees as_neg() will succeed
        let inner = expr.as_neg().expect("is_neg() was true");
        return -expand_powers(&inner);
    }

    // For all other expressions (symbols, numbers), return as-is
    expr.clone()
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
/// Note: Currently unused as expand() uses direct distribute_fully() instead.
#[allow(dead_code)]
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

/// Get distribution rewrite rules (for expanding products over sums)
/// Note: Currently unused as expand() uses direct distribute_fully() instead,
/// but kept for potential future e-graph based expansion use cases.
#[allow(dead_code)]
fn get_distribution_rules() -> Vec<Rewrite<ExprLang, ()>> {
    vec![
        // Distributivity (left and right)
        rewrite!("distrib-left"; "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"),
        rewrite!("distrib-right"; "(* (+ ?a ?b) ?c)" => "(+ (* ?a ?c) (* ?b ?c))"),
        // Note: pow-2 rule doesn't work due to Num(Symbol) not matching literal "2"
        // Power expansion is now handled by expand_powers() function
        // Negation handling for expansion
        // (neg a) * b = neg(a * b)
        rewrite!("neg-mul-left"; "(* (neg ?a) ?b)" => "(neg (* ?a ?b))"),
        // a * (neg b) = neg(a * b)
        rewrite!("neg-mul-right"; "(* ?a (neg ?b))" => "(neg (* ?a ?b))"),
        // (neg a) * (neg b) = a * b (double negation in multiplication)
        rewrite!("neg-neg-mul"; "(* (neg ?a) (neg ?b))" => "(* ?a ?b)"),
        // Distribute negation over addition
        rewrite!("neg-add"; "(neg (+ ?a ?b))" => "(+ (neg ?a) (neg ?b))"),
        // Double negation elimination
        rewrite!("neg-neg"; "(neg (neg ?a))" => "?a"),
        // Multiplication associativity (helps with nested distributions)
        rewrite!("mul-assoc"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        rewrite!("mul-assoc-rev"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),
        // Addition associativity (helps flatten sums)
        rewrite!("add-assoc"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rewrite!("add-assoc-rev"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        // Commutativity (needed for proper expansion)
        rewrite!("mul-comm"; "(* ?a ?b)" => "(* ?b ?a)"),
        rewrite!("add-comm"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        // Basic simplifications needed during expansion
        rewrite!("add-zero"; "(+ ?a 0)" => "?a"),
        rewrite!("zero-add"; "(+ 0 ?a)" => "?a"),
        rewrite!("mul-one"; "(* ?a 1)" => "?a"),
        rewrite!("one-mul"; "(* 1 ?a)" => "?a"),
        rewrite!("mul-zero"; "(* ?a 0)" => "0"),
        rewrite!("zero-mul"; "(* 0 ?a)" => "0"),
        // Handle negation with constants
        rewrite!("neg-zero"; "(neg 0)" => "0"),
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

/// Collect like terms in a polynomial expression with respect to `var`.
///
/// Groups the additive terms of `expr` by their power of `var` and sums the
/// coefficients of each power, producing `Σ_k c_k · var^k`. For example,
/// `3·x + 2·x` collects to `5·x`, and `a·x + b·x` collects to `(a + b)·x`.
///
/// The expression is first fully expanded (so products of sums are distributed)
/// and the result is run through [`simplify`] to canonicalise the rebuilt sum.
/// Terms whose collected coefficient is exactly zero are dropped. If `var` is
/// not a symbol, the expression is returned expanded-and-simplified unchanged,
/// since there is no variable to collect against.
#[must_use]
pub fn collect(expr: &Expression, var: &Expression) -> Expression {
    // Fully expand so the expression is a flat sum of products.
    let expanded = expand(expr);

    // Without a symbol to collect against there is nothing meaningful to group.
    let Some(var_name) = var.as_symbol() else {
        return simplify(&expanded);
    };

    // Decompose into additive terms (subtraction is represented via negation).
    let terms = collect_addends(&expanded);

    // Group terms by the power of `var` they contain. The map value accumulates
    // the coefficient (everything in the term except the `var` factors).
    let mut by_power: HashMap<u32, Vec<Expression>> = HashMap::new();
    let mut order: Vec<u32> = Vec::new();

    for term in &terms {
        let (power, coeff) = split_power_of(term, var_name);
        if !by_power.contains_key(&power) {
            order.push(power);
        }
        by_power.entry(power).or_default().push(coeff);
    }

    // Highest power first gives a conventional polynomial ordering.
    order.sort_unstable_by(|a, b| b.cmp(a));

    let mut result: Option<Expression> = None;
    for power in order {
        let coeffs = by_power.remove(&power).unwrap_or_default();
        let coeff = sum_coefficients(&coeffs);

        // Drop terms whose coefficient collapses to zero.
        if coeff.to_f64().is_some_and(|v| v.abs() < 1e-15) {
            continue;
        }

        // Reattach the var^power factor.
        let term = match power {
            0 => coeff,
            1 => coeff * var.clone(),
            k => coeff * var.clone().pow(&Expression::int(i64::from(k))),
        };

        result = Some(match result {
            Some(acc) => acc + term,
            None => term,
        });
    }

    let collected = result.unwrap_or_else(Expression::zero);
    simplify(&collected)
}

/// Split a single multiplicative term into `(power_of_var, remaining_coefficient)`.
///
/// Counts how many times `var` appears as a factor, accounting for explicit
/// powers `var^n` (with non-negative integer `n`). The coefficient is the
/// product of all remaining factors; if nothing remains it is `1`.
fn split_power_of(term: &Expression, var_name: &str) -> (u32, Expression) {
    // Strip an outer negation, folding the sign into the coefficient.
    if term.is_neg() {
        if let Some(inner) = term.as_neg() {
            let (power, coeff) = split_power_of(&inner, var_name);
            return (power, -coeff);
        }
    }

    let factors = flatten_mul(term);
    let mut power: u32 = 0;
    let mut coeff_factors: Vec<Expression> = Vec::with_capacity(factors.len());

    for factor in factors {
        if factor.as_symbol() == Some(var_name) {
            power += 1;
        } else if let Some((base, exp)) = factor.as_pow() {
            // var^n contributes n to the power when n is a non-negative integer.
            if base.as_symbol() == Some(var_name) {
                if let Some(n) = exp.to_i64() {
                    if n >= 0 {
                        power += u32::try_from(n).unwrap_or(0);
                        continue;
                    }
                }
            }
            coeff_factors.push(factor);
        } else {
            coeff_factors.push(factor);
        }
    }

    let coeff = match coeff_factors.split_first() {
        Some((first, rest)) => {
            let mut acc = first.clone();
            for f in rest {
                acc = acc * f.clone();
            }
            acc
        }
        None => Expression::one(),
    };

    (power, coeff)
}

/// Flatten a (possibly nested) multiplication into its flat list of factors.
fn flatten_mul(expr: &Expression) -> Vec<Expression> {
    if expr.is_mul() {
        // SAFETY: is_mul() guarantees as_mul() succeeds.
        let operands = expr.as_mul().expect("is_mul() was true");
        let mut factors = flatten_mul(&operands[0]);
        factors.extend(flatten_mul(&operands[1]));
        factors
    } else {
        vec![expr.clone()]
    }
}

/// Sum a list of coefficient expressions, folding numeric ones exactly.
///
/// Numeric coefficients are accumulated into a single constant; non-numeric
/// coefficients are kept and combined with the numeric constant via symbolic
/// addition (the constant is omitted when it is exactly zero).
fn sum_coefficients(coeffs: &[Expression]) -> Expression {
    let mut numeric_sum = 0.0_f64;
    let mut symbolic: Vec<Expression> = Vec::new();

    for c in coeffs {
        if let Some(v) = c.to_f64() {
            numeric_sum += v;
        } else {
            symbolic.push(c.clone());
        }
    }

    if symbolic.is_empty() {
        return number_expr(numeric_sum);
    }

    let mut acc = symbolic[0].clone();
    for s in &symbolic[1..] {
        acc = acc + s.clone();
    }
    if numeric_sum.abs() < 1e-15 {
        acc
    } else {
        acc + number_expr(numeric_sum)
    }
}

/// Build a numeric expression from a folded coefficient.
///
/// `Expression::float_unchecked` renders integral values as bare integer
/// literals (e.g. `5.0` becomes the literal `"5"`), so downstream rewrite rules
/// that match on `"0"`/`"1"`/`"2"` continue to fire without any float-to-int
/// cast.
fn number_expr(value: f64) -> Expression {
    Expression::float_unchecked(value)
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

    #[test]
    fn test_collect_combines_like_terms_structurally() {
        // Regression test: `collect` previously delegated to simplify() and did
        // NOT combine like terms, so `3*x + 2*x` stayed a two-term sum. It must
        // now collapse to a single product `5 * x`.
        let x = Expression::symbol("x");
        let expr = Expression::int(3) * x.clone() + Expression::int(2) * x.clone();
        let collected = collect(&expr, &x);

        // Structural assertion: the result is a single multiplicative term, not a
        // sum. This fails for the old no-op behavior.
        assert!(
            collected.is_mul(),
            "expected a single collected term, got {collected}"
        );
        assert!(
            !collected.is_add(),
            "collected form must not be a sum, got {collected}"
        );

        // Numeric coefficient must be exactly 5.
        let mut values = std::collections::HashMap::new();
        values.insert("x".to_string(), 1.0);
        let coeff = collected.eval(&values).expect("eval");
        assert!((coeff - 5.0).abs() < 1e-10, "coefficient was {coeff}");

        // And the collected form is numerically equivalent everywhere.
        for x_val in [0.0, 1.0, -2.5, 7.0] {
            let mut v = std::collections::HashMap::new();
            v.insert("x".to_string(), x_val);
            let lhs = expr.eval(&v).expect("orig");
            let rhs = collected.eval(&v).expect("collected");
            assert!((lhs - rhs).abs() < 1e-10, "mismatch at x={x_val}");
        }
    }

    #[test]
    fn test_collect_symbolic_coefficients() {
        // a*x + b*x must collect to (a + b)*x: a single term whose coefficient is
        // the symbolic sum a + b.
        let a = Expression::symbol("a");
        let b = Expression::symbol("b");
        let x = Expression::symbol("x");
        let expr = a.clone() * x.clone() + b.clone() * x.clone();
        let collected = collect(&expr, &x);

        assert!(
            collected.is_mul(),
            "expected (a+b)*x as a single term, got {collected}"
        );

        // Equivalent under evaluation for arbitrary a, b, x.
        for (a_val, b_val, x_val) in [(2.0, 3.0, 4.0), (1.5, -0.5, 2.0), (0.0, 0.0, 9.0)] {
            let mut v = std::collections::HashMap::new();
            v.insert("a".to_string(), a_val);
            v.insert("b".to_string(), b_val);
            v.insert("x".to_string(), x_val);
            let lhs = expr.eval(&v).expect("orig");
            let rhs = collected.eval(&v).expect("collected");
            assert!(
                (lhs - rhs).abs() < 1e-10,
                "mismatch at a={a_val},b={b_val},x={x_val}: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn test_collect_polynomial_powers() {
        // x^2 + 2*x + 3*x + 1 must collect the x^1 terms to 5*x, leaving
        // x^2 + 5*x + 1 (numerically). Verify across several points.
        let x = Expression::symbol("x");
        let expr = x.clone().pow(&Expression::int(2))
            + Expression::int(2) * x.clone()
            + Expression::int(3) * x.clone()
            + Expression::int(1);
        let collected = collect(&expr, &x);

        for x_val in [-3.0, -1.0, 0.0, 0.5, 2.0, 4.0] {
            let mut v = std::collections::HashMap::new();
            v.insert("x".to_string(), x_val);
            let rhs = collected.eval(&v).expect("collected");
            // x^2 + 5x + 1
            let expected = x_val.mul_add(x_val, 5.0_f64.mul_add(x_val, 1.0));
            assert!(
                (rhs - expected).abs() < 1e-10,
                "at x={x_val}: got {rhs}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_collect_drops_zero_coefficient() {
        // 3*x - 3*x must collect to 0 (zero coefficient term is dropped).
        let x = Expression::symbol("x");
        let expr = Expression::int(3) * x.clone() - Expression::int(3) * x.clone();
        let collected = collect(&expr, &x);

        let mut v = std::collections::HashMap::new();
        v.insert("x".to_string(), 12345.0);
        let val = collected.eval(&v).expect("collected");
        assert!(val.abs() < 1e-10, "expected 0, got {val}");
    }

    #[test]
    fn test_expand_simple_pow2() {
        // Test simple a^2 = a*a
        let a = Expression::symbol("a");
        let two = Expression::from(2);

        let expr = a.clone().pow(&two);
        let expanded = expand(&expr);

        // Should expand to a*a
        let mut values = std::collections::HashMap::new();
        values.insert("a".to_string(), 3.0);
        let exp_val = expanded.eval(&values).expect("eval");
        assert!((exp_val - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_expand_binomial_squared() {
        // Test (a+b)^2 = a^2 + 2ab + b^2
        let a = Expression::symbol("a");
        let b = Expression::symbol("b");
        let two = Expression::from(2);

        let expr = (a.clone() + b.clone()).pow(&two);
        let expanded = expand(&expr);

        // Verify by evaluation at multiple points
        for (a_val, b_val) in [(2.0, 3.0), (1.0, 1.0), (0.0, 5.0)] {
            let mut values = std::collections::HashMap::new();
            values.insert("a".to_string(), a_val);
            values.insert("b".to_string(), b_val);

            let orig_val = expr.eval(&values).expect("eval original");
            let exp_val = expanded.eval(&values).expect("eval expanded");

            // (a+b)^2 should equal expanded form
            assert!(
                (orig_val - exp_val).abs() < 1e-10,
                "Mismatch at a={a_val}, b={b_val}: orig={orig_val}, expanded={exp_val}"
            );
            // Expected: (a+b)^2
            let expected = (a_val + b_val).powi(2);
            assert!(
                (exp_val - expected).abs() < 1e-10,
                "Unexpected value at a={a_val}, b={b_val}: got {exp_val}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_expand_polynomial_constraint() {
        // Test (x+y+z-1)^2 expansion
        // This is used in QUBO constraint expressions
        let x = Expression::symbol("x");
        let y = Expression::symbol("y");
        let z = Expression::symbol("z");
        let one = Expression::from(1);
        let two = Expression::from(2);

        let expr = (x.clone() + y.clone() + z.clone() - one).pow(&two);
        let expanded = expand(&expr);

        // Verify by evaluation at multiple test points
        for (x_val, y_val, z_val) in [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.5, 0.5, 0.0),
        ] {
            let mut values = std::collections::HashMap::new();
            values.insert("x".to_string(), x_val);
            values.insert("y".to_string(), y_val);
            values.insert("z".to_string(), z_val);

            let orig_val = expr.eval(&values).expect("eval original");
            let exp_val = expanded.eval(&values).expect("eval expanded");

            // Both should give same result
            assert!(
                (orig_val - exp_val).abs() < 1e-10,
                "Mismatch at x={x_val}, y={y_val}, z={z_val}: orig={orig_val}, expanded={exp_val}"
            );

            // Expected: (x+y+z-1)^2
            let expected = (x_val + y_val + z_val - 1.0).powi(2);
            assert!(
                (exp_val - expected).abs() < 1e-10,
                "Unexpected value at x={x_val}, y={y_val}, z={z_val}: got {exp_val}, expected {expected}"
            );
        }
    }
}
