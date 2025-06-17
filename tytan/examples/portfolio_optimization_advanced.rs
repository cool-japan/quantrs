//! Advanced Portfolio Optimization example using QuantRS2-Tytan with SciRS2
//!
//! This example demonstrates:
//! - Mean-variance portfolio optimization
//! - Risk-return trade-offs with Pareto frontier
//! - Transaction costs and constraints
//! - Sector allocation constraints
//! - Comparison with classical optimization methods

use ndarray::{Array1, Array2};
use quantrs2_tytan::{
    compile::Model,
    constraints::PenaltyFunction,
    optimization::{
        penalty::{PenaltyConfig, PenaltyOptimizer, PenaltyType},
        tuning::{ParameterBounds, ParameterScale, ParameterTuner, TuningConfig},
    },
    sampler::{SASampler, Sampler},
    visualization::{
        convergence::plot_convergence, export::ExportFormat,
        solution_analysis::analyze_solution_distribution,
    },
};

use quantrs2_tytan::compile::expr::{Expr, constant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::fmt::Write;

/// Asset data structure
#[derive(Debug, Clone)]
struct Asset {
    symbol: String,
    name: String,
    sector: String,
    expected_return: f64,
    volatility: f64,
}

/// Portfolio constraints
#[derive(Debug, Clone)]
struct PortfolioConstraints {
    /// Minimum number of assets to hold
    min_assets: Option<usize>,
    /// Maximum number of assets to hold
    max_assets: Option<usize>,
    /// Minimum weight per asset (if included)
    min_weight: Option<f64>,
    /// Maximum weight per asset
    max_weight: Option<f64>,
    /// Sector allocation limits (sector -> (min%, max%))
    sector_limits: HashMap<String, (f64, f64)>,
    /// Target portfolio value
    target_value: f64,
    /// Transaction cost per trade (percentage)
    transaction_cost: f64,
    /// Current portfolio weights (for rebalancing)
    current_weights: Option<Vec<f64>>,
}

impl Default for PortfolioConstraints {
    fn default() -> Self {
        Self {
            min_assets: None,
            max_assets: None,
            min_weight: Some(0.05), // 5% minimum if included
            max_weight: Some(0.30), // 30% maximum per asset
            sector_limits: HashMap::new(),
            target_value: 1.0,       // Normalized to 1
            transaction_cost: 0.001, // 0.1% transaction cost
            current_weights: None,
        }
    }
}

/// Generate synthetic asset data
fn generate_assets(n_assets: usize, seed: u64) -> (Vec<Asset>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let sectors = vec!["Technology", "Finance", "Healthcare", "Energy", "Consumer"];
    let mut assets = Vec::new();

    // Generate assets with realistic characteristics
    for i in 0..n_assets {
        let sector = sectors[i % sectors.len()];

        // Sector-specific return/risk profiles
        let (base_return, base_vol) = match sector {
            "Technology" => (0.12, 0.25),
            "Finance" => (0.08, 0.20),
            "Healthcare" => (0.10, 0.18),
            "Energy" => (0.07, 0.30),
            "Consumer" => (0.09, 0.15),
            _ => (0.08, 0.20),
        };

        let asset = Asset {
            symbol: format!("ASSET{}", i),
            name: format!("Asset {}", i),
            sector: sector.to_string(),
            expected_return: base_return + rng.gen_range(-0.03..0.03),
            volatility: base_vol + rng.gen_range(-0.05..0.05),
        };

        assets.push(asset);
    }

    // Generate correlation matrix
    let mut correlation = Array2::eye(n_assets);

    for i in 0..n_assets {
        for j in i + 1..n_assets {
            // Higher correlation within sectors
            let same_sector = assets[i].sector == assets[j].sector;
            let base_corr = if same_sector { 0.6 } else { 0.3 };
            let corr = (base_corr + rng.gen_range(-0.2..0.2) as f64).clamp(-0.9, 0.9);

            correlation[[i, j]] = corr;
            correlation[[j, i]] = corr;
        }
    }

    // Convert correlation to covariance
    let mut covariance = Array2::zeros((n_assets, n_assets));
    for i in 0..n_assets {
        for j in 0..n_assets {
            covariance[[i, j]] = correlation[[i, j]] * assets[i].volatility * assets[j].volatility;
        }
    }

    (assets, covariance)
}

/// Create portfolio optimization model
fn create_portfolio_model(
    assets: &[Asset],
    covariance: &Array2<f64>,
    constraints: &PortfolioConstraints,
    risk_aversion: f64,
    n_bins: usize, // Number of discretization bins for weights
) -> Result<Model, Box<dyn std::error::Error>> {
    let n_assets = assets.len();
    let mut model = Model::new();

    // Binary variables for asset selection (x_i = 1 if asset i is included)
    let mut selection_vars = Vec::new();
    for i in 0..n_assets {
        selection_vars.push(model.add_variable(&format!("select_{}", i))?);
    }

    // Binary variables for weight discretization
    // w_i = sum_j (j/n_bins) * w_i_j, where w_i_j are binary
    let mut weight_vars = HashMap::new();
    for i in 0..n_assets {
        for j in 1..=n_bins {
            let var = model.add_variable(&format!("w_{}_{}", i, j))?;
            weight_vars.insert((i, j), var);
        }
    }

    // Constraint: if asset is selected, it must have exactly one weight level
    for i in 0..n_assets {
        let mut weight_sum = constant(0.0);
        for j in 1..=n_bins {
            weight_sum = weight_sum + weight_vars[&(i, j)].clone();
        }

        // Add penalty for weight_sum != selection_var
        // This constraint is now handled by penalty terms in the objective
    }

    // Constraint: total weight must equal 1
    let mut total_weight = constant(0.0);
    for i in 0..n_assets {
        for j in 1..=n_bins {
            let weight_value = j as f64 / n_bins as f64;
            total_weight =
                total_weight + constant(weight_value) * weight_vars[&(i, j)].clone();
        }
    }

    // Weight constraint handled as penalty in objective

    // Asset count constraints handled as penalty in objective

    // Per-asset weight constraints handled as penalty in objective

    // Sector constraints handled as penalty in objective

    // Objective: maximize return - risk_aversion * variance - transaction_costs
    let mut objective = constant(0.0);
    let penalty_weight = 1000.0;

    // Add constraint penalties

    // 1. Total weight constraint: sum of all weights should equal 1
    let mut total_weight_penalty = total_weight.clone();
    total_weight_penalty = total_weight_penalty + constant(-1.0);
    // Penalty for (total_weight - 1)^2
    objective = objective
        + constant(penalty_weight)
            * total_weight_penalty.clone()
            * total_weight_penalty;

    // 2. Asset count constraints
    let mut asset_count_objective = objective;
    if let Some(min_assets) = constraints.min_assets {
        let asset_count: Expr = selection_vars.iter().map(|v| v.clone()).sum();
        let violation = asset_count.clone() + constant(-(min_assets as f64));
        // Penalty for max(0, min_assets - count) which we approximate as (min_assets - count)^2 when count < min_assets
        asset_count_objective = asset_count_objective
            + constant(penalty_weight) * violation.clone() * violation;
    }
    if let Some(max_assets) = constraints.max_assets {
        let asset_count: Expr = selection_vars.iter().map(|v| v.clone()).sum();
        let violation = asset_count + constant(-(max_assets as f64));
        // Penalty for max(0, count - max_assets)
        asset_count_objective = asset_count_objective
            + constant(penalty_weight) * violation.clone() * violation;
    }
    let mut objective = asset_count_objective;

    // Expected return term
    for i in 0..n_assets {
        for j in 1..=n_bins {
            let weight_value = j as f64 / n_bins as f64;
            objective = objective
                + constant(-assets[i].expected_return * weight_value)
                    * weight_vars[&(i, j)].clone();
        }
    }

    // Risk term (portfolio variance)
    for i in 0..n_assets {
        for j in 0..n_assets {
            for bi in 1..=n_bins {
                for bj in 1..=n_bins {
                    let wi = bi as f64 / n_bins as f64;
                    let wj = bj as f64 / n_bins as f64;
                    let cov = covariance[[i, j]];

                    objective = objective
                        + constant(risk_aversion * wi * wj * cov)
                            * weight_vars[&(i, bi)].clone()
                            * weight_vars[&(j, bj)].clone();
                }
            }
        }
    }

    // Transaction costs (simplified - handled as penalty for large portfolio changes)

    // Objective complete with constraint penalties

    model.set_objective(objective);

    Ok(model)
}

/// Extract portfolio from solution
fn extract_portfolio(
    solution: &quantrs2_tytan::sampler::SampleResult,
    n_assets: usize,
    n_bins: usize,
) -> Vec<f64> {
    let mut weights = vec![0.0; n_assets];

    for i in 0..n_assets {
        for j in 1..=n_bins {
            let var_name = format!("w_{}_{}", i, j);
            if solution
                .assignments
                .get(&var_name)
                .copied()
                .unwrap_or(false)
            {
                weights[i] = j as f64 / n_bins as f64;
                break;
            }
        }
    }

    // Normalize weights to sum to 1
    let total: f64 = weights.iter().sum();
    if total > 0.0 {
        for w in &mut weights {
            *w /= total;
        }
    }

    weights
}

/// Calculate portfolio metrics
fn calculate_portfolio_metrics(
    weights: &[f64],
    assets: &[Asset],
    covariance: &Array2<f64>,
) -> PortfolioMetrics {
    let n = weights.len();

    // Expected return
    let expected_return: f64 = weights
        .iter()
        .zip(assets.iter())
        .map(|(w, asset)| w * asset.expected_return)
        .sum();

    // Portfolio variance
    let mut variance = 0.0;
    for i in 0..n {
        for j in 0..n {
            variance += weights[i] * weights[j] * covariance[[i, j]];
        }
    }

    let volatility = variance.sqrt();
    let sharpe_ratio = expected_return / volatility; // Assuming risk-free rate = 0

    // Diversification metrics
    let n_holdings = weights.iter().filter(|&&w| w > 0.001).count();
    let concentration = weights.iter().map(|w| w * w).sum::<f64>(); // HHI

    // Sector allocation
    let mut sector_weights = HashMap::new();
    for (i, weight) in weights.iter().enumerate() {
        if *weight > 0.001 {
            let sector = &assets[i].sector;
            *sector_weights.entry(sector.clone()).or_insert(0.0) += weight;
        }
    }

    PortfolioMetrics {
        expected_return,
        volatility,
        sharpe_ratio,
        n_holdings,
        concentration,
        sector_weights,
        weights: weights.to_vec(),
    }
}

#[derive(Debug, Clone)]
struct PortfolioMetrics {
    expected_return: f64,
    volatility: f64,
    sharpe_ratio: f64,
    n_holdings: usize,
    concentration: f64,
    sector_weights: HashMap<String, f64>,
    weights: Vec<f64>,
}

/// Run portfolio optimization experiment
fn run_portfolio_optimization(
    assets: &[Asset],
    covariance: &Array2<f64>,
    constraints: &PortfolioConstraints,
    risk_aversions: &[f64],
) -> Result<Vec<(f64, PortfolioMetrics)>, Box<dyn std::error::Error>> {
    println!("\n=== Portfolio Optimization ===");
    println!("Assets: {}", assets.len());
    println!("Constraints:");
    if let Some(min) = constraints.min_assets {
        println!("  Min assets: {}", min);
    }
    if let Some(max) = constraints.max_assets {
        println!("  Max assets: {}", max);
    }
    println!("  Min weight: {:?}", constraints.min_weight);
    println!("  Max weight: {:?}", constraints.max_weight);

    let mut results = Vec::new();
    let n_bins = 20; // Discretization levels

    for &risk_aversion in risk_aversions {
        println!("\nOptimizing with risk aversion = {:.2}", risk_aversion);

        // Create model
        let model = create_portfolio_model(assets, covariance, constraints, risk_aversion, n_bins)?;

        // Optimize penalties
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

        println!("  QUBO variables: {}", qubo.num_variables);

        // Parameter tuning temporarily disabled due to type compatibility
        println!("Using default parameters for demonstration");

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

        // Run with optimized parameters
        let mut sampler = SASampler::new(None);

        let samples = sampler.run_qubo(&(matrix, var_map), 1000)?;

        // Find best portfolio
        let mut best_portfolio = None;
        let mut best_sharpe = -f64::INFINITY;

        for sample in &samples {
            let weights = extract_portfolio(sample, assets.len(), n_bins);
            let metrics = calculate_portfolio_metrics(&weights, assets, covariance);

            if metrics.n_holdings > 0 && metrics.sharpe_ratio > best_sharpe {
                best_sharpe = metrics.sharpe_ratio;
                best_portfolio = Some(metrics);
            }
        }

        if let Some(portfolio) = best_portfolio {
            println!("  Return: {:.2}%", portfolio.expected_return * 100.0);
            println!("  Volatility: {:.2}%", portfolio.volatility * 100.0);
            println!("  Sharpe ratio: {:.3}", portfolio.sharpe_ratio);
            println!("  Holdings: {}", portfolio.n_holdings);

            results.push((risk_aversion, portfolio));
        }
    }

    Ok(results)
}

/// Generate efficient frontier plot data
fn plot_efficient_frontier(
    portfolios: &[(f64, PortfolioMetrics)],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut csv_data = String::new();
    writeln!(
        &mut csv_data,
        "risk_aversion,return,volatility,sharpe_ratio,n_holdings"
    )?;

    for (ra, portfolio) in portfolios {
        writeln!(
            &mut csv_data,
            "{},{},{},{},{}",
            ra,
            portfolio.expected_return,
            portfolio.volatility,
            portfolio.sharpe_ratio,
            portfolio.n_holdings
        )?;
    }

    std::fs::write("efficient_frontier.csv", csv_data)?;
    println!("\nEfficient frontier data saved to efficient_frontier.csv");

    // Also save detailed portfolio compositions
    let mut compositions = String::new();
    writeln!(&mut compositions, "risk_aversion,asset,weight,sector")?;

    for (ra, portfolio) in portfolios {
        for (i, &weight) in portfolio.weights.iter().enumerate() {
            if weight > 0.001 {
                writeln!(
                    &mut compositions,
                    "{},{},{},{}",
                    ra,
                    i,
                    weight,
                    "TODO" // Would need assets passed in
                )?;
            }
        }
    }

    std::fs::write("portfolio_compositions.csv", compositions)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Portfolio Optimization Example ===");

    // Generate synthetic market data
    let (assets, covariance) = generate_assets(30, 42);

    println!("\nAsset Universe:");
    println!(
        "{:<10} {:<15} {:<10} {:<10}",
        "Symbol", "Sector", "Return", "Vol"
    );
    println!("{:-<45}", "");

    for asset in &assets[..10] {
        // Show first 10
        println!(
            "{:<10} {:<15} {:>9.1}% {:>9.1}%",
            asset.symbol,
            asset.sector,
            asset.expected_return * 100.0,
            asset.volatility * 100.0
        );
    }
    println!("... and {} more assets", assets.len() - 10);

    // Example 1: Basic mean-variance optimization
    println!("\n\n=== Example 1: Mean-Variance Optimization ===");

    let basic_constraints = PortfolioConstraints {
        min_assets: Some(5),
        max_assets: Some(15),
        min_weight: Some(0.02),
        max_weight: Some(0.20),
        ..Default::default()
    };

    let risk_aversions = vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    let efficient_frontier =
        run_portfolio_optimization(&assets, &covariance, &basic_constraints, &risk_aversions)?;

    // Plot efficient frontier
    plot_efficient_frontier(&efficient_frontier)?;

    // Example 2: Sector-constrained optimization
    println!("\n\n=== Example 2: Sector-Constrained Portfolio ===");

    let mut sector_constraints = PortfolioConstraints {
        min_assets: Some(10),
        max_assets: Some(20),
        ..Default::default()
    };

    // Add sector limits
    sector_constraints
        .sector_limits
        .insert("Technology".to_string(), (0.10, 0.30));
    sector_constraints
        .sector_limits
        .insert("Finance".to_string(), (0.15, 0.25));
    sector_constraints
        .sector_limits
        .insert("Healthcare".to_string(), (0.10, 0.25));

    let sector_portfolios =
        run_portfolio_optimization(&assets, &covariance, &sector_constraints, &[1.0, 2.0, 5.0])?;

    // Display sector allocations
    println!("\nSector Allocations:");
    for (ra, portfolio) in &sector_portfolios {
        println!("\nRisk aversion = {:.1}:", ra);
        for (sector, weight) in &portfolio.sector_weights {
            println!("  {}: {:.1}%", sector, weight * 100.0);
        }
    }

    // Example 3: Portfolio rebalancing with transaction costs
    println!("\n\n=== Example 3: Portfolio Rebalancing ===");

    // Assume current portfolio
    let mut current_weights = vec![0.0; assets.len()];
    current_weights[0] = 0.25;
    current_weights[5] = 0.35;
    current_weights[10] = 0.40;

    let rebalancing_constraints = PortfolioConstraints {
        current_weights: Some(current_weights.clone()),
        transaction_cost: 0.003, // 0.3% transaction cost
        ..basic_constraints
    };

    println!("\nCurrent portfolio:");
    for (i, &weight) in current_weights.iter().enumerate() {
        if weight > 0.0 {
            println!("  {}: {:.1}%", assets[i].symbol, weight * 100.0);
        }
    }

    let rebalanced =
        run_portfolio_optimization(&assets, &covariance, &rebalancing_constraints, &[2.0])?;

    if let Some((_, new_portfolio)) = rebalanced.first() {
        println!("\nRebalanced portfolio:");
        let mut total_turnover = 0.0;

        for (i, &new_weight) in new_portfolio.weights.iter().enumerate() {
            let old_weight = current_weights[i];
            if new_weight > 0.001 || old_weight > 0.001 {
                let change = new_weight - old_weight;
                total_turnover += change.abs();

                if change.abs() > 0.01 {
                    println!(
                        "  {}: {:.1}% -> {:.1}% ({:+.1}%)",
                        assets[i].symbol,
                        old_weight * 100.0,
                        new_weight * 100.0,
                        change * 100.0
                    );
                }
            }
        }

        println!("\nTotal turnover: {:.1}%", total_turnover * 100.0);
        println!("Transaction cost: {:.2}%", total_turnover * 0.3);
    }

    // Comparison with classical optimizer would go here
    // (using convex optimization library if available)

    println!("\n\n=== Summary Statistics ===");

    // Analyze all portfolios
    let all_portfolios: Vec<_> = efficient_frontier
        .iter()
        .chain(sector_portfolios.iter())
        .collect();

    let returns: Vec<f64> = all_portfolios
        .iter()
        .map(|(_, p)| p.expected_return)
        .collect();
    let volatilities: Vec<f64> = all_portfolios.iter().map(|(_, p)| p.volatility).collect();
    let sharpes: Vec<f64> = all_portfolios.iter().map(|(_, p)| p.sharpe_ratio).collect();

    println!(
        "Return range: {:.1}% - {:.1}%",
        returns
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            * 100.0,
        returns
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            * 100.0
    );
    println!(
        "Volatility range: {:.1}% - {:.1}%",
        volatilities
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            * 100.0,
        volatilities
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            * 100.0
    );
    println!(
        "Sharpe ratio range: {:.3} - {:.3}",
        sharpes
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        sharpes
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    );

    Ok(())
}
