//! Finance Industry Optimization
//!
//! This module provides optimization solutions for the finance industry,
//! including portfolio optimization, risk management, and fraud detection.

use std::collections::HashMap;
use crate::qubo::{QuboFormulation, QuboBuilder};
use crate::ising::IsingModel;
use crate::simulator::{AnnealingParams, ClassicalAnnealingSimulator};
use super::{ApplicationError, ApplicationResult, OptimizationProblem, IndustrySolution, IndustryConstraint, IndustryObjective};

/// Portfolio optimization problem
#[derive(Debug, Clone)]
pub struct PortfolioOptimization {
    /// Asset expected returns
    pub expected_returns: Vec<f64>,
    /// Covariance matrix of asset returns
    pub covariance_matrix: Vec<Vec<f64>>,
    /// Investment budget
    pub budget: f64,
    /// Risk tolerance parameter
    pub risk_tolerance: f64,
    /// Minimum position sizes
    pub min_positions: Vec<f64>,
    /// Maximum position sizes
    pub max_positions: Vec<f64>,
    /// Sector constraints (asset -> sector mapping)
    pub sector_constraints: HashMap<usize, String>,
    /// Maximum sector allocation
    pub max_sector_allocation: HashMap<String, f64>,
    /// Transaction costs
    pub transaction_costs: Vec<f64>,
    /// Regulatory constraints
    pub regulatory_constraints: Vec<IndustryConstraint>,
}

impl PortfolioOptimization {
    /// Create a new portfolio optimization problem
    pub fn new(
        expected_returns: Vec<f64>,
        covariance_matrix: Vec<Vec<f64>>,
        budget: f64,
        risk_tolerance: f64,
    ) -> ApplicationResult<Self> {
        let n_assets = expected_returns.len();
        
        if covariance_matrix.len() != n_assets {
            return Err(ApplicationError::InvalidConfiguration(
                "Covariance matrix dimension mismatch".to_string()
            ));
        }
        
        for row in &covariance_matrix {
            if row.len() != n_assets {
                return Err(ApplicationError::InvalidConfiguration(
                    "Covariance matrix is not square".to_string()
                ));
            }
        }
        
        Ok(Self {
            expected_returns,
            covariance_matrix,
            budget,
            risk_tolerance,
            min_positions: vec![0.0; n_assets],
            max_positions: vec![budget; n_assets],
            sector_constraints: HashMap::new(),
            max_sector_allocation: HashMap::new(),
            transaction_costs: vec![0.0; n_assets],
            regulatory_constraints: Vec::new(),
        })
    }
    
    /// Add sector constraint
    pub fn add_sector_constraint(
        &mut self, 
        asset: usize, 
        sector: String, 
        max_allocation: f64
    ) -> ApplicationResult<()> {
        if asset >= self.expected_returns.len() {
            return Err(ApplicationError::InvalidConfiguration(
                "Asset index out of bounds".to_string()
            ));
        }
        
        self.sector_constraints.insert(asset, sector.clone());
        self.max_sector_allocation.insert(sector, max_allocation);
        Ok(())
    }
    
    /// Set position bounds
    pub fn set_position_bounds(&mut self, asset: usize, min: f64, max: f64) -> ApplicationResult<()> {
        if asset >= self.expected_returns.len() {
            return Err(ApplicationError::InvalidConfiguration(
                "Asset index out of bounds".to_string()
            ));
        }
        
        self.min_positions[asset] = min;
        self.max_positions[asset] = max;
        Ok(())
    }
    
    /// Calculate portfolio risk
    pub fn calculate_risk(&self, weights: &[f64]) -> f64 {
        let mut risk = 0.0;
        
        for i in 0..weights.len() {
            for j in 0..weights.len() {
                risk += weights[i] * weights[j] * self.covariance_matrix[i][j];
            }
        }
        
        risk.sqrt()
    }
    
    /// Calculate portfolio return
    pub fn calculate_return(&self, weights: &[f64]) -> f64 {
        weights.iter()
            .zip(self.expected_returns.iter())
            .map(|(w, r)| w * r)
            .sum()
    }
    
    /// Calculate Sharpe ratio
    pub fn calculate_sharpe_ratio(&self, weights: &[f64], risk_free_rate: f64) -> f64 {
        let portfolio_return = self.calculate_return(weights);
        let portfolio_risk = self.calculate_risk(weights);
        
        if portfolio_risk > 1e-8 {
            (portfolio_return - risk_free_rate) / portfolio_risk
        } else {
            0.0
        }
    }
}

impl OptimizationProblem for PortfolioOptimization {
    type Solution = PortfolioSolution;
    type ObjectiveValue = f64;
    
    fn description(&self) -> String {
        format!(
            "Portfolio optimization with {} assets, budget ${:.2}, risk tolerance {:.3}",
            self.expected_returns.len(),
            self.budget,
            self.risk_tolerance
        )
    }
    
    fn size_metrics(&self) -> HashMap<String, usize> {
        let mut metrics = HashMap::new();
        metrics.insert("num_assets".to_string(), self.expected_returns.len());
        metrics.insert("num_sectors".to_string(), self.max_sector_allocation.len());
        metrics.insert("num_constraints".to_string(), self.regulatory_constraints.len());
        metrics
    }
    
    fn validate(&self) -> ApplicationResult<()> {
        if self.expected_returns.is_empty() {
            return Err(ApplicationError::DataValidationError(
                "No assets provided".to_string()
            ));
        }
        
        if self.budget <= 0.0 {
            return Err(ApplicationError::DataValidationError(
                "Budget must be positive".to_string()
            ));
        }
        
        if self.risk_tolerance < 0.0 {
            return Err(ApplicationError::DataValidationError(
                "Risk tolerance must be non-negative".to_string()
            ));
        }
        
        // Check covariance matrix is positive semidefinite (simplified check)
        for i in 0..self.covariance_matrix.len() {
            if self.covariance_matrix[i][i] < 0.0 {
                return Err(ApplicationError::DataValidationError(
                    "Covariance matrix has negative diagonal elements".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    fn to_qubo(&self) -> ApplicationResult<QuboFormulation> {
        let n_assets = self.expected_returns.len();
        let precision = 100; // Discretization precision for continuous weights
        
        let mut builder = QuboBuilder::new();
        
        // Binary variables: x_{i,k} = 1 if asset i gets allocation level k
        let mut var_map = HashMap::new();
        let mut var_counter = 0;
        
        for asset in 0..n_assets {
            for level in 0..precision {
                var_map.insert((asset, level), var_counter);
                var_counter += 1;
            }
        }
        
        // Objective: maximize return - risk_penalty * risk
        for asset in 0..n_assets {
            for level in 0..precision {
                let weight = (level as f64) / (precision as f64);
                let var_idx = var_map[&(asset, level)];
                
                // Return term (to be maximized, so negate)
                let return_coeff = -self.expected_returns[asset] * weight * self.budget;
                builder.add_bias(var_idx, return_coeff);
                
                // Risk penalty term
                let risk_coeff = self.risk_tolerance * weight * weight * 
                    self.covariance_matrix[asset][asset] * self.budget * self.budget;
                builder.add_bias(var_idx, risk_coeff);
            }
        }
        
        // Cross-terms for risk calculation
        for asset1 in 0..n_assets {
            for asset2 in (asset1 + 1)..n_assets {
                let covar = self.covariance_matrix[asset1][asset2];
                if covar.abs() > 1e-8 {
                    for level1 in 0..precision {
                        for level2 in 0..precision {
                            let weight1 = (level1 as f64) / (precision as f64);
                            let weight2 = (level2 as f64) / (precision as f64);
                            let var1 = var_map[&(asset1, level1)];
                            let var2 = var_map[&(asset2, level2)];
                            
                            let risk_cross = 2.0 * self.risk_tolerance * weight1 * weight2 * 
                                covar * self.budget * self.budget;
                            builder.add_coupling(var1, var2, risk_cross);
                        }
                    }
                }
            }
        }
        
        // Constraint: exactly one allocation level per asset
        let constraint_penalty = 1000.0;
        for asset in 0..n_assets {
            // Penalty for not selecting exactly one level
            for level1 in 0..precision {
                for level2 in (level1 + 1)..precision {
                    let var1 = var_map[&(asset, level1)];
                    let var2 = var_map[&(asset, level2)];
                    builder.add_coupling(var1, var2, constraint_penalty);
                }
            }
            
            // Penalty for selecting no level
            let mut constraint_bias = constraint_penalty;
            for level in 0..precision {
                let var_idx = var_map[&(asset, level)];
                builder.add_bias(var_idx, -constraint_bias);
            }
        }
        
        builder.build()
    }
    
    fn evaluate_solution(&self, solution: &Self::Solution) -> ApplicationResult<Self::ObjectiveValue> {
        let portfolio_return = self.calculate_return(&solution.weights);
        let portfolio_risk = self.calculate_risk(&solution.weights);
        
        // Mean-variance objective: return - risk_penalty * risk
        Ok(portfolio_return - self.risk_tolerance * portfolio_risk * portfolio_risk)
    }
    
    fn is_feasible(&self, solution: &Self::Solution) -> bool {
        // Check budget constraint
        let total_investment: f64 = solution.weights.iter().sum();
        if (total_investment - 1.0).abs() > 1e-6 {
            return false;
        }
        
        // Check position bounds
        for (i, &weight) in solution.weights.iter().enumerate() {
            let position_value = weight * self.budget;
            if position_value < self.min_positions[i] || position_value > self.max_positions[i] {
                return false;
            }
        }
        
        // Check sector constraints
        let mut sector_allocations = HashMap::new();
        for (asset, sector) in &self.sector_constraints {
            let allocation = *sector_allocations.entry(sector.clone()).or_insert(0.0);
            sector_allocations.insert(sector.clone(), allocation + solution.weights[*asset]);
        }
        
        for (sector, &max_allocation) in &self.max_sector_allocation {
            if let Some(&allocation) = sector_allocations.get(sector) {
                if allocation > max_allocation {
                    return false;
                }
            }
        }
        
        true
    }
}

/// Portfolio optimization solution
#[derive(Debug, Clone)]
pub struct PortfolioSolution {
    /// Asset weights (sum to 1.0)
    pub weights: Vec<f64>,
    /// Portfolio metrics
    pub metrics: PortfolioMetrics,
}

/// Portfolio performance metrics
#[derive(Debug, Clone)]
pub struct PortfolioMetrics {
    /// Expected return
    pub expected_return: f64,
    /// Portfolio volatility (risk)
    pub volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Value at Risk (VaR)
    pub var_95: f64,
    /// Conditional Value at Risk (CVaR)
    pub cvar_95: f64,
}

impl IndustrySolution for PortfolioSolution {
    type Problem = PortfolioOptimization;
    
    fn from_binary(problem: &Self::Problem, binary_solution: &[i8]) -> ApplicationResult<Self> {
        let n_assets = problem.expected_returns.len();
        let precision = 100;
        
        let mut weights = vec![0.0; n_assets];
        let mut var_idx = 0;
        
        for asset in 0..n_assets {
            for level in 0..precision {
                if var_idx < binary_solution.len() && binary_solution[var_idx] == 1 {
                    weights[asset] = (level as f64) / (precision as f64);
                    break;
                }
                var_idx += 1;
            }
        }
        
        // Normalize weights to sum to 1
        let total_weight: f64 = weights.iter().sum();
        if total_weight > 1e-8 {
            for weight in &mut weights {
                *weight /= total_weight;
            }
        }
        
        // Calculate metrics
        let expected_return = problem.calculate_return(&weights);
        let volatility = problem.calculate_risk(&weights);
        let sharpe_ratio = problem.calculate_sharpe_ratio(&weights, 0.02); // 2% risk-free rate
        
        let metrics = PortfolioMetrics {
            expected_return,
            volatility,
            sharpe_ratio,
            max_drawdown: 0.0, // Would require time series data
            var_95: volatility * 1.645, // Simplified VaR calculation
            cvar_95: volatility * 2.0, // Simplified CVaR calculation
        };
        
        Ok(Self { weights, metrics })
    }
    
    fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("type".to_string(), "Portfolio Optimization".to_string());
        summary.insert("num_assets".to_string(), self.weights.len().to_string());
        summary.insert("expected_return".to_string(), format!("{:.2}%", self.metrics.expected_return * 100.0));
        summary.insert("volatility".to_string(), format!("{:.2}%", self.metrics.volatility * 100.0));
        summary.insert("sharpe_ratio".to_string(), format!("{:.3}", self.metrics.sharpe_ratio));
        
        // Top 5 positions
        let mut indexed_weights: Vec<(usize, f64)> = self.weights.iter().enumerate().map(|(i, &w)| (i, w)).collect();
        indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let top_positions: Vec<String> = indexed_weights.iter().take(5)
            .map(|(i, w)| format!("Asset {}: {:.1}%", i, w * 100.0))
            .collect();
        summary.insert("top_positions".to_string(), top_positions.join(", "));
        
        summary
    }
    
    fn metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("expected_return".to_string(), self.metrics.expected_return);
        metrics.insert("volatility".to_string(), self.metrics.volatility);
        metrics.insert("sharpe_ratio".to_string(), self.metrics.sharpe_ratio);
        metrics.insert("var_95".to_string(), self.metrics.var_95);
        metrics.insert("cvar_95".to_string(), self.metrics.cvar_95);
        
        // Concentration metrics
        let herfindahl_index: f64 = self.weights.iter().map(|w| w * w).sum();
        metrics.insert("concentration_hhi".to_string(), herfindahl_index);
        
        let max_weight = self.weights.iter().fold(0.0, |a, &b| a.max(b));
        metrics.insert("max_position".to_string(), max_weight);
        
        metrics
    }
    
    fn export_format(&self) -> ApplicationResult<String> {
        let mut output = String::new();
        output.push_str("# Portfolio Allocation Report\n\n");
        
        output.push_str("## Asset Allocation\n");
        for (i, &weight) in self.weights.iter().enumerate() {
            if weight > 0.001 { // Only show significant positions
                output.push_str(&format!("Asset {}: {:.2}%\n", i, weight * 100.0));
            }
        }
        
        output.push_str("\n## Risk Metrics\n");
        output.push_str(&format!("Expected Return: {:.2}%\n", self.metrics.expected_return * 100.0));
        output.push_str(&format!("Volatility: {:.2}%\n", self.metrics.volatility * 100.0));
        output.push_str(&format!("Sharpe Ratio: {:.3}\n", self.metrics.sharpe_ratio));
        output.push_str(&format!("VaR (95%): {:.2}%\n", self.metrics.var_95 * 100.0));
        output.push_str(&format!("CVaR (95%): {:.2}%\n", self.metrics.cvar_95 * 100.0));
        
        Ok(output)
    }
}

/// Risk management problem for financial institutions
#[derive(Debug, Clone)]
pub struct RiskManagement {
    /// Portfolio positions
    pub positions: Vec<f64>,
    /// Risk factors and exposures
    pub risk_factors: HashMap<String, Vec<f64>>,
    /// Risk limits by factor
    pub risk_limits: HashMap<String, f64>,
    /// Stress test scenarios
    pub stress_scenarios: Vec<HashMap<String, f64>>,
    /// Current market data
    pub market_data: HashMap<String, f64>,
}

impl RiskManagement {
    /// Create new risk management problem
    pub fn new(positions: Vec<f64>) -> Self {
        Self {
            positions,
            risk_factors: HashMap::new(),
            risk_limits: HashMap::new(),
            stress_scenarios: Vec::new(),
            market_data: HashMap::new(),
        }
    }
    
    /// Add risk factor exposure
    pub fn add_risk_factor(&mut self, name: String, exposures: Vec<f64>, limit: f64) -> ApplicationResult<()> {
        if exposures.len() != self.positions.len() {
            return Err(ApplicationError::InvalidConfiguration(
                "Risk factor exposure dimension mismatch".to_string()
            ));
        }
        
        self.risk_factors.insert(name.clone(), exposures);
        self.risk_limits.insert(name, limit);
        Ok(())
    }
    
    /// Calculate total risk exposure for a factor
    pub fn calculate_factor_exposure(&self, factor: &str) -> f64 {
        if let Some(exposures) = self.risk_factors.get(factor) {
            self.positions.iter()
                .zip(exposures.iter())
                .map(|(pos, exp)| pos * exp)
                .sum()
        } else {
            0.0
        }
    }
    
    /// Run stress test
    pub fn run_stress_test(&self, scenario: &HashMap<String, f64>) -> f64 {
        let mut total_impact = 0.0;
        
        for (factor, &shock) in scenario {
            if let Some(exposures) = self.risk_factors.get(factor) {
                let factor_exposure = self.calculate_factor_exposure(factor);
                total_impact += factor_exposure * shock;
            }
        }
        
        total_impact
    }
}

/// Credit risk assessment problem
#[derive(Debug, Clone)]
pub struct CreditRiskAssessment {
    /// Loan applications
    pub applications: Vec<CreditApplication>,
    /// Risk model parameters
    pub risk_model: CreditRiskModel,
    /// Portfolio constraints
    pub portfolio_constraints: Vec<IndustryConstraint>,
}

/// Credit application data
#[derive(Debug, Clone)]
pub struct CreditApplication {
    /// Application ID
    pub id: String,
    /// Loan amount requested
    pub amount: f64,
    /// Applicant credit score
    pub credit_score: f64,
    /// Debt-to-income ratio
    pub debt_to_income: f64,
    /// Employment history (years)
    pub employment_years: f64,
    /// Collateral value
    pub collateral_value: f64,
    /// Loan purpose
    pub purpose: String,
    /// Additional features
    pub features: HashMap<String, f64>,
}

/// Credit risk model
#[derive(Debug, Clone)]
pub struct CreditRiskModel {
    /// Feature weights
    pub weights: HashMap<String, f64>,
    /// Risk threshold
    pub risk_threshold: f64,
    /// Expected loss rates by risk bucket
    pub loss_rates: Vec<f64>,
}

impl CreditRiskAssessment {
    /// Calculate probability of default for an application
    pub fn calculate_pd(&self, application: &CreditApplication) -> f64 {
        let mut score = 0.0;
        
        // Standard credit features
        score += self.risk_model.weights.get("credit_score").unwrap_or(&0.0) * application.credit_score;
        score += self.risk_model.weights.get("debt_to_income").unwrap_or(&0.0) * application.debt_to_income;
        score += self.risk_model.weights.get("employment_years").unwrap_or(&0.0) * application.employment_years;
        
        // Additional features
        for (feature, value) in &application.features {
            score += self.risk_model.weights.get(feature).unwrap_or(&0.0) * value;
        }
        
        // Convert to probability using logistic function
        1.0 / (1.0 + (-score).exp())
    }
    
    /// Calculate expected loss for a portfolio selection
    pub fn calculate_expected_loss(&self, selection: &[bool]) -> f64 {
        let mut total_loss = 0.0;
        
        for (i, &selected) in selection.iter().enumerate() {
            if selected && i < self.applications.len() {
                let app = &self.applications[i];
                let pd = self.calculate_pd(app);
                let lgd = 0.45; // Loss given default (typical value)
                let ead = app.amount; // Exposure at default
                
                total_loss += pd * lgd * ead;
            }
        }
        
        total_loss
    }
}

/// Utility functions for finance applications

/// Create benchmark portfolio optimization problems
pub fn create_benchmark_problems(num_assets: usize) -> ApplicationResult<Vec<Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>>> {
    let mut problems = Vec::new();
    
    // Problem 1: Conservative portfolio
    let conservative_returns: Vec<f64> = (0..num_assets).map(|i| 0.03 + 0.02 * (i as f64) / (num_assets as f64)).collect();
    let conservative_covar = create_sample_covariance_matrix(num_assets, 0.15);
    let conservative_portfolio = PortfolioOptimization::new(
        conservative_returns,
        conservative_covar,
        1000000.0,
        0.5, // Low risk tolerance
    )?;
    
    problems.push(Box::new(conservative_portfolio) as Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>);
    
    // Problem 2: Aggressive portfolio
    let aggressive_returns: Vec<f64> = (0..num_assets).map(|i| 0.05 + 0.10 * (i as f64) / (num_assets as f64)).collect();
    let aggressive_covar = create_sample_covariance_matrix(num_assets, 0.25);
    let aggressive_portfolio = PortfolioOptimization::new(
        aggressive_returns,
        aggressive_covar,
        1000000.0,
        0.1, // High risk tolerance
    )?;
    
    problems.push(Box::new(aggressive_portfolio) as Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>);
    
    // Problem 3: Sector-constrained portfolio
    let mut sector_portfolio = PortfolioOptimization::new(
        (0..num_assets).map(|i| 0.04 + 0.06 * (i as f64) / (num_assets as f64)).collect(),
        create_sample_covariance_matrix(num_assets, 0.20),
        1000000.0,
        0.3,
    )?;
    
    // Add sector constraints
    for i in 0..num_assets {
        let sector = format!("Sector_{}", i % 5); // 5 sectors
        sector_portfolio.add_sector_constraint(i, sector, 0.3)?; // Max 30% per sector
    }
    
    problems.push(Box::new(sector_portfolio) as Box<dyn OptimizationProblem<Solution = Vec<i8>, ObjectiveValue = f64>>);
    
    Ok(problems)
}

/// Create sample covariance matrix for testing
fn create_sample_covariance_matrix(n: usize, base_volatility: f64) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in 0..n {
            if i == j {
                // Diagonal: individual variances
                matrix[i][j] = base_volatility * base_volatility * (1.0 + 0.5 * (i as f64) / (n as f64));
            } else {
                // Off-diagonal: correlations
                let correlation = 0.1 * (1.0 - (i as f64 - j as f64).abs() / (n as f64));
                let vol_i = (matrix[i][i]).sqrt();
                let vol_j = (matrix[j][j]).sqrt();
                matrix[i][j] = correlation * vol_i * vol_j;
            }
        }
    }
    
    matrix
}

/// Solve portfolio optimization problem
pub fn solve_portfolio_optimization(
    problem: &PortfolioOptimization,
    params: Option<AnnealingParams>,
) -> ApplicationResult<PortfolioSolution> {
    // Convert to QUBO
    let qubo = problem.to_qubo()?;
    
    // Convert to Ising
    let ising = IsingModel::from_qubo(&qubo)?;
    
    // Set up annealing parameters
    let annealing_params = params.unwrap_or_else(|| {
        let mut p = AnnealingParams::default();
        p.num_sweeps = 10000;
        p.num_repetitions = 20;
        p.initial_temperature = 2.0;
        p.final_temperature = 0.01;
        p
    });
    
    // Solve with classical annealing
    let simulator = ClassicalAnnealingSimulator::new(annealing_params)
        .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;
    
    let result = simulator.solve(&ising)
        .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;
    
    // Convert solution back to portfolio
    PortfolioSolution::from_binary(problem, &result.best_spins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_optimization_creation() {
        let returns = vec![0.05, 0.08, 0.06];
        let covar = vec![
            vec![0.04, 0.01, 0.02],
            vec![0.01, 0.09, 0.03],
            vec![0.02, 0.03, 0.05],
        ];
        
        let portfolio = PortfolioOptimization::new(returns, covar, 100000.0, 0.5).unwrap();
        assert_eq!(portfolio.expected_returns.len(), 3);
        assert_eq!(portfolio.budget, 100000.0);
    }
    
    #[test]
    fn test_portfolio_risk_calculation() {
        let returns = vec![0.05, 0.08];
        let covar = vec![
            vec![0.04, 0.01],
            vec![0.01, 0.09],
        ];
        
        let portfolio = PortfolioOptimization::new(returns, covar, 100000.0, 0.5).unwrap();
        
        let weights = vec![0.6, 0.4];
        let risk = portfolio.calculate_risk(&weights);
        
        // Expected: sqrt(0.6^2 * 0.04 + 0.4^2 * 0.09 + 2 * 0.6 * 0.4 * 0.01)
        let expected_risk = (0.36 * 0.04 + 0.16 * 0.09 + 2.0 * 0.6 * 0.4 * 0.01).sqrt();
        
        assert!((risk - expected_risk).abs() < 1e-10);
    }
    
    #[test]
    fn test_portfolio_return_calculation() {
        let returns = vec![0.05, 0.08];
        let covar = vec![
            vec![0.04, 0.01],
            vec![0.01, 0.09],
        ];
        
        let portfolio = PortfolioOptimization::new(returns, covar, 100000.0, 0.5).unwrap();
        
        let weights = vec![0.6, 0.4];
        let portfolio_return = portfolio.calculate_return(&weights);
        
        let expected_return = 0.6 * 0.05 + 0.4 * 0.08;
        assert!((portfolio_return - expected_return).abs() < 1e-10);
    }
    
    #[test]
    fn test_portfolio_validation() {
        // Valid portfolio
        let returns = vec![0.05, 0.08];
        let covar = vec![
            vec![0.04, 0.01],
            vec![0.01, 0.09],
        ];
        
        let portfolio = PortfolioOptimization::new(returns, covar, 100000.0, 0.5).unwrap();
        assert!(portfolio.validate().is_ok());
        
        // Invalid portfolio (negative budget)
        let invalid = PortfolioOptimization::new(vec![0.05], vec![vec![0.04]], -1000.0, 0.5);
        assert!(invalid.is_err());
    }
    
    #[test]
    fn test_sector_constraints() {
        let returns = vec![0.05, 0.08, 0.06];
        let covar = create_sample_covariance_matrix(3, 0.2);
        
        let mut portfolio = PortfolioOptimization::new(returns, covar, 100000.0, 0.5).unwrap();
        
        assert!(portfolio.add_sector_constraint(0, "Tech".to_string(), 0.5).is_ok());
        assert!(portfolio.add_sector_constraint(1, "Tech".to_string(), 0.5).is_ok());
        assert!(portfolio.add_sector_constraint(5, "Finance".to_string(), 0.3).is_err()); // Invalid asset index
    }
    
    #[test]
    fn test_credit_risk_calculation() {
        let app = CreditApplication {
            id: "TEST001".to_string(),
            amount: 50000.0,
            credit_score: 720.0,
            debt_to_income: 0.3,
            employment_years: 5.0,
            collateral_value: 60000.0,
            purpose: "Home".to_string(),
            features: HashMap::new(),
        };
        
        let mut weights = HashMap::new();
        weights.insert("credit_score".to_string(), 0.002);
        weights.insert("debt_to_income".to_string(), -2.0);
        weights.insert("employment_years".to_string(), 0.1);
        
        let risk_model = CreditRiskModel {
            weights,
            risk_threshold: 0.05,
            loss_rates: vec![0.01, 0.03, 0.05, 0.10],
        };
        
        let assessment = CreditRiskAssessment {
            applications: vec![app],
            risk_model,
            portfolio_constraints: Vec::new(),
        };
        
        let pd = assessment.calculate_pd(&assessment.applications[0]);
        assert!(pd > 0.0 && pd < 1.0);
    }
    
    #[test]
    fn test_benchmark_problems() {
        let problems = create_benchmark_problems(5).unwrap();
        assert_eq!(problems.len(), 3);
        
        for problem in &problems {
            assert!(problem.validate().is_ok());
            let metrics = problem.size_metrics();
            assert_eq!(metrics["num_assets"], 5);
        }
    }
}