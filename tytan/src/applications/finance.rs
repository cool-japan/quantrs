//! Finance applications: Portfolio optimization suite.
//!
//! This module provides quantum optimization tools for financial applications
//! including portfolio optimization, risk management, and asset allocation.

// Sampler types available for finance applications
#![allow(dead_code)]

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Portfolio optimization problem
pub struct PortfolioOptimizer {
    /// Asset returns (expected)
    returns: Array1<f64>,
    /// Covariance matrix
    covariance: Array2<f64>,
    /// Risk aversion parameter
    risk_aversion: f64,
    /// Constraints
    constraints: PortfolioConstraints,
    /// Optimization method
    method: OptimizationMethod,
}

#[derive(Debug, Clone, Default)]
pub struct PortfolioConstraints {
    /// Minimum investment per asset
    min_investment: Option<Array1<f64>>,
    /// Maximum investment per asset
    max_investment: Option<Array1<f64>>,
    /// Target return
    target_return: Option<f64>,
    /// Maximum risk (variance)
    max_risk: Option<f64>,
    /// Sector constraints
    sector_constraints: Option<SectorConstraints>,
    /// Cardinality constraint (max number of assets)
    cardinality: Option<usize>,
    /// Transaction costs
    transaction_costs: Option<Array1<f64>>,
    /// Current portfolio (for rebalancing)
    current_portfolio: Option<Array1<f64>>,
}

#[derive(Debug, Clone)]
pub struct SectorConstraints {
    /// Asset to sector mapping
    asset_sectors: Vec<usize>,
    /// Minimum allocation per sector
    min_sector_allocation: Vec<f64>,
    /// Maximum allocation per sector
    max_sector_allocation: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum OptimizationMethod {
    /// Mean-variance optimization (Markowitz)
    MeanVariance,
    /// Conditional Value at Risk (CVaR)
    CVaR { confidence_level: f64 },
    /// Maximum Sharpe ratio
    MaxSharpe { risk_free_rate: f64 },
    /// Risk parity
    RiskParity,
    /// Black-Litterman
    BlackLitterman { views: MarketViews },
    /// Kelly criterion
    KellyCriterion,
}

#[derive(Debug, Clone)]
pub struct MarketViews {
    /// View matrix P
    view_matrix: Array2<f64>,
    /// View expectations Q
    view_expectations: Array1<f64>,
    /// View confidence Omega
    view_confidence: Array2<f64>,
}

impl PortfolioOptimizer {
    /// Create new portfolio optimizer
    pub fn new(
        returns: Array1<f64>,
        covariance: Array2<f64>,
        risk_aversion: f64,
    ) -> Result<Self, String> {
        if returns.len() != covariance.shape()[0] {
            return Err("Returns and covariance dimensions mismatch".to_string());
        }

        Ok(Self {
            returns,
            covariance,
            risk_aversion,
            constraints: PortfolioConstraints::default(),
            method: OptimizationMethod::MeanVariance,
        })
    }

    /// Set optimization method
    pub fn with_method(mut self, method: OptimizationMethod) -> Self {
        self.method = method;
        self
    }

    /// Set constraints
    pub fn with_constraints(mut self, constraints: PortfolioConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Build QUBO formulation
    pub fn build_qubo(
        &self,
        num_bits_per_asset: usize,
    ) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        let n_assets = self.returns.len();
        let total_vars = n_assets * num_bits_per_asset;

        let mut qubo = Array2::zeros((total_vars, total_vars));
        let mut var_map = HashMap::new();

        // Create variable mapping
        for i in 0..n_assets {
            for j in 0..num_bits_per_asset {
                let var_name = format!("asset_{i}_bit_{j}");
                var_map.insert(var_name, i * num_bits_per_asset + j);
            }
        }

        // Build objective based on method
        match &self.method {
            OptimizationMethod::MeanVariance => {
                self.add_mean_variance_objective(&mut qubo, num_bits_per_asset)?;
            }
            OptimizationMethod::CVaR { confidence_level } => {
                self.add_cvar_objective(&mut qubo, num_bits_per_asset, *confidence_level)?;
            }
            OptimizationMethod::MaxSharpe { risk_free_rate } => {
                self.add_sharpe_objective(&mut qubo, num_bits_per_asset, *risk_free_rate)?;
            }
            OptimizationMethod::RiskParity => {
                self.add_risk_parity_objective(&mut qubo, num_bits_per_asset)?;
            }
            OptimizationMethod::BlackLitterman { views } => {
                self.add_black_litterman_objective(&mut qubo, num_bits_per_asset, views)?;
            }
            OptimizationMethod::KellyCriterion => {
                self.add_kelly_criterion_objective(&mut qubo, num_bits_per_asset)?;
            }
        }

        // Add constraints
        self.add_constraints(&mut qubo, num_bits_per_asset)?;

        Ok((qubo, var_map))
    }

    /// Add mean-variance objective
    fn add_mean_variance_objective(
        &self,
        qubo: &mut Array2<f64>,
        bits_per_asset: usize,
    ) -> Result<(), String> {
        let n_assets = self.returns.len();

        // Expected return term (linear)
        for i in 0..n_assets {
            for k in 0..bits_per_asset {
                let weight = (1 << k) as f64 / ((1 << bits_per_asset) - 1) as f64;
                let var_idx = i * bits_per_asset + k;
                qubo[[var_idx, var_idx]] -= self.returns[i] * weight;
            }
        }

        // Risk term (quadratic)
        for i in 0..n_assets {
            for j in 0..n_assets {
                for k1 in 0..bits_per_asset {
                    for k2 in 0..bits_per_asset {
                        let w1 = (1 << k1) as f64 / ((1 << bits_per_asset) - 1) as f64;
                        let w2 = (1 << k2) as f64 / ((1 << bits_per_asset) - 1) as f64;

                        let var_idx1 = i * bits_per_asset + k1;
                        let var_idx2 = j * bits_per_asset + k2;

                        if var_idx1 <= var_idx2 {
                            let coef = self.risk_aversion * self.covariance[[i, j]] * w1 * w2;

                            if var_idx1 == var_idx2 {
                                qubo[[var_idx1, var_idx1]] += coef;
                            } else {
                                qubo[[var_idx1, var_idx2]] += coef / 2.0;
                                qubo[[var_idx2, var_idx1]] += coef / 2.0;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Add CVaR objective
    fn add_cvar_objective(
        &self,
        qubo: &mut Array2<f64>,
        bits_per_asset: usize,
        confidence_level: f64,
    ) -> Result<(), String> {
        // Simplified CVaR approximation
        // In practice, would need scenario-based optimization

        // Use mean-variance as approximation
        self.add_mean_variance_objective(qubo, bits_per_asset)?;

        // Add tail risk penalty
        let tail_weight = 1.0 / (1.0 - confidence_level);
        for i in 0..qubo.shape()[0] {
            for j in 0..qubo.shape()[1] {
                qubo[[i, j]] *= tail_weight;
            }
        }

        Ok(())
    }

    /// Add Sharpe ratio objective
    fn add_sharpe_objective(
        &self,
        qubo: &mut Array2<f64>,
        bits_per_asset: usize,
        risk_free_rate: f64,
    ) -> Result<(), String> {
        // Sharpe ratio is non-linear, use approximation
        // Maximize (return - rf) / sqrt(variance)

        // Adjust returns for risk-free rate
        let adjusted_returns = &self.returns - risk_free_rate;

        // Use modified mean-variance
        let n_assets = adjusted_returns.len();

        for i in 0..n_assets {
            for k in 0..bits_per_asset {
                let weight = (1 << k) as f64 / ((1 << bits_per_asset) - 1) as f64;
                let var_idx = i * bits_per_asset + k;

                // Scale by inverse of portfolio volatility (approximation)
                let mean_var = self
                    .covariance
                    .diag()
                    .mean()
                    .ok_or_else(|| "Empty covariance diagonal".to_string())?;
                let vol_scale = 1.0 / mean_var.sqrt();
                qubo[[var_idx, var_idx]] -= adjusted_returns[i] * weight * vol_scale;
            }
        }

        // Add variance term with higher penalty
        self.add_mean_variance_objective(qubo, bits_per_asset)?;

        Ok(())
    }

    /// Add constraints to QUBO
    fn add_constraints(&self, qubo: &mut Array2<f64>, bits_per_asset: usize) -> Result<(), String> {
        let penalty = 100.0; // Large penalty for constraint violation

        // Budget constraint: sum of weights = 1
        self.add_budget_constraint(qubo, bits_per_asset, penalty)?;

        // Cardinality constraint
        if let Some(max_assets) = self.constraints.cardinality {
            self.add_cardinality_constraint(qubo, bits_per_asset, max_assets, penalty)?;
        }

        // Target return constraint
        if let Some(target) = self.constraints.target_return {
            self.add_return_constraint(qubo, bits_per_asset, target, penalty)?;
        }

        // Risk constraint
        if let Some(max_risk) = self.constraints.max_risk {
            self.add_risk_constraint(qubo, bits_per_asset, max_risk, penalty)?;
        }

        Ok(())
    }

    /// Add budget constraint (weights sum to 1)
    fn add_budget_constraint(
        &self,
        qubo: &mut Array2<f64>,
        bits_per_asset: usize,
        penalty: f64,
    ) -> Result<(), String> {
        let n_assets = self.returns.len();

        // (sum w_i - 1)^2
        for i in 0..n_assets {
            for j in 0..n_assets {
                for k1 in 0..bits_per_asset {
                    for k2 in 0..bits_per_asset {
                        let w1 = (1 << k1) as f64 / ((1 << bits_per_asset) - 1) as f64;
                        let w2 = (1 << k2) as f64 / ((1 << bits_per_asset) - 1) as f64;

                        let var_idx1 = i * bits_per_asset + k1;
                        let var_idx2 = j * bits_per_asset + k2;

                        qubo[[var_idx1, var_idx2]] += penalty * w1 * w2;
                    }
                }
            }
        }

        // Linear term: -2 * sum w_i
        for i in 0..n_assets {
            for k in 0..bits_per_asset {
                let weight = (1 << k) as f64 / ((1 << bits_per_asset) - 1) as f64;
                let var_idx = i * bits_per_asset + k;
                qubo[[var_idx, var_idx]] -= 2.0 * penalty * weight;
            }
        }

        Ok(())
    }

    /// Add cardinality constraint
    fn add_cardinality_constraint(
        &self,
        qubo: &mut Array2<f64>,
        bits_per_asset: usize,
        _max_assets: usize,
        penalty: f64,
    ) -> Result<(), String> {
        // Add binary variables for asset selection
        // Simplified: penalize having too many non-zero weights

        let n_assets = self.returns.len();

        // Count non-zero assets (approximation)
        for i in 0..n_assets {
            for j in i + 1..n_assets {
                // Penalize having both assets
                for k1 in 0..bits_per_asset {
                    for k2 in 0..bits_per_asset {
                        let var_idx1 = i * bits_per_asset + k1;
                        let var_idx2 = j * bits_per_asset + k2;

                        if i != j {
                            qubo[[var_idx1, var_idx2]] += penalty / (n_assets * n_assets) as f64;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Add return constraint
    fn add_return_constraint(
        &self,
        qubo: &mut Array2<f64>,
        bits_per_asset: usize,
        target_return: f64,
        penalty: f64,
    ) -> Result<(), String> {
        // (portfolio_return - target)^2
        let n_assets = self.returns.len();

        // Quadratic term
        for i in 0..n_assets {
            for j in 0..n_assets {
                for k1 in 0..bits_per_asset {
                    for k2 in 0..bits_per_asset {
                        let w1 = (1 << k1) as f64 / ((1 << bits_per_asset) - 1) as f64;
                        let w2 = (1 << k2) as f64 / ((1 << bits_per_asset) - 1) as f64;

                        let var_idx1 = i * bits_per_asset + k1;
                        let var_idx2 = j * bits_per_asset + k2;

                        qubo[[var_idx1, var_idx2]] +=
                            penalty * self.returns[i] * self.returns[j] * w1 * w2;
                    }
                }
            }
        }

        // Linear term
        for i in 0..n_assets {
            for k in 0..bits_per_asset {
                let weight = (1 << k) as f64 / ((1 << bits_per_asset) - 1) as f64;
                let var_idx = i * bits_per_asset + k;
                qubo[[var_idx, var_idx]] -=
                    2.0 * penalty * target_return * self.returns[i] * weight;
            }
        }

        Ok(())
    }

    /// Add risk constraint
    fn add_risk_constraint(
        &self,
        qubo: &mut Array2<f64>,
        bits_per_asset: usize,
        max_risk: f64,
        penalty: f64,
    ) -> Result<(), String> {
        // Penalize portfolios with risk > max_risk
        // This is challenging as it's a quartic term
        // Use linear approximation

        let risk_scale = penalty / max_risk;

        // Add scaled risk term
        let n_assets = self.returns.len();
        for i in 0..n_assets {
            for j in 0..n_assets {
                for k1 in 0..bits_per_asset {
                    for k2 in 0..bits_per_asset {
                        let w1 = (1 << k1) as f64 / ((1 << bits_per_asset) - 1) as f64;
                        let w2 = (1 << k2) as f64 / ((1 << bits_per_asset) - 1) as f64;

                        let var_idx1 = i * bits_per_asset + k1;
                        let var_idx2 = j * bits_per_asset + k2;

                        qubo[[var_idx1, var_idx2]] +=
                            risk_scale * self.covariance[[i, j]] * w1 * w2;
                    }
                }
            }
        }

        Ok(())
    }

    /// Risk parity objective: equal risk contribution penalty
    /// Penalty = sum_{i < j} (RC_i - RC_j)^2, with RC_i discretised via binary weights.
    fn add_risk_parity_objective(
        &self,
        qubo: &mut Array2<f64>,
        bits_per_asset: usize,
    ) -> Result<(), String> {
        let n_assets = self.returns.len();

        // We expand (RC_i - RC_j)^2 = RC_i^2 - 2*RC_i*RC_j + RC_j^2.
        // RC_i = w_i * (sum_k cov[i][k] * w_k).
        // Expanding in binary variables w_i = sum_k b_k * scale_k gives quartic terms;
        // we approximate to quadratic by treating cov row sums as linear coefficients:
        // RC_i ≈ w_i * row_sum_i  →  penalty (RC_i - RC_j)^2 linearised in w-bits.

        // Precompute cov row sums as marginal risk approximations at uniform weights.
        let uniform = 1.0 / n_assets as f64;
        let mut marginal = vec![0.0f64; n_assets];
        for i in 0..n_assets {
            for j in 0..n_assets {
                marginal[i] += self.covariance[[i, j]] * uniform;
            }
        }

        // Penalty coefficient: for each pair (i, j), (RC_i - RC_j)^2
        // = (marginal_i * w_i - marginal_j * w_j)^2 (quadratic in w)
        // Expand with binary encoding.
        for i in 0..n_assets {
            for j in (i + 1)..n_assets {
                let mi = marginal[i];
                let mj = marginal[j];
                // (mi * w_i - mj * w_j)^2 = mi^2 wi^2 - 2*mi*mj*wi*wj + mj^2 wj^2
                // Diagonal terms for each asset-bit
                for k in 0..bits_per_asset {
                    let sc = (1 << k) as f64 / ((1 << bits_per_asset) - 1) as f64;
                    let vi = i * bits_per_asset + k;
                    let vj = j * bits_per_asset + k;
                    qubo[[vi, vi]] += mi * mi * sc * sc;
                    qubo[[vj, vj]] += mj * mj * sc * sc;
                }
                // Cross terms: -2 * mi * mj * wi_{k1} * wj_{k2}
                // Must be added symmetrically to (vi, vj) AND (vj, vi).
                for k1 in 0..bits_per_asset {
                    let sc1 = (1 << k1) as f64 / ((1 << bits_per_asset) - 1) as f64;
                    let vi = i * bits_per_asset + k1;
                    for k2 in 0..bits_per_asset {
                        let sc2 = (1 << k2) as f64 / ((1 << bits_per_asset) - 1) as f64;
                        let vj = j * bits_per_asset + k2;
                        let coef = -2.0 * mi * mj * sc1 * sc2;
                        qubo[[vi, vj]] += coef / 2.0;
                        qubo[[vj, vi]] += coef / 2.0;
                    }
                }
            }
        }

        Ok(())
    }

    /// Black-Litterman objective: build posterior mean then apply mean-variance.
    fn add_black_litterman_objective(
        &self,
        qubo: &mut Array2<f64>,
        bits_per_asset: usize,
        views: &MarketViews,
    ) -> Result<(), String> {
        let n_assets = self.returns.len();
        let tau = 0.05f64;

        // Compute posterior mu using Black-Litterman closed form.
        // mu_post = pi + tau*Sigma*P^T * (P*tau*Sigma*P^T + Omega)^{-1} * (q - P*pi)
        // where pi = self.returns, P = view_matrix, Omega = view_confidence.
        let posterior = if views.view_matrix.shape()[0] == 0 {
            self.returns.clone()
        } else {
            let p = &views.view_matrix;
            let q_vec = &views.view_expectations;
            let omega = &views.view_confidence;
            let k = p.shape()[0]; // number of views

            // tau_sigma = tau * Sigma  (n x n)
            let tau_sigma = tau * &self.covariance;

            // M = P * tau_sigma * P^T + Omega  (k x k)
            let mut m = Array2::<f64>::zeros((k, k));
            for row in 0..k {
                for col in 0..k {
                    let mut val = 0.0f64;
                    for a in 0..n_assets {
                        for b in 0..n_assets {
                            val += p[[row, a]] * tau_sigma[[a, b]] * p[[col, b]];
                        }
                    }
                    m[[row, col]] = val + omega[[row, col]];
                }
            }

            // Invert M via Gauss-Jordan (inline, no new deps).
            let m_inv = gauss_jordan_inverse(&m, k).unwrap_or_else(|| {
                // Fallback: use identity (effectively ignores views)
                Array2::<f64>::eye(k)
            });

            // diff = q - P * pi  (k-vector)
            let mut diff = vec![0.0f64; k];
            for v in 0..k {
                diff[v] = q_vec[v];
                for a in 0..n_assets {
                    diff[v] -= p[[v, a]] * self.returns[a];
                }
            }

            // adj = tau_sigma * P^T * m_inv * diff  (n-vector)
            // Step 1: tmp = m_inv * diff  (k-vector)
            let mut tmp = vec![0.0f64; k];
            for i in 0..k {
                for j in 0..k {
                    tmp[i] += m_inv[[i, j]] * diff[j];
                }
            }
            // Step 2: adj = tau_sigma * P^T * tmp  (n-vector)
            let mut adj = vec![0.0f64; n_assets];
            for a in 0..n_assets {
                for b in 0..n_assets {
                    let mut pt_tmp = 0.0f64;
                    for v in 0..k {
                        pt_tmp += p[[v, b]] * tmp[v];
                    }
                    adj[a] += tau_sigma[[a, b]] * pt_tmp;
                }
            }

            let mut posterior_mu = self.returns.clone();
            for a in 0..n_assets {
                posterior_mu[a] += adj[a];
            }
            posterior_mu
        };

        // Build mean-variance QUBO with posterior returns.
        let n_assets = posterior.len();
        for i in 0..n_assets {
            for k in 0..bits_per_asset {
                let weight = (1 << k) as f64 / ((1 << bits_per_asset) - 1) as f64;
                let var_idx = i * bits_per_asset + k;
                qubo[[var_idx, var_idx]] -= posterior[i] * weight;
            }
        }
        for i in 0..n_assets {
            for j in 0..n_assets {
                for k1 in 0..bits_per_asset {
                    for k2 in 0..bits_per_asset {
                        let w1 = (1 << k1) as f64 / ((1 << bits_per_asset) - 1) as f64;
                        let w2 = (1 << k2) as f64 / ((1 << bits_per_asset) - 1) as f64;
                        let vi = i * bits_per_asset + k1;
                        let vj = j * bits_per_asset + k2;
                        if vi <= vj {
                            let coef = self.risk_aversion * self.covariance[[i, j]] * w1 * w2;
                            if vi == vj {
                                qubo[[vi, vi]] += coef;
                            } else {
                                qubo[[vi, vj]] += coef / 2.0;
                                qubo[[vj, vi]] += coef / 2.0;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Kelly criterion objective: maximize sum_i w_i*mu_i - 0.5 * sum_{i,j} w_i*w_j*cov[i][j]
    fn add_kelly_criterion_objective(
        &self,
        qubo: &mut Array2<f64>,
        bits_per_asset: usize,
    ) -> Result<(), String> {
        let n_assets = self.returns.len();

        // Linear term: -mu_i * w_i  (negate because we minimise)
        for i in 0..n_assets {
            for k in 0..bits_per_asset {
                let weight = (1 << k) as f64 / ((1 << bits_per_asset) - 1) as f64;
                let vi = i * bits_per_asset + k;
                qubo[[vi, vi]] -= self.returns[i] * weight;
            }
        }

        // Quadratic term: +0.5 * cov[i][j] * w_i * w_j
        for i in 0..n_assets {
            for j in 0..n_assets {
                for k1 in 0..bits_per_asset {
                    for k2 in 0..bits_per_asset {
                        let w1 = (1 << k1) as f64 / ((1 << bits_per_asset) - 1) as f64;
                        let w2 = (1 << k2) as f64 / ((1 << bits_per_asset) - 1) as f64;
                        let vi = i * bits_per_asset + k1;
                        let vj = j * bits_per_asset + k2;
                        if vi <= vj {
                            let coef = 0.5 * self.covariance[[i, j]] * w1 * w2;
                            if vi == vj {
                                qubo[[vi, vi]] += coef;
                            } else {
                                qubo[[vi, vj]] += coef / 2.0;
                                qubo[[vj, vi]] += coef / 2.0;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Decode solution to portfolio weights
    pub fn decode_solution(
        &self,
        solution: &HashMap<String, bool>,
        bits_per_asset: usize,
    ) -> Array1<f64> {
        let n_assets = self.returns.len();
        let mut weights = Array1::zeros(n_assets);

        for i in 0..n_assets {
            let mut weight = 0.0;
            for j in 0..bits_per_asset {
                let var_name = format!("asset_{i}_bit_{j}");
                if *solution.get(&var_name).unwrap_or(&false) {
                    weight += (1 << j) as f64 / ((1 << bits_per_asset) - 1) as f64;
                }
            }
            weights[i] = weight;
        }

        // Normalize to sum to 1
        let sum: f64 = weights.sum();
        if sum > 0.0 {
            weights /= sum;
        }

        weights
    }

    /// Calculate portfolio metrics
    pub fn calculate_metrics(&self, weights: &Array1<f64>) -> PortfolioMetrics {
        let expected_return = weights.dot(&self.returns);
        let variance = weights.dot(&self.covariance.dot(weights));
        let volatility = variance.sqrt();

        let sharpe_ratio = if volatility > 0.0 {
            expected_return / volatility
        } else {
            0.0
        };

        // Calculate diversification ratio
        let weighted_vol: f64 = weights
            .iter()
            .zip(self.covariance.diag().iter())
            .map(|(&w, &var)| w * var.sqrt())
            .sum();

        let diversification_ratio = if volatility > 0.0 {
            weighted_vol / volatility
        } else {
            1.0
        };

        PortfolioMetrics {
            expected_return,
            volatility,
            sharpe_ratio,
            diversification_ratio,
            max_drawdown: self.estimate_max_drawdown(weights),
            value_at_risk: self.calculate_var(weights, 0.95),
            conditional_value_at_risk: self.calculate_cvar(weights, 0.95),
        }
    }

    /// Estimate maximum drawdown
    fn estimate_max_drawdown(&self, weights: &Array1<f64>) -> f64 {
        // Simplified estimation based on volatility
        let variance = weights.dot(&self.covariance.dot(weights));
        let volatility = variance.sqrt();

        // Rule of thumb: max drawdown ≈ 2 * annual volatility
        2.0 * volatility
    }

    /// Calculate Value at Risk
    fn calculate_var(&self, weights: &Array1<f64>, _confidence: f64) -> f64 {
        let expected_return = weights.dot(&self.returns);
        let variance = weights.dot(&self.covariance.dot(weights));
        let volatility = variance.sqrt();

        // Parametric VaR assuming normal distribution
        let z_score = 1.645; // 95% confidence
        expected_return - z_score * volatility
    }

    /// Calculate Conditional Value at Risk
    fn calculate_cvar(&self, weights: &Array1<f64>, confidence: f64) -> f64 {
        let var = self.calculate_var(weights, confidence);
        let volatility = weights.dot(&self.covariance.dot(weights)).sqrt();

        // Approximation for normal distribution
        let z_score = 1.645;
        let phi = (-z_score * z_score / 2.0_f64).exp() / (2.0_f64 * PI).sqrt();

        var - volatility * phi / (1.0 - confidence)
    }
}

#[derive(Debug, Clone)]
pub struct PortfolioMetrics {
    pub expected_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub diversification_ratio: f64,
    pub max_drawdown: f64,
    pub value_at_risk: f64,
    pub conditional_value_at_risk: f64,
}

/// Risk parity portfolio optimizer
pub struct RiskParityOptimizer {
    /// Covariance matrix
    covariance: Array2<f64>,
    /// Target risk contributions
    target_contributions: Option<Array1<f64>>,
    /// Convergence tolerance
    tolerance: f64,
    /// Maximum iterations
    max_iterations: usize,
}

impl RiskParityOptimizer {
    /// Create new risk parity optimizer
    pub const fn new(covariance: Array2<f64>) -> Self {
        Self {
            covariance,
            target_contributions: None,
            tolerance: 1e-6,
            max_iterations: 1000,
        }
    }

    /// Set target risk contributions
    pub fn with_target_contributions(mut self, targets: Array1<f64>) -> Self {
        self.target_contributions = Some(targets);
        self
    }

    /// Build QUBO for risk parity
    pub fn build_qubo(
        &self,
        num_bits_per_asset: usize,
    ) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        let n_assets = self.covariance.shape()[0];
        let total_vars = n_assets * num_bits_per_asset;

        let mut qubo = Array2::zeros((total_vars, total_vars));
        let mut var_map = HashMap::new();

        // Create variable mapping
        for i in 0..n_assets {
            for j in 0..num_bits_per_asset {
                let var_name = format!("rp_asset_{i}_bit_{j}");
                var_map.insert(var_name, i * num_bits_per_asset + j);
            }
        }

        // Risk parity objective: minimize sum_i (RC_i - target_i)^2
        // where RC_i = w_i * (Σw)_i / w'Σw is the risk contribution

        // This is a complex non-linear objective, we use iterative approximation
        let initial_weights = Array1::from_elem(n_assets, 1.0 / n_assets as f64);
        let _risk_contribs = self.calculate_risk_contributions(&initial_weights);

        let default_targets = Array1::from_elem(n_assets, 1.0 / n_assets as f64);
        let _targets = self
            .target_contributions
            .as_ref()
            .unwrap_or(&default_targets);

        // Linearize around current weights
        for i in 0..n_assets {
            for k in 0..num_bits_per_asset {
                let weight = (1 << k) as f64 / ((1 << num_bits_per_asset) - 1) as f64;
                let var_idx = i * num_bits_per_asset + k;

                // Gradient of risk contribution
                let gradient = self.risk_contribution_gradient(&initial_weights, i);

                qubo[[var_idx, var_idx]] += gradient * weight;
            }
        }

        // Add budget constraint
        self.add_budget_constraint_rp(&mut qubo, num_bits_per_asset, 100.0)?;

        Ok((qubo, var_map))
    }

    /// Calculate risk contributions
    fn calculate_risk_contributions(&self, weights: &Array1<f64>) -> Array1<f64> {
        let portfolio_risk = weights.dot(&self.covariance.dot(weights)).sqrt();
        let marginal_risks = self.covariance.dot(weights);

        weights * &marginal_risks / portfolio_risk
    }

    /// Risk contribution gradient
    fn risk_contribution_gradient(&self, weights: &Array1<f64>, asset: usize) -> f64 {
        // Numerical gradient (simplified)
        let eps = 1e-6;
        let mut weights_plus = weights.clone();
        weights_plus[asset] += eps;

        let rc_plus = self.calculate_risk_contributions(&weights_plus)[asset];
        let rc = self.calculate_risk_contributions(weights)[asset];

        (rc_plus - rc) / eps
    }

    /// Add budget constraint for risk parity
    fn add_budget_constraint_rp(
        &self,
        qubo: &mut Array2<f64>,
        bits_per_asset: usize,
        penalty: f64,
    ) -> Result<(), String> {
        let n_assets = self.covariance.shape()[0];

        for i in 0..n_assets {
            for j in 0..n_assets {
                for k1 in 0..bits_per_asset {
                    for k2 in 0..bits_per_asset {
                        let w1 = (1 << k1) as f64 / ((1 << bits_per_asset) - 1) as f64;
                        let w2 = (1 << k2) as f64 / ((1 << bits_per_asset) - 1) as f64;

                        let var_idx1 = i * bits_per_asset + k1;
                        let var_idx2 = j * bits_per_asset + k2;

                        qubo[[var_idx1, var_idx2]] += penalty * w1 * w2;
                    }
                }
            }
        }

        Ok(())
    }
}

/// Black-Litterman portfolio optimizer
pub struct BlackLittermanOptimizer {
    /// Market equilibrium returns
    equilibrium_returns: Array1<f64>,
    /// Covariance matrix
    covariance: Array2<f64>,
    /// Risk aversion
    risk_aversion: f64,
    /// Confidence in equilibrium
    tau: f64,
    /// Market views
    views: MarketViews,
}

impl BlackLittermanOptimizer {
    /// Create new Black-Litterman optimizer
    pub fn new(market_weights: Array1<f64>, covariance: Array2<f64>, risk_aversion: f64) -> Self {
        // Calculate equilibrium returns
        let equilibrium_returns = risk_aversion * covariance.dot(&market_weights);

        Self {
            equilibrium_returns,
            covariance,
            risk_aversion,
            tau: 0.05,
            views: MarketViews {
                view_matrix: Array2::zeros((0, 0)),
                view_expectations: Array1::zeros(0),
                view_confidence: Array2::zeros((0, 0)),
            },
        }
    }

    /// Set market views
    pub fn with_views(mut self, views: MarketViews) -> Self {
        self.views = views;
        self
    }

    /// Calculate posterior returns
    pub fn posterior_returns(&self) -> Array1<f64> {
        let _tau_sigma = self.tau * &self.covariance;

        if self.views.view_matrix.shape()[0] == 0 {
            // No views, return equilibrium
            return self.equilibrium_returns.clone();
        }

        // Black-Litterman formula
        let p = &self.views.view_matrix;
        let q = &self.views.view_expectations;
        let omega = &self.views.view_confidence;

        // Calculate posterior mean
        // μ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) [(τΣ)^(-1)μ_eq + P'Ω^(-1)q]

        // Simplified implementation
        let mut posterior = self.equilibrium_returns.clone();

        // Adjust based on views
        for i in 0..p.shape()[0] {
            let view_assets: Vec<_> = (0..p.shape()[1])
                .filter(|&j| p[[i, j]].abs() > 1e-10)
                .collect();

            if !view_assets.is_empty() {
                let view_return = q[i];
                let confidence = 1.0 / omega[[i, i]];

                for &asset in &view_assets {
                    posterior[asset] +=
                        confidence * (view_return - posterior[asset]) * p[[i, asset]];
                }
            }
        }

        posterior
    }

    /// Build QUBO with Black-Litterman returns
    pub fn build_qubo(
        &self,
        num_bits_per_asset: usize,
    ) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        let posterior_returns = self.posterior_returns();

        let optimizer = PortfolioOptimizer::new(
            posterior_returns,
            self.covariance.clone(),
            self.risk_aversion,
        )?;

        optimizer.build_qubo(num_bits_per_asset)
    }
}

/// Transaction cost optimizer
pub struct TransactionCostOptimizer {
    /// Current portfolio
    current_weights: Array1<f64>,
    /// Target optimizer
    target_optimizer: PortfolioOptimizer,
    /// Transaction cost model
    cost_model: TransactionCostModel,
    /// Rebalancing threshold
    threshold: f64,
}

#[derive(Debug, Clone)]
pub enum TransactionCostModel {
    /// Linear costs
    Linear { rates: Array1<f64> },
    /// Quadratic costs (market impact)
    Quadratic { impact_coefficients: Array1<f64> },
    /// Fixed + proportional
    FixedPlusProportional {
        fixed: f64,
        proportional: Array1<f64>,
    },
    /// Custom function
    Custom,
}

impl TransactionCostOptimizer {
    /// Create new transaction cost optimizer
    pub const fn new(
        current_weights: Array1<f64>,
        target_optimizer: PortfolioOptimizer,
        cost_model: TransactionCostModel,
    ) -> Self {
        Self {
            current_weights,
            target_optimizer,
            cost_model,
            threshold: 0.01,
        }
    }

    /// Build QUBO with transaction costs
    pub fn build_qubo(
        &self,
        num_bits_per_asset: usize,
    ) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        // Get base QUBO from target optimizer
        let (mut qubo, var_map) = self.target_optimizer.build_qubo(num_bits_per_asset)?;

        // Add transaction cost terms
        self.add_transaction_costs(&mut qubo, &var_map, num_bits_per_asset)?;

        Ok((qubo, var_map))
    }

    /// Add transaction cost terms
    fn add_transaction_costs(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        bits_per_asset: usize,
    ) -> Result<(), String> {
        let n_assets = self.current_weights.len();

        if let TransactionCostModel::Linear { rates } = &self.cost_model {
            // Cost = sum_i rate_i * |w_i - w_i^current|
            // Approximate with quadratic penalty

            for i in 0..n_assets {
                let current_w = self.current_weights[i];
                let rate = rates[i];

                for k in 0..bits_per_asset {
                    let weight = (1 << k) as f64 / ((1 << bits_per_asset) - 1) as f64;
                    let var_name = format!("asset_{i}_bit_{k}");

                    if let Some(&var_idx) = var_map.get(&var_name) {
                        // Penalize deviation from current weight
                        qubo[[var_idx, var_idx]] += rate * (weight - current_w).powi(2);
                    }
                }
            }
        } else {
            // Other cost models
        }

        Ok(())
    }
}

/// Portfolio rebalancing scheduler
pub struct RebalancingScheduler {
    /// Rebalancing frequency
    frequency: RebalancingFrequency,
    /// Trigger conditions
    triggers: Vec<RebalancingTrigger>,
    /// Cost threshold
    cost_threshold: f64,
}

#[derive(Debug, Clone)]
pub enum RebalancingFrequency {
    /// Fixed period
    Fixed { days: usize },
    /// Calendar-based
    Calendar { schedule: String },
    /// Adaptive
    Adaptive,
}

#[derive(Debug, Clone)]
pub enum RebalancingTrigger {
    /// Deviation from target
    Deviation { threshold: f64 },
    /// Volatility spike
    VolatilitySpike { threshold: f64 },
    /// Correlation breakdown
    CorrelationBreakdown { threshold: f64 },
    /// Custom trigger
    Custom { name: String },
}

impl RebalancingScheduler {
    /// Create new rebalancing scheduler
    pub const fn new(frequency: RebalancingFrequency) -> Self {
        Self {
            frequency,
            triggers: Vec::new(),
            cost_threshold: 0.001,
        }
    }

    /// Add rebalancing trigger
    pub fn add_trigger(mut self, trigger: RebalancingTrigger) -> Self {
        self.triggers.push(trigger);
        self
    }

    /// Check if rebalancing is needed
    pub fn should_rebalance(
        &self,
        current_weights: &Array1<f64>,
        target_weights: &Array1<f64>,
        market_data: &MarketData,
    ) -> bool {
        // Check triggers
        for trigger in &self.triggers {
            match trigger {
                RebalancingTrigger::Deviation { threshold } => {
                    let max_deviation = current_weights
                        .iter()
                        .zip(target_weights.iter())
                        .map(|(&c, &t)| (c - t).abs())
                        .fold(0.0, f64::max);

                    if max_deviation > *threshold {
                        return true;
                    }
                }
                RebalancingTrigger::VolatilitySpike { threshold } => {
                    if market_data.volatility > *threshold {
                        return true;
                    }
                }
                _ => {}
            }
        }

        false
    }
}

#[derive(Debug, Clone)]
pub struct MarketData {
    pub returns: Array1<f64>,
    pub volatility: f64,
    pub correlation: Array2<f64>,
    pub volume: Array1<f64>,
}

/// Gauss-Jordan matrix inverse for a k×k matrix.
/// Returns `None` if the matrix is singular.
fn gauss_jordan_inverse(m: &Array2<f64>, k: usize) -> Option<Array2<f64>> {
    // Build augmented matrix [M | I]
    let mut aug = vec![0.0f64; k * 2 * k];
    for row in 0..k {
        for col in 0..k {
            aug[row * 2 * k + col] = m[[row, col]];
        }
        aug[row * 2 * k + k + row] = 1.0;
    }

    for col in 0..k {
        // Find pivot
        let mut pivot_row = None;
        let mut max_val = 0.0f64;
        for row in col..k {
            let v = aug[row * 2 * k + col].abs();
            if v > max_val {
                max_val = v;
                pivot_row = Some(row);
            }
        }
        let pivot_row = pivot_row?;
        if aug[pivot_row * 2 * k + col].abs() < 1e-12 {
            return None;
        }
        // Swap rows
        if pivot_row != col {
            for j in 0..2 * k {
                aug.swap(col * 2 * k + j, pivot_row * 2 * k + j);
            }
        }
        // Scale pivot row
        let scale = aug[col * 2 * k + col];
        for j in 0..2 * k {
            aug[col * 2 * k + j] /= scale;
        }
        // Eliminate column entries in other rows
        for row in 0..k {
            if row != col {
                let factor = aug[row * 2 * k + col];
                for j in 0..2 * k {
                    let sub = factor * aug[col * 2 * k + j];
                    aug[row * 2 * k + j] -= sub;
                }
            }
        }
    }

    // Extract right half as inverse
    let mut inv = Array2::<f64>::zeros((k, k));
    for row in 0..k {
        for col in 0..k {
            inv[[row, col]] = aug[row * 2 * k + k + col];
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_optimizer() {
        let returns = Array1::from_vec(vec![0.10, 0.12, 0.08]);
        let covariance = Array2::from_shape_vec(
            (3, 3),
            vec![0.01, 0.002, 0.001, 0.002, 0.015, 0.002, 0.001, 0.002, 0.008],
        )
        .expect("Failed to create covariance matrix from shape vector");

        let optimizer = PortfolioOptimizer::new(returns, covariance, 2.0)
            .expect("Failed to create portfolio optimizer");
        let (qubo, var_map) = optimizer.build_qubo(3).expect("Failed to build QUBO");

        assert_eq!(var_map.len(), 9); // 3 assets * 3 bits
    }

    #[test]
    fn test_portfolio_metrics() {
        let returns = Array1::from_vec(vec![0.10, 0.12, 0.08]);
        let covariance = Array2::eye(3) * 0.01;

        let optimizer = PortfolioOptimizer::new(returns, covariance, 2.0)
            .expect("Failed to create portfolio optimizer");
        let mut weights = Array1::from_vec(vec![0.3, 0.5, 0.2]);

        let mut metrics = optimizer.calculate_metrics(&weights);

        assert!((metrics.expected_return - 0.106).abs() < 0.001);
        assert!(metrics.volatility > 0.0);
        assert!(metrics.sharpe_ratio > 0.0);
    }

    #[test]
    fn test_risk_parity() {
        let covariance = Array2::from_shape_vec(
            (3, 3),
            vec![0.01, 0.002, 0.001, 0.002, 0.015, 0.002, 0.001, 0.002, 0.008],
        )
        .expect("Failed to create covariance matrix from shape vector");

        let rp_optimizer = RiskParityOptimizer::new(covariance);
        let (qubo, var_map) = rp_optimizer.build_qubo(3).expect("Failed to build QUBO");

        assert!(!var_map.is_empty());
    }

    fn make_3asset_optimizer(method: OptimizationMethod) -> PortfolioOptimizer {
        let returns = Array1::from_vec(vec![0.10, 0.12, 0.08]);
        let covariance = Array2::from_shape_vec(
            (3, 3),
            vec![0.01, 0.002, 0.001, 0.002, 0.015, 0.002, 0.001, 0.002, 0.008],
        )
        .expect("Failed to build covariance matrix");
        PortfolioOptimizer::new(returns, covariance, 2.0)
            .expect("Failed to build optimizer")
            .with_method(method)
    }

    #[test]
    fn test_risk_parity_qubo_shape() {
        let opt = make_3asset_optimizer(OptimizationMethod::RiskParity);
        let (qubo, var_map) = opt
            .build_qubo(3)
            .expect("risk parity build_qubo should succeed");
        assert!(!var_map.is_empty(), "var_map should be non-empty");
        assert!(!qubo.iter().all(|&v| v == 0.0), "QUBO should be non-zero");
        // Symmetry check on off-diagonal blocks
        let n = qubo.shape()[0];
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (qubo[[i, j]] - qubo[[j, i]]).abs() < 1e-10,
                    "QUBO should be symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_black_litterman_empty_views() {
        let views = MarketViews {
            view_matrix: Array2::zeros((0, 3)),
            view_expectations: scirs2_core::ndarray::Array1::zeros(0),
            view_confidence: Array2::zeros((0, 0)),
        };
        let opt = make_3asset_optimizer(OptimizationMethod::BlackLitterman { views });
        let result = opt.build_qubo(3);
        assert!(
            result.is_ok(),
            "black-litterman with empty views should succeed"
        );
        let (qubo, var_map) = result.expect("expected Ok");
        assert!(!var_map.is_empty());
        assert_eq!(qubo.shape(), [9, 9]);
    }

    #[test]
    fn test_kelly_criterion_qubo() {
        let opt = make_3asset_optimizer(OptimizationMethod::KellyCriterion);
        let (qubo, var_map) = opt
            .build_qubo(3)
            .expect("kelly criterion build_qubo should succeed");
        assert!(!var_map.is_empty());
        assert!(!qubo.iter().all(|&v| v == 0.0), "QUBO should be non-zero");
    }
}
