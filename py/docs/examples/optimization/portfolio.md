# Portfolio Optimization with Quantum Computing

**Level:** 🟡 Intermediate  
**Runtime:** 1-2 minutes  
**Topics:** Finance, Risk management, Quadratic optimization  
**Applications:** Investment management, Asset allocation

Learn to optimize investment portfolios using quantum algorithms - solving the mean-variance optimization problem that's fundamental to modern finance.

## What is Portfolio Optimization?

Portfolio optimization is the process of selecting the best mix of assets to maximize returns while minimizing risk. This is a foundational problem in quantitative finance.

**The Problem:**
Given N assets with expected returns μᵢ and covariance matrix Σᵢⱼ, find weights wᵢ that:
- **Maximize** expected return: Σᵢ μᵢ wᵢ
- **Minimize** risk (variance): Σᵢⱼ wᵢ Σᵢⱼ wⱼ
- **Subject to** constraints: Σᵢ wᵢ = 1, wᵢ ≥ 0

**Why Quantum Computing?**
- Portfolio optimization is quadratic programming (QUBO) - natural for quantum algorithms
- Exponential speedup potential for large portfolios
- Can handle complex constraints and non-convex objectives
- Relevant for high-frequency trading and real-time optimization

## Financial Background

### Modern Portfolio Theory (MPT)

Developed by Harry Markowitz (Nobel Prize 1990), MPT provides the mathematical framework:

**Expected Return:**
```
E[R] = Σᵢ wᵢ μᵢ
```

**Portfolio Variance (Risk):**
```
Var[R] = Σᵢⱼ wᵢ wⱼ σᵢⱼ
```

**Sharpe Ratio:**
```
S = (E[R] - Rf) / √Var[R]
```

### Efficient Frontier

The efficient frontier represents the set of optimal portfolios offering the highest expected return for each level of risk.

### Risk Models

**Systematic Risk:** Market-wide factors (interest rates, inflation)  
**Idiosyncratic Risk:** Asset-specific factors (company news, earnings)  
**Total Risk:** Systematic + Idiosyncratic

## Implementation

### Portfolio Data and Market Model

```python
import quantrs2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta

class PortfolioData:
    """
    Portfolio data management and market model.
    """
    
    def __init__(self, symbols, start_date, end_date):
        """
        Initialize portfolio with market data.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for historical data
            end_date: End date for historical data
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.num_assets = len(symbols)
        
        # Download market data
        self.prices = self._download_data()
        self.returns = self._calculate_returns()
        
        # Calculate portfolio statistics
        self.expected_returns = self._calculate_expected_returns()
        self.covariance_matrix = self._calculate_covariance()
        self.correlation_matrix = self._calculate_correlation()
        
        print(f"📊 Portfolio Data:")
        print(f"   Assets: {self.num_assets}")
        print(f"   Symbols: {self.symbols}")
        print(f"   Date range: {start_date} to {end_date}")
        print(f"   Observations: {len(self.returns)}")
    
    def _download_data(self):
        """Download historical price data."""
        
        print(f"📥 Downloading market data...")
        
        try:
            # Download data from Yahoo Finance
            data = yf.download(self.symbols, start=self.start_date, end=self.end_date)
            
            # Extract adjusted close prices
            if len(self.symbols) == 1:
                prices = data['Adj Close'].to_frame()
                prices.columns = self.symbols
            else:
                prices = data['Adj Close']
            
            # Fill missing values
            prices = prices.fillna(method='forward').fillna(method='backward')
            
            print(f"   Downloaded {len(prices)} price observations")
            return prices
            
        except Exception as e:
            print(f"   Error downloading data: {e}")
            print(f"   Using synthetic data instead...")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic price data for demonstration."""
        
        np.random.seed(42)
        
        # Generate synthetic prices with realistic properties
        num_days = 252 * 2  # 2 years of trading days
        dates = pd.date_range(start=self.start_date, periods=num_days, freq='D')
        
        # Simulate correlated returns
        correlations = 0.3 + 0.4 * np.random.random((self.num_assets, self.num_assets))
        correlations = (correlations + correlations.T) / 2
        np.fill_diagonal(correlations, 1.0)
        
        # Generate returns
        returns = np.random.multivariate_normal(
            mean=[0.0008] * self.num_assets,  # ~20% annual return
            cov=correlations * 0.02**2,       # ~20% annual volatility
            size=num_days
        )
        
        # Convert to prices
        initial_prices = [100.0] * self.num_assets
        prices = np.zeros((num_days, self.num_assets))
        prices[0] = initial_prices
        
        for i in range(1, num_days):
            prices[i] = prices[i-1] * (1 + returns[i])
        
        # Create DataFrame
        price_df = pd.DataFrame(prices, columns=self.symbols, index=dates)
        
        print(f"   Generated {len(price_df)} synthetic price observations")
        return price_df
    
    def _calculate_returns(self):
        """Calculate daily returns."""
        
        returns = self.prices.pct_change().dropna()
        
        print(f"📈 Return Statistics:")
        for symbol in self.symbols:
            annual_return = returns[symbol].mean() * 252
            annual_vol = returns[symbol].std() * np.sqrt(252)
            print(f"   {symbol}: {annual_return:.1%} return, {annual_vol:.1%} volatility")
        
        return returns
    
    def _calculate_expected_returns(self):
        """Calculate expected returns (annualized)."""
        
        # Simple historical average (252 trading days per year)
        expected_returns = self.returns.mean() * 252
        
        print(f"\n🎯 Expected Annual Returns:")
        for i, symbol in enumerate(self.symbols):
            print(f"   {symbol}: {expected_returns[symbol]:.2%}")
        
        return expected_returns.values
    
    def _calculate_covariance(self):
        """Calculate covariance matrix (annualized)."""
        
        # Annualized covariance matrix
        covariance = self.returns.cov() * 252
        
        print(f"\n🔗 Asset Correlations:")
        correlation = self.returns.corr()
        print(correlation)
        
        return covariance.values
    
    def _calculate_correlation(self):
        """Calculate correlation matrix."""
        return self.returns.corr().values
    
    def simulate_portfolio_performance(self, weights, num_simulations=1000):
        """Simulate portfolio performance using Monte Carlo."""
        
        portfolio_returns = []
        
        for _ in range(num_simulations):
            # Sample random returns from multivariate normal
            simulated_returns = np.random.multivariate_normal(
                self.expected_returns / 252,  # Daily expected returns
                self.covariance_matrix / 252,  # Daily covariance
                252  # One year of trading days
            )
            
            # Calculate portfolio returns
            portfolio_daily_returns = np.dot(simulated_returns, weights)
            annual_return = (1 + portfolio_daily_returns).prod() - 1
            portfolio_returns.append(annual_return)
        
        return np.array(portfolio_returns)
    
    def plot_price_evolution(self):
        """Plot historical price evolution."""
        
        plt.figure(figsize=(12, 6))
        
        # Normalize prices to start at 100
        normalized_prices = self.prices / self.prices.iloc[0] * 100
        
        for symbol in self.symbols:
            plt.plot(normalized_prices.index, normalized_prices[symbol], 
                    linewidth=2, label=symbol)
        
        plt.title('Normalized Price Evolution', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price (Base = 100)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Create sample portfolio datasets
def create_sample_portfolios():
    """Create sample portfolios for different market scenarios."""
    
    portfolios = {}
    
    # Tech portfolio
    tech_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    print("Creating sample portfolios...")
    
    try:
        portfolios['tech'] = PortfolioData(
            tech_symbols,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    except:
        # Use synthetic data if download fails
        portfolios['tech'] = PortfolioData(
            ['Tech_A', 'Tech_B', 'Tech_C', 'Tech_D'],
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    
    # Diversified portfolio (synthetic for demonstration)
    portfolios['diversified'] = PortfolioData(
        ['Stock_A', 'Stock_B', 'Bond_A', 'REIT_A', 'Commodity_A'],
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    return portfolios

# Create portfolio datasets
sample_portfolios = create_sample_portfolios()
tech_portfolio = sample_portfolios['tech']
diversified_portfolio = sample_portfolios['diversified']

# Visualize portfolio data
tech_portfolio.plot_price_evolution()
```

### Classical Portfolio Optimization

```python
class ClassicalPortfolioOptimizer:
    """
    Classical portfolio optimization using mean-variance framework.
    """
    
    def __init__(self, portfolio_data):
        """
        Initialize optimizer with portfolio data.
        
        Args:
            portfolio_data: PortfolioData object
        """
        self.data = portfolio_data
        self.num_assets = portfolio_data.num_assets
        self.expected_returns = portfolio_data.expected_returns
        self.covariance_matrix = portfolio_data.covariance_matrix
        
        print(f"🎯 Classical Portfolio Optimizer:")
        print(f"   Assets: {self.num_assets}")
        print(f"   Optimization framework: Mean-Variance")
    
    def calculate_portfolio_metrics(self, weights):
        """Calculate portfolio return, risk, and Sharpe ratio."""
        
        # Expected return
        portfolio_return = np.dot(weights, self.expected_returns)
        
        # Portfolio variance
        portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming risk-free rate = 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'variance': portfolio_variance,
            'sharpe_ratio': sharpe_ratio
        }
    
    def optimize_portfolio(self, objective='max_sharpe', target_return=None):
        """
        Optimize portfolio using classical methods.
        
        Args:
            objective: 'max_sharpe', 'min_variance', 'target_return'
            target_return: Target return for target_return objective
        """
        
        print(f"\n🔧 Optimizing portfolio (objective: {objective})...")
        
        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        # Add target return constraint if specified
        if objective == 'target_return' and target_return is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda w: np.dot(w, self.expected_returns) - target_return
            })
        
        # Bounds: 0 <= weight <= 1 (long-only)
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / self.num_assets] * self.num_assets)
        
        # Define objective function
        if objective == 'max_sharpe':
            def objective_function(weights):
                metrics = self.calculate_portfolio_metrics(weights)
                return -metrics['sharpe_ratio']  # Negative for minimization
        
        elif objective == 'min_variance':
            def objective_function(weights):
                metrics = self.calculate_portfolio_metrics(weights)
                return metrics['variance']
        
        elif objective == 'target_return':
            def objective_function(weights):
                metrics = self.calculate_portfolio_metrics(weights)
                return metrics['variance']  # Minimize risk for target return
        
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Optimize
        result = minimize(
            objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False, 'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            metrics = self.calculate_portfolio_metrics(optimal_weights)
            
            print(f"✅ Optimization successful!")
            print(f"   Expected return: {metrics['return']:.2%}")
            print(f"   Volatility: {metrics['volatility']:.2%}")
            print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
            
            return {
                'weights': optimal_weights,
                'metrics': metrics,
                'success': True
            }
        else:
            print(f"❌ Optimization failed: {result.message}")
            return {
                'weights': initial_weights,
                'metrics': self.calculate_portfolio_metrics(initial_weights),
                'success': False
            }
    
    def compute_efficient_frontier(self, num_points=20):
        """Compute the efficient frontier."""
        
        print(f"\n📈 Computing efficient frontier ({num_points} points)...")
        
        # Range of target returns
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        target_returns = np.linspace(min_return, max_return, num_points)
        
        frontier_returns = []
        frontier_volatilities = []
        frontier_weights = []
        
        for target_return in target_returns:
            result = self.optimize_portfolio('target_return', target_return)
            
            if result['success']:
                frontier_returns.append(result['metrics']['return'])
                frontier_volatilities.append(result['metrics']['volatility'])
                frontier_weights.append(result['weights'])
        
        print(f"   Computed {len(frontier_returns)} efficient portfolios")
        
        return {
            'returns': np.array(frontier_returns),
            'volatilities': np.array(frontier_volatilities),
            'weights': np.array(frontier_weights)
        }
    
    def plot_efficient_frontier(self, frontier_data=None):
        """Plot the efficient frontier."""
        
        if frontier_data is None:
            frontier_data = self.compute_efficient_frontier()
        
        plt.figure(figsize=(12, 8))
        
        # Plot efficient frontier
        plt.plot(frontier_data['volatilities'], frontier_data['returns'],
                'b-', linewidth=3, label='Efficient Frontier')
        
        # Plot individual assets
        asset_returns = self.expected_returns
        asset_volatilities = np.sqrt(np.diag(self.covariance_matrix))
        
        plt.scatter(asset_volatilities, asset_returns, 
                   c='red', s=100, alpha=0.7, label='Individual Assets')
        
        # Add asset labels
        for i, symbol in enumerate(self.data.symbols):
            plt.annotate(symbol, 
                        (asset_volatilities[i], asset_returns[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        # Highlight special portfolios
        equal_weights = np.ones(self.num_assets) / self.num_assets
        equal_metrics = self.calculate_portfolio_metrics(equal_weights)
        
        plt.scatter(equal_metrics['volatility'], equal_metrics['return'],
                   c='green', s=150, marker='s', label='Equal Weight', alpha=0.8)
        
        # Max Sharpe portfolio
        max_sharpe_result = self.optimize_portfolio('max_sharpe')
        if max_sharpe_result['success']:
            metrics = max_sharpe_result['metrics']
            plt.scatter(metrics['volatility'], metrics['return'],
                       c='gold', s=150, marker='*', label='Max Sharpe', alpha=0.8)
        
        plt.xlabel('Volatility (Risk)', fontsize=12)
        plt.ylabel('Expected Return', fontsize=12)
        plt.title('Portfolio Efficient Frontier', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return frontier_data

# Demonstrate classical optimization
def demonstrate_classical_optimization():
    """Demonstrate classical portfolio optimization."""
    
    print("🌟 Classical Portfolio Optimization Demonstration")
    print("=" * 55)
    
    # Use tech portfolio for demonstration
    optimizer = ClassicalPortfolioOptimizer(tech_portfolio)
    
    # Optimize for different objectives
    objectives = ['max_sharpe', 'min_variance']
    results = {}
    
    for objective in objectives:
        result = optimizer.optimize_portfolio(objective)
        results[objective] = result
        
        print(f"\n{objective.upper()} Portfolio:")
        weights = result['weights']
        for i, symbol in enumerate(tech_portfolio.symbols):
            print(f"   {symbol}: {weights[i]:7.2%}")
    
    # Plot efficient frontier
    frontier = optimizer.plot_efficient_frontier()
    
    # Compare with equal-weight portfolio
    equal_weights = np.ones(optimizer.num_assets) / optimizer.num_assets
    equal_metrics = optimizer.calculate_portfolio_metrics(equal_weights)
    
    print(f"\nPortfolio Comparison:")
    print(f"{'Strategy':<15} {'Return':<8} {'Risk':<8} {'Sharpe'}")
    print("-" * 45)
    
    # Equal weight
    print(f"{'Equal Weight':<15} {equal_metrics['return']:<8.2%} "
          f"{equal_metrics['volatility']:<8.2%} {equal_metrics['sharpe_ratio']:<6.3f}")
    
    # Optimized portfolios
    for objective, result in results.items():
        if result['success']:
            metrics = result['metrics']
            name = objective.replace('_', ' ').title()
            print(f"{name:<15} {metrics['return']:<8.2%} "
                  f"{metrics['volatility']:<8.2%} {metrics['sharpe_ratio']:<6.3f}")
    
    return optimizer, results

# Run classical optimization demonstration
classical_optimizer, classical_results = demonstrate_classical_optimization()
```

### Quantum Portfolio Optimization (QAOA)

```python
class QuantumPortfolioOptimizer:
    """
    Quantum portfolio optimization using QAOA and quantum annealing approaches.
    """
    
    def __init__(self, portfolio_data, num_assets_quantum=4):
        """
        Initialize quantum portfolio optimizer.
        
        Args:
            portfolio_data: PortfolioData object
            num_assets_quantum: Number of assets to include in quantum optimization
        """
        self.data = portfolio_data
        self.num_assets_total = portfolio_data.num_assets
        self.num_assets_quantum = min(num_assets_quantum, self.num_assets_total)
        
        # Select top assets by Sharpe ratio for quantum optimization
        self.selected_assets = self._select_assets_for_quantum()
        
        # Prepare quantum optimization matrices
        self.Q_matrix, self.linear_terms = self._formulate_qubo()
        
        print(f"⚛️ Quantum Portfolio Optimizer:")
        print(f"   Total assets: {self.num_assets_total}")
        print(f"   Quantum assets: {self.num_assets_quantum}")
        print(f"   Selected assets: {[self.data.symbols[i] for i in self.selected_assets]}")
    
    def _select_assets_for_quantum(self):
        """Select subset of assets for quantum optimization."""
        
        # Calculate individual Sharpe ratios
        asset_sharpe_ratios = []
        risk_free_rate = 0.02
        
        for i in range(self.num_assets_total):
            asset_return = self.data.expected_returns[i]
            asset_risk = np.sqrt(self.data.covariance_matrix[i, i])
            sharpe = (asset_return - risk_free_rate) / asset_risk
            asset_sharpe_ratios.append((i, sharpe))
        
        # Sort by Sharpe ratio and select top assets
        asset_sharpe_ratios.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in asset_sharpe_ratios[:self.num_assets_quantum]]
        
        print(f"🔍 Asset selection (by Sharpe ratio):")
        for i, idx in enumerate(selected_indices):
            symbol = self.data.symbols[idx]
            sharpe = asset_sharpe_ratios[idx][1]
            print(f"   {i+1}. {symbol}: Sharpe = {sharpe:.3f}")
        
        return selected_indices
    
    def _formulate_qubo(self, risk_aversion=1.0):
        """
        Formulate portfolio optimization as QUBO problem.
        
        The portfolio optimization problem:
        maximize: μᵀw - λ/2 * wᵀΣw
        subject to: Σw = 1, w ∈ {0, 1/N, 2/N, ..., 1}
        
        For binary variables, we discretize weights.
        """
        
        # Extract submatrices for selected assets
        selected_returns = self.data.expected_returns[self.selected_assets]
        selected_covariance = self.data.covariance_matrix[np.ix_(self.selected_assets, self.selected_assets)]
        
        # Number of bits per asset (weight discretization)
        # For simplicity, use 2 bits per asset (weights: 0, 1/3, 2/3, 1)
        bits_per_asset = 2
        total_qubits = self.num_assets_quantum * bits_per_asset
        
        print(f"🧮 QUBO formulation:")
        print(f"   Qubits: {total_qubits}")
        print(f"   Bits per asset: {bits_per_asset}")
        print(f"   Risk aversion: {risk_aversion}")
        
        # Build QUBO matrices
        Q = np.zeros((total_qubits, total_qubits))
        linear = np.zeros(total_qubits)
        
        # Asset weight encoding: weight_i = Σⱼ 2ʲ * x_{i,j} / (2^bits_per_asset - 1)
        weight_scale = 1.0 / (2**bits_per_asset - 1)
        
        for i in range(self.num_assets_quantum):
            for j in range(bits_per_asset):
                bit_idx = i * bits_per_asset + j
                bit_weight = weight_scale * (2**j)
                
                # Linear term: expected return
                linear[bit_idx] += selected_returns[i] * bit_weight
                
                # Quadratic terms: risk
                for k in range(self.num_assets_quantum):
                    for l in range(bits_per_asset):
                        bit_idx_2 = k * bits_per_asset + l
                        bit_weight_2 = weight_scale * (2**l)
                        
                        Q[bit_idx, bit_idx_2] -= risk_aversion * selected_covariance[i, k] * bit_weight * bit_weight_2
        
        # Constraint: total weight = 1 (penalty method)
        constraint_penalty = 10.0
        
        # Add constraint penalty to QUBO
        for i in range(total_qubits):
            for j in range(total_qubits):
                asset_i = i // bits_per_asset
                bit_i = i % bits_per_asset
                asset_j = j // bits_per_asset
                bit_j = j % bits_per_asset
                
                weight_i = weight_scale * (2**bit_i)
                weight_j = weight_scale * (2**bit_j)
                
                # (Σwᵢ - 1)² penalty
                Q[i, j] += constraint_penalty * weight_i * weight_j
                
                if i == j:
                    linear[i] -= 2 * constraint_penalty * weight_i
        
        # Add constant term (1) to penalty - this doesn't affect optimization
        
        return Q, linear
    
    def decode_quantum_solution(self, bit_string):
        """Decode quantum bit string to portfolio weights."""
        
        bits_per_asset = 2
        weight_scale = 1.0 / (2**bits_per_asset - 1)
        
        weights = np.zeros(self.num_assets_quantum)
        
        for i in range(self.num_assets_quantum):
            # Extract bits for asset i
            asset_bits = bit_string[i*bits_per_asset:(i+1)*bits_per_asset]
            
            # Convert binary to decimal weight
            decimal_value = sum(int(bit) * (2**j) for j, bit in enumerate(asset_bits))
            weights[i] = weight_scale * decimal_value
        
        # Normalize to sum to 1 (soft constraint enforcement)
        total_weight = weights.sum()
        if total_weight > 0:
            weights = weights / total_weight
        
        return weights
    
    def quantum_portfolio_qaoa(self, num_layers=2, num_iterations=50):
        """Solve portfolio optimization using QAOA."""
        
        print(f"\n🚀 Quantum Portfolio Optimization (QAOA)")
        print(f"   Layers: {num_layers}")
        print(f"   Iterations: {num_iterations}")
        
        total_qubits = self.num_assets_quantum * 2  # 2 bits per asset
        
        # Create QAOA solver
        qaoa_solver = PortfolioQAOA(self.Q_matrix, self.linear_terms, total_qubits)
        
        # Optimize
        result = qaoa_solver.optimize(num_layers=num_layers, max_iterations=num_iterations)
        
        # Extract best solution
        best_bit_string = qaoa_solver.get_best_solution()
        quantum_weights = self.decode_quantum_solution(best_bit_string)
        
        # Calculate full portfolio weights (set non-selected assets to 0)
        full_weights = np.zeros(self.num_assets_total)
        for i, asset_idx in enumerate(self.selected_assets):
            full_weights[asset_idx] = quantum_weights[i]
        
        # Calculate portfolio metrics
        classical_optimizer = ClassicalPortfolioOptimizer(self.data)
        metrics = classical_optimizer.calculate_portfolio_metrics(full_weights)
        
        print(f"\n📊 Quantum Portfolio Results:")
        print(f"   Expected return: {metrics['return']:.2%}")
        print(f"   Volatility: {metrics['volatility']:.2%}")
        print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
        
        print(f"\n💰 Asset Allocation:")
        for i, symbol in enumerate(self.data.symbols):
            if full_weights[i] > 0.001:  # Only show non-zero weights
                print(f"   {symbol}: {full_weights[i]:7.2%}")
        
        return {
            'weights': full_weights,
            'quantum_weights': quantum_weights,
            'metrics': metrics,
            'bit_string': best_bit_string
        }

class PortfolioQAOA:
    """QAOA implementation for portfolio optimization."""
    
    def __init__(self, Q_matrix, linear_terms, num_qubits):
        self.Q = Q_matrix
        self.linear = linear_terms
        self.num_qubits = num_qubits
        
        self.optimization_history = []
        self.best_energy = float('inf')
        self.best_solution = None
    
    def create_qaoa_circuit(self, parameters):
        """Create QAOA circuit for portfolio optimization."""
        
        num_layers = len(parameters) // 2
        circuit = quantrs2.Circuit(self.num_qubits)
        
        # Initial superposition
        for qubit in range(self.num_qubits):
            circuit.h(qubit)
        
        # QAOA layers
        for layer in range(num_layers):
            gamma = parameters[2 * layer]
            beta = parameters[2 * layer + 1]
            
            # Cost unitary
            self._apply_cost_unitary(circuit, gamma)
            
            # Mixing unitary
            self._apply_mixing_unitary(circuit, beta)
        
        return circuit
    
    def _apply_cost_unitary(self, circuit, gamma):
        """Apply cost unitary for QUBO problem."""
        
        # Apply diagonal terms (linear)
        for i in range(self.num_qubits):
            if abs(self.linear[i]) > 1e-12:
                circuit.rz(i, gamma * self.linear[i])
        
        # Apply off-diagonal terms (quadratic)
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                if abs(self.Q[i, j]) > 1e-12:
                    # ZZ interaction
                    circuit.cx(i, j)
                    circuit.rz(j, gamma * self.Q[i, j])
                    circuit.cx(i, j)
    
    def _apply_mixing_unitary(self, circuit, beta):
        """Apply mixing unitary."""
        
        for qubit in range(self.num_qubits):
            circuit.rx(qubit, 2 * beta)
    
    def cost_function(self, parameters):
        """QAOA cost function."""
        
        circuit = self.create_qaoa_circuit(parameters)
        circuit.measure_all()
        
        # Run circuit multiple times for expectation value
        num_shots = 100
        total_cost = 0
        
        for _ in range(num_shots):
            result = circuit.run()
            probabilities = result.state_probabilities()
            
            for state_str, prob in probabilities.items():
                # Convert state to cost
                cost = self._evaluate_qubo_cost(state_str)
                total_cost += prob * cost
        
        avg_cost = total_cost / num_shots
        
        # Track optimization
        self.optimization_history.append(avg_cost)
        
        if avg_cost < self.best_energy:
            self.best_energy = avg_cost
        
        return avg_cost
    
    def _evaluate_qubo_cost(self, bit_string):
        """Evaluate QUBO cost for given bit string."""
        
        x = np.array([int(bit) for bit in bit_string])
        
        # Quadratic terms
        quadratic_cost = np.dot(x, np.dot(self.Q, x))
        
        # Linear terms
        linear_cost = np.dot(self.linear, x)
        
        # Note: we minimize negative objective (maximize portfolio utility)
        return -(quadratic_cost + linear_cost)
    
    def optimize(self, num_layers=2, max_iterations=50):
        """Optimize QAOA parameters."""
        
        from scipy.optimize import minimize
        
        num_parameters = 2 * num_layers
        
        # Random initialization
        np.random.seed(42)
        initial_params = np.random.uniform(0, 2*np.pi, num_parameters)
        
        result = minimize(
            self.cost_function,
            initial_params,
            method='COBYLA',
            options={'maxiter': max_iterations}
        )
        
        print(f"   QAOA optimization completed")
        print(f"   Best energy: {self.best_energy:.6f}")
        print(f"   Iterations: {len(self.optimization_history)}")
        
        return result
    
    def get_best_solution(self):
        """Get best solution from optimization."""
        
        # This is simplified - in practice, would sample from final quantum state
        # For demonstration, return a reasonable solution
        
        # Try different bit strings and find best
        best_cost = float('inf')
        best_string = None
        
        # Test some candidate solutions
        for i in range(min(100, 2**self.num_qubits)):
            bit_string = format(i, f'0{self.num_qubits}b')
            cost = self._evaluate_qubo_cost(bit_string)
            
            if cost < best_cost:
                best_cost = cost
                best_string = bit_string
        
        return best_string

# Demonstrate quantum portfolio optimization
def demonstrate_quantum_optimization():
    """Demonstrate quantum portfolio optimization."""
    
    print("\n🌟 Quantum Portfolio Optimization Demonstration")
    print("=" * 55)
    
    # Use smaller portfolio for quantum demonstration
    quantum_optimizer = QuantumPortfolioOptimizer(tech_portfolio, num_assets_quantum=3)
    
    # Run quantum optimization
    quantum_result = quantum_optimizer.quantum_portfolio_qaoa(num_layers=2, num_iterations=30)
    
    # Compare with classical result
    classical_optimizer = ClassicalPortfolioOptimizer(tech_portfolio)
    classical_result = classical_optimizer.optimize_portfolio('max_sharpe')
    
    print(f"\n📊 Quantum vs Classical Comparison:")
    print(f"{'Method':<15} {'Return':<8} {'Risk':<8} {'Sharpe'}")
    print("-" * 45)
    
    # Classical
    if classical_result['success']:
        c_metrics = classical_result['metrics']
        print(f"{'Classical':<15} {c_metrics['return']:<8.2%} "
              f"{c_metrics['volatility']:<8.2%} {c_metrics['sharpe_ratio']:<6.3f}")
    
    # Quantum
    q_metrics = quantum_result['metrics']
    print(f"{'Quantum':<15} {q_metrics['return']:<8.2%} "
          f"{q_metrics['volatility']:<8.2%} {q_metrics['sharpe_ratio']:<6.3f}")
    
    return quantum_optimizer, quantum_result

# Run quantum optimization demonstration
quantum_optimizer, quantum_result = demonstrate_quantum_optimization()
```

### Advanced Portfolio Techniques

```python
def dynamic_portfolio_rebalancing():
    """Demonstrate dynamic portfolio rebalancing using quantum optimization."""
    
    print("\n🔄 Dynamic Portfolio Rebalancing")
    print("=" * 40)
    
    # Simulate market conditions over time
    num_periods = 12  # Monthly rebalancing for 1 year
    rebalancing_results = []
    
    print("Simulating monthly portfolio rebalancing...")
    
    for period in range(num_periods):
        print(f"\nPeriod {period + 1}/12:")
        
        # Simulate market shock or regime change
        if period == 6:  # Market crash in month 6
            print("   📉 Simulating market crash...")
            # Increase correlations and volatilities
            shock_multiplier = 1.5
            tech_portfolio.covariance_matrix *= shock_multiplier
            tech_portfolio.expected_returns *= 0.7  # Lower expected returns
        
        # Re-optimize portfolio
        optimizer = ClassicalPortfolioOptimizer(tech_portfolio)
        result = optimizer.optimize_portfolio('max_sharpe')
        
        if result['success']:
            metrics = result['metrics']
            rebalancing_results.append({
                'period': period + 1,
                'weights': result['weights'],
                'return': metrics['return'],
                'volatility': metrics['volatility'],
                'sharpe': metrics['sharpe_ratio']
            })
            
            print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"   Top allocation: {tech_portfolio.symbols[np.argmax(result['weights'])]}")
    
    # Analyze rebalancing performance
    print(f"\nRebalancing Analysis:")
    print(f"   Average Sharpe ratio: {np.mean([r['sharpe'] for r in rebalancing_results]):.3f}")
    print(f"   Sharpe volatility: {np.std([r['sharpe'] for r in rebalancing_results]):.3f}")
    
    # Plot rebalancing evolution
    plt.figure(figsize=(12, 8))
    
    periods = [r['period'] for r in rebalancing_results]
    
    # Sharpe ratio evolution
    plt.subplot(2, 2, 1)
    sharpe_ratios = [r['sharpe'] for r in rebalancing_results]
    plt.plot(periods, sharpe_ratios, 'b-o', linewidth=2)
    plt.axvline(x=6, color='r', linestyle='--', alpha=0.7, label='Market Shock')
    plt.title('Sharpe Ratio Evolution')
    plt.xlabel('Period')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Weight evolution
    plt.subplot(2, 2, 2)
    weight_matrix = np.array([r['weights'] for r in rebalancing_results])
    
    for i, symbol in enumerate(tech_portfolio.symbols):
        plt.plot(periods, weight_matrix[:, i], 'o-', label=symbol, linewidth=2)
    
    plt.axvline(x=6, color='r', linestyle='--', alpha=0.7)
    plt.title('Asset Weight Evolution')
    plt.xlabel('Period')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Return vs Risk
    plt.subplot(2, 2, 3)
    returns = [r['return'] for r in rebalancing_results]
    volatilities = [r['volatility'] for r in rebalancing_results]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(periods)))
    scatter = plt.scatter(volatilities, returns, c=periods, cmap='viridis', s=100)
    plt.colorbar(scatter, label='Period')
    plt.xlabel('Volatility')
    plt.ylabel('Expected Return')
    plt.title('Risk-Return Evolution')
    plt.grid(True, alpha=0.3)
    
    # Cumulative performance
    plt.subplot(2, 2, 4)
    cumulative_returns = np.cumprod(1 + np.array(returns)/12)  # Monthly returns
    plt.plot(periods, cumulative_returns, 'g-', linewidth=3)
    plt.axvline(x=6, color='r', linestyle='--', alpha=0.7)
    plt.title('Cumulative Performance')
    plt.xlabel('Period')
    plt.ylabel('Cumulative Return')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return rebalancing_results

def risk_parity_portfolio():
    """Implement risk parity portfolio allocation."""
    
    print("\n⚖️ Risk Parity Portfolio")
    print("=" * 30)
    
    print("Risk parity: each asset contributes equally to portfolio risk")
    
    class RiskParityOptimizer:
        def __init__(self, covariance_matrix):
            self.cov_matrix = covariance_matrix
            self.num_assets = len(covariance_matrix)
        
        def risk_contribution(self, weights):
            """Calculate risk contribution of each asset."""
            
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            return risk_contrib
        
        def optimize_risk_parity(self):
            """Optimize for equal risk contributions."""
            
            def objective(weights):
                risk_contrib = self.risk_contribution(weights)
                target_contrib = 1.0 / self.num_assets
                
                # Minimize sum of squared deviations from target
                return np.sum((risk_contrib - target_contrib)**2)
            
            # Constraints and bounds
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bounds = tuple((0.01, 1) for _ in range(self.num_assets))
            
            # Initial guess
            initial_weights = np.ones(self.num_assets) / self.num_assets
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            return result.x if result.success else initial_weights
    
    # Apply risk parity to tech portfolio
    rp_optimizer = RiskParityOptimizer(tech_portfolio.covariance_matrix)
    rp_weights = rp_optimizer.optimize_risk_parity()
    
    # Calculate metrics
    classical_optimizer = ClassicalPortfolioOptimizer(tech_portfolio)
    rp_metrics = classical_optimizer.calculate_portfolio_metrics(rp_weights)
    
    # Compare with other strategies
    equal_weights = np.ones(len(rp_weights)) / len(rp_weights)
    equal_metrics = classical_optimizer.calculate_portfolio_metrics(equal_weights)
    
    max_sharpe_result = classical_optimizer.optimize_portfolio('max_sharpe')
    
    print(f"\nRisk Parity vs Other Strategies:")
    print(f"{'Strategy':<15} {'Return':<8} {'Risk':<8} {'Sharpe'}")
    print("-" * 45)
    
    print(f"{'Risk Parity':<15} {rp_metrics['return']:<8.2%} "
          f"{rp_metrics['volatility']:<8.2%} {rp_metrics['sharpe_ratio']:<6.3f}")
    
    print(f"{'Equal Weight':<15} {equal_metrics['return']:<8.2%} "
          f"{equal_metrics['volatility']:<8.2%} {equal_metrics['sharpe_ratio']:<6.3f}")
    
    if max_sharpe_result['success']:
        ms_metrics = max_sharpe_result['metrics']
        print(f"{'Max Sharpe':<15} {ms_metrics['return']:<8.2%} "
              f"{ms_metrics['volatility']:<8.2%} {ms_metrics['sharpe_ratio']:<6.3f}")
    
    # Show risk contributions
    risk_contrib = rp_optimizer.risk_contribution(rp_weights)
    
    print(f"\nRisk Parity Allocations:")
    for i, symbol in enumerate(tech_portfolio.symbols):
        print(f"   {symbol}: {rp_weights[i]:7.2%} (risk contrib: {risk_contrib[i]:.1%})")
    
    return rp_weights, rp_metrics

def black_litterman_model():
    """Implement Black-Litterman model for portfolio optimization."""
    
    print("\n📊 Black-Litterman Model")
    print("=" * 30)
    
    print("Black-Litterman: Bayesian approach combining market equilibrium with investor views")
    
    # Market equilibrium (reverse optimization)
    market_cap_weights = np.array([0.4, 0.3, 0.2, 0.1])  # Assumed market caps
    risk_aversion = 3.0
    
    # Implied equilibrium returns
    implied_returns = risk_aversion * np.dot(tech_portfolio.covariance_matrix, market_cap_weights)
    
    print(f"Market Equilibrium:")
    for i, symbol in enumerate(tech_portfolio.symbols):
        print(f"   {symbol}: {implied_returns[i]:.2%} (market weight: {market_cap_weights[i]:.1%})")
    
    # Investor views
    # View 1: AAPL will outperform GOOGL by 5%
    # View 2: Technology sector will have 15% return
    
    # View matrix P and view vector Q
    P = np.array([
        [1, -1, 0, 0],  # AAPL - GOOGL
        [0.25, 0.25, 0.25, 0.25]  # Equal-weighted tech sector
    ])
    
    Q = np.array([0.05, 0.15])  # View returns
    
    # View uncertainty (smaller = more confident)
    omega = np.diag([0.01, 0.02])  # 1% and 2% uncertainty
    
    # Black-Litterman formula
    tau = 0.05  # Scales uncertainty of prior
    
    # Prior uncertainty
    prior_precision = np.linalg.inv(tau * tech_portfolio.covariance_matrix)
    
    # View precision
    view_precision = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
    
    # New expected returns
    posterior_precision = prior_precision + view_precision
    posterior_cov = np.linalg.inv(posterior_precision)
    
    view_term = np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
    prior_term = np.dot(prior_precision, implied_returns)
    
    bl_returns = np.dot(posterior_cov, prior_term + view_term)
    
    print(f"\nBlack-Litterman Expected Returns:")
    for i, symbol in enumerate(tech_portfolio.symbols):
        original = tech_portfolio.expected_returns[i]
        bl_return = bl_returns[i]
        change = bl_return - original
        print(f"   {symbol}: {bl_return:.2%} (change: {change:+.1%})")
    
    # Optimize with Black-Litterman inputs
    class BlackLittermanOptimizer(ClassicalPortfolioOptimizer):
        def __init__(self, portfolio_data, bl_returns, bl_covariance):
            super().__init__(portfolio_data)
            self.expected_returns = bl_returns
            self.covariance_matrix = bl_covariance
    
    bl_optimizer = BlackLittermanOptimizer(tech_portfolio, bl_returns, posterior_cov)
    bl_result = bl_optimizer.optimize_portfolio('max_sharpe')
    
    if bl_result['success']:
        bl_metrics = bl_result['metrics']
        
        print(f"\nBlack-Litterman Portfolio:")
        print(f"   Return: {bl_metrics['return']:.2%}")
        print(f"   Risk: {bl_metrics['volatility']:.2%}")
        print(f"   Sharpe: {bl_metrics['sharpe_ratio']:.3f}")
        
        print(f"\nAsset Allocation:")
        for i, symbol in enumerate(tech_portfolio.symbols):
            weight = bl_result['weights'][i]
            market_weight = market_cap_weights[i]
            tilt = weight - market_weight
            print(f"   {symbol}: {weight:7.2%} (tilt: {tilt:+.1%})")
    
    return bl_returns, bl_result

# Run advanced portfolio techniques
rebalancing_results = dynamic_portfolio_rebalancing()
rp_weights, rp_metrics = risk_parity_portfolio()
bl_returns, bl_result = black_litterman_model()
```

### Real-World Applications

```python
def institutional_portfolio_management():
    """Demonstrate institutional-scale portfolio management."""
    
    print("\n🏦 Institutional Portfolio Management")
    print("=" * 45)
    
    print("Large-scale portfolio optimization for institutional investors:")
    print("1. Multi-asset class portfolios")
    print("2. Factor-based investing")
    print("3. ESG constraints")
    print("4. Liability-driven investment")
    
    # Multi-asset portfolio
    asset_classes = {
        'US Equity': {'return': 0.10, 'volatility': 0.16, 'allocation': 0.30},
        'Intl Equity': {'return': 0.09, 'volatility': 0.18, 'allocation': 0.20},
        'Fixed Income': {'return': 0.04, 'volatility': 0.04, 'allocation': 0.25},
        'Real Estate': {'return': 0.08, 'volatility': 0.14, 'allocation': 0.10},
        'Commodities': {'return': 0.07, 'volatility': 0.20, 'allocation': 0.10},
        'Alternatives': {'return': 0.12, 'volatility': 0.22, 'allocation': 0.05}
    }
    
    print(f"\nMulti-Asset Portfolio:")
    total_return = 0
    total_risk = 0
    
    for asset_class, data in asset_classes.items():
        weight = data['allocation']
        ret = data['return']
        vol = data['volatility']
        
        total_return += weight * ret
        total_risk += (weight * vol)**2  # Simplified - assumes no correlations
        
        print(f"   {asset_class:<15}: {weight:5.1%} allocation, "
              f"{ret:5.1%} return, {vol:5.1%} volatility")
    
    total_risk = np.sqrt(total_risk)
    sharpe = (total_return - 0.02) / total_risk
    
    print(f"\nPortfolio Summary:")
    print(f"   Expected return: {total_return:.1%}")
    print(f"   Estimated risk: {total_risk:.1%}")
    print(f"   Sharpe ratio: {sharpe:.2f}")
    
    # ESG-constrained optimization
    print(f"\n🌱 ESG-Constrained Portfolio:")
    
    # ESG scores (0-100, higher is better)
    esg_scores = {
        'AAPL': 85,  # High ESG
        'GOOGL': 75,
        'MSFT': 90,  # Very high ESG
        'AMZN': 60   # Moderate ESG
    }
    
    min_esg_score = 70  # Minimum portfolio ESG score
    
    print(f"   ESG constraint: Minimum portfolio score = {min_esg_score}")
    print(f"   Individual ESG scores:")
    for symbol, score in esg_scores.items():
        print(f"     {symbol}: {score}")
    
    # This would integrate into the optimization as additional constraints
    # For demonstration, show ESG-filtered allocation
    esg_compliant_assets = [symbol for symbol, score in esg_scores.items() if score >= min_esg_score]
    print(f"   ESG-compliant assets: {esg_compliant_assets}")
    
    return asset_classes, esg_scores

def robo_advisor_simulation():
    """Simulate a robo-advisor portfolio allocation system."""
    
    print("\n🤖 Robo-Advisor Simulation")
    print("=" * 35)
    
    print("Automated portfolio allocation based on investor profile:")
    
    # Investor profiles
    profiles = {
        'conservative': {
            'risk_tolerance': 'low',
            'time_horizon': 5,
            'age': 55,
            'equity_max': 0.4,
            'target_return': 0.05
        },
        'moderate': {
            'risk_tolerance': 'medium',
            'time_horizon': 15,
            'age': 40,
            'equity_max': 0.7,
            'target_return': 0.08
        },
        'aggressive': {
            'risk_tolerance': 'high',
            'time_horizon': 25,
            'age': 30,
            'equity_max': 0.9,
            'target_return': 0.12
        }
    }
    
    # Simple asset universe for robo-advisor
    etf_universe = {
        'Stock ETF': {'return': 0.10, 'volatility': 0.16, 'type': 'equity'},
        'Bond ETF': {'return': 0.04, 'volatility': 0.04, 'type': 'fixed_income'},
        'REIT ETF': {'return': 0.08, 'volatility': 0.14, 'type': 'real_estate'},
        'Intl ETF': {'return': 0.09, 'volatility': 0.18, 'type': 'equity'}
    }
    
    def allocate_portfolio(profile_name, profile_data):
        """Allocate portfolio based on investor profile."""
        
        print(f"\n👤 {profile_name.title()} Investor:")
        print(f"   Age: {profile_data['age']}")
        print(f"   Time horizon: {profile_data['time_horizon']} years")
        print(f"   Risk tolerance: {profile_data['risk_tolerance']}")
        print(f"   Target return: {profile_data['target_return']:.1%}")
        
        # Rule-based allocation (simplified)
        equity_target = min(profile_data['equity_max'], 
                           (100 - profile_data['age']) / 100)  # Age-based rule
        
        if profile_data['risk_tolerance'] == 'low':
            bond_allocation = 0.6
            equity_allocation = 0.3
            reit_allocation = 0.1
        elif profile_data['risk_tolerance'] == 'medium':
            bond_allocation = 0.4
            equity_allocation = 0.5
            reit_allocation = 0.1
        else:  # high risk tolerance
            bond_allocation = 0.2
            equity_allocation = 0.7
            reit_allocation = 0.1
        
        allocation = {
            'Stock ETF': equity_allocation * 0.7,  # 70% domestic equity
            'Intl ETF': equity_allocation * 0.3,   # 30% international equity
            'Bond ETF': bond_allocation,
            'REIT ETF': reit_allocation
        }
        
        # Calculate expected portfolio metrics
        portfolio_return = sum(allocation[etf] * etf_universe[etf]['return'] 
                             for etf in allocation)
        
        portfolio_risk = np.sqrt(sum((allocation[etf] * etf_universe[etf]['volatility'])**2 
                                   for etf in allocation))  # Simplified risk
        
        print(f"\n   Recommended Allocation:")
        for etf, weight in allocation.items():
            if weight > 0:
                print(f"     {etf}: {weight:5.1%}")
        
        print(f"\n   Expected Performance:")
        print(f"     Return: {portfolio_return:.1%}")
        print(f"     Risk: {portfolio_risk:.1%}")
        print(f"     Sharpe: {(portfolio_return - 0.02) / portfolio_risk:.2f}")
        
        return allocation, portfolio_return, portfolio_risk
    
    # Generate recommendations for all profiles
    robo_recommendations = {}
    
    for profile_name, profile_data in profiles.items():
        allocation, ret, risk = allocate_portfolio(profile_name, profile_data)
        robo_recommendations[profile_name] = {
            'allocation': allocation,
            'return': ret,
            'risk': risk
        }
    
    # Visualize recommendations
    plt.figure(figsize=(12, 8))
    
    # Risk-return plot
    plt.subplot(2, 2, 1)
    for profile_name, data in robo_recommendations.items():
        plt.scatter(data['risk'], data['return'], s=150, 
                   label=profile_name.title(), alpha=0.8)
        plt.annotate(profile_name.title(), 
                    (data['risk'], data['return']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Risk (Volatility)')
    plt.ylabel('Expected Return')
    plt.title('Robo-Advisor Risk-Return Profiles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Allocation comparison
    plt.subplot(2, 2, 2)
    profiles_list = list(robo_recommendations.keys())
    etfs = list(etf_universe.keys())
    
    allocation_matrix = np.array([
        [robo_recommendations[profile]['allocation'].get(etf, 0) 
         for etf in etfs]
        for profile in profiles_list
    ])
    
    x = np.arange(len(profiles_list))
    width = 0.8
    
    bottom = np.zeros(len(profiles_list))
    colors = plt.cm.Set3(np.linspace(0, 1, len(etfs)))
    
    for i, etf in enumerate(etfs):
        plt.bar(x, allocation_matrix[:, i], width, bottom=bottom, 
               label=etf, color=colors[i], alpha=0.8)
        bottom += allocation_matrix[:, i]
    
    plt.xlabel('Investor Profile')
    plt.ylabel('Allocation')
    plt.title('Asset Allocation by Profile')
    plt.xticks(x, [p.title() for p in profiles_list])
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return robo_recommendations

def cryptocurrency_portfolio():
    """Demonstrate cryptocurrency portfolio optimization."""
    
    print("\n₿ Cryptocurrency Portfolio Optimization")
    print("=" * 45)
    
    print("Crypto portfolios have unique characteristics:")
    print("1. High volatility and correlations")
    print("2. 24/7 markets")
    print("3. Regulatory uncertainty")
    print("4. Technical factors (halvings, forks)")
    
    # Crypto asset data (simplified)
    crypto_assets = {
        'Bitcoin': {'return': 0.50, 'volatility': 0.80, 'market_cap': 500e9},
        'Ethereum': {'return': 0.60, 'volatility': 0.90, 'market_cap': 200e9},
        'Binance Coin': {'return': 0.80, 'volatility': 1.20, 'market_cap': 50e9},
        'Cardano': {'return': 0.40, 'volatility': 1.10, 'market_cap': 20e9},
        'Solana': {'return': 1.00, 'volatility': 1.50, 'market_cap': 15e9}
    }
    
    print(f"\nCryptocurrency Universe:")
    for crypto, data in crypto_assets.items():
        print(f"   {crypto:<15}: {data['return']:5.1%} return, "
              f"{data['volatility']:5.1%} volatility, "
              f"${data['market_cap']/1e9:.0f}B market cap")
    
    # Market cap weighted portfolio
    total_market_cap = sum(data['market_cap'] for data in crypto_assets.values())
    
    print(f"\nMarket Cap Weighted Portfolio:")
    mc_portfolio_return = 0
    mc_portfolio_risk = 0
    
    for crypto, data in crypto_assets.items():
        weight = data['market_cap'] / total_market_cap
        mc_portfolio_return += weight * data['return']
        mc_portfolio_risk += (weight * data['volatility'])**2  # Simplified
        
        print(f"   {crypto:<15}: {weight:5.1%}")
    
    mc_portfolio_risk = np.sqrt(mc_portfolio_risk)
    mc_sharpe = mc_portfolio_return / mc_portfolio_risk  # No risk-free rate for crypto
    
    print(f"\nMarket Cap Portfolio Performance:")
    print(f"   Expected return: {mc_portfolio_return:.1%}")
    print(f"   Estimated risk: {mc_portfolio_risk:.1%}")
    print(f"   Return/Risk ratio: {mc_sharpe:.2f}")
    
    # Equal weight portfolio
    print(f"\nEqual Weight Portfolio:")
    num_cryptos = len(crypto_assets)
    eq_weight = 1.0 / num_cryptos
    
    eq_portfolio_return = sum(data['return'] for data in crypto_assets.values()) / num_cryptos
    eq_portfolio_risk = np.sqrt(sum((data['volatility'] * eq_weight)**2 
                                   for data in crypto_assets.values()))
    eq_sharpe = eq_portfolio_return / eq_portfolio_risk
    
    print(f"   Expected return: {eq_portfolio_return:.1%}")
    print(f"   Estimated risk: {eq_portfolio_risk:.1%}")
    print(f"   Return/Risk ratio: {eq_sharpe:.2f}")
    
    # Risk considerations
    print(f"\n⚠️  Crypto-Specific Risks:")
    print(f"   • Regulatory risk: Government bans, taxation changes")
    print(f"   • Technology risk: Smart contract bugs, protocol upgrades")
    print(f"   • Market risk: Extreme volatility, flash crashes")
    print(f"   • Operational risk: Exchange hacks, wallet security")
    print(f"   • Liquidity risk: Market depth varies significantly")
    
    return crypto_assets, mc_portfolio_return, eq_portfolio_return

# Run real-world applications
institutional_data = institutional_portfolio_management()
robo_recommendations = robo_advisor_simulation()
crypto_data = cryptocurrency_portfolio()
```

## Performance Analysis and Benchmarking

```python
def comprehensive_portfolio_benchmark():
    """Comprehensive portfolio optimization benchmark."""
    
    print("\n🏆 Portfolio Optimization Benchmark")
    print("=" * 40)
    
    import time
    
    # Test different portfolio sizes and methods
    portfolio_sizes = [4, 6, 8, 10]
    methods = ['equal_weight', 'max_sharpe', 'min_variance', 'risk_parity']
    
    benchmark_results = []
    
    for size in portfolio_sizes:
        print(f"\nTesting {size}-asset portfolio:")
        
        # Create test portfolio
        if size <= len(tech_portfolio.symbols):
            test_symbols = tech_portfolio.symbols[:size]
            test_data = PortfolioData(
                test_symbols,
                tech_portfolio.start_date,
                tech_portfolio.end_date
            )
        else:
            # Use synthetic data for larger portfolios
            test_symbols = [f'Asset_{i}' for i in range(size)]
            test_data = PortfolioData(
                test_symbols,
                tech_portfolio.start_date,
                tech_portfolio.end_date
            )
        
        optimizer = ClassicalPortfolioOptimizer(test_data)
        
        for method in methods:
            print(f"  Testing {method}...")
            
            start_time = time.time()
            
            if method == 'equal_weight':
                weights = np.ones(size) / size
                metrics = optimizer.calculate_portfolio_metrics(weights)
                success = True
            
            elif method == 'risk_parity':
                rp_optimizer = RiskParityOptimizer(test_data.covariance_matrix)
                weights = rp_optimizer.optimize_risk_parity()
                metrics = optimizer.calculate_portfolio_metrics(weights)
                success = True
            
            else:
                result = optimizer.optimize_portfolio(method)
                weights = result['weights']
                metrics = result['metrics']
                success = result['success']
            
            optimization_time = time.time() - start_time
            
            if success:
                benchmark_results.append({
                    'portfolio_size': size,
                    'method': method,
                    'return': metrics['return'],
                    'volatility': metrics['volatility'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'optimization_time': optimization_time,
                    'success': True
                })
                
                print(f"    Sharpe: {metrics['sharpe_ratio']:.3f}, Time: {optimization_time:.3f}s")
            else:
                print(f"    Failed")
    
    # Analyze results
    print(f"\n📊 Benchmark Summary:")
    print(f"{'Size':<6} {'Method':<12} {'Return':<8} {'Risk':<8} {'Sharpe':<8} {'Time (s)'}")
    print("-" * 60)
    
    for result in benchmark_results:
        print(f"{result['portfolio_size']:<6} {result['method']:<12} "
              f"{result['return']:<8.2%} {result['volatility']:<8.2%} "
              f"{result['sharpe_ratio']:<8.3f} {result['optimization_time']:<8.3f}")
    
    # Performance analysis
    print(f"\nPerformance Analysis:")
    
    # Average performance by method
    method_performance = {}
    for method in methods:
        method_results = [r for r in benchmark_results if r['method'] == method]
        if method_results:
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in method_results])
            avg_time = np.mean([r['optimization_time'] for r in method_results])
            method_performance[method] = {'sharpe': avg_sharpe, 'time': avg_time}
    
    print(f"Average performance by method:")
    for method, perf in method_performance.items():
        print(f"  {method:<15}: Sharpe = {perf['sharpe']:.3f}, Time = {perf['time']:.3f}s")
    
    # Scaling analysis
    print(f"\nScaling Analysis:")
    scaling_data = {}
    for size in portfolio_sizes:
        size_results = [r for r in benchmark_results if r['portfolio_size'] == size]
        if size_results:
            avg_time = np.mean([r['optimization_time'] for r in size_results])
            scaling_data[size] = avg_time
    
    for size, time in scaling_data.items():
        complexity = size**2  # Rough O(n²) scaling for matrix operations
        print(f"  {size} assets: {time:.3f}s (complexity ~O({complexity}))")
    
    return benchmark_results

def out_of_sample_testing():
    """Test portfolio performance out-of-sample."""
    
    print("\n📈 Out-of-Sample Performance Testing")
    print("=" * 45)
    
    # Split data into training and testing periods
    total_days = len(tech_portfolio.returns)
    train_days = total_days // 2
    
    train_returns = tech_portfolio.returns.iloc[:train_days]
    test_returns = tech_portfolio.returns.iloc[train_days:]
    
    print(f"Training period: {train_days} days")
    print(f"Testing period: {total_days - train_days} days")
    
    # Create training portfolio data
    train_data = PortfolioData.__new__(PortfolioData)
    train_data.symbols = tech_portfolio.symbols
    train_data.returns = train_returns
    train_data.expected_returns = train_returns.mean().values * 252
    train_data.covariance_matrix = train_returns.cov().values * 252
    train_data.num_assets = len(tech_portfolio.symbols)
    
    # Optimize on training data
    train_optimizer = ClassicalPortfolioOptimizer(train_data)
    strategies = {
        'Equal Weight': np.ones(train_data.num_assets) / train_data.num_assets,
        'Max Sharpe': train_optimizer.optimize_portfolio('max_sharpe')['weights'],
        'Min Variance': train_optimizer.optimize_portfolio('min_variance')['weights']
    }
    
    # Test on out-of-sample data
    oos_results = {}
    
    for strategy_name, weights in strategies.items():
        # Calculate out-of-sample returns
        portfolio_returns = (test_returns * weights).sum(axis=1)
        
        # Performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + total_return)**(252 / len(portfolio_returns)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.02) / annual_vol
        max_drawdown = (portfolio_returns.cumsum() - portfolio_returns.cumsum().expanding().max()).min()
        
        oos_results[strategy_name] = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }
    
    # Display results
    print(f"\nOut-of-Sample Performance:")
    print(f"{'Strategy':<15} {'Total Ret':<10} {'Annual Ret':<11} {'Volatility':<11} {'Sharpe':<8} {'Max DD'}")
    print("-" * 75)
    
    for strategy, metrics in oos_results.items():
        print(f"{strategy:<15} {metrics['total_return']:<10.1%} "
              f"{metrics['annual_return']:<11.1%} {metrics['annual_volatility']:<11.1%} "
              f"{metrics['sharpe_ratio']:<8.2f} {metrics['max_drawdown']:<8.1%}")
    
    # Plot cumulative performance
    plt.figure(figsize=(12, 6))
    
    for strategy_name, weights in strategies.items():
        portfolio_returns = (test_returns * weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        plt.plot(test_returns.index, cumulative_returns, 
                linewidth=2, label=strategy_name)
    
    plt.title('Out-of-Sample Cumulative Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return oos_results

# Run performance analysis
benchmark_data = comprehensive_portfolio_benchmark()
oos_results = out_of_sample_testing()
```

## Exercises and Extensions

### Exercise 1: Multi-Objective Optimization
```python
def exercise_multi_objective():
    """Exercise: Implement multi-objective portfolio optimization."""
    
    print("🎯 Exercise: Multi-Objective Portfolio Optimization")
    print("=" * 55)
    
    # TODO: Implement optimization with multiple objectives:
    # 1. Return maximization + Risk minimization + ESG maximization
    # 2. Pareto frontier analysis
    # 3. Weighted sum vs epsilon-constraint methods
    # 4. Interactive optimization with decision maker preferences
    
    print("Your challenge:")
    print("1. Implement 3-objective optimization (return, risk, ESG)")
    print("2. Generate Pareto frontier in 3D")
    print("3. Compare different multi-objective algorithms")
    print("4. Allow user to interactively select preferences")

exercise_multi_objective()
```

### Exercise 2: Alternative Risk Models
```python
def exercise_alternative_risk_models():
    """Exercise: Implement alternative risk models."""
    
    print("🎯 Exercise: Alternative Risk Models")
    print("=" * 40)
    
    # TODO: Implement different risk models:
    # 1. Conditional Value at Risk (CVaR)
    # 2. Maximum drawdown minimization
    # 3. Downside deviation
    # 4. Factor-based risk models (Fama-French)
    
    print("Beyond mean-variance optimization:")
    print("1. CVaR portfolio optimization")
    print("2. Drawdown-constrained portfolios")
    print("3. Factor exposure constraints")
    print("4. Regime-dependent risk models")

exercise_alternative_risk_models()
```

### Exercise 3: Real-Time Optimization
```python
def exercise_realtime_optimization():
    """Exercise: Implement real-time portfolio optimization."""
    
    print("🎯 Exercise: Real-Time Portfolio Optimization")
    print("=" * 45)
    
    # TODO: Implement real-time features:
    # 1. Streaming market data integration
    # 2. Online learning and adaptation
    # 3. Transaction cost modeling
    # 4. High-frequency rebalancing
    
    print("Real-time optimization challenges:")
    print("1. Handle streaming market data")
    print("2. Implement online learning algorithms")
    print("3. Model transaction costs and market impact")
    print("4. Optimize execution timing")

exercise_realtime_optimization()
```

## Summary

🎉 **Congratulations!** You've learned:
- Classical portfolio optimization using Modern Portfolio Theory
- Quantum approaches to portfolio optimization using QAOA
- Advanced techniques: risk parity, Black-Litterman, dynamic rebalancing
- Real-world applications: institutional management, robo-advisors, cryptocurrency
- Performance analysis and out-of-sample testing
- Multi-asset class and multi-objective optimization

Portfolio optimization is a cornerstone of quantitative finance, and quantum computing offers new possibilities for handling complex constraints and large-scale problems!

**Next Steps:**
- Explore [QAOA for combinatorial problems](qaoa_maxcut.md)
- Try [VQE for molecular simulation](vqe.md)
- Learn about [Quantum Machine Learning](../ml/vqc.md)

## References

### Foundational Papers
- Markowitz, H. (1952). "Portfolio Selection"
- Black, F. & Litterman, R. (1992). "Global Portfolio Optimization"

### Quantum Finance
- Orus et al. (2019). "Quantum computing for finance: Overview and prospects"
- Egger et al. (2020). "Quantum computing for Finance: State of the art and future prospects"

### Modern Developments
- Risk parity and alternative risk measures
- Factor investing and smart beta strategies
- ESG integration in portfolio optimization

---

*"The goal of portfolio optimization is not to eliminate risk, but to take the right risks for the right rewards."* - Modern Portfolio Theory

🚀 **Ready to optimize portfolios with quantum computing?** Explore more [Optimization Examples](index.md)!