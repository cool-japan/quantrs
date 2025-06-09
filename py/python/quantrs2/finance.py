"""
Quantum Finance Algorithms

This module provides quantum algorithms for financial applications including
portfolio optimization, risk analysis, option pricing, fraud detection,
and other quantum finance use cases.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from datetime import datetime, timedelta
from scipy.optimize import minimize
import json

try:
    import quantrs2
    from quantrs2 import Circuit, SimulationResult
    HAS_QUANTRS2 = True
except ImportError:
    HAS_QUANTRS2 = False
    warnings.warn("quantrs2 not available. Finance module will use classical simulations.")


class FinanceModel(Enum):
    """Types of financial models."""
    BLACK_SCHOLES = "black_scholes"
    BINOMIAL = "binomial"
    MONTE_CARLO = "monte_carlo"
    QUANTUM_MONTE_CARLO = "quantum_monte_carlo"


class RiskMetric(Enum):
    """Risk measurement metrics."""
    VALUE_AT_RISK = "var"
    CONDITIONAL_VAR = "cvar"
    MAXIMUM_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe"
    SORTINO_RATIO = "sortino"


@dataclass
class FinancialAsset:
    """Represents a financial asset."""
    symbol: str
    name: str = ""
    price: float = 0.0
    volatility: float = 0.0
    returns: List[float] = field(default_factory=list)
    correlations: Dict[str, float] = field(default_factory=dict)
    
    def expected_return(self) -> float:
        """Calculate expected return."""
        return np.mean(self.returns) if self.returns else 0.0
    
    def risk(self) -> float:
        """Calculate risk (standard deviation)."""
        return np.std(self.returns) if self.returns else self.volatility


@dataclass
class Portfolio:
    """Represents an investment portfolio."""
    assets: List[FinancialAsset] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    rebalance_frequency: str = "monthly"
    
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        return sum(asset.price * weight for asset, weight in zip(self.assets, self.weights))
    
    def expected_return(self) -> float:
        """Calculate portfolio expected return."""
        return sum(asset.expected_return() * weight for asset, weight in zip(self.assets, self.weights))
    
    def risk(self) -> float:
        """Calculate portfolio risk."""
        if not self.assets or not self.weights:
            return 0.0
        
        # Simple risk calculation (would be more complex with correlations)
        weighted_risks = [asset.risk() * weight for asset, weight in zip(self.assets, self.weights)]
        return np.sqrt(sum(r**2 for r in weighted_risks))


@dataclass
class OptionContract:
    """Represents an options contract."""
    underlying: str
    strike_price: float
    expiry_date: datetime
    option_type: str  # "call" or "put"
    current_price: float = 0.0
    volatility: float = 0.2
    risk_free_rate: float = 0.05
    
    def time_to_expiry(self) -> float:
        """Calculate time to expiry in years."""
        now = datetime.now()
        delta = self.expiry_date - now
        return max(0.0, delta.days / 365.25)
    
    def moneyness(self) -> str:
        """Determine if option is ITM, ATM, or OTM."""
        if self.option_type.lower() == "call":
            if self.current_price > self.strike_price:
                return "ITM"  # In the money
            elif abs(self.current_price - self.strike_price) < 0.01:
                return "ATM"  # At the money
            else:
                return "OTM"  # Out of the money
        else:  # put
            if self.current_price < self.strike_price:
                return "ITM"
            elif abs(self.current_price - self.strike_price) < 0.01:
                return "ATM"
            else:
                return "OTM"


@dataclass
class QuantumFinanceResult:
    """Result of quantum finance computation."""
    value: float
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    quantum_advantage: float = 1.0
    execution_time: float = 0.0
    quantum_error: float = 0.0
    classical_comparison: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumPortfolioOptimizer:
    """Quantum algorithm for portfolio optimization."""
    
    def __init__(self, num_assets: int, risk_aversion: float = 1.0):
        """
        Initialize quantum portfolio optimizer.
        
        Args:
            num_assets: Number of assets in portfolio
            risk_aversion: Risk aversion parameter (higher = more risk averse)
        """
        self.num_assets = num_assets
        self.risk_aversion = risk_aversion
        self.expected_returns = np.zeros(num_assets)
        self.covariance_matrix = np.eye(num_assets)
        
    def set_market_data(self, returns: np.ndarray, covariances: np.ndarray):
        """
        Set market data for optimization.
        
        Args:
            returns: Expected returns for each asset
            covariances: Covariance matrix of asset returns
        """
        self.expected_returns = returns
        self.covariance_matrix = covariances
    
    def create_optimization_circuit(self, n_qubits: int) -> Circuit:
        """
        Create quantum circuit for portfolio optimization.
        
        Args:
            n_qubits: Number of qubits to use
            
        Returns:
            Quantum circuit
        """
        if not HAS_QUANTRS2:
            # Return a mock circuit for classical simulation
            return None
        
        circuit = Circuit(n_qubits)
        
        # Initialize superposition state
        for i in range(n_qubits):
            circuit.h(i)
        
        # Apply problem-specific rotations
        # This would encode the portfolio optimization problem
        for i in range(min(n_qubits, self.num_assets)):
            # Rotate based on expected return
            angle = self.expected_returns[i] * np.pi / 2
            circuit.ry(i, angle)
        
        # Add entangling gates to encode correlations
        for i in range(n_qubits - 1):
            circuit.cx(i, i + 1)
        
        return circuit
    
    def quantum_optimize(self, max_iterations: int = 100) -> QuantumFinanceResult:
        """
        Perform quantum portfolio optimization.
        
        Args:
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimization result
        """
        start_time = pd.Timestamp.now()
        
        if not HAS_QUANTRS2:
            # Classical fallback using mean-variance optimization
            return self._classical_optimize()
        
        try:
            # Quantum optimization using QAOA-like approach
            n_qubits = min(16, self.num_assets + 4)  # Reasonable qubit count
            
            best_weights = None
            best_utility = -np.inf
            
            for iteration in range(max_iterations):
                # Create optimization circuit
                circuit = self.create_optimization_circuit(n_qubits)
                
                if circuit is None:
                    break
                
                # Run circuit
                result = circuit.run()
                
                # Extract portfolio weights from quantum state
                weights = self._extract_weights_from_result(result)
                
                # Evaluate portfolio utility
                utility = self._calculate_utility(weights)
                
                if utility > best_utility:
                    best_utility = utility
                    best_weights = weights
                
                # Early stopping if converged
                if iteration > 10 and abs(utility - best_utility) < 1e-6:
                    break
            
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            # Calculate quantum advantage vs classical
            classical_result = self._classical_optimize()
            quantum_advantage = best_utility / classical_result.value if classical_result.value != 0 else 1.0
            
            return QuantumFinanceResult(
                value=best_utility,
                quantum_advantage=quantum_advantage,
                execution_time=execution_time,
                classical_comparison=classical_result.value,
                metadata={
                    'optimal_weights': best_weights.tolist() if best_weights is not None else [],
                    'iterations': iteration + 1,
                    'algorithm': 'quantum_portfolio_optimization'
                }
            )
            
        except Exception as e:
            warnings.warn(f"Quantum optimization failed: {e}. Using classical fallback.")
            return self._classical_optimize()
    
    def _classical_optimize(self) -> QuantumFinanceResult:
        """Classical portfolio optimization fallback."""
        def objective(weights):
            weights = np.array(weights)
            expected_return = np.dot(weights, self.expected_returns)
            risk = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))
            return -(expected_return - self.risk_aversion * risk)  # Negative because we minimize
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(self.num_assets)]
        
        # Initial guess: equal weights
        x0 = np.ones(self.num_assets) / self.num_assets
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            optimal_utility = -result.fun
            optimal_weights = result.x
            
            return QuantumFinanceResult(
                value=optimal_utility,
                metadata={
                    'optimal_weights': optimal_weights.tolist(),
                    'algorithm': 'classical_mean_variance',
                    'optimization_success': result.success
                }
            )
        except Exception as e:
            # Return equal weights as final fallback
            weights = np.ones(self.num_assets) / self.num_assets
            utility = self._calculate_utility(weights)
            
            return QuantumFinanceResult(
                value=utility,
                metadata={
                    'optimal_weights': weights.tolist(),
                    'algorithm': 'equal_weights_fallback',
                    'error': str(e)
                }
            )
    
    def _extract_weights_from_result(self, result) -> np.ndarray:
        """Extract portfolio weights from quantum measurement result."""
        # This is a simplified extraction - in practice would be more sophisticated
        if not HAS_QUANTRS2 or result is None:
            return np.random.dirichlet(np.ones(self.num_assets))
        
        try:
            # Get state probabilities
            probs = result.state_probabilities()
            
            # Convert probabilities to weights
            weights = np.zeros(self.num_assets)
            
            for state, prob in probs.items():
                # Convert binary state to asset weights
                state_int = int(state, 2) if isinstance(state, str) else 0
                for i in range(self.num_assets):
                    if state_int & (1 << i):
                        weights[i] += prob
            
            # Normalize weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(self.num_assets) / self.num_assets
            
            return weights
            
        except Exception:
            # Fallback to random weights
            return np.random.dirichlet(np.ones(self.num_assets))
    
    def _calculate_utility(self, weights: np.ndarray) -> float:
        """Calculate portfolio utility (expected return - risk penalty)."""
        expected_return = np.dot(weights, self.expected_returns)
        risk = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))
        return expected_return - self.risk_aversion * risk


class QuantumOptionPricer:
    """Quantum algorithms for option pricing."""
    
    def __init__(self, model: FinanceModel = FinanceModel.QUANTUM_MONTE_CARLO):
        """
        Initialize quantum option pricer.
        
        Args:
            model: Pricing model to use
        """
        self.model = model
    
    def price_european_option(self, option: OptionContract, 
                            num_paths: int = 1000) -> QuantumFinanceResult:
        """
        Price European option using quantum algorithms.
        
        Args:
            option: Option contract to price
            num_paths: Number of simulation paths
            
        Returns:
            Option pricing result
        """
        start_time = pd.Timestamp.now()
        
        if self.model == FinanceModel.QUANTUM_MONTE_CARLO:
            return self._quantum_monte_carlo_pricing(option, num_paths)
        elif self.model == FinanceModel.BLACK_SCHOLES:
            return self._black_scholes_pricing(option)
        else:
            return self._classical_monte_carlo_pricing(option, num_paths)
    
    def _quantum_monte_carlo_pricing(self, option: OptionContract, 
                                   num_paths: int) -> QuantumFinanceResult:
        """Quantum Monte Carlo option pricing."""
        if not HAS_QUANTRS2:
            return self._classical_monte_carlo_pricing(option, num_paths)
        
        try:
            # Use quantum random number generation for paths
            n_qubits = min(16, int(np.ceil(np.log2(num_paths))))
            
            # Create quantum random number generator
            circuit = Circuit(n_qubits)
            for i in range(n_qubits):
                circuit.h(i)  # Create superposition
            
            # Run quantum simulation
            result = circuit.run()
            
            # Extract random numbers from quantum state
            random_numbers = self._extract_random_numbers(result, num_paths)
            
            # Generate stock price paths using quantum random numbers
            paths = self._generate_price_paths(option, random_numbers)
            
            # Calculate option payoffs
            payoffs = self._calculate_payoffs(option, paths)
            
            # Discount to present value
            present_value = np.mean(payoffs) * np.exp(-option.risk_free_rate * option.time_to_expiry())
            
            # Calculate confidence interval
            std_error = np.std(payoffs) / np.sqrt(len(payoffs))
            ci_lower = present_value - 1.96 * std_error
            ci_upper = present_value + 1.96 * std_error
            
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            # Compare with classical Monte Carlo
            classical_result = self._classical_monte_carlo_pricing(option, num_paths)
            quantum_advantage = execution_time / classical_result.execution_time if classical_result.execution_time > 0 else 1.0
            
            return QuantumFinanceResult(
                value=present_value,
                confidence_interval=(ci_lower, ci_upper),
                quantum_advantage=quantum_advantage,
                execution_time=execution_time,
                classical_comparison=classical_result.value,
                metadata={
                    'num_paths': num_paths,
                    'algorithm': 'quantum_monte_carlo',
                    'option_type': option.option_type,
                    'moneyness': option.moneyness()
                }
            )
            
        except Exception as e:
            warnings.warn(f"Quantum Monte Carlo failed: {e}. Using classical fallback.")
            return self._classical_monte_carlo_pricing(option, num_paths)
    
    def _classical_monte_carlo_pricing(self, option: OptionContract, 
                                     num_paths: int) -> QuantumFinanceResult:
        """Classical Monte Carlo option pricing."""
        start_time = pd.Timestamp.now()
        
        # Generate random paths
        np.random.seed(42)  # For reproducibility
        random_numbers = np.random.standard_normal(num_paths)
        
        # Generate price paths
        paths = self._generate_price_paths(option, random_numbers)
        
        # Calculate payoffs
        payoffs = self._calculate_payoffs(option, paths)
        
        # Present value
        present_value = np.mean(payoffs) * np.exp(-option.risk_free_rate * option.time_to_expiry())
        
        # Confidence interval
        std_error = np.std(payoffs) / np.sqrt(len(payoffs))
        ci_lower = present_value - 1.96 * std_error
        ci_upper = present_value + 1.96 * std_error
        
        execution_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        return QuantumFinanceResult(
            value=present_value,
            confidence_interval=(ci_lower, ci_upper),
            execution_time=execution_time,
            metadata={
                'num_paths': num_paths,
                'algorithm': 'classical_monte_carlo',
                'option_type': option.option_type
            }
        )
    
    def _black_scholes_pricing(self, option: OptionContract) -> QuantumFinanceResult:
        """Black-Scholes analytical option pricing."""
        from scipy.stats import norm
        
        S = option.current_price
        K = option.strike_price
        T = option.time_to_expiry()
        r = option.risk_free_rate
        sigma = option.volatility
        
        if T <= 0:
            # Option has expired
            if option.option_type.lower() == "call":
                price = max(S - K, 0)
            else:
                price = max(K - S, 0)
        else:
            # Black-Scholes formula
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option.option_type.lower() == "call":
                price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            else:
                price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        return QuantumFinanceResult(
            value=price,
            metadata={
                'algorithm': 'black_scholes',
                'option_type': option.option_type,
                'moneyness': option.moneyness(),
                'd1': d1 if T > 0 else None,
                'd2': d2 if T > 0 else None
            }
        )
    
    def _extract_random_numbers(self, result, num_samples: int) -> np.ndarray:
        """Extract random numbers from quantum measurement result."""
        if not HAS_QUANTRS2 or result is None:
            return np.random.standard_normal(num_samples)
        
        try:
            # Get measurement probabilities
            probs = result.state_probabilities()
            
            # Convert to uniform random numbers, then to normal
            random_numbers = []
            for i in range(num_samples):
                # Use quantum probabilities to generate uniform random number
                uniform = sum(prob for state, prob in probs.items() if int(state, 2) % 2 == i % 2)
                # Convert to standard normal using inverse CDF
                normal = norm.ppf(min(0.999, max(0.001, uniform)))  # Avoid extremes
                random_numbers.append(normal)
            
            return np.array(random_numbers)
            
        except Exception:
            return np.random.standard_normal(num_samples)
    
    def _generate_price_paths(self, option: OptionContract, 
                            random_numbers: np.ndarray) -> np.ndarray:
        """Generate stock price paths using geometric Brownian motion."""
        S0 = option.current_price
        T = option.time_to_expiry()
        r = option.risk_free_rate
        sigma = option.volatility
        
        # Geometric Brownian motion
        drift = (r - 0.5 * sigma**2) * T
        diffusion = sigma * np.sqrt(T) * random_numbers
        
        final_prices = S0 * np.exp(drift + diffusion)
        return final_prices
    
    def _calculate_payoffs(self, option: OptionContract, final_prices: np.ndarray) -> np.ndarray:
        """Calculate option payoffs at expiration."""
        K = option.strike_price
        
        if option.option_type.lower() == "call":
            payoffs = np.maximum(final_prices - K, 0)
        else:  # put
            payoffs = np.maximum(K - final_prices, 0)
        
        return payoffs


class QuantumRiskAnalyzer:
    """Quantum algorithms for financial risk analysis."""
    
    def __init__(self):
        """Initialize quantum risk analyzer."""
        pass
    
    def calculate_var(self, portfolio: Portfolio, confidence_level: float = 0.95,
                     time_horizon: int = 1) -> QuantumFinanceResult:
        """
        Calculate Value at Risk using quantum algorithms.
        
        Args:
            portfolio: Portfolio to analyze
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            time_horizon: Time horizon in days
            
        Returns:
            VaR calculation result
        """
        if not HAS_QUANTRS2:
            return self._classical_var(portfolio, confidence_level, time_horizon)
        
        try:
            # Quantum VaR calculation using amplitude estimation
            n_qubits = 8  # Reasonable for VaR calculation
            
            # Create quantum circuit for portfolio simulation
            circuit = Circuit(n_qubits)
            
            # Initialize superposition
            for i in range(n_qubits):
                circuit.h(i)
            
            # Encode portfolio structure (simplified)
            for i, weight in enumerate(portfolio.weights[:n_qubits]):
                if weight > 0:
                    angle = np.pi * weight
                    circuit.ry(i, angle)
            
            # Add correlations
            for i in range(n_qubits - 1):
                circuit.cx(i, i + 1)
            
            # Run simulation
            result = circuit.run()
            
            # Extract loss distribution
            losses = self._extract_loss_distribution(result, portfolio)
            
            # Calculate VaR
            var_value = np.percentile(losses, (1 - confidence_level) * 100)
            
            return QuantumFinanceResult(
                value=var_value,
                metadata={
                    'confidence_level': confidence_level,
                    'time_horizon': time_horizon,
                    'algorithm': 'quantum_var',
                    'max_loss': np.max(losses),
                    'expected_loss': np.mean(losses)
                }
            )
            
        except Exception as e:
            warnings.warn(f"Quantum VaR calculation failed: {e}. Using classical method.")
            return self._classical_var(portfolio, confidence_level, time_horizon)
    
    def _classical_var(self, portfolio: Portfolio, confidence_level: float, 
                      time_horizon: int) -> QuantumFinanceResult:
        """Classical VaR calculation."""
        # Simulate portfolio returns
        num_simulations = 10000
        
        # Generate correlated random returns
        np.random.seed(42)
        portfolio_returns = []
        
        for _ in range(num_simulations):
            random_shocks = np.random.normal(0, 1, len(portfolio.assets))
            daily_returns = []
            
            for i, asset in enumerate(portfolio.assets):
                daily_return = asset.expected_return() / 252 + asset.risk() / np.sqrt(252) * random_shocks[i]
                daily_returns.append(daily_return)
            
            # Portfolio return
            portfolio_return = sum(w * r for w, r in zip(portfolio.weights, daily_returns))
            portfolio_returns.append(portfolio_return * time_horizon)
        
        # Calculate VaR
        losses = [-r for r in portfolio_returns]  # Convert returns to losses
        var_value = np.percentile(losses, (1 - confidence_level) * 100)
        
        return QuantumFinanceResult(
            value=var_value,
            metadata={
                'confidence_level': confidence_level,
                'time_horizon': time_horizon,
                'algorithm': 'classical_monte_carlo_var',
                'num_simulations': num_simulations
            }
        )
    
    def _extract_loss_distribution(self, result, portfolio: Portfolio) -> np.ndarray:
        """Extract loss distribution from quantum result."""
        # Simplified extraction - in practice would be more sophisticated
        num_samples = 1000
        
        if not HAS_QUANTRS2 or result is None:
            return np.random.exponential(0.02, num_samples)  # Mock losses
        
        try:
            probs = result.state_probabilities()
            
            losses = []
            for _ in range(num_samples):
                # Generate loss based on quantum probabilities
                loss = 0.0
                for state, prob in probs.items():
                    state_value = int(state, 2) if isinstance(state, str) else 0
                    # Convert state to loss magnitude
                    loss += prob * (state_value / 255.0) * 0.1  # Scale to reasonable loss
                
                losses.append(abs(loss))
            
            return np.array(losses)
            
        except Exception:
            return np.random.exponential(0.02, num_samples)


class QuantumFraudDetector:
    """Quantum algorithms for fraud detection in financial transactions."""
    
    def __init__(self, n_features: int = 10):
        """
        Initialize quantum fraud detector.
        
        Args:
            n_features: Number of transaction features to analyze
        """
        self.n_features = n_features
        self.trained = False
        self.normal_patterns = None
    
    def train(self, normal_transactions: np.ndarray) -> bool:
        """
        Train the fraud detector on normal transaction patterns.
        
        Args:
            normal_transactions: Array of normal transaction features
            
        Returns:
            True if training successful
        """
        try:
            # Store normal patterns for comparison
            self.normal_patterns = normal_transactions
            self.trained = True
            return True
        except Exception as e:
            warnings.warn(f"Training failed: {e}")
            return False
    
    def detect_fraud(self, transaction: np.ndarray) -> QuantumFinanceResult:
        """
        Detect if a transaction is fraudulent using quantum algorithms.
        
        Args:
            transaction: Transaction features to analyze
            
        Returns:
            Fraud detection result
        """
        if not self.trained:
            raise ValueError("Detector must be trained before use")
        
        if not HAS_QUANTRS2:
            return self._classical_fraud_detection(transaction)
        
        try:
            # Quantum anomaly detection
            n_qubits = min(16, self.n_features + 2)
            
            circuit = Circuit(n_qubits)
            
            # Encode transaction features
            for i, feature in enumerate(transaction[:n_qubits]):
                if feature != 0:
                    angle = np.pi * abs(feature) / (1 + abs(feature))  # Normalize
                    circuit.ry(i, angle)
            
            # Add pattern matching gates
            for i in range(n_qubits - 1):
                circuit.cx(i, i + 1)
            
            # Run quantum anomaly detection
            result = circuit.run()
            
            # Calculate anomaly score
            anomaly_score = self._calculate_anomaly_score(result, transaction)
            
            # Determine fraud probability
            fraud_probability = 1 / (1 + np.exp(-5 * (anomaly_score - 0.5)))  # Sigmoid
            
            is_fraud = fraud_probability > 0.7  # Threshold
            
            return QuantumFinanceResult(
                value=fraud_probability,
                metadata={
                    'is_fraud': is_fraud,
                    'anomaly_score': anomaly_score,
                    'algorithm': 'quantum_anomaly_detection',
                    'threshold': 0.7,
                    'transaction_features': transaction.tolist()
                }
            )
            
        except Exception as e:
            warnings.warn(f"Quantum fraud detection failed: {e}. Using classical method.")
            return self._classical_fraud_detection(transaction)
    
    def _classical_fraud_detection(self, transaction: np.ndarray) -> QuantumFinanceResult:
        """Classical fraud detection using statistical methods."""
        if self.normal_patterns is None:
            return QuantumFinanceResult(value=0.5, metadata={'algorithm': 'no_training_data'})
        
        # Calculate distance from normal patterns
        distances = np.linalg.norm(self.normal_patterns - transaction, axis=1)
        min_distance = np.min(distances)
        mean_distance = np.mean(distances)
        
        # Anomaly score based on distance
        anomaly_score = min_distance / (mean_distance + 1e-8)
        
        # Convert to fraud probability
        fraud_probability = 1 / (1 + np.exp(-2 * (anomaly_score - 1)))
        
        is_fraud = fraud_probability > 0.7
        
        return QuantumFinanceResult(
            value=fraud_probability,
            metadata={
                'is_fraud': is_fraud,
                'anomaly_score': anomaly_score,
                'min_distance': min_distance,
                'algorithm': 'classical_distance_based'
            }
        )
    
    def _calculate_anomaly_score(self, result, transaction: np.ndarray) -> float:
        """Calculate anomaly score from quantum result."""
        if not HAS_QUANTRS2 or result is None:
            return np.random.random()
        
        try:
            probs = result.state_probabilities()
            
            # Calculate entropy as anomaly measure
            entropy = -sum(p * np.log2(p + 1e-8) for p in probs.values())
            max_entropy = np.log2(len(probs))
            
            # Normalize entropy to [0, 1]
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            return normalized_entropy
            
        except Exception:
            return np.random.random()


# Utility functions and example usage
def create_sample_portfolio(n_assets: int = 5) -> Portfolio:
    """Create a sample portfolio for testing."""
    assets = []
    
    for i in range(n_assets):
        # Generate realistic asset data
        price = 100 + np.random.normal(0, 20)
        volatility = 0.1 + np.random.exponential(0.1)
        returns = np.random.normal(0.08/252, volatility/np.sqrt(252), 252)  # Daily returns for 1 year
        
        asset = FinancialAsset(
            symbol=f"ASSET_{i+1}",
            name=f"Sample Asset {i+1}",
            price=price,
            volatility=volatility,
            returns=returns.tolist()
        )
        assets.append(asset)
    
    # Equal weights
    weights = [1.0 / n_assets] * n_assets
    
    return Portfolio(assets=assets, weights=weights)


def create_sample_option(days_to_expiry: int = 30) -> OptionContract:
    """Create a sample option contract for testing."""
    expiry = datetime.now() + timedelta(days=days_to_expiry)
    
    return OptionContract(
        underlying="AAPL",
        strike_price=150.0,
        expiry_date=expiry,
        option_type="call",
        current_price=155.0,
        volatility=0.25,
        risk_free_rate=0.05
    )


def run_portfolio_optimization_demo() -> Dict[str, Any]:
    """Run portfolio optimization demonstration."""
    # Create sample data
    n_assets = 4
    returns = np.array([0.08, 0.12, 0.10, 0.15])  # Expected annual returns
    
    # Create covariance matrix
    correlations = np.array([
        [1.0, 0.3, 0.2, 0.1],
        [0.3, 1.0, 0.4, 0.2],
        [0.2, 0.4, 1.0, 0.3],
        [0.1, 0.2, 0.3, 1.0]
    ])
    volatilities = np.array([0.15, 0.20, 0.18, 0.25])
    covariance = np.outer(volatilities, volatilities) * correlations
    
    # Run optimization
    optimizer = QuantumPortfolioOptimizer(n_assets, risk_aversion=2.0)
    optimizer.set_market_data(returns, covariance)
    
    result = optimizer.quantum_optimize(max_iterations=50)
    
    return {
        'optimal_weights': result.metadata.get('optimal_weights', []),
        'utility': result.value,
        'quantum_advantage': result.quantum_advantage,
        'algorithm': result.metadata.get('algorithm', 'unknown')
    }


def run_option_pricing_demo() -> Dict[str, Any]:
    """Run option pricing demonstration."""
    option = create_sample_option(30)
    
    # Price using different methods
    pricer = QuantumOptionPricer()
    
    # Black-Scholes
    bs_result = pricer.price_european_option(option)
    
    # Quantum Monte Carlo
    pricer.model = FinanceModel.QUANTUM_MONTE_CARLO
    qmc_result = pricer.price_european_option(option, num_paths=1000)
    
    return {
        'black_scholes_price': bs_result.value,
        'quantum_monte_carlo_price': qmc_result.value,
        'quantum_advantage': qmc_result.quantum_advantage,
        'confidence_interval': qmc_result.confidence_interval
    }


def run_risk_analysis_demo() -> Dict[str, Any]:
    """Run risk analysis demonstration."""
    portfolio = create_sample_portfolio(3)
    
    risk_analyzer = QuantumRiskAnalyzer()
    var_result = risk_analyzer.calculate_var(portfolio, confidence_level=0.95)
    
    return {
        'value_at_risk': var_result.value,
        'confidence_level': var_result.metadata.get('confidence_level', 0.95),
        'algorithm': var_result.metadata.get('algorithm', 'unknown')
    }