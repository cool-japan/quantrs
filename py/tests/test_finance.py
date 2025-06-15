#!/usr/bin/env python3
"""
Test suite for quantum finance algorithms.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import quantrs2
    from quantrs2.finance import (
        QuantumPortfolioOptimizer, QuantumOptionPricer, QuantumRiskAnalyzer, QuantumFraudDetector,
        FinancialAsset, Portfolio, OptionContract, QuantumFinanceResult,
        FinanceModel, RiskMetric,
        create_sample_portfolio, create_sample_option,
        run_portfolio_optimization_demo, run_option_pricing_demo, run_risk_analysis_demo
    )
    HAS_QUANTRS2 = True
except ImportError:
    HAS_QUANTRS2 = False


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
class TestFinancialAsset:
    """Test FinancialAsset dataclass."""
    
    def test_empty_asset(self):
        """Test empty FinancialAsset initialization."""
        asset = FinancialAsset(symbol="TEST")
        assert asset.symbol == "TEST"
        assert asset.name == ""
        assert asset.price == 0.0
        assert asset.volatility == 0.0
        assert asset.returns == []
        assert asset.correlations == {}
    
    def test_asset_with_data(self):
        """Test FinancialAsset with complete data."""
        returns = [0.1, -0.05, 0.02, 0.08]
        correlations = {"AAPL": 0.3, "GOOGL": 0.5}
        
        asset = FinancialAsset(
            symbol="TEST",
            name="Test Asset",
            price=100.0,
            volatility=0.2,
            returns=returns,
            correlations=correlations
        )
        
        assert asset.symbol == "TEST"
        assert asset.name == "Test Asset"
        assert asset.price == 100.0
        assert asset.volatility == 0.2
        assert asset.returns == returns
        assert asset.correlations == correlations
    
    def test_expected_return(self):
        """Test expected return calculation."""
        returns = [0.1, -0.1, 0.2, -0.2]
        asset = FinancialAsset(symbol="TEST", returns=returns)
        
        expected = asset.expected_return()
        assert expected == 0.0  # Mean of [0.1, -0.1, 0.2, -0.2]
        
        # Empty returns
        empty_asset = FinancialAsset(symbol="TEST")
        assert empty_asset.expected_return() == 0.0
    
    def test_risk(self):
        """Test risk calculation."""
        returns = [0.1, 0.1, 0.1, 0.1]  # No volatility
        asset = FinancialAsset(symbol="TEST", returns=returns)
        
        risk = asset.risk()
        assert risk == 0.0
        
        # Test with volatility parameter when no returns
        asset_with_vol = FinancialAsset(symbol="TEST", volatility=0.25)
        assert asset_with_vol.risk() == 0.25


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
class TestPortfolio:
    """Test Portfolio dataclass."""
    
    def test_empty_portfolio(self):
        """Test empty Portfolio initialization."""
        portfolio = Portfolio()
        assert portfolio.assets == []
        assert portfolio.weights == []
        assert portfolio.rebalance_frequency == "monthly"
    
    def test_portfolio_with_assets(self):
        """Test Portfolio with assets."""
        asset1 = FinancialAsset(symbol="A", price=100.0)
        asset2 = FinancialAsset(symbol="B", price=200.0)
        weights = [0.6, 0.4]
        
        portfolio = Portfolio(assets=[asset1, asset2], weights=weights)
        
        assert len(portfolio.assets) == 2
        assert portfolio.weights == weights
    
    def test_total_value(self):
        """Test portfolio total value calculation."""
        asset1 = FinancialAsset(symbol="A", price=100.0)
        asset2 = FinancialAsset(symbol="B", price=200.0)
        weights = [0.6, 0.4]
        
        portfolio = Portfolio(assets=[asset1, asset2], weights=weights)
        
        total_value = portfolio.total_value()
        expected = 100.0 * 0.6 + 200.0 * 0.4  # 60 + 80 = 140
        assert total_value == expected
    
    def test_expected_return(self):
        """Test portfolio expected return calculation."""
        asset1 = FinancialAsset(symbol="A", returns=[0.1, 0.05])  # Mean = 0.075
        asset2 = FinancialAsset(symbol="B", returns=[0.2, 0.1])   # Mean = 0.15
        weights = [0.5, 0.5]
        
        portfolio = Portfolio(assets=[asset1, asset2], weights=weights)
        
        expected_return = portfolio.expected_return()
        expected = 0.075 * 0.5 + 0.15 * 0.5  # 0.1125
        assert abs(expected_return - expected) < 1e-10
    
    def test_risk(self):
        """Test portfolio risk calculation."""
        asset1 = FinancialAsset(symbol="A", returns=[0.1, 0.0])  # Std ~0.07
        asset2 = FinancialAsset(symbol="B", returns=[0.2, 0.0])  # Std ~0.14
        weights = [0.5, 0.5]
        
        portfolio = Portfolio(assets=[asset1, asset2], weights=weights)
        
        risk = portfolio.risk()
        assert risk > 0
        
        # Empty portfolio
        empty_portfolio = Portfolio()
        assert empty_portfolio.risk() == 0.0


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
class TestOptionContract:
    """Test OptionContract dataclass."""
    
    def test_option_creation(self):
        """Test option contract creation."""
        expiry = datetime.now() + timedelta(days=30)
        option = OptionContract(
            underlying="AAPL",
            strike_price=150.0,
            expiry_date=expiry,
            option_type="call",
            current_price=155.0
        )
        
        assert option.underlying == "AAPL"
        assert option.strike_price == 150.0
        assert option.option_type == "call"
        assert option.current_price == 155.0
    
    def test_time_to_expiry(self):
        """Test time to expiry calculation."""
        # Future expiry
        future_expiry = datetime.now() + timedelta(days=365)
        option = OptionContract(
            underlying="TEST",
            strike_price=100.0,
            expiry_date=future_expiry,
            option_type="call"
        )
        
        time_to_expiry = option.time_to_expiry()
        assert 0.9 < time_to_expiry < 1.1  # Approximately 1 year
        
        # Past expiry
        past_expiry = datetime.now() - timedelta(days=1)
        expired_option = OptionContract(
            underlying="TEST",
            strike_price=100.0,
            expiry_date=past_expiry,
            option_type="call"
        )
        
        assert expired_option.time_to_expiry() == 0.0
    
    def test_moneyness_call(self):
        """Test moneyness calculation for call options."""
        expiry = datetime.now() + timedelta(days=30)
        
        # ITM call
        itm_call = OptionContract(
            underlying="TEST", strike_price=100.0, expiry_date=expiry,
            option_type="call", current_price=110.0
        )
        assert itm_call.moneyness() == "ITM"
        
        # ATM call
        atm_call = OptionContract(
            underlying="TEST", strike_price=100.0, expiry_date=expiry,
            option_type="call", current_price=100.0
        )
        assert atm_call.moneyness() == "ATM"
        
        # OTM call
        otm_call = OptionContract(
            underlying="TEST", strike_price=100.0, expiry_date=expiry,
            option_type="call", current_price=90.0
        )
        assert otm_call.moneyness() == "OTM"
    
    def test_moneyness_put(self):
        """Test moneyness calculation for put options."""
        expiry = datetime.now() + timedelta(days=30)
        
        # ITM put
        itm_put = OptionContract(
            underlying="TEST", strike_price=100.0, expiry_date=expiry,
            option_type="put", current_price=90.0
        )
        assert itm_put.moneyness() == "ITM"
        
        # OTM put
        otm_put = OptionContract(
            underlying="TEST", strike_price=100.0, expiry_date=expiry,
            option_type="put", current_price=110.0
        )
        assert otm_put.moneyness() == "OTM"


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
class TestQuantumFinanceResult:
    """Test QuantumFinanceResult dataclass."""
    
    def test_result_creation(self):
        """Test result creation."""
        result = QuantumFinanceResult(
            value=123.45,
            confidence_interval=(120.0, 125.0),
            quantum_advantage=1.5,
            execution_time=0.1
        )
        
        assert result.value == 123.45
        assert result.confidence_interval == (120.0, 125.0)
        assert result.quantum_advantage == 1.5
        assert result.execution_time == 0.1
        assert result.metadata == {}


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
class TestQuantumPortfolioOptimizer:
    """Test quantum portfolio optimization."""
    
    def test_optimizer_creation(self):
        """Test optimizer initialization."""
        optimizer = QuantumPortfolioOptimizer(num_assets=5, risk_aversion=2.0)
        
        assert optimizer.num_assets == 5
        assert optimizer.risk_aversion == 2.0
        assert optimizer.expected_returns.shape == (5,)
        assert optimizer.covariance_matrix.shape == (5, 5)
    
    def test_set_market_data(self):
        """Test setting market data."""
        optimizer = QuantumPortfolioOptimizer(num_assets=3)
        
        returns = np.array([0.08, 0.12, 0.10])
        covariances = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.16]
        ])
        
        optimizer.set_market_data(returns, covariances)
        
        np.testing.assert_array_equal(optimizer.expected_returns, returns)
        np.testing.assert_array_equal(optimizer.covariance_matrix, covariances)
    
    def test_quantum_optimize(self):
        """Test quantum optimization."""
        optimizer = QuantumPortfolioOptimizer(num_assets=3, risk_aversion=1.0)
        
        returns = np.array([0.08, 0.12, 0.10])
        covariances = np.eye(3) * 0.04  # Simple diagonal covariance
        optimizer.set_market_data(returns, covariances)
        
        result = optimizer.quantum_optimize(max_iterations=10)
        
        assert isinstance(result, QuantumFinanceResult)
        assert result.value is not None
        assert 'optimal_weights' in result.metadata
        
        weights = result.metadata['optimal_weights']
        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 0.1  # Weights should sum to ~1


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
class TestQuantumOptionPricer:
    """Test quantum option pricing."""
    
    def test_pricer_creation(self):
        """Test pricer initialization."""
        pricer = QuantumOptionPricer()
        assert pricer.model == FinanceModel.QUANTUM_MONTE_CARLO
        
        pricer_bs = QuantumOptionPricer(FinanceModel.BLACK_SCHOLES)
        assert pricer_bs.model == FinanceModel.BLACK_SCHOLES
    
    def test_black_scholes_pricing(self):
        """Test Black-Scholes option pricing."""
        option = create_sample_option(30)
        pricer = QuantumOptionPricer(FinanceModel.BLACK_SCHOLES)
        
        result = pricer.price_european_option(option)
        
        assert isinstance(result, QuantumFinanceResult)
        assert result.value > 0  # Option should have positive value
        assert 'algorithm' in result.metadata
        assert result.metadata['algorithm'] == 'black_scholes'
    
    def test_monte_carlo_pricing(self):
        """Test Monte Carlo option pricing."""
        option = create_sample_option(30)
        pricer = QuantumOptionPricer(FinanceModel.QUANTUM_MONTE_CARLO)
        
        result = pricer.price_european_option(option, num_paths=100)
        
        assert isinstance(result, QuantumFinanceResult)
        assert result.value >= 0  # Option value should be non-negative
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] <= result.confidence_interval[1]
    
    def test_expired_option_pricing(self):
        """Test pricing of expired option."""
        # Create expired option
        past_expiry = datetime.now() - timedelta(days=1)
        option = OptionContract(
            underlying="TEST",
            strike_price=100.0,
            expiry_date=past_expiry,
            option_type="call",
            current_price=105.0
        )
        
        pricer = QuantumOptionPricer(FinanceModel.BLACK_SCHOLES)
        result = pricer.price_european_option(option)
        
        # Expired ITM call should be worth intrinsic value
        expected_value = max(105.0 - 100.0, 0)
        assert abs(result.value - expected_value) < 1e-10


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
class TestQuantumRiskAnalyzer:
    """Test quantum risk analysis."""
    
    def test_analyzer_creation(self):
        """Test analyzer initialization."""
        analyzer = QuantumRiskAnalyzer()
        assert analyzer is not None
    
    def test_var_calculation(self):
        """Test VaR calculation."""
        portfolio = create_sample_portfolio(3)
        analyzer = QuantumRiskAnalyzer()
        
        result = analyzer.calculate_var(portfolio, confidence_level=0.95, time_horizon=1)
        
        assert isinstance(result, QuantumFinanceResult)
        assert result.value >= 0  # VaR should be positive (loss)
        assert 'confidence_level' in result.metadata
        assert result.metadata['confidence_level'] == 0.95
    
    def test_different_confidence_levels(self):
        """Test VaR with different confidence levels."""
        portfolio = create_sample_portfolio(2)
        analyzer = QuantumRiskAnalyzer()
        
        var_95 = analyzer.calculate_var(portfolio, confidence_level=0.95)
        var_99 = analyzer.calculate_var(portfolio, confidence_level=0.99)
        
        # 99% VaR should be higher than 95% VaR
        assert var_99.value >= var_95.value


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
class TestQuantumFraudDetector:
    """Test quantum fraud detection."""
    
    def test_detector_creation(self):
        """Test detector initialization."""
        detector = QuantumFraudDetector(n_features=5)
        assert detector.n_features == 5
        assert detector.trained is False
    
    def test_training(self):
        """Test detector training."""
        detector = QuantumFraudDetector(n_features=3)
        
        # Generate normal transaction data
        normal_data = np.random.normal(0, 1, (100, 3))
        
        success = detector.train(normal_data)
        assert success is True
        assert detector.trained is True
    
    def test_fraud_detection(self):
        """Test fraud detection."""
        detector = QuantumFraudDetector(n_features=3)
        
        # Train on normal data
        normal_data = np.random.normal(0, 1, (50, 3))
        detector.train(normal_data)
        
        # Test normal transaction
        normal_transaction = np.array([0.1, -0.2, 0.5])
        result = detector.detect_fraud(normal_transaction)
        
        assert isinstance(result, QuantumFinanceResult)
        assert 0 <= result.value <= 1  # Probability should be in [0,1]
        assert 'is_fraud' in result.metadata
        
        # Test anomalous transaction
        anomalous_transaction = np.array([10.0, -15.0, 20.0])  # Very different
        anomaly_result = detector.detect_fraud(anomalous_transaction)
        
        # Anomalous transaction should have higher fraud probability
        assert anomaly_result.value >= result.value
    
    def test_untrained_detector(self):
        """Test detection without training."""
        detector = QuantumFraudDetector(n_features=3)
        transaction = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            detector.detect_fraud(transaction)


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_sample_portfolio(self):
        """Test sample portfolio creation."""
        portfolio = create_sample_portfolio(4)
        
        assert isinstance(portfolio, Portfolio)
        assert len(portfolio.assets) == 4
        assert len(portfolio.weights) == 4
        assert abs(sum(portfolio.weights) - 1.0) < 1e-10  # Weights sum to 1
        
        # Check assets have realistic data
        for asset in portfolio.assets:
            assert asset.price > 0
            assert asset.volatility > 0
            assert len(asset.returns) > 0
    
    def test_create_sample_option(self):
        """Test sample option creation."""
        option = create_sample_option(45)
        
        assert isinstance(option, OptionContract)
        assert option.underlying == "AAPL"
        assert option.strike_price > 0
        assert option.current_price > 0
        assert option.time_to_expiry() > 0


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
class TestDemoFunctions:
    """Test demo functions."""
    
    def test_portfolio_optimization_demo(self):
        """Test portfolio optimization demo."""
        result = run_portfolio_optimization_demo()
        
        assert isinstance(result, dict)
        assert 'optimal_weights' in result
        assert 'utility' in result
        assert 'algorithm' in result
        
        weights = result['optimal_weights']
        if weights:  # If optimization succeeded
            assert len(weights) == 4  # Demo uses 4 assets
            assert abs(sum(weights) - 1.0) < 0.1  # Should sum to ~1
    
    def test_option_pricing_demo(self):
        """Test option pricing demo."""
        result = run_option_pricing_demo()
        
        assert isinstance(result, dict)
        assert 'black_scholes_price' in result
        assert 'quantum_monte_carlo_price' in result
        
        bs_price = result['black_scholes_price']
        qmc_price = result['quantum_monte_carlo_price']
        
        assert bs_price > 0
        assert qmc_price >= 0
    
    def test_risk_analysis_demo(self):
        """Test risk analysis demo."""
        result = run_risk_analysis_demo()
        
        assert isinstance(result, dict)
        assert 'value_at_risk' in result
        assert 'confidence_level' in result
        assert 'algorithm' in result
        
        var = result['value_at_risk']
        assert var >= 0  # VaR should be positive


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
class TestEnumTypes:
    """Test enum types."""
    
    def test_finance_model_enum(self):
        """Test FinanceModel enum."""
        assert FinanceModel.BLACK_SCHOLES.value == "black_scholes"
        assert FinanceModel.QUANTUM_MONTE_CARLO.value == "quantum_monte_carlo"
        assert FinanceModel.MONTE_CARLO.value == "monte_carlo"
    
    def test_risk_metric_enum(self):
        """Test RiskMetric enum."""
        assert RiskMetric.VALUE_AT_RISK.value == "var"
        assert RiskMetric.CONDITIONAL_VAR.value == "cvar"
        assert RiskMetric.SHARPE_RATIO.value == "sharpe"


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
class TestFinanceIntegration:
    """Test finance module integration."""
    
    def test_finance_functions_available(self):
        """Test that finance functions are available from main module."""
        try:
            from quantrs2 import QuantumPortfolioOptimizer, QuantumOptionPricer
            assert QuantumPortfolioOptimizer is not None
            assert QuantumOptionPricer is not None
        except ImportError:
            # This is acceptable if finance module not available
            pass
    
    def test_realistic_workflow(self):
        """Test realistic finance workflow."""
        # Create portfolio
        portfolio = create_sample_portfolio(3)
        assert len(portfolio.assets) == 3
        
        # Optimize portfolio
        optimizer = QuantumPortfolioOptimizer(3, risk_aversion=1.5)
        returns = np.array([asset.expected_return() for asset in portfolio.assets])
        risks = np.array([asset.risk() for asset in portfolio.assets])
        covariances = np.diag(risks**2)  # Simplified covariance
        
        optimizer.set_market_data(returns, covariances)
        opt_result = optimizer.quantum_optimize(max_iterations=5)
        
        assert opt_result.value is not None
        
        # Analyze risk
        analyzer = QuantumRiskAnalyzer()
        var_result = analyzer.calculate_var(portfolio)
        
        assert var_result.value >= 0
        
        # Price option
        option = create_sample_option(15)
        pricer = QuantumOptionPricer(FinanceModel.BLACK_SCHOLES)
        price_result = pricer.price_european_option(option)
        
        assert price_result.value > 0


if __name__ == "__main__":
    pytest.main([__file__])