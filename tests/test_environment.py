"""Unit tests for Trading Environment."""

import pytest
import numpy as np
from src.environment.trading_env import TradingEnvironment
from config.settings import INITIAL_BALANCE


class TestTradingEnvironment:
    """Test Trading Environment."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create mock price data
        self.prices = np.random.uniform(2000, 2100, 1000)
        self.env = TradingEnvironment(self.prices, INITIAL_BALANCE)
    
    def test_env_initialization(self):
        """Test environment initializes correctly."""
        assert self.env.balance == INITIAL_BALANCE
        assert self.env.portfolio_value == INITIAL_BALANCE
        assert self.env.position == 0
    
    def test_env_reset(self):
        """Test environment reset."""
        state = self.env.reset()
        
        assert self.env.balance == INITIAL_BALANCE
        assert self.env.portfolio_value == INITIAL_BALANCE
        assert self.env.position == 0
        assert state is not None
    
    def test_env_step(self):
        """Test environment step."""
        state = self.env.reset()
        action = 0  # Buy
        
        next_state, reward, done, info = self.env.step(action)
        
        assert next_state is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_env_metrics(self):
        """Test environment metrics."""
        self.env.reset()
        metrics = self.env.get_metrics()
        
        assert 'total_profit' in metrics
        assert 'total_return' in metrics
        assert 'max_drawdown' in metrics
        assert 'total_trades' in metrics
        assert 'win_rate' in metrics
