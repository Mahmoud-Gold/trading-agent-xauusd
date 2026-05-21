"""Backtesting Engine."""

import numpy as np
import pandas as pd
from loguru import logger
from config.settings import BACKTEST_INITIAL_BALANCE


class Backtester:
    """Backtest trading strategy on historical data."""
    
    def __init__(self, agent, historical_data):
        """Initialize backtester.
        
        Args:
            agent: Trained agent
            historical_data: Historical OHLCV data
        """
        self.agent = agent
        self.data = historical_data
        self.results = None
        
        logger.info(f"Backtester initialized with {len(historical_data)} data points")
    
    def run(self, initial_balance=BACKTEST_INITIAL_BALANCE):
        """Run backtest.
        
        Args:
            initial_balance: Starting balance
            
        Returns:
            Backtest results
        """
        logger.info(f"Starting backtest with initial balance: ${initial_balance}")
        
        balance = initial_balance
        portfolio_value = initial_balance
        position = 0
        entry_price = 0
        
        trades = []
        equity_curve = [initial_balance]
        returns = []
        
        for i in range(len(self.data) - 1):
            current_price = self.data[i]
            next_price = self.data[i + 1]
            
            # Get action from agent
            state = np.array([balance / initial_balance, portfolio_value / initial_balance])
            action = self.agent.act(state, training=False)
            
            # Execute trade
            if action == 0 and position == 0:  # Buy
                position = 1
                entry_price = current_price
            elif action == 2 and position > 0:  # Sell
                profit = (next_price - entry_price) * 100  # Per unit
                trades.append({
                    'entry': entry_price,
                    'exit': next_price,
                    'profit': profit
                })
                balance += profit
                position = 0
            
            # Update portfolio
            if position > 0:
                portfolio_value = balance + (next_price - entry_price) * 100
            else:
                portfolio_value = balance
            
            equity_curve.append(portfolio_value)
            daily_return = (portfolio_value - equity_curve[-2]) / equity_curve[-2] if i > 0 else 0
            returns.append(daily_return)
        
        # Calculate metrics
        self.results = self._calculate_metrics(
            equity_curve, trades, initial_balance, returns
        )
        
        logger.info(f"Backtest completed - Final Balance: ${self.results['final_balance']:.2f}")
        
        return self.results
    
    @staticmethod
    def _calculate_metrics(equity_curve, trades, initial_balance, returns):
        """Calculate backtest metrics."""
        final_balance = equity_curve[-1]
        total_return = (final_balance - initial_balance) / initial_balance * 100
        
        # Maximum Drawdown
        peak = np.max(equity_curve)
        trough = np.min(equity_curve)
        max_drawdown = (trough - peak) / peak * 100
        
        # Win rate
        winning_trades = sum(1 for t in trades if t['profit'] > 0)
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Sharpe Ratio
        returns_array = np.array(returns)
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'final_balance': final_balance,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades,
            'equity_curve': equity_curve
        }
    
    def get_results(self):
        """Get backtest results."""
        return self.results
