"""Trading Environment for Agent Interaction."""

import numpy as np
from loguru import logger
from config.settings import (
    INITIAL_BALANCE, MAX_ACCOUNT_LOSS_PERCENT, DAILY_DRAWDOWN_LIMIT,
    MAX_POSITION_SIZE, SLIPPAGE_PERCENT, COMMISSION_PERCENT,
    STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT
)


class TradingEnvironment:
    """Simulated trading environment."""
    
    def __init__(self, data, initial_balance=INITIAL_BALANCE):
        """Initialize trading environment.
        
        Args:
            data: Historical price data
            initial_balance: Starting account balance
        """
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.portfolio_value = initial_balance
        
        # Risk limits
        self.max_loss = (MAX_ACCOUNT_LOSS_PERCENT / 100) * initial_balance
        self.daily_drawdown_limit = (DAILY_DRAWDOWN_LIMIT / 100) * initial_balance
        
        # Position tracking
        self.position = 0  # 0=no position, 1=long, -1=short
        self.entry_price = 0
        self.position_size = 0
        
        # Trading history
        self.trades = []
        self.equity_history = [initial_balance]
        self.daily_returns = []
        
        # Current state
        self.current_step = 0
        self.day_open_balance = initial_balance
        
        logger.info(f"Trading environment initialized - Balance: ${initial_balance}")
    
    def reset(self):
        """Reset environment to initial state."""
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.position_size = 0
        self.trades = []
        self.equity_history = [self.initial_balance]
        self.daily_returns = []
        self.current_step = 0
        self.day_open_balance = self.initial_balance
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state vector."""
        if self.current_step < len(self.data):
            price = self.data[self.current_step]
        else:
            price = self.data[-1]
        
        # Normalize state
        state = np.array([
            self.balance / self.initial_balance,
            self.portfolio_value / self.initial_balance,
            price / self.data[0] if self.data[0] != 0 else 1,
            1 if self.position > 0 else (-1 if self.position < 0 else 0),
        ])
        
        return state
    
    def _calculate_reward(self, old_portfolio, new_portfolio):
        """Calculate reward for the action."""
        # Reward based on portfolio change
        profit_loss = new_portfolio - old_portfolio
        reward = profit_loss / self.initial_balance * 100
        
        # Penalty for risk
        if new_portfolio < self.initial_balance * 0.9:
            reward -= 10  # Large penalty for exceeding loss limit
        
        return reward
    
    def step(self, action):
        """Execute one step of trading.
        
        Args:
            action: 0=Buy, 1=Hold, 2=Sell
            
        Returns:
            state, reward, done, info
        """
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True, {}
        
        current_price = self.data[self.current_step]
        old_portfolio = self.portfolio_value
        
        # Apply slippage
        if action == 0:  # Buy
            current_price *= (1 + SLIPPAGE_PERCENT / 100)
        elif action == 2:  # Sell
            current_price *= (1 - SLIPPAGE_PERCENT / 100)
        
        # Process action
        if action == 0 and self.position == 0:  # Buy
            position_size = (self.balance * MAX_POSITION_SIZE) / current_price
            self.position = 1
            self.entry_price = current_price
            self.position_size = position_size
            cost = position_size * current_price * (1 + COMMISSION_PERCENT / 100)
            self.balance -= cost
            
        elif action == 2 and self.position > 0:  # Sell
            proceeds = self.position_size * current_price * (1 - COMMISSION_PERCENT / 100)
            self.balance += proceeds
            self.trades.append({
                'entry': self.entry_price,
                'exit': current_price,
                'size': self.position_size,
                'profit': proceeds - (self.position_size * self.entry_price)
            })
            self.position = 0
            self.position_size = 0
        
        # Update portfolio value
        if self.position > 0:
            self.portfolio_value = self.balance + self.position_size * current_price
        else:
            self.portfolio_value = self.balance
        
        # Calculate reward
        reward = self._calculate_reward(old_portfolio, self.portfolio_value)
        
        # Check termination conditions
        done = False
        info = {}
        
        # Check max loss
        if self.portfolio_value < self.initial_balance * (1 - MAX_ACCOUNT_LOSS_PERCENT / 100):
            done = True
            info['reason'] = 'Max loss exceeded'
            reward -= 50
        
        # Check daily drawdown
        daily_return = (self.portfolio_value - self.day_open_balance) / self.day_open_balance
        if daily_return < -(DAILY_DRAWDOWN_LIMIT / 100):
            done = True
            info['reason'] = 'Daily drawdown exceeded'
            reward -= 30
        
        self.current_step += 1
        self.equity_history.append(self.portfolio_value)
        
        return self._get_state(), reward, done, info
    
    def get_metrics(self):
        """Get trading performance metrics."""
        total_profit = self.portfolio_value - self.initial_balance
        total_return = (total_profit / self.initial_balance) * 100
        max_equity = max(self.equity_history)
        max_drawdown = (max_equity - min(self.equity_history)) / max_equity * 100
        
        win_trades = sum(1 for t in self.trades if t['profit'] > 0)
        total_trades = len(self.trades)
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_profit': total_profit,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'final_balance': self.portfolio_value
        }
