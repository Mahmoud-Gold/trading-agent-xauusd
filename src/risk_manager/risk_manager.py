"""Risk Management Module."""

import numpy as np
from loguru import logger
from config.settings import (
    MAX_ACCOUNT_LOSS_PERCENT, DAILY_DRAWDOWN_LIMIT,
    MAX_POSITION_SIZE, MIN_POSITION_SIZE
)


class RiskManager:
    """Manage trading risks and position sizing."""
    
    def __init__(self, initial_balance):
        """Initialize risk manager.
        
        Args:
            initial_balance: Initial account balance
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.daily_balance = initial_balance
        self.max_loss = (MAX_ACCOUNT_LOSS_PERCENT / 100) * initial_balance
        self.daily_loss_limit = (DAILY_DRAWDOWN_LIMIT / 100) * initial_balance
        
        self.trades_executed = []
        self.daily_trades = []
        
        logger.info(f"RiskManager initialized - Max Loss: ${self.max_loss:.2f}, Daily Limit: ${self.daily_loss_limit:.2f}")
    
    def calculate_position_size(self, current_price, volatility, risk_percent=2.0):
        """Calculate optimal position size based on volatility.
        
        Args:
            current_price: Current market price
            volatility: Price volatility
            risk_percent: Risk percentage per trade
            
        Returns:
            Position size (quantity)
        """
        # Risk amount
        risk_amount = self.current_balance * (risk_percent / 100)
        
        # Position size based on volatility
        position_size = risk_amount / (current_price * volatility)
        
        # Apply limits
        max_size = self.current_balance * (MAX_POSITION_SIZE / 100) / current_price
        min_size = self.current_balance * (MIN_POSITION_SIZE / 100) / current_price
        
        position_size = np.clip(position_size, min_size, max_size)
        
        return position_size
    
    def can_trade(self, action):
        """Check if trade is allowed.
        
        Args:
            action: Trade action (0=Buy, 1=Hold, 2=Sell)
            
        Returns:
            Boolean indicating if trade is allowed
        """
        # Check account loss limit
        loss = self.initial_balance - self.current_balance
        if loss >= self.max_loss:
            logger.warning(f"Trading blocked - Max loss exceeded: ${loss:.2f}")
            return False
        
        # Check daily drawdown limit
        daily_loss = self.daily_balance - self.current_balance
        if daily_loss >= self.daily_loss_limit:
            logger.warning(f"Trading blocked - Daily limit exceeded: ${daily_loss:.2f}")
            return False
        
        return True
    
    def record_trade(self, entry_price, exit_price, quantity, trade_type):
        """Record executed trade.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position quantity
            trade_type: 'long' or 'short'
        """
        profit = quantity * (exit_price - entry_price) if trade_type == 'long' else quantity * (entry_price - exit_price)
        
        trade = {
            'entry': entry_price,
            'exit': exit_price,
            'quantity': quantity,
            'type': trade_type,
            'profit': profit,
            'profit_percent': (profit / (quantity * entry_price)) * 100
        }
        
        self.trades_executed.append(trade)
        self.daily_trades.append(trade)
        self.current_balance += profit
        
        logger.info(f"Trade recorded - Type: {trade_type}, Profit: ${profit:.2f}")
    
    def get_account_status(self):
        """Get current account status."""
        total_profit = self.current_balance - self.initial_balance
        return_percent = (total_profit / self.initial_balance) * 100
        
        max_drawdown = self._calculate_max_drawdown()
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'total_profit': total_profit,
            'return_percent': return_percent,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades_executed),
            'remaining_loss_capacity': self.max_loss - (self.initial_balance - self.current_balance)
        }
    
    @staticmethod
    def _calculate_max_drawdown():
        """Calculate maximum drawdown."""
        # Implementation for max drawdown calculation
        return 0.0
    
    def reset_daily(self):
        """Reset daily tracking."""
        self.daily_balance = self.current_balance
        self.daily_trades = []
        logger.info("Daily tracking reset")
