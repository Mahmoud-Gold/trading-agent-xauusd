"""Global settings and configuration for Trading Agent."""

import os
from datetime import datetime
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TRADING PARAMETERS
# ============================================================================

# Symbol Configuration
SYMBOL = "XAUUSD"  # Gold / US Dollar
TIMEFRAME = "1h"   # 1-hour candles
EXCHANGE = "binance"  # or other broker

# Market Hours (UTC)
MARKET_OPEN_HOUR = 0
MARKET_CLOSE_HOUR = 24
MAX_TRADING_HOURS = 24

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# Account Protection
INITIAL_BALANCE = 10000.0  # Starting balance in USD
MAX_ACCOUNT_LOSS_PERCENT = 10.0  # 10% maximum account loss
DAILY_DRAWDOWN_LIMIT = 5.0  # 5% daily drawdown limit

# Position Management
MAX_POSITION_SIZE = 0.05  # Max 5% of account per trade
MIN_POSITION_SIZE = 0.01  # Min 1% of account per trade

# Stop Loss & Take Profit
STOP_LOSS_PERCENT = 2.0
TAKE_PROFIT_PERCENT = 4.0

# Slippage & Commissions
SLIPPAGE_PERCENT = 0.5  # 0.5% slippage
COMMISSION_PERCENT = 0.1  # 0.1% commission

# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

# DQN Agent Parameters
STATE_SIZE = 64  # State vector dimension
ACTION_SIZE = 3  # Buy, Hold, Sell
MEMORY_SIZE = 10000  # Experience replay buffer
BATCH_SIZE = 32
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 0.001

# Exploration
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.01  # Final exploration rate
EPSILON_DECAY = 0.995  # Decay rate

# Training
TOTAL_STEPS = 5_000_000  # 5 Million steps
UPDATE_TARGET_FREQ = 1000  # Update target network every N steps
LOG_INTERVAL = 100  # Log every N steps

# Noise Injection (Robustness)
NOISE_ENABLED = True
NOISE_LEVEL = 0.05  # 5% noise injection
NOISE_DECAY = 0.9999  # Noise decay rate

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_EPISODES = 1000
EPISODE_STEPS = 1000
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Data normalization
NORMALIZE_DATA = True
FEATURE_SCALING = "standard"  # standard or minmax

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

TECHNICAL_INDICATORS = [
    "RSI",  # Relative Strength Index
    "MACD",  # Moving Average Convergence Divergence
    "BB",  # Bollinger Bands
    "ATR",  # Average True Range
    "SMA",  # Simple Moving Average
    "EMA",  # Exponential Moving Average
]

RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
BB_PERIOD = 20
ATR_PERIOD = 14
SMA_PERIODS = [20, 50, 200]
EMA_PERIODS = [12, 26]

# ============================================================================
# BACKTESTING
# ============================================================================

BACKTEST_START_DATE = "2022-01-01"
BACKTEST_END_DATE = "2024-05-21"
BACKTEST_INITIAL_BALANCE = 10000.0

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# ============================================================================
# DATABASE & API
# ============================================================================

# API Keys (use environment variables in production)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# ============================================================================
# ALERTS & NOTIFICATIONS
# ============================================================================

EMAIL_ALERTS = False
EMAIL_ADDRESS = os.getenv("ALERT_EMAIL", "")
TELEGRAM_ALERTS = False
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# ============================================================================
# PRODUCTION SETTINGS
# ============================================================================

DEBUG_MODE = True
DRY_RUN = True  # Simulate trades without real money
REAL_TRADING = False  # Set to True only when you're sure!
