#!/usr/bin/env python3
"""Main entry point for Trading Agent."""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/trading_agent.log", rotation="500 MB", level="DEBUG")

from config.settings import (
    MODELS_DIR, INITIAL_BALANCE, TRAINING_EPISODES,
    TOTAL_STEPS, DRY_RUN, REAL_TRADING
)
from src.agent.dqn_agent import DQNAgent
from src.environment.trading_env import TradingEnvironment
from src.data_handler.data_processor import DataProcessor
from src.risk_manager.risk_manager import RiskManager
from src.training.trainer import Trainer
from src.backtesting.backtest import Backtester


def load_sample_data(num_samples=5000):
    """Load sample price data."""
    logger.info(f"Loading sample data - {num_samples} samples")
    # Generate synthetic XAUUSD prices
    prices = np.random.uniform(2000, 2100, num_samples)
    logger.info(f"Data loaded - Price range: ${prices.min():.2f}-${prices.max():.2f}")
    return prices


def train_mode(args):
    """Training mode."""
    logger.info("=" * 80)
    logger.info("TRADING AGENT - TRAINING MODE")
    logger.info("=" * 80)
    
    # Load data
    prices = load_sample_data()
    
    # Initialize components
    agent = DQNAgent()
    env = TradingEnvironment(prices, INITIAL_BALANCE)
    trainer = Trainer(agent, env)
    
    # Train
    logger.info(f"Training for {args.episodes} episodes...")
    trainer.train_episodes(episodes=args.episodes)
    
    # Save model
    model_path = MODELS_DIR / f"agent_trained_{args.episodes}ep.h5"
    agent.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Final metrics
    metrics = env.get_metrics()
    logger.info("-" * 80)
    logger.info("TRAINING RESULTS")
    logger.info(f"  Final Balance: ${metrics['final_balance']:.2f}")
    logger.info(f"  Total Return: {metrics['total_return']:.2f}%")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    logger.info(f"  Total Trades: {metrics['total_trades']}")
    logger.info(f"  Win Rate: {metrics['win_rate']:.2f}%")
    logger.info("-" * 80)


def backtest_mode(args):
    """Backtesting mode."""
    logger.info("=" * 80)
    logger.info("TRADING AGENT - BACKTESTING MODE")
    logger.info("=" * 80)
    
    # Load data
    prices = load_sample_data()
    
    # Load agent
    agent = DQNAgent()
    model_path = args.model or str(MODELS_DIR / "agent_trained_1000ep.h5")
    
    try:
        agent.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Training new model...")
        env = TradingEnvironment(prices, INITIAL_BALANCE)
        trainer = Trainer(agent, env)
        trainer.train_episodes(episodes=100)
        agent.save(str(model_path))
    
    # Run backtest
    backtester = Backtester(agent, prices)
    results = backtester.run(initial_balance=INITIAL_BALANCE)
    
    # Display results
    logger.info("-" * 80)
    logger.info("BACKTEST RESULTS")
    logger.info(f"  Final Balance: ${results['final_balance']:.2f}")
    logger.info(f"  Total Return: {results['total_return']:.2f}%")
    logger.info(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
    logger.info(f"  Total Trades: {results['total_trades']}")
    logger.info(f"  Win Rate: {results['win_rate']:.2f}%")
    logger.info(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info("-" * 80)


def trade_mode(args):
    """Live trading mode."""
    if REAL_TRADING and not DRY_RUN:
        logger.warning("=" * 80)
        logger.warning("CAUTION: REAL TRADING ENABLED")
        logger.warning("This will execute real trades with real money!")
        logger.warning("=" * 80)
    elif DRY_RUN:
        logger.info("Running in DRY RUN mode - No real trades will be executed")
    
    logger.info("Live trading mode not fully implemented yet")
    logger.info("Please use training or backtesting modes")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Trading Agent for XAUUSD (Gold/USD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train --episodes 500
  python main.py --mode backtest --model data/models/agent.h5
  python main.py --mode trade --live
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["train", "backtest", "trade"],
        default="train",
        help="Operating mode"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=TRAINING_EPISODES,
        help=f"Number of training episodes (default: {TRAINING_EPISODES})"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (requires real API keys)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        logger.add("logs/trading_agent.log", rotation="500 MB", level="DEBUG")
    
    try:
        if args.mode == "train":
            train_mode(args)
        elif args.mode == "backtest":
            backtest_mode(args)
        elif args.mode == "trade":
            trade_mode(args)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.exception("Traceback:")
        sys.exit(1)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
