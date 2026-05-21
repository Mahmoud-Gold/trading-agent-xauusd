# Trading Agent XAUUSD

🤖 **Advanced AI Trading Agent for Gold (XAUUSD)** with Deep Reinforcement Learning

## 🎯 Features

✅ **Neural Network Agent** - Deep Q-Learning (DQN) architecture
✅ **Risk Management** - Protected from overfitting & slippage
✅ **Noise Injection** - Robust training with 5M+ steps
✅ **Loss Protection** - Max 10% account loss or 5% daily drawdown
✅ **Real-time Trading** - Live market data integration
✅ **Backtesting Engine** - Historical performance analysis
✅ **Auto-learning** - Zero to Hero learning curve

## 📁 Project Structure

```
trading-agent-xauusd/
├── config/              # Configuration files
│   ├── __init__.py
│   └── settings.py      # Global settings
├── data/                # Data storage
│   ├── raw/            # Raw market data
│   ├── processed/      # Processed features
│   └── models/         # Trained models
├── src/
│   ├── __init__.py
│   ├── agent/          # DQN Agent implementation
│   ├── environment/    # Trading environment
│   ├── data_handler/   # Data preprocessing
│   ├── risk_manager/   # Risk management
│   ├── training/       # Training pipeline
│   └── backtesting/    # Backtesting engine
├── tests/              # Unit tests
├── logs/               # Training & trading logs
├── main.py             # Entry point
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/mahmoud-gold/trading-agent-xauusd.git
cd trading-agent-xauusd
pip install -r requirements.txt
```

### Training

```python
python main.py --mode train --episodes 1000
```

### Backtesting

```python
python main.py --mode backtest --model latest
```

### Live Trading

```python
python main.py --mode trade --live
```

## 📊 Configuration

Edit `config/settings.py` for:
- Symbol: XAUUSD (Gold/USD)
- Risk limits: 10% max loss, 5% daily drawdown
- Model hyperparameters
- Trading hours & market conditions

## 🧠 Agent Architecture

**DQN (Deep Q-Network)** with:
- 4-layer neural network
- Experience replay buffer
- Target network stabilization
- Epsilon-greedy exploration
- Noise injection for robustness

## 📈 Performance Metrics

- **Sharpe Ratio** monitoring
- **Win Rate** tracking
- **Drawdown** protection
- **Risk/Reward** optimization

## ⚠️ Risk Management

- **Max Account Loss**: 10%
- **Daily Drawdown Limit**: 5%
- **Position Size**: Dynamic based on volatility
- **Stop Loss**: Automatic
- **Take Profit**: Smart levels

## 📝 License

MIT License - See LICENSE file

## 👨‍💻 Author

**Mahmoud Gold** - AI Trading Development

---

⚡ **Ready to trade like a hero?** Let's go! 🚀
