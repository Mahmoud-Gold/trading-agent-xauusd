"""Data Processing and Feature Engineering."""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config.settings import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, BB_PERIOD, ATR_PERIOD,
    SMA_PERIODS, EMA_PERIODS, FEATURE_SCALING, NORMALIZE_DATA
)


class DataProcessor:
    """Process and engineer features from market data."""
    
    def __init__(self):
        """Initialize data processor."""
        self.scaler = StandardScaler() if FEATURE_SCALING == 'standard' else MinMaxScaler()
        logger.info(f"DataProcessor initialized with {FEATURE_SCALING} scaling")
    
    @staticmethod
    def calculate_rsi(prices, period=RSI_PERIOD):
        """Calculate Relative Strength Index."""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    @staticmethod
    def calculate_macd(prices, fast=MACD_FAST, slow=MACD_SLOW):
        """Calculate MACD."""
        ema_fast = pd.Series(prices).ewm(span=fast).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow).mean().values
        macd = ema_fast - ema_slow
        signal = pd.Series(macd).ewm(span=9).mean().values
        histogram = macd - signal
        
        return macd, signal, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=BB_PERIOD, num_std=2):
        """Calculate Bollinger Bands."""
        sma = pd.Series(prices).rolling(window=period).mean().values
        std = pd.Series(prices).rolling(window=period).std().values
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_atr(high, low, close, period=ATR_PERIOD):
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = np.abs(high - close[:-1])
        tr3 = np.abs(low - close[:-1])
        
        tr = np.maximum(tr1, tr2)
        tr = np.maximum(tr, tr3[:-1])
        
        atr = np.zeros_like(close)
        atr[:period] = np.mean(tr[:period])
        
        for i in range(period, len(close)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
        
        return atr
    
    @staticmethod
    def calculate_sma(prices, periods=SMA_PERIODS):
        """Calculate Simple Moving Averages."""
        smas = {}
        for period in periods:
            smas[f'SMA_{period}'] = pd.Series(prices).rolling(window=period).mean().values
        
        return smas
    
    @staticmethod
    def calculate_ema(prices, periods=EMA_PERIODS):
        """Calculate Exponential Moving Averages."""
        emas = {}
        for period in periods:
            emas[f'EMA_{period}'] = pd.Series(prices).ewm(span=period).mean().values
        
        return emas
    
    def create_features(self, df):
        """Create all technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features
        """
        features = df.copy()
        
        # RSI
        features['RSI'] = self.calculate_rsi(df['close'].values)
        
        # MACD
        macd, signal, histogram = self.calculate_macd(df['close'].values)
        features['MACD'] = macd
        features['MACD_Signal'] = signal
        features['MACD_Histogram'] = histogram
        
        # Bollinger Bands
        upper, sma, lower = self.calculate_bollinger_bands(df['close'].values)
        features['BB_Upper'] = upper
        features['BB_Middle'] = sma
        features['BB_Lower'] = lower
        
        # ATR
        features['ATR'] = self.calculate_atr(
            df['high'].values,
            df['low'].values,
            df['close'].values
        )
        
        # SMAs
        smas = self.calculate_sma(df['close'].values)
        for name, values in smas.items():
            features[name] = values
        
        # EMAs
        emas = self.calculate_ema(df['close'].values)
        for name, values in emas.items():
            features[name] = values
        
        # Returns
        features['Returns'] = df['close'].pct_change()
        features['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Handle NaN values
        features = features.fillna(method='bfill').fillna(method='ffill')
        
        logger.info(f"Features created - Shape: {features.shape}")
        
        return features
    
    def normalize_features(self, features, fit=False):
        """Normalize features to 0-1 range."""
        if fit:
            normalized = self.scaler.fit_transform(features)
        else:
            normalized = self.scaler.transform(features)
        
        return normalized
