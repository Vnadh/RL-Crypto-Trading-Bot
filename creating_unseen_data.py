import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta

def fetch_crypto_data(symbol='BTC/USDT', timeframe='1m', limit=1000):
    """Fetch historical crypto data from Binance"""
    exchange = ccxt.binance()
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp')

def preprocess_data(df):
    """Calculate technical indicators and clean data"""
    # Calculate indicators
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, append=True)
    df['50_ma'] = df.close.rolling(50).mean()
    
    # Clean up column names
    df.rename(columns={
        'RSI_14': 'rsi',
        'MACD_12_26_9': 'macd',
        'BBM_20_2.0': 'bollinger',
    }, inplace=True)
    
    # Handle missing values and numeric types
    df = df.ffill().bfill().select_dtypes(include=[np.number]).astype(np.float32)
    
    return df

if __name__ == "__main__":
    # Example usage
    unseen_btc_data = fetch_crypto_data(limit=100_000)
    unseen_btc_data = preprocess_data(unseen_btc_data)
    unseen_btc_data.to_csv("unseen_btc_data.csv")
    print("Data saved to unseen_btc_data.csv")