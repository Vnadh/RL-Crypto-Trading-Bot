import pandas as pd
from crypto_env import CryptoTradingEnv
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def evaluate_model(model, env, n_episodes=10):
    metrics = {
        'returns': [],
        'sharpe_ratios': [],
        'max_drawdowns': [],
        'win_rate': [],
        'volatility': []
    }
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        portfolio_values = []
        
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _, info = env.step(action)
            portfolio_values.append(info['net_worth'])
        
        returns = calculate_metrics(portfolio_values)
        for k in metrics.keys():
            metrics[k].append(returns[k])
    
    return {k: np.mean(v) for k, v in metrics.items()}

def calculate_metrics(portfolio_values):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    if len(returns) == 0:
        return {k: 0 for k in ['returns', 'sharpe_ratios', 'max_drawdowns', 'win_rate', 'volatility']}
    
    return {
        'returns': (portfolio_values[-1] / portfolio_values[0] - 1) * 100,
        'sharpe_ratios': np.nan_to_num(np.mean(returns) / np.std(returns)) * np.sqrt(365*24),
        'max_drawdowns': (np.min(portfolio_values) / np.max(portfolio_values) - 1) * 100,
        'win_rate': (np.sum(np.array(returns) > 0) / len(returns)) * 100,
        'volatility': np.std(returns) * np.sqrt(365*24)
    }

def buy_and_hold(env):
    obs, _ = env.reset()
    done = False
    portfolio_values = []
    
    while not done:
        action = 1 if env.current_step == 0 else 0
        obs, _, done, _, info = env.step(action)
        portfolio_values.append(info['net_worth'])
    
    return calculate_metrics(portfolio_values)

def visualize_comparison(env, models, model_names):
    plt.figure(figsize=(14, 8))
    colors = ['green', 'blue', 'orange']
    prices = []
    
    for idx, (model, name) in enumerate(zip(models, model_names)):
        obs, _ = env.reset()
        done = False
        portfolio_values = []
        current_prices = []
        
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _, info = env.step(action)
            portfolio_values.append(info['net_worth'])
            current_prices.append(info.get('price', 0))
        
        if idx == 0:  # Capture prices once
            prices = current_prices
        
        plt.plot(portfolio_values, label=name, color=colors[idx], alpha=0.8)
    
    # Plot Buy & Hold
    bh_values = [p/prices[0]*10000 for p in prices] if prices[0] != 0 else []
    plt.plot(bh_values, label='Buy & Hold', linestyle='-', color='red')
    
    plt.title('Portfolio Value Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend()
    #plt.grid(True)
    plt.show()

# Load and prepare data
test_df = pd.read_csv("unseen_btc_data.csv")
df=pd.read_csv("btc_data.csv")
test_df=df+test_df
test_df = (test_df.select_dtypes(include=[np.number])
           .ffill()
           .bfill()
           .astype(np.float32))

# Create test environment
test_env = CryptoTradingEnv(test_df)

# Load trained models
models = {
    'PPO': PPO.load("models/PPO.zip"),
    'A2C': A2C.load("models/A2C.zip"),  # Update with your A2C model path
    'DQN': DQN.load("models/DQN.zip")   # Update with your DQN model path
}

# Evaluate all models
results = {}
for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    results[model_name] = evaluate_model(model, test_env)

# Evaluate Buy & Hold
bh_results = buy_and_hold(test_env)

# Print results
print("\n=== Model Comparison ===")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for k, v in metrics.items():
        print(f"{k:15}: {v:.2f}")

print("\n=== Buy & Hold ===")
for k, v in bh_results.items():
    print(f"{k:15}: {v:.2f}")

# Visualize comparison
visualize_comparison(test_env, list(models.values()), list(models.keys()))