from stable_baselines3 import PPO, A2C, DQN, SAC, DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env
from crypto_env import CryptoTradingEnv
import pandas as pd
import numpy as np

# Load preprocessed data
df = pd.read_csv("btc_data.csv").select_dtypes(include=[np.number]).ffill().bfill().astype(np.float32)

def train_and_evaluate_models(data, models_config):
    results = {}
    
    for model_name, config in models_config.items():
        print(f"\nTraining {model_name}...")
        
        # Create vectorized environment
        env = make_vec_env(lambda: CryptoTradingEnv(data), n_envs=config['n_envs'])
        
        # Initialize model
        model = config['class'](
            config['policy'],
            env,
            verbose=1,
            **config['params']
        )
        
        # Train
        model.learn(total_timesteps=config['timesteps'])
        model.save(f"models/{model_name}")
        
        # Evaluate
        test_env = CryptoTradingEnv(data)
        mean_reward, _ = evaluate_model(model, test_env)
        results[model_name] = mean_reward
        
        env.close()
        test_env.close()
    
    return results

def evaluate_model(model, env, n_episodes=10):
    total_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards), np.std(total_rewards)

# Configuration for different models
MODELS_CONFIG = {
    'PPO': {
        'class': PPO,
        'policy': 'MlpPolicy',
        'n_envs': 4,
        'params': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'n_steps': 256,
            'batch_size': 64,
            'ent_coef':0.01
        },
        'timesteps': 10_00_000
    },
    'A2C': {
        'class': A2C,
        'policy': 'MlpPolicy',
        'n_envs': 4,
        'params': {
            'learning_rate': 7e-4,
            'gamma': 0.95,
            'n_steps': 128,
            'use_rms_prop': True,
             'ent_coef':0.01
        },
        'timesteps': 8_00_000
    },
    'DQN': {
        'class': DQN,
        'policy': 'MlpPolicy',
        'n_envs': 1,  # DQN doesn't support multiple envs
        'params': {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'buffer_size': 100_000,
            'exploration_final_eps': 0.01
        },
        'timesteps': 5_00_000
    },
}

# Train and compare models
results = train_and_evaluate_models(df, MODELS_CONFIG)

print("\nModel Comparison:")
for model_name, reward in results.items():
    print(f"{model_name}: Average Reward = {reward:.2f}")