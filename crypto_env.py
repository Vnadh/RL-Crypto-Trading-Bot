import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CryptoTradingEnv(gym.Env):
    """Cryptocurrency Trading Environment for Reinforcement Learning"""
    
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, df, initial_balance=10000):
        super().__init__()
        
        # Data preparation
        self.df = df.astype(np.float32)
        self.features = self.df.columns.tolist()
        
        # Trading parameters
        self.current_step = 0
        self.initial_balance = np.float32(initial_balance)
        self.balance = self.initial_balance
        self.btc_held = np.float32(0.0)
        
        # Spaces
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(len(self.features) + 2,),  # Features + balance + btc_held
            dtype=np.float32
        )
        
        # Rendering
        self.fig = None
        self.price_ax = None
        self.balance_ax = None
        self.trade_lines = []
        self.render_data = {
            'prices': [],
            'balances': [],
            'trades': []
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.btc_held = np.float32(0.0)
        return self._next_observation(), {}

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        reward = self._calculate_reward()
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'btc_held': self.btc_held,
            'net_worth': self.net_worth
        }
        
        return self._next_observation(), reward, terminated, truncated, info

    def _next_observation(self):
        features = self.df.iloc[self.current_step].values.astype(np.float32)
        portfolio = np.array([self.balance, self.btc_held], dtype=np.float32)
        return np.concatenate([features, portfolio])

    def _take_action(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        if action == 1:  # Buy
            btc_bought = (self.balance * 0.1) / current_price
            self.btc_held += np.float32(btc_bought)
            self.balance -= (btc_bought * current_price) * 1.0002
            self._record_trade('buy', btc_bought, current_price)
            
        elif action == 2:  # Sell
            if self.btc_held > 0:
                btc_sold = self.btc_held * 0.1
                self.btc_held -= np.float32(btc_sold)
                self.balance += (btc_sold * current_price) * 0.9998
                self._record_trade('sell', btc_sold, current_price)

    def _calculate_reward(self):
        current_value = self.net_worth
        previous_value = self._get_previous_value()
        if current_value - previous_value==0:
            return -0.001
        return current_value - previous_value

    def _get_previous_value(self):
        if self.current_step == 0:
            return self.initial_balance
        previous_price = self.df.iloc[self.current_step-1]['close']
        return self.balance + self.btc_held * previous_price

    def _record_trade(self, action_type, amount, price):
        """Store trades with timestamp"""
        self.render_data['trades'].append((
            self.current_step,
            action_type,
            amount,
            price
        ))
        # Auto-clean old trades
        if len(self.render_data['trades']) > 1000:
            self.render_data['trades'] = self.render_data['trades'][-500:]

    @property
    def net_worth(self):
        return self.balance + self.btc_held * self.current_price
    
    @property
    def current_price(self):
        return self.df.iloc[self.current_step]['close']

    def render(self, mode='human', recent_steps=200):
        if mode == 'human':
            self._render_human(recent_steps)

    def _render_human(self, recent_steps=200):
        if self.fig is None:
            plt.ion()
            self.fig, (self.price_ax, self.balance_ax) = plt.subplots(2, 1, figsize=(12, 8))
            self.price_line, = self.price_ax.plot([], [], label='Price', color='blue')
            self.balance_line, = self.balance_ax.plot([], [], label='Net Worth', color='green')
            
            # Initialize plots
            self.price_ax.set_title('BTC/USDT Price')
            self.balance_ax.set_title('Portfolio Value')
            plt.tight_layout()
            plt.show(block=False)

        # Update data buffers
        self.render_data['prices'].append(self.current_price)
        self.render_data['balances'].append(self.net_worth)
        
        # Keep data within window
        window_start = max(0, len(self.render_data['prices']) - recent_steps)
        prices = self.render_data['prices'][window_start:]
        balances = self.render_data['balances'][window_start:]
        steps = np.arange(len(prices))
        
        # Update price plot
        self.price_line.set_data(steps, prices)
        self.price_ax.relim()
        self.price_ax.autoscale_view()
        self.price_ax.set_title(f'BTC/USDT Price (Step {self.current_step})')
        
        # Update balance plot
        self.balance_line.set_data(steps, balances)
        self.balance_ax.relim()
        self.balance_ax.autoscale_view()
        
        # Clear previous trade markers
        for artist in self.price_ax.texts + self.price_ax.lines[2:]:
            artist.remove()
        
        # Plot new trades within window
        window_trades = [t for t in self.render_data['trades'] 
                        if t[0] >= (self.current_step - len(prices))]
        for trade in window_trades[-10:]:  # Show last 10 trades in window
            step, action_type, amount, price = trade
            x_pos = step - (self.current_step - len(prices))
            color = 'lime' if action_type == 'buy' else 'red'
            
            if 0 <= x_pos < len(prices):
                self.price_ax.axvline(x=x_pos, color=color, alpha=0.3)
                self.price_ax.text(x_pos, price, f"{action_type}\n{amount:.4f}", 
                                color=color, ha='center', va='bottom', fontsize=8)

        # Efficient redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    def _clean_old_trades(self, window_start):
        """Remove trades outside the visible window"""
        self.render_data['trades'] = [
            t for t in self.render_data['trades'] 
            if t[0] >= window_start
        ]
    


    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

if __name__ == "__main__":
    # Test environment
    df = pd.read_csv("unseen_btc_data.csv").select_dtypes(include=[np.number]).ffill().bfill().astype(np.float32)
    env = CryptoTradingEnv(df)
    
    # Run sample episode
    obs, _ = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        env.render()
        
        if done:
            break
    
    env.close()
    print("Test completed successfully!")