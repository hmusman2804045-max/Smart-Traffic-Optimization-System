import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from stable_baselines3 import PPO

class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), 
            high=np.array([100, 100, 1, 3]), 
            dtype=np.float32
        )
        self.state = np.array([20.0, 20.0, 0.0, 0.0], dtype=np.float32)
        self.time_steps = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([20.0, 20.0, 0.0, 0.0], dtype=np.float32)
        self.time_steps = 0
        return self.state, {}

    def step(self, action):
        self.time_steps += 1
        curr_density, pred_density, anomaly, sentiment = self.state
        if action == 0:
            curr_density = max(0, curr_density - 5)
        else:
            curr_density = min(100, curr_density + 2)
        reward = -curr_density 
        if anomaly > 0.5: reward -= 50 
        if sentiment > 1: reward -= 20
        self.state = np.array([curr_density, pred_density, anomaly, sentiment], dtype=np.float32)
        terminated = self.time_steps >= self.max_steps
        truncated = False
        return self.state, reward, terminated, truncated, {}

class TrafficOptimizer:
    def __init__(self, model_path="ppo_traffic_model"):
        self.model_path = model_path
        self.env = TrafficEnv()
        self.model = None

    def train(self, total_timesteps=10000):
        print(f" Training Multi-Modal RL agent for {total_timesteps} steps...")
        self.model = PPO("MlpPolicy", self.env, verbose=0)
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(self.model_path)
        print(" Training complete. Multi-modal model saved.")

    def run_inference(self, state_vector):
        if self.model is None:
            if os.path.exists(self.model_path + ".zip"):
                self.model = PPO.load(self.model_path)
            else:
                self.train(5000)
        obs = np.array(state_vector, dtype=np.float32)
        obs = np.clip(obs, self.env.observation_space.low, self.env.observation_space.high)
        obs = np.expand_dims(obs, 0)
        action, _states = self.model.predict(obs, deterministic=True)
        return action[0]

if __name__ == "__main__":
    optimizer = TrafficOptimizer()
    optimizer.train(total_timesteps=10000)
    scenarios = [
        [80, 85, 0, 0],
        [10, 15, 1, 1],
        [30, 70, 0, 2],
    ]
    print("\n Testing Multi-Modal RL Decision logic:")
    print("-" * 40)
    for s in scenarios:
        decision = optimizer.run_inference(s)
        action_text = "GREEN LIGHT" if decision == 0 else "RED LIGHT (Keep clearing others)"
        print(f"State [Dens:{s[0]}, Pred:{s[1]}, Anom:{s[2]}, Sent:{s[3]}] -> Signal: {action_text}")
