import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# --- Hyperparameters ---
ENV_NAME = "CartPole-v1"
GAMMA = 0.99                # Discount factor for future rewards
LR = 1e-3                   # Learning rate
BATCH_SIZE = 64             # Size of random memory sample
BUFFER_CAPACITY = 10000     # Maximum size of memory bank
MIN_REPLAY_SIZE = 1000      # Wait for this many steps before starting training
EPSILON_START = 1.0         # 100% random exploration at the start
EPSILON_END = 0.05          # Maintain a 5% baseline exploration rate
EPSILON_DECAY = 0.995       # Decay exploration rate exponentially per episode
TARGET_UPDATE_FREQ = 10     # Sync Target network every 10 episodes
MAX_EPISODES = 150          # Limit of the engineering run

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. The Deep Q-Network Architecture ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim) # Outputs Q-values for each possible discrete action
        )
        
    def forward(self, x):
        return self.net(x)

# --- 2. The Experience Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.FloatTensor(np.array(state)).to(DEVICE),
                torch.LongTensor(action).unsqueeze(1).to(DEVICE),
                torch.FloatTensor(reward).to(DEVICE),
                torch.FloatTensor(np.array(next_state)).to(DEVICE),
                torch.FloatTensor(done).to(DEVICE))
                
    def __len__(self):
        return len(self.buffer)

# --- 3. The Core Training Loop Engine ---
def train_agent():
    print(f"Booting up Deep Q-Network Controller on {DEVICE}...")
    
    # Initialize modern Gymnasium environment
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Instantiate Dual-Network Systems
    policy_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict()) # Lock initial states identical
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(BUFFER_CAPACITY)
    
    epsilon = EPSILON_START
    episode_rewards = []
    
    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-Greedy Action Selection
            if random.random() < epsilon:
                action = env.action_space.sample() # Explore completely at random
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    action = policy_net(state_t).argmax(dim=1).item() # Exploit best known Q-value
            
            # Step the environment forward based on action selection
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Record the raw transition memory
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Optimization Phase (Executes only if memory bank has sufficient footprint)
            if len(memory) >= MIN_REPLAY_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                
                # Predict current Q values for selected actions
                current_q_values = policy_net(states).gather(1, actions)
                
                # Predict optimal next Q values using the FROZEN Target Network (The Bellman Equation)
                with torch.no_grad():
                    max_next_q_values = target_net(next_states).max(dim=1)[0]
                    expected_q_values = rewards + (GAMMA * max_next_q_values * (1 - dones))
                    
                # Optimize Policy using Mean Squared Error loss
                loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Handle linear exploration decay
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)
        
        # Periodically synchronize target network weights
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        # Logging metrics
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        print(f"Episode [{episode}/{MAX_EPISODES}] | Score: {total_reward:.1f} | 10-Ep Rolling Avg: {avg_reward:.1f} | Epsilon: {epsilon:.3f}")
        
        # CartPole-v1 is officially considered "solved" if the rolling average score clears 475.0 points
        if avg_reward >= 475.0:
            print(f"\n[VICTORY] Physics stabilized! Environment solved in {episode} episodes!")
            break
            
    env.close()
    
    # Generate Professional Diagnostic Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Episode Reward", color="purple", alpha=0.4)
    
    # Calculate rolling window to view true progression
    rolling_windows = [np.mean(episode_rewards[max(0, i-9):i+1]) for i in range(len(episode_rewards))]
    plt.plot(rolling_windows, label="10-Episode Rolling Average", color="darkorange", linewidth=2)
    
    plt.axhline(y=475, color="green", linestyle="--", label="Gymnasium Solved Threshold (475)")
    plt.title("Deep Q-Network Optimization Profile (CartPole-v1)")
    plt.xlabel("Training Episodes")
    plt.ylabel("Cumulative Reward (Steps Balanced)")
    plt.legend()
    plt.grid(True)
    plt.savefig("dqn_training_profile.png")
    print("\nTraining completed. Optimization diagnostic saved to 'dqn_training_profile.png'.")

if __name__ == "__main__":
    train_agent()