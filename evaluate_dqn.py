import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

# --- Configuration ---
ENV_NAME = "CartPole-v1"
WEIGHTS_PATH = "ddqn_cartpole_weights.pt"
EVAL_EPISODES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Identical Network Architecture ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def evaluate_policy():
    print(f"Loading converged policy weights from '{WEIGHTS_PATH}'...")
    
    # Initialize environment with video recording capabilities
    base_env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        base_env, 
        video_folder="./production_rollouts", 
        episode_trigger=lambda episode_id: True,
        name_prefix="perfect_run"
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Load and lock trained brain into evaluation mode
    policy_net = DQN(state_dim, action_dim).to(DEVICE)
    policy_net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    policy_net.eval() 
    
    print("\nStarting evaluation rollouts (Epsilon = 0.00 | Pure Exploitation)...")
    
    for ep in range(1, EVAL_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                # 100% deterministic exploitation of optimal Q-values
                action = policy_net(state_t).argmax(dim=1).item()
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            
        print(f"Evaluation Episode [{ep}/{EVAL_EPISODES}] -> Total Steps Balanced: {episode_reward}")
        
    env.close()
    print("\nEvaluation complete. Video playbacks saved to the './production_rollouts' directory.")

if __name__ == "__main__":
    evaluate_policy()