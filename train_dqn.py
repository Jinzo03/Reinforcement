import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# --- Hyperparameters ---
ENV_NAME        = "CartPole-v1"
GAMMA           = 0.99
LR              = 3e-4
BATCH_SIZE      = 128
BUFFER_CAPACITY = 50000
MIN_REPLAY_SIZE = 2000
EPSILON_START   = 1.0
EPSILON_END     = 0.01
EPSILON_DECAY   = 0.988         # greedy by ~ep 260
TARGET_UPDATE_FREQ = 10
MAX_EPISODES    = 700

# PER parameters
PER_ALPHA   = 0.6               # how much prioritization (0=uniform, 1=full priority)
PER_BETA    = 0.4               # importance sampling start (anneals to 1.0)
PER_EPSILON = 1e-6              # small constant so zero-error transitions still get sampled

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Prioritized Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity  = capacity
        self.alpha     = alpha
        self.buffer    = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos       = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # importance sampling weights to correct for bias
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = torch.FloatTensor(weights).to(DEVICE)

        state, action, reward, next_state, done = zip(*samples)
        return (torch.FloatTensor(np.array(state)).to(DEVICE),
                torch.LongTensor(action).unsqueeze(1).to(DEVICE),
                torch.FloatTensor(reward).to(DEVICE),
                torch.FloatTensor(np.array(next_state)).to(DEVICE),
                torch.FloatTensor(done).to(DEVICE),
                indices,
                weights)

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + PER_EPSILON

    def __len__(self):
        return len(self.buffer)


# --- Dueling DQN ---
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        features   = self.feature_network(x)
        values     = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))


# --- Reward Shaping ---
def shape_reward(state, base_reward, done, step):
    pole_angle    = abs(state[2])
    cart_position = abs(state[0])
    pole_velocity = abs(state[3])

    angle_bonus    = 0.5 * (1.0 - pole_angle    / 0.2095)
    position_bonus = 0.3 * (1.0 - cart_position / 2.4)
    velocity_bonus = 0.2 * (1.0 - pole_velocity / 2.0)

    shaped = base_reward + angle_bonus + position_bonus + velocity_bonus
    if done and step < 499:
        shaped -= 10.0
    return shaped


def train_agent():
    print(f"Booting Dueling Double DQN + PER on {DEVICE}...")

    env = gym.make(ENV_NAME)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DuelingDQN(state_dim, action_dim).to(DEVICE)
    target_net = DuelingDQN(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory    = PrioritizedReplayBuffer(BUFFER_CAPACITY, PER_ALPHA)

    epsilon       = EPSILON_START
    beta          = PER_BETA
    episode_rewards = []
    best_avg_reward = 0.0
    solve_streak    = 0

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0
        done  = False
        step  = 0

        # anneal beta toward 1.0 over training
        beta = min(1.0, PER_BETA + episode * (1.0 - PER_BETA) / MAX_EPISODES)

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    action  = policy_net(state_t).argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            step += 1

            shaped = shape_reward(next_state, reward, done, step)
            memory.push(state, action, shaped, next_state, done)
            state        = next_state
            total_reward += reward

            if len(memory) >= MIN_REPLAY_SIZE:
                states, actions, rewards, next_states, dones, indices, weights = memory.sample(BATCH_SIZE, beta)

                current_q = policy_net(states).gather(1, actions)

                with torch.no_grad():
                    next_actions  = policy_net(next_states).argmax(dim=1, keepdim=True)
                    next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
                    expected_q    = rewards + (GAMMA * next_q_values * (1 - dones))

                td_errors = (current_q.squeeze(1) - expected_q).detach().cpu().numpy()
                memory.update_priorities(indices, td_errors)

                # weighted Huber loss
                loss = (weights * nn.SmoothL1Loss(reduction='none')(
                    current_q, expected_q.unsqueeze(1)).squeeze(1)).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
                optimizer.step()

        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)

        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)

        print(f"Episode [{episode}/{MAX_EPISODES}] | Score: {total_reward:.1f} | "
              f"10-Ep Avg: {avg_reward:.1f} | Epsilon: {epsilon:.3f} | Beta: {beta:.3f}")

        if avg_reward > best_avg_reward and len(episode_rewards) >= 10:
            best_avg_reward = avg_reward
            torch.save(policy_net.state_dict(), "ddqn_cartpole_weights.pt")
            print(f"    --> Checkpoint saved (Avg: {best_avg_reward:.1f})")

        if avg_reward >= 475.0:
            solve_streak += 1
            if solve_streak >= 10:
                print(f"\n[VICTORY] Stably solved at episode {episode}!")
                break
        else:
            solve_streak = 0

    env.close()

    plt.figure(figsize=(12, 5))
    plt.plot(episode_rewards, label="Episode Reward", color="purple", alpha=0.3)
    rolling = [np.mean(episode_rewards[max(0, i-9):i+1]) for i in range(len(episode_rewards))]
    plt.plot(rolling, label="10-Episode Rolling Average", color="darkorange", linewidth=2)
    plt.axhline(y=475, color="green", linestyle="--", label="Solved Threshold (475)")
    plt.title("Dueling Double DQN + PER — CartPole-v1")
    plt.xlabel("Training Episodes")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dqn_training_profile.png")
    print("Plot saved to 'dqn_training_profile.png'.")

if __name__ == "__main__":
    train_agent()