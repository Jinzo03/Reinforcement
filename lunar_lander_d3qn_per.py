import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

ENV_NAME            = "LunarLander-v3"
GAMMA               = 0.99
LR                  = 1e-4
BATCH_SIZE          = 64
BUFFER_CAPACITY     = 100000
MIN_REPLAY_SIZE     = 5000
EPSILON_START       = 1.0
EPSILON_END         = 0.05
EPSILON_DECAY_STEPS = 200000
TAU                 = 5e-4
MAX_EPISODES        = 3000      # extended

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        features   = self.feature_network(state)
        values     = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))


class SumTree:
    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1)
        self.data      = np.zeros(capacity, dtype=object)
        self.write     = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, idx, s):
        while True:
            left  = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s  -= self.tree[left]
                idx = right

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx      = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_steps=300000):
        self.tree         = SumTree(capacity)
        self.alpha        = alpha
        self.beta         = beta_start
        self.beta_start   = beta_start
        self.beta_steps   = beta_steps
        self.step_count   = 0
        self.epsilon      = 1e-5
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.tree.add(self.max_priority, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        indices, priorities = [], []

        self.step_count += 1
        self.beta = min(1.0, self.beta_start + self.step_count * (1.0 - self.beta_start) / self.beta_steps)

        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            indices.append(idx)
            states.append(data[0]);     actions.append(data[1])
            rewards.append(data[2]);    next_states.append(data[3])
            dones.append(data[4])

        probs    = np.array(priorities) / (self.tree.total() + 1e-8)
        weights  = (self.tree.n_entries * probs) ** (-self.beta)
        weights /= weights.max() + 1e-8

        return (torch.FloatTensor(np.array(states)),
                torch.LongTensor(np.array(actions)).unsqueeze(1),
                torch.FloatTensor(np.array(rewards)).unsqueeze(1),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(np.array(dones)).unsqueeze(1),
                indices,
                torch.FloatTensor(weights).unsqueeze(1))

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            p = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            if p > self.max_priority:
                self.max_priority = p


def shape_reward(state, next_state, reward, done):
    pos_x, pos_y, vel_x, vel_y, angle, ang_vel, leg1, leg2 = next_state
    shaped = reward

    if pos_y > 0.1 and not (leg1 or leg2):
        shaped -= 0.1 * pos_y

    if leg1 and leg2:
        shaped += 5.0

    if pos_y < 0.3:
        shaped -= abs(angle) * 2.0

    return shaped


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.online_net  = DuelingDQN(state_dim, action_dim).to(DEVICE)
        self.target_net  = DuelingDQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer   = optim.Adam(self.online_net.parameters(), lr=LR)
        self.memory      = PrioritizedReplayBuffer(BUFFER_CAPACITY)
        self.total_steps = 0

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            return self.online_net(
                torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            ).argmax(dim=1).item()

    def train_step(self):
        if self.memory.tree.n_entries < MIN_REPLAY_SIZE:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(BATCH_SIZE)
        states      = states.to(DEVICE);      actions     = actions.to(DEVICE)
        rewards     = rewards.to(DEVICE);     next_states = next_states.to(DEVICE)
        dones       = dones.to(DEVICE);       weights     = weights.to(DEVICE)

        Q_current = self.online_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            Q_next       = self.target_net(next_states).gather(1, next_actions)
            Q_target     = rewards + (1.0 - dones) * GAMMA * Q_next

        td_errors = torch.abs(Q_current - Q_target).detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors)

        loss = (weights * F.smooth_l1_loss(Q_current, Q_target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(TAU * op.data + (1.0 - TAU) * tp.data)


def main():
    env   = gym.make(ENV_NAME)
    agent = DQNAgent(state_dim=8, action_dim=4)

    score_history       = []
    rolling_avg_history = []
    epsilon             = EPSILON_START
    best_avg            = -np.inf

    print(f"Device: {DEVICE} | Training LunarLander-v3 | MAX_EPISODES: {MAX_EPISODES}")

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        episode_score = 0
        done = False

        while not done:
            action = agent.get_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            shaped = shape_reward(state, next_state, reward, done)
            agent.memory.push(state, action, shaped, next_state, float(done))

            state          = next_state
            episode_score += reward
            agent.total_steps += 1
            epsilon = max(EPSILON_END, EPSILON_START - agent.total_steps / EPSILON_DECAY_STEPS)
            agent.train_step()

        score_history.append(episode_score)
        rolling_avg = np.mean(score_history[-100:])
        rolling_avg_history.append(rolling_avg)

        # save best checkpoint whenever rolling avg improves
        if len(score_history) >= 100 and rolling_avg > best_avg:
            best_avg = rolling_avg
            torch.save(agent.online_net.state_dict(), "lunar_best.pth")
            print(f"    --> Best checkpoint saved (Avg: {best_avg:.1f})")

        if episode % 25 == 0:
            print(f"Ep [{episode}/{MAX_EPISODES}] | Score: {episode_score:.1f} | "
                  f"Avg100: {rolling_avg:.1f} | ε: {epsilon:.3f} | Steps: {agent.total_steps}")

        if rolling_avg >= 200.0 and len(score_history) >= 100:
            print(f"\n[SOLVED] Episode {episode}! Best avg: {best_avg:.1f}")
            torch.save(agent.online_net.state_dict(), "lunar_solved.pth")
            break

    # reload best weights in case final policy regressed
    print(f"\nReloading best checkpoint (avg: {best_avg:.1f})")
    agent.online_net.load_state_dict(torch.load("lunar_best.pth", weights_only=True))
    torch.save(agent.online_net.state_dict(), "lunar_final.pth")

    env.close()

    plt.figure(figsize=(12, 5))
    plt.plot(score_history, alpha=0.3, label="Raw Score", color="blue")
    plt.plot(rolling_avg_history, label="100-Ep Average", color="red", linewidth=2)
    plt.axhline(y=200,  color="green",  linestyle="--", label="Solved (+200)")
    plt.axhline(y=-100, color="orange", linestyle=":",  label="Hover plateau (-100)")
    plt.title("Dueling Double DQN + PER — LunarLander-v3")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lunar_lander_training_profile.png")
    print(f"Done. Best avg: {best_avg:.1f} | Plot saved.")

if __name__ == "__main__":
    main()