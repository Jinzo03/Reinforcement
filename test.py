import torch
import torch.nn as nn

class DualQNetwork(nn.Module):
    """
    The brain of our RL Agent. Maps environmental states to 
    predicted future rewards for each possible action.
    """
    def __init__(self, state_dim, action_dim):
        super(DualQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim) # Outputs a score for [Move Left, Move Right]
        )
        
    def forward(self, state):
        return self.network(state)

import random
from collections import deque

class ExperienceReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        # Samples memories randomly, breaking chronological bias
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.FloatTensor(state), 
                torch.LongTensor(action), 
                torch.FloatTensor(reward), 
                torch.FloatTensor(next_state), 
                torch.FloatTensor(done))
                
    def __len__(self):
        return len(self.buffer)