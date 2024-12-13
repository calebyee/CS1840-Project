import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
import random
from tabular import BattleshipPlacementEnv

class RelativeDQN(nn.Module):
    def __init__(self, board_size=10, num_relative_actions=None):
        super(RelativeDQN, self).__init__()
        
        # Input features:
        # - Board state (board_size x board_size)
        # - Last hit position (2 values: x, y)
        # - Shot history (5 shots x 4 values: x, y, hit, miss)
        
        self.board_size = board_size
        self.num_actions = num_relative_actions
        
        # Convolutional layers for processing the board
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Linear layers for processing last hit position
        self.last_hit_layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        
        # Linear layers for processing shot history
        self.history_layers = nn.Sequential(
            nn.Linear(5 * 4, 64),  # 5 shots, 4 values each
            nn.ReLU()
        )
        
        # Combine all features
        conv_output_size = 64 * board_size * board_size
        combined_size = conv_output_size + 32 + 64
        
        self.combined_layers = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_relative_actions)
        )

    def forward(self, board, last_hit, shot_history):
        # Process board through conv layers
        board = board.unsqueeze(1)  # Add channel dimension
        board_features = self.conv_layers(board)
        
        # Process last hit position
        last_hit_features = self.last_hit_layers(last_hit)
        
        # Process shot history
        history_features = self.history_layers(shot_history)
        
        # Combine all features
        combined = torch.cat([board_features, last_hit_features, history_features], dim=1)
        
        # Output Q-values for each action
        return self.combined_layers(combined)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, board_size, num_actions, device="cpu"):
        self.device = device
        self.board_size = board_size
        self.num_actions = num_actions
        
        # Networks
        self.policy_net = RelativeDQN(board_size, num_actions).to(device)
        self.target_net = RelativeDQN(board_size, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.target_update = 10
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(10000)
        self.steps = 0
        
    def process_state(self, state_dict):
        """Convert state dictionary to tensor inputs"""
        board = torch.FloatTensor(state_dict['board']).to(self.device)
        
        # Process last hit
        if state_dict['last_hit'] is None:
            last_hit = torch.zeros(2).to(self.device)
        else:
            last_hit = torch.FloatTensor(state_dict['last_hit']).to(self.device)
        
        # Process shot history
        history = state_dict['shot_history']
        history_tensor = torch.zeros(5 * 4).to(self.device)
        for i, (pos, hit) in enumerate(history[:5]):
            idx = i * 4
            history_tensor[idx:idx+2] = torch.FloatTensor(pos)
            history_tensor[idx+2] = float(hit)
            history_tensor[idx+3] = float(not hit)
            
        return board, last_hit, history_tensor

    def choose_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            board, last_hit, history = self.process_state(state)
            q_values = self.policy_net(board.unsqueeze(0), 
                                     last_hit.unsqueeze(0), 
                                     history.unsqueeze(0))
            
            # Mask invalid actions
            mask = torch.full((self.num_actions,), float('-inf')).to(self.device)
            mask[valid_actions] = 0
            q_values = q_values.squeeze(0) + mask
            
            return int(q_values.argmax().item())

    def update(self, state, action, reward, next_state, done, valid_next_actions):
        # Store transition
        self.memory.push(state, action, reward, next_state, done)
        
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Process batch
        state_batch = [self.process_state(s) for s in batch[0]]
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = [self.process_state(s) for s in batch[3]]
        done_batch = torch.BoolTensor(batch[4]).to(self.device)
        
        # Compute current Q values
        current_q = self.policy_net(
            torch.stack([s[0] for s in state_batch]),
            torch.stack([s[1] for s in state_batch]),
            torch.stack([s[2] for s in state_batch])
        ).gather(1, action_batch.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q = self.target_net(
                torch.stack([s[0] for s in next_state_batch]),
                torch.stack([s[1] for s in next_state_batch]),
                torch.stack([s[2] for s in next_state_batch])
            ).max(1)[0]
        
        # Compute target Q values
        target_q = reward_batch + (1 - done_batch.float()) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps'] 