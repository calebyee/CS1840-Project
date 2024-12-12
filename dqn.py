import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random
import numpy as np

class BattleshipMemory:
    def __init__(self, max_patterns=1000):
        self.successful_patterns = []
        self.max_patterns = max_patterns
        
    def add_pattern(self, hit_sequence):
        if len(hit_sequence) > 1:
            if len(self.successful_patterns) >= self.max_patterns:
                self.successful_patterns.pop(0)
            self.successful_patterns.append(hit_sequence)
    
    def get_suggested_action(self, current_state, valid_actions):
        if not self.successful_patterns or not valid_actions:
            return None
            
        # Find recent hits in current state
        hits = [(i, j) for i in range(10) for j in range(10) if current_state[i,j] == 1]
        if not hits:
            return None
            
        # Look for similar patterns
        last_hit = hits[-1]
        for pattern in reversed(self.successful_patterns):  # Check recent patterns first
            if len(pattern) > len(hits):
                # Check if current hits match start of pattern
                pattern_start = pattern[:len(hits)]
                if self._matches_pattern(hits, pattern_start):
                    next_pos = pattern[len(hits)]
                    action = next_pos[0] * 10 + next_pos[1]
                    if action in valid_actions:
                        return action
        return None
    
    def _matches_pattern(self, hits, pattern):
        # Normalize positions relative to first hit
        hit_offset = hits[0]
        pattern_offset = pattern[0]
        
        normalized_hits = [(h[0] - hit_offset[0], h[1] - hit_offset[1]) for h in hits]
        normalized_pattern = [(p[0] - pattern_offset[0], p[1] - pattern_offset[1]) for p in pattern]
        
        return normalized_hits == normalized_pattern

class DQN(nn.Module):
    def __init__(self, board_size=10, action_size=100, input_channels=1):
        super(DQN, self).__init__()
        
        self.input_channels = input_channels
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        conv_out_size = 128 * board_size * board_size
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        if len(state.shape) == 2:
            state = state.unsqueeze(0).unsqueeze(0)
            
        # Ensure correct number of channels
        if state.shape[1] != self.input_channels:
            if self.input_channels == 2 and state.shape[1] == 1:
                # Add a second channel of zeros for placement phase
                zeros = torch.zeros_like(state)
                state = torch.cat([state, zeros], dim=1)
            
        x = self.conv(state.float())
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_size, action_size, device="cpu", is_placement_agent=False):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.is_placement_agent = is_placement_agent
        
        # Use different number of input channels based on agent type
        input_channels = 1 if is_placement_agent else 2
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.target_update_frequency = 10
        
        self.policy_net = DQN(state_size, action_size, input_channels).to(device)
        self.target_net = DQN(state_size, action_size, input_channels).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(capacity=10000)
        self.pattern_memory = BattleshipMemory()
        self.current_hit_sequence = []
        self.steps = 0
        
    def get_state_representation(self, state, last_hit_pos=None):
        if self.is_placement_agent:
            return state
        else:
            # For attack phase, create 2-channel state
            shots_grid = state.copy()
            potential_map = np.zeros_like(shots_grid)
            hits = [(i, j) for i in range(10) for j in range(10) if shots_grid[i,j] == 1]
            
            # Add potential targets around hits
            for hit_pos in hits:
                row, col = hit_pos
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    r, c = row + dr, col + dc
                    if 0 <= r < 10 and 0 <= c < 10 and shots_grid[r, c] == 0:
                        potential_map[r, c] = 0.5
            
            # Increase potential for squares adjacent to last hit
            if last_hit_pos is not None:
                row, col = last_hit_pos
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    r, c = row + dr, col + dc
                    if 0 <= r < 10 and 0 <= c < 10 and shots_grid[r, c] == 0:
                        potential_map[r, c] = 1.0  # Higher potential for recent hits
            
            return np.stack([shots_grid, potential_map])

    def calculate_reward(self, hit, sunk_ship=False, game_over=False):
        if game_over:
            return 100
        if sunk_ship:
            return 50
        if hit:
            return 20
        return -1

    def choose_action(self, state, valid_actions):
        if len(valid_actions) == 0:
            return None
            
        if random.random() < self.epsilon:
            return random.choice(list(valid_actions))
        
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.FloatTensor(state).to(self.device)
            q_values = self.policy_net(state)
            
            # Convert valid_actions to tensor indices
            valid_actions_list = [int(x) for x in valid_actions]  # Convert to Python ints
            mask = torch.full((self.action_size,), float('-inf'), device=self.device)
            mask[valid_actions_list] = 0
            
            # Add the mask to q_values
            q_values = q_values.squeeze() + mask
            
            return int(q_values.argmax().item())

    def update(self, state, action, reward, next_state, done, hit=False):
        # Update hit sequence
        if hit:
            row, col = action // 10, action % 10
            self.current_hit_sequence.append((row, col))
        elif self.current_hit_sequence:
            # Store successful hit sequence when it ends
            self.pattern_memory.add_pattern(self.current_hit_sequence)
            self.current_hit_sequence = []
        
        self.memory.push(state, action, reward, next_state, done)
        
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.stack(batch[0]).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.cat(batch[2]).to(self.device)
        next_state_batch = torch.stack(batch[3]).to(self.device)
        done_batch = torch.cat(batch[4]).to(self.device)
        
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch.float()) * self.gamma * next_q_values
        
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save the agent's networks and training state"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'is_placement_agent': self.is_placement_agent
        }, path)

    def load(self, path):
        """Load the agent's networks and training state"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.is_placement_agent = checkpoint['is_placement_agent']

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state)
        if not torch.is_tensor(next_state):
            next_state = torch.FloatTensor(next_state)
        if not torch.is_tensor(reward):
            reward = torch.FloatTensor([reward])
        if not torch.is_tensor(done):
            done = torch.BoolTensor([done])
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
