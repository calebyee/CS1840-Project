import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random
import numpy as np
import torch.nn.functional as F
import math

class ActionPatternTracker:
    def __init__(self):
        self.successful_patterns = []
        self.current_sequence = []
        self.max_patterns = 1000
        
    def add_action(self, action, hit):
        self.current_sequence.append((action, hit))
        if hit and len(self.current_sequence) > 1:
            # Store successful sequences
            if len(self.successful_patterns) >= self.max_patterns:
                self.successful_patterns.pop(0)  # Remove oldest pattern
            self.successful_patterns.append(self.current_sequence.copy())
        if not hit:
            self.current_sequence = []
            
    def get_suggested_action(self, valid_actions, state):
        if not self.successful_patterns or not valid_actions:
            return None
            
        # Get current hits from state
        state_array = np.asarray(state)
        if len(state_array.shape) > 2:
            state_array = state_array.squeeze()  # Remove extra dimensions
            
        # Ensure state array is 2D
        if len(state_array.shape) != 2:
            return None
            
        # Get hits from the 2D state array
        hits = [(i, j) for i in range(state_array.shape[0]) for j in range(state_array.shape[1]) 
                if np.any(state_array[i,j] == 1)]
        
        if not hits:
            return None
            
        # Look for patterns similar to current sequence
        for pattern in reversed(self.successful_patterns):  # Check recent patterns first
            if len(pattern) > len(hits):
                # Check if current hits match start of pattern
                pattern_start = pattern[:len(hits)]
                if self._matches_pattern(hits, pattern_start):
                    next_action = pattern[len(hits)][0]
                    if next_action in valid_actions:
                        return next_action
        return None
    
    def _matches_pattern(self, hits, pattern):
        # Convert hits to action sequence
        hit_actions = [h[0] * 10 + h[1] for h in hits]
        pattern_actions = [p[0] for p in pattern if p[1]]  # Only consider hits
        
        if len(hit_actions) != len(pattern_actions):
            return False
            
        # Check if the sequences match after normalizing positions
        offset = pattern_actions[0] - hit_actions[0]
        return all((ha + offset) % 100 == pa for ha, pa in zip(hit_actions[1:], pattern_actions[1:]))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.success_flags = np.zeros((capacity,), dtype=bool)  # Track successful moves
        self.position = 0
        self.size = 0
        
    def push(self, state, action, reward, next_state, done, hit=False):
        """Save a transition with success tracking"""
        max_priority = max(self.priorities) if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        # Store transition
        self.memory[self.position] = (state, action, reward, next_state, done)
        
        # Increase priority for successful actions
        if hit:
            max_priority *= 2.0
            self.success_flags[self.position] = True
        else:
            self.success_flags[self.position] = False
        
        self.priorities[self.position] = max_priority
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, beta):
        """Sample a batch of transitions with success bonus"""
        if len(self.memory) == 0:
            return None
            
        # Calculate probabilities with success bonus
        priorities = self.priorities[:self.size]
        success_bonus = np.where(self.success_flags[:self.size], 2.0, 1.0)
        probs = (priorities * success_bonus) ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize weights
        
        # Get samples
        batch = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )
    
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions"""
        for idx, priority in zip(indices, priorities.flatten()):
            self.priorities[idx] = priority + 1e-5  # Small constant to ensure non-zero probabilities
            # Maintain success bonus
            if self.success_flags[idx]:
                self.priorities[idx] *= 2.0
    
    def __len__(self):
        """Return the current size of memory"""
        return self.size

class DQN(nn.Module):
    def __init__(self, input_channels, action_size, is_placement_agent=False):
        super(DQN, self).__init__()
        # Store dimensions for shape calculations
        self.input_channels = 1 if is_placement_agent else 2
        
        # Convolutional layers with smaller feature maps
        self.conv1 = nn.Conv2d(self.input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Calculate flattened size for first fully connected layer
        # For 10x10 board: After 2 conv layers with padding, spatial dimensions remain 10x10
        self.conv_output_size = 32 * 10 * 10  # 3200 to match the error message
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 256)
        self.fc2 = nn.Linear(256, action_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Ensure input is 4D: [batch_size, channels, height, width]
        if len(x.shape) == 5:  # If shape is [batch, 1, channels, height, width]
            x = x.squeeze(1)  # Remove the extra dimension
        elif len(x.shape) == 3:  # If shape is [channels, height, width]
            x = x.unsqueeze(0)  # Add batch dimension
        elif len(x.shape) == 2:  # If shape is [height, width]
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, device="cpu", is_placement_agent=False):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.is_placement_agent = is_placement_agent
        
        # Hyperparameters
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05  # Reduced minimum epsilon
        self.epsilon_decay = 0.997  # Faster decay
        self.batch_size = 64
        self.learning_rate = 0.001
        self.tau = 0.001  # soft update parameter
        
        # PER parameters
        self.alpha = 0.6  # prioritization exponent
        self.beta = 0.4  # importance sampling weight
        self.beta_increment = 0.001
        self.max_priority = 1.0
        
        # Networks
        self.policy_net = DQN(state_size, action_size, is_placement_agent).to(device)
        self.target_net = DQN(state_size, action_size, is_placement_agent).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = PrioritizedReplayBuffer(10000, self.alpha)
        
        # Pattern tracking
        self.pattern_tracker = ActionPatternTracker()
        
        # Training metrics
        self.total_steps = 0
        self.episode_rewards = []
        self.hit_rates = []
        self.losses = []
        self.q_value_history = []
        self.epsilons = []
        self.exploration_steps = 0
        self.exploitation_steps = 0
        self.action_values = {}  # Track Q-values for each action
        
        # Q-value thresholds for adaptive exploration
        self.q_value_threshold = 1.0  # Threshold for considering an action high-value
        
        # Reward normalization
        self.reward_deque = deque(maxlen=1000)
        self.reward_mean = 0
        self.reward_std = 1
        
        # Performance tracking
        self.recent_hit_rates = deque(maxlen=100)
        self.recent_rewards = deque(maxlen=100)
        
        # For epsilon decay
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
    
    def update_reward_stats(self, reward):
        """Update running statistics for reward normalization"""
        self.reward_deque.append(reward)
        if len(self.reward_deque) > 1:
            self.reward_mean = np.mean(self.reward_deque)
            self.reward_std = np.std(self.reward_deque) + 1e-8
    
    def normalize_reward(self, reward):
        """Normalize reward using running statistics"""
        self.update_reward_stats(reward)
        if len(self.reward_deque) > 1:
            normalized = (reward - self.reward_mean) / self.reward_std
            return np.clip(normalized, -10, 10)  # Clip normalized rewards for stability
        return reward
    
    def update_epsilon(self):
        """Update epsilon value with much slower decay"""
        avg_q = np.mean(self.q_value_history[-100:]) if self.q_value_history else 0
        avg_hit_rate = np.mean(self.recent_hit_rates) if self.recent_hit_rates else 0
        
        if avg_q > self.q_value_threshold and avg_hit_rate > 0.3:
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.999)  # Even slower decay
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.9995)  # Very slow decay
        
        self.epsilons.append(self.epsilon)
    
    def choose_action(self, state, valid_actions):
        """Choose action using smart exploration and pattern matching"""
        if not valid_actions:
            return None
        
        # Get Q-values and update tracking
        with torch.no_grad():
            q_values = self.policy_net(state)
            valid_q_values = q_values.squeeze()[list(valid_actions)]
            max_q = float(valid_q_values.max())
            self.q_value_history.append(max_q)
            
            # Track Q-values for each action
            for action in valid_actions:
                q_val = float(q_values.squeeze()[action])
                if action not in self.action_values:
                    self.action_values[action] = []
                self.action_values[action].append(q_val)
        
        # Check pattern tracker for suggested action
        suggested_action = self.pattern_tracker.get_suggested_action(valid_actions, state.cpu().numpy().squeeze())
        if suggested_action is not None and random.random() < 0.7:  # 70% chance to use suggested action
            return suggested_action
        
        # Adaptive exploration based on Q-values
        if max_q > self.q_value_threshold:
            explore_prob = self.epsilon * 0.5  # Reduce exploration for high-value states
        else:
            explore_prob = self.epsilon
        
        if random.random() < explore_prob:
            self.exploration_steps += 1
            # Smart exploration using softmax over Q-values
            temperature = 0.5  # Lower temperature = more exploitation
            probs = F.softmax(valid_q_values / temperature, dim=0).cpu().numpy()
            return np.random.choice(list(valid_actions), p=probs)
        
        self.exploitation_steps += 1
        return list(valid_actions)[valid_q_values.argmax().item()]
    
    def update(self, state, action, reward, next_state, done, hit=False, curriculum_info=None):
        """Update agent with improved learning from successful actions"""
        # Track metrics
        self.episode_rewards.append(reward)
        if curriculum_info and 'hit_rate' in curriculum_info:
            hit_rate = curriculum_info['hit_rate']
            self.hit_rates.append(hit_rate)
            self.recent_hit_rates.append(hit_rate)
        
        # Update pattern tracker
        self.pattern_tracker.add_action(action, hit)
        
        # Normalize reward
        normalized_reward = self.normalize_reward(reward)
        
        # Increase priority for transitions that led to hits
        if hit:
            priority_multiplier = 2.0
        else:
            priority_multiplier = 1.0
        
        self.memory.push(state, action, normalized_reward, next_state, done, priority_multiplier)
        
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Rest of the update function remains the same...
        batch = self.memory.sample(self.batch_size, self.beta)
        if batch is None:
            return 0.0
        
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        # Convert to tensors and ensure proper shapes
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        if len(states.shape) == 5:
            states = states.squeeze(1)
        if len(next_states.shape) == 5:
            next_states = next_states.squeeze(1)
        
        current_Q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_Q = self.target_net(next_states).gather(1, next_actions)
            target_Q = rewards.unsqueeze(1) + (self.gamma * next_Q * (1 - dones.unsqueeze(1)))
        
        current_Q = torch.clamp(current_Q, -100, 100)
        target_Q = torch.clamp(target_Q, -100, 100)
        
        td_errors = torch.abs(current_Q - target_Q).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        loss = (weights * F.smooth_l1_loss(current_Q, target_Q, reduction='none')).mean()
        
        self.losses.append(float(loss.item()))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        
        self.update_epsilon()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        self.total_steps += 1
        return float(loss.item())
    
    def get_metrics(self):
        """Return current training metrics"""
        return {
            'episode_rewards': self.episode_rewards,
            'hit_rates': self.hit_rates,
            'losses': self.losses,
            'q_values': self.q_value_history,
            'epsilons': self.epsilons,
            'exploration_ratio': self.exploration_steps / (self.exploration_steps + self.exploitation_steps + 1e-6),
            'action_values': self.action_values
        }

    def get_epsilon(self, steps):
        # Exponential decay
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                 math.exp(-1. * steps / self.epsilon_decay)
        return max(self.epsilon_end, epsilon)  # Ensure we don't go below minimum
    
    def update_curriculum_performance(self, phase, reward):
        """Track performance for each curriculum phase"""
        self.curriculum_performance[phase].append(reward)
        
        # Calculate average performance for current phase
        phase_avg = np.mean(self.curriculum_performance[phase][-100:])  # Last 100 episodes
        return phase_avg

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
                        potential_map[r, c] = 1.0
            
            return np.stack([shots_grid, potential_map])

    def calculate_reward(self, hit, sunk_ship=False, game_over=False):
        if game_over:
            return 100
        if sunk_ship:
            return 50
        if hit:
            return 20
        return -1

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

    def train(self, num_episodes=50):
        total_rewards = []
        for episode in range(num_episodes):
            print(f"\n========================= Episode {episode + 1}/{num_episodes} =========================\n")
            state = self.env.reset()
            done = False
            episode_reward = 0
            steps_in_episode = 0
            
            while not done:
                # Update epsilon based on total steps
                self.epsilon = self.get_epsilon(self.total_steps)
                
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                steps_in_episode += 1
                self.total_steps += 1
                
                # Train after every step if we have enough samples
                if len(self.memory) >= self.batch_size:
                    loss = self.train_step()
                    
                    # Log training metrics every 5 steps
                    if steps_in_episode % 5 == 0:
                        self.log_metrics(episode, steps_in_episode, loss, reward, info)
                
                # Update target network periodically
                if self.total_steps % 100 == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            
            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1} finished with reward: {episode_reward:.2f}, epsilon: {self.epsilon:.4f}")
        
        return total_rewards

    def train_step(self):
        # Update beta for importance sampling
        self.update_beta()
        
        # Sample batch with priorities
        result = self.memory.sample(self.batch_size, self.beta)
        if result is None:
            return 0.0
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices, weights = result
        
        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Ensure proper dimensions for states
        if len(state_batch.shape) == 3:  # (batch, height, width)
            state_batch = state_batch.unsqueeze(1)  # Add channel dimension
        if len(next_state_batch.shape) == 3:
            next_state_batch = next_state_batch.unsqueeze(1)
        
        # Get current Q values
        current_Q = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_Q = self.target_net(next_state_batch).max(1)[0]
            target_Q = reward_batch + (self.gamma * next_Q * (1 - done_batch))
        
        # Compute TD errors for updating priorities
        td_errors = torch.abs(current_Q.squeeze() - target_Q).detach().cpu().numpy()
        
        # Update priorities in buffer
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # Compute loss with importance sampling weights
        loss = (weights * F.smooth_l1_loss(current_Q, target_Q.unsqueeze(1), reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

    def log_metrics(self, episode, steps, loss, reward, info):
        print(f"[Learning Update] Step {steps}")
        print(f"Loss: {loss:.4f}")
        print(f"Reward: {reward:.4f}")
        print(f"Epsilon: {self.epsilon:.4f}")
        if info:
            print(f"Hit Rate: {info.get('hit_rate', 0):.2f}%")

    def select_action(self, state):
        # Convert state to tensor and ensure proper shape
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        # Add batch and channel dimensions if needed
        if len(state.shape) == 2:  # Just height x width
            state = state.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif len(state.shape) == 3:  # Has channel dimension but no batch
            state = state.unsqueeze(0)  # Add batch dimension
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.action_size)

    def update_beta(self):
        """Update beta value for importance sampling"""
        self.beta = min(self.beta_end, self.beta + self.beta_increment)

def train_nested_mdp(init_env, active_env, num_episodes=50):
    """Train the placement agent for the nested MDP.
    
    Args:
        init_env: Initial environment for ship placement
        active_env: Active environment for gameplay
        num_episodes: Number of episodes to train for (default: 50)
    
    Returns:
        Trained DQNAgent
    """
    # Initialize agent
    outer_agent = DQNAgent(
        state_size=init_env.board_size,
        action_size=init_env.action_space.n,
        device="cpu",
        is_placement_agent=True
    )
    
    total_steps = 0
    for episode in range(num_episodes):
        print(f"\n========================= Episode {episode + 1}/{num_episodes} =========================\n")
        
        # Reset environment
        state = init_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Select and perform action
            action = outer_agent.select_action(state)
            next_state, reward, done, info = init_env.step(action)
            
            # Extract hit information from info if available
            hit = info.get('hit', False)
            
            # Store transition and train
            loss = outer_agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                hit=hit,
                curriculum_info=info
            )
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # Log progress every 100 steps
            if total_steps % 100 == 0:
                print(f"Step {total_steps}: Loss = {loss:.4f}, Epsilon = {outer_agent.epsilon:.4f}")
        
        print(f"Episode {episode + 1} finished with reward: {episode_reward:.2f}")
    
    return outer_agent
