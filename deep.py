import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from dqn import DQNAgent
import torch
from collections import deque

# BattleshipPlacementEnv
class BattleshipPlacementEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board_size=10, ship_sizes=[5, 4, 3, 3, 2]):
        super(BattleshipPlacementEnv, self).__init__()
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.current_ship_index = 0
        
        # Single integer action space
        self.action_space = spaces.Discrete(board_size * board_size * 2)  # position * 2 orientations
        self.observation_space = spaces.Box(low=0, high=1, shape=(board_size, board_size), dtype=np.int8)
        
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_ship_index = 0
        self.done = False
        self.ship_positions = []  # Store ship positions and sizes for reward calculation
        return self.board

    def step(self, action):
        """
        Takes an action and returns the next state, reward, done, and info
        Action is now a single integer encoding (row * board_size * 2) + (col * 2) + orientation
        """
        # Convert single integer action back to row, col, orientation
        action = int(action)  # Ensure action is an integer
        
        # Decode action
        orientation = action % 2
        position = action // 2
        row = position // self.board_size
        col = position % self.board_size
        
        ship_size = self.ship_sizes[self.current_ship_index]
        
        # Check if placement is valid
        if not self._is_valid_placement(row, col, ship_size, orientation):
            return self.board, -1, False, {"error": "Invalid placement"}
        
        # Place the ship
        self._place_ship(row, col, ship_size, orientation)
        self.current_ship_index += 1
        
        # Check if all ships are placed
        if self.current_ship_index >= len(self.ship_sizes):
            self.done = True
            reward = 1.0  # Reward for completing placement
        else:
            reward = 0.1  # Small positive reward for valid placement
        
        return self.board, reward, self.done, {}

    def _is_valid_placement(self, row, col, ship_size, orientation):
        if orientation == 0:
            if col + ship_size > self.board_size or np.any(self.board[row, col:col + ship_size] == 1):
                return False
        else:
            if row + ship_size > self.board_size or np.any(self.board[row:row + ship_size, col] == 1):
                return False
        return True

    def _place_ship(self, row, col, ship_size, orientation):
        if orientation == 0:
            self.board[row, col:col + ship_size] = 1
        else:
            self.board[row:row + ship_size, col] = 1

    def render(self, mode='human'):
        print(self.board)

# BattleshipAttackEnv
class BattleshipAttackEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board_size=10):
        super(BattleshipAttackEnv, self).__init__()
        self.board_size = board_size
        self.ship_sizes = [5, 4, 3, 3, 2]  # All standard Battleship ships
        
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.board_size, self.board_size), dtype=np.int8)
        self.reset()
    
    def reset(self):
        self.agent_board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.opponent_board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.agent_shots = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        
        self._place_ships()
        
        return self.agent_shots

    def _place_ships(self):
        """Place ships on the board"""
        self.opponent_board.fill(0)
        
        for ship_size in self.ship_sizes:
            placed = False
            max_attempts = 100
            attempts = 0
            
            while not placed and attempts < max_attempts:
                row = np.random.randint(0, self.board_size)
                col = np.random.randint(0, self.board_size)
                orientation = np.random.randint(0, 2)
                
                if self._is_valid_ship_placement(row, col, ship_size, orientation):
                    self._place_ship(row, col, ship_size, orientation)
                    placed = True
                attempts += 1
            
            if not placed:
                # Silently continue if ship placement fails
                pass

    def _is_valid_ship_placement(self, row, col, ship_size, orientation):
        """Check if ship placement is valid"""
        if orientation == 0:  # horizontal
            if col + ship_size > self.board_size:
                return False
            return not np.any(self.opponent_board[row, col:col + ship_size] == 1)
        else:  # vertical
            if row + ship_size > self.board_size:
                return False
            return not np.any(self.opponent_board[row:row + ship_size, col] == 1)

    def _place_ship(self, row, col, ship_size, orientation):
        """Place a ship on the board"""
        if orientation == 0:  # horizontal
            self.opponent_board[row, col:col + ship_size] = 1
        else:  # vertical
            self.opponent_board[row:row + ship_size, col] = 1

    def step(self, action):
        # Convert action to coordinates
        row = action // self.board_size
        col = action % self.board_size
        
        # Process the shot
        shots = self.agent_shots
        if shots[row, col] != 0:
            return shots, -1, False, {"error": "Position already shot"}
        
        # Update game state
        total_ship_cells = np.sum(self.opponent_board == 1)
        hit_ship_cells = np.sum((self.opponent_board == 1) & (shots == 1))
        
        # Calculate hit rate and efficiency
        total_shots = np.sum(np.abs(shots))
        hit_rate = hit_ship_cells / total_shots if total_shots > 0 else 0
        efficiency = hit_ship_cells / total_shots if total_shots > 0 else 0
        
        # Record shot and check hit
        if self.opponent_board[row, col] == 1:
            shots[row, col] = 1
            streak_bonus = 1.0 * (hit_ship_cells / total_shots if total_shots > 0 else 0)
            reward = 3.0 + streak_bonus
        else:
            shots[row, col] = -1
            base_penalty = -0.1
            reward = base_penalty
        
        # Add efficiency bonus
        if efficiency > 0.3:
            reward += 0.5 * efficiency
        
        # Game ends only when all ships are sunk
        done = hit_ship_cells == total_ship_cells
        
        if done:
            reward += 5.0 * efficiency
        
        return shots, reward, done, {
            "total_ships": total_ship_cells,
            "hits": hit_ship_cells,
            "hit_rate": hit_rate,
            "efficiency": efficiency,
        }

    def render(self, mode='human'):
        print("Agent's Board (our ships):")
        print(self.agent_board)
        print("\nOpponent's Board (their ships):")
        print(self.opponent_board)
        print("\nAgent's Shots:")
        print(self.agent_shots)
        print("\nOpponent's Shots:")
        print(self.opponent_shots)

def get_valid_actions(env, shots):
    """Helper function to get valid actions (unshot positions)"""
    valid_actions = []
    for i in range(env.board_size):
        for j in range(env.board_size):
            if shots[i, j] == 0:  # Position hasn't been shot at
                action = i * env.board_size + j
                valid_actions.append(action)
    return valid_actions

def get_valid_placement_actions(env):
    """Get valid actions for ship placement"""
    valid_actions = []
    ship_size = env.ship_sizes[env.current_ship_index]
    
    # For each position and orientation
    for row in range(env.board_size):
        for col in range(env.board_size):
            for orientation in range(2):  # 0: horizontal, 1: vertical
                # Check if placement would be valid
                if orientation == 0:  # horizontal
                    if col + ship_size <= env.board_size:
                        if not np.any(env.board[row, col:col + ship_size]):
                            # Convert to single action index
                            action = (row * env.board_size * 2) + (col * 2) + orientation
                            valid_actions.append(action)
                else:  # vertical
                    if row + ship_size <= env.board_size:
                        if not np.any(env.board[row:row + ship_size, col]):
                            # Convert to single action index
                            action = (row * env.board_size * 2) + (col * 2) + orientation
                            valid_actions.append(action)
    
    # print(f"[DEBUG] Generated {len(valid_actions)} valid actions for ship size {ship_size}")
    # print(f"[DEBUG] Sample actions: {valid_actions[:5]}")
    return valid_actions

def train_nested_mdp(init_env, active_env, num_episodes=50):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agents
    outer_agent = DQNAgent(
        state_size=init_env.board_size,
        action_size=init_env.action_space.n,
        device=device,
        is_placement_agent=True
    )
    
    inner_agent = DQNAgent(
        state_size=active_env.board_size,
        action_size=active_env.action_space.n,
        device=device,
        is_placement_agent=False
    )
    
    # Training loop
    for episode in range(num_episodes):
        print(f"\n{'='*25} Episode {episode+1}/{num_episodes} {'='*25}")
        
        # Reset environments
        init_state = init_env.reset()
        active_state = active_env.reset()
        
        # Place ships first
        init_state_tensor = torch.FloatTensor(init_state).unsqueeze(0).to(device)
        valid_init_actions = get_valid_placement_actions(init_env)
        
        if not valid_init_actions:
            print("[ERROR] No valid initialization actions!")
            continue
            
        action = outer_agent.choose_action(init_state_tensor, valid_init_actions)
        next_init_state, init_reward, init_done, init_info = init_env.step(action)
        
        if 'error' in init_info:
            print(f"[ERROR] Init step error: {init_info['error']}")
            continue
        
        # Play the game
        episode_reward = 0
        done = False
        total_shots = 0
        hits = 0
        
        while not done:
            # Create 2-channel state
            shots = (active_state == -1).astype(np.float32)
            hits_map = (active_state == 1).astype(np.float32)
            state_tensor = torch.FloatTensor(np.stack([shots, hits_map])).unsqueeze(0).to(device)
            
            # Get valid actions
            valid_actions = set(range(active_env.action_space.n))
            for i in range(active_env.board_size):
                for j in range(active_env.board_size):
                    action_idx = i * active_env.board_size + j
                    if active_state[i, j] != 0:
                        valid_actions.discard(action_idx)
            
            if not valid_actions:
                break
            
            # Choose and perform action
            action = inner_agent.choose_action(state_tensor, list(valid_actions))
            if action is None:
                break
            
            next_state, reward, done, info = active_env.step(action)
            
            if 'error' in info:
                print(f"[ERROR] Step error: {info['error']}")
                continue
            
            # Update statistics
            total_shots += 1
            if reward > 0:
                hits += 1
            
            # Create next state tensor
            next_shots = (next_state == -1).astype(np.float32)
            next_hits = (next_state == 1).astype(np.float32)
            next_state_tensor = torch.FloatTensor(np.stack([next_shots, next_hits])).unsqueeze(0).to(device)
            
            # Update inner agent
            loss = inner_agent.update(
                state_tensor, 
                action, 
                reward, 
                next_state_tensor, 
                done,
                hit=(reward > 0),
                curriculum_info=info
            )
            
            # Print progress every 5 moves
            if total_shots % 5 == 0:
                hit_rate = hits / total_shots if total_shots > 0 else 0
                print(f"Move {total_shots}: Reward={reward:.2f}, Hit Rate={hit_rate:.2f}, Epsilon={inner_agent.epsilon:.2f}")
            
            active_state = next_state
            episode_reward += reward
        
        # Update outer agent
        next_init_state_tensor = torch.FloatTensor(next_init_state).unsqueeze(0).to(device)
        outer_agent.update(
            init_state_tensor,
            action,
            episode_reward,
            next_init_state_tensor,
            init_done
        )
        
        # Print episode summary
        print(f"\nEpisode Summary:")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Hit Rate: {hit_rate:.2f}")
        print(f"Total Shots: {total_shots}")
        print(f"Epsilon: {inner_agent.epsilon:.2f}")
    
    # Plot training metrics
    metrics = inner_agent.get_metrics()
    
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 3, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot hit rates
    plt.subplot(2, 3, 2)
    plt.plot(metrics['hit_rates'])
    plt.title('Hit Rates')
    plt.xlabel('Episode')
    plt.ylabel('Hit Rate')
    
    # Plot losses
    plt.subplot(2, 3, 3)
    plt.plot(metrics['losses'])
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    
    # Plot Q-values
    plt.subplot(2, 3, 4)
    plt.plot(metrics['q_values'])
    plt.title('Average Q-Values')
    plt.xlabel('Step')
    plt.ylabel('Q-Value')
    
    # Plot epsilon decay
    plt.subplot(2, 3, 5)
    plt.plot(metrics['epsilons'])
    plt.title('Epsilon Decay')
    plt.xlabel('Step')
    plt.ylabel('Epsilon')
    
    # Plot exploration ratio
    plt.subplot(2, 3, 6)
    plt.axhline(y=metrics['exploration_ratio'], color='r', linestyle='-')
    plt.title('Exploration Ratio')
    plt.xlabel('Step')
    plt.ylabel('Ratio')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    
    return outer_agent, inner_agent

# Create and train the environments
if __name__ == "__main__":
    init_env = BattleshipPlacementEnv()
    active_env = BattleshipAttackEnv()
    train_nested_mdp(init_env, active_env)
