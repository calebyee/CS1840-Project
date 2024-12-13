import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from dqn import DQNAgent
import torch

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
        self.max_board_size = board_size
        self.curr_board_size = 6  # Start with smaller board
        self.curr_ship_sizes = [5, 4]  # Start with only large ships
        self.curriculum_phase = 0
        self.episodes_per_phase = 15
        self.max_phases = 4
        self.min_phase_performance = 0.2
        self.phase_hit_rates = []
        
        # Dynamic move limit calculation
        self.base_moves = {  # Base moves per phase
            0: 14,  # Phase 0: (5+4)*1.5 ≈ 14 moves
            1: 18,  # Phase 1: (5+4+3)*1.5 ≈ 18 moves
            2: 22,  # Phase 2: (5+4+3+3)*1.5 ≈ 22 moves
            3: 26,  # Phase 3: (5+4+3+3+2)*1.5 ≈ 26 moves
        }
        self.efficiency_bonus = 0  # Additional moves based on performance
        self.max_efficiency_bonus = 8  # Maximum additional moves
        
        self.action_space = spaces.Discrete(self.max_board_size * self.max_board_size)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_board_size, self.max_board_size), dtype=np.int8)
        self.reset()

    def get_move_limit(self):
        """Calculate dynamic move limit based on phase and performance"""
        base_moves = self.base_moves[self.curriculum_phase]
        return base_moves + self.efficiency_bonus

    def update_move_limit(self, hit_rate):
        """Update move limit based on performance"""
        if hit_rate > 0.3:  # Good performance
            self.efficiency_bonus = min(self.max_efficiency_bonus, self.efficiency_bonus + 1)
        elif hit_rate < 0.15:  # Poor performance
            self.efficiency_bonus = max(0, self.efficiency_bonus - 1)

    def update_curriculum(self, hit_rate):
        """Update curriculum difficulty based on performance"""
        self.phase_hit_rates.append(hit_rate)
        
        # Only consider advancing phase after minimum episodes
        if len(self.phase_hit_rates) >= self.episodes_per_phase:
            avg_hit_rate = np.mean(self.phase_hit_rates[-self.episodes_per_phase:])
            
            if avg_hit_rate >= self.min_phase_performance:
                old_phase = self.curriculum_phase
                self.curriculum_phase = min(self.max_phases, self.curriculum_phase + 1)
                
                if old_phase != self.curriculum_phase:
                    # Update board size
                    self.curr_board_size = min(self.max_board_size, 6 + self.curriculum_phase)
                    
                    # Update ship sizes
                    if self.curriculum_phase == 1:
                        self.curr_ship_sizes = [5, 4, 3]  # Add cruiser
                    elif self.curriculum_phase == 2:
                        self.curr_ship_sizes = [5, 4, 3, 3]  # Add second cruiser
                    elif self.curriculum_phase == 3:
                        self.curr_ship_sizes = [5, 4, 3, 3, 2]  # Add destroyer
                    
                    # Reset hit rates for new phase
                    self.phase_hit_rates = []
                    return True
        return False

    def _get_valid_actions(self):
        """Get valid actions for current curriculum phase"""
        valid_actions = set()
        
        # Calculate the actual playable area based on current phase
        for i in range(self.curr_board_size):
            for j in range(self.curr_board_size):
                # Map to full board coordinates
                scaled_i = int(i * (self.max_board_size / self.curr_board_size))
                scaled_j = int(j * (self.max_board_size / self.curr_board_size))
                action = scaled_i * self.max_board_size + scaled_j
                valid_actions.add(action)
        
        return valid_actions

    def reset(self):
        # Initialize full-size boards
        self.agent_board = np.zeros((self.max_board_size, self.max_board_size), dtype=np.int8)
        self.opponent_board = np.zeros((self.max_board_size, self.max_board_size), dtype=np.int8)
        self.agent_shots = np.zeros((self.max_board_size, self.max_board_size), dtype=np.int8)
        
        # Place ships according to current curriculum
        self._place_curriculum_ships()
        
        return self.agent_shots

    def _place_curriculum_ships(self):
        """Place ships based on current curriculum phase"""
        # Clear existing ships
        self.opponent_board.fill(0)
        
        # Calculate scaling factor for ship placement
        scale = self.max_board_size / self.curr_board_size
        
        for ship_size in self.curr_ship_sizes:
            placed = False
            max_attempts = 100
            attempts = 0
            
            while not placed and attempts < max_attempts:
                # Scale coordinates to current board size
                row = int(np.random.randint(0, self.curr_board_size) * scale)
                col = int(np.random.randint(0, self.curr_board_size) * scale)
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
            if col + ship_size > self.max_board_size:
                return False
            return not np.any(self.opponent_board[row, col:col + ship_size] == 1)
        else:  # vertical
            if row + ship_size > self.max_board_size:
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
        row = action // self.max_board_size
        col = action % self.max_board_size
        
        # Check if action is valid for current curriculum
        valid_actions = self._get_valid_actions()
        if action not in valid_actions:
            return self.agent_shots, -1, False, {"error": "Invalid action for current phase"}
        
        # Get current move count and limit
        moves_made = np.sum(np.abs(self.agent_shots))
        max_moves = self.get_move_limit()
        
        # Check if exceeded max moves
        if moves_made >= max_moves:
            return self.agent_shots, -2, True, {"error": "Exceeded maximum moves"}
        
        # Rest of the step function...
        shots = self.agent_shots
        if shots[row, col] != 0:
            return shots, -1, False, {"error": "Position already shot"}
            
        # Record shot and check hit
        if self.opponent_board[row, col] == 1:
            shots[row, col] = 1
            reward = 2.0  # Increased reward for hits
        else:
            shots[row, col] = -1
            # Penalty increases with number of misses
            miss_count = np.sum(shots == -1)
            reward = -0.2 - (0.1 * (miss_count / max_moves))  # Progressive penalty
            
        # Update game state
        total_ship_cells = np.sum(self.opponent_board == 1)
        hit_ship_cells = np.sum((self.opponent_board == 1) & (shots == 1))
        
        # Calculate hit rate and efficiency
        total_shots = np.sum(np.abs(shots))
        hit_rate = hit_ship_cells / total_shots if total_shots > 0 else 0
        efficiency = hit_ship_cells / moves_made if moves_made > 0 else 0
        
        # Update move limit based on performance
        self.update_move_limit(hit_rate)
        
        # Add efficiency bonus/penalty
        if efficiency > 0.3:  # Reward high efficiency
            reward += 0.5 * efficiency
        
        # Check if game is done
        done = hit_ship_cells == total_ship_cells or moves_made >= max_moves
        
        if done:
            # Add final reward based on efficiency
            if hit_ship_cells == total_ship_cells:
                moves_remaining = max_moves - moves_made
                reward += 2.0 + (moves_remaining / max_moves) * 3.0  # Bonus for finishing early
            
            # Update curriculum based on performance
            phase_changed = self.update_curriculum(hit_rate)
            if phase_changed:
                reward += 5.0
        
        return shots, reward, done, {
            "total_ships": total_ship_cells,
            "hits": hit_ship_cells,
            "hit_rate": hit_rate,
            "efficiency": efficiency,
            "moves_made": moves_made,
            "max_moves": max_moves,
            "curriculum_phase": self.curriculum_phase
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
    """Train the agents with a simplified training loop"""
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
        state_size=active_env.max_board_size,
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
        moves_made = 0
        hits = 0
        total_shots = 0
        
        while not done:
            # Create 2-channel state
            shots = (active_state == -1).astype(np.float32)
            hits_map = (active_state == 1).astype(np.float32)
            state_tensor = torch.FloatTensor(np.stack([shots, hits_map])).unsqueeze(0).to(device)
            
            # Get valid actions
            valid_actions = set(range(active_env.action_space.n))
            curr_valid = active_env._get_valid_actions()
            valid_actions = valid_actions.intersection(curr_valid)
            
            # Remove already shot positions
            for i in range(active_env.max_board_size):
                for j in range(active_env.max_board_size):
                    action_idx = i * active_env.max_board_size + j
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
            moves_made += 1
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
            
            # Print progress
            hit_rate = hits / total_shots if total_shots > 0 else 0
            if moves_made % 5 == 0:  # Print every 5 moves
                print(f"Move {moves_made}: Reward={reward:.2f}, Hit Rate={hit_rate:.2f}, Epsilon={inner_agent.epsilon:.2f}")
            
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
        print(f"Moves Made: {moves_made}")
        print(f"Epsilon: {inner_agent.epsilon:.2f}")
        if 'curriculum_phase' in info:
            print(f"Curriculum Phase: {info['curriculum_phase']}")
    
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
