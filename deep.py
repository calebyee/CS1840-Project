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
import random

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

# StandardBattleshipStrategy
class StandardBattleshipStrategy:
    def __init__(self, board_size=10):
        self.board_size = board_size
        self.hits_to_investigate = []  # Stack of hits we need to check around
        self.direction = None  # Once we find two hits, we know ship direction
        self.tried_positions = set()

    def get_next_position_in_direction(self, hit_pos, direction, board_state):
        """Get next position to try in current direction"""
        x, y = hit_pos
        if direction == 'horizontal':
            # Only try left/right
            next_pos = [(x, y + 1), (x, y - 1)]
        else:  # vertical
            # Only try up/down
            next_pos = [(x + 1, y), (x - 1, y)]
            
        # Filter valid positions not yet tried and not containing our own ships
        valid_next = [(x, y) for x, y in next_pos 
                     if self.is_valid_position(x, y, board_state) and board_state[x, y] != 1]
        
        return valid_next[0] if valid_next else None

    def choose_action(self, board_state):
        if board_state is None:
            return random.randint(0, self.board_size * self.board_size - 1)
        
        # If we have hits to investigate, use hunt mode
        if self.hits_to_investigate:
            hit_pos = self.hits_to_investigate[-1]
            
            # If we know direction, keep going that direction
            if self.direction:
                next_pos = self.get_next_position_in_direction(hit_pos, self.direction, board_state)
                if next_pos:
                    x, y = next_pos
                    return x * self.board_size + y
                else:
                    # Hit end of ship, try other direction from first hit
                    self.direction = 'vertical' if self.direction == 'horizontal' else 'horizontal'
                    first_hit = self.hits_to_investigate[0]
                    next_pos = self.get_next_position_in_direction(first_hit, self.direction, board_state)
                    if next_pos:
                        x, y = next_pos
                        return x * self.board_size + y
                    else:
                        # No more positions to try, reset hunt
                        self.hits_to_investigate = []
                        self.direction = None
                        return self.choose_action(board_state)
            
            # No direction yet, try cardinal directions only
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:  # Try cardinal directions
                new_x = hit_pos[0] + dx
                new_y = hit_pos[1] + dy
                
                if self.is_valid_position(new_x, new_y, board_state) and board_state[new_x, new_y] != 1:
                    return new_x * self.board_size + new_y
            
            # No valid adjacent positions, remove this hit
            self.hits_to_investigate.pop()
            return self.choose_action(board_state)

        # No hits to investigate, choose random untried position
        valid_positions = [
            (x, y) for x in range(self.board_size) 
            for y in range(self.board_size) 
            if self.is_valid_position(x, y, board_state) and board_state[x, y] != 1
        ]
        
        if valid_positions:
            x, y = random.choice(valid_positions)
            return x * self.board_size + y
        return None

    def update(self, action, hit, board_state):
        """
        Update strategy based on the result of the last shot
        Args:
            action: The position that was fired at
            hit: Whether the shot was a hit
            board_state: Current state of the board
        """
        x = action // self.board_size
        y = action % self.board_size
        self.tried_positions.add((x, y))

        if hit:
            self.hits_to_investigate.append((x, y))
            
            # If we have multiple hits, try to determine direction
            if len(self.hits_to_investigate) >= 2:
                hit1 = self.hits_to_investigate[-2]
                hit2 = self.hits_to_investigate[-1]
                if hit1[0] == hit2[0]:  # Same row
                    self.direction = 'horizontal'
                else:  # Same column
                    self.direction = 'vertical'

    def is_valid_position(self, x, y, board_state):
        return (0 <= x < self.board_size and 
                0 <= y < self.board_size and 
                (x, y) not in self.tried_positions and
                board_state[x, y] == 0)  # Check if position hasn't been shot at

# ImprovedBattleshipStrategy
class ImprovedBattleshipStrategy:
    def __init__(self, board_size=10):
        self.board_size = board_size
        self.reset()
        
    def reset(self):
        """Reset all strategy state"""
        self.probability_map = np.ones((self.board_size, self.board_size))
        self.hits = set()
        self.misses = set()
        self.current_targets = []
        self.ship_sizes = [5, 4, 3, 3, 2]
        self.remaining_ships = self.ship_sizes.copy()
        self.mode = 'hunt'  # 'hunt' or 'target'
        
    def choose_action(self, board_state):
        """Choose next shot using probability density"""
        # Update probability map based on board state
        self.update_probability_map(board_state)
        
        if self.mode == 'target' and self.current_targets:
            # Target mode: Check around hits
            return self.get_target_shot()
        else:
            # Hunt mode: Use probability density
            return self.get_hunt_shot()
            
    def update_probability_map(self, board_state):
        """Update probability density based on game state"""
        # Reset probability map
        self.probability_map.fill(0)
        
        # Calculate probabilities for each remaining ship
        for ship_size in self.remaining_ships:
            for row in range(self.board_size):
                for col in range(self.board_size):
                    # Check horizontal placement
                    if self.can_place_ship(row, col, ship_size, 'horizontal', board_state):
                        self.probability_map[row, col:col+ship_size] += 1
                    # Check vertical placement
                    if self.can_place_ship(row, col, ship_size, 'vertical', board_state):
                        self.probability_map[row:row+ship_size, col] += 1
                        
        # Zero out known positions
        for x, y in self.hits | self.misses:
            self.probability_map[x, y] = 0
            
    def can_place_ship(self, row, col, size, orientation, board_state):
        """Check if ship can be placed at position"""
        if orientation == 'horizontal':
            if col + size > self.board_size:
                return False
            for c in range(col, col + size):
                if (row, c) in self.misses or board_state[row, c] == -1:
                    return False
        else:
            if row + size > self.board_size:
                return False
            for r in range(row, row + size):
                if (r, col) in self.misses or board_state[r, col] == -1:
                    return False
        return True
        
    def get_target_shot(self):
        """Choose shot in target mode"""
        for x, y in self.current_targets:
            # Check cardinal directions
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                new_x, new_y = x + dx, y + dy
                if (self.is_valid_position(new_x, new_y) and 
                    (new_x, new_y) not in self.hits | self.misses):
                    return new_x * self.board_size + new_y
        # No valid targets, switch to hunt mode
        self.mode = 'hunt'
        return self.get_hunt_shot()
        
    def get_hunt_shot(self):
        """Choose shot in hunt mode using probability density"""
        # Get position with highest probability
        valid_positions = np.where(self.probability_map > 0)
        if len(valid_positions[0]) == 0:
            return None
            
        probabilities = self.probability_map[valid_positions]
        max_prob_idx = np.argmax(probabilities)
        x, y = valid_positions[0][max_prob_idx], valid_positions[1][max_prob_idx]
        return x * self.board_size + y
        
    def update(self, action, hit, board_state):
        """Update strategy based on shot result"""
        x = action // self.board_size
        y = action % self.board_size
        
        if hit:
            self.hits.add((x, y))
            self.current_targets.append((x, y))
            self.mode = 'target'
        else:
            self.misses.add((x, y))
            
    def is_valid_position(self, x, y):
        """Check if position is within board"""
        return 0 <= x < self.board_size and 0 <= y < self.board_size

# BattleshipAttackEnv
class BattleshipAttackEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board_size=10):
        super(BattleshipAttackEnv, self).__init__()
        self.board_size = board_size
        self.ship_sizes = [5, 4, 3, 3, 2]  # Add ship sizes
        self.action_space = spaces.Discrete(board_size * board_size)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(board_size, board_size), dtype=np.int8)
        
        # Initialize boards
        self.agent_board = np.zeros((board_size, board_size), dtype=np.int8)
        self.opponent_board = np.zeros((board_size, board_size), dtype=np.int8)
        self.agent_shots = np.zeros((board_size, board_size), dtype=np.int8)
        self.opponent_shots = np.zeros((board_size, board_size), dtype=np.int8)
        
        self.agent_ships_remaining = 17  # 5 + 4 + 3 + 3 + 2
        self.opponent_ships_remaining = 17
        self.sunk_ships = []  # Track which ships have been sunk
        
        self._place_ships()  # Place opponent's ships

    def reset(self):
        # Reset boards
        self.agent_board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.opponent_board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.agent_shots = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.opponent_shots = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        
        self.agent_ships_remaining = 17
        self.opponent_ships_remaining = 17
        
        self._place_ships()
        return self.agent_shots

    def _place_ships(self):
        """Place ships on the board"""
        self.opponent_board.fill(0)
        self.ship_positions = []  # Reset ship positions
        
        for ship_size in self.ship_sizes:
            placed = False
            while not placed:
                row = np.random.randint(0, self.board_size)
                col = np.random.randint(0, self.board_size)
                orientation = np.random.randint(0, 2)
                
                if self._is_valid_ship_placement(row, col, ship_size, orientation):
                    self._place_ship(row, col, ship_size, orientation)
                    placed = True

    def _is_valid_ship_placement(self, row, col, ship_size, orientation):
        """Check if ship placement is valid"""
        # Add board parameter to check against
        def check_placement(board):
            if orientation == 0:  # horizontal
                if col + ship_size > self.board_size:
                    return False
                return not np.any(board[row, col:col + ship_size] == 1)
            else:  # vertical
                if row + ship_size > self.board_size:
                    return False
                return not np.any(board[row:row + ship_size, col] == 1)
        
        # Use opponent_board by default if no board specified
        return check_placement(self.opponent_board)

    def _place_ship(self, row, col, ship_size, orientation):
        """Place a ship and store its position"""
        if orientation == 0:  # horizontal
            self.opponent_board[row, col:col + ship_size] = 1
            self.ship_positions.append(((row, col), orientation, ship_size))
        else:  # vertical
            self.opponent_board[row:row + ship_size, col] = 1
            self.ship_positions.append(((row, col), orientation, ship_size))

    def _check_ship_sunk(self, row, col):
        """Check if hitting this position sunk a ship"""
        # Check horizontal ship
        left = col
        while left > 0 and self.opponent_board[row, left-1] == 1:
            left -= 1
        right = col
        while right < self.board_size-1 and self.opponent_board[row, right+1] == 1:
            right += 1
            
        # Check if all positions in horizontal ship are hit
        ship_coords = [(row, c) for c in range(left, right+1)]
        if all(self.agent_shots[r, c] == 1 for r, c in ship_coords):
            if ship_coords not in self.sunk_ships:
                self.sunk_ships.append(ship_coords)
                return True
            
        # Check vertical ship
        top = row
        while top > 0 and self.opponent_board[top-1, col] == 1:
            top -= 1
        bottom = row
        while bottom < self.board_size-1 and self.opponent_board[bottom+1, col] == 1:
            bottom += 1
            
        # Check if all positions in vertical ship are hit
        ship_coords = [(r, col) for r in range(top, bottom+1)]
        if all(self.agent_shots[r, c] == 1 for r, c in ship_coords):
            if ship_coords not in self.sunk_ships:
                self.sunk_ships.append(ship_coords)
                return True
               
        return False

    def _place_agent_ships(self):
        """Place agent's ships randomly"""
        self.agent_board.fill(0)
        for ship_size in self.ship_sizes:
            placed = False
            while not placed:
                row = np.random.randint(0, self.board_size)
                col = np.random.randint(0, self.board_size)
                orientation = np.random.randint(0, 2)
                # Create temporary board for checking placement
                temp_board = self.agent_board.copy()
                if self._is_valid_ship_placement(row, col, ship_size, orientation):
                    if orientation == 0:  # horizontal
                        self.agent_board[row, col:col + ship_size] = 1
                    else:  # vertical
                        self.agent_board[row:row + ship_size, col] = 1
                    placed = True

    def random_shot(self, board_state):
        """Generate a random valid shot"""
        valid_positions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.opponent_shots[i, j] == 0:  # Position hasn't been shot at
                    valid_positions.append(i * self.board_size + j)
        
        if valid_positions:
            return random.choice(valid_positions)
        return None

    def step(self, action):
        if action is None:
            return self.agent_shots, 0, True, {}
            
        # Agent's turn
        row = action // self.board_size
        col = action % self.board_size
        
        # Track hits/misses
        agent_hit = False
        opponent_hit = False
        reward = 0
        
        # Process agent's action
        if self.opponent_board[row, col] == 1:
            self.agent_shots[row, col] = 1
            self.opponent_ships_remaining -= 1
            reward = 3.0
            agent_hit = True
            if self._check_ship_sunk(row, col):
                reward = 5.0
        else:
            self.agent_shots[row, col] = -1
            reward = -0.1
            
        # Opponent's turn (using test.py's random_shot)
        opponent_action = self._random_shot()
        if opponent_action is not None:
            opp_row, opp_col = opponent_action
            if self.agent_board[opp_row, opp_col] == 1:
                self.opponent_shots[opp_row, opp_col] = 1
                self.agent_ships_remaining -= 1
                opponent_hit = True
            else:
                self.opponent_shots[opp_row, opp_col] = -1
        
        # Check if game is over
        done = (self.opponent_ships_remaining <= 0) or (self.agent_ships_remaining <= 0)
        
        info = {
            "agent_ships_remaining": self.agent_ships_remaining,
            "opponent_ships_remaining": self.opponent_ships_remaining,
            "agent_hit": agent_hit,
            "opponent_hit": opponent_hit,
            "winner": "agent" if self.opponent_ships_remaining <= 0 else "opponent" if self.agent_ships_remaining <= 0 else None
        }
        
        return self.agent_shots, reward, done, info

    def _random_shot(self):
        """Generate a random valid shot (from test.py)"""
        valid_positions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.opponent_shots[i, j] == 0:  # Position hasn't been shot at
                    valid_positions.append((i, j))
        
        if valid_positions:
            return random.choice(valid_positions)
        return None

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
            if shots[i, j] == 0 and env.agent_board[i, j] == 0:
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

def prepare_state_tensor(state):
    """
    Prepare the state for input to the neural network.
    Creates a 2-channel input: one for shots and one for hits
    """
    # First channel: shots (-1 for misses, 1 for hits, 0 for no shot)
    shots = state.copy()
    
    # Second channel: binary mask of hits (1 where hits occurred, 0 elsewhere)
    hits = (state == 1).astype(np.float32)
    
    # Stack channels and convert to tensor
    state_tensor = torch.FloatTensor(np.stack([shots, hits])).unsqueeze(0)
    
    return state_tensor

def train_nested_mdp(init_env, active_env, num_episodes=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agents
    inner_agent = DQNAgent(
        state_size=active_env.board_size * 2,
        action_size=active_env.action_space.n,
        device=device,
        is_placement_agent=False
    )
    
    # Track statistics
    wins = 0
    total_games = 0
    episode_infos = []  # Store episode results
    
    for episode in range(num_episodes):
        print(f"\n{'='*25} Episode {episode+1}/{num_episodes} {'='*25}")
        
        active_state = active_env.reset()
        episode_reward = 0
        done = False
        total_moves = 0
        hits = 0
        shots = 0
        
        while not done:
            # Agent's turn
            valid_actions = get_valid_actions(active_env, active_state)
            state_tensor = prepare_state_tensor(active_state)
            action, was_explored = inner_agent.choose_action(state_tensor, valid_actions)
            next_state, reward, done, info = active_env.step(action)
            
            # Update statistics
            total_moves += 1
            shots += 1
            if info["agent_hit"]:
                hits += 1
            
            # Update agent
            inner_agent.update(state_tensor, action, reward, prepare_state_tensor(next_state), 
                             done, was_explored, info["agent_hit"])
            
            active_state = next_state
            episode_reward += reward
            
            # Print progress every 5 moves
            if total_moves % 5 == 0:
                print(f"Move {total_moves}: Reward={reward:.2f}, "
                      f"Hit Rate={hits/shots:.2f}, "
                      f"Agent Ships={info['agent_ships_remaining']}, "
                      f"Opponent Ships={info['opponent_ships_remaining']}, "
                      f"Epsilon={inner_agent.epsilon:.2f}")
        
        # Update statistics
        total_games += 1
        if info['winner'] == 'agent':
            wins += 1
        episode_infos.append(info)  # Store episode info for plotting
        
        # Print episode summary
        winner = info['winner']
        print(f"\nGame Over!")
        print(f"Winner: {winner}")
        print(f"Total Moves: {total_moves}")
        print(f"Final Hit Rate: {hits/shots:.2f}")
        print(f"Agent Ships Remaining: {info['agent_ships_remaining']}/17")
        print(f"Opponent Ships Remaining: {info['opponent_ships_remaining']}/17")
        print(f"Win Rate: {wins/total_games:.2f}")
    
    # Plot training metrics
    metrics = inner_agent.get_metrics()
    
    plt.figure(figsize=(15, 10))
    
    # Plot rewards (now in 2x4 grid with win rate)
    plt.subplot(2, 4, 1)
    num_completed_episodes = len(metrics['episode_rewards'])
    episodes = list(range(1, num_completed_episodes + 1))
    episode_rewards = metrics['episode_rewards']
    plt.plot(episodes, episode_rewards, '-o', markersize=2)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot win rate
    plt.subplot(2, 4, 7)
    # Track actual wins from info['winner']
    wins_cumulative = np.cumsum([1 if info.get('winner') == 'agent' else 0 for info in episode_infos])
    episodes_completed = range(1, len(wins_cumulative) + 1)
    plt.plot(episodes_completed, [wins_cumulative[i]/(i+1) for i in range(len(wins_cumulative))], '-o', markersize=2)
    plt.title('Win Rate Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    
    # Plot hit rates
    plt.subplot(2, 4, 2)
    hit_rates = metrics['hit_rates']
    if len(hit_rates) > num_completed_episodes:
        hit_rates = hit_rates[:num_completed_episodes]
    elif len(hit_rates) < num_completed_episodes:
        hit_rates.extend([0] * (num_completed_episodes - len(hit_rates)))
    plt.plot(episodes, hit_rates, '-o', markersize=2)
    plt.title('Hit Rates')
    plt.xlabel('Episode')
    plt.ylabel('Hit Rate')
    
    # Plot losses
    plt.subplot(2, 4, 3)
    plt.plot(metrics['losses'])
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    
    # Plot Q-values
    plt.subplot(2, 4, 4)
    plt.plot(metrics['q_values'])
    plt.title('Average Q-Values')
    plt.xlabel('Step')
    plt.ylabel('Q-Value')
    
    # Plot epsilon decay
    plt.subplot(2, 4, 5)
    plt.plot(metrics['epsilons'])
    plt.title('Epsilon Decay')
    plt.xlabel('Step')
    plt.ylabel('Epsilon')
    
    # Plot exploration ratio
    plt.subplot(2, 4, 6)
    plt.axhline(y=metrics['exploration_ratio'], color='r', linestyle='-')
    plt.title('Exploration Ratio')
    plt.xlabel('Step')
    plt.ylabel('Ratio')
    
    # Add game length plot
    plt.subplot(2, 4, 8)
    game_lengths = metrics['game_lengths']
    plt.plot(episodes, game_lengths, '-o', markersize=2)
    plt.title('Game Length')
    plt.xlabel('Episode')
    plt.ylabel('Number of Moves')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    
    return inner_agent

# Create and train the environments
if __name__ == "__main__":
    init_env = BattleshipPlacementEnv()
    active_env = BattleshipAttackEnv()
    train_nested_mdp(init_env, active_env)
