import numpy as np
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import os
import pickle
import random

# QLearningAgent
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration
        
        # Initialize Q-table with small random values
        self.q_table = {}

    def _get_state_key(self, state):
        """Convert state dictionary to a hashable key"""
        if isinstance(state, dict):
            board = state['board']
            last_hit = state['last_hit']
            shot_history = state['shot_history']
            
            # Convert board to tuple for hashing
            board_tuple = tuple(map(tuple, board))
            
            # Convert shot history to tuple of tuples
            history_tuple = tuple((pos, hit) for pos, hit in shot_history)
            
            # Create hashable state representation
            return (board_tuple, last_hit, history_tuple)
        else:
            # Handle flat state arrays (for placement phase)
            return tuple(state.flatten())

    def choose_action(self, state, valid_actions):
        state_key = self._get_state_key(state)
        
        # Initialize state in Q-table if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.random.uniform(low=0, high=0.1, size=self.action_size)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            # Choose best valid action
            valid_q_values = [(action, self.q_table[state_key][action]) for action in valid_actions]
            return max(valid_q_values, key=lambda x: x[1])[0]

    def update(self, state, action, reward, next_state, valid_next_actions, hit=None):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Initialize states in Q-table if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.random.uniform(low=0, high=0.1, size=self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.random.uniform(low=0, high=0.1, size=self.action_size)
        
        # Get maximum Q-value for valid next actions only
        next_q_values = [self.q_table[next_state_key][a] for a in valid_next_actions]
        best_next_value = max(next_q_values) if next_q_values else 0
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - current_q
        self.q_table[state_key][action] += self.lr * td_error

    def decay_exploration(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# BattleshipPlacementEnv
class BattleshipPlacementEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board_size=10, ship_sizes=[5, 4, 3, 3, 2]):
        super(BattleshipPlacementEnv, self).__init__()
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.current_ship_index = 0

        self.action_space = spaces.MultiDiscrete([board_size, board_size, 2])
        self.observation_space = spaces.Box(low=0, high=1, shape=(board_size, board_size), dtype=np.int8)

        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_ship_index = 0
        self.done = False
        self.ship_positions = []  # Store ship positions and sizes for reward calculation
        return self.board

    def step(self, action):
        row, col, orientation = action
        ship_size = self.ship_sizes[self.current_ship_index]

        if not self._is_valid_placement(row, col, ship_size, orientation):
            return self.board, -1, False, {"error": "Invalid placement"}

        self._place_ship(row, col, ship_size, orientation)
        self.ship_positions.append((row, col, orientation, ship_size))
        self.current_ship_index += 1

        if self.current_ship_index >= len(self.ship_sizes):
            self.done = True
            reward = 0
        else:
            reward = 0

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
        
        # Define relative actions as a grid around the last hit
        # Format: (dx, dy) for each possible relative position
        self.relative_actions = []
        for dx in range(-board_size+1, board_size):
            for dy in range(-board_size+1, board_size):
                if dx == 0 and dy == 0:  # Skip (0,0) as it's not a valid move
                    continue
                self.relative_actions.append((dx, dy))
        
        # Add random action as the last option
        self.relative_actions.append(None)  # Random action
        
        self.action_space = spaces.Discrete(len(self.relative_actions))
        self.observation_space = spaces.Box(low=0, high=1, shape=(board_size, board_size), dtype=np.int8)
        
        # Shot history for each player
        self.agent_shot_history = []  # List of (pos, hit) tuples
        self.opponent_shot_history = []
        
        # Last hit positions
        self.agent_last_hit = None
        self.opponent_last_hit = None
        
        self.reset()

    def reset(self):
        self.agent_board = np.zeros((self.board_size, self.board_size))
        self.opponent_board = np.zeros((self.board_size, self.board_size))
        self.agent_shots = np.zeros((self.board_size, self.board_size))
        self.opponent_shots = np.zeros((self.board_size, self.board_size))
        self.agent_shot_history = []
        self.opponent_shot_history = []
        self.agent_last_hit = None
        self.opponent_last_hit = None
        return self.agent_shots

    def get_valid_relative_actions(self, is_opponent=False):
        """Get valid relative actions based on last hit and board state"""
        valid_actions = []
        shots = self.opponent_shots if is_opponent else self.agent_shots
        last_hit = self.opponent_last_hit if is_opponent else self.agent_last_hit
        
        # If no hits yet, only random action is valid
        if last_hit is None:
            return [len(self.relative_actions) - 1]  # Random action index
        
        # Check each relative action
        for i, action in enumerate(self.relative_actions[:-1]):  # Exclude random action
            if action is not None:
                dx, dy = action
                new_x = last_hit[0] + dx
                new_y = last_hit[1] + dy
                
                # Check if position is valid and hasn't been shot
                if (0 <= new_x < self.board_size and 
                    0 <= new_y < self.board_size and 
                    shots[new_x, new_y] == 0):
                    valid_actions.append(i)
        
        # Always include random action as fallback
        valid_actions.append(len(self.relative_actions) - 1)
        return valid_actions

    def relative_to_absolute(self, action, last_hit):
        """Convert relative action to absolute board position"""
        if action == len(self.relative_actions) - 1:  # Random action
            valid_positions = []
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.agent_shots[i, j] == 0:
                        valid_positions.append((i, j))
            if not valid_positions:
                return None
            return random.choice(valid_positions)
        
        if last_hit is None:
            # First shot or no hits yet - shoot randomly
            valid_positions = []
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.agent_shots[i, j] == 0:
                        valid_positions.append((i, j))
            if not valid_positions:
                return None
            return random.choice(valid_positions)
        
        dx, dy = self.relative_actions[action]
        new_x = last_hit[0] + dx
        new_y = last_hit[1] + dy
        
        if (0 <= new_x < self.board_size and 
            0 <= new_y < self.board_size):
            return (new_x, new_y)
        return None

    def step(self, action, is_opponent=False):
        # Use appropriate variables based on whose turn it is
        shots = self.opponent_shots if is_opponent else self.agent_shots
        target_board = self.agent_board if is_opponent else self.opponent_board
        shot_history = self.opponent_shot_history if is_opponent else self.agent_shot_history
        last_hit = self.opponent_last_hit if is_opponent else self.agent_last_hit
        
        # Convert relative action to absolute position
        position = self.relative_to_absolute(action, last_hit)
        if position is None:
            return shots, -1, False, {'error': 'Invalid position'}
        
        row, col = position
        if shots[row, col] != 0:  # Position already shot
            return shots, -1, False, {'error': 'Position already shot'}

        # Process the shot
        hit = target_board[row, col] == 1
        shots[row, col] = 1 if hit else -1
        
        # Update shot history and last hit
        shot_history.append(((row, col), hit))
        if hit:
            if is_opponent:
                self.opponent_last_hit = (row, col)
            else:
                self.agent_last_hit = (row, col)
        
        # Check if all ships are sunk
        if np.sum(target_board == 1) == np.sum((shots == 1) & (target_board == 1)):
            return shots, 10, True, {'game_over': True}
        
        return shots, 1 if hit else -1, False, {}

    def get_state_representation(self, is_opponent=False):
        """Get a state representation that includes shot history"""
        shots = self.opponent_shots if is_opponent else self.agent_shots
        shot_history = self.opponent_shot_history if is_opponent else self.agent_shot_history
        last_hit = self.opponent_last_hit if is_opponent else self.agent_last_hit
        
        # Create a state that includes:
        # - Current shot board
        # - Last hit position (if any)
        # - Recent shot history
        return {
            'board': shots.copy(),
            'last_hit': last_hit,
            'shot_history': shot_history[-5:] if shot_history else []  # Last 5 shots
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


def train_against_random(init_env, active_env, outer_episodes=20, inner_episodes=100):
    # Initialize agent
    agent_outer = QLearningAgent(
        state_size=init_env.board_size * init_env.board_size,
        action_size=np.prod(init_env.action_space.nvec)
    )
    agent_inner = QLearningAgent(
        state_size=active_env.board_size * active_env.board_size,
        action_size=len(active_env.relative_actions)
    )

    turns_per_game = []
    agent_wins = []  # 1 for win, 0 for loss
    
    for episode in range(outer_episodes):
        print(f"\nStarting episode {episode + 1}/{outer_episodes}")

        # Agent placement phase
        agent_env = BattleshipPlacementEnv()
        agent_state = agent_env.reset()
        agent_done = False
        agent_rewards = []

        while not agent_done:
            valid_actions = []
            for action in range(np.prod(agent_env.action_space.nvec)):
                row, col, orientation = np.unravel_index(action, agent_env.action_space.nvec)
                if agent_env._is_valid_placement(row, col, agent_env.ship_sizes[agent_env.current_ship_index], orientation):
                    valid_actions.append(action)
            
            action = agent_outer.choose_action(agent_state.flatten(), valid_actions)
            row, col, orientation = np.unravel_index(action, agent_env.action_space.nvec)
            next_state, reward, agent_done, info = agent_env.step((row, col, orientation))
            
            if 'error' not in info:
                agent_rewards.append((agent_state, action, next_state))
                print(f"Agent placement: Ship {agent_env.current_ship_index}, Position: ({row}, {col}), Orientation: {orientation}")
            
            agent_state = next_state

        # Random opponent placement
        opponent_env = BattleshipPlacementEnv()
        opponent_board = np.zeros((init_env.board_size, init_env.board_size))
        for ship_size in init_env.ship_sizes:
            placed = False
            while not placed:
                row = random.randint(0, init_env.board_size-1)
                col = random.randint(0, init_env.board_size-1)
                orientation = random.randint(0, 1)
                if orientation == 0:  # horizontal
                    if col + ship_size <= init_env.board_size:
                        if not np.any(opponent_board[row, col:col+ship_size]):
                            opponent_board[row, col:col+ship_size] = 1
                            placed = True
                else:  # vertical
                    if row + ship_size <= init_env.board_size:
                        if not np.any(opponent_board[row:row+ship_size, col]):
                            opponent_board[row:row+ship_size, col] = 1
                            placed = True

        # Battle phase
        active_env.reset()
        active_env.agent_board = agent_env.board.copy()
        active_env.opponent_board = opponent_board
        
        inner_state = active_env.get_state_representation()
        inner_done = False
        agent_turns = 0
        opponent_turns = 0
        max_shots = init_env.board_size * init_env.board_size
        current_player = 1  # Track whose turn it is

        while not inner_done and (agent_turns + opponent_turns < max_shots * 2):
            if current_player == 1:
                # Agent's turn
                valid_actions = active_env.get_valid_relative_actions()
                inner_action = agent_inner.choose_action(inner_state, valid_actions)
                next_state, reward, inner_done, info = active_env.step(inner_action)

                if 'error' not in info:
                    hit = reward > 0
                    agent_turns += 1
                    print(f"Turn {agent_turns + opponent_turns}: Agent shot - {'Hit!' if hit else 'Miss'}")
                    
                    valid_next_actions = active_env.get_valid_relative_actions()
                    next_state_rep = active_env.get_state_representation()
                    agent_inner.update(inner_state, inner_action, reward, next_state_rep, valid_next_actions, hit)
                    
                    if inner_done:
                        print(f"Agent wins! All opponent ships sunk in {agent_turns} shots!")
                        agent_wins.append(1)
                        break

                    inner_state = next_state_rep
                current_player = 2  # Switch to opponent's turn
            
            else:
                # Opponent's turn (hunt-and-target)
                opponent_action = hunt_and_target_shot(active_env, active_env.opponent_last_hit)
                next_state, reward, opponent_done, opp_info = active_env.step(opponent_action, is_opponent=True)
                
                if 'error' not in opp_info:
                    hit = reward > 0
                    opponent_turns += 1
                    print(f"Turn {agent_turns + opponent_turns}: Opponent shot - {'Hit!' if hit else 'Miss'}")
                    
                    if opponent_done:
                        print(f"Opponent wins! All agent ships sunk in {opponent_turns} shots!")
                        agent_wins.append(0)
                        inner_done = True
                        break

                    inner_state = active_env.get_state_representation()
                current_player = 1  # Switch to agent's turn

        total_turns = agent_turns + opponent_turns
        turns_per_game.append(total_turns)
        
        # Update placement agent
        for i, (state, action, next_state) in enumerate(agent_rewards):
            valid_next_actions = []
            for next_action in range(np.prod(init_env.action_space.nvec)):
                row, col, orientation = np.unravel_index(next_action, init_env.action_space.nvec)
                if i + 1 < len(init_env.ship_sizes) and init_env._is_valid_placement(row, col, init_env.ship_sizes[i + 1], orientation):
                    valid_next_actions.append(next_action)
            agent_outer.update(state.flatten(), action, 1.0, next_state.flatten(), valid_next_actions)

        # Decay exploration rates
        agent_outer.decay_exploration()
        agent_inner.decay_exploration()

        print(f"\nEpisode {episode + 1} completed in {total_turns} total shots")
        print(f"Agent shots: {agent_turns}, Opponent shots: {opponent_turns}")
        print("-" * 50)

    return agent_outer, agent_inner, turns_per_game, agent_wins

def hunt_and_target_shot(env, last_hit=None):
    """
    Implements a hunt-and-target strategy:
    - If there's a hit, try adjacent squares
    - Otherwise, shoot randomly at unshot squares
    """
    # If we have a last hit, try adjacent squares first
    if last_hit is not None:
        x, y = last_hit
        # Try adjacent squares (North, South, East, West)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < env.board_size and 
                0 <= new_y < env.board_size and 
                env.opponent_shots[new_x, new_y] == 0):
                # Convert to relative action index
                for i, action in enumerate(env.relative_actions[:-1]):  # Exclude random action
                    if action == (dx, dy):
                        return i
                
    # If no last hit or no valid adjacent squares, use random action
    return len(env.relative_actions) - 1  # Random action index

if __name__ == "__main__":
    # Create instances of the environments
    init_env = BattleshipPlacementEnv()
    active_env = BattleshipAttackEnv()

    # Train against random opponent
    agent_outer, agent_inner, turns_per_game, agent_wins = train_against_random(init_env, active_env, outer_episodes=501, inner_episodes=100)
    
    # Plot training results
    plt.figure(figsize=(10, 6))
    
    # Plot turns per game with color-coded points
    episodes = range(1, len(turns_per_game) + 1)
    
    # Create scatter plot with different colors for wins/losses
    for i, (turns, winner) in enumerate(zip(turns_per_game, agent_wins), 1):
        if winner == 1:
            plt.scatter(i, turns, c='blue', alpha=0.6)  # Agent wins
        else:
            plt.scatter(i, turns, c='red', alpha=0.6)   # Opponent wins
    
    # Add connecting line
    plt.plot(episodes, turns_per_game, color='gray', alpha=0.3)
    
    plt.xlabel('Episode')
    plt.ylabel('Turns per Game')
    plt.title('Turns per Game During Training\nBlue: Agent wins, Red: Opponent wins')
    plt.grid(True)
    
    # Print final statistics
    agent_win_rate = sum(win == 1 for win in agent_wins) / len(agent_wins)
    print(f"\nFinal Statistics:")
    print(f"Agent Win Rate: {agent_win_rate:.2%}")
    print(f"Average turns per game: {np.mean(turns_per_game):.1f}")
    
    plt.tight_layout()
    plt.show()