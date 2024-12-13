import numpy as np
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import os
import pickle

# QLearningAgent
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration
        
        # Initialize Q-table with small random values instead of zeros
        self.q_table = np.random.uniform(low=0, high=0.1, size=(state_size, action_size))

    def choose_action(self, state, valid_actions=None):
        state_idx = self._get_state_index(state)
        
        if valid_actions is None:
            valid_actions = range(self.action_size)
        
        # Epsilon-greedy with valid actions only
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            # Get Q-values for valid actions only
            valid_q_values = self.q_table[state_idx, valid_actions]
            return valid_actions[np.argmax(valid_q_values)]

    def update(self, state, action, reward, next_state, valid_next_actions=None):
        state_idx = self._get_state_index(state)
        next_state_idx = self._get_state_index(next_state)
        
        if valid_next_actions is None:
            valid_next_actions = range(self.action_size)
        
        # Get maximum Q-value for valid next actions only
        next_q_values = self.q_table[next_state_idx, valid_next_actions]
        best_next_value = np.max(next_q_values) if len(next_q_values) > 0 else 0
        
        # Q-learning update
        current_q = self.q_table[state_idx, action]
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - current_q
        self.q_table[state_idx, action] += self.lr * td_error

    def _get_state_index(self, state):
        # Convert state array to index, ensuring it's within bounds
        if isinstance(state, np.ndarray):
            state = state.flatten()
        return hash(state.tobytes()) % self.state_size

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
        self.action_space = spaces.MultiDiscrete([board_size, board_size])
        self.observation_space = spaces.Box(low=-1, high=1, shape=(board_size, board_size), dtype=np.int8)
        self.reset()

    def reset(self):
        self.agent_board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.opponent_board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.agent_shots = np.zeros((self.board_size, self.board_size), dtype=np.int8)  # Shots by agent
        self.opponent_shots = np.zeros((self.board_size, self.board_size), dtype=np.int8)  # Shots by opponent
        self.done = False
        return self.agent_shots

    def step(self, action, is_opponent=False):
        row, col = action
        shots = self.opponent_shots if is_opponent else self.agent_shots
        target_board = self.agent_board if is_opponent else self.opponent_board

        # Check if shot is valid
        if shots[row, col] != 0:
            return shots, -1, False, {"error": "Invalid shot"}

        # Record the shot and check if it's a hit
        if target_board[row, col] == 1:
            shots[row, col] = 1
            reward = 1
        else:
            shots[row, col] = -1
            reward = 0

        # Update the appropriate shots board
        if is_opponent:
            self.opponent_shots = shots
        else:
            self.agent_shots = shots

        # Check if all ships are hit (game ending condition)
        if is_opponent:
            total_ship_cells = np.sum(self.agent_board == 1)
            hit_ship_cells = np.sum((self.agent_board == 1) & (self.opponent_shots == 1))
            self.done = hit_ship_cells == total_ship_cells
        else:
            total_ship_cells = np.sum(self.opponent_board == 1)
            hit_ship_cells = np.sum((self.opponent_board == 1) & (self.agent_shots == 1))
            self.done = hit_ship_cells == total_ship_cells

        return shots, reward, self.done, {
            "total_ships": total_ship_cells,
            "hits": hit_ship_cells,
            "game_over": self.done
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

# Training Loop
def train_nested_mdp(init_env, active_env, outer_episodes=20, inner_episodes=100):
    # Initialize agents with correct state/action spaces
    outer_agent = QLearningAgent(
        state_size=init_env.board_size * init_env.board_size,
        action_size=np.prod(init_env.action_space.nvec)
    )
    
    inner_agent = QLearningAgent(
        state_size=active_env.board_size * active_env.board_size,
        action_size=active_env.board_size * active_env.board_size  # For attack phase, action is just row*col
    )

    previous_outer_agent = deepcopy(outer_agent)  # Initialize the previous outer agent
    previous_inner_agent = deepcopy(inner_agent)  # Initialize the previous inner agent

    turns_per_game = []

    for episode in range(outer_episodes):
        print(f"Starting episode {episode + 1}/{outer_episodes}")

        # Reset lists and environment
        outer_rewards = []
        outer_state = init_env.reset()
        outer_done = False

        # Place ships
        while not outer_done:
            action = outer_agent.choose_action(outer_state.flatten())
            row, col, orientation = np.unravel_index(action, init_env.action_space.nvec)
            next_state, reward, outer_done, info = init_env.step((row, col, orientation))
            
            if 'error' not in info:
                outer_rewards.append((outer_state, action, next_state))
                print(f"Valid placement: Ship {init_env.current_ship_index}, Position: ({row}, {col}), Orientation: {orientation}")
            
            outer_state = next_state

        # Set up battle phase
        active_env.reset()
        active_env.agent_board = init_env.board.copy()
        
        # Create opponent's board
        previous_init_env = BattleshipPlacementEnv()
        prev_state = previous_init_env.reset()
        prev_done = False
        
        while not prev_done:
            prev_action = previous_outer_agent.choose_action(prev_state.flatten())
            row, col, orientation = np.unravel_index(prev_action, previous_init_env.action_space.nvec)
            prev_state, _, prev_done, info = previous_init_env.step((row, col, orientation))
        
        active_env.opponent_board = previous_init_env.board.copy()

        print(f"\nBattle Phase Setup - Episode {episode + 1}")
        print(f"Agent's board (our ships):\n{active_env.agent_board}")
        print(f"Opponent's board (their ships):\n{active_env.opponent_board}")

        # Battle phase
        inner_state = active_env.agent_shots
        inner_done = False
        agent_turns = 0
        opponent_turns = 0
        max_shots = init_env.board_size * init_env.board_size

        while not inner_done and (agent_turns < max_shots and opponent_turns < max_shots):
            # Get valid actions for agent
            valid_actions = get_valid_actions(active_env, active_env.agent_shots)
            
            # Current agent's turn
            inner_action = inner_agent.choose_action(inner_state.flatten(), valid_actions)
            row, col = np.unravel_index(inner_action, (active_env.board_size, active_env.board_size))
            next_state, reward, inner_done, info = active_env.step((row, col), is_opponent=False)

            if 'error' not in info:
                hit_status = "Hit!" if reward > 0 else "Miss"
                agent_turns += 1
                print(f"Turn {agent_turns + opponent_turns}: Agent shot at ({row}, {col}) - {hit_status}")
                
                # Modify reward to encourage hits and discourage misses
                if reward > 0:  # Hit
                    reward = 10
                else:  # Miss
                    reward = -1
                
                # Get valid actions for next state
                valid_next_actions = get_valid_actions(active_env, next_state)
                inner_agent.update(inner_state.flatten(), inner_action, reward, next_state.flatten(), valid_next_actions)
                
                if 'game_over' in info and info['game_over']:
                    print(f"Agent wins! All opponent ships sunk in {agent_turns} shots!")
                    reward += 50  # Bonus reward for winning
                    inner_agent.update(inner_state.flatten(), inner_action, reward, next_state.flatten(), valid_next_actions)
                    inner_done = True
                    break

                inner_state = next_state

                if not inner_done:
                    # Opponent's turn
                    valid_opponent_move = False
                    while not valid_opponent_move and not inner_done:
                        opponent_action = previous_inner_agent.choose_action(inner_state.flatten())
                        opp_row, opp_col = np.unravel_index(opponent_action, active_env.action_space.nvec)
                        next_state, opp_reward, opponent_done, opp_info = active_env.step((opp_row, opp_col), is_opponent=True)
                        
                        if 'error' not in opp_info:
                            hit_status = "Hit!" if opp_reward > 0 else "Miss"
                            opponent_turns += 1
                            print(f"Turn {agent_turns + opponent_turns}: Opponent shot at ({opp_row}, {opp_col}) - {hit_status}")
                            if 'game_over' in opp_info and opp_info['game_over']:
                                print(f"Opponent wins! All agent ships sunk in {opponent_turns} shots!")
                                inner_done = True
                                break
                            valid_opponent_move = True
                            inner_state = next_state

        total_turns = agent_turns + opponent_turns
        turns_per_game.append(total_turns)

        # Calculate ship rewards based on opponent's shots
        ship_rewards = []
        for row, col, orientation, size in init_env.ship_positions:
            if orientation == 0:  # horizontal
                ship_cells = active_env.agent_board[row, col:col + size]
                shot_cells = active_env.opponent_shots[row, col:col + size]
            else:  # vertical
                ship_cells = active_env.agent_board[row:row + size, col]
                shot_cells = active_env.opponent_shots[row:row + size, col]
            
            unhit_cells = np.sum((ship_cells == 1) & (shot_cells != 1))
            ship_rewards.append(unhit_cells)

        # Verify we have the correct number of rewards
        assert len(outer_rewards) == len(ship_rewards), f"Mismatch between placements ({len(outer_rewards)}) and rewards ({len(ship_rewards)})"

        # Update outer agent with rewards
        for i, (state, action, next_state) in enumerate(outer_rewards):
            outer_agent.update(state.flatten(), action, ship_rewards[i], next_state.flatten())

        # Print final game state
        print(f"\nEpisode {episode + 1} completed in {total_turns} total shots")
        print(f"Agent shots: {agent_turns}, Opponent shots: {opponent_turns}")
        print("\nFinal Game State:")
        print("\nAgent's Board (our ships):")
        print(active_env.agent_board)
        print("\nOpponent's Board (their ships):")
        print(active_env.opponent_board)
        print("\nAgent's Shots (hits=1, misses=-1):")
        print(active_env.agent_shots)
        print("\nOpponent's Shots (hits=1, misses=-1):")
        print(active_env.opponent_shots)
        print("\nFinal Score:")
        print(f"Agent hits: {np.sum(active_env.agent_shots == 1)}/{np.sum(active_env.opponent_board == 1)}")
        print(f"Opponent hits: {np.sum(active_env.opponent_shots == 1)}/{np.sum(active_env.agent_board == 1)}")
        print("\nShip rewards (unhit cells per ship):", ship_rewards)
        print("-" * 50)

        outer_agent.decay_exploration()
        inner_agent.decay_exploration()

        previous_outer_agent = deepcopy(outer_agent)
        previous_inner_agent = deepcopy(inner_agent)

        if episode % 100 == 0:
            # Create checkpoint directory if it doesn't exist
            os.makedirs("checkpoints", exist_ok=True)
            # Save agents
            with open(f"checkpoints/placement_agent_{episode}.pkl", 'wb') as f:
                pickle.dump(outer_agent, f)
            with open(f"checkpoints/attack_agent_{episode}.pkl", 'wb') as f:
                pickle.dump(inner_agent, f)

    # Plot the number of turns per game
    plt.plot(range(1, len(turns_per_game) + 1), turns_per_game)
    plt.xlabel('Episode')
    plt.ylabel('Turns per Game')
    plt.title('Turns Taken per Game During Training')
    plt.show()

    return outer_agent, inner_agent

if __name__ == "__main__":
    # Create instances of the environments
    init_env = BattleshipPlacementEnv()
    active_env = BattleshipAttackEnv()

    # Pass the instances to the training function
    train_nested_mdp(init_env, active_env, outer_episodes=1000, inner_episodes=100)
