import numpy as np
from copy import deepcopy
import gym
from gym import spaces
import matplotlib.pyplot as plt

# QLearningAgent
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

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
        self.agent_board = None
        self.opponent_board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.shots = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.done = False
        return self.shots

    def step(self, action):
        row, col = action

        if self.shots[row, col] != 0:
            return self.shots, -1, False, {"error": "Invalid shot"}

        if self.opponent_board[row, col] == 1:
            self.shots[row, col] = 1
            reward = 1
        else:
            self.shots[row, col] = -1
            reward = 0

        if np.sum((self.opponent_board == 1) & (self.shots == 1)) == np.sum(self.opponent_board == 1):
            self.done = True

        return self.shots, reward, self.done, {}

    def render(self, mode='human'):
        print("Shots:")
        print(self.shots)

# Training Loop
def train_nested_mdp(init_env, active_env, outer_episodes=500, inner_episodes=100):
    outer_agent = QLearningAgent(state_size=init_env.observation_space.shape[0] * init_env.observation_space.shape[1],
                                 action_size=np.prod(init_env.action_space.nvec))
    inner_agent = QLearningAgent(state_size=active_env.observation_space.shape[0] * active_env.observation_space.shape[1],
                                 action_size=np.prod(active_env.action_space.nvec))

    previous_outer_agent = deepcopy(outer_agent)  # Initialize the previous outer agent
    previous_inner_agent = deepcopy(inner_agent)  # Initialize the previous inner agent

    turns_per_game = []

    for episode in range(outer_episodes):
        print(f"Starting episode {episode + 1}/{outer_episodes}")

        # Outer environment for the current agent
        outer_state = init_env.reset()
        outer_done = False
        outer_rewards = []

        while not outer_done:
            action = outer_agent.choose_action(outer_state.flatten())
            row, col, orientation = np.unravel_index(action, init_env.action_space.nvec)
            next_state, _, outer_done, _ = init_env.step((row, col, orientation))
            outer_rewards.append((outer_state, action, next_state))
            outer_state = next_state

        # Outer environment for the previous agent
        previous_init_env = deepcopy(init_env)
        previous_board = previous_init_env.board.copy()

        # Inner environment to compute rewards for each ship placement
        active_env.agent_board = init_env.board.copy()
        active_env.opponent_board = previous_board  # Opponent uses the previous outer agent's board

        # Inner agent plays against the previous inner agent
        inner_state = active_env.reset()
        inner_done = False
        turn_count = 0

        while not inner_done:
            # Current inner agent takes a shot
            inner_action = inner_agent.choose_action(inner_state.flatten())
            row, col = np.unravel_index(inner_action, active_env.action_space.nvec)
            next_state, reward, inner_done, _ = active_env.step((row, col))
            inner_agent.update(inner_state.flatten(), inner_action, reward, next_state.flatten())
            inner_state = next_state
            turn_count += 1

            # Previous inner agent takes a shot (opponent)
            opponent_action = previous_inner_agent.choose_action(inner_state.flatten())
            row, col = np.unravel_index(opponent_action, active_env.action_space.nvec)
            _, _, _, _ = active_env.step((row, col))

        turns_per_game.append(turn_count)  # Track the number of turns in this game

        # Calculate specific rewards for each ship placement
        ship_rewards = []
        for row, col, orientation, size in init_env.ship_positions:
            if orientation == 0:
                ship_cells = active_env.agent_board[row, col:col + size]
            else:
                ship_cells = active_env.agent_board[row:row + size, col]

            unhit_cells = np.sum(ship_cells == 1)
            ship_rewards.append(unhit_cells)

        # Update outer agent with specific rewards for each placement
        for i, (state, action, next_state) in enumerate(outer_rewards):
            outer_agent.update(state.flatten(), action, ship_rewards[i], next_state.flatten())

        outer_agent.decay_exploration()
        inner_agent.decay_exploration()

        # Update the previous agents to the current agents
        previous_outer_agent = deepcopy(outer_agent)
        previous_inner_agent = deepcopy(inner_agent)

        print(f"Episode {episode + 1}: Ship placement rewards = {ship_rewards}, Turns taken = {turn_count}")

    # Plot the number of turns per game
    plt.plot(range(1, len(turns_per_game) + 1), turns_per_game)
    plt.xlabel('Episode')
    plt.ylabel('Turns per Game')
    plt.title('Turns Taken per Game During Training')
    plt.show()

    return outer_agent, inner_agent
