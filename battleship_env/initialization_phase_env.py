# Initialization phase environment via OpenAI Gym

import gym
from gym import spaces
import numpy as np

class BattleshipInitializationEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, board_size=10, ship_sizes=[5, 4, 3, 3, 2], active_env=None):
        """
        Initialization phase of Battleship.
        
        Args:
        - board_size (int): Size of the board (NxN).
        - ship_sizes (list): Sizes of the ships to place.
        - active_env (object): An instance of the active phase environment.
        """
        super(BattleshipInitializationEnv, self).__init__()
        
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.active_env = active_env  # Active phase environment
        
        # Action space: (row, col, orientation) for ship placement
        # Orientation: 0 for horizontal, 1 for vertical
        self.action_space = spaces.MultiDiscrete([board_size, board_size, 2])
        
        # Observation space: The current state of the board
        self.observation_space = spaces.Box(low=0, high=1, 
                                            shape=(board_size, board_size), dtype=np.int8)
        
        self.reset()

    def reset(self):
        # Reset the board
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        
        # Remaining ships to place
        self.remaining_ships = list(self.ship_sizes)
        self.current_ship_index = 0
        
        self.done = False
        return self.board

    def step(self, action):
        if self.done:
            raise Exception("Initialization phase is done. Please reset.")
        
        row, col, orientation = action
        
        # Get the size of the current ship to place
        if self.current_ship_index >= len(self.remaining_ships):
            raise Exception("No more ships to place.")
        
        ship_size = self.remaining_ships[self.current_ship_index]
        
        # Check if placement is valid
        if not self._is_valid_placement(row, col, ship_size, orientation):
            raise ValueError(f"Invalid placement for ship of size {ship_size} at ({row}, {col}).")
        
        # Place the ship
        self._place_ship(row, col, ship_size, orientation)
        
        # Move to the next ship
        self.current_ship_index += 1
        
        # Check if all ships are placed
        if self.current_ship_index >= len(self.remaining_ships):
            self.done = True
            reward = self._calculate_reward()
        else:
            reward = 0  # Reward is calculated after all ships are placed
        
        return self.board, reward, self.done, {}

    def render(self, mode='human'):
        print("Current board:")
        print(self.board)

    def _is_valid_placement(self, row, col, size, orientation):
        """
        Checks if a ship can be placed starting at (row, col) with the given orientation.
        """
        if orientation == 0:  # Horizontal
            if col + size > self.board_size:
                return False  # Ship hangs over the edge
            if np.any(self.board[row, col:col + size] == 1):
                return False  # Overlaps another ship
        else:  # Vertical
            if row + size > self.board_size:
                return False  # Ship hangs over the edge
            if np.any(self.board[row:row + size, col] == 1):
                return False  # Overlaps another ship
        return True

    def _place_ship(self, row, col, size, orientation):
        """
        Places a ship on the board.
        """
        if orientation == 0:  # Horizontal
            self.board[row, col:col + size] = 1
        else:  # Vertical
            self.board[row:row + size, col] = 1

    def _calculate_reward(self):
        """
        Simulates a game using the active phase environment and calculates the reward.
        """
        if self.active_env is None:
            raise Exception("Active phase environment is not provided.")
        
        # Pass the initialized board to the active phase environment
        self.active_env.agent_board = self.board.copy()
        self.active_env.reset()
        
        # Simulate the game
        done = False
        while not done:
            action = self.active_env.action_space.sample()  # Random opponent moves
            _, _, done, _ = self.active_env.step(action)
        
        # Calculate reward: number of unhit ship cells
        reward = np.sum(self.active_env.agent_board == 1)
        return reward

    def close(self):
        pass
