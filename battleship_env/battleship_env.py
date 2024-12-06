import gym
from gym import spaces
import numpy as np

class BattleshipActivePhaseEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, agent_board, opponent_board, opponent_ships):
        """
        Initialize the Battleship environment.

        Args:
        - agent_board (ndarray): The player's board (not used in this phase).
        - opponent_board (ndarray): The opponent's board with ship placements.
        - opponent_ships (list): List of ship sizes for the opponent.
        """
        super(BattleshipActivePhaseEnv, self).__init__()
        
        # Game configurations
        self.grid_size = agent_board.shape[0]
        self.agent_board = agent_board  # The agent's board (ships placed)
        self.opponent_board = opponent_board  # The opponent's board (ships placed)
        
        # Ship tracking for the opponent
        self.opponent_ships = opponent_ships
        self.remaining_ship_segments = {size: size for size in opponent_ships}
        
        # Action space: Target a cell on the grid
        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)
        
        # Observation space: What the agent knows about the opponent's board
        # 0: Unknown, 1: Miss, 2: Hit, 3: Sunk
        self.observation_space = spaces.Box(low=0, high=3, 
                                            shape=(self.grid_size, self.grid_size), dtype=np.int8)
        
        # Initialize the game
        self.reset()

    def reset(self):
        # Reset the agent's knowledge of the opponent's board
        self.agent_view = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Reinitialize ship tracking
        self.remaining_ship_segments = {size: size for size in self.opponent_ships}
        self.done = False
        return self.agent_view

    def step(self, action):
        if self.done:
            raise Exception("Game is already over. Please reset the environment.")
        
        # Translate action into grid coordinates
        row, col = divmod(action, self.grid_size)
        
        # Check if the action is valid
        if self.agent_view[row, col] != 0:
            raise ValueError(f"Cell ({row}, {col}) already targeted. Choose another action.")
        
        reward = 0  # Default reward is 0 (miss)
        
        # Determine if the shot is a hit or miss
        if self.opponent_board[row, col] == 1:  # Hit
            self.agent_view[row, col] = 2  # Mark as hit
            reward = 1
            
            # Update ship tracking
            self._update_ship_tracking(row, col)
        else:  # Miss
            self.agent_view[row, col] = 1  # Mark as miss
        
        # Check if all opponent ships are sunk
        if all(count == 0 for count in self.remaining_ship_segments.values()):
            self.done = True
        
        return self.agent_view, reward, self.done, {}

    def render(self, mode='human'):
        print("Agent's view of the opponent's board:")
        print(self.agent_view)

    def _update_ship_tracking(self, row, col):
        """
        Updates the ship tracking after a hit and checks if any ship is sunk.
        """
        # Check all ships to see if this hit corresponds to one
        for size, count in self.remaining_ship_segments.items():
            if count > 0:  # Only check unsunk ships
                # Count how many cells of this ship remain on the board
                hits_remaining = np.sum(self.opponent_board == 1)
                
                # If the count drops to 0, the ship is sunk
                if hits_remaining == 0:
                    self.remaining_ship_segments[size] = 0
                    self._mark_sunk_ship(size)
                    print(f"Ship of size {size} has been sunk!")

    def _mark_sunk_ship(self, size):
        """
        Marks a sunk ship on the agent's view of the board.
        """
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.opponent_board[row, col] == 1:
                    self.agent_view[row, col] = 3  # Mark as sunk

    def close(self):
        pass


