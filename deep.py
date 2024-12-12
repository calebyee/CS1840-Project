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

def get_valid_placement_actions(env):
    """Get valid actions for ship placement"""
    valid_actions = []
    ship_sizes = [5, 4, 3, 3, 2]  # Standard Battleship ship sizes
    
    if not hasattr(env, 'current_ship_index'):
        return list(range(np.prod(env.action_space.nvec)))
        
    ship_size = ship_sizes[env.current_ship_index]
    
    for row in range(env.board_size):
        for col in range(env.board_size):
            for orientation in range(2):  # 0: horizontal, 1: vertical
                action = np.ravel_multi_index((row, col, orientation), env.action_space.nvec)
                # Create temporary copy of board to test placement
                temp_board = env.board.copy()
                if orientation == 0:  # horizontal
                    if col + ship_size <= env.board_size:
                        # Check if placement would overlap with existing ships
                        if not np.any(temp_board[row, col:col + ship_size]):
                            valid_actions.append(action)
                else:  # vertical
                    if row + ship_size <= env.board_size:
                        # Check if placement would overlap with existing ships
                        if not np.any(temp_board[row:row + ship_size, col]):
                            valid_actions.append(action)
    return valid_actions

def train_nested_mdp(init_env, active_env, outer_episodes=50, inner_episodes=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize agents with correct type
    outer_agent = DQNAgent(
        state_size=init_env.board_size,
        action_size=np.prod(init_env.action_space.nvec),
        device=device,
        is_placement_agent=True
    )
    
    inner_agent = DQNAgent(
        state_size=active_env.board_size,
        action_size=active_env.board_size * active_env.board_size,
        device=device,
        is_placement_agent=False
    )

    previous_outer_agent = deepcopy(outer_agent)
    previous_inner_agent = deepcopy(inner_agent)

    turns_per_game = []
    
    for episode in range(outer_episodes):
        print(f"Starting episode {episode + 1}/{outer_episodes}")

        # Reset environments and tracking variables
        outer_state = init_env.reset()
        outer_done = False
        current_hit_sequence = []

        # Ship placement phase
        while not outer_done:
            # Get valid placement actions
            valid_actions = get_valid_placement_actions(init_env)
            
            state_tensor = torch.FloatTensor(outer_state).unsqueeze(0)
            action = outer_agent.choose_action(state_tensor, valid_actions)
            
            if action is None:
                print("No valid actions available for ship placement")
                break
                
            row, col, orientation = np.unravel_index(action, init_env.action_space.nvec)
            next_state, reward, outer_done, info = init_env.step((row, col, orientation))
            
            if 'error' not in info:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                outer_agent.update(state_tensor, action, reward, next_state_tensor, outer_done)
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
            # Get valid actions for previous agent
            valid_actions = get_valid_placement_actions(previous_init_env)
            
            prev_state_tensor = torch.FloatTensor(prev_state).unsqueeze(0)
            prev_action = previous_outer_agent.choose_action(prev_state_tensor, valid_actions)
            
            if prev_action is None:
                print("No valid actions available for opponent ship placement")
                break
                
            row, col, orientation = np.unravel_index(prev_action, previous_init_env.action_space.nvec)
            prev_state, _, prev_done, info = previous_init_env.step((row, col, orientation))
        
        active_env.opponent_board = previous_init_env.board.copy()

        print(f"\nBattle Phase Setup - Episode {episode + 1}")
        print(f"Agent's board (our ships):\n{active_env.agent_board}")
        print(f"Opponent's board (their ships):\n{active_env.opponent_board}")

        # Battle phase
        inner_state = inner_agent.get_state_representation(active_env.agent_shots)
        inner_done = False
        agent_turns = 0
        opponent_turns = 0
        max_shots = init_env.board_size * init_env.board_size

        while not inner_done and (agent_turns < max_shots and opponent_turns < max_shots):
            # Get valid actions
            valid_actions = get_valid_actions(active_env, active_env.agent_shots)
            
            # Agent's turn
            inner_action = inner_agent.choose_action(inner_state, valid_actions)
            if inner_action is None:
                break
                
            row, col = np.unravel_index(inner_action, (active_env.board_size, active_env.board_size))
            next_state, reward, inner_done, info = active_env.step((row, col), is_opponent=False)

            if 'error' not in info:
                # Check if it was a hit and if a ship was sunk
                hit = reward > 0
                sunk_ship = 'ship_sunk' in info and info['ship_sunk']
                game_over = 'game_over' in info and info['game_over']
                
                # Calculate enhanced reward
                shaped_reward = inner_agent.calculate_reward(hit, sunk_ship, game_over)
                
                hit_status = "Hit!" if hit else "Miss"
                agent_turns += 1
                print(f"Turn {agent_turns + opponent_turns}: Agent shot at ({row}, {col}) - {hit_status}")
                
                # Get next state representation
                next_state_rep = inner_agent.get_state_representation(
                    next_state, 
                    last_hit_pos=(row, col) if hit else None
                )
                
                # Update the agent
                inner_agent.update(inner_state, inner_action, shaped_reward, next_state_rep, inner_done, hit=hit)
                
                if game_over:
                    print(f"Agent wins! All opponent ships sunk in {agent_turns} shots!")
                    inner_done = True
                    break

                inner_state = next_state_rep

                # Opponent's turn
                if not inner_done:
                    valid_opponent_actions = get_valid_actions(active_env, active_env.opponent_shots)
                    opponent_state = inner_agent.get_state_representation(active_env.opponent_shots)
                    
                    valid_opponent_move = False
                    while not valid_opponent_move and not inner_done:
                        opponent_action = previous_inner_agent.choose_action(opponent_state, valid_opponent_actions)
                        if opponent_action is None:
                            break
                            
                        opp_row, opp_col = np.unravel_index(opponent_action, (active_env.board_size, active_env.board_size))
                        next_state, opp_reward, opponent_done, opp_info = active_env.step((opp_row, opp_col), is_opponent=True)
                        
                        if 'error' not in opp_info:
                            hit_status = "Hit!" if opp_reward > 0 else "Miss"
                            opponent_turns += 1
                            print(f"Turn {agent_turns + opponent_turns}: Opponent shot at ({opp_row}, {opp_col}) - {hit_status}")
                            valid_opponent_move = True
                            
                            next_state_rep = inner_agent.get_state_representation(
                                next_state,
                                last_hit_pos=(opp_row, opp_col) if opp_reward > 0 else None
                            )
                            inner_state = next_state_rep
                            
                            if opponent_done:
                                print(f"Opponent wins! All agent ships sunk in {opponent_turns} shots!")
                                inner_done = True

        # Print final game state and statistics
        total_turns = agent_turns + opponent_turns
        turns_per_game.append(total_turns)
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
        print("-" * 50)

        # Update previous agents
        previous_outer_agent = deepcopy(outer_agent)
        previous_inner_agent = deepcopy(inner_agent)

        # Save models periodically
        # if (episode + 1) % 10 == 0:
        #     outer_agent.save(f'outer_agent_episode_{episode+1}.pth')
        #     inner_agent.save(f'inner_agent_episode_{episode+1}.pth')

    # Plot training results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(turns_per_game) + 1), turns_per_game)
    plt.xlabel('Episode')
    plt.ylabel('Turns per Game')
    plt.title('Turns Taken per Game During Training')
    plt.grid(True)
    plt.show()

    return outer_agent, inner_agent

# Create and train the environments
if __name__ == "__main__":
    init_env = BattleshipPlacementEnv()
    active_env = BattleshipAttackEnv()
    train_nested_mdp(init_env, active_env, outer_episodes=50, inner_episodes=100)
