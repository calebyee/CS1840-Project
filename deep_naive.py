import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
from dqn import DQNAgent
import random
from deep_selfplay import BattleshipAttackEnv
from deep_selfplay import BattleshipPlacementEnv

# Reuse BattleshipPlacementEnv and BattleshipAttackEnv from deep.py
# ... [Copy the environment classes from deep.py] ...

def random_ship_placement(env):
    """Place ships randomly on the board"""
    board = np.zeros((env.board_size, env.board_size), dtype=np.int8)
    ship_positions = []
    
    for ship_size in env.ship_sizes:
        placed = False
        while not placed:
            # Random position and orientation
            row = random.randint(0, env.board_size - 1)
            col = random.randint(0, env.board_size - 1)
            orientation = random.randint(0, 1)  # 0: horizontal, 1: vertical
            
            # Check if placement is valid
            if orientation == 0:  # horizontal
                if col + ship_size <= env.board_size:
                    if not np.any(board[row, col:col + ship_size]):
                        board[row, col:col + ship_size] = 1
                        ship_positions.append((row, col, ship_size, True))
                        placed = True
            else:  # vertical
                if row + ship_size <= env.board_size:
                    if not np.any(board[row:row + ship_size, col]):
                        board[row:row + ship_size, col] = 1
                        ship_positions.append((row, col, ship_size, False))
                        placed = True
    
    return board, ship_positions

def random_shot(env, shots):
    """Generate a random valid shot"""
    valid_positions = []
    for i in range(env.board_size):
        for j in range(env.board_size):
            if shots[i, j] == 0:  # Position hasn't been shot at
                valid_positions.append((i, j))
    
    if valid_positions:
        return random.choice(valid_positions)
    return None

def hunt_and_target_shot(env, shots):
    """
    Enhanced hunt and target strategy:
    1. Random shots until a hit is found (hunt mode)
    2. Try adjacent squares after a hit (target mode)
    3. When multiple hits are found, try to follow the line
    4. Continue until ship is sunk, then return to hunt mode
    """
    board_size = env.board_size
    
    # Find all hits and analyze their patterns
    hits = []
    hit_lines = []  # Store lines of consecutive hits
    
    # First pass: collect all hits
    for i in range(board_size):
        for j in range(board_size):
            if shots[i, j] == 1:
                hits.append((i, j))
    
    # If we have multiple hits, look for lines and patterns
    if len(hits) >= 2:
        # Group hits into lines
        visited = set()
        for hit in hits:
            if hit in visited:
                continue
                
            current_line = [hit]
            visited.add(hit)
            
            # Check horizontal and vertical directions
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                r, c = hit[0] + dr, hit[1] + dc
                while (r, c) in hits and (r, c) not in visited:
                    current_line.append((r, c))
                    visited.add((r, c))
                    r, c = r + dr, c + dc
            
            if len(current_line) > 1:
                hit_lines.append(sorted(current_line))
    
    # If we have a line of hits, try to extend it
    for line in hit_lines:
        if len(line) >= 2:
            # Get direction of the line
            is_horizontal = line[0][0] == line[1][0]
            
            # Try both ends of the line
            for end in [line[0], line[-1]]:
                # Check both directions perpendicular to the line
                if is_horizontal:
                    # Check left and right of the line
                    for dc in [-1, 1]:
                        r, c = end[0], end[1] + dc
                        if (0 <= r < board_size and 0 <= c < board_size and 
                            shots[r, c] == 0):
                            return (r, c)
                else:
                    # Check up and down of the line
                    for dr in [-1, 1]:
                        r, c = end[0] + dr, end[1]
                        if (0 <= r < board_size and 0 <= c < board_size and 
                            shots[r, c] == 0):
                            return (r, c)
    
    # If we have single hits, try adjacent squares
    for hit in hits:
        # Check if this hit has any unexplored adjacent squares
        adjacents = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            r, c = hit[0] + dr, hit[1] + dc
            if (0 <= r < board_size and 0 <= c < board_size and 
                shots[r, c] == 0):
                adjacents.append((r, c))
        
        if adjacents:
            # If we have a previous hit in a line, prioritize that direction
            for prev_hit in hits:
                if prev_hit != hit:
                    # Check if hits are in line (same row or column)
                    if prev_hit[0] == hit[0]:  # same row
                        # Prioritize horizontal shots
                        horizontal_options = [pos for pos in adjacents 
                                           if pos[0] == hit[0]]
                        if horizontal_options:
                            return random.choice(horizontal_options)
                    elif prev_hit[1] == hit[1]:  # same column
                        # Prioritize vertical shots
                        vertical_options = [pos for pos in adjacents 
                                         if pos[1] == hit[1]]
                        if vertical_options:
                            return random.choice(vertical_options)
            
            # If no line found, try any adjacent square
            return random.choice(adjacents)
    
    # Hunt mode: random shot in unexplored squares
    valid_positions = []
    for i in range(board_size):
        for j in range(board_size):
            if shots[i, j] == 0:  # Position hasn't been shot at
                valid_positions.append((i, j))
    
    if valid_positions:
        return random.choice(valid_positions)
    return None

def train_against_random(init_env, active_env, episodes=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize agent
    agent = DQNAgent(
        state_size=active_env.board_size,
        action_size=active_env.board_size * active_env.board_size,
        device=device,
        is_placement_agent=False
    )

    placement_agent = DQNAgent(
        state_size=init_env.board_size,
        action_size=np.prod(init_env.action_space.nvec),
        device=device,
        is_placement_agent=True
    )

    turns_per_game = []
    wins = 0
    game_outcomes = []  # Track who won each game (True for agent, False for opponent)
    
    for episode in range(episodes):
        print(f"Starting episode {episode + 1}/{episodes}")
        
        # Reset epsilon at the start of each episode
        agent.reset_epsilon()
        placement_agent.reset_epsilon()
        
        # Agent ship placement phase
        outer_state = init_env.reset()
        outer_done = False
        
        while not outer_done:
            valid_actions = get_valid_placement_actions(init_env)
            state_tensor = torch.FloatTensor(outer_state).unsqueeze(0)
            action = placement_agent.choose_action(state_tensor, valid_actions)
            
            if action is None:
                print("No valid actions available for ship placement")
                break
                
            row, col, orientation = np.unravel_index(action, init_env.action_space.nvec)
            next_state, reward, outer_done, info = init_env.step((row, col, orientation))
            
            if 'error' not in info:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                placement_agent.update(state_tensor, action, reward, next_state_tensor, outer_done)
            
            outer_state = next_state

        # Set up battle phase
        active_env.reset()
        
        # Set up agent's board and register ships
        active_env.agent_board = init_env.board.copy()
        # Register agent's ships from placement phase
        for ship_pos in init_env.ship_positions:
            row, col, orientation, ship_size = ship_pos
            active_env.add_ship(row, col, ship_size, orientation == 0, is_opponent=False)
        
        # Random opponent ship placement
        opponent_board, opponent_ships = random_ship_placement(init_env)
        active_env.opponent_board = opponent_board
        
        # Register opponent's ships
        for row, col, size, is_horizontal in opponent_ships:
            active_env.add_ship(row, col, size, is_horizontal, is_opponent=True)

        # Battle phase
        inner_state = agent.get_state_representation(active_env.agent_shots)
        inner_done = False
        agent_turns = 0
        opponent_turns = 0
        max_shots = init_env.board_size * init_env.board_size
        
        print(f"Starting epsilon: {agent.epsilon:.4f}")

        while not inner_done and (agent_turns < max_shots and opponent_turns < max_shots):
            # Agent's turn
            valid_actions = get_valid_actions(active_env, active_env.agent_shots)
            inner_action = agent.choose_action(inner_state, valid_actions)
            
            if inner_action is None:
                break
                
            row, col = np.unravel_index(inner_action, (active_env.board_size, active_env.board_size))
            next_state, reward, inner_done, info = active_env.step((row, col), is_opponent=False)

            if 'error' not in info:
                hit = reward > 0
                sunk_ship = 'ship_sunk' in info and info['ship_sunk']
                game_over = 'game_over' in info and info['game_over']
                
                shaped_reward = agent.calculate_reward(hit, sunk_ship, game_over)
                agent_turns += 1
                
                # Print hit/sink information for agent
                if hit:
                    # print(f"Agent hit at ({row}, {col})!")
                    if sunk_ship:
                        ship_type = info.get('ship_type', 'Unknown ship')
                        print(f"Agent sunk the {ship_type} in {agent_turns} shots!")
                # else:
                #     print(f"Agent missed at ({row}, {col})")
                
                next_state_rep = agent.get_state_representation(
                    next_state, 
                    last_hit_pos=(row, col) if hit else None
                )
                
                agent.update(inner_state, inner_action, shaped_reward, next_state_rep, inner_done, hit=hit)
                
                if game_over:
                    print(f"Agent wins! All opponent ships sunk in {agent_turns} shots!")
                    inner_done = True
                    wins += 1
                    game_outcomes.append(True)  # Agent win

                inner_state = next_state_rep

                # Opponent's turn using hunt and target strategy
                if not inner_done:
                    opponent_action = hunt_and_target_shot(active_env, active_env.opponent_shots)
                    if opponent_action is None:
                        break
                        
                    opp_row, opp_col = opponent_action
                    next_state, opp_reward, opponent_done, opp_info = active_env.step(
                        (opp_row, opp_col), 
                        is_opponent=True
                    )
                    
                    opponent_turns += 1
                    
                    # Print opponent hit/sink information
                    if opp_reward > 0:
                        #print(f"Opponent hit at ({opp_row}, {opp_col})!")
                        if 'ship_sunk' in opp_info and opp_info['ship_sunk']:
                            ship_type = opp_info.get('ship_type', 'Unknown ship')
                            print(f"Opponent sunk the {ship_type} in {opponent_turns} shots!")
                    # else:
                    #     print(f"Opponent missed at ({opp_row}, {opp_col})")
                    
                    if opponent_done:
                        print(f"Hunt-and-target opponent wins! All agent ships sunk in {opponent_turns} shots!")
                        inner_done = True
                        game_outcomes.append(False)  # Opponent win

        print(f"Ending epsilon: {agent.epsilon:.4f}")
        
        total_turns = agent_turns + opponent_turns
        turns_per_game.append(total_turns)
        print(f"\nEpisode {episode + 1} completed in {total_turns} total shots")
        print("\nFinal Score:")
        print(f"Agent hits: {np.sum(active_env.agent_shots == 1)}/{np.sum(active_env.opponent_board == 1)}")
        print(f"Opponent hits: {np.sum(active_env.opponent_shots == 1)}/{np.sum(active_env.agent_board == 1)}")
        print("-" * 50)

    win_rate = wins / episodes
    print(f"\nFinal Statistics:")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average turns per game: {np.mean(turns_per_game):.1f}")

    # Plot training results with colored points
    plt.figure(figsize=(12, 5))
    
    # First subplot: Game lengths with colored points
    plt.subplot(1, 2, 1)
    episodes_range = range(1, len(turns_per_game) + 1)
    
    # Plot points with different colors based on winner
    for i, (turns, agent_won) in enumerate(zip(turns_per_game, game_outcomes), 1):
        color = 'blue' if agent_won else 'red'
        plt.scatter(i, turns, c=color, alpha=0.6)
    
    plt.plot(episodes_range, turns_per_game, color='gray', alpha=0.3)  # Connection line
    plt.xlabel('Episode')
    plt.ylabel('Turns per Game')
    plt.title('Turns Taken per Game During Training\nBlue: Agent wins, Red: Opponent wins')
    plt.grid(True)
    
    # Second subplot: Moving average
    # plt.subplot(1, 2, 2)
    # window_size = 5
    # moving_avg = np.convolve(turns_per_game, np.ones(window_size)/window_size, mode='valid')
    # plt.plot(range(window_size, len(turns_per_game) + 1), moving_avg)
    # plt.xlabel('Episode')
    # plt.ylabel('Moving Average Turns')
    # plt.title(f'{window_size}-Episode Moving Average')
    # plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    return agent, placement_agent

# Helper functions from deep.py
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

if __name__ == "__main__":
    init_env = BattleshipPlacementEnv()
    active_env = BattleshipAttackEnv()
    train_against_random(init_env, active_env, episodes=200)