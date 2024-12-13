import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import os
import glob
from tabular import BattleshipPlacementEnv, BattleshipAttackEnv, QLearningAgent, get_valid_actions

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

def evaluate_agent(placement_agent, attack_agent, num_games=20):
    """Evaluate agent against random opponent"""
    wins = 0
    total_moves = []
    
    for game in range(num_games):
        # Initialize environments
        init_env = BattleshipPlacementEnv()
        active_env = BattleshipAttackEnv()
        
        # Place ships
        state = init_env.reset()
        done = False
        while not done:
            valid_actions = []
            for action in range(np.prod(init_env.action_space.nvec)):
                row, col, orientation = np.unravel_index(action, init_env.action_space.nvec)
                if init_env._is_valid_placement(row, col, init_env.ship_sizes[init_env.current_ship_index], orientation):
                    valid_actions.append(action)
            
            action = placement_agent.choose_action(state.flatten(), valid_actions)
            row, col, orientation = np.unravel_index(action, init_env.action_space.nvec)
            state, _, done, _ = init_env.step((row, col, orientation))
        
        # Set up battle phase
        active_env.reset()
        active_env.agent_board = init_env.board.copy()
        
        # Random opponent board setup
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
        
        active_env.opponent_board = opponent_board
        
        # Battle phase
        state = active_env.agent_shots
        done = False
        moves = 0
        
        while not done and moves < 200:  # Add move limit to prevent infinite games
            # Agent's turn
            valid_actions = []
            for i in range(active_env.board_size):
                for j in range(active_env.board_size):
                    if active_env.agent_shots[i, j] == 0:
                        valid_actions.append(i * active_env.board_size + j)
            
            action = attack_agent.choose_action(state.flatten(), valid_actions)
            if action is None:
                break
                
            row, col = np.unravel_index(action, (active_env.board_size, active_env.board_size))
            state, _, agent_done, _ = active_env.step((row, col))
            moves += 1
            
            if agent_done:
                wins += 1
                break
            
            # Random opponent's turn
            opponent_action = random_shot(active_env, active_env.opponent_shots)
            if opponent_action is None:
                break
                
            opp_row, opp_col = opponent_action
            _, _, opponent_done, _ = active_env.step((opp_row, opp_col), is_opponent=True)
            moves += 1
            
            if opponent_done:
                break
            
        total_moves.append(moves)
    
    return wins / num_games, np.mean(total_moves)

def find_checkpoints(checkpoint_dir):
    """Find all available checkpoint pairs and their episode numbers"""
    placement_files = glob.glob(os.path.join(checkpoint_dir, "placement_agent_*.pkl"))
    attack_files = glob.glob(os.path.join(checkpoint_dir, "attack_agent_*.pkl"))
    
    # Extract episode numbers and ensure both placement and attack checkpoints exist
    episodes = []
    for p_file in placement_files:
        episode = int(p_file.split("_")[-1].split(".")[0])
        if os.path.exists(os.path.join(checkpoint_dir, f"attack_agent_{episode}.pkl")):
            episodes.append(episode)
    
    return sorted(episodes)

def test_and_plot(checkpoint_dir):
    # Find all available checkpoints
    episodes = find_checkpoints(checkpoint_dir)
    
    if not episodes:
        print("No checkpoint pairs found in directory:", checkpoint_dir)
        return
    
    win_rates = []
    avg_moves = []
    
    print(f"Found checkpoints for episodes: {episodes}")
    
    for episode in episodes:
        # Load agents at this checkpoint
        placement_path = f"{checkpoint_dir}/placement_agent_{episode}.pkl"
        attack_path = f"{checkpoint_dir}/attack_agent_{episode}.pkl"
        
        print(f"\nEvaluating checkpoint from episode {episode}...")
        
        # Load the agents
        with open(placement_path, 'rb') as f:
            placement_agent = pickle.load(f)
        with open(attack_path, 'rb') as f:
            attack_agent = pickle.load(f)
        
        # Evaluate
        win_rate, moves = evaluate_agent(placement_agent, attack_agent)
        win_rates.append(win_rate)
        avg_moves.append(moves)
        print(f"Episode {episode}: Win Rate = {win_rate:.2%}, Avg Moves = {moves:.1f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Win rate plot
    plt.subplot(1, 2, 1)
    plt.plot(episodes, win_rates, 'b-', label='Win Rate')
    plt.xlabel('Training Episodes')
    plt.ylabel('Win Rate')
    plt.title('Agent Win Rate vs Random Opponent')
    plt.grid(True)
    
    # Average moves plot
    plt.subplot(1, 2, 2)
    plt.plot(episodes, avg_moves, 'r-', label='Avg Moves')
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Moves per Game')
    plt.title('Average Game Length')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Specify the path where agent checkpoints are saved
    checkpoint_dir = "checkpoints"
    test_and_plot(checkpoint_dir) 