import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import os
import glob
from tabular_relative import BattleshipAttackEnv, QLearningAgent
from tabular import BattleshipPlacementEnv
from dqn_relative import DQNAgent

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

def evaluate_agent(placement_agent, attack_agent, num_games=10):
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
            
            # Convert state to dictionary format for DQN
            state_dict = {
                'board': state,
                'last_hit': None,
                'shot_history': []
            }
            
            action = placement_agent.choose_action(state_dict, valid_actions)
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
        state = active_env.get_state_representation()  # Get dictionary state
        done = False
        moves = 0
        
        while not done and moves < 200:  # Add move limit to prevent infinite games
            # Agent's turn
            valid_actions = active_env.get_valid_relative_actions()
            action = attack_agent.choose_action(state, valid_actions)
            
            next_state, reward, agent_done, info = active_env.step(action)
            moves += 1
            
            if 'error' in info:
                continue
                
            if agent_done:
                wins += 1
                break
            
            # Random opponent's turn
            opponent_action = len(active_env.relative_actions) - 1  # Random action
            _, _, opponent_done, _ = active_env.step(opponent_action, is_opponent=True)
            moves += 1
            
            if opponent_done:
                break
            
            state = next_state
            
        total_moves.append(moves)
    
    return wins / num_games, np.mean(total_moves)

def find_checkpoints(checkpoint_dir):
    """Find all available checkpoint pairs and their episode numbers"""
    agent_placement_files = glob.glob(os.path.join(checkpoint_dir, "agent_placement_*.pt"))
    
    # Extract episode numbers and ensure all necessary files exist
    episodes = []
    for p_file in agent_placement_files:
        episode_str = p_file.split("_")[-1].split(".")[0]
        if episode_str == "final":
            continue  # Skip the final checkpoint
        
        episode = int(episode_str)
        if all(os.path.exists(os.path.join(checkpoint_dir, f"agent_{prefix}_{episode}.pt")) 
               for prefix in ["placement", "attack"]):
            episodes.append(episode)
    
    return sorted(episodes)

def test_and_plot(checkpoint_dir):
    # Find all available checkpoints
    episodes = find_checkpoints(checkpoint_dir)
    
    if not episodes:
        print("No checkpoint pairs found in directory:", checkpoint_dir)
        return
    
    agent_win_rates = []
    agent_avg_moves = []
    
    print(f"Found checkpoints for episodes: {episodes}")
    
    for episode in episodes:
        print(f"\nEvaluating checkpoint from episode {episode}...")
        
        # Initialize agents with correct number of actions
        init_env = BattleshipPlacementEnv()
        active_env = BattleshipAttackEnv()
        
        placement_action_size = np.prod(init_env.action_space.nvec)
        attack_action_size = len(active_env.relative_actions)
        
        # Create and load agents
        placement_agent = DQNAgent(board_size=init_env.board_size, 
                                 num_actions=placement_action_size)
        attack_agent = DQNAgent(board_size=active_env.board_size, 
                              num_actions=attack_action_size)
        
        placement_agent.load(f"{checkpoint_dir}/agent_placement_{episode}.pt")
        attack_agent.load(f"{checkpoint_dir}/agent_attack_{episode}.pt")
        
        # Evaluate agents
        win_rate, avg_moves = evaluate_agent(placement_agent, attack_agent)
        
        agent_win_rates.append(win_rate)
        agent_avg_moves.append(avg_moves)
        
        print(f"Episode {episode}: Win Rate = {win_rate:.2%}, Avg Moves = {avg_moves:.1f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Win rate plot
    plt.subplot(1, 2, 1)
    plt.plot(episodes, agent_win_rates, 'b-', label='Agent')
    plt.xlabel('Training Episodes')
    plt.ylabel('Win Rate vs Random')
    plt.title('Agent Win Rates vs Random Opponent')
    plt.legend()
    plt.grid(True)
    
    # Average moves plot
    plt.subplot(1, 2, 2)
    plt.plot(episodes, agent_avg_moves, 'b-', label='Agent')
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Moves per Game')
    plt.title('Average Game Length vs Random')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Specify the path where agent checkpoints are saved
    checkpoint_dir = "checkpoints_deep_relative"
    test_and_plot(checkpoint_dir) 