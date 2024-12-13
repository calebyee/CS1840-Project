import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from tabular import BattleshipPlacementEnv
from tabular_relative import BattleshipAttackEnv
from dqn_relative import DQNAgent

def train_against_random(init_env, active_env, outer_episodes=20, inner_episodes=100, device="cpu"):
    # Initialize DQN agents for both phases
    num_relative_actions = len(active_env.relative_actions)
    placement_action_size = np.prod(init_env.action_space.nvec)
    
    agent_outer = DQNAgent(board_size=init_env.board_size, 
                          num_actions=placement_action_size, 
                          device=device)
    agent_inner = DQNAgent(board_size=active_env.board_size, 
                          num_actions=num_relative_actions, 
                          device=device)

    turns_per_game = []
    agent_wins = []  # 1 for win, 0 for loss
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints_deep_relative", exist_ok=True)
    
    for episode in range(outer_episodes):
        print(f"\nStarting episode {episode + 1}/{outer_episodes}")
        
        # Agent ship placement phase
        state = init_env.reset()
        done = False
        agent_rewards = []  # Store (state, action, next_state) for batch updating
        
        while not done:
            valid_actions = []
            for action in range(placement_action_size):
                row, col, orientation = np.unravel_index(action, init_env.action_space.nvec)
                if init_env._is_valid_placement(row, col, init_env.ship_sizes[init_env.current_ship_index], orientation):
                    valid_actions.append(action)
            
            # Convert state for DQN
            state_dict = {
                'board': state,
                'last_hit': None,
                'shot_history': []
            }
            
            action = agent_outer.choose_action(state_dict, valid_actions)
            row, col, orientation = np.unravel_index(action, init_env.action_space.nvec)
            next_state, _, done, info = init_env.step((row, col, orientation))
            
            if 'error' not in info:
                agent_rewards.append((state_dict, action, {
                    'board': next_state,
                    'last_hit': None,
                    'shot_history': []
                }))
                print(f"Agent placement: Ship {init_env.current_ship_index}, Position: ({row}, {col}), Orientation: {orientation}")
            
            state = next_state

        agent_board = init_env.board.copy()

        # Random opponent placement
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
        active_env.agent_board = agent_board
        active_env.opponent_board = opponent_board
        
        state = active_env.get_state_representation()
        done = False
        agent_turns = 0
        opponent_turns = 0
        max_shots = init_env.board_size * init_env.board_size
        current_player = 1  # Track whose turn it is

        while not done and (agent_turns + opponent_turns < max_shots * 2):
            if current_player == 1:
                # Agent's turn
                valid_actions = active_env.get_valid_relative_actions()
                action = agent_inner.choose_action(state, valid_actions)
                next_state, reward, done, info = active_env.step(action)

                if 'error' not in info:
                    hit = reward > 0
                    agent_turns += 1
                    print(f"Turn {agent_turns + opponent_turns}: Agent shot - {'Hit!' if hit else 'Miss'}")
                    
                    valid_next_actions = active_env.get_valid_relative_actions()
                    agent_inner.update(state, action, reward, next_state, done, valid_next_actions)
                    
                    if done:
                        print(f"Agent wins! All opponent ships sunk in {agent_turns} shots!")
                        agent_wins.append(1)
                        break

                    state = next_state
                current_player = 2  # Switch to opponent's turn
            
            else:
                # Opponent's turn (random)
                opponent_action = len(active_env.relative_actions) - 1  # Random action
                next_state, reward, opponent_done, opp_info = active_env.step(opponent_action, is_opponent=True)
                
                if 'error' not in opp_info:
                    hit = reward > 0
                    opponent_turns += 1
                    print(f"Turn {agent_turns + opponent_turns}: Opponent shot - {'Hit!' if hit else 'Miss'}")
                    
                    if opponent_done:
                        print(f"Opponent wins! All agent ships sunk in {opponent_turns} shots!")
                        agent_wins.append(0)
                        done = True
                        break

                    state = active_env.get_state_representation()
                current_player = 1  # Switch to agent's turn

        # Update placement agent
        for i, (state, action, next_state) in enumerate(agent_rewards):
            valid_next_actions = []
            for next_action in range(placement_action_size):
                row, col, orientation = np.unravel_index(next_action, init_env.action_space.nvec)
                if i + 1 < len(init_env.ship_sizes) and init_env._is_valid_placement(row, col, init_env.ship_sizes[i + 1], orientation):
                    valid_next_actions.append(next_action)
            agent_outer.update(state, action, 1.0, next_state, i == len(agent_rewards)-1, valid_next_actions)

        total_turns = agent_turns + opponent_turns
        turns_per_game.append(total_turns)

        # Save checkpoints every 20 episodes
        if episode % 10 == 0:
            print(f"Saving checkpoint at episode {episode}")
            agent_outer.save(f"checkpoints_deep_relative/agent_placement_{episode}.pt")
            agent_inner.save(f"checkpoints_deep_relative/agent_attack_{episode}.pt")

        print(f"\nEpisode {episode + 1} completed in {total_turns} total shots")
        print(f"Agent shots: {agent_turns}, Opponent shots: {opponent_turns}")
        print("-" * 50)

    # Save final models
    agent_outer.save(f"checkpoints_deep_relative/agent_placement_final.pt")
    agent_inner.save(f"checkpoints_deep_relative/agent_attack_final.pt")

    return agent_outer, agent_inner, turns_per_game, agent_wins

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create instances of the environments
    init_env = BattleshipPlacementEnv()
    active_env = BattleshipAttackEnv()

    # Train against random opponent
    agent_placement, agent_attack, turns_per_game, agent_wins = train_against_random(
        init_env, 
        active_env, 
        outer_episodes=101, 
        inner_episodes=100,
        device=device
    )
    
    # Plot training results
    plt.figure(figsize=(10, 6))
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
    plt.title('Deep Q-Learning Training Results\nBlue: Agent wins, Red: Opponent wins')
    plt.grid(True)
    
    # Print final statistics
    agent_win_rate = sum(win == 1 for win in agent_wins) / len(agent_wins)
    print(f"\nFinal Statistics:")
    print(f"Agent Win Rate: {agent_win_rate:.2%}")
    print(f"Average turns per game: {np.mean(turns_per_game):.1f}")
    
    plt.tight_layout()
    plt.show() 