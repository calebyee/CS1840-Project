# Q-learning agent implementation
import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_shape, action_space, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Q-Learning Agent.

        Args:
        - state_shape (tuple): Shape of the state space (flattened for Q-table indexing).
        - action_space (gym.spaces): Action space of the environment.
        - alpha (float): Learning rate.
        - gamma (float): Discount factor.
        - epsilon (float): Exploration rate.
        """
        self.state_shape = state_shape
        self.action_space = action_space
        self.q_table = {}  # Initialize Q-table as a dictionary
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _get_q_values(self, state):
        """Retrieve Q-values for a state, initializing if not present."""
        state_key = tuple(state.flatten())  # Flatten state for Q-table key
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        return self.q_table[state_key]

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        q_values = self._get_q_values(state)
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()  # Explore
        return np.argmax(q_values)  # Exploit

    def update(self, state, action, reward, next_state):
        """Update the Q-table using the Q-learning update rule."""
        q_values = self._get_q_values(state)
        next_q_values = self._get_q_values(next_state)
        td_target = reward + self.gamma * np.max(next_q_values)
        td_error = td_target - q_values[action]
        q_values[action] += self.alpha * td_error

def train_nested_mdp(init_env, active_env, outer_agent, inner_agent, outer_episodes=500, inner_episodes=100):
    """
    Train Q-learning agents for the nested MDP.

    Args:
    - init_env: Instance of BattleshipInitializationEnv (outer MDP).
    - active_env: Instance of BattleshipActivePhaseEnv (inner MDP).
    - outer_agent: QLearningAgent for the outer MDP.
    - inner_agent: QLearningAgent for the inner MDP.
    - outer_episodes (int): Number of episodes to train the outer agent.
    - inner_episodes (int): Number of episodes to train the inner agent.
    """
    for episode in range(outer_episodes):
        # Outer MDP (Initialization Phase)
        outer_state = init_env.reset()
        outer_done = False
        total_outer_reward = 0

        while not outer_done:
            # Outer agent selects a ship placement action
            outer_action = outer_agent.choose_action(outer_state)
            outer_next_state, outer_reward, outer_done, _ = init_env.step(outer_action)
            total_outer_reward += outer_reward

            # Outer agent updates Q-table
            outer_agent.update(outer_state, outer_action, outer_reward, outer_next_state)
            outer_state = outer_next_state

        # Inner MDP (Active Phase)
        for inner_episode in range(inner_episodes):
            inner_state = active_env.reset()
            inner_done = False
            total_inner_reward = 0

            while not inner_done:
                # Inner agent selects an attack action
                inner_action = inner_agent.choose_action(inner_state)
                inner_next_state, inner_reward, inner_done, _ = active_env.step(inner_action)
                total_inner_reward += inner_reward

                # Inner agent updates Q-table
                inner_agent.update(inner_state, inner_action, inner_reward, inner_next_state)
                inner_state = inner_next_state

        # Use the outcome of the inner MDP to inform the outer MDP
        final_outer_reward = total_inner_reward
        print(f"Episode {episode + 1}/{outer_episodes}: Total Reward = {final_outer_reward}")

