# Battleship MDP Environment

This repository contains a custom Battleship environment built with OpenAI Gym, a Q-learning agent, and self-play logic. It also includes scripts for training, evaluation, and testing.


Final repository structure:

CS1840-Project/
├── README.md
├── setup.py
├── requirements.txt
├── battleship/                 # Main package
│   ├── __init__.py
│   ├── environments/           # Environments for outer and inner MDP
│   │   ├── __init__.py
│   │   ├── outer_env.py        # Outer MDP for ship placement
│   │   ├── inner_env.py        # Inner MDP for active gameplay
│   │   ├── utils.py            # Shared utilities for environments
│   ├── agents/                 # RL agents and training logic
│   │   ├── __init__.py
│   │   ├── outer_agent.py      # RL agent for ship placement
│   │   ├── inner_agent.py      # RL agent for active gameplay
│   ├── policies/               # Pre-defined policies (if applicable)
│   │   ├── __init__.py
│   │   ├── random_policy.py    # Random actions for testing
│   │   ├── heuristic_policy.py # Example of heuristic-based strategy
│   ├── models/                 # Models and training logic
│   │   ├── __init__.py
│   │   ├── nested_mdp.py       # Implementation of the nested MDP
│   └── utils/                  # General utility functions
│       ├── __init__.py
│       ├── board_utils.py      # Functions for board manipulation
│       ├── visualization.py    # Rendering/plotting helpers
├── tests/                      # Unit and integration tests
│   ├── test_outer_env.py
│   ├── test_inner_env.py
│   ├── test_nested_mdp.py
│   ├── test_agents.py
├── notebooks/                  # Jupyter notebooks for experiments
│   ├── analysis.ipynb
│   ├── training_results.ipynb
├── data/                       # Data storage for experiments (optional)
│   ├── example_boards/
│   │   ├── board_1.json
│   │   ├── board_2.json
│   ├── results/
│       ├── training_metrics.csv
│       ├── performance_plots/
└── scripts/                    # CLI scripts for running experiments
    ├── train_outer.py          # Train the outer MDP agent
    ├── train_inner.py          # Train the inner MDP agent
    ├── evaluate_nested.py      # Evaluate the full nested MDP
