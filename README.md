# Battleship MDP Environment

This repository contains a custom Battleship environment built as a final project for CS1840: Reinforcement Learning. The authors are Caleb Yee, Pranav Pendri, and Priyanka Kaul. It is organized into 5 folders: 

1. Initial Exploration
This folder contains our first attempts at the nested MDP approach to a Battleship setting. We included these files as additional context to demonstrate our early thought processes and coding implementations. 

2. Tabular 
These files contain our implementations with tabular q-learning. These are the next step up from our initial exploration files as they reflect further adjustments to our initial approach. 

3. DQN 
Here, we instead use deep q-learning combined with a neural network (deep q-network, or DQN) to select actions. We also no longer use self-play to train the agents, rather, we train the agent against a baseline strategy. 

4. Final Implementation
These files contain our final approach to the problem we set out to tackle. The outcomes are reflected in the results section of our final report. 

5. Testing 
Here, we include code to test and plot our results.