# Abstract base class for agents
# Coding baseline policy that we'll test the agent against 
import random 

# pri starts here, coding the baseline policy place 
# include ship placement somewhere!! 

class Baseline: # do i need anything more here? 
    def __init__(self):
        self.shots = [] # list with a history of spots tried 
        self.hits = [] # list with a history of spots that we've hit 

    def reset(self): 
        """
        Resets back to default 
        """
        self.shots = []
        self.hits = [] 

    def get_neighbors(self, spot):
        """
        Gets the neighbors of a specific spot 
        """
        x, y = spot
        neighbors = [(x + 1, y), (x + 1, y + 1), (x + 1, y - 1), (x, y + 1), (x, y - 1), (x - 1, y), (x - 1, y + 1), (x - 1, y - 1)]
        
        valid_neighbors = [(i, j) for (i, j) in neighbors if 0 <= i <= 9 and 0 <= j <= 9]
    
        return valid_neighbors


    def act(self, epsilon): 
        """
        Selects an action 
        uses possible_spots to keep track of spots that i could hit 
        opportunity to expand here with an epsilon-greedy approach 
        """
        grid = [(row, col) for row in range(10) for col in range(10)]
        possible_spots = list(set(grid) - set(self.shots))

        if not self.hits: # if no spots have been hit
            choice = random.choice(possible_spots)
        else: # if some spots have been hit 
            hit_spot = self.hits[-1] # gets most recently hit spot 
            neighbors = self.get_neighbors(hit_spot)
            possible_neighbors = list(set(neighbors) & set(possible_spots)) # intersection of neighbors and possible spots
            if possible_neighbors: # if list is empty 
                choice = random.choice(possible_neighbors) 
            else: 
                choice = random.choice(possible_spots)

        self.shots.append(choice) 
        return choice 

