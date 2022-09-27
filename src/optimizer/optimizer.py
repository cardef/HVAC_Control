import numpy as np
import math

class Annealer():

    def __init__(self, n_features, energy_model, temp_model, cost_fn, past_values, initial_temp = 90, final_temp = .1, alpha=0.01):
        
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.n_features = n_features
        self.energy_model = energy_model
        self.temp_model = temp_model
        self.cost_fn = cost_fn
        self.past_values = past_values
        
    
    def simulated_annealing(self, initial_state):
        """Peforms simulated annealing to find a solution"""     
        current_temp = self.initial_temp

        # Start by initializing the current state with the initial state
        current_state = initial_state
        solution = current_state

        while current_temp > self.final_temp:
            neighbor = self.get_neighbor(current_state)

            # Check if neighbor is best so far
            cost_diff = self.get_cost(current_state) - self.get_cost(neighbor)

            # if the new solution is better, accept it
            if cost_diff > 0:
                solution = neighbor
            # if the new solution is not better, accept it with a probability of e^(-cost/temp)
            else:
                if np.random.uniform(0, 1) < math.exp(-cost_diff / current_temp):
                    solution = neighbor
            # decrement the temperature
            current_temp -= self.alpha

        return solution

    def get_cost(self, state, cost_fn):
        """Calculates cost of the argument state for your solution."""
        
        
    def get_neighbors(self, state, param_space):
        """Returns neighbors of the argument state for your solution."""
        mask = np.random.random_sample(state.size) > 0.7
        random_steps = np.zeros(state.size)
        random_steps[mask] = np.random.choice(np.arange(-2,3), np.sum(mask))
        limits, steps = zip(*param_space.values())
        inf_limits, up_limits = zip(limits)
        new_state = state + random_steps*np.array(steps)
        new_state = np.where(new_state<inf_limits, inf_limits, new_state)
        new_state = np.where(new_state>up_limits, up_limits, new_state)
        
        return new_state