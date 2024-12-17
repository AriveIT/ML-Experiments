import numpy as np

class KBandits:

    def __init__(self, k, mean_dist_parameters, sd_dist_parameters):
        self.k = k
        self.reward_distributions = [] # k * (mean, sd)

        for _ in range(k):
            mean = np.random.normal(mean_dist_parameters[0], mean_dist_parameters[1])
            sd = (np.random.normal(sd_dist_parameters[0], sd_dist_parameters[1])) ** 2 # sd is positive
            self.reward_distributions.append([mean, sd])

        self.optimal_action = np.argmax([row[0] for row in self.reward_distributions])
        self.optimal_action_expected_reward = self.reward_distributions[self.optimal_action][0]
    
    def take_action(self, a):
        parameters = self.reward_distributions[a]
        return np.random.normal(parameters[0], parameters[1])