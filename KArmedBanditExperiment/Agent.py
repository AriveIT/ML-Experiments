import numpy as np
from KBandits import KBandits

class Agent:

    def run(k, initial_value, eps, bandits, n_steps):
        optimal_action_percentages = [] # for stats
        average_rewards = []
        n_optimal_action = 0

        Q = [] # action value estimates
        N = [] # number of times taken each action

        for i in range(k):
            Q.append(initial_value)
            N.append(1)


        for i in range(n_steps):
            a = np.argmax(Q) if np.random.rand() > eps else np.random.choice(k)
            r = bandits.take_action(a)


            # update values
            N[a] += 1
            Q[a] += (1 / float(N[a])) * (r - Q[a]) # incrementally update the action value estimate

            # for plotting
            if a == bandits.optimal_action:
                n_optimal_action += 1
            optimal_action_percentages.append(n_optimal_action / (i + 1))
            
            avg_reward = r if i == 0 else (1 / float(i + 1)) * (r - average_rewards[-1])
            average_rewards.append(avg_reward)

            # see how things are going so far
            # if i < k or i == n_steps - 1:
            #     print(Q)
            #     print([param[0] for param in bandits.reward_distributions])
            #     print()
        
        return optimal_action_percentages, average_rewards


