import numpy as np
import matplotlib.pyplot as plt

from KBandits import KBandits
from Agent import Agent

# driver parameters
initial_values_for_tests = [0, 1, 2, 5, 10, 100]
colours = ['b', 'g', 'r', 'c', 'm', 'y']

# kbandits parameters
k = 5
mean_mean = 0
mean_sd = 1

sd_mean = 1
sd_sd = 0

# agent parameters
n_steps = 1000
eps = 0.0
initial_values = 5

def main():
    fig, axes = plt.subplots(1, 1, figsize=(12,8))

    # -----------------------
    # Constant variance
    # -----------------------
    bandits = KBandits(k, (mean_mean, mean_sd), (sd_mean, sd_sd))

    # all in one plot
    for i in range(len(initial_values_for_tests)):
        # print("-" * 10 + f" {initial_values_for_tests[i]} " + "-" * 10)
        optimal_action_percentages, _ = Agent.run(k, initial_values_for_tests[i], eps, bandits, n_steps)
        axes.plot(optimal_action_percentages, label=f"init: {initial_values_for_tests[i]}", color=colours[i])
 
    print(bandits.reward_distributions)

    axes.title.set_text("Reward Distribution for arm i is N(X_i, 1) where X_i ~ N(0, 1)")
    axes.set_ylabel("% Optimal Moves")
    axes.legend()

    # bandits2 = KBandits(k, (mean_mean, mean_sd), (1, 0))

    # for i in range(len(initial_values_for_tests)):
    #     # print("-" * 10 + f" {initial_values_for_tests[i]} " + "-" * 10)
    #     optimal_action_percentages, _ = Agent.run(k, initial_values_for_tests[i], eps, bandits2, n_steps)
    #     axes[1].plot(optimal_action_percentages, label=f"init: {initial_values_for_tests[i]}", color=colours[i])

    # axes[1].title.set_text("Reward Distribution for arm i is N(X_i, 1) where X_i ~ N(0, 1)")
    # axes[1].set_ylabel("% Optimal Moves")
    # axes[1].legend()


    plt.show()

if __name__ == "__main__":
    main()