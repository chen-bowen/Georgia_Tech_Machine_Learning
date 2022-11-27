from collections import defaultdict

import gym
import numpy as np
from hiive.mdptoolbox import example
from hiive.mdptoolbox.mdp import PolicyIteration, QLearning, ValueIteration

from src.config.config import (  # pylint: disable=import-error
    ALPHA,
    ALPHA_DECAY,
    EPSILON,
    EPSILON_DECAY,
    MAX_ITERATIONS,
)


class ForestManagementAgent:
    """
    Implementation of of the environment solving agent given the environment object
    Agent is capable of solving the environment using
    value iteration, policy iteration and Q learning
    """

    def __init__(self, gamma, problem_size):
        self.gamma = gamma
        self.value_iteration_stats = None
        self.policy_iteration_stats = None
        self.q_learning_stats = None
        self.init_transition_and_rewards(problem_size)

    def init_transition_and_rewards(self, problem_size):
        """Initalize transition matrix and rewards maxtrix"""
        self.T, self.R = example.forest(S=problem_size)

    def value_iteration(self):
        """
        Value iteration algorithm,
        Returns the policy, set the relevant statistics attribute
        """
        experiment = ValueIteration(
            self.T, self.R, gamma=self.gamma, epsilon=EPSILON, max_iter=MAX_ITERATIONS
        )
        # collect statistics
        self.value_iteration_stats, policy = self.get_algorithm_stats(experiment)
        return policy

    def policy_iteration(self):
        """
        Value iteration algorithm,
        Returns the policy, set the relevant statistics attribute
        """
        experiment = PolicyIteration(
            self.T,
            self.R,
            gamma=self.gamma,
            max_iter=MAX_ITERATIONS,
            eval_type="iterative",
        )
        # collect statistics
        self.policy_iteration_stats, policy = self.get_algorithm_stats(experiment)
        return policy

    def q_learning(self):
        """
        Q learning algorithm,
        Returns the policy, set the relevant statistics attribute
        """
        experiment = QLearning(
            self.T,
            self.R,
            gamma=self.gamma,
            alpha=ALPHA,
            alpha_decay=ALPHA_DECAY,
            epsilon_decay=EPSILON_DECAY,
            n_iter=MAX_ITERATIONS,
        )
        # collect statistics
        self.q_learning_stats, policy = self.get_algorithm_stats(experiment)
        return policy

    def get_algorithm_stats(self, experiment):
        """
        Get the statistics from experiment,
        returns the policy and relevant statistics
        """
        # run the experiment
        runs = experiment.run()
        # get time, number of iterations and max reward
        terminal_stats = runs[-1]
        # get policy
        policy = np.array(experiment.policy)
        # collect statistics
        data = defaultdict()
        data["gamma"] = self.gamma
        data["time"] = terminal_stats["Time"]
        data["iterations"] = terminal_stats["Iteration"]
        data["reward"] = terminal_stats["Max V"]
        return data, policy
