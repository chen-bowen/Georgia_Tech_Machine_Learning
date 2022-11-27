from collections import defaultdict

import gym
import numpy as np
from hiive.mdptoolbox.mdp import PolicyIteration, QLearning, ValueIteration

from src.config.config import (  # pylint: disable=import-error
    ALPHA,
    ALPHA_DECAY,
    EPSILON,
    EPSILON_DECAY,
    MAX_ITERATIONS,
)


class FrozenLakeAgent:
    """
    Implementation of of the environment solving agent given the environment object
    Agent is capable of solving the environment using
    value iteration, policy iteration and Q learning
    """

    def __init__(self, environment_args, gamma):
        self.env = gym.make(**environment_args)
        self.gamma = gamma
        self.env_size = list(map(int, environment_args["map_name"].split("x")))
        self.value_iteration_stats = None
        self.policy_iteration_stats = None
        self.q_learning_stats = None
        self.init_transition_and_rewards()

    def init_transition_and_rewards(self):
        """Initalize transition matrix and rewards maxtrix"""
        self.T = np.zeros(
            (self.env.action_space.n, np.prod(self.env_size), np.prod(self.env_size))
        )
        self.R = np.zeros(
            (self.env.action_space.n, np.prod(self.env_size), np.prod(self.env_size))
        )
        old_state = np.inf
        # fill in the transition matrix and rewards matrix
        for square in self.env.P:
            for action in self.env.P[square]:
                for i in range(len(self.env.P[square][action])):
                    new_state = self.env.P[square][action][i][1]
                    if new_state == old_state:
                        self.T[action][square][self.env.P[square][action][i][1]] = (
                            self.T[action][square][old_state]
                            + self.env.P[square][action][i][0]
                        )
                        self.R[action][square][self.env.P[square][action][i][1]] = (
                            self.R[action][square][old_state]
                            + self.env.P[square][action][i][2]
                        )
                    else:
                        self.T[action][square][
                            self.env.P[square][action][i][1]
                        ] = self.env.P[square][action][i][0]
                        self.R[action][square][
                            self.env.P[square][action][i][1]
                        ] = self.env.P[square][action][i][2]
                    old_state = self.env.P[square][action][i][1]

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
        (
            self.value_iteration_stats["avg_steps"],
            self.value_iteration_stats["success_rate"],
        ) = self.get_score(policy, print_info=True)
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
        )  # eval_type="iterative"
        # collect statistics
        self.policy_iteration_stats, policy = self.get_algorithm_stats(experiment)
        (
            self.policy_iteration_stats["avg_steps"],
            self.policy_iteration_stats["success_rate"],
        ) = self.get_score(policy, print_info=True)
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
        )  # eval_type="iterative"
        # collect statistics
        self.q_learning_stats, policy = self.get_algorithm_stats(experiment)
        (
            self.q_learning_stats["avg_steps"],
            self.q_learning_stats["success_rate"],
        ) = self.get_score(policy, print_info=True)
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

    def get_score(self, policy, print_info=False, episodes=1000):
        """
        Get the score of the policy.

        https://medium.com/analytics-vidhya/solving-the-frozenlake-environment-from-openai-gym-using-value-iteration-5a078dffe438

        :param env: (Gym Environment) The environment
        :param policy: (Gym Policy) The policy
        :param print_info: (bool) Whether to print the information
        """
        misses = 0
        steps_list = []
        for _ in range(episodes):
            observation = self.env.reset()[0]
            steps = 0
            while True:
                # choose action
                action = policy[observation]
                # take the step according to the action
                observation, reward, done, _, _ = self.env.step(action)
                steps += 1
                if done and reward == 1:  # success
                    steps_list.append(steps)
                    break
                elif done and reward == 0:  # fail
                    misses += 1
                    break
        ave_steps = np.mean(steps_list)
        pct_success = 1 - (misses / episodes)

        if print_info:
            print("----------------------------------------------")
            print(
                f"You took an average of {round(ave_steps, 2)} steps to get the frisbee"
            )
            print(
                f"And you fell in the hole {round(1 - pct_success, 2)*100} % of the times"
            )
            print("----------------------------------------------")

        return ave_steps, pct_success
