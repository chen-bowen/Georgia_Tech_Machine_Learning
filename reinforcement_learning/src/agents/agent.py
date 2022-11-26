import gym
import numpy as np
from hiive.mdptoolbox.mdp import PolicyIteration, QLearning, ValueIteration

from src.config.config import (  # pylint: disable=import-error，no-name-in-module
    ALPHA,
    MAX_ITERATIONS,
)


class FrozenLakeAgent:
    """
    Implementation of of the environment solving agent given the environment object
    Agent is capable of solving the environment using
    value iteration, policy iteration and Q learning
    """

    def __init__(self, environment_args, eps=1e-10):
        self.env = gym.make(**environment_args)
        self.eps = eps
        self._init_q_values()
        self._init_convergence_metrics()

    def _init_convergence_metrics(self):
        """Initialize storages for rewards vs iteration on different algorithms"""
        self.value_iteration_rewards = []
        self.policy_iteration_rewards = []
        self.q_learning_rewards = []

    def _init_q_values(self):
        """Initialize Q values based on the environment"""
        num_actions = self.env.nA
        num_states = self.env.nS
        self.q_value = (
            np.zeros(num_states * num_actions)
            .reshape(num_states, num_actions)
            .astype(np.float32)
        )

    def policy_iteration(self):
        """Policy-Iteration algorithm"""
        # initialize a random policy
        policy = np.random.choice(self.env.nA, size=(self.env.nS))

        # iterate until max iterations
        for i in range(MAX_ITERATIONS):
            # get old policy value for all states
            old_policy_value = self.get_policy_value_function(policy)
            # given the old value functions, obtain the optimal policy
            new_policy = self.get_optimal_policy(old_policy_value)
            # extract the policy and evaluate the rewards
            self.policy_iteration_rewards.append(self.env.evaluate_policy(policy))
            # if converged, stop
            if np.all(policy == new_policy):
                print(f"Policy-Iteration converged at step #{i + 1}.")
                self.policy_iteration_rewards.append(self.env.evaluate_policy(policy))
                return policy

            policy = new_policy
        print(f"Policy-Iteration never converged at iteration # {i + 1}.")
        return policy

    def value_iteration(self):
        """Value-iteration algorithm"""
        # initialize value-function
        v = np.zeros(self.env.nS)
        for i in range(MAX_ITERATIONS):
            # store the previous value
            prev_v = np.copy(v)
            # loop through all states
            for s in range(self.env.nS):
                # get the value all taking each action at a certain state
                q_sa = [
                    sum([p * (r + prev_v[s_]) for p, s_, r, _ in self.env.P[s][a]])
                    for a in range(self.env.nA)
                ]
                # set the state value to be the maximum of q values
                v[s] = max(q_sa)
            # extract the policy and evaluate the rewards
            if i % 50 == 0:
                policy = self.get_optimal_policy(v)
                self.value_iteration_rewards.append(self.env.evaluate_policy(policy))

            if np.sum(np.fabs(prev_v - v)) <= self.eps:
                print(f"Value-Iteration converged at iteration # {i + 1}.")
                policy = self.get_optimal_policy(v)
                self.value_iteration_rewards.append(self.env.evaluate_policy(policy))
                return policy

        print(f"Value-Iteration never converged at iteration # {i + 1}.")
        policy = self.get_optimal_policy(v)
        return policy

    def update_q_values(self):
        # one episode learning
        state = self.env.environment_obj.reset()[0]
        # self.env.render()

        for _ in range(MAX_ITERATIONS):
            act = self.env.environment_obj.action_space.sample()  # random
            next_state, reward, done, _, _ = self.env.environment_obj.step(act)
            self.q_learning_rewards.append(reward)
            q_next_max = np.max(self.q_value[next_state])
            # Q <- Q + a(Q' - Q)
            # <=> Q <- (1-a)Q + a(Q')
            self.q_value[state][act] = (1 - ALPHA) * self.q_value[state][
                act
            ] + ALPHA * (reward + self.env.gamma * q_next_max)
            # self.env.render()
            if done:
                self.q_learning_rewards.append(reward)
                return
            else:
                state = next_state

    def q_learning(self):
        # update q values
        for i in range(MAX_ITERATIONS):
            self.update_q_values()
        # reset the environment, use q values to make decisions
        state = self.env.environment_obj.reset()[0]
        for j in range(MAX_ITERATIONS):
            act = np.argmax(self.q_value[state])
            next_state, reward, done, _, _ = self.env.environment_obj.step(act)
            self.q_learning_rewards.append(reward)
            # environment solved
            if done:
                print(f"Q learning converged at step # {i + j + 1}.")
                self.q_learning_rewards.append(reward)
                # get optimal policy
                policy = self.q_value.argmax(axis=1)
                return policy
            else:
                state = next_state
        print(f"Q learning never converged at step # {i + j + 1}.")
        policy = self.q_value.argmax(axis=1)
        return policy

    def get_optimal_policy(self, value_function):
        """Get the optimal policy given a value-function"""
        policy = np.zeros(self.env.nS)
        # loop through all states in the environment
        for s in range(self.env.nS):
            # initialize the q value for taking action a at state s
            q_sa = np.zeros(self.env.nA)
            # loop through all actions in the action spaces
            for a in range(self.env.nA):
                # (probability, next state, reward, done)
                for p, next_state, reward, _ in self.env.P[s][a]:
                    # calculate the sum of discounted reward
                    q_sa[a] += p * (
                        reward + self.env.gamma * value_function[next_state]
                    )
            # get the optimal policy from the state action pair that resulted in the largest Q value
            policy[s] = np.argmax(q_sa)
        return policy

    def get_policy_value_function(self, policy):
        """Iteratively evaluate the value-function under policy"""
        # initialize the value function
        v = np.zeros(self.env.nS)
        while True:
            # store the previous value function
            prev_v = np.copy(v)
            # loop through all the states
            for s in range(self.env.nS):
                # get action from state
                policy_action = policy[s]
                # get weighted sum for the cumulative rewards
                v[s] = sum(
                    [
                        p * (r + self.env.gamma * prev_v[s_])
                        for p, s_, r, _ in self.env.P[s][policy_action]
                    ]
                )
            # stop when value converged
            if np.sum((np.fabs(prev_v - v))) <= self.eps:
                break
        return v
