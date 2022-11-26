import gym
import numpy as np

from src.config.config import MAX_ITERATIONS


class Environment:
    """Generate environment given name"""

    def __init__(self, env_args, render=False, gamma=1.0, n=100):
        self.environment_obj = gym.make(**env_args)
        self.render = render
        self.gamma = gamma
        self.n = n
        self.get_environment_attributes()

    def get_environment_attributes(self):
        self.nS = self.environment_obj.observation_space.n
        self.nA = self.environment_obj.action_space.n
        self.P = self.environment_obj.P

    def run_episode(self, policy):
        """Runs an episode and return the total reward"""
        obs = self.environment_obj.reset()[0]
        total_reward = 0
        step_idx = 0
        while True and step_idx < MAX_ITERATIONS:
            # print(f"step: {step_idx}")
            if self.render:
                self.environment_obj.render()
            # collect observations and rewards
            obs, reward, done, _, _ = self.environment_obj.step(int(policy[obs]))
            # calculate the discounted total reward
            total_reward += self.gamma**step_idx * reward
            # increment step
            step_idx += 1
            if done:
                break
        return total_reward

    def evaluate_policy(self, policy):
        """
        execute the entire policy and generate a list of scores
        accumulated by executing the policy
        """
        scores = np.mean([self.run_episode(policy) for _ in range(self.n)])
        return scores
