import gym


class Environment:
    """Generate environment given name"""

    def __init__(self, environment_name, render=False, gamma=1.0, n=1000):
        self.env = gym.make(environment_name)
        self.render = render
        self.gamma = gamma
        self.n = n

    def run_episode(self, policy):
        """Runs an episode and return the total reward"""
        obs = self.env.reset()
        total_reward = 0
        step_idx = 0
        while True:
            if self.render:
                self.env.render()
            # collect observations and rewards
            obs, reward, done, _ = self.env.step(int(policy[obs]))
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
        scores = [self.run_episode(policy) for _ in range(self.n)]
        return scores
