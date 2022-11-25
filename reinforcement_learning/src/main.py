from collections import defaultdict

from src.agents.agent import Agent
from src.config.config import GAMMA_LIST
from src.environment.environment import Environment


def frozen_lake_analysis(env_args):
    """Analyze the frozen lake agent given the problem size"""
    # initialize list for collecting reward scores
    reward_map = defaultdict(lambda: defaultdict(list))
    for gamma in GAMMA_LIST:
        # create environment and agent
        environment = Environment(env_args=env_args, gamma=gamma)
        agent = Agent(environment)
        # use policy iteration from the agent to solve the Frozen Lake problem
        optimal_policy_from_policy_iteration = agent.policy_iteration()
        reward_map["policy_iteration"][gamma] = agent.policy_iteration_rewards
        # use value iteration
        optimal_policy_from_value_iteration = agent.value_iteration()
        reward_map["value_iteration"][gamma] = agent.value_iteration_rewards
        # use q learning
        optimal_policy_from_q_learning = agent.q_learning()
        reward_map["q_learning"][gamma] = agent.q_learning_rewards


if __name__ == "__main__":
    pass
