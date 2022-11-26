import time
from collections import defaultdict

from matplotlib import pyplot as plt

from src.agents.agent import Agent
from src.config.config import (  # pylint: disable=no-name-in-module
    FROZEN_LAKE_MAPS,
    GAMMA_LIST,
)
from src.environment.environment import Environment
from src.visualization.visualize import visualize_policy


def frozen_lake_analysis(env_args):
    """Analyze the frozen lake agent given the problem size"""
    # initialize list for collecting reward scores
    reward_map = defaultdict(lambda: defaultdict(list))
    time_map = defaultdict(lambda: defaultdict(list))

    for gamma in GAMMA_LIST:
        # create environment and agent
        print(f"Gamma: {gamma}")
        environment = Environment(env_args=env_args, gamma=gamma)
        agent = Agent(environment)
        agent.env.environment_obj.reset()
        # use policy iteration
        start_time = time.perf_counter()
        optimal_policy_policy_iteration = agent.policy_iteration()
        end_time = time.perf_counter()
        reward_map["policy_iteration"][gamma] = agent.policy_iteration_rewards
        time_map["policy_iteration"][gamma] = end_time - start_time
        # use value iteration
        start_time = time.perf_counter()
        optimal_policy_value_iteration = agent.value_iteration()
        end_time = time.perf_counter()
        reward_map["value_iteration"][gamma] = agent.value_iteration_rewards
        time_map["value_iteration"][gamma] = end_time - start_time
        # use q learning
        start_time = time.perf_counter()
        optimal_policy_q_learning = agent.q_learning()
        end_time = time.perf_counter()
        reward_map["q_learning"][gamma] = agent.q_learning_rewards
        time_map["q_learning"][gamma] = end_time - start_time

    # visualize the optimal policy from policy iteration
    visualize_policy(
        optimal_policy_policy_iteration,
        shape=tuple(map(int, env_args["map_name"].split("x"))),
        name=env_args["map_name"],
        title=f"Frozen Lake {env_args['map_name']} Policy Iteration",
    )
    plt.savefig(
        f"./reports/figures/frozenLake_{env_args['map_name']}_policy_iter.jpg", dpi=150
    )
    # visualize the optimal policy from value iteration
    visualize_policy(
        optimal_policy_value_iteration,
        shape=tuple(map(int, env_args["map_name"].split("x"))),
        name=env_args["map_name"],
        title=f"Frozen Lake {env_args['map_name']} Value Iteration",
    )
    plt.savefig(
        f"./reports/figures/frozenLake_{env_args['map_name']}_value_iter.jpg", dpi=150
    )
    # visualize the optimal policy from q learning
    visualize_policy(
        optimal_policy_q_learning,
        shape=tuple(map(int, env_args["map_name"].split("x"))),
        name=env_args["map_name"],
        title=f"Frozen Lake {env_args['map_name']} Q Learning",
    )

    plt.savefig(
        f"./reports/figures/frozenLake_{env_args['map_name']}_q_learning.jpg", dpi=150
    )
    breakpoint()


if __name__ == "__main__":
    # small frozen lake problem
    frozen_lake_analysis(
        dict(
            id="FrozenLake-v1",
            desc=FROZEN_LAKE_MAPS["4x4"],
            map_name="4x4",
            is_slippery=True,
        )
    )
    # big frozen lake problem
    # frozen_lake_analysis(
    #     dict(
    #         id="FrozenLake-v1",
    #         desc=FROZEN_LAKE_MAPS["8x8"],
    #         map_name="8x8",
    #         is_slippery=False,
    #     )
    # )
