from collections import defaultdict

import pandas as pd
from matplotlib import pyplot as plt

from src.agents.forest_management import ForestManagementAgent
from src.agents.frozen_lake import FrozenLakeAgent
from src.config.config import (  # pylint: disable=no-name-in-module, import-error
    FROZEN_LAKE_MAPS,
    GAMMA_LIST,
)
from src.visualization.visualize import (
    visualize_forest,
    visualize_policy_frozen_lake,
    visualize_statistics_forest,
    visualize_statistics_frozen_lake,
)


def frozen_lake_analysis(env_args):
    """Analyze the frozen lake agent given the problem size"""
    # initialize list for collecting reward scores
    policy_map = defaultdict(list)
    statistics_map = defaultdict(list)
    # loop through the gamma
    for gamma in GAMMA_LIST:
        # create environment and agent
        print(f"Gamma: {gamma}")
        agent = FrozenLakeAgent(env_args, gamma)
        # use policy iteration
        optimal_policy_policy_iteration = agent.policy_iteration()
        policy_map["policy_iteration"].append(optimal_policy_policy_iteration)
        statistics_map["policy_iteration"].append(agent.policy_iteration_stats)
        # # use value iteration
        optimal_policy_value_iteration = agent.value_iteration()
        policy_map["value_iteration"].append(optimal_policy_value_iteration)
        statistics_map["value_iteration"].append(agent.value_iteration_stats)
        # use q learning
        optimal_policy_q_learning = agent.q_learning()
        policy_map["q_learning"].append(optimal_policy_q_learning)
        statistics_map["q_learning"].append(agent.q_learning_stats)

    # convert statistics map to dataframe
    stats_list = []
    for method, stats in statistics_map.items():
        stats = pd.DataFrame.from_dict(stats)
        stats["method"] = method
        stats_list.append(stats)
    statistics = pd.concat(stats_list)
    statistics.to_csv(f"./reports/stats/frozenLake_{env_args['map_name']}.csv")

    # visualize the optimal policy from policy iteration
    visualize_policy_frozen_lake(
        optimal_policy_policy_iteration,
        shape=tuple(map(int, env_args["map_name"].split("x"))),
        name=env_args["map_name"],
        title=f"Frozen Lake {env_args['map_name']} Policy Iteration",
    )
    plt.savefig(
        f"./reports/figures/frozenLake_{env_args['map_name']}_policy_iter.jpg", dpi=150
    )
    # visualize the optimal policy from value iteration
    visualize_policy_frozen_lake(
        optimal_policy_value_iteration,
        shape=tuple(map(int, env_args["map_name"].split("x"))),
        name=env_args["map_name"],
        title=f"Frozen Lake {env_args['map_name']} Value Iteration",
    )
    plt.savefig(
        f"./reports/figures/frozenLake_{env_args['map_name']}_value_iter.jpg", dpi=150
    )
    # visualize the optimal policy from q learning
    visualize_policy_frozen_lake(
        optimal_policy_q_learning,
        shape=tuple(map(int, env_args["map_name"].split("x"))),
        name=env_args["map_name"],
        title=f"Frozen Lake {env_args['map_name']} Q Learning",
    )

    plt.savefig(
        f"./reports/figures/frozenLake_{env_args['map_name']}_q_learning.jpg", dpi=150
    )

    visualize_statistics_frozen_lake(
        f"./reports/stats/frozenLake_{env_args['map_name']}.csv", "policy_iteration"
    )
    plt.suptitle("Frozen Lake Statistics - Policy Iteration")
    plt.savefig(
        f"./reports/figures/frozenLake_{env_args['map_name']}_policy_iter_anylze.jpg",
        dpi=150,
    )

    visualize_statistics_frozen_lake(
        f"./reports/stats/frozenLake_{env_args['map_name']}.csv", "value_iteration"
    )
    plt.suptitle("Frozen Lake Statistics - Value Iteration")
    plt.savefig(
        f"./reports/figures/frozenLake_{env_args['map_name']}_value_iter_anylze.jpg",
        dpi=150,
    )

    visualize_statistics_frozen_lake(
        f"./reports/stats/frozenLake_{env_args['map_name']}.csv", "q_learning"
    )
    plt.suptitle("Frozen Lake Statistics - Q Learning")
    plt.savefig(
        f"./reports/figures/frozenLake_{env_args['map_name']}_q_learning_anylze.jpg",
        dpi=150,
    )


def forest_management_analysis(problem_size):
    """Analyze the forest management agent given the problem size"""
    # initialize list for collecting reward scores
    policy_map = defaultdict(list)
    statistics_map = defaultdict(list)
    # loop through the gamma
    for gamma in GAMMA_LIST:
        # create environment and agent
        agent = ForestManagementAgent(gamma, problem_size)
        # use policy iteration
        optimal_policy_policy_iteration = agent.policy_iteration()
        policy_map["policy_iteration"].append(optimal_policy_policy_iteration)
        statistics_map["policy_iteration"].append(agent.policy_iteration_stats)
        # use value iteration
        optimal_policy_value_iteration = agent.value_iteration()
        policy_map["value_iteration"].append(optimal_policy_value_iteration)
        statistics_map["value_iteration"].append(agent.value_iteration_stats)
        # use q learning
        optimal_policy_q_learning = agent.q_learning()
        policy_map["q_learning"].append(optimal_policy_q_learning)
        statistics_map["q_learning"].append(agent.q_learning_stats)

    stats_list = []
    for method, stats in statistics_map.items():
        stats = pd.DataFrame.from_dict(stats)
        stats["method"] = method
        stats_list.append(stats)
    statistics = pd.concat(stats_list)
    statistics.to_csv(f"./reports/stats/forestManagement_{problem_size}.csv")

    visualize_forest(
        optimal_policy_policy_iteration,
        problem_size=problem_size,
        title=f"Forest Management ({problem_size}) Policy Iteration",
    )
    visualize_forest(
        optimal_policy_value_iteration,
        problem_size=problem_size,
        title=f"Forest Management ({problem_size}) Value Iteration",
    )
    visualize_forest(
        optimal_policy_q_learning,
        problem_size=problem_size,
        title=f"Forest Management ({problem_size}) Q Learning",
    )

    visualize_statistics_forest(
        f"./reports/stats/forestManagement_{problem_size}.csv", "policy_iteration"
    )
    plt.suptitle("Forest Management Statistics - Policy Iteration")
    plt.savefig(
        f"./reports/figures/forestManagement_{problem_size}_policy_iter_anylze.jpg",
        dpi=150,
    )

    visualize_statistics_forest(
        f"./reports/stats/forestManagement_{problem_size}.csv", "value_iteration"
    )
    plt.suptitle("Forest Management Statistics - Value Iteration")
    plt.savefig(
        f"./reports/figures/forestManagement_{problem_size}_value_iter_anylze.jpg",
        dpi=150,
    )

    visualize_statistics_forest(
        f"./reports/stats/forestManagement_{problem_size}.csv", "q_learning"
    )
    plt.suptitle("Forest Management Statistics - Q Learning")
    plt.savefig(
        f"./reports/figures/forestManagement_{problem_size}_q_learning_anylze.jpg",
        dpi=150,
    )


if __name__ == "__main__":
    # small frozen lake problem
    # frozen_lake_analysis(
    #     dict(
    #         id="FrozenLake-v1",
    #         desc=FROZEN_LAKE_MAPS["4x4"],
    #         map_name="4x4",
    #         is_slippery=True,
    #     )
    # )
    # big frozen lake problem
    # frozen_lake_analysis(
    #     dict(
    #         id="FrozenLake-v1",
    #         # desc=FROZEN_LAKE_MAPS["8x8"],
    #         map_name="8x8",
    #         is_slippery=False,
    #     )
    # )
    forest_management_analysis(problem_size=25)
    forest_management_analysis(problem_size=625)
