from collections import defaultdict

import numpy as np


def get_score(env, policy, print_info=False, episodes=1000):
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
        observation = env.reset()[0]
        steps = 0
        while True:
            # choose action
            action = policy[observation]
            # take the step according to the action
            observation, reward, done, _, _ = env.step(action)
            steps += 1
            if done and reward == 1:  # success
                steps_list.append(steps)
                break
            elif done and reward == 0:  # fail
                misses += 1
                break
    ave_steps = np.mean(steps_list)
    std_steps = np.std(steps_list)
    pct_fail = (misses / episodes) * 100

    if print_info:
        print("----------------------------------------------")
        print(f"You took an average of {ave_steps} steps to get the frisbee")
        print(f"And you fell in the hole {pct_fail} % of the times")
        print("----------------------------------------------")

    return ave_steps, std_steps, pct_fail
