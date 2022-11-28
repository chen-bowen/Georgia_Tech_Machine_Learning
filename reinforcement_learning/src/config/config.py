RANDOM_SEED = 100
MAX_ITERATIONS = 1e5
# Q learning params
ALPHA = 0.2  # learning rate
EPSILON = 1e-5
ALPHA_DECAY = 0.999
EPSILON_DECAY = 0.9
GAMMA_LIST = [0.1, 0.2, 0.3, 0.4, 0.6, 0.9, 1.0]
LEARNING_LIMIT = 1000
FROZEN_LAKE_MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}
TERM_STATE_MAP = {
    "4x4": [i for i, ltr in enumerate("".join(FROZEN_LAKE_MAPS["4x4"])) if ltr == "H"],
    "8x8": [i for i, ltr in enumerate("".join(FROZEN_LAKE_MAPS["8x8"])) if ltr == "H"],
}
GOAL_STATE_MAP = {"4x4": [15], "8x8": [63], "50x50": [2499]}
CMAP = "cool"
