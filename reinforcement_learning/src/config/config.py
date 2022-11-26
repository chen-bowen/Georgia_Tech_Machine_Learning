RANDOM_SEED = 100
MAX_ITERATIONS = 10000
# Q learning params
ALPHA = 0.2  # learning rate
GAMMA_LIST = [1.0, 0.5, 0.2, 0.1]
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
    "4x4": [5, 7, 11, 12],
    "8x8": [19, 29, 35, 41, 42, 46, 49, 52, 54, 59],
}
GOAL_STATE_MAP = {"4x4": [15], "8x8": [63], "50x50": [2499]}
CMAP = "cool"
