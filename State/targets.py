import numpy as np

def generate_random_targets(num_targets=3):
    """
    Generate random target centers in table coordinate space (0 to 1).
    Returns list of (x,y) coordinates.
    """
    targets = []

    # Restrict targets to the central 60% of the table 
    # Can't place objects on top of ArUco markers
    for _ in range(num_targets):
        x = np.random.uniform(0.2, 0.8)
        y = np.random.uniform(0.2, 0.8)
        targets.append((x, y))

    return targets