class DataGeneratorCfg:
    n_samples = 300
    centers = [[-1, 0.5], [1, 0], [1,1]]
    cluster_std = 0.5
    random_state = 0

class APCfg:
    n_iterations = 100
    damping = 0.5
    preference = -50 #'MEDIAN', 'MINIMUM' or a value

