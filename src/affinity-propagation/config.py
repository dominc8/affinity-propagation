class DataGeneratorCfg:
    n_samples = 300
    centers = [[-1, 0.5], [1, 0], [1,1]]
    cluster_std = 0.5
    random_state = None 

class APCfg:
    n_iterations = 300
    damping = 0.8
    preference = -50 #'MEDIAN', 'MINIMUM' or a value

class MainCfg:
    generate_new_data=True              # Saves it to data folder
    show_iterations=False               # Showing every iteration on figure
    outfilename="output/result.png"     # None for not saving
    outgifname="output/result.gif"      # None for not saving
