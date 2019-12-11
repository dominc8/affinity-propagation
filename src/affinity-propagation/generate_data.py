from config import DataGeneratorCfg
from sklearn.datasets.samples_generator import make_blobs
import numpy as np

def generate():
    data, true_labels = make_blobs(n_samples=DataGeneratorCfg.n_samples, centers=DataGeneratorCfg.centers, cluster_std=DataGeneratorCfg.cluster_std, random_state=DataGeneratorCfg.random_state)
    print("Generating new data!")
    np.savetxt("data/data.txt", data)
    np.savetxt("data/true_labels.txt", true_labels)
    return data

