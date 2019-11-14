from generate_data import generate
from ap import AP
import numpy as np

def similarity(p1, p2):
    return -((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    generate()
    X = np.loadtxt("../data/data.txt")
    ap = AP(X, similarity)
    ap.categorize(show_iterations=False, outfilename="../output/result.png", outgifname="../output/result.gif")

if __name__ == "__main__":
    main()
    input()

