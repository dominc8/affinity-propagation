from generate_data import generate
from config import APCfg
from fgconverter import FGConverter
import numpy as np
import matplotlib.pyplot as plt
import timeit
from itertools import cycle
import imageio
from io import BytesIO

def similarity(p1, p2):
    return -((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    damping = APCfg.damping
    generate()
    X = np.loadtxt("../data/data.txt")
    data_size = len(X)
    print("X: ", X)
    r = np.zeros([data_size, data_size])
    a = np.zeros([data_size, data_size])
    s = np.zeros([data_size, data_size])

    for i in range(0, data_size):
        for k in range(0, data_size):
            s[i][k] = similarity(X[i], X[k])

    if APCfg.preference == 'MEDIAN':
        preference = np.median(s)
    elif APCfg.preference == 'MINIMUM':
        preference = np.amin(s)
    else:
        try:
            preference = float(APCfg.preference)
        except ValueError:
            print("Couldn't parse APCfg.preference as float, treats as MEDIAN!")
            preference = np.median(s)
    
    np.fill_diagonal(s, preference)

    fgc = FGConverter()

    for iter in range (APCfg.n_iterations):
        
        print("iteration ", iter)
        
        t1 = timeit.default_timer()

        sa = s + a
        rows = np.arange(data_size)
        np.fill_diagonal(sa, -np.inf)

        idx_max = np.argmax(sa, axis=1)
        first_max = sa[rows, idx_max]

        sa[rows, idx_max] = -np.inf
        second_max = sa[rows, np.argmax(sa, axis=1)]

        max_sa = np.zeros_like(r) + first_max[:, None]
        max_sa[rows, idx_max] = second_max

        r = r * damping + (1 - damping) * (s - max_sa)


        t2 = timeit.default_timer()
        print("r timing[ms]: ", 1000*(t2 - t1))

        t1 = timeit.default_timer()

        k_k_idx = np.arange(data_size)
        # set a(i, k)
        a_temp = np.array(r)
        a_temp[a_temp < 0] = 0
        np.fill_diagonal(a_temp, 0)
        a_temp = a_temp.sum(axis=0) # columnwise sum
        a_temp = a_temp + r[k_k_idx, k_k_idx]

        # broadcasting of columns 'r(k, k) + sum(max(0, r(i', k))) to rows.
        a_temp = np.ones(a.shape) * a_temp

        # For every column k, subtract the positive value of k. 
        # This value is included in the sum and shouldn't be
        a_temp -= np.clip(r, 0, np.inf)
        a_temp[a_temp > 0] = 0

        # set(a(k, k))
        w = np.array(r)
        np.fill_diagonal(w, 0)

        w[w < 0] = 0

        a_temp[k_k_idx, k_k_idx] = w.sum(axis=0) # column wise sum
        a = a * damping + (1 - damping) * a_temp

        t2 = timeit.default_timer()
        print("a timing[ms]: ", 1000*(t2 - t1))

        c = a + r

        labels = np.argmax(c, axis=1)
        exemplars = np.unique(labels)
        colors = dict(zip(exemplars, cycle('bgrcmyk')))

        fig = plt.figure(0)
        fig.clf()
        for i in range(len(labels)):
            x_t = X[i][0]
            y_t = X[i][1]

            if i in exemplars:
                exemplar = i
                edge = 'k'
                ms = 10
            else:
                exemplar = labels[i]
                ms = 3
                edge = None
                plt.plot([x_t, X[exemplar][0]], [y_t, X[exemplar][1]], color=colors[exemplar])
            plt.plot(x_t, y_t, 'o', markersize=ms, markeredgecolor=edge, color=colors[exemplar])

        plt.ion()
        plt.show()
        plt.pause(0.0001)

        fgc.add_fig(fig, False)
        #figures.append(fig)

    fgc.make_gif("../data/fgc.gif")

#     make_gif(figures, "test.gif")


if __name__ == "__main__":
    main()
    input()

