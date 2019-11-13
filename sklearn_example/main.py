import numpy as np
import matplotlib.pyplot as plt
import timeit
from itertools import cycle
import imageio
from io import BytesIO

def similarity(p1, p2):
    return -((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def make_gif(figures, filename, fps=10, **kwargs):
    images = []
    for fig in figures:
        output = BytesIO()
        fig.savefig(output)
        plt.close(fig)  
        output.seek(0)
        images.append(imageio.imread(output))
    imageio.mimsave(filename, images, fps=fps, **kwargs)

def main():
    damping = 0.5
    X = np.loadtxt("data.txt")
    #X = [[1, 1], [0.9, 1.1], [4, 4], [3.8, 3.9], [1.2, 1.1], [4.3, 3.9], [0.7, 0.9], [3.6, 4.4]]
    data_size = len(X)
    print("X: ", X)
    r = np.zeros([data_size, data_size])
    a = np.zeros([data_size, data_size])
    s = np.zeros([data_size, data_size])

    for i in range(0, data_size):
        for k in range(0, data_size):
            s[i][k] = similarity(X[i], X[k])
    #np.fill_diagonal(s, np.median(s))
    #np.fill_diagonal(s, np.amin(s))
    np.fill_diagonal(s, -50)

    figures = []

    for iter in range (100):
        
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


#         for i in range(0, data_size):
#             for k in range(0, data_size):
#                 sai = s[i] + a[i]
#                 sai[k] = -np.inf
#                 sai[i] = -np.inf
#                 r[i][k] = r[i][k] * damping + (1 - damping) * (s[i][k] - np.amax(sai))

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
#         for k in range(0, data_size):
#             temp = np.maximum(r[:,k], np.zeros(data_size))
#             temp[k] = 0
#             a[k][k] = a[k][k] * damping + (1 - damping) * (np.sum(temp))
#     
#         for i in range(0, data_size):
#             for k in range(0, data_size):
#                 if k == i:
#                     continue
#                 max_rk = np.maximum(r[:,k], np.zeros(data_size))
#                 max_rk[i] = 0
#                 max_rk[k] = 0
#                 a[i][k] = a[i][k] * damping + (1 - damping) * (min(0, np.sum(max_rk) + r[k][k]))


        t2 = timeit.default_timer()
        print("a timing[ms]: ", 1000*(t2 - t1))

        c = a + r

        #print("r: ", r)
        #print("a: ", a)
        #print("c: ", c)

#         np.savetxt("diag.txt", np.diagonal(c))
#         np.savetxt("max_rows.txt", np.amax(c, axis=1))

#         prototypes = np.argmax(c, axis=1)
#         out = {}
#         for i in range(data_size):
#             if prototypes[i] in out:
#                 out[prototypes[i]].append(i)
#             else:
#                 out[prototypes[i]] = [i]
        #print("out: ", out)
#         plt.clf()
#         ax = plt.gca()
#         for key in out:
#             whatever = np.take(X, out[key], axis=0)
#             color = next(ax._get_lines.prop_cycler)['color']
#             plt.plot(whatever[:,0], whatever[:,1], '.', color = color)
#             plt.plot(X[key][0], X[key][1], 'o', color = color)
        labels = np.argmax(c, axis=1)
        exemplars = np.unique(labels)
        colors = dict(zip(exemplars, cycle('bgrcmyk')))

        fig = plt.figure()
        plt.clf()
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
        figures.append(fig)


        #input()

    make_gif(figures, "test.gif")


if __name__ == "__main__":
    main()
    input()
