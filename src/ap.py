import numpy as np
from config import APCfg
from fgconverter import FGConverter
import matplotlib.pyplot as plt
from itertools import cycle

class AP:
    def __init__(self, input_data, f_similarity): 
        self.X = input_data
        self.similarity = f_similarity
        self.data_size = len(input_data)
        
        self.S = np.zeros([self.data_size, self.data_size])
        self.A = np.zeros([self.data_size, self.data_size])
        self.R = np.zeros([self.data_size, self.data_size])
        
        self._calculate_S()

        try:
            damping = float(APCfg.damping)
            if damping < 0.5 or damping > 1:
                print("APCfg.damping value too high (", damping, "), setting to 0.5")
                damping = 0.5
        except ValueError:
            print("Couldn't parse APCfg.damping as float, treats as 0.5!")
            damping = 0.5

        self.damping = damping

        self.labels = []
        self.exemplars = []
        self._change_count = 0
        self.fgc = FGConverter()

    
    def categorize(self, show_iterations=True, outfilename=None, outgifname=None):
        for iteration in range(APCfg.n_iterations):
            print("Iteration ", iteration)
            self._update_R()
            self._update_A()
            self._update_clusters()
            if outgifname == None:
                self._display(show = show_iterations)
            else:
                self._display(show = show_iterations, make_gif = True)


            if self._change_count > 10:
                print("_change_count > 10, breaking loop")
                break

        if outfilename != None:
            self._display(save_to_file = True, filename = outfilename)

        if outgifname != None:
            self.fgc.make_gif(outgifname)

#         fig = plt.figure(0)
#         fig.clf()
#         colors = dict(zip(self.exemplars, cycle('bgrcmyk')))
# 
#         for i in range(len(self.labels)):
#             x = self.X[i][0]
#             y = self.X[i][1]
# 
#             if i in self.exemplars:
#                 exemplar = i
#                 edge = 'k'
#                 ms = 10
#             else:
#                 exemplar = self.labels[i]
#                 ms = 3
#                 edge = None
#                 plt.plot([x, self.X[exemplar][0]], [y, self.X[exemplar][1]], color=colors[exemplar])
#             plt.plot(x, y, 'o', markersize=ms, markeredgecolor=edge, color=colors[exemplar])
#         plt.show()




    def _calculate_S(self):
        for i in range(self.data_size):
            for k in range(self.data_size):
                self.S[i][k] = self.similarity(self.X[i], self.X[k])

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

        np.fill_diagonal(self.S, preference)

    def _update_R(self):
        SA = self.S + self.A
        rows_indices = np.arange(self.data_size)

        # We need to find maximum value for every row excluding the diagonal elements
        np.fill_diagonal(SA, -np.inf)

        indices_max = np.argmax(SA, axis=1)
        values_max = SA[rows_indices, indices_max]

        # The found max values work for every element except where they were found, so we need to find for them the second maximum value
        SA[rows_indices, indices_max] = -np.inf
        values_max2 = SA[rows_indices, np.argmax(SA, axis=1)]

        # Now we create SA with max values, as the algorithm says
        SA_max = np.zeros_like(self.R) + values_max[:, None]
        SA_max[rows_indices, indices_max] = values_max2

        self.R = self.R * self.damping + (1 - self.damping) * (self.S - SA_max)

    def _update_A(self):
        indices = np.arange(self.data_size)

        A_next = np.array(self.R)
        A_next[A_next < 0] = 0
        np.fill_diagonal(A_next, 0)
        # We should also exclude i index for itself, but now it would be hard to vectorize, so we ignore it for now, and take it into account later

        # Sum columnwise
        A_next = A_next.sum(axis=0)
        A_next = A_next + self.R[indices, indices]
        
        # Reshaping to square matrix
        A_next = np.ones(self.A.shape) * A_next

        # Subtracting the elements which we didn't ignore before
        A_next -= np.clip(self.R, 0, np.inf)
        A_next[A_next > 0] = 0

        # Setting the diagonal of A, which is slightly different
        A_diag = np.array(self.R)
        np.fill_diagonal(A_diag, 0)
        A_diag[A_diag < 0] = 0

        # Merging diagonal of A into A
        A_next[indices, indices] = A_diag.sum(axis=0)

        self.A = self.A * self.damping + (1 - self.damping) * A_next

    def _update_clusters(self):
        C = self.A + self.R
        labels = np.argmax(C, axis=1)
        exemplars = np.unique(labels)

        if np.array_equal(labels, self.labels) and np.array_equal(np.sort(exemplars,axis=0), np.sort(self.exemplars,axis=0)):
            self._change_count += 1
        else:
            self._change_count = 0

        self.labels = labels
        self.exemplars = exemplars

    def _display(self, figure_number = 0, show = True, save_to_file = False, filename = "outfile.png", make_gif = False):
        fig = plt.figure(figure_number)
        fig.clf()
        colors = dict(zip(self.exemplars, cycle('bgrcmyk')))

        for i in range(len(self.labels)):
            x = self.X[i][0]
            y = self.X[i][1]

            if i in self.exemplars:
                exemplar = i
                edge = 'k'
                ms = 10
            else:
                exemplar = self.labels[i]
                ms = 3
                edge = None
                plt.plot([x, self.X[exemplar][0]], [y, self.X[exemplar][1]], color=colors[exemplar])
            plt.plot(x, y, 'o', markersize=ms, markeredgecolor=edge, color=colors[exemplar])
        if show:
            plt.ion()
            plt.show()
            plt.pause(0.0001)
        if save_to_file:
            plt.savefig(filename)
        if make_gif:
            self.fgc.add_fig(fig, False)
        
