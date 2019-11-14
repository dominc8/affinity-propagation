import numpy as np
from config import APCfg

class AP:
    def __init__(self, input_data, f_similarity): 
        self.X = input_data
        self.similarity = f_similarity
        self.data_size = len(input_data)
        
        self.S = np.zeros([data_size, data_size])
        self.A = np.zeros([data_size, data_size])
        self.R = np.zeros([data_size, data_size])
        
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


