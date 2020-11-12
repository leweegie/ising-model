import math
import numpy as np
import scipy
from numpy.random import rand
import matplotlib.pyplot as plt
from algorithm import Algorithm

class Glauber(Algorithm):

    def flip(self):
        
        #   get coords of spin flip
        i = np.random.randint(self.n)
        j = np.random.randint(self.n)

        self.state[i,j] = -self.state[i,j]

        energy_diff = self.delta_e(self.state, i, j)

        if energy_diff > 0:
            check = self.probability(energy_diff)
            random = rand(1)
            if check < random:
                self.state[i,j] = -self.state[i,j]