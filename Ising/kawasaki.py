import math
import numpy as np
import scipy
from numpy.random import rand
import matplotlib.pyplot as plt
from algorithm import Algorithm

class Kawasaki(Algorithm):

    def flip(self):

        while True:
            #   get coords of spin flips
            i = np.random.randint(self.n)
            j = np.random.randint(self.n)
            x = np.random.randint(self.n)
            y = np.random.randint(self.n) 

            if (i != x) and (j != y):
                if self.state[i,j] != self.state[x,y]:

                    self.state[i,j] = - self.state[i,j]
                    self.state[x,y] = - self.state[x,y]
                    break
        
        #   calculate delta E
        energy_diff_1 = self.delta_e(self.state, i, j)
        energy_diff_2 = self.delta_e(self.state, x, y)

        energy_diff = energy_diff_1 + energy_diff_2

        #   check for nearest neighbours and adjust delta E
        xdiff = abs(x-i) % self.n
        ydiff = abs(y-j) % self.n

        if ((i == x) and (ydiff == 1)) or ((j == y) and (xdiff ==1)):
            energy_diff = energy_diff - 4

        if energy_diff > 0:
            check = self.probability(energy_diff)
            random = rand(1)
            if check < random:
                self.state[i,j] = -self.state[i,j]
                self.state[x,y] = -self.state[x,y]