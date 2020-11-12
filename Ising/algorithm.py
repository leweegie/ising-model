import math
import numpy as np
import scipy
from numpy.random import rand
import matplotlib.pyplot as plt

class Algorithm(object):

    def __init__(self, n, T):
        self.n = n
        self.state = 2 * np.random.randint(2, size = (n, n)) - 1
        self.T = T
        self.J = 1.0
        self.kb = 1.0

    def delta_e(self, lattice, i, j):
        Si      = lattice[i,j]
        top     = lattice[i, (j-1)%self.n]
        bottom  = lattice[i, (j+1)%self.n]
        left    = lattice[(i-1)%self.n, j]
        right   = lattice[(i+1)%self.n, j]

        e = - 2 * self.J * Si * (top + bottom + left + right)
        return e

    #probability of flip
    def probability(self, delta_e):
        p = math.exp((-delta_e) / (self.kb * self.T))
        return p

    def magnetisation(self):
        m = abs(np.sum(self.state))
        return m

    def total_energy(self):
        e = 0
        for i in range(self.n):
            for j in range(self.n):
                left    = self.state[(i-1)%self.n, j]
                top     = self.state[i, (j-1)%self.n]
                Si      = self.state[i, j]
                e       += Si * top
                e       += Si * left

        e = - e * self.J
        return e