import math
import numpy as np
import scipy
from numpy.random import rand
import matplotlib.pyplot as plt
from algorithm import Algorithm
from glauber import Glauber
from kawasaki import Kawasaki
from astropy.stats import jackknife_resampling
from astropy.stats import jackknife_stats
import sys

#susceptibility function
def susceptibility(variance, N, kB, T):
    X = (1/(N * kB * T)) * (variance)
    return X

#heat capacity function
def heat_capacity(variance, N, kB, T):
    C = ( 1 / ( N* kB * (T*T) )) * (variance)
    return C

#variance function
def variance(data):
    mean = np.mean(data)
    sq = np.mean(data ** 2)
    meansq = mean ** 2
    v = sq - meansq
    return v

#function for jackknife errors
def jackknife(data, N, kb, T, method):
    resamples = jackknife_resampling(data)
    x_resamples = [method(variance(resamples[i]), N, kb, T) for i in range (len(resamples))]
    v = variance(data)
    c = method(v, N, kb, T)
    return np.sqrt(np.sum([(x_resamples[i] - c) * (x_resamples[i]-c) for i in range (len(x_resamples))]))

def plot_graphs_glauber(n, nsweeps):
    
    #create arrays
    av_mags = []
    av_energies = []
    heat_capacities = []
    susceptibilities = []
    c_errors = []
    s_errors = []

    #create list of temps and create glauber lattice
    temps = np.linspace(1,3,20)
    g = Glauber(n, temps[0])
    g.state = np.ones((n,n))    #set all spins to up
    for x in range(len(temps)):
        
        g.T = temps[x]
        mags = []
        energies = []
        for i in range (n*n*nsweeps):
            g.flip()
            if (i%(10*n*n) ==0):
                if (i > 200*n*n):
                    mags.append(g.magnetisation())
                    energies.append(g.total_energy())

        #variance of mags and energies
        m_v = variance(np.asarray(mags))
        e_v = variance(np.asarray(energies))
        N = n * n

        #calculate susceptibility, heat capacity, and errors using jackknife function
        s = susceptibility(m_v, N, g.kb, temps[x])
        c = heat_capacity(e_v, N, g.kb, temps[x])
        c_e = jackknife(np.array(energies), N, g.kb, temps[x], heat_capacity)
        
        #append values to lists
        susceptibilities.append(s)
        heat_capacities.append(c)
        c_errors.append(c_e)
        average_m = np.sum(mags)/len(mags)
        average_e = np.sum(energies)/len(energies)
        av_mags.append(average_m)
        av_energies.append(average_e)

    np.savetxt('magnetisation_g.dat', np.column_stack([temps, av_mags]))
    np.savetxt('susceptibility_g.dat', np.column_stack([temps, susceptibilities]))
    np.savetxt('energy_g.dat', np.column_stack([temps, av_energies]))
    np.savetxt('heatcapacity_g.dat', np.column_stack([temps, heat_capacities, c_errors]))

def plot_graphs_kawasaki(n, nsweeps):
    
    #create arrays
    av_energies = []
    heat_capacities = []
    c_errors = []

    #create lists of temps and create kawasaki lattice
    temps = np.linspace(1,3,20)
    g = Kawasaki(n, temps[0])
    #set lattice to half up, half down
    g.state = np.zeros([n,n])
    a = int(n/2)
    g.state[:,:a]+=1
    g.state[:,a:]-=1

    for x in range(len(temps)):
        
        g.T = temps[x]

        energies = []
        for i in range (n*n*nsweeps):
            g.flip()
            if (i%(10*n*n) ==0):
                if (i > 200* n * n):
                    energies.append(g.total_energy())

        #variance of energy
        e_v = variance(np.asarray(energies))

        N = n * n

        #calculate heat capacity and errors using jackknife function
        c = heat_capacity(e_v, N, g.kb, g.T)
        c_e = jackknife(np.array(energies), N, g.kb, g.T, heat_capacity)

        #append to lists
        heat_capacities.append(c)
        c_errors.append(c_e)
        average_e = np.sum(energies)/len(energies)
        av_energies.append(average_e)
        print(x)

    np.savetxt('energy_k.dat', np.column_stack([temps, av_energies]))
    np.savetxt('heatcapacity_k.dat', np.column_stack([temps, heat_capacities, c_errors]))

#funtion to animate lattice of spins
def animation(n, nsweeps, temp):

    method = input("Glauber = 1 or Kawasaki = 2\n")

    if method == 1:
        g = Glauber(n, temp)
    if method == 2:
        g = Kawasaki(n, temp)

    fig = plt.figure()
    im=plt.imshow(g.state, animated=True)

    for i in range (n * n * nsweeps):

        g.flip() #perform flip
        if (i%(10*n*n) == 0):
            plt.cla()
            im=plt.imshow(g.state, animated=True, vmin = -1, vmax = 1)
            plt.draw()
            plt.pause(0.0001)

def plot():

    
    te_k    = np.loadtxt('energy_k.dat')[:,0]
    e_k     = np.loadtxt('energy_k.dat')[:,1]
    tc_k    = np.loadtxt('heatcapacity_k.dat')[:,0]
    c_k     = np.loadtxt('heatcapacity_k.dat')[:,1]
    c_k_e   = np.loadtxt('heatcapacity_k.dat')[:,2]
    
    te_g    = np.loadtxt('energy_g.dat')[:,0]
    e_g     = np.loadtxt('energy_g.dat')[:,1]
    tc_g    = np.loadtxt('heatcapacity_g.dat')[:,0]
    c_g     = np.loadtxt('heatcapacity_g.dat')[:,1]
    c_g_e   = np.loadtxt('heatcapacity_g.dat')[:,2]
    tm_g    = np.loadtxt('magnetisation_g.dat')[:,0]
    m_g     = np.loadtxt('magnetisation_g.dat')[:,1]
    ts_g    = np.loadtxt('susceptibility_g.dat')[:,0]
    s_g     = np.loadtxt('susceptibility_g.dat')[:,1]

    
    #temp vs energy kawasaki
    plt.plot(te_k, e_k, linewidth = 0.3)
    plt.scatter(te_k, e_k, s=8)
    plt.xlabel('Temperature')
    plt.ylabel('Energy')
    plt.title('Temperature vs Energy (Kawasaki)')
    plt.savefig('t_e_k.png')
    plt.show()

    #temp vs C kawasaki
    plt.plot(tc_k, c_k, linewidth = 0.3)
    plt.scatter(tc_k, c_k, s = 8)
    plt.xlabel('Temperature')
    plt.ylabel('<C>')
    plt.errorbar(tc_k, c_k, linewidth = 0.5, yerr = c_k_e)
    plt.title('Temperature vs Specific Heat Capacity (Kawasaki)')
    plt.savefig('t_c_k.png')
    plt.show()
    
    #temp vs E glauber
    plt.plot(te_g, e_g, linewidth = 0.3)
    plt.scatter(te_g, e_g, s = 8)
    plt.xlabel('Temperature')
    plt.ylabel('Energy')
    plt.title('Temperature vs Energy (Glauber)')
    plt.savefig('t_e_g.png')
    plt.show()

    #temp vs C kawasaki
    plt.plot(tc_g, c_g, linewidth = 0.3)
    plt.scatter(tc_g, c_g, s = 8)
    plt.xlabel('Temperature')
    plt.ylabel('<C>')
    plt.errorbar(tc_g, c_g, linewidth = 0.5, yerr = c_g_e)
    plt.title('Temperature vs Specific Heat Capacity (Glauber)')
    plt.savefig('t_c_g.png')
    plt.show()

    #temp vs M glauber
    plt.plot(tm_g, m_g, linewidth = 0.3)
    plt.scatter(tm_g, m_g, s = 8)
    plt.xlabel('Temperature')
    plt.ylabel('Magnitisation')
    plt.title('Temperature vs Magnitisation (Glauber)')
    plt.savefig('t_m_g.png')
    plt.show()

    #temp vs susceptibility glauber
    plt.plot(ts_g, s_g, linewidth = 0.3)
    plt.scatter(ts_g, s_g, s = 8)
    plt.xlabel('Temperature')
    plt.ylabel('Susceptibility')
    plt.title('Temperature vs Susceptibility (Glauber)')
    plt.savefig('t_s_g.png')
    plt.show()


def main():
    n = int(sys.argv[1])
    nsweeps = int(sys.argv[2])
    temp = float(sys.argv[3])

    #animation(n, nsweeps, temp)

    #plot_graphs_glauber(n, nsweeps)
    #plot_graphs_kawasaki(n, nsweeps)

    plot()


main()