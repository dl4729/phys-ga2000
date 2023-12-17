#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:30:54 2023

@author: nancyshi
"""

import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import numba
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure
from NYU import NYU

# time-dependent external magnetic field function
def time_dependent_H(t,w=1,A=1):
    '''
    

    Parameters
    ----------
    t : int
        Time (units of timesteps)
    w : float, optional
        Frequency
    A : float, optional
        Amplitude

    Returns
    -------
    float
        External field at a particular timestep. For constant field, return constant.

    '''
    return 10*np.sin(np.pi*.00005*t)
    #return 1

def pos_dependent_H(N):
    '''
    

    Parameters
    ----------
    N : int
        Lattice Size

    Returns
    -------
    A : array
        NxN array of magnetic field values at each position in an NxN lattice.

    '''
    A=np.zeros([N,N])
    #A=NYU
    return A


# initialize lattice, it can be all ups and all downs, or a certain percentage
def initialize_lattice(N, threshold):
    '''
    

    Parameters
    ----------
    N : int
        Lattice Size
    threshold : float
        Filling parameter; determines what fraction of the lattice is initialized to spin up or spin down.

    Returns
    -------
    lattice : int array
        A lattice of 1s and -1s corresponding to the spin at a particular point.

    '''
    init_random = np.random.random((N, N))
    lattice = np.where(init_random >= threshold, 1, -1)
    return lattice

# Calculate the energy based on the given equation using convolution wrap mode for periodic boundary condition
def Energy(lattice, H_func, H_func_pos, t):
    '''
    

    Parameters
    ----------
    lattice : int array
        Ising model lattice of 1s and -1s
    H_func : function
        Time-dependent external field function.
    H_func_pos : array (2D)
        Array of external field at each lattice point for position dependent field.
    t : int
        Time in timesteps

    Returns
    -------
    float
        Total energy of the system due to neighbor-neighbor interactions and field coupling

    '''
    kernal = generate_binary_structure(2, 1)
    kernal[1][1] = False
    total = -lattice * convolve(lattice, kernal, mode='wrap', cval=0)
    return total.sum() - H_func(t) * lattice.sum() - (H_func_pos * lattice).sum()


def metropolis(N, spin_arr, times, BJ, E, H_func, H_func_pos):
    '''
    

    Parameters
    ----------
    N : int
        Lattice size
    spin_arr : int array
        Array of 1s and -1s for Ising model
    times : int
        ending time of Ising model run
    BJ : float
        Energy parameter J/kT
    E : float
        Initial energy of the system.
    H_func : function
        Time-dependent field function.
    H_func_pos : 2d float array
        Position dependent field array

    Returns
    -------
    net_spins : float array
        Total spin of the system (sum of the ending lattice) at each timestep
    net_energy : float array
        Final energy
    spin_arr_copy : float array
        Final lattice state

    '''
    spin_arr_copy = spin_arr.copy()
    net_spins = np.zeros(times - 1)
    net_energy = np.zeros(times - 1)

    for t in range(0, times - 1):
        # Pick a random point on the array and flip the spin
        x = randint(0, N)
        y = np.random.randint(0, N)
        spin_i = spin_arr_copy[x, y]
        spin_f = -spin_i

        # Compute energy change after flipping
        neighbors = [(x - 1, y), ((x + 1)%N, y), (x, y - 1), (x, (y + 1)%N)]
        E_i = sum([-spin_i * spin_arr_copy[nx, ny] for nx, ny in neighbors])
        E_f = sum([-spin_f * spin_arr_copy[nx, ny] for nx, ny in neighbors])

        # Update spin
        #dE = E_f - E_i - H_func(time_values[t]) * (spin_f - spin_i)
        dE = E_f - E_i - H_func(t) * (spin_f - spin_i) - H_func_pos[x,y]*(spin_f-spin_i)
        if dE > 0 and np.random.random() < np.exp(-BJ * dE):
            spin_arr_copy[x, y] = spin_f
            E += dE
        elif dE <= 0:
            spin_arr_copy[x, y] = spin_f
            E += dE

        net_spins[t] = spin_arr_copy.sum()
        net_energy[t] = E

    return net_spins, net_energy, spin_arr_copy

# plot the result
def plot_metropolis_results(N, spins, energies, BJ, E):
    '''
    

    Parameters
    ----------
    N : int
        Lattice size
    spins : int array
        Net spin at each timestep
    energies : float array
        Total energy at each timestep
    BJ : float
        Energy parameter J/kT
    E : float
        Initial energy of the system

    Returns
    -------
    None.

    '''
    """Plot the evolution of average spin and energy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(spins / N ** 2)
    axes[0].set_xlabel('Algorithm Time Steps')
    axes[0].set_ylabel(r'Average Spin $\bar{m}$')
    axes[0].grid()

    axes[1].plot(energies)
    axes[1].set_xlabel('Algorithm Time Steps')
    axes[1].set_ylabel(r'Energy $E/J$')
    axes[1].grid()

    fig.tight_layout()
    fig.suptitle(rf'Evolution of Average Spin and Energy for $\beta J={BJ}$', y=1.07, size=18)
    plt.show()

# get the average spin, energy and energy std
def Spin_energy(N, lattice, BJs, H_func, H_pos):
    '''
    

    Parameters
    ----------
    N : int
        Lattice size
    spin_arr : int array
        Array of 1s and -1s for Ising model
    times : int
        ending time of Ising model run
    BJ : float
        Energy parameter J/kT
    H_func : function
        Time-dependent field function.
    H_func_pos : 2d float array
        Position dependent field array

    Returns
    -------
    ms : int array
        Mean spin at each energy parameter (temperature) step
    E_means : float array
        Mean energy in the array at each energy parameter (temperature) step
    E_stds : float array
        Standard deviation of energy at each energy parameter (temperature) step

    '''
    """Get average spin, energy mean, and energy std for a range of BJ values."""
    L=lattice
    ms = np.zeros(len(BJs))
    E_means = np.zeros(len(BJs))
    E_stds = np.zeros(len(BJs))

    for i, bj in enumerate(BJs):
        num = 50000
        spins, energies, spin_arr = metropolis(N, L, num, bj, Energy(L, H_func, H_pos, 0), H_func, H_pos)
        ms[i] = spins[-num//2:].mean() / N ** 2
        E_means[i] = energies[-num:].mean()
        E_stds[i] = energies[-num:].std()
        L=spin_arr
    return ms, E_means, E_stds


def find_response(spin, H,w):
    T_fl=int(2/w)
    print(T_fl)
    max_H=H[:T_fl].argmax(axis=0)
    max_spin=spin[:T_fl].argmax(axis=0)
    return max_spin-max_H
    


cmap = plt.cm.spring

N = 50
lattice_n = initialize_lattice(N, 1)
lattice_p = initialize_lattice(N, 0)
lattice_r = initialize_lattice(N, .5)
H_p=pos_dependent_H(N)
#plt.imshow(lattice_n, cmap=cmap)
BJs = np.arange(0.1, 2, 0.005)
time_values = np.arange(0, 200000, 1)

spins, energies, spin_arr = metropolis(N, lattice_p, 100000, 1, Energy(lattice_p, time_dependent_H, H_p , 0), time_dependent_H, H_p)
plot_metropolis_results(N, spins, energies, 1, Energy(lattice_p, time_dependent_H, H_p, 0))

t=np.arange(0,100000-1,1)
plt.plot(t,spins/N**2, label="Spin")
plt.plot(t,.1*time_dependent_H(t), '--', label="Renormalized H-field")
plt.legend()
plt.xlabel('Algorithm Timesteps')
plt.ylabel('Mean Spin')
plt.show()

'''
print(find_response(spins, time_dependent_H(t),.00005))


Omega=5*1/np.arange(0,10000,10)
responses=[]
for w in Omega:
    H_t=lambda time: 10*np.cos(np.pi*w*time)
    spins,energies,spin_arr=metropolis(N, lattice_n, 100000, 1, Energy(lattice_n, H_t, H_p , 0), H_t, H_p)
    responses.append(find_response(spins, H_t(t), w))
 
    


#ms_n, E_means_n, E_stds_n = Spin_energy(N, lattice_n, BJs, time_dependent_H, H_p)
#ms_p, E_means_p, E_stds_p = Spin_energy(N, lattice_p, BJs, time_dependent_H, H_p)

plt.figure(figsize=(8, 5))
# plt.plot(1/BJs, ms_n, 'o--')
plt.plot(1 / BJs, ms_p, 'o--')
plt.xlabel('Temperature (J/k)')
plt.ylabel('Magnetization')
plt.legend(facecolor='white', framealpha=1)
plt.title('Magnetization vs. Temperature, H=-1')
plt.show()
#plt.imshow(spin_arr, cmap=cmap)
'''

'''
ms_r, E_means_r, E_stds_r = Spin_energy(N, lattice_r, BJs[::-1], time_dependent_H, H_p)
plt.figure(figsize=(8, 5))
# plt.plot(1/BJs, ms_n, 'o--')
plt.plot(1 / BJs[::-1], ms_r, 'o--')
plt.xlabel('Temperature (J/k)')
plt.ylabel('Magnetization')
plt.legend(facecolor='white', framealpha=1)
plt.title('Magnetization vs. Temperature, H=1')
plt.show()
#plt.imshow(spin_arr, cmap=cmap)
'''
