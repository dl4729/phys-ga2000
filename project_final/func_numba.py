#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:59:07 2023

@author: pagabb
"""

import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import numba
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure

#initialize lattice, it can be all ups and all downs, or vertain percentage
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
def Energy(lattice):
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
    return total.sum()

@numba.njit("Tuple((f8[:], f8[:], i8[:, :]))(i8, i8[:,:], i8, f8, i8)", nogil=True)
def metropolis(N, spin_arr, times, BJ, E):
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
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)

    for t in range(0, times-1):
        # Pick random point on array and flip spin
        x = randint(0, N)
        y = np.random.randint(0, N)
        spin_i = spin_arr_copy[x, y]
        spin_f = -spin_i
        
        # Compute energy change after flipping
        neighbors = [(x-1, y), ((x+1)%N, y), (x, y-1), (x, (y+1)%N)]
        E_i = sum([-spin_i * spin_arr_copy[nx, ny] for nx, ny in neighbors])
        E_f = sum([-spin_f * spin_arr_copy[nx, ny] for nx, ny in neighbors])

        # Update spin 
        dE = E_f - E_i
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
    
    axes[0].plot(spins/N**2)
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
def Spin_energy(N, lattice, BJs, ts):
    '''
    

    Parameters
    ----------
    N : int
        Lattice size
    spin_arr : int array
        Array of 1s and -1s for Ising model
    BJ : float
        Energy parameter J/kT
    ts : int
        ending time of Ising model run


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
        spins, energies, spin_arr= metropolis(N, L, ts, bj, Energy(L))
        ms[i] = spins[-ts//2:].mean() / N**2
        E_means[i] = energies[-ts//2:].mean()
        E_stds[i] = energies[-ts//2:].std()
        L=spin_arr

    return ms, E_means, E_stds