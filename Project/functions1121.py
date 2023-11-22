#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: belper
"""
import numpy as np
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
        Lattice size (generates an NxN square lattice)
    threshold : float
        Fraction of spin up points in the lattice

    Returns
    -------
    lattice : arr
        2D Matrix of spins (1 or -1)

    '''
    init_random = np.random.random((N, N))
    lattice = np.where(init_random >= threshold, 1, -1)
    return lattice

#Calculate the energy based on the given equaiton using convolution constant mode: sets
#all value om the rim as the defined constant
def Energy(lattice):
    '''
    

    Parameters
    ----------
    lattice : arr
        Matrix of spins (1 or -1)

    Returns
    -------
    E : float
        Energy stored in the lattice spins

    '''
    kernal = generate_binary_structure(2, 1)
    kernal[1][1] = False
    total = -lattice * convolve(lattice, kernal, mode='constant', cval=0)
    E=total.sum()
    return E

@numba.njit("UniTuple(f8[:], 2)(f8, i8[:,:], i8, f8, i8)", nogil=True)
def metropolis(N, spin_arr, times, l, runstr=""):
    '''
    

    Parameters
    ----------
    N : int
        Lattice size
    spin_arr : arr
        Spin matrix (of 1, -1)
    times : int
        Times at which lattice is updated
    l : Efloat
        J/KT energy constant
    runstr : str, optional
        output filename. The default is "".

    Returns
    -------
    net_spins : float
        Net spin (magnetization)
    net_energy : float
        Net energy kT/J

    '''
    if runstr=="":
        runstr=f"sz{N}_t{times}_l{l}_spins"
    
    spin_arr_copy = spin_arr.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    E=l*Energy(spin_arr)
    
    for t in range(0,times-1):
        #Pick a point to flip
        x, y = np.random.randint(0, N, 2)
        spin_i = spin_arr_copy[x, y]
        spin_f = -spin_i
        #get neighbors with periodic boundary conditions
        neighbors = [((x-1)%N, y%N), ((x+1)%N, y%N), (x%N, (y-1)%N), (x%N, (y+1)%N)]
        E_i = sum([-spin_i * spin_arr_copy[nx, ny] for nx, ny in neighbors])
        E_f = sum([-spin_f * spin_arr_copy[nx, ny] for nx, ny in neighbors])
        
        dE = l*(E_f-E_i)
        if (dE>0 and np.random.random()<np.exp(-dE)) or dE<=0:
            spin_arr_copy[x,y]=spin_f
            E+=dE
            net_spins[t] = spin_arr_copy.sum()
            net_energy[t] = E
        
    np.savetxt(runstr, spin_arr_copy)
    return net_spins,net_energy

# get the average spin, energy and energy std
def Spin_energy(N, lattice, ls):
    '''
    

    Parameters
    ----------
    N : int
        Lattice size
    lattice : arr
        Spin lattice
    ls: arr
        List of energy constants (effectively an inverse temperature list)

    Returns
    -------
    ms : float
        Mean magnetizations
    E_means : float
        Mean energies
    E_stds : float
        Standard deviation of energies

    '''
    ms = np.zeros(len(ls))
    E_means = np.zeros(len(ls))
    E_stds = np.zeros(len(ls))

    for i, bj in enumerate(ls):
        spins, energies = metropolis(N, lattice, 1000000, bj, Energy(lattice))
        ms[i] = spins[-100000:].mean() / N**2
        E_means[i] = energies[-100000:].mean()
        E_stds[i] = energies[-100000:].std()

    return ms, E_means, E_stds
        

