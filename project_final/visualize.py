#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:34:13 2023

@author: nancyshi
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve, generate_binary_structure
#from functions import initialize_lattice, Energy, metropolis,plot_metropolis_results, Spin_energy
from externalfieldfunction_nonumba import initialize_lattice, Energy, metropolis,plot_metropolis_results, Spin_energy, time_dependent_H, pos_dependent_H
from ipywidgets import interact
from NYU import NYU

N = 200
def display_spin_field(field, N, ts):
    '''
    

    Parameters
    ----------
    field : int array
        Ising model lattice
    N : int
        Lattice size
    ts : int
        timestep to display the spin lattice

    Returns
    -------
    None.

    '''
    plt.imshow(field, cmap='gray')
    plt.title(f"Visualization for N = {N}, Ts = {ts}")
    plt.show()

initial = initialize_lattice(N, 0.50)
display_spin_field(initial, N, ts = 0)

ts_list = [50000, 100000, 200000, 400000, 600000, 800000,1000000, 1500000, 2000000, 3000000, 4000000, 5000000]
ts_list2 = [20000, 40000, 60000, 80000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 1000000]
for i in ts_list2:
    net_spins, net_energy, spin_arr_copy = metropolis(N, initial, i, 0.2, Energy(initial, time_dependent_H, NYU, 0), time_dependent_H, NYU)
    display_spin_field(spin_arr_copy, N, i)


