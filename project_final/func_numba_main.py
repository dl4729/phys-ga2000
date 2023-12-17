#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 12:37:44 2023

@author: nancyshi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure
from func_numba import initialize_lattice, Energy, metropolis, plot_metropolis_results, Spin_energy
cmap = plt.cm.spring

N = 100
ts = 1000000
lattice_n = initialize_lattice(N, 1)
lattice_p = initialize_lattice(N, 0)
#plt.imshow(lattice_n, cmap = cmap)
BJs = np.arange(0.1, 2, 0.01)
spins, energies, spin_arr= metropolis(N, lattice_n, ts, 0.2, Energy(lattice_n))
plot_metropolis_results(N, spins, energies, 0.7, Energy(lattice_n) )


ms_n, E_means_n, E_stds_n = Spin_energy(N, lattice_n, BJs, ts)
ms_p, E_means_p, E_stds_p = Spin_energy(N, lattice_p, BJs, ts)

plt.figure(figsize=(8, 5))
#plt.plot(1/BJs, ms_n, 'o--')
plt.plot(1/BJs, ms_p, 'o--')
plt.title(f"Magnetization vs. Temperature for N = {N}")
plt.xlabel('Temperature (k/J)')
plt.ylabel('Magnetization')
#plt.legend(facecolor='white', framealpha=1)
plt.show()
#plt.imshow(spin_arr, cmap = cmap)
