#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 12:37:44 2023

@author: nancyshi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure

from functions1121 import initialize_lattice, Energy, metropolis, Spin_energy
from plotting import plot_metropolis_results



N = 50  
lattice_n = initialize_lattice(N, 1)
lattice_p = initialize_lattice(N, 0)
plt.imshow(lattice_p)
BJs = np.arange(0.1, 2, 0.05)
spins, energies = metropolis(N, lattice_n, 100000, 0.2, Energy(lattice_n))
plot_metropolis_results(N, spins, energies, 0.7, Energy(lattice_n) )

ms_n, E_means_n, E_stds_n = Spin_energy(N, lattice_n, BJs)
ms_p, E_means_p, E_stds_p = Spin_energy(N, lattice_p, BJs)

plt.figure(figsize=(8, 5))
#plt.plot(1/BJs, ms_n, 'o--')
plt.plot(1/BJs, ms_p, 'o--')
plt.xlabel(r'$\left(\frac{k}{J}\right)T$')
plt.ylabel(r'$\bar{m}$')
plt.legend(facecolor='white', framealpha=1)
plt.show()
