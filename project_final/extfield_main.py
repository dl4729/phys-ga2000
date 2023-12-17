#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 09:33:16 2023

@author: pagabb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure
from externalfieldfunction_nonumba import initialize_lattice, Energy, metropolis, plot_metropolis_results, Spin_energy, pos_dependent_H, time_dependent_H
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

ms_n, E_means_n, E_stds_n = Spin_energy(N, lattice_n, BJs, time_dependent_H, H_p)
ms_p, E_means_p, E_stds_p = Spin_energy(N, lattice_p, BJs, time_dependent_H, H_p)

plt.figure(figsize=(8, 5))
# plt.plot(1/BJs, ms_n, 'o--')
plt.plot(1 / BJs, ms_p, 'o--')
plt.xlabel('Temperature (J/k)')
plt.ylabel('Magnetization')
plt.legend(facecolor='white', framealpha=1)
plt.title('Magnetization vs. Temperature, H=-1')
plt.show()
#plt.imshow(spin_arr, cmap=cmap)



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



