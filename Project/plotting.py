#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: belper
"""
import matplotlib.pyplot as plt

def plot_metropolis_results(N, spins, energies,l):
    '''
    

    Parameters
    ----------
    N : int
        Lattice size
    spins : arr
       spin matrix
    energies : arr
        Energies
    l : arr
        Energy constants (inverse temperatures)

    Returns
    -------
    None.

    '''
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(spins/N**2)
    axes[0].set_xlabel('Algorithm Time Steps')
    axes[0].set_ylabel(r'Average Spin $\bar{m}$')
    axes[0].grid()
    
    axes[1].plot(energies)
    axes[1].set_xlabel('Algorithm Time Steps')
    axes[1].set_ylabel(r'Energy $E/l$')
    axes[1].grid()
    
    fig.tight_layout()
    fig.suptitle(rf'Evolution of Average Spin and Energy for l={l}$', y=1.07, size=18)
    plt.show()