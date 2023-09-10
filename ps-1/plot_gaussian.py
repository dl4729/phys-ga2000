# -*- coding: utf-8 -*-
"""
Dylan Lane
Problem Set 01
PHYS-GA2000
Computational Physics
Professor Blanton
"""

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x,mean,std):
    """
    Parameters
    ----------
    x : Numpy Array
        Array of x-values over which the Gaussian is to be plotted
    mean : numerical (typically float)
        The mean which determines the center of the Gaussian.
    std : numerical (typically float)
        Standard deviation which determines the width of the Gaussian.

    Returns
    -------
    Numpy Array.

    """
    norm = np.sqrt(2*np.pi)*std
    G = (1/norm)*np.exp(-((x-mean)**2)/(2*std**2))
    return G
    
x=np.arange(-10., 10., .05)
sigma=3.
mean=0.

plt.plot(x,gaussian(x,mean,sigma))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gaussian: Mean=0., St. Dev=3.')
plt.savefig('gaussian.png')
