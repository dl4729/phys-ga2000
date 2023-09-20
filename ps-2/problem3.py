#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 23:45:20 2023

@author: belper
"""

#Mandelbrot Set

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

def magnitude(z):
    return np.sqrt(np.real(z)**2 + np.imag(z)**2)
    
#Generate c
N = 1000
grid_param = 4
#4x4 grid

a = np.linspace(-grid_param/2, grid_param/2, N+1)
b = np.linspace(-grid_param/2, grid_param/2, N+1)*1j
c = np.zeros((N+1, N+1), dtype='complex128')
for i in range(len(a)):
    c[i]=a[i]+b

iters=np.zeros((N+1, N+1))
z = np.zeros((N+1, N+1), dtype='complex128')
steps=1000
for s in range(steps):
    z[magnitude(z)>=2]=np.inf
    iters[z==np.inf]=s
    z = z*z+c
    
iters[iters==0]=steps
mod=magnitude(z)


mandelbrot = c[mod<2]
n_mandelbrot = c[np.logical_or(np.logical_or(mod>=2, np.isnan(mod)),np.isinf(mod))]
#plt.scatter(np.real(mandelbrot), np.imag(mandelbrot), color='k',s=.001)
#plt.scatter(np.real(n_mandelbrot), np.imag(n_mandelbrot), color='w',s=.001)
plt.scatter(np.real(c), np.imag(c), c=iters,s=.01)
plt.xlabel('Real(z) (unitless)')
plt.ylabel('Imag(z) (unitless)')
plt.savefig('Mandelbrot')




