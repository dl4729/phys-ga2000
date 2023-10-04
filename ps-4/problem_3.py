#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 23:05:41 2023

@author: belper
"""

import numpy as np
import matplotlib.pyplot as plt
from gaussian_quad import gaussxwab
from math import factorial

#a
def H(n,x):
    h=[np.ones(np.shape(x)),2*x]
    if n<2:
        return h[n]
    else:
        for i in range(2,n+1):
            h.append(2*x*h[i-1]-2*(i-1)*h[i-2])
    return h[n]



def inv_coeff_2(n):
    return (2**n)*factorial(n)*np.sqrt(np.pi)

x=np.linspace(-4,4,100)
for n in range(4):
    plt.plot(x, H(n,x)*np.exp(-1/2*x**2)/np.sqrt(inv_coeff_2(n)),label=f"n={n}")
plt.legend()
plt.xlabel('x')
plt.ylabel('psi')
plt.savefig('psi_a')
plt.show()

#b
x=np.linspace(-10,10,1000)
plt.plot(x,H(30,x)*np.exp(-1/2*x**2)/np.sqrt(inv_coeff_2(30)))
plt.xlabel('x')
plt.ylabel('psi')
plt.savefig('psi_b')
plt.show()

#c
def integrand(n,x):
    return(x**2)*(H(n,x)*np.exp(-1/2*x**2))**2

z,w=gaussxwab(100,-np.pi/2,np.pi/2)
var_5=np.sum(w*integrand(5,np.tan(z))/(np.cos(z)**2))
print(np.sqrt(var_5/inv_coeff_2(5)))

#d
from scipy.special import roots_hermite
x,w=roots_hermite(7)
var_5_gh = np.sum(w*(x**2)*(np.abs(H(5,x))**2))
print(np.sqrt(var_5_gh/inv_coeff_2(5)))




