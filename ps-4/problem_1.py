#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 21:10:36 2023

@author: belper
"""

import numpy as np
import matplotlib.pyplot as plt
from gaussian_quad import gaussxwab

V=.001 #m^3
p=6.022e28
theta=428.
k_b = 1.38e-23


def f_debye(x):
    return (x**4 * np.exp(x))/((np.exp(x)-1)**2)


def cv(T,N):
    x, w = gaussxwab(N, 0, theta/T)
    gauss=np.sum(w*f_debye(x))
    return 9*V*p*((T/theta)**3)*k_b*gauss


T=np.linspace(5,500,100)
spec_heat=[]
for t in T:
    spec_heat.append(cv(t,59))
plt.plot(T,spec_heat)
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (J/K)')
plt.savefig('HeatCapacity50')

plt.show()

N_ar=[1,10,20,30,40,50,60,70]
for n in N_ar:
    spec_heat=[]
    for t in T:
        spec_heat.append(cv(t,n))
    plt.plot(T,spec_heat,label=f"N={n}")


plt.legend()
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (J/K)')
plt.savefig('HeatCapacity_all')
plt.show()
        
    


    