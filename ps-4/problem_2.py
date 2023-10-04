#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 22:17:50 2023

@author: belper
"""

import numpy as np
import matplotlib.pyplot as plt
from gaussian_quad import gaussxwab

def V(x):
    return x**4

def integrand(x,a):
    return 1/np.sqrt(V(a)-V(x))

def T(a):
    N=20
    x,w = gaussxwab(N,0,a)
    gauss=np.sum(w*integrand(x,a))
    return np.sqrt(8)*gauss

A=np.linspace(0,2,101)
periods=[]
for a in A:
    periods.append(T(a))
plt.plot(A, periods)
plt.xlabel("Amplitude (a)")
plt.ylabel("Period")
plt.savefig("problem2")