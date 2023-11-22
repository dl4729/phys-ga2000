#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:52:29 2023

@author: nancyshi
"""

import numpy as np
import matplotlib.pyplot as plt

def theoretical_magnetization(T, J, k):
    Tc = 2.269185 * J / k  
    if T > Tc:
        return 0
    else:
        z = np.exp(-2*J/(k*T))
        return (1 + z**2)**0.25 * (1 - 6*z**2 + z**4)**0.125

J = 1  
k = 1  
temperatures = np.linspace(1, 10, 500)  # Sample temperatures
magnetizations = [theoretical_magnetization(T, J, k) for T in temperatures]

plt.plot(temperatures, magnetizations, '-o', label="Theoretical")
plt.xlim(0, 10)
plt.xlabel("Temperature T")
plt.ylabel("Magnetization M")
plt.title("Theoretical Magnetization vs. Temperature")
plt.legend()
plt.show()
