#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 21:50:38 2023

@author: belper
"""
import numpy as np
import matplotlib.pyplot as plt

tau=3.053*60.

N=1000
z=np.random.rand(N)
mu = np.log(2)/tau

t=-(1/mu)*np.log(1-z)

t_sort=np.sort(t)
decayed=np.linspace(1,1000, 1000)
plt.plot(t_sort, N-decayed)
plt.xlabel('t (s)')
plt.ylabel('Number of atoms')
plt.savefig('FasterDecay')
