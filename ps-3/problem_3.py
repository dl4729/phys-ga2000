#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 21:21:18 2023

@author: belper
"""

import numpy as np
import matplotlib.pyplot as plt
import random

N_Bi213=10000
N_Tl=0
N_Pb=0
N_Bi209=0

P_Pb=97.91/100.
P_Tl = 1-P_Pb

tmax=20000.
dt=1.0
t_steps=np.arange(0.0,tmax, dt)

tau_Bi213 = 46.*60.
tau_Tl=2.2*60.
tau_Pb=3.3*60.

p_Bi213 = 1-2**(-dt/tau_Bi213)
p_Tl = 1-2**(-dt/tau_Tl)
p_Pb = 1-2**(-dt/tau_Pb)

Bi213_pts=[]
Tl_pts=[]
Pb_pts=[]
Bi209_pts=[]


for t in t_steps:
    Bi213_pts.append(N_Bi213)
    Tl_pts.append(N_Tl)
    Pb_pts.append(N_Pb)
    Bi209_pts.append(N_Bi209)
    
    decay=0
    for i in range(N_Pb):
        if random.random()<p_Pb:
            decay+=1
    N_Pb-=decay
    N_Bi209+=decay
    
    decay=0
    for i in range(N_Tl):
        if random.random()<p_Tl:
            decay+=1
    N_Tl-=decay
    N_Pb+=decay
    
    decay=0
    decay_Pb=0
    decay_Tl=0
    for i in range(N_Bi213):
        if random.random()<p_Bi213:
            decay+=1
            if random.random()<P_Pb:
                decay_Pb+=1
            else:
                decay_Tl+=1
    N_Bi213-=decay
    N_Pb+=decay_Pb
    N_Tl+=decay_Tl
    
plt.plot(t_steps, Bi213_pts, label="Bi-213")
plt.plot(t_steps, Tl_pts, label="Tl-209")
plt.plot(t_steps, Pb_pts, label="Pb-209")
plt.plot(t_steps, Bi209_pts, label="Bi-209")
plt.legend()
plt.xlabel("t(s)")
plt.ylabel("Number of Atoms")
plt.savefig('problem3')
    
    