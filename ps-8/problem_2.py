#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 09:44:01 2023

@author: belper
"""

import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np

sigma=10
r=28
b=8/3

def num_solv(f, t_range, y0, t):
    sol=scipy.integrate.solve_ivp(f, t_range, y0, t_eval=t, method='LSODA')
    t=sol.t
    out=sol.y
    return t,out

def lorenz(t,w):
    x=w[0]
    y=w[1]
    z=w[2]
    dx=sigma*(y-x)
    dy=r*x-y-x*z
    dz=x*y-b*z
    return [dx,dy,dz]
 


exp_fps=1000
t_span=[0,50]
t=np.arange(*t_span,1/exp_fps)
y0=[0,1,0]

t,out=num_solv(lorenz, t_span, y0, t)
x=out[0,:]
y=out[1,:]
z=out[2,:]

plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('y')
plt.savefig('yt')
plt.show()

plt.plot(x,z)
plt.xlabel('x')
plt.ylabel('z')
plt.savefig('attractor')
plt.show()


    



    