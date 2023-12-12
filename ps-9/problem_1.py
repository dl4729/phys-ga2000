#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:10:54 2023

@author: pagabb
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy.linalg as la
from banded import banded

L=10**-8
N=1000
a=L/N
m=9.109e-31
h=1e-18
x0=L/2
sigma=1e-10
k=5e10
hbar = 1.05e-34

a1=1+h*(1j*hbar/(2*m*a**2))
a2=-h*(1j*hbar/(4*m*a**2))
b1=1-h*(1j*hbar/(2*m*a**2))
b2=-a2

tridiag=(np.tri(N-1,N-1,1)*np.tri(N-1,N-1,1).T)
A=a2*tridiag
np.fill_diagonal(A, a1)
B=b2*tridiag
np.fill_diagonal(B,b1)

Ainv=la.inv(A)

def psi_0(x,x0,sigma,k):
    return np.exp(-(x-x0)**2 / (2*sigma**2))*np.exp(1j*k*x)

def cn_step(psi,A,B):
    v=B.dot(psi)
    #psi_out = banded(A,v,1,1)
    psi_out=Ainv.dot(v)
    #psi_out=la.solve(A, v)

    return psi_out

def cn(psi_0, A,B, t_f):
    t=list(range(t_f))
    psi_array=[psi_0]
    for i in t:
        #psi_array.append(cn_step(psi_array[i], A,B))
        psi_next=cn_step(psi_array[i],A,B)
        psi_next/=np.sum(np.abs(psi_next)**2)
        psi_array.append(psi_next)
    return psi_array

x=np.linspace(1, N-1, N-1)
x=a*x

psi_cn = cn(psi_0(x,x0,sigma,k), A,B, 3000)
times=list(range(0,3000,300))
for i in times:
    #plt.xlim(.4,.6)
    plt.plot(x, psi_cn[i])
    #plt.xlim(.4e-8,.6e-8)
    #plt.ylim(-1,1)
    plt.xlabel('x (m)')
    plt.ylabel('Wave function')
    plt.title(f'psi({i})')
    plt.savefig(f"{i}waveform")
    plt.show()






    
    
    
    
    
    
    
    
