#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 20:46:52 2023

@author: belper
"""

import numpy as np
import matplotlib.pyplot as plt
import time

N=np.arange(10, 201, 20)

#For loop
t_loop=np.zeros(len(N))
for m,n in enumerate(N):
    A=np.zeros((n,n),float)
    B=np.zeros((n,n),float)
    C=np.zeros((n,n),float)
    start=time.time()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j] += A[i,k]*B[k,j]
    end=time.time()
    t_loop[m]=end-start
                
plt.plot(N, t_loop)
plt.xlabel('N')
plt.ylabel('t (s)')
plt.savefig('N vs t (loop)')
plt.show()

plt.plot(N**3, t_loop)
plt.xlabel('N**3')
plt.ylabel('t (s)')
plt.savefig('N vs t_cube')
plt.show()


#Dot
t_dot = np.zeros(len(N))
for m,n in enumerate(N):
    A=np.zeros((n,n),float)
    B=np.zeros((n,n),float)
    C=np.zeros((n,n),float)
    start=time.time()
    C=np.dot(A,B)
    end=time.time()
    t_dot[m]=end-start

plt.plot(N,t_dot)
plt.xlabel('N')
plt.ylabel('t (s)')
plt.savefig('N vs t (dot)')
plt.show()

plt.plot(N,t_loop, label='Loop')
plt.plot(N,t_dot, label='Dot')
plt.xlabel('N')
plt.ylabel('t (s)')
plt.legend()
plt.savefig('N vs t both')
plt.show()
