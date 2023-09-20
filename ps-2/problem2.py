#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 23:44:39 2023

@author: belper
"""
import time
import numpy as np


def magnitude(i,j,k):
    return np.sqrt(i**2+j**2+k**2)

def contribution(i,j,k):
    """
    

    Parameters
    ----------
    i : integer
        x coordinate
    j : integer
        y coordinate
    k : zinteger
        z coordinate

    Returns
    -------
    float which is the contribution to the Madelung constant up to a factor of 1/(4*pi*epsilon_0*a)

    """
    if np.max(np.abs((i,j,k)))==0:
        return 0
    else: 
        return (-1)**(i+j+k)/magnitude(i,j,k)
    
L=100

s=time.time()
grid=np.arange(-L, L+.1, 1)
i,j,k=np.meshgrid(grid,grid,grid)








M = contribution(i,j,k)
M[M==np.inf]=0
e=time.time()
print(M.sum())
print(e-s)


