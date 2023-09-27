#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 19:56:38 2023

@author: belper
"""
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x*(x-1)

def deriv_f(x):
    return 2*x - 1

def diff(f, x, delta):
    return (f(x+delta)-f(x))/delta

#Part 1
delta = 10.**-2

print(diff(f,1,delta))
print(deriv_f(1))


#Part 2
delta=np.power(10,-np.linspace(2,14,7))
plt.plot(np.linspace(2,14,7), np.log10(np.abs(diff(f,1,delta)-deriv_f(1))))
plt.xlabel('-(log(delta))')
plt.ylabel('Absolute error (log scale)')
plt.savefig('Derivative')



    