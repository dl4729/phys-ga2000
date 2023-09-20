#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 18:57:51 2023

@author: belper
"""

import numpy as np

def quadratic1(a,b,c):
    r1 = (-b + np.sqrt(b**2-4*a*c))/(2*a)
    r2 = (-b - np.sqrt(b**2-4*a*c))/(2*a)
    return r1, r2

def quadratic2(a,b,c):
    r1 = (2*c)/(-b - np.sqrt(b**2 - 4*a*c))
    r2 = (2*c)/(-b + np.sqrt(b**2 - 4*a*c))
    return r1, r2


def quadratic(a,b,c):
    if b>0:
        r1 = (2*c)/(-b - np.sqrt(b**2 - 4*a*c))
        r2 = (-b - np.sqrt(b**2-4*a*c))/(2*a)
    else:
        r1 = (2*c)/(-b + np.sqrt(b**2 - 4*a*c))
        r2 = (-b + np.sqrt(b**2-4*a*c))/(2*a)
    x1 = np.max((r1,r2))
    x2 = np.min((r1,r2))
    return x1,x2
