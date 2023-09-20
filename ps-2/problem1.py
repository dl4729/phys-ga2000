#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 23:28:30 2023

@author: belper
"""

import numpy as np


#get_bits taken from:
#https://nbviewer.org/github/blanton144/computational-grad/blob/main/docs/notebooks/intro.ipynb
#Written by Professor Blanton
def get_bits(number):
    """For a NumPy quantity, return bit representation
    
    Inputs:
    ------
    number : NumPy value
        value to convert into list of bits
        
    Returns:
    -------
    bits : list
       list of 0 and 1 values, highest to lowest significance
    """
    bytes = number.tobytes()
    bits = []
    for byte in bytes:
        bits = bits + np.flip(np.unpackbits(np.uint8(byte)), np.uint8(0)).tolist()
    return list(reversed(bits))

bitlist=get_bits(np.float32(100.98763))
sign=bitlist[0]
exponent=bitlist[1:9]
mantissa=bitlist[10:32]

print(f"Sign Bit: {sign}\nExponent Bit: {exponent}\nMantissa: {mantissa}")

print(np.float64(100.98763)-np.float32(100.98763))