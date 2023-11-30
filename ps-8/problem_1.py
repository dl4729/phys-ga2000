#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 20:08:32 2023

@author: belper
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft

def fftfile(filename):
    '''
    

    Parameters
    ----------
    filename : str
        filename

    Returns
    -------
    None.

    '''
    file=np.genfromtxt(filename+".txt")
    plt.plot(file)
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.savefig(filename+'_waveform')
    plt.show()
    
    t=np.asarray(list(enumerate(file)))
    t=t/44100
    T=t[1,0]-t[0,0]
    N=len(file)
    y=scipy.fft.fft(file)
    y=np.abs(y[:N//2])
    x=scipy.fft.fftfreq(N, T)[:N//2]
    plt.plot(x,2/N*y)
    plt.xlim(0,10000)
    fmax=x[y==np.max(y)]
    plt.axvline(fmax, 0,1, c='r')
    print(filename+f": {fmax}")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Fourier Coefficients')
    plt.savefig(filename+'_fourier')
    plt.show()
    
    
    return y,x, fmax

fftfile('piano')
fftfile('trumpet')