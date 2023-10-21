#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: belper
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.fft import fft, fftfreq

def is_float(string):
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False

#load data
data = []
with open('signal.dat', 'r') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split('|')
        for i in k:
            if is_float(i):
                data.append(float(i)) 

data = np.array(data, dtype='float')
time = data[::2]
signal = data[1::2]

t=time/np.max(time)

#a
def plot_data():
    plt.scatter(t, signal, label='Data')
    plt.xlabel('t/tmax')
    plt.ylabel('Signal')


plot_data()
plt.savefig('p2data')
plt.show()

#b
t=time/np.max(time)
N=3
A=np.zeros((len(time), N+1))
for i in range(N+1):
    A[:,i]=t**i

(U,w,VT)=la.svd(A,full_matrices=False)
def inv_diag(w):
    winv=np.zeros(np.shape(w))
    for j in range(len(w)):
        if w[j]>1.e-15:
            winv[j]=1/w[j]
    return np.diag(winv)

winv=inv_diag(w)
Ainv=VT.transpose().dot(winv).dot(U.transpose())
c=Ainv.dot(signal)
y=A.dot(c)

plot_data()
plt.scatter(t,y,label='Model', c='r')
plt.savefig('cubicfit')
plt.show()

#c
plt.scatter(t,signal-y, label="residuals")
plt.plot(t,np.zeros(np.shape(t)), label="y=0", linestyle="-",c='k')
plt.legend()
plt.xlabel('t/tmax')
plt.ylabel('y')
plt.savefig('cubicresidue')
plt.show()

#d
plot_data()
for N in range(5,26,5):
    A=np.zeros((len(time), N+1))
    for i in range(N+1):
        A[:,i]=t**i

    (U,w,VT)=la.svd(A,full_matrices=False)
    def inv_diag(w):
        winv=np.zeros(np.shape(w))
        for j in range(len(w)):
            if w[j]>1.e-15:
                winv[j]=1/w[j]
        return np.diag(winv)

    winv=inv_diag(w)
    Ainv=VT.transpose().dot(winv).dot(U.transpose())
    c=Ainv.dot(signal)
    y=A.dot(c)

    plt.scatter(t,y,label=f'Model N={N}',s=5)
plt.legend()
plt.savefig('polynomials')
plt.show()

N=25
A=np.zeros((len(time), N+1))
for i in range(N+1):
    A[:,i]=t**i

(U,w,VT)=la.svd(A,full_matrices=False)
def inv_diag(w):
    winv=np.zeros(np.shape(w))
    for j in range(len(w)):
        if np.abs(w[j])>1.e-15:
            winv[j]=1/w[j]
    return np.diag(winv)

winv=inv_diag(w)
Ainv=VT.transpose().dot(winv).dot(U.transpose())
c=Ainv.dot(signal)
y=A.dot(c)
plt.scatter(t,signal-y, label="residuals")
plt.plot(t,np.zeros(np.shape(t)), label="y=0", linestyle="-",c='k')
plt.legend()
plt.xlabel('t/tmax')
plt.ylabel('y')
plt.savefig('25residue')
plt.show()

print(np.max(w)/np.min(w[w!=0]))

#e
T=np.max(time)/2.
omega=4*np.pi
N=10
A=np.zeros((len(time),2*N+2))
A[:,0]=1   
A[:,1]=t    
for i in range(2,2*N+2,2):
    A[:,i]=np.cos(i*omega*t)
    A[:,i+1]=np.sin(i*omega*t)
(U,w,VT)=la.svd(A,full_matrices=False)
def inv_diag(w):
    winv=np.zeros(np.shape(w))
    for j in range(len(w)):
        if np.abs(w[j])>1.e-15:
            winv[j]=1/w[j]
    return np.diag(winv)

winv=inv_diag(w)
winv=np.diag(1/w)
Ainv=VT.transpose().dot(winv).dot(U.transpose())
c=Ainv.dot(signal)
y=A.dot(c)

plot_data()
plt.scatter(t,y,label='Model', c='r',s=10)
plt.legend()
plt.savefig('Fourierfit')
plt.show()

plt.scatter(t,signal-y, label="residuals")
plt.plot(t,np.zeros(np.shape(t)), label="y=0", linestyle="-",c='k')
plt.legend()
plt.xlabel('t/tmax')
plt.ylabel('y')
plt.savefig('fourierresidue')
plt.show()

print(np.max(w)/np.min(w[w!=0]))
    
    


        
    

