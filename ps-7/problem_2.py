#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: belper
"""

import numpy as np
import matplotlib as pyplot
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt

dat=pd.read_csv('survey.csv')

def p(x,b0,b1):
    return 1/(1+np.exp(-(b0+b1*x)))

xs = dat['age'].to_numpy()
ys = dat['recognized_it'].to_numpy()
x_sort = np.argsort(xs)
xs = xs[x_sort]
ys = ys[x_sort]


def log_likelihood(beta,x,y):
    log_list=[]
    b0=beta[0]
    b1=beta[1]
    offset=1e-15
    for i in range(len(xs)):
        p_i=p(xs[i],b0,b1)
        log=ys[i]*np.log(p_i/(1-p_i+offset) + offset)+np.log(1-p_i+offset)
        log_list.append(log)
    return -np.sum(np.array(log_list), axis=-1)


guess=[-10,0]

def covariance(hess_inv, var):
    return hess_inv*var

def uncertainty(hess_inv, var):
    cov=covariance(hess_inv,var)
    return np.sqrt(np.diag(cov))

minimum = scipy.optimize.minimize(log_likelihood, x0=guess, args=(xs,ys))
hess_inv=minimum.hess_inv
var = minimum.fun/(len(ys)-len(guess)) 

cov=covariance(hess_inv,var)
unc=uncertainty(hess_inv,var)
print('Optimal parameters and error:\n\tp: ' , minimum.x)
print('Covariance matrix of optimal parameters:\n\tC: ' , covariance( hess_inv,  var))

beta=minimum.x
x_prob=np.linspace(0,np.max(xs),101)
plt.scatter(xs,ys,label="Data")
plt.plot(x_prob, p(x_prob, beta[0],beta[1]), 'r', label="Logistic Fit")
plt.xlabel('Age (years)')
plt.ylabel('Answer')
plt.legend()
plt.savefig("logistic")
plt.show()


        




