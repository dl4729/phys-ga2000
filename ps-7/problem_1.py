#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: belper
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def f(x):
    return ((x-0.3)**2)*np.exp(x)

def f2(x):
    return x**2

x=np.arange(-10,10,0.1)
plt.plot(x,f(x))
plt.ylim(0,2)
plt.show()

w=.5*(3-np.sqrt(5))#golden section
        
            

def s_quad_interp(a, b, c):
    """
    inverse quadratic interpolation
    """
    #my convention is flipped; i use s_quad_interpt(a,c,b)
    epsilon = 1e-7 #for numerical stability
    s0 = a*f(b)*f(c) / (epsilon + (f(a)-f(b))*(f(a)-f(c)))
    s1 = b*f(a)*f(c) / (epsilon + (f(b)-f(a))*(f(b)-f(c)))
    s2 = c*f(a)*f(b) / (epsilon + (f(c)-f(a))*(f(c)-f(b)))
    return s0+s1+s2


def brent1d(a,c,f,epsilon=1e-7):
    b=np.mean([a,c])
    if np.abs(f(b))>=np.abs(f(a)):
        temp=a
        a=b
        b=temp
    elif np.abs(f(b))>=np.abs(f(c)):
        temp=c
        c=b
        b=temp
    error_list=[np.abs(a-b)]
    b_list=[b]
    flag=True
    while (np.abs(a-b)>=epsilon):
        if (a>c):
            temp=a
            a=c
            c=temp
        s=s_quad_interp(a,b,c)
        if s>=b:
            flag=True
        elif (flag==False and np.abs(s-b)>=np.abs(b_list[-2]-b_list[-1])):
            flag=True
        elif (flag==True and np.abs(s-b)>=np.abs(b-c)):
            flag=True
        else: flag=False
        if flag==True:
            b=a+(w*(c-a))
            x=c-(w*(c-a))
            if f(x)>f(b):
                c=x
            else:
                a=b
                b=x
        else: #s<b if this is being called
            if f(s)>f(b):
                a=s
            else:
                a=b
                b=s
        #if np.abs(f(a))<np.abs(f(b)):
            #temp=b
            #b=a
            #a=temp
        #elif np.abs(f(b))>=np.abs(f(c)):
            #temp=c
            #c=b
            #b=temp
        error_list.append(np.abs(a-b))
        b_list.append(b)
    return b, b_list, error_list

(result,b_list,error_list)=brent1d(0,1,f)

x=list(range(len(b_list)))
print("Minimum: ", result)
plt.plot(x,b_list)
plt.xlabel("Iterations")
plt.ylabel("b")
plt.savefig("b")
plt.show()

plt.plot(x,np.log(error_list))
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.savefig("error")
plt.show()
        

def brack(f,a,c):
    b=a+(w*(c-a))
    if np.abs(f(b))>=np.abs(f(a)):
        temp=a
        a=b
        b=temp
    elif np.abs(f(b))>=np.abs(f(c)):
        temp=c
        c=b
        b=temp
    if (a>c):
        temp=a
        a=c
        c=temp
    return (a,b,c)
    
print("Brent: ",scipy.optimize.brent(f, brack=(0,1),tol=1e-7,full_output=True))
    
    
    
    


