#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: belper
"""

import numpy as np
from gaussian_quad import gaussxwab
import matplotlib.pyplot as plt

#a
def f(x,a):
    return x**(a-1)*np.exp(-x)

x=np.linspace(0,5,501)
for a in range(2,5):
    plt.plot(x,f(x,a),label=f'a={a}')
    
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.savefig('1a')
plt.show()

#e
order=20
def f_2(x,a):
    return np.exp((a-1)*np.log(x)-x)

def gamma(a):
    z,w=gaussxwab(order,0,1)
    return np.sum(w*f_2(z*(a-1)/(1-z),a)*(a-1)/((1-z)**2))

print(f'a=3/2: gamma={gamma(3/2)}')
print(f'a=3: gamma={gamma(3)}')
print(f'a=6: gamma={gamma(6)}')
print(f'a=10: gamma={gamma(10)}')




    
