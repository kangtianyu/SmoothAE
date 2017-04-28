'''
Created on Apr 25, 2017

@author: tianyu
'''
import numpy as np

alpha = 0.6

A = [[0,1,1,1,1,1,1,1,1],
     [1,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0]]
X0 = [0,1,1,1,1,1,1,1,1]

Xt = X0
for i in range(5000):
#     Xt = alpha * np.dot(Xt,A) + np.multiply(1-alpha,X0)
    Xt = (1-alpha) * np.dot(Xt,A) + alpha * X0
    print(Xt)