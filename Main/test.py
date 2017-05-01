'''
Created on Apr 25, 2017

@author: tianyu
'''
import numpy as np
import numpy.random as rng

alpha = 0.6

A = [[0,0,1,1,1,1,1,1,1],
     [0,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0]]

D = np.diag(np.float64(1.0)/np.sqrt(np.sum(A,0)))

W = np.dot(np.dot(D,A),D)
W = np.where(np.isnan(W), 0, W)

print(np.float64(1.0) / 0.0)
print((np.float64(1.0) / 0.0)*0)

print(W)

X0 = [[0,1,1,1,1,1,1,1,1]]

Xt = rng.rand(9)
print(Xt)
print("---")
for i in range(100):
#     Xt = alpha * np.dot(Xt,A) + np.multiply(1-alpha,X0)
    Xt = (1-alpha) * np.dot(Xt,W) + np.multiply(alpha,X0)
    print(Xt)