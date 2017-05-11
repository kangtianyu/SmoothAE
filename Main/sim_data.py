'''
Created on May 7, 2017

@author: tianyu
'''

import numpy as np
import scipy.sparse as sp
import numpy.random as rng
import time
import os
from six.moves import cPickle

FORCE_RECOMPUTE = False
# FORCE_RECOMPUTE = True

FEATURE_SIZE = 50
SAMPLE_NUM = 200
alpha = 0.7
labels = list(range(FEATURE_SIZE))

fileName = "sim002"

cwd = os.getcwd()
logStart = time.strftime("%Y_%m_%d_%H_%M_%S")
logpath = cwd + "/../log/" + logStart + "/"
if not os.path.exists(logpath):
    os.makedirs(logpath)
    
with open(logpath + time.strftime("%Y_%m_%d_%H_%M_%S") + ".txt","w") as tfile:
    tfile.write(time.strftime("%c") + '\n')
    
def log(str):
    with open(logpath + logStart + ".txt","a") as tfile:
        tfile.write(str + '\n')
          
def dmpnp(str,info):
    np.savetxt(logpath + str + ".txt", info ,fmt='%0.6f')
    
afpath = cwd + "/../data/" + fileName + ".txt"
if FORCE_RECOMPUTE or (not os.path.isfile(afpath)):    
    connections = [(0,7),(1,7),(2,7),(3,7),(4,7),(10,11),(10,15),(10,16),(11,15),(11,16),(15,16),(13,14),(13,19),(14,19),(18,19)]
    sub_keys = [[0,1,2,3,4,7],
                [10,11,15,16],
                [13,14,18,19]]
    sub_types = [[1,0,0],
                 [0,0,1],
                 [0,1,0]]
    
    A = np.zeros((FEATURE_SIZE,FEATURE_SIZE))
    for con in connections:
        A[con[0]][con[1]] = 1
        A[con[1]][con[0]] = 1
    D = np.diag(np.sum(A,0))
    L = sp.csr_matrix(D - A)
    D_nh = np.diag(np.float64(1.0)/np.sqrt(np.sum(A,0)))
    D_nh = np.where(np.isinf(D_nh), 0, D_nh)
    W = np.dot(np.dot(D_nh,A),D_nh)
    W_SUM = np.sum(W)
    
    B = np.zeros((3,FEATURE_SIZE))
    for i in range(len(sub_keys)):
        for j in sub_keys[i]:
            B[i][j] = 1
    
    mat = []
    for i in range(len(sub_types)):
        type = sub_types[i]
        x0 = np.dot(type,B)
        xt = np.copy(x0)
        while True:
            xt1 = alpha * np.transpose(np.dot(W,(np.transpose([xt]))))[0] + np.multiply(x0,1-alpha)
            if np.sum((xt1-xt)**2)<1e-100:
                break
            xt = xt1
        w_sum = np.copy(xt)
        for j in range(SAMPLE_NUM):
            res = np.zeros(FEATURE_SIZE)
            while np.count_nonzero(res) == 0:
                res = list(np.greater(w_sum+0.1,rng.rand(FEATURE_SIZE)*3)*1.0)
            res.append(i)
            mat.append(res)
    rng.shuffle(mat)
    mat = np.transpose(mat)
    samples = np.transpose(mat[0:len(mat)-1])
    groundTruth = np.transpose(mat[len(mat)-1])
    
    with open(afpath,"wb") as tfile:
        for obj in [W,L,W_SUM,samples,groundTruth]:
            cPickle.dump(obj, tfile, protocol=cPickle.HIGHEST_PROTOCOL)
else:    
    with open(afpath,"rb") as tfile:
        W = cPickle.load(tfile)
        L = cPickle.load(tfile)
        W_SUM = cPickle.load(tfile)
        samples = cPickle.load(tfile)
        groundTruth = cPickle.load(tfile)