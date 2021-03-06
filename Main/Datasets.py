'''
Created on Apr 21, 2017

@author: tianyu
'''
import os
import re
import scipy.sparse as sp
import theano.sparse.basic as S
import numpy as np
import json
import os.path
import time
from six.moves import cPickle

FORCE_RECOMPUTE = False
if __name__ == '__main__':
    FORCE_RECOMPUTE = True
if FORCE_RECOMPUTE:
    print("FORCE RECOMPUTE")
# FORCE_RECOMPUTE = False

# Working directory
# dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()

# File path
file_network = "PC_NCI"
file_data = "UCEC"

# gene id to uid (which used in graph)
gid2uid = {}

# uid (which used in graph) to gene id
uid2gid = {}

# gene name to gene id
gname2gid = {}

# gene id to label position
gid2pos = {}

# edge information
edgeInfo = {}

# data 
labels = []
samples = []
groundTruth = []
 
# Read data file
with open(cwd + "/../data/" + file_data + ".txt") as tfile:    
    x = re.split("\s+",tfile.readline().strip())
    del x[0]
    labels = x
    for idx in range(len(labels)):
        gid2pos[labels[idx]] = idx
    for line in tfile:        
        x = re.split("\s+",line.strip())
        groundTruth.append(float(x.pop(0)))
        samples.append(list(map(float,x)))

samples = samples[0:20]

# Read ents file
with open(cwd + "/../data/" + file_network + ".ents") as tfile:
    tfile.readline()
    for line in tfile:
        x = re.split("\s+",line.strip())
        gid2uid[x[2]] = x[0]
        uid2gid[x[0]] = x[2]
        gname2gid[x[1]] = x[2]

# Read rels file
with open(cwd + "/../data/" + file_network + ".rels") as tfile:
    tfile.readline()
    for line in tfile:
        x = re.split("\s+",line.strip())
        if x[1] in uid2gid and x[2] in uid2gid:
            if not uid2gid[x[1]] in edgeInfo:
                edgeInfo[uid2gid[x[1]]] = {}
            if not uid2gid[x[2]] in edgeInfo:
                edgeInfo[uid2gid[x[2]]] = {}
            edgeInfo[uid2gid[x[1]]][uid2gid[x[2]]] = x[3]
            edgeInfo[uid2gid[x[2]]][uid2gid[x[1]]] = x[3]

# Adjacency matrix
afpath = cwd + "/../data/" + file_network + "_adj_matrix.txt"
if FORCE_RECOMPUTE or (not os.path.isfile(afpath)):
    data = []
    rows = []
    columns = []
    NON_ZEROS = 0
    for i in range(len(labels)):
        if labels[i] in edgeInfo:
            for j in edgeInfo[labels[i]].keys():
                if j in gid2pos:
                    data.append(1);
                    columns.append(i)
                    rows.append(gid2pos[j])
                    NON_ZEROS += 1
    
    __ADJ_MAT = sp.csr_matrix((data,(rows,columns)), shape=(len(labels), len(labels)))
    ADJ_DIAG_SUM = S.sp_sum(__ADJ_MAT,1).eval()
    D = np.diag(ADJ_DIAG_SUM)
    L = sp.csr_matrix(S.sub(D, __ADJ_MAT))
    # D_star = D^(-1/2) where D is a diagonal matrix with diagonal entries equals the row sum of ADJ_MAT
    ADJ_DIAG_SQRT = np.sqrt(ADJ_DIAG_SUM)
    ADJ_DIAG_INV = np.float64(1.0)/ADJ_DIAG_SQRT
    D_star = sp.csr_matrix((ADJ_DIAG_INV,(range(len(labels)),range(len(labels)))), shape=(len(labels), len(labels)))
    
    W = S.dot(S.dot(D_star,__ADJ_MAT),D_star).eval()
    W = sp.csr_matrix(np.where(np.isnan(W), 0, W))
    W_SUM = S.sp_sum(W).eval()
    with open(afpath,"wb") as tfile:
        for obj in [W,L,W_SUM]:
            cPickle.dump(obj, tfile, protocol=cPickle.HIGHEST_PROTOCOL)
else:
    with open(afpath,"rb") as tfile:
        W = cPickle.load(tfile)
        L = cPickle.load(tfile)
        W_SUM = cPickle.load(tfile)

logStart = time.strftime("%Y_%m_%d_%H_%M_%S")
logpath = cwd + "/../log/" + logStart + "/"
if not os.path.exists(logpath):
    os.makedirs(logpath)
    
with open(logpath + time.strftime("%Y_%m_%d_%H_%M_%S") + ".txt","w") as tfile:
    tfile.write(time.strftime("%c") + '\n')
    
def log(str):
    with open(logpath + logStart + ".txt","a") as tfile:
        tfile.write(str + '\n')
        
def dmp(str,info):
    with open(logpath + str + ".txt","w") as tfile:
        tfile.write(time.strftime("%c") + '\n')
        tfile.write(info + '\n')

def dmpnp(str,info):
    np.savetxt(logpath + str + ".txt", info ,fmt='%0.6f')

def dmpj(str,jv): 
    with open(logpath + str + ".txt","w") as tfile:
        json.dump(jv, tfile)
        
# f = cwd + "/../data/" + file_data
# np.savetxt(f + "_adj_matrixD.txt", D ,fmt='%0.0f')
# np.savetxt(f + "_adj_matrixA.txt", A ,fmt='%0.0f')
# np.savetxt(f + "_adj_matrixL.txt", L ,fmt='%0.0f')

# dmpnp("L",L)

print("Data load complete")