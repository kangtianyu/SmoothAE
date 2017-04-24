'''
Created on Apr 21, 2017

@author: tianyu
'''
import os
import re
import scipy.sparse as sp
import numpy as np

# Working directory
# dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()

# File path
file_ents = "PC_NCI.ents"
file_rels = "PC_NCI.rels"
file_data = "UCEC.txt"

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
with open(cwd + "/../data/" + file_data) as tfile:    
    x = re.split("\s+",tfile.readline().strip())
    del x[0]
    labels = x
    for idx in range(len(labels)):
        gid2pos[labels[idx]] = idx
    for line in tfile:        
        x = re.split("\s+",line.strip())
        groundTruth.append(float(x.pop(0)))
        samples.append(map(float,x))
tfile.close()

# Read ents file
with open(cwd + "/../data/" + file_ents) as tfile:
    tfile.readline()
    for line in tfile:
        x = re.split("\s+",line.strip())
        gid2uid[x[2]] = x[0]
        uid2gid[x[0]] = x[2]
        gname2gid[x[1]] = x[2]
tfile.close()

# Read rels file
with open(cwd + "/../data/" + file_rels) as tfile:
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
tfile.close()

# Adjacency matrix
data = []
rows = []
columns = []
list = []
NON_ZEROS = 0
for i in range(len(labels)):
    if labels[i] in edgeInfo:
        t = 0
        for j in edgeInfo[labels[i]].keys():
            if j in gid2pos:
                data.append(1);
                columns.append(i)
                rows.append(gid2pos[j])
                NON_ZEROS += 1
                t += 1
        list.append(t)
D = np.diag(list)
ADJ_MAT = sp.csr_matrix((data,(rows,columns)), shape=(len(labels), len(labels)))
L = D - ADJ_MAT

print("Data load complete")

# samples = [samples[0]]