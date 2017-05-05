'''
Created on May 3, 2017

@author: tianyu
'''

import os
import re
import sys
import numpy.random as rng

if len(sys.argv)>1:
    arg = sys.argv[1]

cwd = os.getcwd()
file_data = "UCEC"

chg = int(arg)

groundTruth = []
with open(cwd + "/../data/" + file_data + ".txt") as tfile:    
    x = re.split("\s+",tfile.readline().strip())
    del x[0]
    labels = x
    for line in tfile:        
        x = re.split("\s+",line.strip())
        groundTruth.append(int(x.pop(0)))

idx = []
tr = 0
tot = 0
idx = groundTruth[0:20]
sample_num = len(idx)
x = list(range(sample_num))
rng.shuffle(x)
chgidx = x[0:chg]
print(chgidx)
for i in chgidx:
    while True:
        t = rng.randint(0,3)
        if not t == idx[i]:
            idx[i] = t
            break;
for m in range(sample_num -1):
    for n in range(m+1,sample_num):
        if((groundTruth[m]==groundTruth[n]) and idx[m] == idx[n] or ((not groundTruth[m]==groundTruth[n]) and (not idx[m] == idx[n]))):
            tr +=1
        tot += 1
print(str(tr) + "/" + str(tot) + ":" + str(float(tr)/tot))