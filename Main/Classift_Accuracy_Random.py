'''
Created on May 3, 2017

@author: tianyu
'''

import os
import re
import numpy.random as rng

cwd = os.getcwd()
file_data = "UCEC"

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
idx = rng.randint(0,3,20)
sample_num = len(idx)
for m in range(sample_num -1):
    for n in range(m+1,sample_num):
        if((groundTruth[m]==groundTruth[n]) and idx[m] == idx[n] or ((not groundTruth[m]==groundTruth[n]) and (not idx[m] == idx[n]))):
            tr +=1
        tot += 1
print(str(tr) + "/" + str(tot) + ":" + str(float(tr)/tot))