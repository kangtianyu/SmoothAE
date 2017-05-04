'''
Created on May 3, 2017

@author: tianyu
'''

import os
import re
import sys

arg = sys.argv[1]
cwd = os.getcwd()
file_data = "UCEC"
if arg == "":
    folder_test = "2017_05_03_15_43_17"
else:
    folder_test = arg
    
groundTruth = []
with open(cwd + "/../data/" + file_data + ".txt") as tfile:    
    x = re.split("\s+",tfile.readline().strip())
    del x[0]
    labels = x
    for line in tfile:        
        x = re.split("\s+",line.strip())
        groundTruth.append(int(x.pop(0)))

i = 1
filepath = cwd + "/../log/" + folder_test + "/H_" + str(i) + ".txt"
while os.path.isfile(filepath):
    idx = []
    tr = 0
    tot = 0
    with open(filepath) as tfile:
        for line in tfile:        
            x = list(map(float,re.split("\s+",line.strip())))
            idx.append(x.index(max(x)))
    sample_num = len(idx)
    for m in range(sample_num -1):
        for n in range(m+1,sample_num):
            if((groundTruth[m]==groundTruth[n]) and idx[m] == idx[n] or ((not groundTruth[m]==groundTruth[n]) and (not idx[m] == idx[n]))):
               tr +=1
            tot += 1
    print(str(i) + ":" + str(tr) + "/" + str(tot) + ":" + str(float(tr)/tot))
    i += 1
    filepath = cwd + "/../log/" + folder_test + "/H_" + str(i) + ".txt"