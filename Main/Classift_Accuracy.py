'''
Created on May 3, 2017

@author: tianyu
'''

import os
import re
import sys


if len(sys.argv) > 1:
    arg = sys.argv[1]
    arg2 = ""
    if len(sys.argv) > 2:
        arg2 = sys.argv[2]
else:
    print("no parameter")
    exit()

cwd = os.getcwd()
file_data = "UCEC"

folder_test = arg
num = -1
if not arg2 == "":
    num = int(arg2)

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
    if num == i:        
        for k in range(len(idx)):    
            print(groundTruth[k],idx[k])
        break
    i += 1
    filepath = cwd + "/../log/" + folder_test + "/H_" + str(i) + ".txt"

if num<0:
    for k in range(len(idx)):    
        print(groundTruth[k],idx[k])