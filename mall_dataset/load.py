from scipy.io import loadmat
import pandas as pd
import csv

gt=loadmat('./mall_gt.mat')

# print(type(gt['frame']), gt['frame'].shape) # (1, 2000)
# print(type(gt['count']), gt['count'].shape) # (2000, 1)
# print(type(gt['frame'][0][0]), '\n', gt['frame'][0][0], '\n',  gt['frame'][0][0].shape)
# print(type(gt['count'][0][0]), gt['count'][0][0].shape)
# print(gt['frame'][0][0][0][0][0])

# x = list(tuple(i) for i in gt['frame'][0][0][0][0][0])
# print(x)
# new_x = "\""+ str(x) + "\""
# print(new_x)

with open('gt.csv', 'w', newline='') as f:
    thewriter=csv.writer(f)
    thewriter.writerow(['filename', 'count', 'locations'])
    prefix = "seq_"
    for i in range(1, 2001):
        fn = prefix + str(i).zfill(6) + ".jpg"  # file's name
        cnt = str(gt['count'][i-1][0])
        locs = list(tuple(j) for j in gt['frame'][0][i-1][0][0][0])
        locs = str(locs)
        thewriter.writerow([fn, cnt, locs]) 

