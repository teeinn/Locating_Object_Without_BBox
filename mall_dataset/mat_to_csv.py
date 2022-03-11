from scipy.io import loadmat
import pandas as pd
import csv

gt = loadmat('./mall_gt.mat')
with open('gt_1.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['filename', 'count', 'locations'])
    prefix = "seq_"

    for i in range(1, 2001):
        fn = prefix + str(i).zfill(6) + ".jpg"  # file's name
        cnt = str(gt['count'][i - 1][0])  # the number of each images' objs
        locs = list(tuple(j) for j in gt['frame'][0][i - 1][0][0][0])  # Locations of each images' objs.
        new_list = []
        for point in locs:
            x, y = point[0], point[1]
            #csv for mat (y, x)
            new_point = (y, x)
            new_list.append(new_point)
        locs = str(new_list)
        thewriter.writerow([fn, cnt, locs])