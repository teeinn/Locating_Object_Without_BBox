import csv
import ast

original_csv = '../towncenter/towncenter_gt.csv'
new_csv = './towncenter_gt.csv'


with open(new_csv, 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['filename', 'count', 'locations'])

    with open(original_csv, 'r', encoding='utf-8', newline='') as readFile:
        read = csv.reader(readFile)
        for row in read:
            locs = row[2]
            locs = ast.literal_eval(locs)

            new_list = []
            for point in locs:
                x, y = point[0], point[1]
                new_point = (y, x)
                new_list.append(new_point)
            locs_ = str(new_list)
            thewriter.writerow([row[0], row[1], locs_])
    readFile.close()
f.close()


