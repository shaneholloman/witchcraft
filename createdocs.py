import csv
import sys

with open(sys.argv[1], newline='') as csvfile:
    data = csv.reader(csvfile, delimiter = '\t')
    for (num, txt) in data:
        dst = open("documents/%06d.txt" % int(num), "w")
        dst.write(txt)
        dst.close()
