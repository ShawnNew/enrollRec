import json
import csv
import pprint

path1 = "/Users/apple/Desktop/enrollmentData1.json"
path2 = "/Users/apple/Desktop/stuDataOfTenSchools.csv"

file2 = open(path2).read().decode('utf8')
reader = csv.DictReader(file2)
dict_list = []
for line in reader:
    dict_list.append(line)

pprint.pprint(dict_list)

#enrollmentData2 = csv_dict_list(file2)
#pprint.pprint(enrollmentData2)


file1 = open(path1 , 'r')
enrollmentData1 = json.load(file1)

#pprint.pprint (enrollmentData[0])
