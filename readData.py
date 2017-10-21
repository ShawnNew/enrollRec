import json
import pprint

path = "/Users/apple/Desktop/CEE/enrollmentData.json"
file = open(path , 'r')
enrollmentData = json.load(file)
pprint.pprint (enrollmentData[0])
