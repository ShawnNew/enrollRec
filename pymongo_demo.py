import pprint
import pymongo

def main():

    # 1. Connect to MongoDB instance running on localhost
    client = pymongo.MongoClient()

    # Access the 'restaurants' collection in the 'test' database
    collection = client.enrollmentData.collection_v2

    # 2. Insert 
#    new_documents = [
#       {"name":"Sun Bakery Trattoria", "stars":4, "categories":["Pizza","Pasta","Italian","Coffee","Sandwiches"]},
#        {"name":"Blue Bagels Grill", "stars":3, "categories":["Bagels","Cookies","Sandwiches"]},
#        {"name":"Hot Bakery Cafe","stars":4,"categories":["Bakery","Cafe","Coffee","Dessert"]},
#        {"name":"XYZ Coffee Bar","stars":5,"categories":["Coffee","Cafe","Bakery","Chocolates"]},
#        {"name":"456 Cookies Shop","stars":4,"categories":["Bakery","Cookies","Cake","Coffee"]}]

#    collection.insert_many(new_documents)

    # 3. Query 
#    for collection_v2 in collection.find():
#       pprint.pprint(collection_v2)

    # 4. Create Index 
    indexSchoolBased = collection.create_index([
        ('school', pymongo.ASCENDING),
        ('studentsInfo.stuScore', pymongo.ASCENDING)
        ])
    print(indexSchoolBased)
    pprint.pprint(indexSchoolBased)
    

    # 5. Perform aggregation
    # Group in a certain school and divide the data into different groups
    # by major.
    pipeline = [
        {"$match": {"school": "华侨大学"}},
        {"$group": {"_id": "$major", "count": {"$sum": 1}}}]
#    pprint.pprint(list(collection.aggregate(pipeline)))

if __name__ == '__main__':
    main()
