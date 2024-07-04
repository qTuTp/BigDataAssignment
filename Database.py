import pymongo

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd

"""
For databaseAccess, please create a new py file name databaseAccess and add
databaseAccess = {
    "username": "<username>",
    "password": "<password>"
}
into it. And replace <username> and <password> with you database accces credential
"""
from databaseAccess import databaseAccess

def getDatabase():
    uri = f"mongodb+srv://{databaseAccess["username"]}:{databaseAccess["password"]}@bigdata.qu7vocu.mongodb.net/?appName=BigData"    
    # Create a new client and connect to the server
    client = MongoClient(uri)
    
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    
    return client["Covid19"]

def getCollectionInDataFrame(collectionName: str) -> pd.DataFrame:
    dbName = getDatabase()
    collection = dbName[collectionName]
    cursor = collection.find({})
    documents = list(cursor)

    df = pd.DataFrame(documents)
    df = df.drop(columns=['_id'])

    print(df)

if __name__ == "__main__":
    populationDB = getDatabase()
