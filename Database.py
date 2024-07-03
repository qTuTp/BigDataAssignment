import pymongo

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

def getDatabase():
    uri = "mongodb+srv://<user>:<password>@bigdata.qu7vocu.mongodb.net/?appName=BigData"
    
    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))
    
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    
    return client["Covid19"]

if __name__ == "__main__":
    populationDB = getDatabase()
