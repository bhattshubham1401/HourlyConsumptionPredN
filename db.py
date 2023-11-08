from pymongo import MongoClient


class MongoConnector:
    def __init__(self, mongo_url):
        self.client = MongoClient(mongo_url)
        self.db = self.client['PVVNL4']


