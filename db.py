import pymongo
from pymongo import MongoClient
import bcrypt

def get_database():
    CONNECTION_STRING = "mongodb+srv://ansamr76:VgEm70ifP5ED7Ape@cluster0.m0ztu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(CONNECTION_STRING)
    return client['brain_tumor_db']

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def add_chat_history(db, username, interaction):
    """Add a chat interaction to user's history"""
    return db.chat_history.update_one(
        {"username": username},
        {"$push": {"interactions": interaction}},
        upsert=True
    )

def get_chat_history(db, username):
    """Get user's chat history"""
    history = db.chat_history.find_one({"username": username})
    return history.get('interactions', []) if history else []

def clear_chat_history(db, username):
    """Clear user's chat history"""
    return db.chat_history.update_one(
        {"username": username},
        {"$set": {"interactions": []}},
        upsert=True
    )

__all__ = ['get_database', 'hash_password', 'verify_password', 
           'add_chat_history', 'get_chat_history', 'clear_chat_history']
