
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
from typing import Optional, List
# ------------------------------------------------------------------
from .user_schema import UserSchema

class DatabaseManager:
    def __init__(self, connection_str = "MONGODB_CONNECTION_STRING", db_name="HACK-CX-Hackathon", collection_name="users"):
        try:
            load_dotenv()
            self.uri = os.getenv(connection_str)
            self.client = MongoClient(self.uri, server_api=ServerApi('1'))
            self.client.admin.command('ping')
            self.collection = self.client.get_database(db_name).get_collection(collection_name)

            print("Successfully initialized DatabaseManager and connected to MongoDB!")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")

    def create_user(self, user_data) -> dict:
        """Create a new user"""
        try:
            existing_user = self.collection.find_one({"user_id": user_data["user_id"]})
            if existing_user:
                print(f"Error: User with user_id '{user_data['user_id']}' already exists!")
                return {"success": False, "message": f"User with user_id '{user_data['user_id']}' already exists!"}
            
            result = self.collection.insert_one(user_data)
            print(f"User created successfully with ID: {result.inserted_id}")
            return {"success": True, "message": f"User created successfully with ID: {result.inserted_id}"}
        
        except Exception as e:
            print(f"Error creating user: {e}")
            return {"success": False, "message": str(e)}

    def get_user_by_id(self, user_id) -> Optional[UserSchema]:
        try:
            user = self.collection.find_one({"user_id": user_id})
            if user:
                print(f"User found: {user}")
                return user
            else:
                print(f"No user found with user_id: {user_id}")
                return None
        except Exception as e:
            print(f"Error finding user: {e}")
            return None

    def get_all_users(self) -> List[UserSchema]:
        try:
            users = list(self.collection.find())
            print(f"Found {len(users)} users:")
            return users
        except Exception as e:
            print(f"Error getting users: {e}")
            return []

    def update_user_info(self, user_id, update_data) -> dict:
        try:        
            result = self.collection.update_one(
                {"user_id": user_id},
                {"$set": update_data}
            )
            
            if result.matched_count > 0:
                print(f"Successfully updated user '{user_id}' with data: {update_data}")
                return {"success": True, "message": f"User '{user_id}' updated successfully."}
            else:
                print(f"No user found with user_id: {user_id}")
                return {"success": False, "message": f"No user found with user_id: {user_id}"}
        
        except Exception as e:
            print(f"Error updating user: {e}")
            return {"success": False, "message": str(e)}
        
    def delete_user(self, user_id) -> dict:
        try:
            result = self.collection.delete_one({"user_id": user_id})
            if result.deleted_count > 0:
                print(f"Successfully deleted user with user_id: {user_id}")
                return {"success": True, "message": f"User '{user_id}' deleted successfully."}
            else:
                print(f"No user found with user_id: {user_id}")
                return {"success": False, "message": f"No user found with user_id: {user_id}"}
        except Exception as e:
            print(f"Error deleting user: {e}")
            return {"success": False, "message": str(e)}
