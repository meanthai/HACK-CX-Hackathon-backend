
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class UserSchema(BaseModel):
    """
    User Schema:
    - user_name: str (required)
    - user_id: str (required, unique)
    - user_current_acc_balance: float (default: 0.0)
    - created_at: datetime (auto-generated)
    - updated_at: datetime (auto-updated)
    """
    user_name: str
    user_id: str
    user_current_acc_balance: float = 0.0
    total_freq_deposit: int = 0
    total_freq_credit_loan: int = 0
    total_freq_stock_investment: int = 0
    last_deposit_timestamp: Optional[datetime] = None
    last_credit_loan_timestamp: Optional[datetime] = None
    last_stock_investment_timestamp: Optional[datetime] = None
    past_conversations: str = Field(description="Past conversations with the user")

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

    def create_user(self, user_data):
        """Create a new user"""
        try:
            existing_user = self.collection.find_one({"user_id": user_data["user_id"]})
            if existing_user:
                print(f"Error: User with user_id '{user_data['user_id']}' already exists!")
                return None
            
            result = self.collection.insert_one(user_data)
            print(f"User created successfully with ID: {result.inserted_id}")
            return result.inserted_id
        
        except Exception as e:
            print(f"Error creating user: {e}")
            return None

    def get_user_by_id(self, user_id):
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

    def get_all_users(self):
        try:
            users = list(self.collection.find())
            print(f"Found {len(users)} users:")
            return users
        except Exception as e:
            print(f"Error getting users: {e}")
            return []

    def update_user_info(self, user_id, update_data):
        try:        
            result = self.collection.update_one(
                {"user_id": user_id},
                {"$set": update_data}
            )
            
            if result.matched_count > 0:
                print(f"Successfully updated user '{user_id}' with data: {update_data}")
                return True
            else:
                print(f"No user found with user_id: {user_id}")
                return False
        
        except Exception as e:
            print(f"Error updating user: {e}")
            return False

def run_examples(db_manager=None):
    """Run example CRUD operations"""
    print("\n" + "="*50)
    print("RUNNING CRUD EXAMPLES")
    print("="*50)
    
    print("\n1. CREATING SAMPLE USERS:")
    print("-" * 30)
    
    sample_users = [
        {
            "user_name": "John Doe",
            "user_id": "john_doe_001",
            "user_current_acc_balance": 1500.50,
            "total_freq_deposit"   : 5,
            "total_freq_credit_loan": 2,
            "total_freq_stock_investment": 3
        },
        {
            "user_name": "Jane Smith",
            "user_id": "jane_smith_002",
            "user_current_acc_balance": 2750.25,
            "total_freq_deposit"   : 10,
            "total_freq_credit_loan": 1,
            "total_freq_stock_investment": 4
        },
        {
            "user_name": "Bob Johnson",
            "user_id": "bob_johnson_003",
            "user_current_acc_balance": 1200.00
        }
    ]
    
    for user in sample_users:
        db_manager.create_user(user)
    
    print("\n2. READING USERS:")
    print("-" * 30)
    
    db_manager.get_all_users()
    
    print("\nGetting specific user:")
    db_manager.get_user_by_id("john_doe_001")
    
    print("\n3. UPDATING USERS:")
    print("-" * 30)
    

    db_manager.update_user_info("jane_smith_002", {
        "user_name": "Jane Smith-Wilson",
        "user_current_acc_balance": 3000.00
    })
    
    print("\nUsers after updates:")
    db_manager.get_all_users()
    
    print("\n4. DELETING USER:")
    print("-" * 30)
        
    print("\nFinal user list:")
    db_manager.get_all_users()


if __name__ == "__main__":
    db_manager = DatabaseManager()

    run_examples(db_manager=db_manager)